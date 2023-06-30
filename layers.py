import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
ids = list(range(torch.cuda.device_count()))
device = torch.device("cuda:" + str(ids[0]) if torch.cuda.is_available() else "cpu")

class SpatialAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, ngrid, nfeat, nhid, dropout, alpha):
        super(SpatialAttentionLayer, self).__init__()
        self.dropout = dropout
        self.nfeat = nfeat
        self.nhid = nhid
        self.ngrid = ngrid

        self.W = nn.Parameter(torch.zeros(size=(self.nfeat, self.nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.selfa_out = SelfAttentionLayer(nhid, nhid, 2 * nhid, dropout, alpha)
        self.selfa_in = SelfAttentionLayer(nhid, nhid, 2 * nhid, dropout, alpha)
        self.selfa_geo = SelfAttentionLayer(nhid, nhid, 2 * nhid, dropout, alpha)

    def forward(self, features, geo_mask):
        # Get the semantic mask for out-degree neighbors and in-degree neighbors
        matrix = features[:, :, :, :, :self.ngrid]
        sem_mask_out = matrix.div(matrix.sum(4, keepdim=True) + 1)
        sem_mask_in = matrix.div(matrix.sum(3, keepdim=True) + 1)

        # Get a shared linear transformation
        features = features.matmul(self.W)
        feat_out = sem_mask_out.matmul(features)
        feat_in = (sem_mask_in.permute(0, 1, 2, 4, 3)).matmul(features)
        feat_geo = geo_mask.matmul(features)

        # Calculate attentions among nodes
        feat_list = []
        attention_out, feat_out = self.selfa_out(features, feat_out)
        feat_list.append(attention_out.matmul(feat_out))
        attention_in, feat_in = self.selfa_in(features, feat_in)
        feat_list.append(attention_in.matmul(feat_in))
        attention_geo, feat_geo = self.selfa_geo(features, feat_geo)
        feat_list.append(attention_geo.matmul(feat_geo))
        combine_feat = torch.cat([features] + feat_list, 4)
        combine_feat = F.dropout(combine_feat, self.dropout, training=self.training)

        return combine_feat

    # def __repr__(self):
    #     return self.__class__.__name__ + ' (' + str(self.nfeat) + ' -> ' + str(self.nhid) + ')'


class TemporalAttentionLayer(nn.Module):
    """
    Attention layer for temporal trend.
    """

    def __init__(self, nhid, p, dropout):
        super(TemporalAttentionLayer, self).__init__()
        self.dropout = dropout

        self.dpa = DotProductAttentionLayer(nhid, p, self.dropout)

    def forward(self, combine_list, gt_embed):
        te_feat = self.dpa(combine_list, gt_embed)
        te_feat = F.dropout(te_feat, self.dropout, training=self.training)
        return te_feat

    # def __repr__(self):
    #     return self.__class__.__name__ + ' (' + str(self.dropout) + ' -> ' + str(self.embed_sp) + ')'


class TransferringAttentionLayer(nn.Module):
    """
    Self-Attention layer is used to calculate the relationship between grids and project the demand into destination grid.
    """
    def __init__(self, nhid, ngrid, dropout, alpha):
        super(TransferringAttentionLayer, self).__init__()
        self.nhid = nhid
        self.dropout = dropout

        self.selfa = SelfAttentionLayer(nhid * 4, nhid * 4, 2 * nhid * 4, dropout, alpha)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, demand, st_feat):
        attention, st_feat = self.selfa(st_feat, st_feat)
        od_matrix = self.leakyrelu(demand * attention)
        od_matrix = F.dropout(od_matrix, self.dropout, training=self.training)
        return od_matrix


class DotProductAttentionLayer(nn.Module):
    """
    Dot product Attention layer like Transformer.
    """

    def __init__(self, nhid, nre, dropout):
        super(DotProductAttentionLayer, self).__init__()
        self.dropout = dropout
        self.nre = nre
        self.Wk = nn.Parameter(torch.zeros(size=(nhid*4, nhid*4)))
        nn.init.xavier_uniform_(self.Wk.data, gain=1.414)
        self.Wq = nn.Parameter(torch.zeros(size=(nhid*4, nhid*4)))
        nn.init.xavier_uniform_(self.Wq.data, gain=1.414)
        self.Wv = nn.Parameter(torch.zeros(size=(nhid*4, nhid*4)))
        nn.init.xavier_uniform_(self.Wv.data, gain=1.414)

    def forward(self, source, target):
        d = target.size(-1)
        target = target.matmul(self.Wq).repeat(1, self.nre, 1).view(source.shape[0], source.shape[1], source.shape[2],
                                                                    source.shape[3], source.shape[4])
        attention = target.matmul(source.matmul(self.Wk).transpose(-2, -1)) / math.sqrt(d)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        feat = torch.matmul(attention, source.matmul(self.Wv))
        if self.nre == 7:
            feat = feat.sum(2, keepdim=True)
        else:
            feat = feat.sum(1, keepdim=True)
        return feat


class SelfAttentionLayer(nn.Module):
    """
    Self-Attention layer is used to calculate the relationship between grids and project the demand into destination grid.
    """
    def __init__(self, wd1, wd2, ad, dropout, alpha):
        super(SelfAttentionLayer, self).__init__()
        self.ad = ad
        self.dropout = dropout

        self.W = nn.Parameter(torch.zeros(size=(wd1, wd2)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(ad, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, feat1, feat2):
        feat1, feat2 = feat1.matmul(self.W), feat2.matmul(self.W)
        N = feat1.shape[3]
        feat_ = torch.cat([feat1.repeat(1, 1, 1, 1, N).view(feat1.shape[0], feat1.shape[1], feat1.shape[2],
            N * N, -1), feat2.repeat(1, 1, 1, N, 1)], dim=4).view(feat1.shape[0], feat1.shape[1], feat1.shape[2], N, -1, self.ad)
        attention = F.softmax(self.leakyrelu(torch.matmul(feat_, self.a).squeeze(5)), dim=4)
        # The reason for using leakyrelu is to increase the nonlinearity. ------From GAT
        attention = F.dropout(attention, self.dropout, training=self.training)

        return attention, feat2



