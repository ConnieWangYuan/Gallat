import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn import init

from layers import SpatialAttentionLayer, TemporalAttentionLayer, DotProductAttentionLayer, TransferringAttentionLayer
from utils import l1_regularization, l2_regularization, accuracy

ids = list(range(torch.cuda.device_count()))
device = torch.device("cuda:" + str(ids[0]) if torch.cuda.is_available() else "cpu")


class Gallat(nn.Module):
    def __init__(self, args):
        """dynamic Graph with ALL ATtention"""
        super(Gallat, self).__init__()
        self.nchannel = args.nchannel
        self.p = args.p
        self.ninput = args.ninput
        self.ngrid = args.ngrid
        self.w_d = args.w_d
        self.w_od = args.w_od
        # self.if_cuda = args.cuda
        self.scale = args.scale
        self.dropout = args.dropout

        self.embed_dow = nn.Embedding(7, 4)
        self.embed_hour = nn.Embedding(24, 8)
        self.embed_grid = nn.Embedding(400, 16)
        self.embed_sp = 28  # The embedding dimension of (dow, time, grid)

        # Change the embedding dimension of three fields to fit the following calculation.
        self.dense = nn.Linear(self.embed_sp, 4 * args.nhid)

        # Spatial layer, we use GAT to embed geographical and semantic information
        self.sp_atts = SpatialAttentionLayer(args.ngrid, args.nfeat, args.nhid, args.dropout, args.alpha)

        # Temporal layer, we use the former seven historical time slots and short-term data to predict the current
        # demands through four chennals
        self.te_channels = nn.ModuleList()
        for i in range(args.nchannel):
            channel = TemporalAttentionLayer(args.nhid, args.p, args.dropout)
            self.te_channels.append(channel)
        self.dpa = DotProductAttentionLayer(args.nhid, self.nchannel, args.dropout)

        # Transferring layer, self-attention parameter to calculate the od matrix
        self.outlayer = nn.Linear(4 * args.nhid, 1)  # Output the demand of all grids
        self.leakyrelu = nn.LeakyReLU(args.alpha)
        self.trans = TransferringAttentionLayer(args.nhid, args.ngrid, args.dropout, args.alpha)

    def forward(self, gt, feat, geo_mask):
        # Embedding for three fields, dow, hour, grid_id
        grid_vec = self.embed_grid(feat[:, :, :, :, -3:-2].to(torch.int64)).squeeze(4)
        dow_vec = self.embed_dow(feat[:, :, :, :, -2:-1].to(torch.int64)).squeeze(4)
        hour_vec = self.embed_hour(feat[:, :, :, :, -1:].to(torch.int64)).squeeze(4)
        feat = torch.cat([feat[:, :, :, :, :-3], grid_vec, dow_vec, hour_vec], 4)

        grid_vec = self.embed_grid(gt[:, :, -3:-2].to(torch.int64)).squeeze(2)
        dow_vec = self.embed_dow(gt[:, :, -2:-1].to(torch.int64)).squeeze(2)
        hour_vec = self.embed_hour(gt[:, :, -1:].to(torch.int64)).squeeze(2)
        gt_embed = torch.cat([grid_vec, dow_vec, hour_vec], 2)
        gt_embed = self.dense(gt_embed)

        chennal_list = []
        for c in range(self.nchannel):
            combine_list = []
            for p in range(self.p):
                sp_matrix = self.sp_atts(feat[:, c:c + 1, p:p + 1, :, :], geo_mask)
                combine_list.append(sp_matrix)
            sp_feat = torch.cat(combine_list, 2)
            te_matrix = self.te_channels[c](sp_feat, gt_embed)
            chennal_list.append(te_matrix)
        channel_feat = torch.cat(chennal_list, 1)
        st_feat = self.dpa(channel_feat, gt_embed)

        demand = self.leakyrelu(self.outlayer(st_feat))
        demand = F.dropout(demand, self.dropout, training=self.training)
        od_matrix = self.trans(demand, st_feat)
        demand = demand.squeeze(4).squeeze(1).squeeze(1)
        od_matrix = od_matrix.squeeze(1).squeeze(1)

        return demand, od_matrix

    def predict(self, gt, feat, geo_mask):
        demand, od_matrix = self.forward(gt, feat, geo_mask)
        d_gt = gt[:, :, -5:-4].squeeze(-1)
        od_gt = gt[:, :, :self.ngrid]
        loss = torch.mul(F.smooth_l1_loss(demand, d_gt), self.w_d) + \
               torch.mul(F.smooth_l1_loss(od_matrix, od_gt), self.w_od)
        demand = demand.mul(self.scale)
        od_matrix = od_matrix.mul(self.scale)
        d_gt = d_gt.mul(self.scale)
        od_gt = od_gt.mul(self.scale)
        d_acc = accuracy(demand, d_gt)
        od_acc = accuracy(od_matrix, od_gt)

        return d_acc, od_acc, loss
