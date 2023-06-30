import datetime
import numpy as np

import torch

# Directories of datasets
from sklearn.metrics import mean_squared_error
ids = list(range(torch.cuda.device_count()))
device = torch.device("cuda:" + str(ids[0]) if torch.cuda.is_available() else "cpu")


# path_geo = '/nfs/volume-527-1/wangyuandong_backup/DDWP_Datasets/100grids_geo_mask.npy'
# path1 = '/nfs/volume-527-1/wangyuandong_backup/DDWP_Datasets/100grids_June_Matrix_Hour.npy'
# path2 = '/nfs/volume-527-1/wangyuandong_backup/DDWP_Datasets/100grids_July_Matrix_Hour.npy'
# path3 = '/nfs/volume-527-1/wangyuandong_backup/DDWP_Datasets/100grids_August_Matrix_Hour.npy'
# path4 = '/nfs/volume-527-1/wangyuandong_backup/DDWP_Datasets/100grids_September_Matrix_Hour.npy'
# path1 = '/nfs/volume-527-1/wangyuandong_backup/DDWP_Datasets/shanghai_100grids_June_Matrix_Hour.npy'
# path2 = '/nfs/volume-527-1/wangyuandong_backup/DDWP_Datasets/shanghai_100grids_July_Matrix_Hour.npy'
# path3 = '/nfs/volume-527-1/wangyuandong_backup/DDWP_Datasets/shanghai_100grids_August_Matrix_Hour.npy'
# path4 = '/nfs/volume-527-1/wangyuandong_backup/DDWP_Datasets/shanghai_100grids_September_Matrix_Hour.npy'

path_geo = '/nfs/volume-527-1/wangyuandong_backup/DDWP_Datasets/geo_mask.npy'
path1 = '/nfs/volume-527-1/wangyuandong_backup/DDWP_Datasets/June_Matrix_Hour.npy'
path2 = '/nfs/volume-527-1/wangyuandong_backup/DDWP_Datasets/July_Matrix_Hour.npy'
path3 = '/nfs/volume-527-1/wangyuandong_backup/DDWP_Datasets/August_Matrix_Hour.npy'
path4 = '/nfs/volume-527-1/wangyuandong_backup/DDWP_Datasets/September_Matrix_Hour.npy'
# path1 = '/nfs/volume-527-1/wangyuandong_backup/DDWP_Datasets/Shanghai_June_Matrix_Hour.npy'
# path2 = '/nfs/volume-527-1/wangyuandong_backup/DDWP_Datasets/Shanghai_July_Matrix_Hour.npy'
# path3 = '/nfs/volume-527-1/wangyuandong_backup/DDWP_Datasets/Shanghai_August_Matrix_Hour.npy'
# path4 = '/nfs/volume-527-1/wangyuandong_backup/DDWP_Datasets/Shanghai_September_Matrix_Hour.npy'
# paths = [path1]
# paths = [path1, path2]
# paths = [path1, path2, path3]
paths = [path1, path2, path3, path4]
print('data_paths =', paths)


def load_data(ngrid, train_day, test_day):
    """Load datasets"""
    df_list = []
    # load all data and concatenate into a df
    for path in paths:
        # print '########################################################'
        print('Loading {} dataset...'.format(path))
        df = np.load(path)
        df_list.append(df)
    df = np.concatenate(df_list, axis=0)
    del df_list

    train_size = ngrid * 24 * (train_day)
    test_size = ngrid * 24 * (test_day)
    train_df = df[:train_size]
    print('train_df.shape=', train_df.shape)

    test_df = df[(df.shape[0] - test_size):]
    print('test_df.shape=', test_df.shape)
    del df
    geo_mask = np.load(path_geo)

    return train_df, test_df, geo_mask


# def load_features_tensor(matrix):
#     feat_in =matrix.t()
#
#     feat_o = torch.sum(matrix, 1, keepdim=True)
#     feat_i = torch.sum(feat_in, 1, keepdim=True)
#
#     feat_data = torch.cat([matrix, feat_in, feat_o, feat_i], 1)
#
#     return feat_data
#
# def load_features(matrix):
#     feat_in = matrix.T
#
#     feat_o = np.sum(matrix, 1).reshape(matrix.shape[0], 1)
#     feat_i = np.sum(feat_in, 1).reshape(feat_in.shape[0], 1)
#
#     feat_data = np.concatenate([matrix, feat_in, feat_o, feat_i], 1)
#
#     return feat_data
#
#
# def load_sample(gt_date, day, df, period, ngrid):
#     """Load the four-item input for each sample"""
#     gt_dow = gt_date.weekday()
#     gt_hour = gt_date.hour
#     start_index = (gt_hour + day * 24) * ngrid
#     gt = df[start_index: start_index + ngrid]
#     a_list = []
#     b_list = []
#     c_list = []
#     d_list = []
#     afeat_list = []
#     bfeat_list = []
#     cfeat_list = []
#     dfeat_list = []
#     adow_list = []
#     bdow_list = []
#     cdow_list = []
#     ddow_list = []
#     ahour_list = []
#     bhour_list = []
#     chour_list = []
#     dhour_list = []
#     for p in range(period):
#         a_start = start_index - (p + 1) * 24 * ngrid
#         b_start = start_index - (p + 1) * 24 * ngrid - ngrid
#         c_start = start_index - (p + 1) * 24 * ngrid + ngrid
#         d_start = start_index - (p + 1) * ngrid
#         a_tmp = np.array(df[a_start: a_start + ngrid])
#         a_features = load_features(a_tmp)
#         a_dow = (gt_date - datetime.timedelta(days=p, hours=0)).weekday()
#         a_hour = (gt_date - datetime.timedelta(days=p, hours=0)).hour
#         b_tmp = np.array(df[b_start: b_start + ngrid])
#         b_features = load_features(b_tmp)
#         b_dow = (gt_date - datetime.timedelta(days=p, hours=1)).weekday()
#         b_hour = (gt_date - datetime.timedelta(days=p, hours=1)).hour
#         c_tmp = np.array(df[c_start: c_start + ngrid])
#         c_features = load_features(c_tmp)
#         c_dow = (gt_date - datetime.timedelta(days=p) + datetime.timedelta(hours=1)).weekday()
#         c_hour = (gt_date - datetime.timedelta(days=p) + datetime.timedelta(hours=1)).hour
#         d_tmp = np.array(df[d_start: d_start + ngrid])
#         d_features = load_features(d_tmp)
#         d_dow = (gt_date - datetime.timedelta(hours=p + 1)).weekday()
#         d_hour = (gt_date - datetime.timedelta(hours=p + 1)).hour
#         a_list.append(a_tmp)
#         b_list.append(b_tmp)
#         c_list.append(c_tmp)
#         d_list.append(d_tmp)
#         afeat_list.append(a_features)
#         bfeat_list.append(b_features)
#         cfeat_list.append(c_features)
#         dfeat_list.append(d_features)
#         adow_list.append(a_dow)
#         bdow_list.append(b_dow)
#         cdow_list.append(c_dow)
#         ddow_list.append(d_dow)
#         ahour_list.append(a_hour)
#         bhour_list.append(b_hour)
#         chour_list.append(c_hour)
#         dhour_list.append(d_hour)
#     data = np.concatenate(a_list + b_list + c_list + d_list)
#     features = np.concatenate(afeat_list + bfeat_list + cfeat_list + dfeat_list)
#     dow = np.concatenate([np.array(adow_list).reshape(1, 7), np.array(bdow_list).reshape(1, 7),
#                           np.array(cdow_list).reshape(1, 7), np.array(ddow_list).reshape(1, 7)])
#     hour = np.concatenate([np.array(ahour_list).reshape(1, 7), np.array(bhour_list).reshape(1, 7),
#                            np.array(chour_list).reshape(1, 7), np.array(dhour_list).reshape(1, 7)])
#
#     return torch.FloatTensor(data), torch.FloatTensor(features), \
#            torch.LongTensor(dow), torch.LongTensor(hour), \
#            torch.FloatTensor(gt), torch.LongTensor([gt_dow]), torch.LongTensor([gt_hour])
#

# def accuracy1(pred, gt):
#     # Calculate mae and mape
#     gt = gt.cpu().detach().numpy()
#     pred = pred.cpu().detach().numpy()
#     mae_list = []
#     mape_list = []
#     mse_list = []
#     rmse_list = []
#     threhold = [0, 3, 5, 10]
#     maes = np.absolute(pred - gt)
#     mae = np.mean(maes)
#     mae_list.append(mae)
#     mape = np.mean(maes / (gt + 1))  # In case the denominator equalling to 0, we add 1 to it.
#     mape_list.append(mape)
#
#     mse = mean_squared_error(pred, gt)
#     mse_list.append(mse)
#     rmse = np.sqrt(mse)
#     rmse_list.append(rmse)
#
#     # the mape for all samples that above threhold
#     for th in threhold:
#         eval_indexs = np.where(gt > th)
#         tmp_gt = gt[eval_indexs]
#         tmp_pred = pred[eval_indexs]
#         mae_t = np.mean(maes[eval_indexs])
#         mae_list.append(mae_t)
#         mape_t = np.mean(maes[eval_indexs] / tmp_gt)
#         mape_list.append(mape_t)
#         mse_t = mean_squared_error(tmp_pred, tmp_gt)
#         mse_list.append(mse_t)
#         rmse_t = np.sqrt(mse_t)
#         rmse_list.append(rmse_t)
#     # print("mae_list:", mae_list)
#     # print("mape_list:", mape_list)
#     # print("mse_list:", mse_list)
#     # print("rmse_list:", rmse_list)
#
#     return [mae_list, mape_list, mse_list, rmse_list]

def accuracy(pred, gt):
    # Calculate mae, mape, mse, rmse
    threhold = [0, 3, 5, 10]
    maes = torch.abs(pred - gt)
    mae = torch.mean(maes)
    mape = torch.mean(maes / (gt + 1))  # In case the denominator equalling to 0, we add 1 to it.
    mse_function = torch.nn.MSELoss()
    mse = mse_function(pred, gt)
    rmse = torch.sqrt(mse)

    acc = torch.cat([mae.unsqueeze(0), mape.unsqueeze(0), mse.unsqueeze(0), rmse.unsqueeze(0)], 0).unsqueeze(1)

    # the mape for all samples that above threhold
    for th in threhold:
        eval_indexs = torch.where(gt > th)
        tmp_gt = gt[eval_indexs]
        tmp_pred = pred[eval_indexs]
        mae_t = torch.mean(maes[eval_indexs])
        mape_t = torch.mean(maes[eval_indexs] / tmp_gt)
        mse_t = mse_function(tmp_pred, tmp_gt)
        rmse_t = torch.sqrt(mse_t)
        acc = torch.cat([acc, torch.cat([mae_t.unsqueeze(0), mape_t.unsqueeze(0), mse_t.unsqueeze(0),
                                         rmse_t.unsqueeze(0)], 0).unsqueeze(1)], 1)

    return acc


def l1_regularization(model, loss, cuda):
    param_count = 0
    alpha = 0.0001
    l1_regularization = torch.zeros(1)
    if cuda:
        l1_regularization = l1_regularization.cuda()
    for param in model.parameters():
        l1_regularization = l1_regularization + torch.sum(torch.abs(param))
        param_count = param_count + 1
    # print 'l1_regularization=', l1_regularization
    l1_regularization = torch.div(l1_regularization, param_count).squeeze(0)
    loss = loss + alpha * l1_regularization

    return loss


def l2_regularization(model, loss, cuda):
    param_count = 0
    alpha = 0.0001
    l2_regularization = torch.zeros(1)
    if cuda:
        l2_regularization = l2_regularization.cuda()
    for param in model.parameters():
        l2_regularization = l2_regularization + torch.sum(torch.pow(torch.abs(param), 2))
        param_count = param_count + 1
    # print 'l1_regularization=', l1_regularization
    l2_regularization = torch.div(l2_regularization, param_count).squeeze(0)
    loss = loss + alpha * l2_regularization

    return loss


def getparam(model):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        #   print('layer:' + str(list(i.size())))
        for j in i.size():
            l *= j
        #  print('layer sum' + str(l))
        k = k + l
    print('total parameter size: ' + str(k))


# This is for debugging
if __name__ == "__main__":
    for (month, path) in [('June', path1), ('July', path2), ('August', path3), ('September', path4)]:
        df = np.load(path)
        ngrid = 400
        for sample in [100, 144, 196, 256, 324]:
            start_index = int((ngrid-sample)/2)
            tmp_list = []
            topath = '../../../../volume-527-1/wangyuandong_backup/DDWP_Datasets/shanghai_' + str(sample) + 'grids_' + month + '_Matrix_Hour.npy'
            for i in range(int(df.shape[0]/400)):
                tmp = df[(i*ngrid + start_index):(i*ngrid + start_index + sample), start_index:(start_index + sample)]
                tmp_list.append(tmp)
            df1 = np.concatenate(tmp_list, axis=0)
            np.save(topath, df1)
            df_check = np.load(topath)
            print(topath + ' is done!')