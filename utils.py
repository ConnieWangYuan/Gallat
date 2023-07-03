import datetime
import numpy as np

import torch

# Directories of datasets
from sklearn.metrics import mean_squared_error
ids = list(range(torch.cuda.device_count()))
device = torch.device("cuda:" + str(ids[0]) if torch.cuda.is_available() else "cpu")

path_geo = '/nfs/volume-527-1/wangyuandong_backup/DDWP_Datasets/geo_mask.npy'
path1 = '/nfs/volume-527-1/wangyuandong_backup/DDWP_Datasets/June_Matrix_Hour.npy'
path2 = '/nfs/volume-527-1/wangyuandong_backup/DDWP_Datasets/July_Matrix_Hour.npy'
path3 = '/nfs/volume-527-1/wangyuandong_backup/DDWP_Datasets/August_Matrix_Hour.npy'
path4 = '/nfs/volume-527-1/wangyuandong_backup/DDWP_Datasets/September_Matrix_Hour.npy'
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
