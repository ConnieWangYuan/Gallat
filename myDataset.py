import datetime
import random
import time
import argparse
import numpy as np

import torch
from torch.utils.data import Dataset

train_start = '20190601'
# test_start = '20190610'
test_start = '20190917'
print('train_start =', train_start)
print('test_start =', test_start)


class myDataset(Dataset):
    def __init__(self, df, args, mode):
        """
        Args:
            df: trainning dataset or testing dataset
            day (int): which day from the beginning of the dataset
            hour (int): which hour
        """
        self.df = df
        self.c = args.nchannel
        self.p = args.p
        self.ninput = args.ninput
        self.ngrid = args.ngrid
        self.mode = mode
        # self.geo_mask = geo_mask

        if self.mode.data.item() == 1:
            self.t_day = args.train_day
        else:
            self.t_day = args.test_day

    def worker(self, gt_date, day):
        gt_dow = gt_date.weekday()
        gt_hour = gt_date.hour
        start_index = (gt_hour + day * 24) * self.ngrid
        gt = self.df[start_index: start_index + self.ngrid]
        # gt = np.c_[gt, np.sum(gt, 1), np.sum(gt.T, 1), np.arange(gt.shape[0]), np.full((gt.shape[0]), gt_dow),
        #            np.full((gt.shape[0]), gt_hour)
        gt = np.column_stack((gt, np.sum(gt, 1), np.sum(gt.T, 1), np.arange(gt.shape[0]), np.full((gt.shape[0]), gt_dow),
                         np.full((gt.shape[0]), gt_hour)))
        a_arr, b_arr, c_arr, d_arr = np.zeros(gt.shape[0]), np.zeros(gt.shape[0]), \
                                         np.zeros(gt.shape[0]), np.zeros(gt.shape[0])
        for p in range(self.p):
            a_start = start_index - (p + 1) * 24 * self.ngrid
            b_start = start_index - (p + 1) * 24 * self.ngrid - self.ngrid
            c_start = start_index - (p + 1) * 24 * self.ngrid + self.ngrid
            d_start = start_index - (p + 1) * self.ngrid

            a = np.array(self.df[a_start: a_start + self.ngrid])
            a_dow = (gt_date - datetime.timedelta(days=p, hours=0)).weekday()
            a_hour = (gt_date - datetime.timedelta(days=p, hours=0)).hour
            # a = np.c_[a, a.T, np.sum(a, 1), np.sum(a.T, 1), np.arange(a.shape[0]), np.full((a.shape[0]), a_dow),
            #               np.full((a.shape[0]), a_hour)]
            a = np.column_stack((a, a.T, np.sum(a, 1), np.sum(a.T, 1), np.arange(a.shape[0]), np.full((a.shape[0]), a_dow),
                      np.full((a.shape[0]), a_hour)))

            b = np.array(self.df[b_start: b_start + self.ngrid])
            b_dow = (gt_date - datetime.timedelta(days=p, hours=1)).weekday()
            b_hour = (gt_date - datetime.timedelta(days=p, hours=1)).hour
            # b = np.c_[b, b.T, np.sum(b, 1), np.sum(b.T, 1), np.arange(b.shape[0]), np.full((b.shape[0]), b_dow),
            #           np.full((b.shape[0]), b_hour)]
            b = np.column_stack((b, b.T, np.sum(b, 1), np.sum(b.T, 1), np.arange(b.shape[0]), np.full((b.shape[0]), b_dow),
                      np.full((b.shape[0]), b_hour)))


            c = np.array(self.df[c_start: c_start + self.ngrid])
            c_dow = (gt_date - datetime.timedelta(days=p) + datetime.timedelta(hours=1)).weekday()
            c_hour = (gt_date - datetime.timedelta(days=p) + datetime.timedelta(hours=1)).hour
            # c = np.c_[c, c.T, np.sum(c, 1), np.sum(c.T, 1), np.arange(c.shape[0]), np.full((c.shape[0]), c_dow),
            #           np.full((c.shape[0]), c_hour)]
            c = np.column_stack((c, c.T, np.sum(c, 1), np.sum(c.T, 1), np.arange(c.shape[0]), np.full((c.shape[0]), c_dow),
                      np.full((c.shape[0]), c_hour)))

            d = np.array(self.df[d_start: d_start + self.ngrid])
            d_dow = (gt_date - datetime.timedelta(hours=p + 1)).weekday()
            d_hour = (gt_date - datetime.timedelta(hours=p + 1)).hour
            # d = np.c_[d, d.T, np.sum(d, 1), np.sum(d.T, 1), np.arange(d.shape[0]), np.full((d.shape[0]), d_dow),
            #           np.full((d.shape[0]), d_hour)]
            d = np.column_stack((d, d.T, np.sum(d, 1), np.sum(d.T, 1), np.arange(d.shape[0]), np.full((d.shape[0]), d_dow),
                      np.full((d.shape[0]), d_hour)))

            # a, b, c, d = a.reshape(a.shape[0]*a.shape[1]), b.reshape(b.shape[0] * b.shape[1]), \
            #              c.reshape(c.shape[0] * c.shape[1]), d.reshape(d.shape[0] * d.shape[1])
            if p == 0:
                a_arr, b_arr, c_arr, d_arr = a, b, c, d
            else:
                # a_arr, b_arr, c_arr, d_arr = np.r_[a_arr, a], np.r_[b_arr, b], \
                #                                      np.r_[c_arr, c], np.r_[d_arr, d],
                a_arr, b_arr, c_arr, d_arr = np.row_stack((a_arr, a)), np.row_stack((b_arr, b)), \
                                             np.row_stack((c_arr, c)), np.row_stack((d_arr, d)),


        # feat = np.r_[a_arr, b_arr, c_arr, d_arr].reshape(self.c, self.p, self.ngrid, self.ninput)
        feat = np.row_stack((a_arr, b_arr, c_arr, d_arr)).reshape(self.c, self.p, self.ngrid, self.ninput)

        gt, feat = torch.FloatTensor(gt), torch.FloatTensor(feat)

        return gt, feat

    def __getitem__(self, index):
        hour = random.sample(range(0, 24), 1)[0]
        day = random.sample(range(self.p, self.t_day), 1)[0]
        if day == self.p and hour == 0:
            hour = hour + 1
        if self.mode.data.item() == 1:
            start = train_start
        else:
            start = test_start
        gt_date = datetime.datetime.strptime(start, '%Y%m%d') + datetime.timedelta(days=day, hours=hour)

        return self.worker(gt_date, day)


    def __len__(self):
        return int(len(self.df)/self.ngrid) - 7 * 24









