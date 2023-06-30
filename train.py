
from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import re
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import load_data, accuracy, l1_regularization, getparam
from models import Gallat
from myDataset import myDataset

t_total = time.time()
print('train.py_StartTime:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# Training settings
parser = argparse.ArgumentParser()
# parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--if_cuda', action='store_true', default=True, help='If use GPU in training.')
parser.add_argument('--ninput', type=int, default=805, help='The input dimension of the dataloader')
parser.add_argument('--ngrid', type=int, default=400, help='Number of grids.')
parser.add_argument('--nfeat', type=int, default=830, help='The input dimension of the features for attention layer')
parser.add_argument('--nhid', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--nout', type=int, default=400, help='Number of output units.')
parser.add_argument('--nchannel', type=int, default=4, help='Number of channels in attention layer.')
parser.add_argument('--p', type=int, default=7, help='p-days data before current time is considered when predicting.')
parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--scale', type=float, default=582.0, help='# train.max() = 581, test.max() = 582')
parser.add_argument('--lr', type=float, default=0.00001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--w_d', type=float, default=0.8, help='The weight of demand loss.')
parser.add_argument('--w_od', type=float, default=0.2, help='The weight of od matrix loss.')

# These parameters can be set small values during debugging
parser.add_argument('--train_day', type=int, default=108, help='Number of trainning day.') # 108 This must be more than 7
parser.add_argument('--test_day', type=int, default=21, help='Number of testing day.') # 21 This must be more than 7
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.') # 100
parser.add_argument('--batch_size', type=int, default=2, help='Number of batches in each epoch.') # 24
parser.add_argument('--nworkers', type=int, default=6, help='Number of workers in each epoch.') # depend on the number of GPU


args = parser.parse_args()
args.cuda = args.if_cuda and torch.cuda.is_available()
print('if_cuda =', args.cuda)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# device = torch.device("cuda:" + re.split(r",", os.environ['CUDA_VISIBLE_DEVICES'])[0] if args.cuda else "cpu")
# ids = list(range(torch.cuda.device_count()))
# print("GPU_ids=", ids)
# device = torch.device("cuda:" + str(ids[0]) if args.cuda else "cpu")

torch.backends.cudnn.benchmark = True
print('args=', args)

# Load data. df is the concatenation of matrix (ngrid * ngrid) in each time slot
train_df, test_df, geo_mask = load_data(args.ngrid, args.train_day, args.test_day)
geo_mask = torch.FloatTensor(geo_mask)

# Normalization of df
train_df = train_df / args.scale
test_df = test_df / args.scale

# Define model and optimizer
model = Gallat(args)
print('model =', model)
getparam(model)

if args.cuda:
    model = model.cuda()
    num = 1
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        args.nworkers = torch.cuda.device_count() * args.nworkers
        num = 4
        model = nn.DataParallel(model)
        # torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
        # model = DistributedDataParallel(model)
    else:
        print("Let's use", torch.cuda.device_count(), "GPU!")
geo_mask = geo_mask.repeat(args.batch_size * num, 1).view(args.batch_size * num, 1, 1, args.ngrid, args.ngrid)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=1, factor=0.3, patience=10)


def run_model(df, mode):
    with torch.no_grad():
        model.eval()
        global geo_mask
        od_dataset = myDataset(df, args, mode)
        val_loader = torch.utils.data.DataLoader(dataset=od_dataset,
                                                 batch_size=args.batch_size * num,
                                                 shuffle=False, num_workers=args.nworkers)
        global geo_mask

        for sample, (gt, feat) in enumerate(val_loader):
            if args.cuda:
                # geo_mask.repeat(args.batch_size, 1).view(args.batch_size, args.ngrid, args.ngrid)
                gt, feat, geo_mask = gt.cuda(), feat.cuda(), geo_mask.cuda()
            if torch.cuda.device_count() > 1:
                d_acc, od_acc, val_loss = model.module.predict(gt, feat, geo_mask)
            else:
                d_acc, od_acc, val_loss = model.predict(gt, feat, geo_mask)
            if sample == 0:
                d_acc_final, od_acc_final, loss_final = d_acc, od_acc, val_loss
            else:
                d_acc_final, od_acc_final, loss_final = torch.add(d_acc_final, d_acc), \
                                                        torch.add(od_acc_final, od_acc), torch.add(loss_final, val_loss)
        d_acc_final, od_acc_final, loss_final = d_acc_final/(sample + 1), od_acc_final/(sample + 1), loss_final/(sample + 1)

    return d_acc_final, od_acc_final, loss_final


def train_epoch(epoch):
    # Trainning process
    print('epoch = ', epoch)
    mode = torch.ones(1)
    model.train()
    print('-------------------------------------------------------')
    od_dataset = myDataset(train_df, args, mode)
    train_loader = torch.utils.data.DataLoader(dataset=od_dataset, batch_size=args.batch_size * num,
                                                    shuffle=False, num_workers=args.nworkers)

    global geo_mask
    loss = 0
    batch = 0
    t = time.time()
    optimizer.zero_grad()
    for sample, (gt, feat) in enumerate(train_loader):
        if args.cuda:
            gt, feat, geo_mask = gt.cuda(), feat.cuda(), geo_mask.cuda()
        demand, od_matrix = model(gt, feat, geo_mask)
        train_loss = torch.mul(F.smooth_l1_loss(demand, gt[:, :, :args.ngrid].sum(1)), args.w_d) + \
               torch.mul(F.smooth_l1_loss(od_matrix, gt[:, :, :args.ngrid]), args.w_od)
        train_loss.backward()
        loss += train_loss
        if ((sample+1) * args.batch_size * num) % 24 == 0:
            optimizer.step()
            print('batch: {:02d}'.format(batch), 'time: {:.4f}s'.format(time.time() - t),
                  'train_loss: {:.4f}'.format(float(loss)/24.0))
            batch += 1
            loss = 0
            t = time.time()
            optimizer.zero_grad()

    # Validation process
    t = time.time()
    d_val_acc, od_val_acc, val_loss = run_model(test_df, torch.zeros(1))
    print('val_loss: {:.4f}'.format(val_loss), 'time: {:.4f}s'.format(time.time() - t))
    print('demand_acc_val: mae: {0[0][0]}, 0mae: {0[0][1]}, 3mae: {0[0][2]}, '
             '5mae: {0[0][3]}, 10mae: {0[0][4]}'.format(d_val_acc))
    print('               mape: {0[1][0]}, 0mape: {0[1][1]}, 3mape: {0[1][2]}, '
             '5mape: {0[1][3]}, 10mape: {0[1][4]}'.format(d_val_acc))
    print('                mse: {0[2][0]}, 0mse: {0[2][1]}, 3mse: {0[2][2]}, '
          '5mse: {0[2][3]}, 10mse: {0[2][4]}'.format(d_val_acc))
    print('               rmse: {0[3][0]}, 0rmse: {0[3][1]}, 3rmse: {0[3][2]}, '
          '5rmse: {0[3][3]}, 10rmse: {0[3][4]}'.format(d_val_acc))
    print('    od_acc_val: mae: {0[0][0]}, 0mae: {0[0][1]}, 3mae: {0[0][2]}, '
          '5mae: {0[0][3]}, 10mae: {0[0][4]}'.format(od_val_acc))
    print('               mape: {0[1][0]}, 0mape: {0[1][1]}, 3mape: {0[1][2]}, '
          '5mape: {0[1][3]}, 10mape: {0[1][4]}'.format(od_val_acc))
    print('                mse: {0[2][0]}, 0mse: {0[2][1]}, 3mse: {0[2][2]}, '
          '5mse: {0[2][3]}, 10mse: {0[2][4]}'.format(od_val_acc))
    print('               rmse: {0[3][0]}, 0rmse: {0[3][1]}, 3rmse: {0[3][2]}, '
          '5rmse: {0[3][3]}, 10rmse: {0[3][4]}'.format(od_val_acc))

    return val_loss


def compute_test():
    # Testing process
    t = time.time()
    d_test_acc, od_test_acc, test_loss = run_model(test_df, torch.zeros(1))
    print('test_loss: {:.4f}'.format(test_loss), 'time: {:.4f}s'.format(time.time() - t))
    print('demand_acc_test: mae: {0[0][0]}, 0mae: {0[0][1]}, 3mae: {0[0][2]}, '
          '5mae: {0[0][3]}, 10mae: {0[0][4]}'.format(d_test_acc))
    print('               mape: {0[1][0]}, 0mape: {0[1][1]}, 3mape: {0[1][2]}, '
          '5mape: {0[1][3]}, 10mape: {0[1][4]}'.format(d_test_acc))
    print('                mse: {0[2][0]}, 0mse: {0[2][1]}, 3mse: {0[2][2]}, '
          '5mse: {0[2][3]}, 10mse: {0[2][4]}'.format(d_test_acc))
    print('               rmse: {0[3][0]}, 0rmse: {0[3][1]}, 3rmse: {0[3][2]}, '
          '5rmse: {0[3][3]}, 10rmse: {0[3][4]}'.format(d_test_acc))
    print('    od_acc_test: mae: {0[0][0]}, 0mae: {0[0][1]}, 3mae: {0[0][2]}, '
          '5mae: {0[0][3]}, 10mae: {0[0][4]}'.format(od_test_acc))
    print('               mape: {0[1][0]}, 0mape: {0[1][1]}, 3mape: {0[1][2]}, '
          '5mape: {0[1][3]}, 10mape: {0[1][4]}'.format(od_test_acc))
    print('                mse: {0[2][0]}, 0mse: {0[2][1]}, 3mse: {0[2][2]}, '
          '5mse: {0[2][3]}, 10mse: {0[2][4]}'.format(od_test_acc))
    print('               rmse: {0[3][0]}, 0rmse: {0[3][1]}, 3rmse: {0[3][2]}, '
          '5rmse: {0[3][3]}, 10rmse: {0[3][4]}'.format(od_test_acc))

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    val_loss = train_epoch(epoch)
    loss_values.append(val_loss)
    scheduler.step(val_loss)
    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1
    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
compute_test()

print('Total_time: {:.4f}s'.format(time.time() - t_total))