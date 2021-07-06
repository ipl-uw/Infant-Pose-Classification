import os
import time
import json
import argparse
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from dataset import InfantDataset
from posenet import PoseNet
from utils.HeatmapProc import get_keypoints
from Visualize import drawCar

def parse_args():
    parser = argparse.ArgumentParser(description='Train HG.')
    parser.add_argument('-md', '--modeldir', type=str, dest='model_dir',
                        help='file name to save trained model')
    parser.add_argument('-dr', '--data_root', type=str, dest='data_root', default='/home/zhongyu/data/infant_synthetic',
                        help='directory to load data')
    parser.add_argument('-ld', '--logdir', type=str, dest='log_dir', default='./results/',
                        help='directory to save trained model')
    args = parser.parse_args()
    return args


def create_dir_for_new_model(name='infantHG'):
    model_name = name + '-' + time.strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(os.path.join(train_model_path, model_name)):
        os.makedirs(os.path.join(train_model_path, model_name))
    return model_name


if __name__ == "__main__":

    args = parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    train_model_path = args.log_dir

    model_name = None
    epoch_start = 0
    iter_start = 0
    if args.model_dir is not None and os.path.exists(os.path.join(train_model_path, args.model_dir)):
        model_dir = args.model_dir
        models_loaded = sorted(os.listdir(os.path.join(train_model_path, model_dir)))
        for fid, file in enumerate(models_loaded):
            if not file.endswith('.pkl'):
                del models_loaded[fid]
        if len(models_loaded) == 0:
            model_dir = create_dir_for_new_model()
        else:
            model_name = models_loaded[-2]
            epoch_start = int(float(model_name.split('.')[0].split('_')[1]))
    else:
        model_dir = create_dir_for_new_model('infantHG')

    writer = SummaryWriter(os.path.join(train_model_path, model_dir))

    n_epoch = 40
    batch_size = 8
    lr = 1e-4
    stacked_num = 6

    idataset = InfantDataset(root_dir=args.data_root)
    dataloader = DataLoader(idataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # print training configurations
    print("Training dataset length: %d" % len(idataset))
    print("Number of epoches: %d" % n_epoch)
    print("Batch size: %d" % batch_size)
    print("Number of iterations in each epoch: %d" % int(len(idataset) // batch_size))
    model = PoseNet(nstack=stacked_num, inp_dim=256, oup_dim=13).cuda()
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

    iter_count = 0
    if model_name is not None:
        checkpoint = torch.load(os.path.join(train_model_path, '%s/%s' % (model_dir, model_name)))
        if 'optimizer_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_start = checkpoint['epoch']
            loss_cp = checkpoint['loss']
            if 'iter_count' in checkpoint:
                iter_count = checkpoint['iter_count']
        else:
            model.load_state_dict(checkpoint)

    for epoch in range(epoch_start, n_epoch):

        tic_load = time.time()

        for iter, (data, heatmap_gt, bbox_gt, keypoints_gt) in enumerate(dataloader):

            tic = time.time()
            optimizer.zero_grad()  # zero the parameter gradients
            heatmap_preds = model(data.float().cuda())
            heatmap_gt = heatmap_gt * 100

            loss_heatmap = 0
            for i in range(stacked_num):
                loss_cur = criterion(heatmap_preds[:, i, :, :, :], heatmap_gt.float().cuda())
                loss_heatmap += loss_cur
            loss_heatmap.backward()
            optimizer.step()
            writer.add_scalar('data/loss_all', loss_heatmap.item(), iter_count)
            writer.add_scalar('data/loss_heatmap', loss_cur.item(), iter_count)
            iter_count += 1

            # print statistics
            print('epoch %d, iter %d: loss: %.8f | load time: %.4f | backward time: %.4f | max value: %.4f' %
                  (epoch + 1, iter + 1, loss_heatmap.item(), tic - tic_load, time.time() - tic, torch.max(heatmap_preds).cpu().item()), end='\r')

            tic_load = time.time()

        status_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_heatmap,
                'iter_count': iter_count,
                }
        save_model_path = os.path.join(train_model_path,
                                        '%s/infantHG_%02d.pkl' %
                                        (model_dir, epoch+1))
        torch.save(status_dict, save_model_path)

        scheduler.step()

    print('Training Finished.')
