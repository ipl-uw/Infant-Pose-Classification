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
    parser = argparse.ArgumentParser(description='Test HG.')
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

    result = []

    model_name = None
    epoch_start = 0
    iter_start = 0
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

    n_epoch = 40
    batch_size = 4
    stacked_num = 6

    cfdataset = InfantDataset(root_dir=args.data_root, mode='validate')
    dataloader = DataLoader(cfdataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # print training configurations
    print("Training dataset length: %d" % len(cfdataset))
    print("Number of epoches: %d" % n_epoch)
    print("Batch size: %d" % batch_size)
    print("Number of iterations in each epoch: %d" % int(len(cfdataset) // batch_size))
    model = PoseNet(nstack=stacked_num, inp_dim=256, oup_dim=13).cuda()
    model.eval()
    distance = []

    f = []
    for i in range(11):
        f.append(open('poseEst2d_%d.txt' % i, 'w'))

    iter_count = 0
    if model_name is not None:
        checkpoint = torch.load(os.path.join(train_model_path, '%s/%s' % (model_dir, model_name)))
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch_start = checkpoint['epoch']
        loss_cp = checkpoint['loss']
        if 'iter_count' in checkpoint:
            iter_count = checkpoint['iter_count']
    else:
        exit(1)

    tic_load = time.time()
    for iter, (data, bbox_gt, keypoints_gt, image_id) in enumerate(dataloader):

        tic = time.time()
        heatmap_preds = model(data.float().cuda())

        iter_count += 1

        for i in range(len(image_id)):
            x, y, w, h = bbox_gt[i].numpy()
            keypoints_pred = np.array(get_keypoints(heatmap_preds[i, stacked_num - 1,...].cpu().detach().numpy()))
            keypoints_pred[:, 0] = keypoints_pred[:, 0] * w / 96 + x
            keypoints_pred[:, 1] = keypoints_pred[:, 1] * h / 64 + y
            keypoints_gt_t = keypoints_gt[i].numpy().reshape((13, 3)).copy()
            distance.append((np.sum(np.abs(keypoints_gt_t - keypoints_pred)[:, :2])) / 13)
            # if distance[-1] > 50:
            # keypoints_pred = np.array(get_keypoints(heatmap_preds[i, stacked_num - 1,...].cpu().detach().numpy()))
            # keypoints_pred[:, 0] = keypoints_pred[:, 0] * 4
            # keypoints_pred[:, 1] = keypoints_pred[:, 1] * 4
            # img = data[i, :, :, :].cpu().detach().numpy()

            result.append({
                "category_id": 1,
                "image_id": int(image_id[i]),
                "keypoints": keypoints_pred.reshape((39)).tolist(),
                "score": 1,
                "bbox": [x, y, w, h]
            })

            f[int(image_id[i]) // 10000].write('%d,' % (int(image_id[i]) % 10000))
            for j in range(13):
                f[int(image_id[i]) // 10000].write('%f,%f,%f,' % (keypoints_pred[j, 0], keypoints_pred[j, 1], keypoints_pred[j, 2]))
            f[int(image_id[i]) // 10000].write('%f,%f,%f\n' % ((keypoints_pred[1, 0] + keypoints_pred[2, 0]) / 2, (keypoints_pred[1, 1] + keypoints_pred[2, 1]) / 2, (keypoints_pred[1, 2] + keypoints_pred[2, 2]) / 2))

            # gt_img = img.copy()
            # keypoints_gt_t[:, 0] = (keypoints_gt_t[:, 0] - x) * 256 / w
            # keypoints_gt_t[:, 1] = (keypoints_gt_t[:, 1] - y) * 256 / h
            # keypoints_gt_t = keypoints_gt_t.astype(np.int)
            # img = drawCar(img, keypoints_pred.astype(np.int), thre=1)
            # gt_img = drawCar(gt_img, keypoints_gt_t, thre=0.5)
            # res = np.hstack((img, gt_img))
            # plt.imshow(res)
            # plt.show()
        
        print('iter %d: | load time: %.4f | backward time: %.4f | ave error: %.4f' %
                (iter + 1, tic - tic_load, time.time() - tic, sum(distance) / len(distance)))

        tic_load = time.time()

    json.dump(result, open('hg_result.json', 'w'))

    print('Finished Test.')
