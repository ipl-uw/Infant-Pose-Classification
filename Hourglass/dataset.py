import torch
import os
import numpy as np
import ast
import json
import matplotlib.pyplot as plt
import math
import pickle
import random
from PIL import Image

import h5py
from pycocotools.coco import COCO

from utils.ImgTransfer import Filp, AdjustBrightness, AdjustContrast, ColorEnhance
from utils.HeatmapProc import get_keypoints

class InfantDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, mode='train', gen_heatmap=False):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.bboxes = []
        self.skeleton =  []
        self.avaliable_points = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]
        self.transforms = [ColorEnhance, AdjustContrast, AdjustBrightness]
        # if mode == 'train':
        #     self.annotation = json.load(open(os.path.join(root_dir, 'annotations', 'pose_label_train.json')))
        # else:
        #     self.annotation = json.load(open(os.path.join(root_dir, 'annotations', 'pose_label_validate.json')))
        if mode == 'train':
            self.coco = COCO(os.path.join(root_dir, 'annotations', 'pose_label_train.json'))
        else:
            self.coco = COCO(os.path.join(root_dir, 'annotations', 'pose_label_validate.json'))
        self.anno_ids = sorted(self.coco.getAnnIds())
        self.length = len(self.coco.loadAnns(self.coco.getAnnIds()))
        self.mode = mode
        if gen_heatmap:
            self.gen_heatmap()
        if mode == 'train':
            if os.path.exists(os.path.join(self.root_dir, 'heatmap.hdf5')):
                self.heatmaps = h5py.File(os.path.join(self.root_dir, 'heatmap.hdf5'), 'r')
            else:
                self.heatmaps = None

    def gen_heatmap(self):
        heatmap_file = h5py.File(os.path.join(self.root_dir, 'heatmap.hdf5'), 'w')
        heatmaps = heatmap_file.create_dataset("heatmaps", (self.length, 12, 64, 64))
        for i in range(self.length):
            print('processing [%05d/%05d]' % (i, self.length), end='\r')
            label = self.coco.loadAnns([i])[0]
            heatmaps[i, :, :, :] = self._gen_heatmap(label)
            # keypoints = 
            
    def _gen_heatmap(self, label, sigma=2, gaussian_thres=36):
        confmap = np.zeros((13, 64, 96), dtype=float)
        keypoints = label['keypoints']
        bbox_x, bbox_y, w, h = label['bbox']

        if w == 0:
            print('\nerror_label')
            return confmap

        for idx, point_idx in enumerate(self.avaliable_points):
            keypoint = keypoints[point_idx * 3:point_idx * 3 + 2]
            keypoint[0] = (keypoint[0] - bbox_x) * (96 / w)
            keypoint[1] = (keypoint[1] - bbox_y) * (64 / h)
            for i in range(64):
                for j in range(96):
                    distant = ((keypoint[1] - i) ** 2 + (keypoint[0] - j) ** 2) / sigma**2
                    if distant < gaussian_thres:  # threshold for confidence maps
                        value = np.exp(- distant / 2) / (2 * math.pi)
                        confmap[idx, i, j] = value if value > confmap[idx, i, j] else confmap[idx, i, j]
        return confmap

    def _transform(self, img, heatmap):
        for transform in self.transforms:
            if random.random() < 0.2:
                img, heatmap = transform(img, heatmap)
        return img, heatmap
    
    def __getitem__(self, index):
        label = self.coco.loadAnns([self.anno_ids[index]])[0]
        bbox_x, bbox_y, w, h = label['bbox']
        while(w == 0):
            index = random.randint(0, self.length - 1)
            label = self.coco.loadAnns([index])[0]
            bbox_x, bbox_y, w, h = label['bbox']

        # image_path = self.coco.loadImgs([label['image_id']])[0]['file_name']
        img = Image.open(os.path.join(self.root_dir, 'images', '%06d.png' % label['image_id']))
        img = img.crop([bbox_x, bbox_y, bbox_x + w, bbox_y + h])
        img = img.resize((384, 256))
        if self.mode == 'train':
            if self.heatmaps is None:
                heatmap = self._gen_heatmap(label)
            else:
                heatmap = self.heatmaps['heatmaps'][index, :, :, :]
            img, heatmap = self._transform(img, heatmap)
            return np.array(img)[:,:,:3] / 255, torch.from_numpy(heatmap.copy()), np.array(label['bbox']), np.array(label['keypoints'])
        else:
            return np.array(img)[:,:,:3] / 255, np.array(label['bbox']), np.array(label['keypoints']), self.anno_ids[index]
    
    def __len__(self):
        return self.length


if __name__ == "__main__":
    A = InfantDataset('/home/zhongyu/data/infant_synthetic')
    print(A[0])
