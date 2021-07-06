import json
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import random

import torch

from config import pose2id, ave_thigh_len


class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, save_pickle=False, from_pickle=False, is_train=True, train_set_ratio=0.8, is_gan=False,
                 is_trans_aug=False, is_rot_aug=False):
        # np.random.seed(seed=int(time.time()))
        if not is_train:
            self.rnd = np.random.RandomState(3)
        else:
            self.rnd = np.random.RandomState(int(time.time()))
        self.is_train = is_train
        self.is_gan = is_gan
        self.is_trans_aug = is_trans_aug
        self.is_rot_aug = is_rot_aug
        if not from_pickle:
            self.data = self.preprocess(os.path.join(data_root, "data.json"), os.path.join(data_root, "label.json"))
            assert self.data['pose2d'].shape[0] == self.data['pose3d'].shape[0] == len(self.data['label'])
            self.normalize()
            if save_pickle:
                train_test_set = np.array([i for i in range(len(self.data['label']))])
                np.random.shuffle(train_test_set)
                split_len = int(train_set_ratio * len(train_test_set))
                self.train_data = {'pose2d': self.data['pose2d'][train_test_set[:split_len]],
                                   'pose3d': self.data['pose3d'][train_test_set[:split_len]],
                                   'label': self.data['label'][train_test_set[:split_len]],
                                   'joint_list': self.data['joint_list'][train_test_set[:split_len]]}
                self.test_data = {'pose2d': self.data['pose2d'][train_test_set[split_len:]],
                                  'pose3d': self.data['pose3d'][train_test_set[split_len:]],
                                  'label': self.data['label'][train_test_set[split_len:]],
                                  'joint_list': self.data['joint_list'][train_test_set[split_len:]]}
                pickle.dump(self.train_data, open(os.path.join(data_root, "processed_train_data.pkl"), 'wb'))
                pickle.dump(self.test_data, open(os.path.join(data_root, "processed_test_data.pkl"), 'wb'))
        else:
            self.train_data = pickle.load(open(os.path.join(data_root, "processed_train_data.pkl"), 'rb'))
            self.test_data = pickle.load(open(os.path.join(data_root, "processed_test_data.pkl"), 'rb'))
            assert self.train_data['pose2d'].shape[0] == self.train_data['pose3d'].shape[0] == len(self.train_data['label'])
            assert self.test_data['pose2d'].shape[0] == self.test_data['pose3d'].shape[0] == len(self.test_data['label'])

        if self.is_gan:
            self.rand_list = []
            for i in range(7, 14):
                self.rand_list.append([1 for j in range(i)])
                self.rand_list[-1].extend([0 for j in range(13 - i)])

    def __len__(self):
        if self.is_train:
            return len(self.train_data['label'])
        else:
            return len(self.test_data['label'])

    def rand_rotation_matrix(self, deflection=1.0, randnums=None):
        """
        Creates a random rotation matrix.

        deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
        rotation. Small deflection => small perturbation.
        randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
        """
        # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

        if randnums is None:
            randnums = self.rnd.uniform(size=(3,))

        theta, phi, z = randnums

        theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
        phi = phi * 2.0 * np.pi  # For direction of pole deflection.
        z = z * 2.0 * deflection  # For magnitude of pole deflection.

        # Compute a vector V used for distributing points over the sphere
        # via the reflection I - V Transpose(V).  This formulation of V
        # will guarantee that if x[1] and x[2] are uniformly distributed,
        # the reflected points will be uniform on the sphere.  Note that V
        # has length sqrt(2) to eliminate the 2 in the Householder matrix.

        r = np.sqrt(z)
        Vx, Vy, Vz = V = (
            np.sin(phi) * r,
            np.cos(phi) * r,
            np.sqrt(2.0 - z)
        )

        st = np.sin(theta)
        ct = np.cos(theta)

        R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

        # Construct the rotation matrix  ( V Transpose(V) - I ) R.

        M = (np.outer(V, V) - np.eye(3)).dot(R)
        return M

    def __getitem__(self, index):
        if self.is_trans_aug:
            random_trans = self.rnd.normal(0, 5, (3, ))
        else:
            random_trans = np.array([0, 0, 0])
        if self.is_rot_aug:
            random_rot = self.rand_rotation_matrix()
        else:
            random_rot = np.eye(3)
        if self.is_gan:
            if self.is_train:
                pose3d = np.matmul(random_rot, self.train_data['pose3d'][index, :].T).T
                pose3d[self.train_data['joint_list'][index] > 0.5] += random_trans
                return self.train_data['pose2d'][index, :], pose3d, \
                       self.train_data['label'][index], self.train_data['joint_list'][index], \
                       np.array(random.sample(self.rand_list[random.randint(0, 6)], 13)), \
                       np.array(random.sample(self.rand_list[random.randint(0, 6)], 13))
            else:
                pose3d = np.matmul(random_rot, self.test_data['pose3d'][index, :].T).T
                pose3d[self.train_data['joint_list'][index] > 0.5] += random_trans
                return self.test_data['pose2d'][index, :], pose3d, \
                       self.test_data['label'][index], self.test_data['joint_list'][index]
        else:
            if self.is_train:
                pose3d = np.matmul(random_rot, self.train_data['pose3d'][index, :].T).T
                pose3d[self.train_data['joint_list'][index] > 0.5] += random_trans
                return self.train_data['pose2d'][index, :], pose3d, \
                       self.train_data['label'][index], self.train_data['first_level_scores'][index]
            else:
                pose3d = np.matmul(random_rot, self.test_data['pose3d'][index, :].T).T
                pose3d[self.test_data['joint_list'][index] > 0.5] += random_trans
                return self.test_data['pose2d'][index, :], pose3d, \
                       self.test_data['label'][index], self.test_data['first_level_scores'][index]


    def normalize(self):
        for i in range(len(self.data['label'])):
            if np.sum(self.data['pose3d'][i, 8, :]) != 0 and np.sum(self.data['pose3d'][i, 10, :]) != 0:
                thigh_len = np.linalg.norm(np.abs(self.data['pose3d'][i, 8, :] - self.data['pose3d'][i, 10, :]))
                self.data['pose3d'][i,...] = self.data['pose3d'][i,...] / thigh_len
            else:
                self.data['pose3d'][i,...] = self.data['pose3d'][i,...] / ave_thigh_len

    @staticmethod
    def preprocess(data_json, label_json):
        raw_data = json.load(open(data_json))['AnnotationList']
        raw_label = json.load(open(label_json))
        new_2dpose = np.zeros((13, 2))
        new_3dpose = np.zeros((13, 3))
        new_joint = np.zeros((13, ))
        current_frame_num = 0
        label_id = 0
        labeled_2dpose = []
        labeled_3dpose = []
        labeled_joint = []
        joint_count = 0
        label = []
        for item in raw_data:
            if current_frame_num != item['FrameNumber']:
                current_frame_num = item['FrameNumber']
                if joint_count > 6:
                    labeled_2dpose.append(new_2dpose)
                    labeled_3dpose.append(new_3dpose)
                    labeled_joint.append(new_joint)
                    label.append(pose2id[raw_label[label_id]['MotorActionFriendlyName']])
                new_2dpose = np.zeros((13, 2))
                new_3dpose = np.zeros((13, 3))
                new_joint = np.zeros((13,))
                joint_count = 0
                if current_frame_num > raw_label[label_id]['EndFrame']:
                    label_id += 1
                    if label_id == len(raw_label):
                        break
            if current_frame_num < raw_label[label_id]['StartFrame']:
                continue
            if type(item['WorldX']) != str:
                new_2dpose[item['Part'], :] = [item['LocationX'], item['LocationY']]
                new_3dpose[item['Part'], :] = [item['WorldX'], item['WorldY'], item['WorldZ']]
                new_joint[item['Part']] = 1
                joint_count += 1
        return {'pose2d': np.array(labeled_2dpose), 'pose3d': np.array(labeled_3dpose), 'label': np.array(label),
                'joint_list': np.array(labeled_joint)}

skeletons = [[0, 2], [0, 1], [1, 2], [2, 4], [4, 6], [1, 3], [3, 5], [7, 8], [2, 8], [1, 7], [8, 10], [10, 12], [7, 9],
             [9, 11]]

def plot_pose(pose3d=None):
    if pose3d is None:
        train_set = pickle.load(open('processed_test_data.pkl', 'rb'))

        for i, pose3d in enumerate(train_set["pose3d"]):
            if sum(train_set['joint_list'][i]) > 11:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                for i in range(pose3d.shape[0]):
                    if np.sum(pose3d[i]) != 0:
                        ax.scatter(pose3d[i, 0], pose3d[i, 1], pose3d[i, 2])
                for skeleton in skeletons:
                    if np.sum(pose3d[skeleton[0]]) * np.sum(pose3d[skeleton[1]]) != 0:
                        ax.plot([pose3d[skeleton[0], 0], pose3d[skeleton[1], 0]],
                                [pose3d[skeleton[0], 1], pose3d[skeleton[1], 1]],
                                [pose3d[skeleton[0], 2], pose3d[skeleton[1], 2]])
                plt.show()
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(pose3d.shape[0]):
            if np.sum(pose3d[i]) != 0:
                ax.scatter(pose3d[i, 0], pose3d[i, 1], pose3d[i, 2])
        for skeleton in skeletons:
            if np.sum(pose3d[skeleton[0]]) * np.sum(pose3d[skeleton[1]]) != 0:
                ax.plot([pose3d[skeleton[0], 0], pose3d[skeleton[1], 0]],
                        [pose3d[skeleton[0], 1], pose3d[skeleton[1], 1]],
                        [pose3d[skeleton[0], 2], pose3d[skeleton[1], 2]])
        plt.show()


if __name__ == '__main__':
    # test = PoseDataset('data', save_pickle=True, from_pickle=False)
    test = PoseDataset('data', save_pickle=False, from_pickle=True, is_train=False, is_trans_aug=True, is_rot_aug=True)
    print(test[0][1])
    plot_pose(test[0][1])
