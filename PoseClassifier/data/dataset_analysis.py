import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import collections
import pickle
import math
from statistics import stdev

from PoseClassifier.posedataset import PoseDataset

def distribution_analyze():
    pkl_list = ['processed_train_data.pkl', 'processed_test_data.pkl']
    hist = {}
    for pkl in pkl_list:
        train_set = pickle.load(open(pkl, 'rb'))
        for joint_list in train_set["joint_list"]:
            if sum(joint_list) not in hist.keys():
                hist[sum(joint_list)] = 1
            else:
                hist[sum(joint_list)] += 1
    count = 0
    hist = collections.OrderedDict(sorted(hist.items(), reverse=False))
    for key in sorted(hist.keys()):
        count += hist[key]
    print(count)
    D = hist
    print(D.keys())
    plt.bar(range(7, 13), D.values(), align='center')
    print()
    plt.show()


def orientation_analyze():
    train_set = PoseDataset('./', from_pickle=True, save_pickle=False, is_trans_aug=False, is_rot_aug=True, is_train=False)
    vector = []
    total_length = []
    for _, pose3d, _ in train_set:
        if np.sum(pose3d[2]) * np.sum(pose3d[8]) != 0:
            vector.append(pose3d[2] - pose3d[8])
            total_length.append(math.sqrt(vector[-1][0]**2 + vector[-1][1]**2 + vector[-1][2]**2))
    vector = np.array(vector)
    print(sum(total_length) / len(total_length), stdev(total_length))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vector[:, 0], vector[:, 1], vector[:, 2])
    plt.show()


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
    plot_pose()
