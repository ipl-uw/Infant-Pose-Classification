import argparse
import os
import numpy as np
import sklearn.metrics

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from posenet import PoseNet
from posenetwm import PoseNetwM
from baseline import BaseNet
from posenetwot import PoseNetwoT
from posedataset import PoseDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Pose Classifier')
    parser.add_argument('-m', dest='model', type=str, default='PoseNet')
    parser.add_argument('-r', dest='resume', type=str, default='')
    parser.add_argument('--trans_aug', dest='trans_aug', action='store_true')
    parser.add_argument('--rot_aug', dest='rot_aug', action='store_true')
    args = parser.parse_args()
    return args


def label_trans(original_label):
    temp = original_label.numpy()
    length = len(temp)   
    res = np.zeros(length)    
    for i in range(length):
        if((temp[i]==0) or (temp[i]==1) or (temp[i]==2) or (temp[i]==3)):
            res[i] = 0 #prone
        elif((temp[i]==4) or (temp[i]==5) or (temp[i]==6)):
            res[i] = 1 #sitting
        elif(temp[i]==7):
            res[i] = 2 #standing     
        else:
            res[i] = 3
    res = torch.from_numpy(res)
    res = res.int()
    return res


def scores_trans(original_scores):
    temp = original_scores.numpy()
    m,n = temp.shape
    res = np.zeros((m,11))
    for i in range(m):
        res[i,0] = temp[i,0]
        res[i,1] = temp[i,0]
        res[i,2] = temp[i,0]
        res[i,3] = temp[i,0]
        res[i,4] = temp[i,1]
        res[i,5] = temp[i,1]
        res[i,6] = temp[i,1]
        res[i,7] = temp[i,2]
        res[i,8] = temp[i,3]
        res[i,9] = temp[i,3]
        res[i,10] = temp[i,3]
    res = torch.from_numpy(res)
    return res




if __name__ == '__main__':
    args = parse_args()

    dataset = PoseDataset('data', from_pickle=True, save_pickle=False, is_trans_aug=args.trans_aug, is_rot_aug=args.rot_aug)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    test_dataset = PoseDataset('data', from_pickle=True, save_pickle=False, is_train=False, is_trans_aug=args.trans_aug, is_rot_aug=args.rot_aug)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    if args.model == 'PoseNet':
        net = PoseNet()
    elif args.model == 'BaseNet':
        net = BaseNet()
    elif args.model == 'PoseNetwM':
        net = PoseNetwM()
    elif args.model == 'PoseNetwoT':
        net = PoseNetwoT()
    net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    epoch_start = 0

    if len(args.resume) != 0:
        checkpoint = torch.load(args.resume)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        net.load_state_dict(checkpoint['model_state_dict'])
        epoch_start = checkpoint['epoch']

    #criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()

    epoch = 60
    num_iter = len(dataset) // 128 + 1
    for i in range(epoch_start, epoch):
        net.train()
        total_loss = 0
        #for idx, (pose_2d, pose_3d, label) in enumerate(dataloader):
        for idx, (pose_2d, pose_3d, label, first_level_scores) in enumerate(dataloader):
            optimizer.zero_grad()
            net.zero_grad()

            #label_m = label_trans(label)
            #label = label_m.cuda().long()
            label = label.cuda().long()

            pose_3d = pose_3d.cuda().float()

            temp_scores = scores_trans(first_level_scores)
            temp_scores = temp_scores.cuda().float()

            #first_level_scores = first_level_scores.cuda().float()
            #pred = net(pose_3d)
            pred = net(pose_3d , temp_scores)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            if idx % 10 == 0:
                print('\r epoch \t\t[%d/%d] | iter \t\t[%d/%d] | loss: \t%f' % (i, epoch, idx, num_iter, loss.item()), end='')
            total_loss += loss.item()
        print('\n average loss: %f' % (total_loss / num_iter))
        scheduler.step()
        if i%10==9:
            net.eval()
            nb_classes = 11

            class_label_list = [[0 for b in range(len(test_dataset))] for a in range(nb_classes)]
            class_pred_list = [[0 for b in range(len(test_dataset))] for i in range(nb_classes)]

            with torch.no_grad():
                for idx, (_, pose_3d, label, first_level_scores) in enumerate(test_dataloader):
                        pose_3d = pose_3d.cuda().float()

                        temp_scores = scores_trans(first_level_scores)
                        temp_scores = temp_scores.cuda().float()
                        #first_level_scores = first_level_scores.cuda().float()
                        
                        #label_m = label_trans(label)
                        #label = label_m.cuda().long()
                        
                        label = label.cuda().long()
                        outputs = net(pose_3d, temp_scores)
                        _, preds = torch.max(outputs, 1)
                        for cid, (t, p) in enumerate(zip(label.view(-1), preds.view(-1))):
                            class_label_list[t.long()][idx * 128 + cid] = 1
                            class_pred_list[p.long()][idx * 128 + cid] = 1

            total_AP = 0
            for a in range(nb_classes):
                total_AP += (sklearn.metrics.average_precision_score(class_label_list[a], class_pred_list[a]) * sum(class_label_list[a]))
                print('AP for %02d: %.04f' % (a, sklearn.metrics.average_precision_score(class_label_list[a], class_pred_list[a])))
            print('mAP: %.04f' % (total_AP / len(test_dataset)))

            # confusion_matrix = torch.zeros(nb_classes, nb_classes)
            # with torch.no_grad():
            #     for idx, (_, pose_3d, label) in enumerate(test_dataloader):
            #         pose_3d = pose_3d.cuda().float()
            #         label = label.cuda().long()
            #         outputs = net(pose_3d)
            #         _, preds = torch.max(outputs, 1)
            #         for t, p in zip(label.view(-1), preds.view(-1)):
            #             confusion_matrix[t.long(), p.long()] += 1
            # confusion_matrix.int()
            # # print(confusion_matrix)
            # print(confusion_matrix.diag() / confusion_matrix.sum(1))
            # print(torch.sum(confusion_matrix.diag()) / torch.sum(confusion_matrix))
            # correct = 0
            # total = 0
            # with torch.no_grad():
            #     for _, pose_3d, label in test_dataloader:
            #         pose_3d = pose_3d.cuda().float()
            #         label = label.cuda().long()
            #         outputs = net(pose_3d)
            #         predicted = torch.max(outputs.data, 1)
            #         total += label.size(0)
            #         correct += (predicted[1] == label).sum().item()
            # print(correct, total)
            # print('Accuracy of the network on the %d test poses: %.04f %%' % (total, 100 * correct / total))
            status_dict = {
                'epoch': i,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            save_model_path = os.path.join('net/%s/model_%03d.pth' % (args.model, i + 1))
            torch.save(status_dict, save_model_path)
