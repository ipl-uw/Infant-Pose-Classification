import argparse
import numpy as np
import cv2

import torch
import random
import numpy as np
from posenet import PoseNet
from posenetwm import PoseNetwM
from baseline import BaseNet
from posenetwot import PoseNetwoT
from posedataset import PoseDataset
from config import ave_thigh_len, pose2id, id2pose

def parse_args():
    parser = argparse.ArgumentParser(description='Pose Classifier')
    parser.add_argument('-m', dest='model', type=str, default='PoseNet')
    parser.add_argument('-r', dest='resume', type=str, default='net/PoseNet/model_100.pth')
    parser.add_argument('-d', dest='data', type=str, default='data/sample_7/psEst3d_smooth.txt')
    args = parser.parse_args()
    return args





def stable_output(predict,window_size=7):
    pred_length = len(predict)
    k = round((window_size-1)/2)   
    temp = predict.astype('int64')
             
    for i in range(pred_length):
        if(i<=k):
            predict[i] = findmode(temp[0:window_size])
        elif(i>=pred_length-k-1):
            predict[i] = findmode(temp[pred_length-window_size:pred_length])
        else:
            predict[i] = findmode(temp[i-k-1:i+k])
            
    return predict


def findmode(data_array):
    counts = np.bincount(data_array)
    return np.argmax(counts)







if __name__ == '__main__':
    args = parse_args()
    assert args.data != '' and args.resume != ''

    data_file = open(args.data)
    data = []
    pose_mappings = [[2, 2], [5, 1], [3, 4], [4, 6], [6, 3], [7, 5], [8, 8], [11, 7], [9, 10], [10, 12], [11, 7], [12, 9], [13, 11], [0, 0]]
    coarse_level_id = ["Prone", "Sitting", "Standing", "Supine"]
    
    
    line_number = 0
    for line in data_file:
        line_number = line_number+1
        line = line.rstrip().split(',')[1:]
        pose3d = np.zeros((13, 3))
        for i in range(13):
            pose3d[i,:] = line[i * 3: i * 3 + 3]
        if np.sum(pose3d[8, :]) != 0 and np.sum(pose3d[10, :]) != 0:
            thigh_len = np.linalg.norm(np.abs(pose3d[8, :] - pose3d[10, :]))
            pose3d /= thigh_len
        else:
            pose3d /= ave_thigh_len
        data.append(pose3d)
    data = torch.FloatTensor(np.array(data)).cuda()
    
        
    first_level_scores_file = open('data/sample_7/classification results.txt')
    coarse_level = np.zeros((line_number,11))
    coarse_res = np.zeros(line_number)
    count = 0
    for line in first_level_scores_file:
        if (count>4.5):
            line = line.rstrip().split(',')[2:]
            coarse_res[count-5] = line[0]
            coarse_level[count-5,0] = line[1]
            coarse_level[count-5,1] = line[1]
            coarse_level[count-5,2] = line[1]
            coarse_level[count-5,3] = line[1]
            coarse_level[count-5,4] = line[2]
            coarse_level[count-5,5] = line[2]
            coarse_level[count-5,6] = line[2]
            coarse_level[count-5,7] = line[3]
            coarse_level[count-5,8] = line[4]
            coarse_level[count-5,9] = line[4]
            coarse_level[count-5,10] = line[4]
        count = count+1      
    coarse_level = torch.from_numpy(coarse_level)
    coarse_level = coarse_level.cuda().float()
        
    if args.model == 'PoseNet':
        net = PoseNet()
    elif args.model == 'BaseNet':
        net = BaseNet()
    elif args.model == 'PoseNetwM':
        net = PoseNetwM()
    elif args.model == 'PoseNetwoT':
        net = PoseNetwoT()
    net.cuda()

    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['model_state_dict'])

    outputs = net(data,coarse_level)
    
    _, preds = torch.max(outputs, 1)
    print(preds)

    preds = preds.cpu().detach().numpy()
    preds = stable_output(preds,7)
    preds = stable_output(preds,11)
    print(preds)
    

    cap = cv2.VideoCapture('data/sample_7/sample_7_huber.avi')
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 100)
    fontScale = 1.5
    fontColor = (0, 0, 0)
    lineType = 2

    # Read until video is completed
    count = 5
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        cv2.putText(frame, 'Coarse level: '+coarse_level_id[int(coarse_res[count-5])],
                    (10, 50),
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        cv2.putText(frame, 'Fine level: '+id2pose[preds[count-5]],
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        cv2.imwrite('out/%04d.jpg' % count, frame)
        count += 1

