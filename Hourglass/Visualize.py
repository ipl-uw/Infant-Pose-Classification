import numpy as np
import glob
import cv2
import sys,os

skeletons =  [[0, 2], [1, 3], [0, 1], [2, 3], [9, 11], [10, 12], [9, 10], [11, 12], [4, 0], [4, 9], [4, 5], [5, 1], [5, 10], [6, 2], [6, 11], [7, 3], [7, 12], [6, 7]]

def drawCar(img,keypoints,bb=None, thre=2):
#    if  keypoints[0,2] >50 and keypoints[2,2] >50:
#        cv2.line(img,tuple(keypoints[0,0:2]),tuple(keypoints[2,0:2]),(255,0,0),5)
#    if  keypoints[1,2] >50 and keypoints[3,2] >50:
#        cv2.line(img,tuple(keypoints[1,0:2]),tuple(keypoints[3,0:2]),(0,0,255),5)
#    if  keypoints[0,2] >50 and keypoints[1,2] >50:
#        cv2.line(img,tuple(keypoints[0,0:2]),tuple(keypoints[1,0:2]),(0,255,0),5)
#    if  keypoints[3,2] >50 and keypoints[2,2] >50:
#        cv2.line(img,tuple(keypoints[3,0:2]),tuple(keypoints[2,0:2]),(128,128,0),5)
    threshold = thre
    # wheels
    if  keypoints[1,2] >threshold:
        cv2.circle(img,tuple(keypoints[1,0:2]),3,(0,255,0),3)
    if  keypoints[2,2] >threshold:
        cv2.circle(img,tuple(keypoints[2,0:2]),3,(64,255,255),3)
    if  keypoints[3,2] >threshold:
        cv2.circle(img,tuple(keypoints[3,0:2]),3,(128,255,128),3)
    if  keypoints[0,2] >threshold:
        cv2.circle(img,tuple(keypoints[0,0:2]),3,(128,255,0),3)

    for skeleton in skeletons:
        if  keypoints[skeleton[0],2] >threshold and keypoints[skeleton[1],2] >threshold:
            cv2.line(img,tuple(keypoints[skeleton[0],0:2]),tuple(keypoints[skeleton[1],0:2]),(255,0,0),2)

    # top of car
    if  keypoints[9,2] >threshold:
        	cv2.circle(img,tuple(keypoints[9,0:2]),3,(255,128,128),3)
    if  keypoints[11,2] >threshold:
        	cv2.circle(img,tuple(keypoints[11,0:2]),3,(128,128,128),3)
    if  keypoints[12,2] >threshold:
        	cv2.circle(img,tuple(keypoints[12,0:2]),3,(0,128,255),3)
    if  keypoints[13,2] >threshold:
        	cv2.circle(img,tuple(keypoints[13,0:2]),3,(0,255,255),3)
        
    # front head lights
    if  keypoints[4,2] >threshold:
        	cv2.circle(img,tuple(keypoints[4,0:2]),3,(0,255,0),3)

    if  keypoints[5,2] >threshold:
        cv2.circle(img,tuple(keypoints[5,0:2]),3,(128,0,0),3)

    # back head lights
    if  keypoints[6,2] >threshold:
        	cv2.circle(img,tuple(keypoints[6,0:2]),3,(255,0,0),3)
    
    if  keypoints[7,2] >threshold:
        	cv2.circle(img,tuple(keypoints[7,0:2]),3,(255,0,128),5)
    # mirrror
    if  keypoints[8,2] >threshold:
        	cv2.circle(img,tuple(keypoints[8,0:2]),3,(128,0,128),5)
    
    if  keypoints[10,2] >threshold:
        cv2.circle(img,tuple(keypoints[10,0:2]),3,(0,128,128),5)
    
    return img
    
    #cv2.line(img,tuple(keypoints[0,0:2]),tuple(keypoints[2,0:2]),(255,0,0),1)
    #cv2.line(img,tuple(keypoints[4,0:2]),tuple(keypoints[6,0:2]),(0,255,0),1)
    #cv2.line(img,tuple(keypoints[5,0:2]),tuple(keypoints[7,0:2]),(0,255,0),1)
    #cv2.line(img,tuple(keypoints[4,0:2]),tuple(keypoints[5,0:2]),(0,255,0),1)
    #cv2.line(img,tuple(keypoints[6,0:2]),tuple(keypoints[7,0:2]),(0,255,0),1)

    #cv2.line(img,tuple(keypoints[1]),tuple(keypoints[2]),(0,255,0),5)
    #cv2.line(img,tuple(keypoints[2]),tuple(keypoints[3]),(0,255,0),5)
    #cv2.line(img,tuple(keypoints[0]),tuple(keypoints[3]),(0,255,0),5)
    #print(keypoints[0]- [20,20])

def drawPerson(img,keypoints,bb=None):
    threshold = 10
    # wheels
    if  keypoints[0,2] >threshold:
        	cv2.circle(img,tuple(keypoints[0,0:2]),3,(128,255,0),2)
    if  keypoints[1,2] >threshold:
        cv2.circle(img,tuple(keypoints[1,0:2]),3,(0,255,0),2)
    if  keypoints[2,2] >threshold:
        cv2.circle(img,tuple(keypoints[2,0:2]),3,(64,255,255),2)
    if  keypoints[3,2] >threshold:
        	cv2.circle(img,tuple(keypoints[3,0:2]),3,(128,255,128),2)
    if  keypoints[4,2] >threshold:
        	cv2.circle(img,tuple(keypoints[4,0:2]),3,(0,255,0),2)
    if  keypoints[5,2] >threshold:
        cv2.circle(img,tuple(keypoints[5,0:2]),3,(128,0,0),2)
    if  keypoints[6,2] >threshold:
        	cv2.circle(img,tuple(keypoints[6,0:2]),3,(255,0,0),2)
    if  keypoints[7,2] >threshold:
        	cv2.circle(img,tuple(keypoints[7,0:2]),3,(255,0,128),2)
    if  keypoints[8,2] >threshold:
        	cv2.circle(img,tuple(keypoints[8,0:2]),3,(128,0,128),2)
    if  keypoints[9,2] >threshold:
        cv2.circle(img,tuple(keypoints[9,0:2]),3,(0,128,128),2)
    if  keypoints[10,2] >threshold:
        	cv2.circle(img,tuple(keypoints[10,0:2]),3,(255,128,128),2)
    if  keypoints[11,2] >threshold:
        	cv2.circle(img,tuple(keypoints[11,0:2]),3,(128,128,128),2)
    if  keypoints[12,2] >threshold:
        	cv2.circle(img,tuple(keypoints[12,0:2]),3,(0,128,255),2)
    if  keypoints[13,2] >threshold:
        	cv2.circle(img,tuple(keypoints[13,0:2]),3,(0,255,255),2)
    if  keypoints[14,2] >threshold:
        	cv2.circle(img,tuple(keypoints[14,0:2]),3,(0,255,64),2)
    if  keypoints[15,2] >threshold:
        	cv2.circle(img,tuple(keypoints[15,0:2]),3,(0,64,255),2)

    if  keypoints[0,2] >threshold and keypoints[1,2] >threshold:
        cv2.line(img,tuple(keypoints[0,0:2]),tuple(keypoints[1,0:2]),(0,255,0),2)
    if  keypoints[1,2] >threshold and keypoints[2,2] >threshold:
        cv2.line(img,tuple(keypoints[1,0:2]),tuple(keypoints[2,0:2]),(0,255,0),2)
    if  keypoints[2,2] >threshold and keypoints[6,2] >threshold:
        cv2.line(img,tuple(keypoints[2,0:2]),tuple(keypoints[6,0:2]),(0,255,0),2)
    if  keypoints[3,2] >threshold and keypoints[4,2] >threshold:
        cv2.line(img,tuple(keypoints[3,0:2]),tuple(keypoints[4,0:2]),(0,255,0),2)
    if  keypoints[3,2] >threshold and keypoints[6,2] >threshold:
        cv2.line(img,tuple(keypoints[3,0:2]),tuple(keypoints[6,0:2]),(0,255,0),2)
    if  keypoints[4,2] >threshold and keypoints[5,2] >threshold:
        cv2.line(img,tuple(keypoints[4,0:2]),tuple(keypoints[5,0:2]),(0,255,0),2)
    if  keypoints[6,2] >threshold and keypoints[8,2] >threshold:
        cv2.line(img,tuple(keypoints[6,0:2]),tuple(keypoints[8,0:2]),(0,255,0),2)
    if  keypoints[8,2] >threshold and keypoints[9,2] >threshold:
        cv2.line(img,tuple(keypoints[8,0:2]),tuple(keypoints[9,0:2]),(0,255,0),2)
    if  keypoints[13,2] >threshold and keypoints[8,2] >threshold:
        cv2.line(img,tuple(keypoints[13,0:2]),tuple(keypoints[8,0:2]),(0,255,0),2)
    if  keypoints[10,2] >threshold and keypoints[11,2] >threshold:
        cv2.line(img,tuple(keypoints[10,0:2]),tuple(keypoints[11,0:2]),(0,255,0),2)
    if  keypoints[11,2] >threshold and keypoints[12,2] >threshold:
        cv2.line(img,tuple(keypoints[11,0:2]),tuple(keypoints[12,0:2]),(0,255,0),2)
    if  keypoints[12,2] >threshold and keypoints[8,2] >threshold:
        cv2.line(img,tuple(keypoints[12,0:2]),tuple(keypoints[8,0:2]),(0,255,0),2)
    if  keypoints[13,2] >threshold and keypoints[14,2] >threshold:
        cv2.line(img,tuple(keypoints[13,0:2]),tuple(keypoints[14,0:2]),(0,255,0),2)
    if  keypoints[14,2] >threshold and keypoints[15,2] >threshold:
        cv2.line(img,tuple(keypoints[14,0:2]),tuple(keypoints[15,0:2]),(0,255,0),2)
        
 

class BoundingBox(object):
    """
    A 2D bounding box
    """
    def __init__(self, points):
        if len(points) == 0:
            raise ValueError("Can't compute bounding box of empty list")
        self.minx, self.miny = float("inf"), float("inf")
        self.maxx, self.maxy = float("-inf"), float("-inf")
        for x, y in points:
            # Set min coords
            if x < self.minx:
                self.minx = x
            if y < self.miny:
                self.miny = y
            # Set max coords
            if x > self.maxx:
                self.maxx = x
            elif y > self.maxy:
                self.maxy = y
    @property
    def width(self):
        return self.maxx - self.minx
    @property
    def height(self):
        return self.maxy - self.miny
    def __repr__(self):
        return "BoundingBox({}, {}, {}, {})".format(
            self.minx, self.maxx, self.miny, self.maxy)
# Usage example:

def _compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    if xA < xB and yA < yB:
        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0

    assert iou >= 0
    assert iou <= 1.01

    return iou

def get_keypoints(name):
	kp_all_new = []
	with open(name) as f:
		lines = f.readlines()
	bb = []
	points = []
	class_name = []
	for line in lines:
		bb.append(np.array(line.split(',')[1:5]).astype(np.float))
		points.append(np.array(line.split(',')[5:-1]).astype(np.float))
		class_name.append(line.split(',')[-1].split('\n')[0]) 
	for bb_loop,bb_num in enumerate(bb):
		points_array = np.array(points[bb_loop])#.splitlines()[0].split(','))
		points_arranged = points_array.reshape(int(len(points_array)/3),3)
		kp = points_arranged[:,0:3]
		kp = np.round(kp.astype(np.float)).astype(np.int)
		kp[:,0] = bb[bb_loop][0] + kp[:,0]*(bb[bb_loop][2]/64)
		kp[:,1] = bb[bb_loop][1] + kp[:,1]*(bb[bb_loop][3]/64)
		kp[:,2] = np.round(points_arranged[:,2].astype(np.float)*100).astype(np.int)
		kp_all_new.append(kp)
	return kp_all_new
             
def count_pck(bb_gt,bb_computed,alpha,B):
    #print(bb_gt,bb_computed)
    count = 0
    count_inliers = 0
    for loop,point in enumerate(bb_computed):
        if point[2]>10 and loop != 8 and loop != 9:
            dist = np.linalg.norm(point[0:2]-bb_gt[loop,0:2])
            #dist = np.linalg.norm(point[0]-bb_gt[loop,0])
            #print(point[0:2],bb_gt[loop,0:2],dist)
            #print(dist,alpha*B)
            count = count+1
            if dist<alpha*B:
                count_inliers = count_inliers+1
    return count_inliers,count
