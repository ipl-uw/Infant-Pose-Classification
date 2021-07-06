import numpy as np

def get_keypoints(heatmap):
    keypoints = []
    for i in range(heatmap.shape[0]):
        index = np.argmax(heatmap[i, :, :])
        keypoints.append([int(index % heatmap.shape[2]), int(index // heatmap.shape[2]), np.max(heatmap[i, :, :])])
    return keypoints
