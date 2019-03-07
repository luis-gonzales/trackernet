import cv2
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, 'src/')
from generator import get_feat_and_label
from helpers import sigmoid, preds_to_bbox


# Parse input arg
parser = argparse.ArgumentParser()
parser.add_argument('data', help='JSON file storing frames and bounding box', type=str)
args = parser.parse_args()

json_file = args.data


# Read in dictionary and process
file = open(json_file, mode='r')
data = json.load(file)
file.close()
imgs, label = get_feat_and_label(data)


# Visualize images (TrackerNet trained on RGB images; OpenCV uses BGR)
img_a = imgs[0][:, :, ::-1]
img_b = imgs[1][:, :, ::-1]
cv2.imwrite('test/img_a.jpg', img_a.astype(np.uint8) )
cv2.imwrite('test/img_b.jpg', img_b.astype(np.uint8) )


# Print training label and interpreted label
print('Raw label =\n', label)
obj_idx = int( np.nonzero(label[:,0])[0] )

x, y, w, h = preds_to_bbox(label[obj_idx, 1:], obj_idx, grid_sz=64, anchor_sz=96)
print('\nInterpreted:')
print('Top-left x =', x)
print('Top-left y =', y)
print('Width =', w)
print('Height =', h)
