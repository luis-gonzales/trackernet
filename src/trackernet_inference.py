import cv2
import json
import argparse
import numpy as np
import tensorflow as tf

from helpers import sigmoid
from tracknet_loss import tracknet_loss
from generator import get_feat_and_label


def preds_to_bbox(t_bbox, obj_row):
	# Convert CNN output to (top-left x, top-left y, width, height)

	tx, ty, tw, th = t_bbox

	cx, cy = (obj_row % 3), (obj_row // 3)	# offsets in [0, 1, 2]

	bx = 64 * (sigmoid(tx) + cx)
	by = 64 * (sigmoid(ty) + cx)
	bw = 96 * np.exp(tw)
	bh = 96 * np.exp(th)
	
	x = int(round(bx - bw/2))
	y = int(round(by - bh/2))
	w = int(round(bw))
	h = int(round(bh))

	return x, y, w, h


# Grab model and JSON file
parser = argparse.ArgumentParser()
parser.add_argument('model_file', help='saved model file (.h5)', type=str)
parser.add_argument('json_file', help='json file with adjacent frame association', type=str)

args = parser.parse_args()
model_file = args.model_file
json_file  = args.json_file


# Load model, data to perform inference on
model = tf.keras.models.load_model(model_file, custom_objects={'tracknet_loss': tracknet_loss})

file = open(json_file, mode='r')
input_data = json.load(file)
file.close()


# Transform input JSON to usable input to CNN
[feat_a, feat_b], _ = get_feat_and_label(input_data)
feat_a = np.expand_dims(feat_a, axis=0)
feat_b = np.expand_dims(feat_b, axis=0)


# Run prediction
y_hat = model.predict([feat_a, feat_b])
y_hat = y_hat[0,:,:]			# Single batch assumed


# Find row with highest confidence
conf_logits, bboxes, _ = np.split(y_hat, [1,5], axis=1)
confs = sigmoid(conf_logits)

obj_row  = int(np.argmax(confs))
max_conf = float(confs[obj_row])


# If high enough confidence, draw rectangle on second frame
if max_conf > 0.8:

	# Convert from CNN output to (top-left x, top-left y, width, height) w.r.t. `feat_b`
	x, y, w, h = preds_to_bbox(bboxes[obj_row], obj_row)

	# Parameters corresponding to `bbox_a`
	x_a, y_a, w_a, h_a = input_data['bbox_a']

	center_x_a, center_y_a = x_a + w_a/2, y_a + h_a/2
	max_dim   = max(w_a, h_a)
	img_ratio = 2*max_dim/192	# FOV and `feat_a` shape used

	# Transform (x,y,w,h) w.r.t. `feat_b` to (xx,yy,ww,hh) w.r.t. second frame
	xx = round(x*img_ratio + center_x_a - max_dim)
	yy = round(y*img_ratio + center_y_a - max_dim)
	ww = round(w * img_ratio)
	hh = round(h * img_ratio)

	# Read in second frame, draw appropriate rectangle, and save image
	out_img = cv2.imread(input_data['frame_b'])
	cv2.rectangle(out_img, (xx, yy), (xx+ww, yy+hh), (0,255,255), thickness=2)
	cv2.imwrite(json_file[:-4] + 'jpg', out_img)
