import cv2
import numpy as np

from sklearn.utils import shuffle
from helpers import sigmoid

def pad_zeros(img, size):
	'''
	Fit image into shape `size`, padding with zeros at the bottom and
	right as necessary. Assumes `img` is already no bigger than (size x size)
	'''
	h, w = img.shape[:2]

	pix_w, pix_h = size - w, size - h

	return cv2.copyMakeBorder(img, top=0, bottom=pix_h, left=0,
							  right=pix_w, borderType=cv2.BORDER_CONSTANT)


def resize_with_ar(img, size):
	'''
	Resize `img` while maintaining aspect ratio
	'''
	h, w = img.shape[:2]
	
	ratio = float(size) / max(h, w)

	new_w, new_h = int(w*ratio), int(h*ratio)

	return cv2.resize(img, (new_w, new_h)), ratio


def extract_mult(img, bbox, mult):
	'''
	Extract subset of image centered at `bbox` with a "field of view" multiplier
	of `mult`.
	'''
	# Params of original `bbox`
	x, y, w, h = bbox
	max_dim = max(w,h)

	# Coordinates for output
	x1 = round(x+w/2 - max_dim*mult/2)
	y1 = round(y+h/2 - max_dim*mult/2)
	x2 = x1 + mult*max_dim
	y2 = y1 + mult*max_dim

	# Clipping
	img_h, img_w = img.shape[:2]
	x1, y1 = max(0,x1), max(0,y1)
	x2, y2 = min(img_w,x2), min(img_h,y2)

	return img[y1:y2, x1:x2, :], (x1,y1)


def inv_sigmoid(x):
	lim = 4.5
	if (x > 0.9998) and (x < 1.0002): return lim	# prevent div by zero
	
	ans = np.log( x / (1-x) )
	if ans > lim: return lim
	elif ans < -lim: return -lim
	else: return ans


def in_img(img_dims, bbox, min_pix=3):
	'''Boolean describing whether object at bboxx is within image (img_dims)'''
	img_w, img_h = img_dims
	x,y,w,h = bbox

	in_img = ( (x+w > min_pix) and (x < (img_w - min_pix)) ) \
		 and ( (y+h > min_pix) and (y < (img_h - min_pix)) )

	return in_img


def anchor_parse(vals):

	prev_sz, cur_sz = 96, 192
	
	if not in_img((cur_sz,cur_sz), vals):
		return -1, (0,0,0,0)

	x, y, w, h = vals
		
	# Handle extremes
	if y+h > cur_sz:
		h = cur_sz - y
	if x+w > cur_sz:
		w = cur_sz - x
	if x < 0:
		#w += x
		x = 0
	if y < 0:
		#h += y
		y = 0

	center_x, center_y = round(x + w/2), round(y + h/2)

	x_idx, y_idx = center_x // (cur_sz//3), center_y // (cur_sz//3)

	idx = 3*y_idx + x_idx

	# Normalize in [0.0, 1.0] and map to CNN parameter
	b_x = (center_x % (cur_sz/3)) / (cur_sz/3)
	b_y = (center_y % (cur_sz/3)) / (cur_sz/3)
	t_x = inv_sigmoid(b_x)
	t_y = inv_sigmoid(b_y)

	# Map to CNN parameter given fixed anchor box
	anchor_sz = 96	# square
	tw = np.log(w/anchor_sz)
	th = np.log(h/anchor_sz)

	return idx, (t_x, t_y, tw, th)


def hsv_aug(img, hue_aug, sat_aug, min_coeff):
	'''Add hue and sat shift based on random coefficient'''
	hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

	if hue_aug:
		hue_coeff = np.random.uniform(low=min_coeff, high=1.0)
		hsv[:,:,0] = hue_coeff * hsv[:,:,0]

	if sat_aug:
		sat_coeff = np.random.uniform(low=min_coeff, high=1.0)
		hsv[:,:,1] = sat_coeff * hsv[:,:,1]

	return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def get_feat_and_label(dict_desc, data_aug=False):
	'''
	Load images and create ground-truth label based on dict_desc,
	which is expected to contain:
		'frame_a': path to the "previous" frame
		'frame_b': path to the "current" frame
		'bbox_a': bounding box for object of interest in `frame_a` in the
				  form of (top-left x, top-left y, w, h)
		'bbox_b': similar to `bbox_a` but for `frame_b`; if omitted, it's
				  assumed that object of interest has left the scene

	First iteration of TrackerNet fixes the "previous" crop to 96 x 96
	and the "current" crop to 192 x 192
	'''

	prev_sz, cur_sz = 96, 192

	# Image slicing for "previous" frame
	img_a = cv2.imread(dict_desc['frame_a'])

	x_a, y_a, w_a, h_a = dict_desc['bbox_a']	# x, y correspond to top-left coord
	
	# Handle edge bases
	if x_a < 0:
		w_a = w_a + x_a
		x_a = 0
	if y_a < 0:
		h_a = h_a + y_a
		y_a = 0
	if y_a + h_a > img_a.shape[0]:
		h_a = img_a.shape[0] - y_a
	if x_a+w_a > img_a.shape[1]:
		w_a = img_a.shape[1] - x_a

	# Extract crop, resize, and pad
	feat_a = img_a[y_a : y_a+h_a, x_a : x_a+w_a, :]
	feat_a, _ = resize_with_ar(feat_a, prev_sz)
	feat_a = pad_zeros(feat_a, prev_sz)
	feat_a = feat_a[:, :, ::-1] # RGB


	# Image slicing for "current" frame (extract, resize, and pad)
	img_b = cv2.imread(dict_desc['frame_b'])
	feat_b, (x_orig, y_orig) = extract_mult(img_b, dict_desc['bbox_a'], mult=2)
	feat_b, ratio_b = resize_with_ar(feat_b, cur_sz)
	feat_b = pad_zeros(feat_b, cur_sz)
	feat_b = feat_b[:, :, ::-1]	# RGB


	# Data augmentation
	if data_aug:
		
		aug_hue_a, aug_sat_a, aug_hue_b, aug_sat_b = \
			np.random.rand(4) > 0.5

		if aug_sat_a or aug_hue_a:
			feat_a = hsv_aug(feat_a, aug_hue_a, aug_sat_a, min_coeff=0.3)

		if aug_sat_b or aug_hue_b:
			feat_b = hsv_aug(feat_b, aug_hue_b, aug_sat_b, min_coeff=0.3)


	# Create ground-truth label
	label = np.zeros((9,5), dtype=np.float32)
	if ('bbox_b' in dict_desc):
		x_b, y_b, w_b, h_b = dict_desc['bbox_b']

		# Re-scale coords for "current" frame crop
		x = round((x_b - x_orig)*ratio_b)
		y = round((y_b - y_orig)*ratio_b)
		w, h = round(w_b*ratio_b), round(h_b*ratio_b)

		idx, vals = anchor_parse((x, y, w, h))	# map to CNN parameters
		print('idx =', idx)

		if idx != -1:
			label[idx, 0] = 1.0
			label[idx, 1:] = vals

	return [feat_a, feat_b], label


def generator(gen_entries, abs_path, batch_sz):

	num_entries = len(gen_entries)

	while True:
		entries = shuffle(gen_entries)	# shuffle at every epoch

		for offset in range(0, num_entries, batch_sz):

			batch_entries = entries[offset : offset+batch_sz]
			imgs1, imgs2, labels = [], [], []

			for entry in batch_entries:
				input_entry = {'frame_a': abs_path + entry['frame_a'],
							   'frame_b': abs_path + entry['frame_b'],
							   'bbox_a': entry['bbox_a'],
							   'bbox_b': entry['bbox_b']}

				X_cur, y_cur = get_feat_and_label(input_entry, data_aug=True)
				imgs1.append(X_cur[0])
				imgs2.append(X_cur[1])
				labels.append(y_cur)

			yield [np.array(imgs1), np.array(imgs2)], np.array(labels)
