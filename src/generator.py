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
	# vals is tuple w/ (x, y, w, h)
	#print('--- in anchor_parse ---')
	if not in_img((192,192), vals):
		#print('outside of 192 x 192 crop')
		return -1, (0,0,0,0)

	x, y, w, h = vals
	#print(x,y,w,h)

	#in_frame = (y < 192) and (x < 192) and (x+w > 0) and (y+h > 0)
	##print('in_frame =', in_frame)

	#if not in_frame:
		

	if y+h > 192:
		h = 192 - y
	if x+w > 192:
		w = 192 - x
	if x < 0:
		w = w + x
		x = 0
	if y < 0:
		h = h + y
		y = 0
	#print('new bbox =', x, y, w, h)

	center_x, center_y = round(x + w/2), round(y + h/2)

	#if center_y > 192: center_y = 192

	#print('center =', center_x, center_y)

	x_idx, y_idx = center_x // 64, center_y // 64
	##print('idx =', x_idx, y_idx)

	idx = 3*y_idx + x_idx
	

	b_x = (center_x % 64) / 64
	b_y = (center_y % 64) / 64
	#print('b_x, b_y =', b_x, b_y)

	t_x = inv_sigmoid(b_x)
	t_y = inv_sigmoid(b_y)

	#print('b_x, b_y =', b_x, b_y)
	#print('t_x, t_y =', t_x, t_y)
	#print('recovered =', sigmoid(t_x), sigmoid(t_y))

	tw = np.log(w/96)
	th = np.log(h/96)

	#print('recovered w and h =', 144*np.exp(tw), 144*np.exp(th))

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
	# prev: 144 x 144     96 x  96
	# cur:  288 x 288    192 x 192
	

	img_a = cv2.imread(dict_desc['frame_a'])

	tl_x_a, tl_y_a, w_a, h_a = dict_desc['bbox_a']
	
	if tl_x_a < 0:
		w_a = w_a + tl_x_a
		tl_x_a = 0
	if tl_y_a < 0:
		h_a = h_a + tl_y_a
		tl_y_a = 0
	if tl_y_a + h_a > img_a.shape[0]:
		h_a = img_a.shape[0] - tl_y_a
	if tl_x_a+w_a > img_a.shape[1]:
		w_a = img_a.shape[1] - tl_x_a

	feat_a = img_a[tl_y_a : tl_y_a+h_a, tl_x_a : tl_x_a+w_a, :]


	img_b = cv2.imread(dict_desc['frame_b'])
	feat_b, (x_orig, y_orig) = extract_mult(img_b, dict_desc['bbox_a'], mult=2)
	

	feat_a, _ = resize_with_ar(feat_a, 96)
	feat_a = pad_zeros(feat_a, 96)
	feat_a = feat_a[:, :, ::-1] # RGB

	feat_b, ratio_b = resize_with_ar(feat_b, 192)
	feat_b = pad_zeros(feat_b, 192)
	feat_b = feat_b[:, :, ::-1]	# RGB


	if data_aug:
		# Boolean vars
		aug_hue_a, aug_sat_a, aug_hue_b, aug_sat_b = \
			np.random.rand(4) > 0.5

		if aug_sat_a or aug_hue_a:
			feat_a = hsv_aug(feat_a, aug_hue_a, aug_sat_a, min_coeff=0.3)

		if aug_sat_b or aug_hue_b:
			feat_b = hsv_aug(feat_b, aug_hue_b, aug_sat_b, min_coeff=0.3)


	# Create
	label = np.zeros((9,5), dtype=np.float32)
	if ('bbox_b' in dict_desc):
		x_b, y_b, w_b, h_b = dict_desc['bbox_b']

		x = round((x_b - x_orig)*ratio_b)
		y = round((y_b - y_orig)*ratio_b)
		w, h = round(w_b*ratio_b), round(h_b*ratio_b)
		#print('new bbox', x, y, w, h)

		idx, vals = anchor_parse((x, y, w, h))

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
