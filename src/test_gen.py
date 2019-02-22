import cv2
import json
import numpy as np

from generator import get_feat_and_label

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def inv_sigmoid(x):
	#print('inv_sigmoid x =', x)
	lim = 5
	if (x == 0): return -lim
	elif (x > 0.9998) and (x < 1.0002): return lim
	ans = np.log( x / (1-x) )

	if ans > lim: return lim
	elif ans < -lim: return -lim
	else: return ans

print( inv_sigmoid(0.0078125) )
print( sigmoid(-5), sigmoid(5) )


def json_to_list(json_file, batch_sz):
	file = open(json_file, mode='r')
	data = json.load(file)

	# Make divisible by batch_sz
	excess = len(data[1]) % batch_sz
	if excess != 0:
		data[1] = data[1][:-excess]

	file.close()
	return data


abs_path, train_set = json_to_list('data/data_train_local.json', batch_sz=16)
print( 'original length =', len(train_set) )

cur_dict = train_set[61]
new_dict = {'frame_a': abs_path + cur_dict['frame_a'],
			'frame_b': abs_path + cur_dict['frame_b'],
			'bbox_a': cur_dict['bbox_a'],
			'bbox_b': cur_dict['bbox_b']}
#print('new_dict =\n', new_dict)

imgs, label = get_feat_and_label(new_dict)
#my_gen = generator(train_set, abs_path, batch_sz=1)
#imgs, label = next(my_gen)

#print( type(imgs), type(label) )
img_a = imgs[0][:,:,:]
img_a = img_a[:, :, ::-1]*255
#print(img_a.shape)
cv2.imwrite('test_a.jpg', img_a.astype(np.uint8))

img_b = imgs[1][:,:,:]
img_b = img_b[:, :, ::-1]*255
#print(img_b.shape)
cv2.imwrite('test_b.jpg', img_b.astype(np.uint8))

lab = label
print(lab.shape)
print(lab)

t = np.nonzero(lab[:,0])
#print( t )
#print( type(t), len(t) )
#print( type(t[0]) )
idx = int(t[0])
print('idx =', idx)

t = lab[idx, 1:]
print(64*sigmoid(t[0]))
print(64*sigmoid(t[1]))
print(96*np.exp(t[2]))
print(96*np.exp(t[3]))
