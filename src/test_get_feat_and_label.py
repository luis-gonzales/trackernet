import cv2
import numpy as np

from generator import get_feat_and_label


my_dict = {'frame_a': '/Users/luisgonzales/Insight/Project/TrackingNet-devkit/TrackingNet/TRAIN_0/frames/a3In51YCqMg_0/0.jpg',
		   'frame_b': '/Users/luisgonzales/Insight/Project/TrackingNet-devkit/TrackingNet/TRAIN_0/frames/a3In51YCqMg_0/15.jpg',
		   'bbox_a': [233,173,93,61],
		   'bbox_b': [193,154,94,59]}

imgs, label = get_feat_and_label(my_dict)

print( type(imgs), type(label) )
print( type(imgs[0]), imgs[0].shape )


cv2.imwrite('img_a.jpg', (imgs[0][:, :, ::-1]*255).astype(np.uint8) )
cv2.imwrite('img_b.jpg', (imgs[1][:, :, ::-1]*255).astype(np.uint8) )
