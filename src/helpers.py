import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))
	
def preds_to_bbox(t_bbox, obj_row, grid_sz, anchor_sz):
	# Convert CNN output to (top-left x, top-left y, width, height)

	tx, ty, tw, th = t_bbox

	cx, cy = (obj_row % 3), (obj_row // 3)	# offsets in [0, 1, 2]

	bx = grid_sz * (sigmoid(tx) + cx)
	by = grid_sz * (sigmoid(ty) + cy)
	bw = anchor_sz * np.exp(tw)
	bh = anchor_sz * np.exp(th)
	
	x = int(round(bx - bw/2))
	y = int(round(by - bh/2))
	w = int(round(bw))
	h = int(round(bh))

	return x, y, w, h
	