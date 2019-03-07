import tensorflow as tf

def trackernet_loss(y_true, y_pred):	

	# Separate objectness from bounding box params
	y_true_conf, y_true_bbox = tf.split(y_true, [1, 4], axis=2)
	y_pred_conf, y_pred_bbox = tf.split(y_pred, [1, 4], axis=2)


	# Calculate loss from objectness
	loss_confs = \
		tf.losses.sigmoid_cross_entropy(multi_class_labels=y_true_conf,
										logits=y_pred_conf)


	# Compute loss for bounding box params only where obj exists
	lambda_coords = 5						# amplification parameter

	obj_mask = tf.equal(y_true_conf, 1.0)
	obj_mask = tf.tile(obj_mask, [1,1,4])	# repeat for four bbox params

	y_true_bbox_obj = tf.boolean_mask(y_true_bbox, obj_mask)
	y_pred_bbox_obj = tf.boolean_mask(y_pred_bbox, obj_mask)

	loss_coords = \
		tf.losses.mean_squared_error(labels=y_true_bbox_obj,
									 predictions=y_pred_bbox_obj)


	return lambda_coords*loss_coords + loss_confs
	