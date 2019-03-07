import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda


def parse_cfg(cfg_file):
	'''
	Store each layer as a dict
	'''

	# List w/ each element equal to a line from the cfg file
	file = open(cfg_file, 'r')
	lines = file.read().split('\n')                 # Store the lines in a list
	lines = [x for x in lines if len(x) > 0]        # Get rid of the empty lines 
	lines = [x for x in lines if x[0] != '#']       # Get rid of comments
	lines = [x.rstrip().lstrip() for x in lines]	# Get rid of fringe whitespaces

	block = {}
	blocks = []

	for line in lines:
		if line[0] == '[':							# Beginning of a block
			if len(block) != 0:         			# If block is not empty, implies it is storing a previous array; append
				blocks.append(block) 
				block = {}            
			block['type'] = line[1:-1].rstrip()		# E.g., block['type'] = 'net', 'convolutional', etc
		else:
			key, value = line.split('=')
			block[key.rstrip()] = value.lstrip()	# E.g., block['batch'] = '64'
	blocks.append(block)							# Append last block

	file.close()
	return blocks


def reshape(x):
	import tensorflow as tf
	h, w = x.get_shape()[1:3]
	return tf.reshape(x, [-1, h*w, 5])	# Force to 5 columns


def conv(x, dict_desc, bn_momentum, bn_epsilon, relu_alpha, post_name,
		 conv_trainable=True, bn_trainable=True):
	'''
	Unified convolutional layer with batch norm and activation layers

	Input args:
	x				Previous activation map (tensor) on which to perform convolution
	layer 			Dictionary containing layer parameters
	bn_momentum		Batch norm momentum value
	bn_epsilon		Batch norm epsilon value
	relu_alpha 		ReLU alpha value
	post_name		Name to append to op names
	conv_trainable	Boolean defining whether conv kernels are trainable
	bn_trainable	Boolean defining whether batch norm params are trainable
	'''
	
	batch_norm  = True if (dict_desc['batch_normalize'] == '1') else False
	num_filters = int(dict_desc['filters'])
	kernel_size = int(dict_desc['size'])
	stride      = int(dict_desc['stride'])
	activation  = dict_desc['activation']

	if kernel_size == 3:
		if stride == 1:
			x = tf.keras.layers.ZeroPadding2D( ((1,1),(1,1)) )(x)
		elif stride == 2:
			x = tf.keras.layers.ZeroPadding2D( ((1,0),(1,0)) )(x)
	
	conv_name = 'conv' + post_name
	x = tf.layers.conv2d(x, filters=num_filters, kernel_size=(kernel_size,kernel_size),
						 strides=(stride,stride), padding='valid', use_bias=(not batch_norm),
						 kernel_initializer=tf.initializers.he_normal(),
						 bias_initializer=tf.zeros_initializer(),
						 trainable=conv_trainable, name=conv_name)

	if batch_norm:
		bn_name = 'bn' + post_name
		x = tf.keras.layers.BatchNormalization(axis=-1, momentum=bn_momentum,
											   epsilon=bn_epsilon,
											   trainable=bn_trainable, name=bn_name)(x)

	if activation == 'leaky':
		act_name = 'relu' + post_name
		x = tf.keras.layers.LeakyReLU(alpha=relu_alpha, name=act_name)(x)

	return x


def divide(x):
	import tensorflow as tf
	return tf.divide(x, 255)	# [0,255] -> [0.0, 1.0]


def build_model(cfg_head, cfg_tail):
	print('Building model...')

	head_blocks = parse_cfg(cfg_head)
	

	# Parse hyperparams
	net_info = head_blocks[0]
	
	det_width     = int(net_info['det_width'])
	det_height    = int(net_info['det_height'])
	fov_mult      = int(net_info['fov_mult'])
	bn_momentum   = float(net_info['bn_momentum'])
	bn_epsilon    = float(net_info['bn_epsilon'])
	relu_alpha    = float(net_info['relu_alpha'])
	learning_rate = float(net_info['learning_rate'])
	adam_weight_decay = float(net_info['adam_weight_decay'])

	prev_ops_stop = int(net_info['prev_ops_stop'])	# Layer from head at which to stop ops for `prev` frame


	# Define inputs and normalize [0,255] -> [0.0, 1.0]
	input_prev = Input(shape=(det_height, det_width, 3),
					   dtype=tf.float32, name='input_prev')
	input_cur  = Input(shape=(fov_mult*det_height, fov_mult*det_width, 3),
					   dtype=tf.float32, name='input_cur')

	x_prev = Lambda(divide)(input_prev)
	x_cur  = Lambda(divide)(input_cur)


	# Build "head" of CNN
	prev_act_maps = []
	cur_act_maps  = []
	for idx, layer in enumerate(head_blocks[1:]):
		#print('-------\nLayer:', idx)
		#print('layer[type] =', layer['type'])

		if layer['type'] == 'convolutional':
			x_cur = conv(x_cur, layer, bn_momentum=bn_momentum,
					     bn_epsilon=bn_epsilon, relu_alpha=relu_alpha,
					     post_name='_cur_' + str(idx), conv_trainable=False, bn_trainable=False)
			
			if idx < prev_ops_stop:
				x_prev = conv(x_prev, layer, bn_momentum=bn_momentum,
						  	  bn_epsilon=bn_epsilon, relu_alpha=relu_alpha,
							  post_name='_prev_' + str(idx), conv_trainable=False, bn_trainable=False)

		elif layer['type'] == 'shortcut':
			shortcut_name = 'res_cur_' + str(idx)
			from_ = int(layer['from'])
			x_cur = tf.keras.layers.add([x_cur, cur_act_maps[from_]], name=shortcut_name)

			if idx < prev_ops_stop:
				shortcut_name = 'res_prev_' + str(idx)
				x_prev = tf.keras.layers.add([x_prev, prev_act_maps[from_]], name=shortcut_name)

		prev_act_maps.append(x_prev)
		cur_act_maps.append(x_cur)


	# Concatenate before building "tail"
	x = tf.keras.layers.concatenate([x_prev, x_cur])	# [?, 3, 3, 1536]


	# Build "tail" of CNN
	x_act_maps = []
	tail_blocks = parse_cfg(cfg_tail)
	for idx, layer in enumerate(tail_blocks):
		#print('-------\nLayer:', idx)
		#print('layer[type] =', layer['type'])

		if layer['type'] == 'convolutional':
			x = conv(x, layer, bn_momentum=bn_momentum,
					 bn_epsilon=bn_epsilon, relu_alpha=relu_alpha,
					 post_name='_x_' + str(idx))

		elif layer['type'] == 'shortcut':
			shortcut_name = 'res_x_' + str(idx)
			from_ = int(layer['from'])
			x = tf.keras.layers.add([x, x_act_maps[from_]], name=shortcut_name)

		x_act_maps.append(x)


	# Reshape 3 x 3 x 5 into 9 x 5
	x = Lambda(reshape)(x)
	
	print('Done!')
	return Model(inputs=[input_prev, input_cur], outputs=x), head_blocks


def load_weights(weights_file, model, blocks):
	print('Setting weights...')
	file = open(weights_file, 'r')

	# Skip header info including major, minor, and subversion numbers, and training data seen
	header = np.fromfile(file, dtype=np.int32, count=5)

	for idx, layer in enumerate(blocks):
		#print('-------\nLayer:', idx)
		#print('layer[type] =', layer['type'])

		# Load weights (conv kernels and BN params) for convolutional layers
		if layer['type'] == 'convolutional':
			
			conv_name = 'conv_cur_' + str(idx)

			bias_weights = []
			size, _, channels, filters = model.get_layer(conv_name).get_weights()[0].shape

			if ('batch_normalize' in layer):
				betas        = np.fromfile(file, dtype=np.float32, count=filters)
				gammas       = np.fromfile(file, dtype=np.float32, count=filters)
				moving_means = np.fromfile(file, dtype=np.float32, count=filters)
				moving_vars  = np.fromfile(file, dtype=np.float32, count=filters)

				bn_name = 'bn_cur_' + str(idx)
				model.get_layer(bn_name).set_weights([gammas, betas, moving_means, moving_vars])

				if idx < 12:
					bn_name_2 = 'bn_prev_' + str(idx)
					model.get_layer(bn_name_2).set_weights([gammas, betas, moving_means, moving_vars])

			else:
				biases = np.fromfile(file, dtype=np.float32, count=filters)
				bias_weights.append(biases)

			conv_weights = np.fromfile(file, dtype=np.float32, count=size*size*channels*filters)

			conv_weights = np.reshape(conv_weights, newshape=(-1,filters), order='F')
			conv_weights = np.reshape(conv_weights, newshape=(-1, channels, filters), order='F')
			conv_weights = [np.reshape(conv_weights, newshape=(size,size,channels,filters), order='C')]

			model.get_layer(conv_name).set_weights(conv_weights + bias_weights)

			if idx < 12:
				conv_name_2 = 'conv_prev_' + str(idx)
				model.get_layer(conv_name_2).set_weights(conv_weights + bias_weights)

	file.close()
	print('Done!')
	return model
