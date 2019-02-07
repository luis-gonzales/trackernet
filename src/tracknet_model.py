import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


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


class ReshapeLayer(tf.keras.layers.Layer):

	def __init__(self):
		super(ReshapeLayer, self).__init__()

	def call(self, inputs):
		h, w = inputs.get_shape()[1:3]
		return tf.reshape(inputs, [-1, h*w, 5])


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


def conv_object(dict_desc, prev_depth, bn_momentum, bn_epsilon, relu_alpha, post_name,
				conv_trainable=True, bn_trainable=True):

	inp = Input(shape=(None,None,prev_depth), dtype=tf.float32, name='input'+post_name)
	x = inp

	x = conv(x, dict_desc, bn_momentum=bn_momentum, bn_epsilon=bn_epsilon,
			 relu_alpha=relu_alpha, post_name=post_name)

	return Model(inputs=inp, outputs=x, name='conv_obj'+post_name)








def build_model(cfg_file):

	cfg_blocks = parse_cfg(cfg_file)
	print('cfg_blocks =\n', cfg_blocks)
	net = cfg_blocks[0]


	# Hyperparams
	det_width     = int(net['det_width'])
	det_height    = int(net['det_height'])
	fov_mult      = int(net['fov_mult'])
	bn_momentum   = float(net['bn_momentum'])
	bn_epsilon    = float(net['bn_epsilon'])
	relu_alpha    = float(net['relu_alpha'])
	learning_rate = float(net['learning_rate'])
	adam_weight_decay = float(net['adam_weight_decay'])
	

	# Define inputs
	input_t_m1 = Input(shape=(det_height, det_width, 3),
					   dtype=tf.float32, name='input_t_m1')
	input_t    = Input(shape=(fov_mult*det_height, fov_mult*det_width, 3),
					   dtype=tf.float32, name='input_t')
	x_t_m1 = input_t_m1
	x_t    = input_t


	# Shared conv operation
	shared_conv_0_dict = {'batch_normalize': '1', 'filters': '16', 'size': '3',
						  'stride': '1', 'activation': 'leaky'}
	shared_conv_0 = conv_object(shared_conv_0_dict, prev_depth=3, bn_momentum=bn_momentum,
								bn_epsilon=bn_epsilon, relu_alpha=relu_alpha,
								post_name='_shared_conv_0',
								conv_trainable=True, bn_trainable=True)
	x_t_m1 = shared_conv_0(x_t_m1)
	x_t    = shared_conv_0(x_t)

	print('Shared conv:')
	print(x_t_m1)
	print(x_t)



	t_m1_act_maps = []
	for idx, layer in enumerate(cfg_blocks[1:]):
		print('-------\nLayer:', idx)
		print('layer[type] =', layer['type'])

		if layer['type'] == 'convolutional':
			x_t_m1 = conv(x_t_m1, layer, bn_momentum=bn_momentum,
						  bn_epsilon=bn_epsilon, relu_alpha=relu_alpha,
						  post_name='_conv_t_m1_' + str(idx))
			print(x_t_m1)

		elif layer['type'] == 'shortcut':
			shortcut_name = 'res_t_m1_' + str(idx)
			from_ = int(layer['from'])
			x_t_m1 = tf.keras.layers.add([x_t_m1, t_m1_act_maps[from_]], name=shortcut_name)
			print(x_t_m1)

		t_m1_act_maps.append(x_t_m1)

	'''
	print('\nConvs for frame t-1')
	# Conv operations for frame t-1
	t_m1_conv_dict = {'batch_norm': True, 'filters': 32, 'size': 3,
					  'stride': 2, 'activation': 'leaky'}
	x_t_m1 = conv(x_t_m1, t_m1_conv_dict, bn_momentum=bn_momentum,
				  bn_epsilon=bn_epsilon, relu_alpha=relu_alpha, post_name='_t_m1_0')
	print(x_t_m1)

	#t_m1_conv_dict = {'batch_norm': True, 'filters': 16, 'size': 1,
	#				  'stride': 1, 'activation': 'leaky'}
	#x1_t_m1 = conv(x0_t_m1, t_m1_conv_dict, bn_momentum=bn_momentum,
	#			  bn_epsilon=bn_epsilon, relu_alpha=relu_alpha, post_name='_t_m1_1')
	#print(x1_t_m1)

	#t_m1_conv_dict = {'batch_norm': True, 'filters': 32, 'size': 3,
	#				  'stride': 1, 'activation': 'leaky'}
	#x2_t_m1 = conv(x1_t_m1, t_m1_conv_dict, bn_momentum=bn_momentum,
	#			  bn_epsilon=bn_epsilon, relu_alpha=relu_alpha, post_name='_t_m1_2')
	#print(x2_t_m1)

	#x_t_m1 = tf.keras.layers.add([x0_t_m1, x2_t_m1])
	#print(x_t_m1)



	t_m1_conv_dict = {'batch_norm': True, 'filters': 64, 'size': 3,
					  'stride': 2, 'activation': 'leaky'}
	x_t_m1 = conv(x_t_m1, t_m1_conv_dict, bn_momentum=bn_momentum,
				  bn_epsilon=bn_epsilon, relu_alpha=relu_alpha, post_name='_t_m1_1')
	print(x_t_m1)

	#t_m1_conv_dict = {'batch_norm': True, 'filters': 32, 'size': 1,
	#				  'stride': 1, 'activation': 'leaky'}
	#x4_t_m1 = conv(x3_t_m1, t_m1_conv_dict, bn_momentum=bn_momentum,
	#			   bn_epsilon=bn_epsilon, relu_alpha=relu_alpha, post_name='_t_m1_4')
	#print(x4_t_m1)

	#t_m1_conv_dict = {'batch_norm': True, 'filters': 64, 'size': 3,
	#				  'stride': 1, 'activation': 'leaky'}
	#x5_t_m1 = conv(x4_t_m1, t_m1_conv_dict, bn_momentum=bn_momentum,
	#			  bn_epsilon=bn_epsilon, relu_alpha=relu_alpha, post_name='_t_m1_5')
	#print(x5_t_m1)

	#x_t_m1 = tf.keras.layers.add([x3_t_m1, x5_t_m1])
	#print(x_t_m1)


	t_m1_conv_dict = {'batch_norm': True, 'filters': 128, 'size': 3,
					  'stride': 2, 'activation': 'leaky'}
	x_t_m1 = conv(x_t_m1, t_m1_conv_dict, bn_momentum=bn_momentum,
				  bn_epsilon=bn_epsilon, relu_alpha=relu_alpha, post_name='_t_m1_2')
	print(x_t_m1)

	t_m1_conv_dict = {'batch_norm': True, 'filters': 256, 'size': 3,
					  'stride': 2, 'activation': 'leaky'}
	x_t_m1 = conv(x_t_m1, t_m1_conv_dict, bn_momentum=bn_momentum,
				  bn_epsilon=bn_epsilon, relu_alpha=relu_alpha, post_name='_t_m1_3')
	print(x_t_m1)

	t_m1_conv_dict = {'batch_norm': True, 'filters': 512, 'size': 3,
					  'stride': 2, 'activation': 'leaky'}
	x_t_m1 = conv(x_t_m1, t_m1_conv_dict, bn_momentum=bn_momentum,
				  bn_epsilon=bn_epsilon, relu_alpha=relu_alpha, post_name='_t_m1_4')
	print(x_t_m1)
	'''



	print('\nConvs for frame t')
	# Conv operations for frame t
	'''
	t_conv_dict = {'batch_norm': True, 'filters': 32, 'size': 3,
				   'stride': 2, 'activation': 'leaky'}
	x_t = conv(x_t, t_conv_dict, bn_momentum=bn_momentum,
			   bn_epsilon=bn_epsilon, relu_alpha=relu_alpha, post_name='_t_0')
	print(x_t)

	t_conv_dict = {'batch_norm': True, 'filters': 64, 'size': 3,
				   'stride': 2, 'activation': 'leaky'}
	x_t = conv(x_t, t_conv_dict, bn_momentum=bn_momentum,
			   bn_epsilon=bn_epsilon, relu_alpha=relu_alpha, post_name='_t_1')
	print(x_t)

	t_conv_dict = {'batch_norm': True, 'filters': 128, 'size': 3,
				   'stride': 2, 'activation': 'leaky'}
	x_t = conv(x_t, t_conv_dict, bn_momentum=bn_momentum,
			   bn_epsilon=bn_epsilon, relu_alpha=relu_alpha, post_name='_t_2')
	print(x_t)

	t_conv_dict = {'batch_norm': True, 'filters': 256, 'size': 3,
				   'stride': 2, 'activation': 'leaky'}
	x_t = conv(x_t, t_conv_dict, bn_momentum=bn_momentum,
			   bn_epsilon=bn_epsilon, relu_alpha=relu_alpha, post_name='_t_3')
	print(x_t)

	t_conv_dict = {'batch_norm': True, 'filters': 512, 'size': 3,
				   'stride': 2, 'activation': 'leaky'}
	x_t = conv(x_t, t_conv_dict, bn_momentum=bn_momentum,
			   bn_epsilon=bn_epsilon, relu_alpha=relu_alpha, post_name='_t_4')
	print(x_t)
	'''


	t_act_maps = []
	for idx, layer in enumerate(cfg_blocks[1:]):
		print('-------\nLayer:', idx)
		print('layer[type] =', layer['type'])

		if layer['type'] == 'convolutional':
			x_t = conv(x_t, layer, bn_momentum=bn_momentum,
						  bn_epsilon=bn_epsilon, relu_alpha=relu_alpha,
						  post_name='_conv_t_' + str(idx))
			print(x_t)

		elif layer['type'] == 'shortcut':
			shortcut_name = 'res_t_' + str(idx)
			from_ = int(layer['from'])
			x_t = tf.keras.layers.add([x_t, t_act_maps[from_]], name=shortcut_name)
			print(x_t)

		t_act_maps.append(x_t)
	print('---')

	t_conv_dict = {'batch_normalize': '1', 'filters': '1024', 'size': '3',
				   'stride': '2', 'activation': 'leaky'}
	x_t = conv(x_t, t_conv_dict, bn_momentum=bn_momentum,
			   bn_epsilon=bn_epsilon, relu_alpha=relu_alpha, post_name='_t_5')
	print(x_t)


	print('\nMerge')
	# Merge and perform final convolutions
	x = tf.keras.layers.concatenate([x_t_m1, x_t])	# [?, 3, 3, 1536]
	print(x)


	conv_dict = {'batch_normalize': '1', 'filters': '128', 'size': '1',
				   'stride': '1', 'activation': 'leaky'}
	x = conv(x, conv_dict, bn_momentum=bn_momentum,
			 bn_epsilon=bn_epsilon, relu_alpha=relu_alpha, post_name='_post_merge_0')
	print(x)


	conv_dict = {'batch_normalize': '0', 'filters': '5', 'size': '1',
				   'stride': '1', 'activation': 'linear'}
	x = conv(x, conv_dict, bn_momentum=bn_momentum,
			 bn_epsilon=bn_epsilon, relu_alpha=relu_alpha, post_name='_post_merge_1')
	print(x)


	x = ReshapeLayer()(x)
	print(x)

	return Model(inputs=[input_t_m1, input_t], outputs=x), net


