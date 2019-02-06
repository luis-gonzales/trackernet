import json
import argparse
import tensorflow as tf

from generator import generator
from tracknet_model import build_model
from tracknet_loss import tracknet_loss


# DETERMINE WHERE TO PLACE RANDOM SEED!!!
# CFG IS NOT OPTIONAL!

# Parse for config file
parser = argparse.ArgumentParser()
parser.add_argument('-cfg', '--config',
					help="path to config file", type=str)
args = parser.parse_args()
cfg_file = args.config


# Build model and parse hyperparams
print('Building model...')
model, cfg_blocks = build_model(cfg_file)
model.summary()
print('cfg_blocks =\n', cfg_blocks)


batch_sz    = int(cfg_blocks['batch_size'])
num_epochs	= 3
learning_rate     = float(cfg_blocks['learning_rate'])
adam_weight_decay = float(cfg_blocks['adam_weight_decay'])


# Load train, val set and create Python generators

def json_to_list(json_file, batch_sz):
	file = open(json_file, mode='r')
	data = json.load(file)

	# Make divisible by batch_sz
	excess = len(data[1]) % batch_sz
	if excess != 0:
		data[1] = data[1][:-excess]

	file.close()
	return data

abs_path, train_set = json_to_list('data/data_train_local.json', batch_sz)
_, val_set   = json_to_list('data/data_val_local.json', batch_sz)
print('abs_path =', type(abs_path), abs_path)

print('Number of training examples:\t', len(train_set))
print('Number of validation examples:\t', len(val_set))


# gen
train_gen = generator(train_set, abs_path, batch_sz=batch_sz)
val_gen   = generator(val_set, abs_path, batch_sz=batch_sz)


# Training
#adam = tf.keras.optimizers.Adam(lr=0.02)
adam = tf.contrib.opt.AdamWOptimizer(weight_decay=adam_weight_decay, learning_rate=learning_rate)
model.compile(loss=tracknet_loss, optimizer=adam)
print('Training...')
model.fit_generator(generator=train_gen,
	                steps_per_epoch=len(train_set)//batch_sz,
	      		    epochs=num_epochs,
	      		    validation_data=val_gen,
	      		    validation_steps=len(val_set)//batch_sz)


