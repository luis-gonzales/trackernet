import json
import pickle
import argparse
import tensorflow as tf

from generator import generator
from tracknet_model import build_model, load_weights
from tracknet_loss import tracknet_loss



# DETERMINE WHERE TO PLACE RANDOM SEED!!!

# Parse config files (head and tail:q)
parser = argparse.ArgumentParser()
parser.add_argument('cfg_front', help='config file for front of CNN', type=str)
parser.add_argument('cfg_tail', help='config file for tail of CNN', type=str)
args = parser.parse_args()

cfg_front = args.cfg_front	# e.g., data/sample.jpg
cfg_tail = args.cfg_tail
print('cfg_front =', cfg_front)
print('cfg_tail =', cfg_tail)


# Build model and parse hyperparams
print('Building model...')
model, cfg_blocks = build_model(cfg_front, cfg_tail)
model.summary()


load_weights('yolov3.weights', model, cfg_blocks[1:])

net_info = cfg_blocks[0]

batch_sz    = int(net_info['batch_size'])
num_epochs	= int(net_info['num_epochs'])
learning_rate     = float(net_info['learning_rate'])
adam_weight_decay = float(net_info['adam_weight_decay'])



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
#print('abs_path =', type(abs_path), abs_path)
print(len(train_set), len(val_set))

#val_set = train_set[:32]
#train_set = train_set[:32] + train_set[:32]


print('Number of training examples:\t', len(train_set))
print('Number of validation examples:\t', len(val_set))
print('Batch size =', batch_sz)


# gen
train_gen = generator(train_set, abs_path, batch_sz=batch_sz)
val_gen   = generator(val_set, abs_path, batch_sz=batch_sz)


# Training
adam = tf.keras.optimizers.Adam(lr=0.02)
#adam = tf.contrib.opt.AdamWOptimizer(weight_decay=adam_weight_decay, learning_rate=learning_rate)
model.compile(loss=tracknet_loss, optimizer=adam)
print('Training...')
#print('steps =', len(train_set)//batch_sz, len(val_set)//batch_sz)
checkpointer = tf.keras.callbacks.ModelCheckpoint('model/checkpoints/' + 'weights.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss')
history = model.fit_generator( \
					generator=train_gen,
	                steps_per_epoch=len(train_set)//batch_sz,
	      		    epochs=num_epochs,
	      		    callbacks=[checkpointer],
	      		    validation_data=val_gen,
	      		    validation_steps=len(val_set)//batch_sz)


def write_pickle(file_name, data):
	outfile = open(file_name, mode='wb')
	pickle.dump(data, outfile)
	outfile.close()


write_pickle('model/train_loss.pickle', history.history['loss'])
write_pickle('model/val_loss.pickle', history.history['val_loss'])

# final model should be taken from checkpoints (data/checkpoints)
