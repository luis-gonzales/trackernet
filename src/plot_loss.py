import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt


def get_data(file):
	infile = open(file, mode='rb')
	data = pickle.load(infile)
	infile.close()
	return data


# Parse command-line inputs
parser = argparse.ArgumentParser()
parser.add_argument('train_loss', help='pickle file to training loss', type=str)
parser.add_argument('val_loss', help='pickle file to validation loss', type=str)
args = parser.parse_args()

train_file = args.train_loss
val_file   = args.val_loss


# Load data
train_loss = get_data(train_file)
val_loss   = get_data(val_file)
train_loss = train_loss[:16]
val_loss   = val_loss[:16]


# Plot data
epochs = np.arange(1, len(train_loss)+1)

plt.plot(epochs, train_loss)
plt.plot(epochs, val_loss)


# Customize
axis_fsize = 18
legend_fsize = 18
tick_fsize = 18

plt.xlabel('Epochs', fontsize=axis_fsize)
plt.ylabel('Loss', fontsize=axis_fsize)

plt.gca().set_xlim([1, len(epochs)])
plt.gca().set_ylim([0, 1.5])

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_color('white')
plt.gca().spines['left'].set_color('white')

plt.gca().xaxis.label.set_color('white')
plt.gca().yaxis.label.set_color('white')

plt.gca().tick_params(axis='x', colors='white')
plt.gca().tick_params(axis='y', colors='white')
plt.tick_params(axis='both', which='major', labelsize=tick_fsize)

plt.grid(axis='y', linestyle='--')

plt.xticks([2, 4, 6, 8, 10, 12, 14, 16])

plt.gca().legend(['Train', 'Validation'], fontsize=legend_fsize)

plt.savefig('model/loss_curves.png', bbox_inches='tight', dpi=300, transparent=True)
