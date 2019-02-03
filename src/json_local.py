import json
import argparse
from os import path

'''
Command-line arg:
	-dp or --datapath: Absolute path to dataset (<abs>/TrackingNet-devkit/)
'''

# Retrieve absolute path to dataset
parser = argparse.ArgumentParser()
parser.add_argument('-dp', '--datapath',
					help="full path to TrackingNet dataset", type=str)
args = parser.parse_args()
abs_path = args.datapath

if abs_path[-1] != '/':
	abs_path = abs_path + '/'


# Helper that prepends absolute path to frame paths
def full_path(json_file, abs_path):	

	# Load
	file = open(json_file, mode='r')
	entries = json.load(file)
	file.close()

	data = []

	# Augment to existing path
	for entry in entries:

		frame_a_path = abs_path + entry['frame_a']
		frame_b_path = abs_path + entry['frame_b']

		if path.isfile(frame_a_path) and path.isfile(frame_b_path):
			entry['frame_a'] = frame_a_path
			entry['frame_b'] = frame_b_path
			data.append(entry)

	# Write
	out_file = open(json_file[:-5] + '_local.json', 'w')
	json.dump(data, out_file)
	out_file.close()

full_path('data/data_train.json', abs_path)
full_path('data/data_val.json', abs_path)
full_path('data/data_test.json', abs_path)
