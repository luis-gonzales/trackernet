import sys
import json
import argparse

from os import path

'''
Command-line arg:
	abs_path: Absolute path to dataset (<abs>/TrackingNet-devkit/)
'''


# Parse command-line input
parser = argparse.ArgumentParser()
parser.add_argument('abs_path', help='absolute path to TrackingNet', type=str)
args = parser.parse_args()

abs_path = args.abs_path
if abs_path[-1] != '/':
	abs_path = abs_path + '/'		# Force '/' for consistency


# Helper that prepends absolute path to frame paths
def full_path(json_file, abs_path):	

	# Load existing labels
	file = open(json_file, mode='r')
	entries = json.load(file)
	file.close()

	# Append if data exists locally
	out_entries = []
	for entry in entries:
		path_a = abs_path + entry['frame_a']
		path_b = abs_path + entry['frame_b']

		if path.isfile(path_a) and path.isfile(path_b):
			out_entries.append(entry)

	out = [abs_path, out_entries]

	# Write out to local JSON
	new_file = json_file[:-5] + '_local.json'
	out_file = open(new_file, 'w')
	json.dump(out, out_file)
	out_file.close()
	print('# of samples in', new_file, '\t', len(out_entries))

full_path('data/data_train.json', abs_path)
full_path('data/data_val.json', abs_path)
full_path('data/data_test.json', abs_path)
