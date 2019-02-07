import sys
import json
import argparse
from os import path

'''
Command-line arg:
	abs_path: Absolute path to dataset (<abs>/TrackingNet-devkit/)
'''

# Retrieve absolute path to dataset
abs_path = sys.argv[1]

if abs_path[-1] != '/':
	abs_path = abs_path + '/'


# Helper that prepends absolute path to frame paths
def full_path(json_file, abs_path):	

	# Load
	file = open(json_file, mode='r')
	entries = json.load(file)
	file.close()

	out_entries = []

	for entry in entries:
		#print(entry)

		path_a = abs_path + entry['frame_a']
		path_b = abs_path + entry['frame_b']

		#print('paths =', path_a, path_b)

		if path.isfile(path_a) and path.isfile(path_b):
			#print('file exists')
			out_entries.append(entry)
		#else:
			#print('file does NOT exist')


	out = [abs_path, out_entries]

	# Write
	out_file = open(json_file[:-5] + '_local.json', 'w')
	json.dump(out, out_file)
	out_file.close()

full_path('data/data_train.json', abs_path)
full_path('data/data_val.json', abs_path)
full_path('data/data_test.json', abs_path)
