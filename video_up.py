from utils.prepare_images import ImageSplitter
from Models import get_model
import torch.nn as nn
from test import Uscale_splitted_frames
import torch
import sys
import os

# If cuda device is available use it
if torch.cuda.is_available():
	DEVICE = torch.device("cuda")
else:
	DEVICE = torch.device("cpu")
	print('Upscaling will be performed on CPU, this may take longer than expected')


if __name__ == "__main__":
	#Parameters
	args_cmd = sys.argv[1:]
	if len(args_cmd)==5:
		pass
	else:
		raise InputError('Inputs provided must be 5, not ' + str(len(args_cmd)))
 	#args 1 : Path where the initial video clips are located
	path_to_clips = args_cmd[0] #"video"

	# args 2 : Path of to be used a cache, be sure to have a fast read and write as well as some space
	# using hard drives will slow the process.
	cache_path = args_cmd[1] # 'cache' # Folder must be empty !

	# args 3 model and args 4 weights
	model_type = args_cmd[2]#'UpConv_7'
	path_weights = args_cmd[3] #"model_check_points/Upconv_7/anime/noise3_scale2.0x_model.json"
	model = get_model(model_type ,path_weights)

	#arg 5, Batch size telling how many images you'll be processing simutanuously increase this as much as your VRAM allows it
	batch_size=int(args_cmd[4])
	
	# Path where the upscaled videos will be written
	output_path = path_to_clips + "/results"

	# Path to pretrained data model to be imported

	Uscale_splitted_frames(model, cache_path+"/frames", cache_path+'/results/',batch_size, DEVICE)
