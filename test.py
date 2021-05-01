from utils.prepare_images import ImageSplitter
from Models import get_model
import torch.nn as nn
from load_and_write import loader,DatasetNamed,DataWritter,register_fn


import torch
from torch.utils.data import DataLoader,Dataset
# If cuda device is available use it
if torch.cuda.is_available():
	DEVICE = torch.device("cuda")
else:
	DEVICE = torch.device("cpu")

def Uscale_splitted_frames(model, path_initial_frames, output_path,batch_size, device):
	"""
	function to upscale images at a given with a given model  on a specified device
	image are splitted to use Waifu2x and run on a modest gpu

	:param: model: torch model compatible with upscaling
	:param: path_initial_frames: path to the images to be upscaled
	:param: output_path: path for upscale frames to be saved
	:param: batch_size: number of image to be treated at the same time
	:param: device: device on which the computation is performed
	"""

	#Preparing dataset and dataloader, the images
	dataset = DatasetNamed(path_initial_frames,loader=loader,extensions=('jpg','png'))
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=None,
			   batch_sampler=None, num_workers=2, collate_fn=None,
			   pin_memory=False, drop_last=False, timeout=0,
			   worker_init_fn=None, prefetch_factor=2,
			   persistent_workers=False)

	#setting model to device
	model.to(device)
	model = model.float() 
	

	for data,names in iter(dataloader):
		
		data=data.to(device)
		# overlapping split
		# if input image is too large, then split it into overlapped patches 
		# details can be found at [here](https://github.com/nagadomi/waifu2x/issues/238)
		# The image splitter splits the image in non even patches meaning patches can't be batched and processed altogether
		# However they can be parallized between images and this is our improvement here
		with torch.no_grad():
			# I recoded the image splitter so as to directly work with batch tensors
		   img_splitter = ImageSplitter(seg_size=64, scale_factor=2, boarder_pad_size=7)
		   # Cut the images in batched patches
		   img_patches = img_splitter.split_img_tensor(data, img_pad=0)
		# Why we process in a loop: patches don't have the same width or height but we process the same patch on the batch size
		with torch.no_grad():
			out = [model(i) for i in img_patches]
		# Merging the patches in a batched manner
		img_upscale = img_splitter.merge_img_tensor(out)

		# Writting the images using a dummy custom dataloader for allowing multiproccessing could be upgraded with asynchronous writting
		datawriter = DataLoader(DataWritter(img_upscale.cpu(),names), batch_size=1, shuffle=False, sampler=None,
				   batch_sampler=None, num_workers=min(8,batch_size), collate_fn=lambda x: register_fn(x,output_path),
				   pin_memory=False, drop_last=False, timeout=0,
				   worker_init_fn=None, prefetch_factor=2,
				   persistent_workers=False)
		# Writting images
		for _ in iter(datawriter):
			pass
		#Log the images that were dealt with
		print(names)

if __name__ == "__main__":
	#Parameters

	#Path where the initial frames are located
	path_initial_frames = "frames"

	# Path of the model weights
	path_weights ="trainings/model_mh_00014.pk"
	# path_weights = "model_check_points/Upconv_7/anime/noise3_scale2.0x_model.json"
	
	# Path where the upscaled frames will be written
	output_path = "results/"

	# Batch size telling how many images you'll be processing simutanuously increase this as much as your VRAM allows it
	batch_size=1
	# Path to pretrained data model to be imported

	#Loading model and weights
	model = get_model('CARN_V2',path_weights)
	# model = get_model('UpConv_7',path_weights)

	Uscale_splitted_frames(model, path_initial_frames, output_path,batch_size, DEVICE)
