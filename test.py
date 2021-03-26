from utils.prepare_images import ImageSplitter
from Models import UpConv_7
from load_and_write import loader,DatasetNamed,DataWritter,register_fn

import torch
from torch.utils.data import DataLoader,Dataset


if __name__ == "__main__":
	#Parameters
	path_initial_frames = "/media/will/227E8A467E8A1329/Users/willi/Documents/MX/frames"
	path_weights = "/home/will/Documents/github/Upscale/model_check_points/Upconv_7/anime/noise3_scale2.0x_model.json"
	output_path = "/media/will/227E8A467E8A1329/Users/willi/Documents/MX/results/"
	# Batch size telling how many images you'll be processing simutanuously increase this as much as your VRAM allows it
	batch_size=7

	#Loading model and weights
	model = UpConv_7()
	model.load_pre_train_weights(json_file=path_weights)

	#Preparing dataset and dataloader, the images
	dataset = DatasetNamed(path_initial_frames,loader=loader,extensions=('jpg','png'))
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=None,
			   batch_sampler=None, num_workers=2, collate_fn=None,
			   pin_memory=False, drop_last=False, timeout=0,
			   worker_init_fn=None, prefetch_factor=2,
			   persistent_workers=False)

	if torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	#setting model to device
	model.to(device)
	model = model.float() 

	for data,names in iter(dataloader):
		
		data=data.to(device)
		

		# overlapping split
		# if input image is too large, then split it into overlapped patches 
		# details can be found at [here](https://github.com/nagadomi/waifu2x/issues/238)
		# The image splitter splits the image in non even patches meaning patches can't be batched in the same image
		# However they can be parallized between images and this is our improvement here
		with torch.no_grad():
			# I recoded the image splitter so as to directly work with batch tensors
		   img_splitter = ImageSplitter(seg_size=64, scale_factor=2, boarder_pad_size=3)

		   img_patches = img_splitter.split_img_tensor(data, img_pad=0)
		# Why we process in a loop: patches don't have the same width or height but we process the same patch on the batch size
		with torch.no_grad():
			out = [model(i) for i in img_patches]
		# Merging the patches in a batched manner
		img_upscale = img_splitter.merge_img_tensor(out)

		# Writting the images using a dummy custom dataloader for allowing multiproccessing
		datawriter = DataLoader(DataWritter(img_upscale,names), batch_size=1, shuffle=False, sampler=None,
				   batch_sampler=None, num_workers=min(8,batch_size), collate_fn=lambda x: register_fn(x,output_path),
				   pin_memory=False, drop_last=False, timeout=0,
				   worker_init_fn=None, prefetch_factor=2,
				   persistent_workers=False)
		# Writting images
		for _ in iter(datawriter):
			pass
		print(names)
