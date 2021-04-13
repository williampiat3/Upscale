from utils.prepare_images import ImageSplitter
from Models import UpConv_7
from load_and_write import loader,DatasetNamed,DataWritter,register_fn,TrainingLoaderLRHR,MedianFilter,PreprocessLR

import torch
from torch.utils.data import DataLoader,Dataset
from torchvision.utils import save_image
from utils.losses import laplacian_loss
import torch.optim as optim
import numpy as np

if __name__ == "__main__":
	root="/media/will/227E8A467E8A1329/Users/willi/Documents/Batman/frames"
	# rename("/media/will/227E8A467E8A1329/Users/willi/Documents/Batman/frames/LR")
	# remap("/media/will/227E8A467E8A1329/Users/willi/Documents/Batman/frames/LR","/media/will/227E8A467E8A1329/Users/willi/Documents/Batman/frames/HR",window=24)

	# lr = lr.unsqueeze(0)
	# reducer = nn.Upsample(size=(540, 718), scale_factor=None, mode='nearest', align_corners=None)
	# upscaler = nn.Upsample(size=(1080, 1436), scale_factor=None, mode='nearest', align_corners=None)
	# input_image = upscaler(reducer(lr)).squeeze()
	path_weights = "model_check_points/Upconv_7/anime/noise3_scale2.0x_model.json"
	# Path where the upscaled frames will be written
	# Batch size telling how many images you'll be processing simutanuously increase this as much as your VRAM allows it
	batch_size=40

	#Loading model and weights
	model = UpConv_7()
	model.load_pre_train_weights(json_file=path_weights)

	dataset = TrainingLoaderLRHR(root,loader=loader,extensions=('jpg','png'),batch_size=batch_size,transform_lr=None,transform_hr=None)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, sampler=None,
			   batch_sampler=None, num_workers=4, collate_fn=None,
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
	optimizer = optim.SGD(model.parameters(),lr=0.0000000001,momentum=0.9)
	median_filter = MedianFilter(2)
	log = []

	for data_lr,data_hr in iter(dataloader):
		
		data_lr=data_lr.to(device)
		data_hr = median_filter(data_hr.to(device))
		# Why we process in a loop: patches don't have the same width or height but we process the same patch on the batch size
		img_upscale = model(data_lr)
		# Merging the patches in a batched manner
		error = torch.sum((img_upscale-data_hr)**2)/batch_size + 0.5*torch.sum(laplacian_loss(img_upscale,data_hr,insist=1.1))/batch_size
		error.backward()
		optimizer.step()
		optimizer.zero_grad()
		log.append(error.item())
		if len(log)==30:
			print(np.mean(log))
			log=[]

		# save_image(img_upscale[[0]],'upscaled.png')
		# save_image(data_hr[[0]],'original.png')
		# exit()


