from utils.prepare_images import ImageSplitter
from Models import UpConv_7,Discriminant,CARN_V2
from load_and_write import loader,DatasetNamed,DataWritter,register_fn,TrainingLoaderLRHR,MedianFilter,PreprocessLR,LRHRDataset

import torch
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
from torchvision.utils import save_image
from utils.losses import laplacian_loss
import torch.optim as optim
import numpy as np
from torchvision import get_image_backend

def plot_example(path,model,name,k):
	with torch.no_grad():
		# I recoded the image splitter so as to directly work with batch tensors
		img = loader(path).cuda().unsqueeze(0)
		img_splitter = ImageSplitter(seg_size=64, scale_factor=2, boarder_pad_size=3)
		# Cut the images in batched patches
		img_patches = img_splitter.split_img_tensor(img, img_pad=0)
		#	 Why we process in a loop: patches don't have the same width or height but we process the same patch on the batch size
		out = [model(i) for i in img_patches]
		# Merging the patches in a batched manner
		img_upscale = img_splitter.merge_img_tensor(out)
		save_image(img_upscale,name+"{:05d}.png".format(k))

def regular_training(model,dataloader,device,optimizer,batch_size,epochs):
	log = []
	k=6


	for epoch in range(epochs):

		for data_lr,data_hr in iter(dataloader):
			data_lr=data_lr.to(device)
			data_hr=data_hr.to(device)

			

			img_upscale = model(data_lr+ 0.01*torch.randn(*data_lr.shape,device=device))
			# img_upscale = img_upscale[...,6:-6,6:-6]
			# Merging the patches in a batched manner
			error = torch.sum(torch.abs(data_hr-img_upscale))/batch_size +10*torch.sum(laplacian_loss(img_upscale,data_hr,insist=1.1))/batch_size
			error.backward()
			optimizer.step()
			optimizer.zero_grad()
			log.append(error.item())




			
			if len(log)==3000:
				print(np.mean(log))
				log=[]
				
				torch.save(model,"model_{:05d}.pk".format(k))
				plot_example('lr.png',model,"trainings/image_",k)
				k+=1
				
					
					
def GAN_training(model,disc,dataloader,device,optimizer,optimizer_disc,batch_size,epochs):
	log = []
	log_disk = []
	k=2

	for epoch in range(epochs):

		for data_lr,data_hr in iter(dataloader):
			data_lr=data_lr.to(device)
			data_hr=data_hr.to(device)
			if data_lr.shape[-1]!=64 or data_lr.shape[-2]!=64:
				continue
			batch_size = data_lr.shape[0]

			if k>1:

				img_upscale = model(data_lr+ 0.001*torch.randn(*data_lr.shape,device=device))
				input_fake = img_upscale.unfold(2,64,64).unfold(3,64,64).transpose(1,2).transpose(2,3).reshape(batch_size*4,3,64,64)
				probits = disc(input_fake)
				error = -torch.sum(probits)/(batch_size)
				
				(error+0.0001*(torch.sum(torch.abs(data_hr-img_upscale))/batch_size+10*torch.sum(laplacian_loss(img_upscale,data_hr,insist=1.1))/batch_size)).backward()
				optimizer.step()
				optimizer.zero_grad()
				optimizer_disc.zero_grad()
				log.append(error.item())

			with torch.no_grad():
				img_upscale = model(data_lr+ 0.001*torch.randn(*data_lr.shape,device=device))
			input_fake =img_upscale.unfold(2,64,64).unfold(3,64,64).transpose(1,2).transpose(2,3).reshape(batch_size*4,3,64,64)


			input_real = data_hr.unfold(2,64,64).unfold(3,64,64).transpose(1,2).transpose(2,3).reshape(batch_size*4,3,64,64)

			probs_real = disc(data_hr)
			probs_fake = disc(input_fake)
			error_disc = torch.sum(probs_fake)-torch.sum(probs_real)
			error_disc.backward()
			optimizer_disc.step()
			optimizer.zero_grad()
			optimizer_disc.zero_grad()
			with torch.no_grad():
				for parameter in disc.parameters():
					parameter.data = torch.clamp(parameter.data,-0.1,0.1)
			log_disk.append(error_disc.item())

			if len(log_disk)==3000:
				print(np.mean(log_disk))
				log_disk=[]
				if k<=1:
					k+=1







			
			if len(log)==3000:
				print(np.mean(log))
				log=[]
				if k>1:
					torch.save(model,"trainings/model_gan_{:05d}.pk".format(k))
					plot_example('lr.png',model,"trainings/imggan",k)
				k+=1
				




if __name__ == "__main__":
	root="/media/will/227E8A467E8A1329/Users/willi/Documents/Batman/frames/test"




	path_weights = "model_check_points/Upconv_7/anime/noise3_scale2.0x_model.json"

	batch_size=20

	#Loading model and weights
	# model = UpConv_7()
	# model.load_pre_train_weights(json_file=path_weights)
	model = CARN_V2(color_channels=3, mid_channels=64,
                 scale=2, activation=nn.LeakyReLU(0.1),
                 SEBlock=True, conv=nn.Conv2d,
                 atrous=(1, 1, 1), repeat_blocks=3,
                 single_conv_size=3, single_conv_group=1)
	# model = torch.load("model_00010.pk")
	disc = Discriminant()

	# dataset = LRHRDataset(root,loader=loader,extensions=('jpg','png'),transform_lr=None,transform_hr=None)
	dataset = TrainingLoaderLRHR(root,loader=loader,extensions=('jpg','png'),batch_size=batch_size,transform_lr=None,transform_hr=None)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, sampler=None,
			   batch_sampler=None, num_workers=4, collate_fn=None,
			   pin_memory=False, drop_last=False, timeout=0,
			   worker_init_fn=None, prefetch_factor=4,
			   persistent_workers=False)

	
	if torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	#setting model to device
	model.to(device)
	disc.to(device)
	# learning_rate = 1e-6
	# weight_decay = 1e-6
	optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0, amsgrad=True)
	optimizer_disc = optim.Adam(disc.parameters(), lr=1e-6, weight_decay=1e-3, amsgrad=True)
	epochs=100000
	# regular_training(model,dataloader,device,optimizer,batch_size,epochs)
	GAN_training(model,disc,dataloader,device,optimizer,optimizer_disc,batch_size,epochs)


