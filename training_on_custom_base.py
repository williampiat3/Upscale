from utils.prepare_images import ImageSplitter
from Models import UpConv_7,Discriminant,CARN_V2
from load_and_write import loader,random_compression_loader,TrainingLoaderLRHR,MedianFilter,LRHRDataset

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
from torchvision.utils import save_image
from utils.losses import laplacian_loss
import torch.optim as optim
import numpy as np

def plot_example(path,model,name,k):
	with torch.no_grad():
		img = loader(path).cuda().unsqueeze(0)
		img_splitter = ImageSplitter(seg_size=64, scale_factor=2, boarder_pad_size=3)
		# Cut the images in batched patches
		img_patches = img_splitter.split_img_tensor(img, img_pad=0)
		#	 Why we process in a loop: patches don't have the same width or height but we process the same patch on the batch size
		out = [model(i) for i in img_patches]
		# Merging the patches in a batched manner
		img_upscale = img_splitter.merge_img_tensor(out)
		save_image(img_upscale,name+"{:05d}.png".format(k))

def make_patches(image,size_patch,batch_size):
	return image.unfold(2,size_patch,size_patch).unfold(3,size_patch,size_patch).transpose(1,2).transpose(2,3).reshape(batch_size*4,3,size_patch,size_patch)


def regular_training(model,dataloader,device,optimizer,batch_size,epochs):
	"""
	Traditionnal training for superRes 
	"""
	log = []
	k=0


	for epoch in range(epochs):

		for data_lr,data_hr in iter(dataloader):
			data_lr=data_lr.to(device)
			data_hr=data_hr.to(device)

			

			img_upscale = model(data_lr)
			## Computing content loss with a laplacian loss to insist on borders
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
				
					
					
def GAN_training(model,disc,dataloader,device,optimizer,optimizer_disc,batch_size,epochs,pretrain_disc=0,batches_logs=4000):
	"""
	Training using a GAN structure, netter for the likelyhood of the upscaled images
	"""
	# Logs lists
	log = []
	log_disc = []
	# counter for logs
	k=0


	for epoch in range(epochs):

		for data_lr,data_hr in iter(dataloader):
			data_lr=data_lr.to(device)
			data_hr=data_hr.to(device)
			#if the data is not made of patches of 64x64 we pass to the next one
			if data_lr.shape[-1]!=64 or data_lr.shape[-2]!=64:
				continue


			#getting batch size
			batch_size = data_lr.shape[0]

			# Discriminant pass

			# Upscaling with no gradient flow
			with torch.no_grad():
				img_upscale = model(data_lr)

			# Making the 64x64 patches for the discriminant
			input_fake =make_patches(img_upscale,64,batch_size)
			input_real = make_patches(data_hr,64,batch_size)

			# Processing through discriminant
			probs_real = disc(input_real)
			probs_fake = disc(input_fake)
			# Computing error and gradient step discriminant (wasserstein GAN loss and gradient update)
			error_disc = torch.sum(probs_fake)-torch.sum(probs_real)
			error_disc.backward()
			optimizer_disc.step()
			optimizer.zero_grad()
			optimizer_disc.zero_grad()
			# Clipping weights of discriminant as we have wasserstein GAN
			with torch.no_grad():
				for parameter in disc.parameters():
					parameter.data = torch.clamp(parameter.data,-0.1,0.1)
			# Logs Disc
			log_disc.append(error_disc.item())

			#Logs for the discriminant
			if len(log_disc)==batches_logs:
				print("Loss Discriminant: ",np.mean(log_disc))
				log_disc=[]
				pretrain_disc-=1

			# If we are pretraining the discriminant we skip the upscaler's update
			if pretrain_disc>0:
				continue



			# Upscaler pass

			# Upscaling with graph tracing
			img_upscale = model(data_lr)
			# Making patches for the discriminant
			input_fake = make_patches(img_upscale,64,batch_size)
			# Discriminant pass
			probits = disc(input_fake)
			# Computing the log likelyhood of the samples being real according to the discriminant
			# The upscaler wants to fool the discriminant therefore it has to reduce -p_fake
			p_fake = torch.sum(probits)/(batch_size)

			# Computing content loss with a laplacian loss to insist on borders
			content_loss = (torch.sum(torch.abs(data_hr-img_upscale))/batch_size+10*torch.sum(laplacian_loss(img_upscale,data_hr,insist=1.1))/batch_size)

			# Backprop and update upscaler, the 0.00005 has been set minimal to avoid collapse mode leaving room for the GAN structure
			(-p_fake+0.000005*content_loss).backward()
			optimizer.step()
			optimizer.zero_grad()
			optimizer_disc.zero_grad()

			# Logs 
			log.append(error.item())


			# Periodical logs for upscaler
			if len(log)==batches_logs:
				print("Loss Upscaler: ",np.mean(log))
				log=[]
				torch.save(model,"trainings/model_gan_{:05d}.pk".format(k))
				plot_example('trainings/lr.png',model,"trainings/imggan",k)
				k+=1



				




if __name__ == "__main__":

	# In this folder add an "HR" folder with high res images and a "LR" folder with low res images, tuples are made between images with the same name
	root="/media/will/227E8A467E8A1329/Users/willi/Documents/Batman/frames"






	batch_size=20

	#Loading model and weights

	# model = UpConv_7()
	# path_weights = "model_check_points/Upconv_7/anime/noise3_scale2.0x_model.json"
	# model.load_pre_train_weights(json_file=path_weights)


	model = CARN_V2(color_channels=3, mid_channels=64,
				 scale=2, activation=nn.LeakyReLU(0.1),
				 SEBlock=True, conv=nn.Conv2d,
				 atrous=(1, 1, 1), repeat_blocks=3,
				 single_conv_size=3, single_conv_group=1)
	# model = torch.load("model_00010.pk")
	disc = Discriminant()

	# Loader that loads only small patches: can be run on a small device like jetson nano
	# The compression noise is directly injected through the "random_compression loader" we add some gaussian blur on the low res images as a preprocessing (the gaussian kernel has by default a variying sigma)
	dataset = TrainingLoaderLRHR(root,loader=random_compression_loader,extensions=('jpg','png'),batch_size=batch_size,transform_lr=torchvision.transforms.GaussianBlur(5),transform_hr=None)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, sampler=None,
			   batch_sampler=None, num_workers=4, collate_fn=None,
			   pin_memory=False, drop_last=False, timeout=0,
			   worker_init_fn=None, prefetch_factor=4,
			   persistent_workers=False)

	
	if torch.cuda.is_available():
		device = torch.device("cuda")
	else:
		device = torch.device("cpu")
		print("No CUDA capable device detected, this will run on CPU, expect this to be very slow")

	#setting model to device
	model.to(device)
	disc.to(device)

	optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0, amsgrad=True)

	# Weight decay because the parameters are clipped
	optimizer_disc = optim.Adam(disc.parameters(), lr=1e-7, weight_decay=1e-3, amsgrad=True)
	epochs=100000
	# regular_training(model,dataloader,device,optimizer,batch_size,epochs)
	GAN_training(model,disc,dataloader,device,optimizer,optimizer_disc,batch_size,epochs,pretrain_disc=0,batches_logs=4000)


