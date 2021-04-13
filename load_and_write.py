from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader,Dataset
from torchvision.utils import save_image
from PIL import Image
from torchvision.transforms.functional import to_tensor
import torch
import os
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from os import listdir
from os.path import isfile, join
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
import torch.optim as optim
import numpy as np
import shutil
from utils.losses import laplacian_loss
import random


def loader(string_path):
	#Function to load the images for the DatasetFolder
	img = Image.open(string_path).convert("RGB")
	return to_tensor(img)

def make_dataset(
	directory,
	extensions,
	is_valid_file):
	"""Generates a list of samples of a form (path_to_sample, class).

	Args:
		directory (str): root dataset directory
		class_to_idx (Dict[str, int]): dictionary mapping class name to class index
		extensions (optional): A list of allowed extensions.
			Either extensions or is_valid_file should be passed. Defaults to None.
		is_valid_file (optional): A function that takes path of a file
			and checks if the file is a valid file
			(used to check of corrupt files) both extensions and
			is_valid_file should not be passed. Defaults to None.

	Raises:
		ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.

	Returns:
		List[Tuple[str, int]]: samples of a form (path_to_sample, class)
	"""
	instances = []
	directory = os.path.expanduser(directory)
	both_none = extensions is None and is_valid_file is None
	both_something = extensions is not None and is_valid_file is not None
	if both_none or both_something:
		raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
	if extensions is not None:
		def is_valid_file(x: str) -> bool:
			return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
	is_valid_file = cast(Callable[[str], bool], is_valid_file)
   
	target_dir_lr = os.path.join(directory, "LR")
	target_dir_hr = os.path.join(directory, "HR")
	onlyfiles_lr = [f for f in listdir(target_dir_lr) if isfile(join(target_dir_lr, f))]
	onlyfiles_hr = [f for f in listdir(target_dir_hr) if isfile(join(target_dir_hr, f))]
	for f in sorted(onlyfiles_lr):
		if f in onlyfiles_hr:
			item = join(target_dir_lr, f),join(target_dir_hr, f)

			instances.append(item)
	return instances

class DatasetNamed(DatasetFolder):
	"""
	The initial dataset folder creates labels according to the folder structure of the folder given in parameters
	however for our case we just use the reading code of DatasetFolder so as read the image files
	"""
	def __getitem__(self,index):
		path, target = self.samples[index]
		sample = self.loader(path)
		if self.transform is not None:
			sample = self.transform(sample)
		if self.target_transform is not None:
			target = self.target_transform(target)
		#returning the name along with the sampled image so as to keep track of the ranking of the image 
		return sample, path.split('/')[-1]

class LRHRDataset(DatasetFolder):
	def __init__(self, root,loader,extensions=None,transform_lr=None,transform_hr=None,is_valid_file=None):
		# super(LRHRDataset, self).__init__(root, transform=transform,loader=loader,
		# 									target_transform=None,extensions=extensions,is_valid_file=is_valid_file)
		self.root=root
		self.transform_lr=transform_lr
		self.transform_hr = transform_hr
		samples = self.make_dataset(self.root, extensions, is_valid_file)
		if len(samples) == 0:
			msg = "Found 0 files in subfolders of: {}\n".format(self.root)
			if extensions is not None:
				msg += "Supported extensions are: {}".format(",".join(extensions))
			raise RuntimeError(msg)

		self.loader = loader
		self.extensions = extensions

		self.samples = samples




	@staticmethod
	def make_dataset(directory,extensions, is_valid_file) :
		return make_dataset(directory, extensions=extensions, is_valid_file=is_valid_file)

	def __getitem__(self,index):
		path_lr, path_hr = self.samples[index]
		sample_lr = self.loader(path_lr)
		sample_hr = self.loader(path_hr)
		if self.transform_lr is not None:
			sample_lr = self.transform_lr(sample_lr)
		if self.transform_hr is not None:
			sample_hr = self.transform_hr(sample_hr)
		#returning the name along with the sampled image so as to keep track of the ranking of the image 
		return sample_lr, sample_hr

def remap(folder1,folder2,window=24):
	onlyfiles_lr = [f for f in listdir(folder1) if isfile(join(folder1, f))]
	onlyfiles_hr = [f for f in listdir(folder2) if isfile(join(folder2, f))]
	upscaler = nn.Upsample(size=(1080, 1436), scale_factor=None, mode='nearest', align_corners=None)
	for k,f in enumerate(sorted(onlyfiles_lr)):
		min_file=f
		min_value=np.inf
		path_init = os.path.join(folder1, f)
		image_init = loader(path_init)[...,161:-161]
		image_init=upscaler(image_init.unsqueeze(0)).squeeze()
		for i in range(window):
			f2 = onlyfiles_hr[max(0,i-window//2+k)]
			path_temp = os.path.join(folder2, f2)
			image2 = loader(path_temp)
			distance = torch.sum(torch.abs(image_init-image2))
			if distance<min_value:
				min_file=f2
				min_value=distance
		print(f,min_file)
def resize_base():
	path_init="/media/will/227E8A467E8A1329/Users/willi/Documents/Batman/frames/HLR"
	path_dest = "/media/will/227E8A467E8A1329/Users/willi/Documents/Batman/frames/LR"
	onlyfiles_lr = [f for f in listdir(path_init) if isfile(join(path_init, f))]
	reducer = nn.Upsample(size=(540, 718), scale_factor=None, mode='bilinear', align_corners=True)
	for f in onlyfiles_lr:
		img = loader(join(path_init, f)).cuda()[...,161:-161]
		img = reducer(img.unsqueeze(0)).squeeze()
		save_image(img,join(path_dest, f))


def rename(folder1):
	onlyfiles_lr = [f for f in listdir(folder1) if isfile(join(folder1, f))]
	for k,f in enumerate(sorted(onlyfiles_lr)):
		path_init = os.path.join(folder1, f)
		# print(path_init,os.path.join(folder1, "thumb{:09d}.png".format(k)))
		shutil.move(path_init,os.path.join(folder1, "thumb{:09d}.png".format(k)))


class TrainingLoaderLRHR(LRHRDataset):

	def __init__(self, root,loader,batch_size,extensions=None,transform_lr=None,transform_hr=None,is_valid_file=None,patch_size=64):
		super(TrainingLoaderLRHR,self).__init__(root,loader,extensions=extensions,transform_lr=transform_lr,transform_hr=transform_hr,is_valid_file=is_valid_file)
		self.patch_size=patch_size
		self.counter=batch_size
		self.batch_size=batch_size

	def __getitem__(self,index):
		path_lr, path_hr = self.samples[index]
		sample_lr = self.loader(path_lr)
		sample_hr = self.loader(path_hr)
		if self.transform_lr is not None:
			sample_lr = self.transform_lr(sample_lr)

		_,height,width = sample_lr.size()
		# indice_w= random.randint(0,width-(width - math.ceil(width//self.patch_size-1)*self.patch_size))
		# indice_h = random.randint(0,height-(height - math.ceil(height//self.patch_size-1)*self.patch_size))
		if self.counter ==self.batch_size:
			self.i = random.choice(range(self.patch_size, height, self.patch_size))
			self.j= random.choice(range(self.patch_size, width, self.patch_size))
			self.counter=1
		else: 
			self.counter+=1
		patch_lr = sample_lr[...,self.i:min(self.i + self.patch_size, height),self.j:min(self.j + self.patch_size , width)]
		patch_hr = sample_hr[...,2*self.i:2*min(self.i + self.patch_size, height),2*self.j:2*min(self.j + self.patch_size, width)]
		if self.transform_hr is not None:
			patch_hr = self.transform_hr(patch_hr)
		return patch_lr,patch_hr





class DataWritter(Dataset):
	"""
	Class made in order to write examples
	"""
	def __init__(self,images,names):
		self.image=images
		self.paths=names
	def __getitem__(self,idx):
		return self.image[idx],self.paths[idx]
	def __len__(self):
		return len(self.paths)

def register_fn(batch,register_path):
	"""
	Dummy collate fn in order to register the images in a multiprocessed manner
	THe output of this function is irrelevant as we just use the multiprocessed save
	"""
	for image,path in batch:
		save_image(image, register_path+path, nrow=1)
	return torch.zeros(3)



class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


class MedianFilter():
	def __init__(self,size):
		self.pad = nn.ReplicationPad2d(size)
		self.pool=MedianPool2d(kernel_size=1+2*size)
	def __call__(self,x):
		x= self.pad(x)
		x = self.pool(x)
		return x
class PreprocessLR():
	def __init__(self,size=(540, 718)):
		self.reducer = nn.Upsample(size=size, scale_factor=None, mode='bilinear', align_corners=True)
	def __call__(self,x):
		return self.reducer(x[...,161:-161].unsqueeze(0)).squeeze()


if __name__ == "__main__":
	#resize_base()
	exit()
	root="/media/will/227E8A467E8A1329/Users/willi/Documents/Batman/frames"
	# rename("/media/will/227E8A467E8A1329/Users/willi/Documents/Batman/frames/LR")
	# remap("/media/will/227E8A467E8A1329/Users/willi/Documents/Batman/frames/LR","/media/will/227E8A467E8A1329/Users/willi/Documents/Batman/frames/HR",window=24)
	# dataset = LRHRDataset(root,loader=loader,extensions=('jpg','png'),transform_lr=PreprocessLR(),transform_hr=MedianFilter(2))
	# dataset = LRHRDataset(root,loader=loader,extensions=('jpg','png'),transform_lr=None,transform_hr=None)
	
	dataset = TrainingLoaderLRHR(root,loader=loader,extensions=('jpg','png'),transform_lr=PreprocessLR(),transform_hr=MedianFilter(2))
	lr,hr = dataset[5000]
	# lr = lr.unsqueeze(0)
	# reducer = nn.Upsample(size=(540, 718), scale_factor=None, mode='nearest', align_corners=None)
	# upscaler = nn.Upsample(size=(1080, 1436), scale_factor=None, mode='nearest', align_corners=None)
	# input_image = upscaler(reducer(lr)).squeeze()

	
	save_image(hr, "hr_patch.png", nrow=1)
	save_image(lr, "lr_patch.png", nrow=1)

	exit()
	input_image.requires_grad=True
	optimizer = optim.Adam([input_image],lr=0.001)
	for i in range(10000):
		loss = laplacian_loss(input_image,output_image)
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		if i%100==0:
			save_image(input_image, "test{:05d}.png".format(i//100), nrow=1)

	exit()

	laplacian = (laplacian(hr))/2+0.5
	average = nn.ReplicationPad2d(2)(laplacian.unsqueeze(0))
	pool= MedianPool2d(kernel_size=5)
	hr_median = pool(nn.ReplicationPad2d(2)(hr.unsqueeze(0))).squeeze()
	average = pool(average).squeeze()
	save_image(laplacian, "laplacian.png", nrow=1)
	save_image(average,"median_filter.png",nrow=1)
	save_image(torch.abs(laplacian-average)*5,"noise.png",nrow=1)
	save_image(hr_median,"hr_median.png",nrow=1)
	# save_image(hr, "hr.png", nrow=1)
