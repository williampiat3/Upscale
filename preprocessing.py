from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader,Dataset
from torchvision.utils import save_image
from load_and_write import MedianFilter,loader
from PIL import Image
from torchvision.transforms.functional import to_tensor
import torch
import os
from os import listdir
from os.path import isfile, join
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
import torch.optim as optim
import numpy as np
import shutil
import random
from tqdm import tqdm

def remap(folder1,folder2,window=24):
	onlyfiles_lr = list(sorted([f for f in listdir(folder1) if isfile(join(folder1, f))]))
	onlyfiles_hr = [f for f in listdir(folder2) if isfile(join(folder2, f))]
	upscaler = nn.Upsample(size=(1080, 1436), scale_factor=None, mode='nearest', align_corners=None)
	for k,f in enumerate(sorted(onlyfiles_lr)):
		if k<=1000:
			continue
		min_file=f
		min_value=np.inf
		path_init = os.path.join(folder1, f)
		image_init = loader(path_init)
		image_init=upscaler(image_init.unsqueeze(0)).squeeze()
		for i in range(window):
			f2 = onlyfiles_lr[max(0,i-window//2+k)]
			path_temp = os.path.join(folder2, f2)
			image2 = loader(path_temp)
			distance = torch.sum(torch.abs(image_init-image2))
			if distance<min_value:
				min_file=f2
				min_value=distance
		print(f,min_file)
def resize_base():
	path_init="/media/will/227E8A467E8A1329/Users/willi/Documents/Batman/frames/HR"
	# path_dest = "/home/will/Dataset/episodes/LR3"
	onlyfiles_lr = [f for f in listdir(path_init) if isfile(join(path_init, f))]
	# reducer = nn.Upsample(size=(540, 718), scale_factor=None, mode='bilinear', align_corners=True)
	for f in tqdm(onlyfiles_lr):
		img = loader(join(path_init, f))
		if img.shape[2]!=1436 or img.shape[1]!=1080:
			print(f)
		# img = reducer(img.unsqueeze(0)).squeeze()
		# save_image(img[...,121:-121],join(path_dest, f))


def rename(folder1):
	onlyfiles_lr = [f for f in listdir(folder1) if isfile(join(folder1, f))]
	for k,f in enumerate(sorted(onlyfiles_lr)):
		number = int(f.split('a')[1].split('.')[0])
		path_init = os.path.join(folder1, f)
		# print(path_init,os.path.join("/media/will/227E8A467E8A1329/Users/willi/Documents/Batman/episodes/LRJ", "justice{:09d}.png".format(number-1)))
		shutil.move(path_init,os.path.join(folder1+"temp","dupa{:09d}.png".format(number+11)))
if __name__ == "__main__":
	resize_base()
	# exit()
	# root="/media/will/227E8A467E8A1329/Users/willi/Documents/Batman/frames"
	# rename("/media/will/227E8A467E8A1329/Users/willi/Documents/Batman/episodes/LRS2")
	# remap("/media/will/227E8A467E8A1329/Users/willi/Documents/Batman/episodes/LRB","/media/will/227E8A467E8A1329/Users/willi/Documents/Batman/episodes/HRB",window=40)
	# exit()