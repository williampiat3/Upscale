import copy
import glob
import os
from multiprocessing.dummy import Pool as ThreadPool

from PIL import Image
from torchvision.transforms.functional import to_tensor

import torch.nn as nn
import torch


class ImageSplitter:
	# key points:
	# Boarder padding and over-lapping img splitting to avoid the instability of edge value
	# Thanks Waifu2x's author nagadomi for suggestions (https://github.com/nagadomi/waifu2x/issues/238)
	"""
	I changed the original splitter as is only accepted pil images  and wasn't working in a batched manner
	"""

	def __init__(self, seg_size=48, scale_factor=2, boarder_pad_size=3):
		self.seg_size = seg_size
		self.scale_factor = scale_factor
		self.pad_size = boarder_pad_size
		self.height = 0
		self.width = 0
		self.upsampler = nn.Upsample(scale_factor=scale_factor, mode='bilinear')

	def split_img_tensor(self, img_tensor, img_pad=0):
		"""
		Function to split the images into patches, img_tensor are batched tensors of images (cuda or not)
		"""


	   # Using 2D replication padding on the batch				 
		img_tensor = nn.ReplicationPad2d(self.pad_size)(img_tensor)
		batch, channel, height, width = img_tensor.size()
		self.batch = batch
		self.height = height
		self.width = width
		patch_box = []
		# avoid the residual part is smaller than the padded size
		if height % self.seg_size < self.pad_size or width % self.seg_size < self.pad_size:
			self.seg_size += self.scale_factor * self.pad_size

		# split image into over-lapping pieces
		for i in range(self.pad_size, height, self.seg_size):
			for j in range(self.pad_size, width, self.seg_size):
				part = img_tensor[:, :,
					   (i - self.pad_size):min(i + self.pad_size + self.seg_size, height),
					   (j - self.pad_size):min(j + self.pad_size + self.seg_size, width)]
				if img_pad > 0:
					part = nn.ZeroPad2d(img_pad)(part)
				patch_box.append(part)
		return patch_box

	def merge_img_tensor(self, list_img_tensor):
		"""
		Function to remerge the images once the've been scaled
		"""
		out = torch.zeros((self.batch, 3, self.height * self.scale_factor, self.width * self.scale_factor))
		img_tensors = copy.copy(list_img_tensor)
		rem = self.pad_size * 2

		pad_size = self.scale_factor * self.pad_size
		seg_size = self.scale_factor * self.seg_size
		height = self.scale_factor * self.height
		width = self.scale_factor * self.width
		for i in range(pad_size, height, seg_size):
			for j in range(pad_size, width, seg_size):
				part = img_tensors.pop(0)
				part = part[:, :, rem:-rem, rem:-rem]
				# might have error
				if len(part.size()) > 3:
					_, _, p_h, p_w = part.size()
					out[:, :, i:i + p_h, j:j + p_w] = part

		out = out[:, :, rem:-rem, rem:-rem]
		return out

