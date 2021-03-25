from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader,Dataset
from torchvision.utils import save_image
from PIL import Image
from torchvision.transforms.functional import to_tensor
import torch


def loader(string_path):
	#Function to load the images for the DatasetFolder
	img = Image.open(string_path).convert("RGB")
	return to_tensor(img)

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