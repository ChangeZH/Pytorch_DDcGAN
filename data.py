import os
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class Load_Dataset(Dataset):
	"""docstring for Load_Dataset"""
	def __init__(self,root_dir,transforms):
		super(Load_Dataset, self).__init__()
		self.root_dir=root_dir
		self.transforms=transforms
		self.imageidlist=self.load_list()

	def __len__(self):
		return len(self.imageidlist)

	def __getitem__(self,id):
		vis,ir=self.load_image(id)
		W,H=vis.size
		vis=self.transforms(vis)
		ir=self.transforms(ir)
		return [vis,ir]

	def load_list(self):
		return [[os.path.join(self.root_dir,x,os.listdir(os.path.join(self.root_dir,x))[0]),\
				os.path.join(self.root_dir,x,os.listdir(os.path.join(self.root_dir,x))[1])] \
				for x in os.listdir(self.root_dir)]

	def load_image(self,image_id):
		image_path=self.imageidlist[image_id]
		vis=Image.open(image_path[0])
		ir=Image.open(image_path[1])
		vis=vis.convert('L')
		ir=ir.convert('L')
		# if len(image.split())==1:
		# 	image=Image.merge('RGB',(image,image,image))
		return vis,ir

if __name__=='__main__':
	root_dir='./datasets/TNO'
	val_dataset=Load_Dataset(root_dir,
							transforms.Compose([transforms.Resize((512,512)),
												transforms.ToTensor()]))
	val_generator=DataLoader(val_dataset,batch_size=2,shuffle=True)
	num_iter_per_epoch=len(val_generator)
	print(num_iter_per_epoch)

	for iter,data in enumerate(val_generator):
		print('iter',iter,data[0].shape,data[1].shape)
