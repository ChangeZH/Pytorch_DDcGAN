import os
import cv2
import numpy as np
from scipy.ndimage import filters
from PIL import Image
from collections import Counter
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class Classification_Datasets(Dataset):
	"""docstring for Classification_Datasets"""

	def __init__(self, root_dir, sensors, transform=None):
		super(Classification_Datasets, self).__init__()
		self.root_dir = root_dir
		self.transform = transform
		self.sensors = sensors
		self.obj_list = os.listdir(os.path.join(self.root_dir, self.sensors[0]))
		self.img_list = {j: os.listdir(os.path.join(self.root_dir, self.sensors[0], j)) for j in self.obj_list}
		self.img_path = {k: [] for k in self.sensors}
		self.img_label = []
		for i in self.obj_list:
			for j in self.img_list[i]:
				for k in self.sensors:
					self.img_path[k].append(os.path.join(self.root_dir, k, i, j))
					self.img_label.append(self.obj_list.index(i))

	def __getitem__(self, index):
		img = {i: Image.open(self.img_path[i][index]) for i in self.sensors}
		label = self.img_label[index]
		if self.transform is not None:
			img = {i: self.transform(img[i]) for i in img}
		img_data = {'img': img, 'label': label}
		return img_data

	def __len__(self):
		length = [len(self.img_path[i]) for i in self.img_path]
		return length[0]


if __name__ == '__main__':
	datasets = Classification_Datasets(root_dir='../../datasets/NWPU-RESISC45/', sensors=['RGB'],
	                                   transform=transforms.Compose(
		                                   [transforms.Resize((512, 512)), transforms.ToTensor()]))
	train = DataLoader(datasets, 1, True)
	for i, data in enumerate(train):
		print(data['img'], data['label'])
