import os
import cv2
import numpy as np
from scipy.ndimage import filters
from PIL import Image
from collections import Counter
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class Fusion_Datasets(Dataset):
	"""docstring for Fusion_Datasets"""

	def __init__(self, root_dir, sensors, transform=None):
		super(Fusion_Datasets, self).__init__()
		self.root_dir = root_dir
		self.transform = transform
		self.sensors = sensors
		self.img_list = {i: os.listdir(os.path.join(self.root_dir, i)) for i in self.sensors}
		self.img_path = {i: [os.path.join(self.root_dir, i, j) for j in os.listdir(os.path.join(self.root_dir, i))]
		                 for i in self.sensors}

	def __getitem__(self, index):
		img_data = {}
		for i in self.sensors:
			img = Image.open(self.img_path[i][index])
			# if i == 'Inf':
			# 	img = np.array(img)
			# 	imx = np.zeros(img.shape)
			# 	filters.sobel(img, 1, imx)
			# 	imy = np.zeros(img.shape)
			# 	filters.sobel(img, 0, imy)
			# 	magnitude = np.sqrt(imx ** 2 + imy ** 2)
			# 	magnitude.dtype = np.uint
			# 	img = Image.fromarray(magnitude)
			# img = Image.fromarray(img)
			if self.transform is not None:
				img = self.transform(img)
			img_data.update({i: img})
		return img_data

	def __len__(self):
		img_num = [len(self.img_list[i]) for i in self.img_list]
		img_counter = Counter(img_num)
		assert len(img_counter) == 1, 'Sensors Has Different len'
		return img_num[0]


if __name__ == '__main__':
	datasets = Fusion_Datasets(root_dir='../../datasets/TNO/', sensors=['Vis', 'Inf'],
	                           transform=transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()]))
	train = DataLoader(datasets, 1, False)
	print(len(train))
	for i, data in enumerate(train):
		print(data)
