import os
import torch
import numpy as np
import torchvision
from loss import *
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

loader=transforms.Compose([transforms.ToTensor()])  
unloader=transforms.ToPILImage()

def image_save(tensor,name):
	image=tensor.cpu().clone()
	image=image.squeeze(0)
	image=unloader(image)
	image.save(name)

def image_loader(image_name,gpu,shape=320):
	img=Image.open(image_name)#.convert('RGB')
	image=img.resize((shape,shape))
	image=loader(image).unsqueeze(0)
	image=image.to('cuda:'+str(gpu), torch.float32)
	if image.shape[1]>1:
		image=image[:,0,:,:]
		image=torch.reshape(image,(1,1,shape,shape))
	return image

def load_train_data(path,batch,batch_size,gpu):
	dirname=os.listdir(path)
	imgname=[]
	for i in dirname:
		img=os.listdir(path+i)
		img=[path+i+'/'+img[0],path+i+'/'+img[1]]
		imgname.append(img)
	for i in range(batch*batch_size,min(len(imgname),(batch+1)*batch_size)):
		if i==batch*batch_size:
			train_data1=image_loader(imgname[i][0],gpu,256)
			train_data2=image_loader(imgname[i][1],gpu,64)
		else:
			data1=image_loader(imgname[i][0],gpu,256)
			data2=image_loader(imgname[i][1],gpu,64)
			train_data1=torch.cat((train_data1,data1),0)
			train_data2=torch.cat((train_data2,data2),0)
	return train_data1,train_data2,len(imgname)

if __name__=='__main__':
	img1,img2,img=load_train_data('../../dataset/TNO/',0,2,0)
	print(img1.shape,img2.shape)