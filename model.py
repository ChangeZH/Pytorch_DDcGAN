import os
import torch
from data import *
import torch.nn as nn

class Generator(nn.Module):
	"""docstring for Generator"""
	def __init__(self):
		super(Generator, self).__init__()
		
		self.Vis_DeCon=nn.Sequential(
			nn.ConvTranspose2d(1,1,1,1))
		self.IR_DeCon=nn.Sequential(
			nn.ConvTranspose2d(1,1,4,4))

		self.conv1=nn.Sequential(
			nn.Conv2d(2,16,3,1,1),
			nn.BatchNorm2d(16),
			nn.ReLU())

		self.conv2=nn.Sequential(
			nn.Conv2d(16,16,3,1,1),
			nn.BatchNorm2d(16),
			nn.ReLU())

		self.conv3=nn.Sequential(
			nn.Conv2d(32,16,3,1,1),
			nn.BatchNorm2d(16),
			nn.ReLU())

		self.conv4=nn.Sequential(
			nn.Conv2d(48,16,3,1,1),
			nn.BatchNorm2d(16),
			nn.ReLU())

		self.conv5=nn.Sequential(
			nn.Conv2d(64,16,3,1,1),
			nn.BatchNorm2d(16),
			nn.ReLU())

		self.Decoder1=nn.Sequential(
			nn.Conv2d(80,128,3,1,1),
			nn.BatchNorm2d(128),
			nn.ReLU())

		self.Decoder2=nn.Sequential(
			nn.Conv2d(128,64,3,1,1),
			nn.BatchNorm2d(64),
			nn.ReLU())

		self.Decoder3=nn.Sequential(
			nn.Conv2d(64,32,3,1,1),
			nn.BatchNorm2d(32),
			nn.ReLU())

		self.Decoder4=nn.Sequential(
			nn.Conv2d(32,16,3,1,1),
			nn.BatchNorm2d(16),
			nn.ReLU())

		self.Decoder5=nn.Sequential(
			nn.Conv2d(16,1,3,1,1),)
			# nn.BatchNorm2d(1),
			# nn.ReLU())

	def forward(self,vis,ir):
		vis=self.Vis_DeCon(vis)
		ir=self.IR_DeCon(ir)
		x=torch.cat((vis,ir),1)

		x1=self.conv1(x)
		x2=self.conv2(x1)
		x3=self.conv3(torch.cat((x1,x2),1))
		x4=self.conv4(torch.cat((x1,x2,x3),1))
		x5=self.conv5(torch.cat((x1,x2,x3,x4),1))
		x=torch.cat((x1,x2,x3,x4,x5),1)

		x=self.Decoder1(x)
		x=self.Decoder2(x)
		x=self.Decoder3(x)
		x=self.Decoder4(x)
		x=self.Decoder5(x)
		return x

class Discriminator_v(nn.Module):
	"""docstring for Discriminator_v"""
	def __init__(self):
		super(Discriminator_v, self).__init__()
		self.conv1=nn.Sequential(
			nn.Conv2d(1,16,3,2,1),
			nn.BatchNorm2d(16),
			nn.ReLU())

		self.conv2=nn.Sequential(
			nn.Conv2d(16,32,3,2,1),
			nn.BatchNorm2d(32),
			nn.ReLU())

		self.conv3=nn.Sequential(
			nn.Conv2d(32,64,3,2,1),
			nn.BatchNorm2d(64),
			nn.ReLU())

		self.fc=nn.Sequential(
			nn.Linear(65536,1),
			nn.Sigmoid())

	def forward(self,v):
		batch_szie=v.shape[0]
		v=self.conv1(v)
		v=self.conv2(v)
		v=self.conv3(v)
		v=v.view((batch_szie,-1))
		v=self.fc(v)
		return v

class Discriminator_i(nn.Module):
	"""docstring for Discriminator_i"""
	def __init__(self):
		super(Discriminator_i, self).__init__()
		self.conv1=nn.Sequential(
			nn.Conv2d(1,16,3,2,1),
			nn.BatchNorm2d(16),
			nn.ReLU())

		self.conv2=nn.Sequential(
			nn.Conv2d(16,32,3,2,1),
			nn.BatchNorm2d(32),
			nn.ReLU())

		self.conv3=nn.Sequential(
			nn.Conv2d(32,64,3,2,1),
			nn.BatchNorm2d(64),
			nn.ReLU())

		self.fc=nn.Sequential(
			nn.Linear(4096,1),
			nn.Sigmoid())

	def forward(self,i):
		batch_szie=i.shape[0]
		i=self.conv1(i)
		i=self.conv2(i)
		i=self.conv3(i)
		i=i.view((batch_szie,-1))
		i=self.fc(i)
		return i

class DDcGAN(nn.Module):
	"""docstring for DDcGAN"""
	def __init__(self, if_train=False):
		super(DDcGAN, self).__init__()
		self.if_train=if_train

		self.G=Generator()
		self.Dv=Discriminator_v()
		self.Di=Discriminator_i()
		self.down=nn.Sequential(
			nn.AvgPool2d(3,2,1),
			nn.AvgPool2d(3,2,1))

	def forward(self,vis,ir):
		fusion_v=self.G(vis,ir)
		# image_save(fusion_v[0:1,:,:,:],'./test/'+str(len(os.listdir('./test')))+'.jpg')
		# image_save(fusion_v[1:2,:,:,:],'./test/'+str(len(os.listdir('./test')))+'.jpg')
		if self.if_train:
			fusion_i=self.down(fusion_v)
			score_v=self.Dv(vis)
			score_i=self.Di(ir)
			score_Gv=self.Dv(fusion_v)
			score_Gi=self.Di(fusion_i)
			return fusion_v,fusion_i,score_v,score_i,score_Gv,score_Gi
		else:
			return fusion_v

if __name__=='__main__':
	vis=torch.rand((1,1,256,256))
	ir=torch.rand((1,1,64,64))
	model=Discriminator_i()
	output=model(ir)
	# model=Generator()
	# output=model(vis,ir)
	print(output.shape)