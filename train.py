import os
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
from model import DDcGAN
from data import Load_Dataset
from torchvision import transforms
from loss import L_con,L_adv_G,L_G,L_Dv,L_Di
from torch.utils.data import Dataset, DataLoader

parser=argparse.ArgumentParser(description='train')

parser.add_argument('--epoch',type=int,default=24,help='config file')
parser.add_argument('--bs',type=int,default=2,help='config file')
parser.add_argument('--lr',type=float,default=0.00001,help='config file')
parser.add_argument('--device',type=float,default=None,help='config file')

args=parser.parse_args()

def train_Generator(model,l_Gmax,l_min,loss_G,loss_adv,loss_dv,loss_di,vis,ir,max_step,lr):
	for i in model.G.parameters():
		i.requires_grad=True
	for i in model.Dv.parameters():
		i.requires_grad=False
	for i in model.Di.parameters():
		i.requires_grad=False
	opt=torch.optim.RMSprop(model.parameters(),lr)
	for mini_step in range(0,max_step+1):
		fusion_v,fusion_i,score_v,score_i,score_Gv,score_Gi=model(vis,ir)
		loss_v=loss_dv(score_v,score_Gv)
		loss_i=loss_di(score_i,score_Gi)
		loss_adv_G=loss_adv(score_Gv,score_Gi)
		loss_g=loss_G(vis,ir,fusion_v,score_Gv,score_Gi)
		print('G step:',mini_step,'\tloss_adv_G:',loss_adv_G.item(),'\tloss_g:',loss_g.item())
		if loss_v<l_min or loss_i<l_min:
			opt.zero_grad()
			loss_adv_G.backward()
			opt.step()
		elif loss_g>l_Gmax:
			opt.zero_grad()
			loss_g.backward()
			opt.step()
		# opt.zero_grad()
		# loss_g.backward()
		# opt.step()
		if loss_g<=l_Gmax:
			break

	return model

def train_Discriminators(model,l_max,loss_G,loss_dv,loss_di,vis,ir,max_step,lr):
	for i in model.G.parameters():
		i.requires_grad=False
	for i in model.Dv.parameters():
		i.requires_grad=True
	for i in model.Di.parameters():
		i.requires_grad=False
	opt=torch.optim.SGD(model.parameters(),lr)
	for mini_step in range(0,max_step+1):
		_,_,score_v,_,score_Gv,_=model(vis,ir)
		loss_v=loss_dv(score_v,score_Gv)
		# print(loss_v,score_v,score_Gv)
		print('Dv step:',mini_step,'\tloss_v:',loss_v.item())
		if loss_v<=l_max:
			break
		opt.zero_grad()
		loss_v.backward()
		opt.step()

	for i in model.G.parameters():
		i.requires_grad=False
	for i in model.Dv.parameters():
		i.requires_grad=False
	for i in model.Di.parameters():
		i.requires_grad=True
	opt=torch.optim.SGD(model.parameters(),lr)
	for mini_step in range(0,max_step+1):
		_,_,_,score_i,_,score_Gi=model(vis,ir)
		loss_i=loss_di(score_i,score_Gi)
		print('Di step:',mini_step,'\tloss_i:',loss_i.item())
		if loss_i<=l_max:
			break
		opt.zero_grad()
		loss_i.backward()
		opt.step()
	fusion_v,fusion_i,score_v,score_i,score_Gv,score_Gi=model(vis,ir)
	L_G=loss_G(vis,ir,fusion_v,score_Gv,score_Gi)
	return model,L_G

def main():
	l_max=1.3
	l_min=1.0
	max_step=20
	batch_size=args.bs
	root_dir='./datasets/TNO'
	train_dataset=Load_Dataset(root_dir,
							transforms.Compose([transforms.Resize((256,256)),
												transforms.ToTensor()]))
	train_generator=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
	num_iter_per_epoch=len(train_generator)
	train_bar=tqdm(train_generator)
	if args.device:
		model=DDcGAN(if_train=True).to('cuda:'+str(args.device))
	else:
		model=DDcGAN(if_train=True)
	model.train()

	Loss_G=L_G()
	Loss_adv_G=L_adv_G()
	Loss_Dv=L_Dv()
	Loss_Di=L_Di()

	for epoch in range(0,args.epoch):
		print('epoch:',epoch)
		for iter,data in enumerate(train_bar):
			if args.device:
				vis=data[0].to('cuda:'+str(args.device))
				ir=data[1].to('cuda:'+str(args.device))
			else:
				vis=data[0]
				ir=data[1]

			model,loss_G=train_Discriminators(model,l_max,Loss_G,Loss_Dv,Loss_Di,vis,ir,max_step,args.lr)
			L_G_max=0.8*loss_G
			model=train_Generator(model,L_G_max,l_min,Loss_G,Loss_adv_G,Loss_Dv,Loss_Di,vis,ir,max_step,args.lr)
            # progress_bar.set_description(
            #     'Epoch: {}/{}.ssim loss: {:.5f}.Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
            #         step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, ssim_loss.item(),cls_loss.item(),
            #         reg_loss.item(), loss.item()))
		torch.save(model,'./model/model'+str(epoch)+'.pth')

if __name__=='__main__':
	main()
