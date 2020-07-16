import math
from model import *
from data import FusionDataset

parser = argparse.ArgumentParser(description = 'train')

parser.add_argument('--epoch', type = int, default = 2500, help = 'config file')
parser.add_argument('--bs', type = int, default = 4, help = 'config file')
parser.add_argument('--lr', type = float, default = 0.001, help = 'config file')
parser.add_argument('--test', type = bool, default = True, help = 'config file')

args = parser.parse_args()

def train():
	epoch=args.epoch
	batch_size=args.bs
	lr=args.lr
	criterion = SSIM_Loss()#nn.MSELoss()#
	model=VIFNet_bn().cuda()

	train_dataset = FusionDataset('./data/TNO/', transforms.Compose([transforms.Resize((512, 512)),
																	 transforms.ToTensor()]))
	train_generator = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

	opt=torch.optim.Adam(model.parameters(),lr)
	for i in range(1,epoch+1):
		allloss=0
		with tqdm(total=len(train_generator)) as train_bar:
			for x, y in enumerate(train_generator):
				input1=y['img_1'].cuda()
				input2=y['img_2'].cuda()
				train_bar.update(1)
				output=model(input1,input2)
				loss = criterion(input1,input2,output)
				opt.zero_grad()
				loss.backward()
				opt.step()
				allloss=allloss+loss
				# image_save(output,'./output/'+str(i)+'_'+str(j)+'.jpg')
				# if i//800-i/800==0:
				# 	lr=lr/10
				# 	opt=torch.optim.Adam(model.parameters(),lr)
				train_bar.set_description('epoch:%s loss:%.5f'%(i,allloss))
		# torch.save(model,'./model/model'+str(i)+'.pth')

if __name__=='__main__':
	train()
