import os
import sys
import torch
from tqdm import tqdm
import torch.optim as optim
sys.path.append(".")
from core.model import build_model
from torchvision import transforms
from torch.utils.data import DataLoader
from core.utils import load_config, debug
from core.dataset.fusion_datasets import Fusion_Datasets
from core.loss.loss import Generator_Loss, Discriminator_Loss


def train_Discriminator(epoch, opt, datasets_generator, GAN_Model, Discriminator_Loss, Discriminator_Train_config):
	for i in GAN_Model.Generator.parameters():
		i.requires_grad = False
	for i in GAN_Model.Discriminator.parameters():
		i.requires_grad = True
	num_iter = len(datasets_generator)
	train_times_per_epoch = Discriminator_Train_config['train_times_per_epoch']
	min_loss_per_epoch = Discriminator_Train_config['min_loss_per_epoch']
	train_times = 0
	min_loss = min_loss_per_epoch
	while train_times < train_times_per_epoch and min_loss >= min_loss_per_epoch:
		allloss = 0
		with tqdm(total=num_iter) as train_Discriminator_bar:
			for index, data in enumerate(datasets_generator):
				if torch.cuda.is_available():
					for i in data:
						data[i] = data[i].cuda()
				Generator_feats, Discriminator_feats, confidence = GAN_Model(data)
				D_loss = Discriminator_Loss(Generator_feats, Discriminator_feats, confidence)
				opt.zero_grad()
				D_loss.backward()
				opt.step()
				allloss = allloss + D_loss
				train_Discriminator_bar.set_description('\tepoch:%s Train_D iter:%s loss:%.5f' %
				                                        (epoch, index, allloss / num_iter))
				train_Discriminator_bar.update(1)
			min_loss = allloss / num_iter
			train_times += 1


def train_Generator(epoch, opt, datasets_generator, GAN_Model, Generator_Loss, Generator_Train_config):
	for i in GAN_Model.Generator.parameters():
		i.requires_grad = True
	for i in GAN_Model.Discriminator.parameters():
		i.requires_grad = False
	num_iter = len(datasets_generator)
	train_times_per_epoch = Generator_Train_config['train_times_per_epoch']
	min_loss_per_epoch = Generator_Train_config['min_loss_per_epoch']
	train_times = 0
	min_loss = min_loss_per_epoch
	while train_times < train_times_per_epoch and min_loss >= min_loss_per_epoch:
		allloss = 0
		with tqdm(total=num_iter) as train_Generator_bar:
			for index, data in enumerate(datasets_generator):
				if torch.cuda.is_available():
					for i in data:
						data[i] = data[i].cuda()
				Generator_feats, Discriminator_feats, confidence = GAN_Model(data)
				G_loss = Generator_Loss(data, Generator_feats, Discriminator_feats, confidence)
				opt.zero_grad()
				G_loss.backward()
				opt.step()
				allloss = allloss + G_loss
				debug(epoch, data, Generator_feats,
				      mean=Generator_Train_config['mean'], std=Generator_Train_config['std'])
				train_Generator_bar.set_description('\tepoch:%s Train_G iter:%s loss:%.5f' %
				                                    (epoch, index, allloss / num_iter))
				train_Generator_bar.update(1)
			min_loss = allloss / num_iter
			train_times += 1


def runner():
	project_name = 'GAN_G1_D2'

	try:
		os.mkdir(f'./weights/{project_name}/')
		os.mkdir(f'./weights/{project_name}/Generator/')
		os.mkdir(f'./weights/{project_name}/Discriminator/')
	except:
		pass

	config = load_config(f'./config/{project_name}.yaml')
	GAN_Model = build_model(config)

	Datasets_config = config['Dataset']
	Generator_config = config['Generator']
	Discriminator_config = config['Discriminator']
	Base_Train_config = config['Train']['Base']
	Generator_Train_config = config['Train']['Generator']
	Discriminator_Train_config = config['Train']['Discriminator']

	mean, std = Datasets_config['mean'], Datasets_config['std']
	input_size = Datasets_config['input_size']
	train_dataloader = Fusion_Datasets(root_dir=Datasets_config['root_dir'], sensors=Datasets_config['sensors'],
	                                   transform=transforms.Compose([transforms.Resize((input_size, input_size)),
	                                                                 transforms.ToTensor(),
	                                                                 transforms.Normalize(mean, std)]))
	train_generator = DataLoader(train_dataloader, batch_size=Base_Train_config['batch_size'], shuffle=False)

	G_Loss = Generator_Loss(Generator_config)
	D_Loss = Discriminator_Loss(Discriminator_config)

	if torch.cuda.is_available():
		Generator = GAN_Model.Generator.cuda()
		Discriminator = GAN_Model.Discriminator.cuda()

	opt_generator = eval('optim.' + Generator_Train_config['opt'])(Generator.parameters(), Generator_Train_config['lr'])
	opt_discriminator = eval('optim.' + Discriminator_Train_config['opt'])(Discriminator.parameters(),
	                                                                       Discriminator_Train_config['lr'])

	for epoch in range(1, Base_Train_config['epoch'] + 1):
		# exit()
		train_Discriminator(epoch, opt_discriminator, train_generator, GAN_Model, D_Loss, Discriminator_Train_config)
		train_Generator(epoch, opt_generator, train_generator, GAN_Model, G_Loss, Generator_Train_config)
		torch.save(Generator, f'./weights/{project_name}/Generator/Generator_{epoch}.pth')
		torch.save(Discriminator, f'./weights/{project_name}/Discriminator/Discriminator_{epoch}.pth')


if __name__ == '__main__':
	runner()
