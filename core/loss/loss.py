import torch
import torch.nn as nn


class Generator_Loss(nn.Module):
	"""docstring for Generator_Loss"""

	def __init__(self, Generator_config):
		super(Generator_Loss, self).__init__()
		self.Loss_adv_weight = Generator_config['Loss_adv_weight']
		self.Loss_Dist_weight = Generator_config['Loss_Dist_weight']
		self.Dist_Loss = Generator_config['Dist_Loss']
		self.MSELoss = nn.MSELoss()

	def forward(self, Input, Generator, Discriminator, confidence):
		# Generator_name = [i for i in Generator]
		# batch_size = Generator[Generator_name[0]].shape[0]
		Discriminator_name = [i for i in Discriminator]

		loss_dist = 0
		for i in self.Dist_Loss:
			loss_dist = loss_dist + self.MSELoss(Generator[i[0]], Input[i[1]])

		loss_adv = 0
		for i in Discriminator_name:
			loss_adv = loss_adv + self.MSELoss(Discriminator[i], confidence[i])

		return self.Loss_Dist_weight * loss_dist + self.Loss_adv_weight * loss_adv


class Discriminator_Loss(nn.Module):
	"""docstring for Discriminator_Loss"""

	def __init__(self, Discriminator_config):
		super(Discriminator_Loss, self).__init__()
		self.MSELoss = nn.MSELoss()

	def forward(self, Generator, Discriminator, confidence):
		# Generator = Generator['Generator']
		# Generator_name = [i for i in Generator]
		Discriminator_name = [i for i in Discriminator]

		loss_adv = 0
		for i in Discriminator_name:
			loss_adv = loss_adv + self.MSELoss(Discriminator[i], confidence[i])
		return loss_adv
