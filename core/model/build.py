import torch
import torch.nn as nn
from core.model.base import Model


class build_model(nn.Module):
	"""docstring for build_model"""

	def __init__(self, config):
		super(build_model, self).__init__()

		Generator_config = config['Generator']

		self.Generator_name = Generator_config['Generator_Name']
		self.Generator_input = {G: input for G, input in
		                        zip(self.Generator_name, Generator_config['Input_Datasets'])}
		self.Generator = nn.ModuleDict({i: Model({i: config['Struct'][i]}) for i in self.Generator_name})

		Discriminator_config = config['Discriminator']
		self.Discriminator_name = Discriminator_config['Discriminator_Name']
		self.Discriminator_input = {D: input for D, input in
		                            zip(self.Discriminator_name, Discriminator_config['Input_Datasets'])}
		self.Discriminator = nn.ModuleDict({i: Model({i: config['Struct'][i]}) for i in self.Discriminator_name})

	def forward(self, inputs):

		Generator_feats = {}
		for G in self.Generator:
			Generator_inputs = inputs.copy()
			Generator_inputs = {i: Generator_inputs[i] for i in self.Generator_input[G]}
			Generator_feat = self.Generator[G](Generator_inputs)
			Generator_feat = Generator_feat[[i for i in Generator_feat][-1]]
			Generator_feats.update({G: Generator_feat})

		Discriminator_feats = {}
		confidence = {}
		for D in self.Discriminator:
			Discriminator_inputs = inputs.copy()
			# Discriminator_inputs.update({'Generator': Generator_feats['Generator']})
			Discriminator_inputs.update(Generator_feats)
			Discriminator_inputs = {i: Discriminator_inputs[i] for i in self.Discriminator_input[D]}

			confidence.update({D: torch.cat(
				[torch.zeros(Discriminator_inputs[i].shape[0]).to(Discriminator_inputs[i].device)
				 if i in self.Generator else
				 torch.ones(Discriminator_inputs[i].shape[0]).to(Discriminator_inputs[i].device)
				 for i in Discriminator_inputs], dim=0)})

			Discriminator_feat = self.Discriminator[D](Discriminator_inputs)
			Discriminator_feat = Discriminator_feat[[i for i in Discriminator_feat][-1]]
			Discriminator_feats.update({D: Discriminator_feat.squeeze()})

		return Generator_feats, Discriminator_feats, confidence
