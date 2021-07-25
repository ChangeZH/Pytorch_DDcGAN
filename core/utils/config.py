import yaml
from collections import Counter


def Check_Generator(config):
	Generator_config = config['Generator']
	Generator_name = Generator_config['Generator_Name']
	Generator_counter = Counter(Generator_name)
	assert len(Generator_counter) == len(Generator_name), \
		f'The Same Name is in the {Generator_name} '

def Check_Discriminator(config):
	Discriminator_config = config['Discriminator']
	Discriminator_name = Discriminator_config['Discriminator_Name']
	Discriminator_counter = Counter(Discriminator_name)
	assert len(Discriminator_counter) == len(Discriminator_name), \
		f'The Same Name is in the {Discriminator_name} '


def load_config(filename):
	with open(filename, 'r') as f:
		config = yaml.safe_load(f)
		Check_Generator(config)
		Check_Discriminator(config)
		return config
