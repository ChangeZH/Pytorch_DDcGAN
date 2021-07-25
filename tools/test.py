import torch
from PIL import Image
from torchvision import transforms
from core.model.build import build_model
from core.utils.config import load_config

config = load_config('../config/Pan-GAN.yaml')
GAN_Model = build_model(config)
vis_img = Image.open('../demo/test_vis.jpg')
inf_img = Image.open('../demo/test_inf.jpg')
trans = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
vis_img = trans(vis_img)
inf_img = trans(inf_img)
data = {'Vis': vis_img.unsqueeze(0), 'Inf': inf_img.unsqueeze(0)}
GAN_Model.Generator.load_state_dict(torch.load('../weights/Generator/Generator_50.pth').state_dict())
Generator_feats, Discriminator_feats = GAN_Model(data)
untrans = transforms.Compose([transforms.ToPILImage()])

img = untrans(Generator_feats['Generator'][0])
print(img.size)
img.save('test_result.jpg')
