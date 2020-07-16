import os
from loss import *
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FusionDataset(Dataset):
    """docstring for FusionDataset"""

    def __init__(self, root_path, transform=None):
        super(FusionDataset, self).__init__()
        self.root_path = root_path
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
        self.dir_list = os.listdir(self.root_path)

    def __len__(self):
        return len(self.dir_list)

    def __getitem__(self, id):
        return self.load_img(id)

    def load_img(self, id):
        img_path = os.path.join(self.root_path, self.dir_list[id])
        img_list = os.listdir(img_path)
        img_path_1 = os.path.join(img_path, img_list[0])
        img_path_2 = os.path.join(img_path, img_list[1])
        img_1 = Image.open(img_path_1)
        img_2 = Image.open(img_path_2)
        img_1=img_1.convert('L')
        img_2 = img_2.convert('L')
        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
        return {'img_1': img_1, 'img_2': img_2}


if __name__ == '__main__':
    train_dataset = FusionDataset('./data/TNO/', transforms.Compose([transforms.Resize((512, 512)),
                                                                     transforms.ToTensor()]))
    train_generator = DataLoader(train_dataset, batch_size=2, shuffle=True)
    for x, y in enumerate(train_generator):
        print(x, y['img_1'].shape, y['img_2'].shape)
# img1,img2,img=load_train_data('./data/TNO/',1,1)
# print(img1.shape,img2.shape)
