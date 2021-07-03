import glob
import os 
import torch

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as udata 

class ImageDataset(udata.Dataset):
    def __init__(self,path,noisy_image="",clean_image="",temps=None):
        self.noisy_image = sorted(list(glob.glob(os.path.join(path,noisy_image)+"/*")))
        self.clean_image = sorted(list(glob.glob(os.path.join(path,clean_image)+"/*")))
    def __getitem__(self,index):
        noisy = Image.open(self.noisy_image[index%len(self.noisy_image)])
        clean = Image.open(self.clean_image[index%len(self.clean_image)])
        temps = transforms.Compose([transforms.ToTensor()])
        if(noisy.size!= clean.size):
            noisy_tfm = transform.Compose([transforms.Resize((clean.size[1],clean.size[0])),transforms.ToTensor()])
            noisy = noisy_tfm(noisy)
        else:
            noisy = temps(noisy)
        clean = temps(clean)
	return {'input':noisy,'target':clean}
    def __len__(self):
        return len(self.noisy_image)
