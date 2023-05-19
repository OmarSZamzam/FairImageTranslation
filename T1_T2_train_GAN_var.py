import nibabel as nib
import io
import os
import random
import math
import numpy as np
from skimage.transform import resize
import glob
import pickle
from scipy import ndimage
import tqdm
import torch
from monai.networks.nets import unet
from torch.utils.data import DataLoader, Dataset
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
import torch.nn as nn
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
threshhold = 10

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, sample_number = 4, transform=None):
        self.sample_number = sample_number
        self.img_dir = img_dir
        self.files = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx])
        
        vol = pickle.load(open(img_path, 'rb'))
        T1, T2 = vol[0], vol[1]
        
        T1, T2 = T1[:, :-1, :], T2[:, :-1, :]
        T1, T2 = resize(T1, (208, 248, 208)), resize(T2, (208, 248, 208))
        T1, T2 = np.pad(T1, ((0,0), (4,4), (0,0)), mode='constant', constant_values=0), np.pad(T2, ((0,0), (4,4), (0,0)), mode='constant', constant_values=0)
        
        T1, T2 = T1.transpose((2,1,0)), T2.transpose((2,1,0))
        
        msk_normal = (~np.all(T1 == 0,axis=(1,2))) # Remove empty planes
        choices = np.arange(len(msk_normal))[msk_normal]
        
        sample_idx = np.array(random.choices(choices,k = self.sample_number))
        
        coord = sample_idx[:, np.newaxis] / T1.shape[0]
        coord = coord - 0.5
        
        T1, T2 = T1[sample_idx], T2[sample_idx]
        max_batch1, max_batch2 = T1.max(axis = 1), T2.max(axis = 1)
        max_batch1, max_batch2 = max_batch1.max(axis = 1), max_batch2.max(axis = 1)
        T1, T2 = T1/max_batch1.reshape((-1, 1, 1)), T2/max_batch2.reshape((-1, 1, 1))
        T1, T2 = T1.astype(np.float32), T2.astype(np.float32)
        
        T1, T2 = torch.from_numpy(T1), torch.from_numpy(T2)
        
        return T1, T2, coord, self.files[idx]
    
    
    
train_dsetHCP = CustomImageDataset(img_dir='/scratch1/zamzam/HCP_nt_train')
train_dsetCamCan = CustomImageDataset(img_dir='/scratch1/akrami/CAMCAN_nt_train')

train_dset = torch.utils.data.ConcatDataset([train_dsetHCP, train_dsetCamCan])
train_loader = DataLoader(train_dset, batch_size=20,shuffle=True,num_workers=1)

val_dsetHCP = CustomImageDataset(img_dir='/scratch1/zamzam/HCP_nt_val')
val_dsetCamCan = CustomImageDataset(img_dir='/scratch1/akrami/CAMCAN_nt_val')

val_dset = torch.utils.data.ConcatDataset([val_dsetHCP, val_dsetCamCan])
val_loader = DataLoader(val_dset, batch_size=20,shuffle=True,num_workers=1)





import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = unet.UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=3
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        ndf =32
        nc = 1
        self.main = nn.Sequential(
            # input is ``(nc) x 256 x 208`
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size.
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size.
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Flatten(),
            nn.Linear(13 * 10, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)

generator = Generator()
discriminator = Discriminator()

adversarial_loss = nn.BCELoss()
loss = torch.nn.MSELoss()

generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

generator.to(device)
discriminator.to(device)

init_loss = torch.nn.MSELoss(reduction = 'none')
for epoch in range(1000):
    total_loss = 0
    for i, data in enumerate(train_loader):
        # Training the generator
        generator_optimizer.zero_grad()

        T1, T2, _, _ = data
        T1, T2 = T1.view(-1,1,T1.shape[2],T1.shape[3]), T2.view(-1,1,T2.shape[2],T2.shape[3]) 
        T1, T2 = T1.to(device), T2.to(device)

        output = generator(T1)

        initial = torch.mean(init_loss(output, T2), dim = (1, 2, 3))
        mse = torch.mean(initial) + torch.var(initial)
        
        # Adversarial loss
        valid = torch.ones(output.size(0), 1).to(device)
        fake = torch.zeros(output.size(0), 1).to(device)

        

        gen_loss = adversarial_loss(discriminator(output), valid)
        

        if epoch> threshhold:
            g_loss = mse + gen_loss
        else:
             g_loss = mse

        g_loss.backward()
        generator_optimizer.step()

        total_loss += mse

        # Training the discriminator
        discriminator_optimizer.zero_grad()

        real_loss = adversarial_loss(discriminator(T2), valid)
        fake_loss = adversarial_loss(discriminator(output.detach()), fake)

        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        discriminator_optimizer.step()

    print('  * train  ' +
          f'Loss: {total_loss/len(train_dset):.7f}, ')

    total_loss = 0

    for i, data in enumerate(val_loader):
        
        with torch.no_grad():
            T1, T2, _, _ = data
            T1, T2 = T1.view(-1,1,T1.shape[2],T1.shape[3]), T2.view(-1,1,T2.shape[2],T2.shape[3]) 
            T1, T2 = T1.to(device), T2.to(device)

            output = generator(T1)

            mse = loss(output, T2)

            total_loss += mse
            
    torch.save(generator.state_dict(), f'/scratch1/akrami/Projects/T1_T2/models/GAN/T1_T2{epoch}_var_4s.pt')
    print('  * val  ' +
          f'Loss: {total_loss/len(val_dset):.7f}, ')
    
    print(f'epoch{epoch})
    
    
    
