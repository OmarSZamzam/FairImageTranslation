import nibabel as nib
import scipy
import io
import os
import random
import math
import numpy as np
from skimage.transform import resize
import glob
import pickle
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm.notebook import tqdm
import torch
from scipy import stats
from monai.networks.nets import unet
from torch.utils.data import DataLoader, Dataset
import csv
import pandas as pd

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('/scratch1/zamzam/HCP_1200.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # skip the header row
    HCP_info = [[row[0], row[4], row[3]] for row in reader]
HCP_info = np.array(HCP_info);

df = pd.read_csv('/scratch1/zamzam/CAMCAN.csv', header=None, skiprows=1)
CamCan_info = df.iloc[:, :3].apply(lambda x: x.str[2:] if x.name == 0 else x.str[0] if x.name == 2 else x).to_numpy()

print(np.shape(HCP_info))
print(np.shape(CamCan_info))

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, info, sample_number = 1, transform=None):
        self.sample_number = sample_number
        self.img_dir = img_dir
        self.files = os.listdir(img_dir)
        self.transform = transform
        
        self.files = np.array(self.files).reshape(-1, 1)
        self.files = np.hstack((self.files,  np.array(['A'] * len(self.files)).reshape(-1, 1)))

        for i in range(len(self.files)):
            for j in range(len(info)):
                if self.files[i,0] == info[j,0]:
                    self.files[i,1] = info[j,2]
                    break
                    
        self.dset = 'CC' in self.files[0,0]
        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx,0])
        
        # vol = np.load(img_path)
        # vol = vol['data1']
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
        
        return T1, T2
    


train_dsetHCP = CustomImageDataset(img_dir='/scratch1/zamzam/HCP_nt_train', info = HCP_info)
train_dsetCamCan = CustomImageDataset(img_dir='/scratch1/akrami/CAMCAN_nt_train', info = CamCan_info)

train_dset = torch.utils.data.ConcatDataset([train_dsetHCP, train_dsetCamCan])
train_loader = DataLoader(train_dset, batch_size=20,shuffle=True,num_workers=1)

val_dsetHCP = CustomImageDataset(img_dir='/scratch1/zamzam/HCP_nt_val', info = HCP_info)
val_dsetCamCan = CustomImageDataset(img_dir='/scratch1/akrami/CAMCAN_nt_val', info = CamCan_info)

val_dset = torch.utils.data.ConcatDataset([val_dsetHCP, val_dsetCamCan])
val_loader = DataLoader(val_dset, batch_size=20,shuffle=True,num_workers=1)

test_dsetHCP = CustomImageDataset(img_dir='/scratch1/zamzam/HCP_nt_test', info = HCP_info)
test_dsetCamCan = CustomImageDataset(img_dir='/scratch1/akrami/CAMCAN_nt_test', info = CamCan_info)

test_dset = torch.utils.data.ConcatDataset([test_dsetHCP, test_dsetCamCan])
test_loader = DataLoader(test_dset, batch_size=20,shuffle=True,num_workers=1)

test_loaderHCP = DataLoader(test_dsetHCP, batch_size=20,shuffle=True,num_workers=1)
test_loaderCamCan = DataLoader(test_dsetCamCan, batch_size=20,shuffle=True,num_workers=1)



model = unet.UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=3).to(device)

optimizer = torch.optim.Adam(model.parameters(), 1e-4)
init_loss = torch.nn.MSELoss(reduction = 'none')
loss = torch.nn.MSELoss()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(count_parameters(model))

for epoch in range(1000):
    total_loss = 0
    for i, data in enumerate(tqdm(train_loader)):
        
        optimizer.zero_grad()
        
        T1, T2, _, _, _ = data
        # T1, T2 = T1.swapaxes(0,1), T2.swapaxes(0,1)
        T1, T2 = T1.to(device), T2.to(device)
        
        output = model(T1)
        
        initial = torch.mean(init_loss(output, T2), dim = (1, 2, 3))
        mse = torch.mean(initial) + torch.var(initial)
        
        mse.backward()
        
        optimizer.step()
        
        total_loss += mse
        
    print('  * train  ' +
    f'Loss: {total_loss/len(train_dset):.7f}, ')
    
    if epoch%2 == 0:
    
        total_loss = 0

        for i, data in enumerate(tqdm(val_loader)):
            with torch.no_grad():
                T1, T2, _, _, _ = data
                #T1, T2 = T1.swapaxes(0,1), T2.swapaxes(0,1)
                T1, T2 = T1.to(device), T2.to(device)

                output = model(T1)

                mse = loss(output, T2)

                total_loss += mse

        print('  * val  ' +
        f'Loss: {total_loss/len(val_dset):.7f}, ')

    torch.save(generator.state_dict(), f'/scratch1/akrami/Projects/T1_T2/models/Unet/T1_T2{epoch}_unet_var.pt')