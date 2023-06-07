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
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
from monai.networks.nets import unet
from torch.utils.data import DataLoader, Dataset
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
train_loader = DataLoader(train_dset, batch_size=2,shuffle=True,num_workers=1)

val_dsetHCP = CustomImageDataset(img_dir='/scratch1/zamzam/HCP_nt_val')
val_dsetCamCan = CustomImageDataset(img_dir='/scratch1/akrami/CAMCAN_nt_val')

val_dset = torch.utils.data.ConcatDataset([val_dsetHCP, val_dsetCamCan])
val_loader = DataLoader(val_dset, batch_size=2,shuffle=True,num_workers=1)



class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Classifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc = nn.Linear(256 * 256 * 208, num_classes)  # Assuming input size of (256, 208)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        output = x

        return output

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Downsample
        self.down_conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.down_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.down_conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.down_conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        # Upsample
        self.up_conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.up_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.up_conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.up_conv4 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Downsample
        x = self.relu(self.down_conv1(x))
        x = self.relu(self.down_conv2(x))
        x = self.relu(self.down_conv3(x))
        x = self.relu(self.down_conv4(x))

        latent = x

        # Upsample
        x = self.relu(self.up_conv1(x))
        x = self.relu(self.up_conv2(x))
        x = self.relu(self.up_conv3(x))
        x = self.up_conv4(x)

        return x, latent
    


# Create the model
in_channels = 1
out_channels = 1
num_classes = 1  # Assuming a single output for binary classification
model = UNet(in_channels, out_channels)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
loss = torch.nn.MSELoss(reduction = 'none')

classifier = Classifier(512, num_classes)
classifier = classifier.to(device)
optimizerC = torch.optim.Adam(classifier.parameters(), 1e-4)
lossC = torch.nn.BCEWithLogitsLoss()


for epoch in range(1000):
    total_loss = 0
    for i, data in enumerate(tqdm(train_loader)):
        
        classifier.zero_grad()
        
        T1, T2, _, _ = data
        T1, T2 = T1.view(-1,1,T1.shape[2],T1.shape[3]), T2.view(-1,1,T2.shape[2],T2.shape[3]) 
        T1, T2 = T1.to(device), T2.to(device)
        
        output, latent = model(T1)
        
        L = torch.mean(loss(output, T2), dim=(1,2,3))
        
        sorted_tensor, _ = torch.sort(L, descending=True)
        
        sorted_tensor = torch.zeros_like(sorted_tensor)
        
        sorted_tensor[:4] = 1
        
        
        LC = lossC(classifier(latent.detach()), sorted_tensor.unsqueeze(1).detach())
                
        LC.backward()
        
        optimizerC.step()
        
        model.zero_grad()
        
        LC = lossC(classifier(latent.detach()), sorted_tensor.unsqueeze(1).detach())
                
        mse = torch.mean(L) - LC
        
        mse.backward()
        
        optimizer.step()
    
        
        total_loss += mse
        
    print('  * train  ' +
    f'Loss: {total_loss/len(train_dset):.7f}, ')
    
    if epoch%2==0:
        total_loss = 0
            
        for i, data in enumerate(tqdm(val_loader)):
            with torch.no_grad():
                T1, T2, _, _ = data
                # T1, T2 = T1.swapaxes(0,1), T2.swapaxes(0,1)
                T1, T2 = T1.view(-1,1,T1.shape[2],T1.shape[3]), T2.view(-1,1,T2.shape[2],T2.shape[3]) 
                T1, T2 = T1.to(device), T2.to(device)

                output, latent = model(T1)

                mse = loss(output, T2)
                
                total_loss += torch.mean(mse)
                
        print('  * val  ' +
        f'Loss: {total_loss/len(val_dset):.7f}, ')
    if epoch%10==0:
        torch.save(model.state_dict(), '/home1/zamzam/Fairness/modelsClassifier/model{}.pth'.format(epoch))