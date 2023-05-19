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
    def __init__(self, img_dir, info, sample_number = 4, transform=None):
        self.sample_number = sample_number
        self.img_dir = img_dir
        self.files = os.listdir(img_dir)
        self.transform = transform
        
        self.files = np.array(self.files).reshape(-1, 1)
        self.files = np.hstack((self.files,  np.array(['A'] * len(self.files)).reshape(-1, 1)))
        
        self.dset = int('CC' in self.files[0,0])

        for i in range(len(self.files)):
            for j in range(len(info)):
                if self.files[i,0][0+6*self.dset:-3] == info[j,0]:
                    self.files[i,1] = info[j,2]
                    break

        

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
        
        return T1, T2, self.files[idx,1], self.files[idx,0], self.dset
    


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

trainHCP = os.listdir('/scratch1/zamzam/HCP_nt_train')
for i in range(len(trainHCP)):
    trainHCP[i] = trainHCP[i][:-3]
trainCamCan = os.listdir('/scratch1/akrami/CAMCAN_nt_train')
for i in range(len(trainCamCan)):
    trainCamCan[i] = trainCamCan[i][6:-3]
    
trainHCP = np.array(trainHCP).reshape(-1, 1)
trainHCP = np.hstack((trainHCP, np.array(['A'] * len(trainHCP)).reshape(-1, 1)))

trainCamCan = np.array(trainCamCan).reshape(-1, 1)
trainCamCan = np.hstack((trainCamCan,  np.array(['A'] * len(trainCamCan)).reshape(-1, 1)))

for i in range(len(trainHCP)):
    for j in range(len(HCP_info)):
        if trainHCP[i,0] == HCP_info[j,0]:
            trainHCP[i,1] = HCP_info[j,2]
            break
for i in range(len(trainCamCan)):
    for j in range(len(CamCan_info)):
        if trainCamCan[i,0][:] == CamCan_info[j,0]:
            trainCamCan[i,1] = CamCan_info[j,2]
            break
            

trainHCP = os.listdir('/scratch1/zamzam/HCP_nt_train')
for i in range(len(trainHCP)):
    trainHCP[i] = trainHCP[i][:-3]
trainCamCan = os.listdir('/scratch1/akrami/CAMCAN_nt_train')
for i in range(len(trainCamCan)):
    trainCamCan[i] = trainCamCan[i][6:-3]
    
trainHCP = np.array(trainHCP).reshape(-1, 1)
trainHCP = np.hstack((trainHCP, np.array(['A'] * len(trainHCP)).reshape(-1, 1)))

trainCamCan = np.array(trainCamCan).reshape(-1, 1)
trainCamCan = np.hstack((trainCamCan,  np.array(['A'] * len(trainCamCan)).reshape(-1, 1)))

for i in range(len(trainHCP)):
    for j in range(len(HCP_info)):
        if trainHCP[i,0] == HCP_info[j,0]:
            trainHCP[i,1] = HCP_info[j,2]
            break
for i in range(len(trainCamCan)):
    for j in range(len(CamCan_info)):
        if trainCamCan[i,0][:] == CamCan_info[j,0]:
            trainCamCan[i,1] = CamCan_info[j,2]
            break
            

confusion_matrix = [[0,0],[0,0]]
confusion_matrix = np.array(confusion_matrix)
for i in range(len(trainHCP)):
    if trainHCP[i,1]=='M':
        confusion_matrix[0,0]+=1
    if trainHCP[i,1]=='F':
        confusion_matrix[0,1]+=1
        
for i in range(len(trainCamCan)):
    if trainCamCan[i,1]=='M':
        confusion_matrix[1,0]+=1
    if trainCamCan[i,1]=='F':
        confusion_matrix[1,1]+=1
        

ratios = np.sum(confusion_matrix)/confusion_matrix
ratios = ratios/np.sum(ratios)
print(confusion_matrix)
print(ratios)




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

adversarial_loss = nn.BCELoss(reduction = 'none')
init_loss = torch.nn.MSELoss(reduction = 'none')
loss = torch.nn.MSELoss()

generator_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

generator.to(device)
discriminator.to(device)


for epoch in range(1000):
    total_loss = 0
    for i, data in enumerate(tqdm.tqdm(train_loader)):
        # Training the generator
        generator_optimizer.zero_grad()

        T1, T2, s, _, d = data
        #T1, T2 = T1.swapaxes(0,1), T2.swapaxes(0,1)
        T1, T2 = T1.view(-1,1,T1.shape[2],T1.shape[3]), T2.view(-1,1,T2.shape[2],T2.shape[3]) 
        T1, T2 = T1.to(device), T2.to(device)
        s,d = np.repeat(np.array(s), 4), np.repeat(np.array(d), 4)

        output = generator(T1)

        initial = torch.mean(init_loss(output, T2), dim = (1, 2, 3))
        
        # Adversarial loss
        valid = torch.ones(output.size(0), 1).to(device)
        fake = torch.zeros(output.size(0), 1).to(device) 

        gen_loss = torch.mean(adversarial_loss(discriminator(output), valid), dim = (1, 2, 3))
        
        if epoch >threshhold:
            g_loss = mse + gen_loss
        else:
            g_loss = mse

        losses = torch.zeros((4))
        
        losses[0] = torch.nan_to_num(torch.mean(g_loss[(np.array(s)=='M') & (np.array(d)==1)]), nan=100) #100 is a big number to avoid nan
        losses[1] = torch.nan_to_num(torch.mean(g_loss[(np.array(s)=='F') & (np.array(d)==1)]), nan=100)
        losses[2] = torch.nan_to_num(torch.mean(g_loss[(np.array(s)=='M') & (np.array(d)==0)]), nan=100)
        losses[3] = torch.nan_to_num(torch.mean(g_loss[(np.array(s)=='F') & (np.array(d)==0)]), nan=100)
        
        mse = torch.mean(g_loss) + torch.mean((losses - torch.min(losses))[losses!=100])

        g_loss.backward()
        generator_optimizer.step()

        total_loss += mse

        # Training the discriminator
        discriminator_optimizer.zero_grad()

        real_loss = torch.mean(adversarial_loss(discriminator(T2), valid))
        fake_loss = torch.mean(adversarial_loss(discriminator(output.detach()), fake))

        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        discriminator_optimizer.step()

    print('  * train  ' +
          f'Loss: {total_loss/len(train_dset):.7f}, ')

    total_loss = 0

    for i, data in enumerate(tqdm.tqdm(val_loader)):
        torch.save(generator.state_dict(), f'/scratch1/akrami/Projects/T1_T2/models/GAN/T1_T2{epoch}_b20_w.pt')
        with torch.no_grad():
            T1, T2, _, _ = data
            #T1, T2 = T1.swapaxes(0,1), T2.swapaxes(0,1)
            T1, T2 = T1.view(-1,1,T1.shape[2],T1.shape[3]), T2.view(-1,1,T2.shape[2],T2.shape[3]) 
            T1, T2 = T1.to(device), T2.to(device)

            output = generator(T1)

            mse = loss(output, T2)

            total_loss += mse

    print('  * val  ' +
          f'Loss: {total_loss/len(val_dset):.7f}, ')
    
    
    
