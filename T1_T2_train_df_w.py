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
import torch
from monai.networks.nets import unet
from torch.utils.data import DataLoader, Dataset
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet
from generative.inferers import DiffusionInferer
from generative.networks.schedulers.ddpm import DDPMScheduler
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import torch.nn.functional as F
import sys


with tqdm(total=100, dynamic_ncols=False) as pbar:
    for i in range(100):
        # Do something
        pbar.update(1)

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
        
        return T1, T2, self.files[idx,1], self.files[idx,0], int(self.dset)


model = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=2,
    out_channels=1,
    num_channels=(64, 64, 64),
    attention_levels=(False, False, True),
    num_res_blocks=1,
    num_head_channels=64,
    with_conditioning=False,
)
model.to(device)



scheduler = DDPMScheduler(num_train_timesteps=1000)
optimizer = torch.optim.Adam(params=model.parameters(), lr=2.5e-5)
inferer = DiffusionInferer(scheduler)


n_epochs = 50 #4000
val_interval = 2 #50
epoch_loss_list = []
val_epoch_loss_list = []
pre_train =False
pre_epoch = 0

if pre_train:
    pre_epoch = 154
    model.load_state_dict(torch.load( f'/scratch1/akrami/Projects/T1_T2/models/T1_T2{pre_epoch}_b20.pt'))
    print('loaded the pre train model')




scaler = GradScaler()


for epoch in range(n_epochs):
    model.train()
    epoch_loss = 0
   # print('pass1')
    for step, data in enumerate(tqdm(train_loader,file=sys.stdout,position=0, leave=True)):
        
        
        
        T1, T2, s, _, d = data
        images, seg = T1.to(device), T2.to(device)
        

        ratio = ratios[d[0],int(s[0]=='F')]
        mse = loss(output, T2) * ratio
        
        
        
        #images = data["image"].to(device)
        #seg = data["label"].to(device)  # this is the ground truth segmentation
        #print('pass2')
        optimizer.zero_grad(set_to_none=True)
        timesteps = torch.randint(0, 1000, (len(images),)).to(device)  # pick a random time step t

        with autocast(enabled=True):
            # Generate random noise
            noise = torch.randn_like(seg).to(device)
            noisy_seg = scheduler.add_noise(
                original_samples=seg, noise=noise, timesteps=timesteps
            )  # we only add noise to the segmentation mask
            combined = torch.cat(
                (images, noisy_seg), dim=1
            )  # we concatenate the brain MR image with the noisy segmenatation mask, to condition the generation process
            prediction = model(x=combined, timesteps=timesteps)
            # Get model prediction
            loss = F.mse_loss(prediction.float(), noise.float())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
          
        print('  * train  ' +
          f'Loss: {epoch_loss/len(train_loader):.7f}, ')

    epoch_loss_list.append(epoch_loss / (step + 1))
    if (epoch) % val_interval == 0:
        model.eval()
        val_epoch_loss = 0
        for step, data in enumerate(tqdm(val_loader,file=sys.stdout,position=0, leave=True)):
            torch.save(model.state_dict(), f'/scratch1/akrami/Projects/T1_T2/models/T1_T2{epoch+pre_epoch}_b20.pt')
            T1, T2, s, _, d = data
            images, seg = T1.to(device), T2.to(device)
            timesteps = torch.randint(0, 1000, (len(images),)).to(device)
            with torch.no_grad():
                with autocast(enabled=True):
                    noise = torch.randn_like(seg).to(device)
                    noisy_seg = scheduler.add_noise(original_samples=seg, noise=noise, timesteps=timesteps)
                    combined = torch.cat((images, noisy_seg), dim=1)
                    prediction = model(x=combined, timesteps=timesteps)
                    val_loss = F.mse_loss(prediction.float(), noise.float())
            val_epoch_loss += val_loss.item()
        #print("Epoch", epoch, "Validation loss", val_epoch_loss / (step + 1))
        print('  * val  ' +
          f'Loss: {val_epoch_loss/len(val_loader):.7f}, ')
        val_epoch_loss_list.append(val_epoch_loss / (step + 1))



torch.save(model.state_dict(), f'/scratch1/akrami/Projects/T1_T2/models/T1_T2{epoch+pre_epoch}_b20.pt')
total_time = time.time() - total_start
print(f"train diffusion completed, total time: {total_time}.")
plt.style.use("seaborn-bright")
plt.title("Learning Curves Diffusion Model", fontsize=20)
plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_loss_list, color="C0", linewidth=2.0, label="Train")
plt.plot(
    np.linspace(val_interval, n_epochs, int(n_epochs / val_interval)),
    val_epoch_loss_list,
    color="C1",
    linewidth=2.0,
    label="Validation",
)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(prop={"size": 14})
plt.show()
plt.savefig(f'./"result_translation/loss.png')