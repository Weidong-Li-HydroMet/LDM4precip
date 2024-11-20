from torch.utils.data import DataLoader, Dataset
import os, random, sys, yaml
import torch, torchvision, netCDF4
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from tqdm import tqdm
sys.path.append(r'/root/lwd/SDM_program')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data
opt = read_yaml('/root/lwd/SDM_program/opt.yaml')

sys.path.append(opt['path']['model_save_path'])

# dataset 
from datetime import datetime, timedelta
from dataset import random_sample_fixed
print(opt['dataset']['sart_date'])
start_date = datetime(opt['dataset']['sart_date'], 1, 1)
end_date = datetime(opt['dataset']['end_date'], 12, 31)
current_date = start_date
date_list = []
while current_date <= end_date:
    date_list.append(current_date.strftime("%Y%m%d"))
    current_date += timedelta(days=1)

lats_lr, lons_lr = np.arange(50, 24.5, -0.25), np.arange(235, 293.5, 0.25)
lats_sr, lons_sr = np.arange(50, 24.5, -1/24), np.arange(235, 293.5, 1/24)
region = 'A'
if region == 'A':
    random_lon = 241
    random_lat = 38
elif region == 'B':
    random_lon = 269
    random_lat = 32
elif region == 'C':
    random_lon = 269
    random_lat = 43
lon_lr_idx = np.where(lons_lr == random_lon)[0][0]
lat_lr_idx = np.where(lats_lr == random_lat)[0][0]
'''
I stored the region data separately as nc file. 
If not, the random_sample_fixed function was used to cut out the regional patch from CONUS
'''
condition = {}
prism = {}
for current_date in tqdm(date_list):
    # condition: 'z', 'q', 't', 'u', 'v'
    path_meteo = r'/dynamical_input/meteo_%s.nc' % current_date
    with netCDF4.Dataset(opt['path']['path_data_root'] + path_meteo) as ds:
        v_meteo = ds.variables['data'][:].data
    path_t2m = r'/T2M/t2m_%s.nc' % current_date
    with netCDF4.Dataset(opt['path']['path_data_root'] + path_t2m) as ds:
        t2m_1d = ds.variables['data'][:].data

    # dynamical_input = random_sample_fixed(np.concatenate((v_meteo[:-24*3], t2m_1d), axis=0), 
    #                                       lon_lr_idx, lat_lr_idx, r=24)
    dynamical_input = np.concatenate((v_meteo, t2m_1d), axis=0)
    condition[current_date] = dynamical_input
    del dynamical_input, v_meteo, t2m_1d
    # condition with meteo without t2m
    # path_meteo = r'/dynamical_input/meteo_%s.nc' % current_date
    # v_meteo = netCDF4.Dataset(opt['path']['path_data_root'] + path_meteo).variables['data'][:].data
    # condition[current_date] = v_meteo
    # precip
    path_pr_sr = r'/pr_sr_nc/pr_sr_%s.nc' % current_date
    pr_sr = netCDF4.Dataset(opt['path']['path_data_root'] + path_pr_sr).variables['data'][:].data
    # pr_sr = random_sample_fixed(pr_sr, lon_lr_idx*6, lat_lr_idx*6, r=96)
    pr_sr[pr_sr==-9999] = np.nan
    pr_sr[np.isnan(pr_sr)] = 0
    prism[current_date] = pr_sr
    del pr_sr

# condition['h'] = netCDF4.Dataset(opt['path']['path_data_root'] + r'/usa_dem.nc').variables['data'][:].data
condition['h'] = random_sample_fixed(np.load(opt['path']['path_data_root'] + r'/dem_us_lr.npy'), 
                                          lon_lr_idx, lat_lr_idx, r=24)
# sample 8°*8° patch 10 times on one day from the CONUS
data_list = [date for date in date_list for _ in range(10)]
data_len = len(data_list)
train_data_list = data_list[:int(data_len*0.9)]
val_data_list = data_list[int(data_len*0.9):int(data_len)]

class dicDataset(Dataset):
    def __init__(self, data_list, batch_size=64):
        self.data_list = data_list
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):

        pr_transformed = np.log(prism[self.data_list[index]] + 1)[np.newaxis, :, :]
        
        condition_day_sampled = condition[self.data_list[index]]
        condition_dem_sampled = condition['h'][np.newaxis, :, :]
        condition_sampled = np.concatenate((condition_day_sampled, condition_dem_sampled), axis=0)

        return {'pr':torch.from_numpy(pr_transformed), 
                'condition':torch.from_numpy(condition_sampled)}

train_dataset =dicDataset(train_data_list)
val_dataset = dicDataset(val_data_list)
batch_size = opt['train']['batch_size']
num_workers=8
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                    shuffle=True, pin_memory=True, num_workers=num_workers) 
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                    shuffle=True, pin_memory=True, num_workers=num_workers)

# VAE
from vae_precip import VAE
num_variables = 1
num_hiddens = 256 
num_residual_layers = 2 
num_residual_hiddens = 48  
embedding_dim = 4 
vae = VAE(num_variables, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 embedding_dim).to(device)
vae.load_state_dict(torch.load(opt['path']['model_save_path'] + 'vae_precip.pt')['model_state_dict'])
vae.eval()

# cUNet
from cUNet import ResUnet
cUNet = ResUnet(in_channels=opt['cUNet']['in_channels'], out_channels=opt['cUNet']['out_channels'], 
                n_feat=opt['cUNet']['n_feat'])
cUNet.to(device)
cUNet.load_state_dict(torch.load(opt['path']['model_save_path'] + 'UNet_con.pt')['model_state_dict'])
cUNet.eval()

# ddpm
from diffusion.ddpm import DDPM_cfg
ddpm = DDPM_cfg(opt)
optG = torch.optim.Adam(list(ddpm.parameters()), lr=opt['train']["optimizer"]["lr"])

train_loss_list, val_loss_list = [], []
best_val_loss = float('inf') 
patience = 200  
counter = 0 
best_train_path = opt['path']['model_save_path'] + 'SDM_best_train.pt'

# load ddpm
load_path = opt['path']['model_save_path'] + 'SDM_gen.pt'
load_path_train = opt['path']['model_save_path'] + 'SDM_best_train.pt'
if os.path.exists(load_path):
    print('Loading model from {}'.format(load_path))
    checkpoint = torch.load(load_path)
    ddpm.load_state_dict(checkpoint['model_state_dict'])
    optG.load_state_dict(checkpoint['optimizer_state_dict'])
elif os.path.exists(load_path_train):
    print('Loading model from {}'.format(load_path_train))
    checkpoint = torch.load(load_path_train)
    ddpm.load_state_dict(checkpoint['model_state_dict'])
    optG.load_state_dict(checkpoint['optimizer_state_dict'])
else:
    print('the first time to train')


# train
lrate = opt['train']['optimizer']['lr']
n_epoch = opt['train']['n_epoch']
x_p_c = {}
pbar = tqdm(range(n_epoch))
for epoch in pbar:
    ddpm.train()
    epoch_loss_list = []
    # optG.param_groups[0]['lr'] = lrate*(1-epoch/(n_epoch+1))/2
    # pbar = tqdm(train_dataloader)
    for dictdata in train_dataloader:
        optG.zero_grad()
        dictdata['pr'] = torch.as_tensor(dictdata['pr'])
        dictdata['pr'] = dictdata['pr'].to(device)
        z = vae.encoder(dictdata['pr'])
        z = vae._pre_vq_conv(z)

        x_p_c['pr'] = z
        dictdata['condition'] = torch.as_tensor(dictdata['condition'])
        x_p_c['condition'] = cUNet(dictdata['condition'].to(device))
        l_pix = ddpm(x_p_c) 
        l_pix.backward()
        optG.step()

        epoch_loss_list.append(l_pix.item())
    train_mean_loss = np.mean(epoch_loss_list)
    train_loss_list.append(train_mean_loss)
    
    # val
    epoch_loss_list = []
    ddpm.eval()
    with torch.no_grad():
        # pbar1 = tqdm(val_dataloader)
        for dictdata in val_dataloader:
            dictdata['pr'] = torch.as_tensor(dictdata['pr'])
            dictdata['pr'] = dictdata['pr'].to(device)
            z = vae.encoder(dictdata['pr'])
            z = vae._pre_vq_conv(z)

            x_p_c['pr'] = z
            dictdata['dictdata'] = torch.as_tensor(dictdata['condition'])
            x_p_c['condition'] = cUNet(dictdata['condition'].to(device))
            l_pix = ddpm(x_p_c) # self.netG(self.data) # loss of pixel
            
            # pbar1.set_description(f"val loss: {l_pix.item():.4f}")
            epoch_loss_list.append(l_pix.item())
    val_mean_loss = np.mean(epoch_loss_list)
    val_loss_list.append(val_mean_loss)
    pbar.set_description(f"train loss: {train_mean_loss:.4f}; val loss: {val_mean_loss:.4f}")

    if val_mean_loss < best_val_loss:
        best_val_loss = val_mean_loss
        counter = 0
        state_dict = ddpm.state_dict()
        optimizer_state_dict = optG.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save({
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer_state_dict,}, best_train_path)
        print('best_val_loss:', best_val_loss)
    else:
        counter += 1

    if counter >= patience:
        print("Best val loss is {}, No improvement in val loss for {} epochs.".format(best_val_loss, patience))
    if (epoch+1) % 10:
        state_dict = ddpm.state_dict()
        optimizer_state_dict = optG.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save({
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer_state_dict,}, best_train_path)

gen_path = opt['path']['model_save_path'] + 'SDM_gen.pt'
# gen
state_dict = ddpm.state_dict()
optimizer_state_dict = optG.state_dict()
for key, param in state_dict.items():
    state_dict[key] = param.cpu()
torch.save({
    'model_state_dict': state_dict, 
    'optimizer_state_dict': optimizer_state_dict,}, gen_path)