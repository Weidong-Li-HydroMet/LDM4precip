import math, torch
import torch.nn.functional as F
from inspect import isfunction
import torch.nn as nn
import numpy as np
import functools
from functools import partial
from diffusion.srUNet import UNet
from diffusion.block import make_beta_schedule
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_device(x, device):
    if isinstance(x, dict):
        for key, item in x.items():
            if item is not None:
                x[key] = item.to(device)
    elif isinstance(x, list):
        for item in x:
            if item is not None:
                item = item.to(device)
    else:
        x = x.to(device)
    return x


class DDPM_cfg(nn.Module):
    def __init__(self, opt):
        super(DDPM_cfg, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # denoising UNet
        self.denoise_fn = set_device(UNet(
            in_channel=opt['unet']['in_channel'], 
            out_channel=opt['unet']['out_channel'], 
            norm_groups=opt['unet']['norm_groups'], 
            inner_channel=opt['unet']['inner_channel'],
            channel_mults=opt['unet']['channel_multiplier'],
            attn_res=opt['unet']['attn_res'], 
            res_blocks=opt['unet']['res_blocks'],
            dropout=opt['unet']['dropout'], 
            image_size=opt['diffusion']['image_size'],
            noise_level_channel = opt['unet']['inner_channel'] 
        ), self.device)
        # ddpm
        self.channels = opt['diffusion']['channels']
        self.image_size = opt['diffusion']['image_size']
        self.is_guide = opt['diffusion']['is_guide']
        self.guide_w = opt['diffusion']['guide_w']
        self.loss_type = 'l2'
        self.set_loss()
        self.set_new_noise_schedule(opt['beta_schedule']['train'])

    def set_loss(self):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss().to(self.device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss().to(self.device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device) 
        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas 
        alphas = 1. - betas 
        alphas_cumprod = np.cumprod(alphas, axis=0) 
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1]) 
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod)) 

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20)))) 
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    ######## guide
    def predict_start_from_noise(self, x, t, noise): # denoising
        if self.is_guide:
            length = int(x.shape[0]/2)
            x = x[:length]
            noise1 = noise[:length]
            noise2 = noise[length:]
            noise_w = (1+self.guide_w)*noise1 - self.guide_w*noise2
        else:
            noise_w = noise
        return self.sqrt_recip_alphas_cumprod[t] * x - self.sqrt_recipm1_alphas_cumprod[t] * noise_w
    
    def q_posterior(self, x_recon, x, t):  # denoising
        if self.is_guide:
            length = int(x.shape[0]/2)
            x = x[:length]
            
        posterior_mean = self.posterior_mean_coef1[t] * x_recon + self.posterior_mean_coef2[t] * x
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped
    
    ######## guide
    def p_mean_variance(self, x, t, x_c): # denoising

        batch_size = x.shape[0] 
        x = x.repeat(2,1,1,1)
        x_c = x_c.repeat(2,1,1,1)

        context_mask = torch.zeros_like(x_c).to(device)
        context_mask[:batch_size] = 1.
        x_c = x_c * context_mask

        noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t+1]])
        noise_level = noise_level.repeat(batch_size*2, 1).to(x.device)
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(torch.cat([x_c, x], dim=1), noise_level))
        model_mean, posterior_log_variance = self.q_posterior(x_recon=x_recon, x=x, t=t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, x_c=None): # denoising

        model_mean, model_log_variance = self.p_mean_variance(x=x, t=t, x_c=x_c)
        shape = (self.batch_size, *x.shape[1:]) if self.is_guide else x.shape
        z = torch.randn(shape).to(self.device) if t > 0 else torch.zeros(shape).to(self.device)
        
        return model_mean + z * (0.5 * model_log_variance).exp()

    @torch.no_grad() 
    def super_resolution(self, data, is_guide=False, guide_w=0): # denoising
        self.is_guide = is_guide
        self.guide_w = guide_w
        self.batch_size = data['pr'].shape[0]
        x_in = set_device(data, self.device)
        device = self.betas.device
        x_c = x_in['condition']
        x = x_in['pr']
        for i in reversed(range(0, self.num_timesteps)):
            x = self.p_sample(x, i, x_c=x_c)
        return x

    def forward(self, data, *args, **kwargs): # noising
        x_in = set_device(data, self.device)
        x_precip = x_in['pr'] if isinstance(x_in, dict) else x_in
        batch_size = x_precip.shape[0]
        t = np.random.randint(1, self.num_timesteps + 1) # time step
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=batch_size
            )
        ).to(x_precip.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(batch_size, -1) # shape to (b, ...)
        noise = torch.randn_like(x_precip)
        continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1)
        x_noisy = continuous_sqrt_alpha_cumprod * x_precip + (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        if self.is_guide: # if mask is true, the 'condition' will be masked
            # dropout context with some probability p = 0.1
            context_mask = torch.bernoulli(torch.zeros_like(x_in['condition'])+ 0.1).to(device)
            context_mask = (1-context_mask)
            x_in['condition'] = x_in['condition'] * context_mask

        x_recon = self.denoise_fn(torch.cat([x_in['condition'], x_noisy], dim=1), 
                                continuous_sqrt_alpha_cumprod) # UNet to predict the noise in each step
        loss = self.loss_func(noise, x_recon)

        return loss

