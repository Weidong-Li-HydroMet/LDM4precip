import math, torch
import torch.nn.functional as F
from inspect import isfunction
import torch.nn as nn
from diffusion.block import PositionalEncoding, FeatureWiseAffine, Swish, Upsample, Downsample, exists, default
from inspect import isfunction
from torch import nn

# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(), 
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(), 
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )
    def forward(self, x):
        return self.block(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level) 
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)

class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2) 

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input
# ResNet block with attention
class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if (self.with_attn):
            x = self.attn(x)
        return x

class UNet(nn.Module):
    def __init__(
        self, 
        in_channel=2, 
        out_channel=1,
        norm_groups=16,
        inner_channel=64, 
        channel_mults=[1,2,4,8],
        attn_res=[128],
        res_blocks=2,
        dropout=0.1, 
        with_noise_level_emb=True,
        image_size=48, 
        noise_level_channel = 64 
    ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel), 
                nn.Linear(inner_channel, inner_channel * 4), 
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel) 
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        # down 
        inner_channel = default(inner_channel, in_channel)
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel] 
        now_res = image_size
        self.start_conv = nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1) 
        downs = []
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1) 
            use_attn = now_res in attn_res 
            channel_mult = inner_channel * channel_mults[ind] 
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, 
                    norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)
        
        # mid
        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False),
        ])
        
        # up
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = now_res in attn_res
            channel_mult = inner_channel * channel_mults[ind] 
            for _ in range(0, res_blocks):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)
        
        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, time):
        t = self.noise_level_mlp(time) if exists(self.noise_level_mlp) else None
        x = self.start_conv(x)
        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
                feats.append(x)
            else:
                x = layer(x)
            
        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
        
        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)






