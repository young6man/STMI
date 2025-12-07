import torch.nn as nn
import einops
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from modeling.backbones.vit_pytorch import trunc_normal_
import textwrap
from .inp import Aggregation_Block
from functools import partial
from .HyperCompute import HyperComputeModule

class STR_CHI(nn.Module):
    def __init__(self, dim=512, learn_tokens=4, threshold=5.0):
        super(STR_CHI, self).__init__()

        self.num_tokens = learn_tokens
        self.threshold = threshold
        self.dim = dim
        C = self.dim
        self.hyper_module = HyperComputeModule(c1=C, c2=C, threshold=self.threshold, hidden_size=C)
        self.mlp_proj = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim),
        )
        #self.hyper_module = HyperComputeModule(c1=C, c2=C, init_threshold=5.0, hidden_size=C)        
        self.inp_tokens_rgb = nn.Parameter(torch.randn(1, self.num_tokens, self.dim))  
        self.inp_tokens_nir = nn.Parameter(torch.randn(1, self.num_tokens, self.dim))
        self.inp_tokens_tir = nn.Parameter(torch.randn(1, self.num_tokens, self.dim))
        self.inp_block = Aggregation_Block(
            dim=self.dim, num_heads=8, mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
        self.inp_block1 = Aggregation_Block(
            dim=self.dim, num_heads=8, mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
        self.inp_block2 = Aggregation_Block(
            dim=self.dim, num_heads=8, mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
        self.inp_block3 = Aggregation_Block(
            dim=self.dim, num_heads=8, mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
        self.inp_self_attn = Aggregation_Block(
            dim=self.dim, num_heads=8, mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

        self.cross_attn = Aggregation_Block(
            dim=self.dim, num_heads=8, mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
        embed_dim = self.dim

        self.proj_q = nn.Linear(embed_dim, embed_dim)
        self.proj_k = nn.Linear(embed_dim, embed_dim)
        self.proj_v = nn.Linear(embed_dim, embed_dim)
        self.proj_out = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(0.1)

        self.proj_q_cross = nn.Linear(embed_dim, embed_dim)
        self.proj_k_cross = nn.Linear(embed_dim, embed_dim)
        self.proj_v_cross = nn.Linear(embed_dim, embed_dim)
        self.cross_proj_out = nn.Linear(embed_dim, embed_dim)
        self.cross_proj_drop = nn.Dropout(0.1)

        self.attn_drop = nn.Dropout(0.1)
        
    def forward(self, x, y, z, boss):
        B, N, C = x.size()

        inp_tokens_rgb = torch.cat((self.inp_tokens_rgb.expand(B, -1, -1), boss[:, 3].unsqueeze(1)), dim=1)
        inp_tokens_nir = torch.cat((self.inp_tokens_nir.expand(B, -1, -1), boss[:, 3].unsqueeze(1)), dim=1)
        inp_tokens_tir = torch.cat((self.inp_tokens_tir.expand(B, -1, -1), boss[:, 3].unsqueeze(1)), dim=1)
        out_rgb = self.inp_block1(inp_tokens_rgb, x)  
        out_nir = self.inp_block2(inp_tokens_nir, y)
        out_tir = self.inp_block3(inp_tokens_tir, z)

        boss = boss[:, :3]

        fused = torch.cat([out_rgb, out_nir, out_tir], dim=1)  

        fused = fused.permute(0, 2, 1).contiguous()
        fused = self.hyper_module(fused) 
        fused = fused.permute(0, 2, 1).contiguous()

        q = self.proj_q_cross(boss)   
        k = self.proj_k_cross(fused)  
        v = self.proj_v_cross(fused)  
        attn = torch.bmm(q, k.transpose(1, 2)) / C**0.5 
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.bmm(attn, v)  
        fusion_out = boss + self.cross_proj_drop(self.cross_proj_out(out))  

        fea = fusion_out.permute(0, 2, 1)  
        vision = torch.flatten(fea[:, :, :3], start_dim=1, end_dim=2)  

        return vision
