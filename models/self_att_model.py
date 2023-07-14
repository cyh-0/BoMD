from typing import ForwardRef
import torch
from models.densenet import densenet121
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from utils.helper_functions import sim_score
import torchvision
from timm.models.layers import Mlp, DropPath, trunc_normal_, lecun_normal_
import numpy as np


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(self, img_size=(16, 30), dim_in=1280, embed_dim=768, norm_layer=None):
        super().__init__()
        H, W = img_size[0], img_size[1]
        self.img_size = img_size

        # self.proj = nn.Conv2d(1280, 512, kernel_size=3, stride=patch_size)
        self.proj = nn.Linear(dim_in, embed_dim)
        self.num_patches = H * W
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.proj(x)
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SELF_ATTENTION(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        depth=2,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        dim_in=1204,
    ):
        super().__init__()
        self.vision_embed = PatchEmbed(
            img_size=(16, 16), dim_in=dim_in, embed_dim=embed_dim
        )
        self.label_embed = Mlp(in_features=1024, out_features=1024)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.vision_embed.num_patches, embed_dim)
        )
        # self.pos_embed_2 = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward_features(self, x, l_emb):
        x = self.vision_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        l_emb = self.label_embed(l_emb)
        x_in = torch.cat((x, l_emb.unsqueeze(0).repeat(x.shape[0], 1, 1)), dim=1)
        out = self.blocks(x_in)
        out = self.norm(out)
        return out

    def forward(self, x, l_emb):
        x = self.forward_features(x, l_emb)
        return x


import torch.nn.functional as F
from models.seesaw import NormalizedLinear as seesaw_ln


class CXR_SELFATT(nn.Module):
    def __init__(self, args, wordvec_array):
        super().__init__()
        self.args = args
        self.vision_backbone = torchvision.models.densenet121(pretrained=True)
        # self.vision_backbone.classifier = nn.Linear(1024, args.num_classes)
        self.att_module = SELF_ATTENTION(dim_in=1024, embed_dim=1024, depth=1)

        self.lb_fc = seesaw_ln(14, 1)
        self.cls = seesaw_ln(2048, 14)

        self.word_vec = wordvec_array.squeeze().transpose(1, 0).cuda()
        # self.word_vec = torch.tensor(np.load("./embeddings/nih_openi_biober_14.npy")).cuda()

    def forward(self, x):
        # fea -> [B, 1024, 16, 16]
        fm = self.vision_backbone.features(x)
        x_att = self.att_module(fm, self.word_vec)

        z = F.relu(fm, inplace=True)
        z = F.adaptive_avg_pool2d(z, (1, 1))
        z = torch.flatten(z, 1)

        lb_att = x_att[:, -14:, :]

        lb_att = rearrange(lb_att, "b n c -> b c n")
        lb_fea = self.lb_fc(lb_att).squeeze()

        fea_final = torch.cat((z, lb_fea), dim=1)

        logits = self.cls(fea_final)

        return logits
