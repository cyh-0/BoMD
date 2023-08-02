from typing import ForwardRef
import torch
from models.densenet import densenet121
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from utils.helper_functions import sim_score

class CosineLoss(nn.Module):
    def forward(self, t_emb, v_emb):
        a_norm = v_emb / v_emb.norm(dim=1)[:, None]
        b_norm = t_emb / t_emb.norm(dim=1)[:, None]
        loss = 1 - torch.mean(torch.diagonal(torch.mm(a_norm, b_norm.t()), 0))

        return loss
class model_mid(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = densenet121(pretrained=True)
        self.backbone.classifier = nn.Linear(1024, args.num_classes)
        self.backbone.classifier = nn.Identity()
        self.fm_ln = nn.Linear(1024, args.embed_len * args.num_pd)

    def forward(self, x):
        # fea -> [B, 1024, 16, 16]
        x = self.backbone(x)
        visual_feats = self.fm_ln(x)

        return visual_feats
