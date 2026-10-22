"""model.py - KwsNet, the network the firmware runs (Ai/kws_engine.c).

Only these sizes may be changed (the C code reads them from the model):
T (frames per window, <= 101), c1 <= 32, c2 <= 64, c3 <= 96, hid <= 128,
ncls <= 8, spk_dim <= 64. The layer types / order must stay as they are.

    x: raw log-mel [N, T, 40] -> (x - in_mean) / in_std
    conv1 3x3 (1->c1, pad 1) + BN + ReLU + MaxPool 2
    conv2 3x3 (c1->c2, pad 1) + BN + ReLU + MaxPool 2
    conv3 3x3 (c2->c3, pad 1) + BN + ReLU + mean over time and frequency -> g
    keyword: Linear(c3, hid) + ReLU + Linear(hid, ncls)       -> logits
    speaker: Linear(c3, spk_dim), divided by its length        -> emb (optional)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

BANDS = 40


class KwsNet(nn.Module):
    def __init__(self, ncls, T=61, c1=16, c2=32, c3=48, hid=64, spk_dim=32):
        super().__init__()
        self.cfg = dict(ncls=ncls, T=T, c1=c1, c2=c2, c3=c3, hid=hid, spk_dim=spk_dim)
        self.register_buffer("in_mean", torch.zeros(BANDS))
        self.register_buffer("in_std", torch.ones(BANDS))
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(c3)
        self.fc1 = nn.Linear(c3, hid)
        self.fc2 = nn.Linear(hid, ncls)
        self.spk = nn.Linear(c3, spk_dim) if spk_dim > 0 else None

    def trunk(self, x):
        x = (x - self.in_mean) / self.in_std
        x = x.unsqueeze(1)                                  # [N, 1, T, 40]
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.relu(self.bn3(self.conv3(x)))
        return x.mean(dim=(2, 3))                           # [N, c3]

    def forward(self, x):
        g = self.trunk(x)
        logits = self.fc2(F.relu(self.fc1(g)))
        emb = F.normalize(self.spk(g), dim=1) if self.spk is not None else None
        return logits, emb
