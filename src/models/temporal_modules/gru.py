from dataclasses import dataclass
from dataclasses import dataclass
import os
import time
from typing import Optional
import numpy as np
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy

import random

@dataclass(frozen=True)
class GRU_Para:
    RNN_DIM: int = 256
    NUM_LAYERS: int = 2
    DROPOUT: float = 0.5
    BIDIR: bool = True

class BiGRUFrameHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, para: GRU_Para):
        super().__init__()
        self.para = para

        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=para.RNN_DIM,
            num_layers=para.NUM_LAYERS,
            batch_first=True,
            bidirectional=para.BIDIR,
            dropout=para.DROPOUT if para.NUM_LAYERS > 1 else 0.0,
        )

        self.drop = nn.Dropout(para.DROPOUT)

        out_dim = para.RNN_DIM * (2 if para.BIDIR else 1)
        self.head = nn.Linear(out_dim, n_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = z
        y, _ = self.gru(x)
        y = self.drop(y)
        logits = self.head(y)
        return logits.permute(0, 2, 1)
    
