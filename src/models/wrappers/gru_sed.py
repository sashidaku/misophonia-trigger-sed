from dataclasses import dataclass
from dataclasses import dataclass
import os
import time
from typing import Optional
import numpy as np

from src.models.encorders.common_audio_encorder import EncodeSpec
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy

class SEDModelEncoderRNN(nn.Module):
    def __init__(self, *, encoder, rnn_head: nn.Module, use_amp: bool = False):
        super().__init__()
        self.encoder = encoder
        self.rnn_head = rnn_head
        self.use_amp = use_amp

        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        with torch.no_grad():
            z = self.encoder(
                audio,
                spec=EncodeSpec(out="cnn_seq", detach=True, use_amp=self.use_amp),
                esn=None,
            )

        logits = self.rnn_head(z)
        return logits