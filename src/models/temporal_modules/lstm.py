from dataclasses import dataclass
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.nn as nn


@dataclass(frozen=True)
class LSTM_Para:
    RNN_DIM: int = 256
    NUM_LAYERS: int = 2
    DROPOUT: float = 0.5
    BIDIR: bool = True

class BiLSTMFrameHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, para: LSTM_Para):
        super().__init__()
        self.para = para

        self.proj = nn.Linear(in_dim, para.RNN_DIM)

        self.lstm = nn.LSTM(
            input_size=para.RNN_DIM,
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
        x = self.proj(z)
        y, _ = self.lstm(x)
        y = self.drop(y)
        logits = self.head(y)
        return logits.permute(0, 2, 1)