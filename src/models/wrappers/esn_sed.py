import os

from src.models.encorders.common_audio_encorder import EncodeSpec
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.nn as nn
from typing import Literal




class SEDModelEncoderESNRidge(nn.Module):
    def __init__(self, *, encoder, esn: nn.Module, W: torch.Tensor, b: torch.Tensor, use_amp: bool = False, esn_input_from="cnn"):
        super().__init__()
        self.encoder = encoder
        self.esn = esn
        self.use_amp = use_amp
        self.esn_input_from = esn_input_from

        self.register_buffer("W", W)
        self.register_buffer("b", b)

        self.encoder.eval()
        self.esn.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for p in self.esn.parameters():
            p.requires_grad_(False)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        with torch.no_grad():
            h = self.encoder(
                audio,
                spec=EncodeSpec(out="esn_seq", detach=True, use_amp=self.use_amp, esn_input_from=self.esn_input_from),
                esn=self.esn,
            )
            B, T, Hdim = h.shape
            logits = h.reshape(-1, Hdim) @ self.W + self.b
            logits = logits.reshape(B, T, -1).permute(0, 2, 1)
        return logits

class SEDModelEncoderESNReadout(nn.Module):
    """
    推論用:
      audio -> encoder(cnn_seq) -> esn -> readout -> logits (B,C,T)
    """
    def __init__(self, *, encoder, esn: nn.Module, readout: nn.Module, use_amp: bool = False, esn_input_from: Literal["cnn","mel"]="cnn"):
        super().__init__()
        self.encoder = encoder
        self.esn = esn
        self.readout = readout
        self.use_amp = use_amp

        self.encoder.eval()
        self.esn.eval()
        self.readout.eval()
        self.esn_input_from = esn_input_from
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for p in self.esn.parameters():
            p.requires_grad_(False)
        for p in self.readout.parameters():
            p.requires_grad_(False)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        with torch.no_grad():
            h = self.encoder(
                audio,
                spec=EncodeSpec(out="esn_seq", detach=True, use_amp=self.use_amp, esn_input_from=self.esn_input_from),
                esn=self.esn,
            )

            B, T, Hdim = h.shape
            logits = self.readout(h.reshape(-1, Hdim))
            logits = logits.reshape(B, T, -1).permute(0, 2, 1)
        return logits