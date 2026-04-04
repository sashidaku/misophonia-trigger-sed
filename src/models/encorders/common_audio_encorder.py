from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncodeSpec:
    out: Literal["mel", "cnn_seq", "cnn_pool", "cnn_logits", "esn_seq", "esn_pool"] = "cnn_seq"
    pool: Literal["mean", "max"] = "mean"
    detach: bool = True
    use_amp: bool = False
    std_input: Optional[bool] = None
    logits_key: Optional[str] = None
    esn_input_from: Literal["cnn", "mel"] = "cnn"


class CommonAudioEncoder(nn.Module):

    def __init__(
        self,
        *,
        mel_extractor: nn.Module,
        fmn: nn.Module,
        device: torch.device,
        default_std_input: bool = False,
        std_eps: float = 1e-8,
        expected_T: Optional[int] = None,
    ):
        super().__init__()
        self.mel_extractor = mel_extractor
        self.fmn = fmn
        self.device = device

        self.default_std_input = default_std_input
        self.std_eps = std_eps
        self.expected_T = expected_T
        
        self.register_buffer("cnn_mean", None, persistent=False)
        self.register_buffer("cnn_std",  None, persistent=False)
        self.register_buffer("mel_mean", None, persistent=False)
        self.register_buffer("mel_std",  None, persistent=False)

    def _ensure_BT(self, wave: torch.Tensor) -> torch.Tensor:
        if wave.dim() == 1:
            wave = wave.unsqueeze(0)
        elif wave.dim() == 3:
            wave = wave.squeeze(1)
        elif wave.dim() != 2:
            raise ValueError(f"Unexpected wave shape: {tuple(wave.shape)}")
        return wave

    def _fmn_to_BTD(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.dim() != 3:
            raise ValueError(f"Unexpected fmn output shape: {tuple(feats.shape)}")

        expT = self.expected_T

        if expT is not None:
            if feats.shape[1] == expT:
                return feats
            if feats.shape[2] == expT:
                return feats.transpose(1, 2)

            raise ValueError(
                f"Cannot align fmn output to (B,T,D): got {tuple(feats.shape)}, expected_T={expT}"
            )

        if feats.shape[1] > feats.shape[2]:
            feats = feats.transpose(1, 2)
        return feats
    
    def _mel_to_BTD(self, mels: torch.Tensor) -> torch.Tensor:
        if mels.dim() == 4:
            m = mels.squeeze(1)
        elif mels.dim() == 3:
            m = mels
        else:
            raise ValueError(f"Unexpected mel shape: {tuple(mels.shape)}")

        if m.dim() != 3:
            raise ValueError(f"Unexpected mel shape after squeeze: {tuple(m.shape)}")

        expT = self.expected_T

        # expT があれば、それに一致する次元を time とみなす
        if expT is not None:
            if m.shape[1] == expT and m.shape[2] != expT:
                mel_seq = m
            elif m.shape[2] == expT and m.shape[1] != expT:
                mel_seq = m.transpose(1, 2)
            elif m.shape[1] == expT and m.shape[2] == expT:
                raise ValueError(f"Ambiguous mel shape: {tuple(m.shape)} expected_T={expT}")
            else:
                if m.shape[1] < m.shape[2]:
                    m_bft = m
                else:
                    m_bft = m.transpose(1, 2)
                m_bft = F.interpolate(m_bft, size=expT, mode="linear", align_corners=False)
                mel_seq = m_bft.transpose(1, 2)  # (B,T,F)
        else:
            mel_seq = m.transpose(1, 2) if m.shape[1] < m.shape[2] else m

        return mel_seq
    
    def _to_BCT_logits(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Unexpected logits shape: {tuple(x.shape)}")

        expT = self.expected_T
        if expT is not None:
            if x.shape[2] == expT:
                return x
            if x.shape[1] == expT:
                return x.transpose(1, 2)

        if 200 <= x.shape[1] <= 400 and x.shape[2] > x.shape[1]:
            return x.transpose(1, 2)
        return x 

    def _force_T_BTD(self, x: torch.Tensor) -> torch.Tensor:
        expT = self.expected_T
        if expT is None or x.shape[1] == expT:
            return x

        x_bdt = x.transpose(1, 2)
        x_bdt = F.interpolate(x_bdt, size=expT, mode="linear", align_corners=False)
        return x_bdt.transpose(1, 2)
    
    @torch.no_grad()
    def set_feature_norm(self, mean: torch.Tensor, std: torch.Tensor, *, source: Literal["cnn", "mel"] = "cnn"):
        if mean is None or std is None:
            if source == "cnn":
                self.cnn_mean = self.cnn_std = None
            else:
                self.mel_mean = self.mel_std = None
            return

        mean = mean.detach()
        std  = std.detach()
        if mean.dim() == 1: mean = mean.view(1, 1, -1)
        if std.dim() == 1:  std  = std.view(1, 1, -1)

        mean = mean.to(self.device)
        std  = std.to(self.device)

        if source == "cnn":
            self.cnn_mean, self.cnn_std = mean, std
        else:
            self.mel_mean, self.mel_std = mean, std
        
    def _maybe_std_input(self, x: torch.Tensor, *, source: Literal["cnn", "mel"]) -> torch.Tensor:
        if source == "cnn":
            mean, std = self.cnn_mean, self.cnn_std
        else:
            mean, std = self.mel_mean, self.mel_std

        if (mean is not None) and (std is not None):
            return (x - mean) / (std + self.std_eps)

        mean_b = x.mean(dim=(0, 1), keepdim=True)
        std_b  = x.std(dim=(0, 1), keepdim=True) + self.std_eps
        return (x - mean_b) / std_b

    def forward(
        self,
        wave: torch.Tensor,
        spec: EncodeSpec,
        *,
        esn: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        wave = self._ensure_BT(wave).to(self.device)

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if (spec.use_amp and wave.is_cuda) else nullcontext()
        )

        with amp_ctx, torch.set_grad_enabled(not spec.detach):
            mels = self.mel_extractor(wave)
            feat_seq = None
            if spec.out == "mel":
                return mels.detach() if spec.detach else mels

            if spec.out == "cnn_logits":
                out = self.fmn(mels)

                out = extract_logits_any(out, key=spec.logits_key)

                logits_BCT = self._to_BCT_logits(out)
                return logits_BCT.detach() if spec.detach else logits_BCT
            need_cnn_seq = (
                spec.out in ["cnn_seq", "cnn_pool"]
                or (spec.out in ["esn_seq", "esn_pool"] and spec.esn_input_from == "cnn")
            )
            if need_cnn_seq:
                feats_raw = self.fmn.fmn(mels)
                feat_seq = self._fmn_to_BTD(feats_raw)

                if spec.out == "cnn_seq":
                    return feat_seq.detach() if spec.detach else feat_seq

                if spec.out == "cnn_pool":
                    pooled = feat_seq.mean(1) if spec.pool == "mean" else feat_seq.max(1).values
                    return pooled.detach() if spec.detach else pooled

            if spec.out in ["esn_seq", "esn_pool"]:
                if esn is None:
                    raise ValueError("spec.out is esn_* but esn is None")

                if spec.esn_input_from == "mel":
                    x_src = self._mel_to_BTD(mels)
                    x_src = self._force_T_BTD(x_src)
                    source = "mel"
                else:
                    if feat_seq is None:
                        feats_raw = self.fmn.fmn(mels)
                        feat_seq = self._fmn_to_BTD(feats_raw)
                    x_src = self._force_T_BTD(feat_seq)
                    source = "cnn"
                    

                std_input = self.default_std_input if spec.std_input is None else bool(spec.std_input)
                x_in = self._maybe_std_input(x_src, source=source) if std_input else x_src

                h_seq = esn(x_in)

                if spec.out == "esn_seq":
                    return h_seq.detach() if spec.detach else h_seq

                pooled = h_seq.mean(1) if spec.pool == "mean" else h_seq.max(1).values
                return pooled.detach() if spec.detach else pooled

@torch.no_grad()
def compute_cnnseq_mean_std(encoder, train_loader, device, use_amp=False):
    sum_ = None
    sumsq = None
    n = 0

    for audio, labels, *rest in train_loader:
        audio = audio.to(device)

        x = encoder(
            audio,
            spec=EncodeSpec(out="cnn_seq", detach=True, use_amp=use_amp, std_input=False),
            esn=None,
        )  # (B,T,D)

        x = x.float().cpu()
        if sum_ is None:
            D = x.shape[-1]
            sum_ = torch.zeros(D, dtype=torch.float64)
            sumsq = torch.zeros(D, dtype=torch.float64)

        sum_ += x.sum(dim=(0,1)).double()
        sumsq += (x*x).sum(dim=(0,1)).double()
        n += x.shape[0] * x.shape[1]

    mean = (sum_ / n).float()
    var  = (sumsq / n).float() - mean*mean
    std  = torch.sqrt(torch.clamp(var, min=1e-8))
    return mean, std

def extract_logits_any(out, *, key: str | None = None) -> torch.Tensor:
    if torch.is_tensor(out):
        return out

    if hasattr(out, "to_tuple") and callable(getattr(out, "to_tuple")):
        out = out.to_tuple()

    if isinstance(out, dict):
        if key is not None:
            v = out.get(key, None)
            if torch.is_tensor(v):
                return v
            raise ValueError(f"logits_key='{key}' not found or not a Tensor. keys={list(out.keys())}")

        for k in ["strong_logits", "framewise_logits", "frame_logits", "logits", "strong", "pred", "y", "output"]:
            v = out.get(k, None)
            if torch.is_tensor(v):
                return v

        c3 = [v for v in out.values() if torch.is_tensor(v) and v.dim() == 3]
        if len(c3) >= 1:
            return c3[0]

        for v in out.values():
            if torch.is_tensor(v):
                return v

        raise ValueError(f"dict output contains no Tensor. keys={list(out.keys())}")

    if isinstance(out, (tuple, list)):
        c3 = [v for v in out if torch.is_tensor(v) and v.dim() == 3]
        if len(c3) >= 1:
            return c3[0]
        c = [v for v in out if torch.is_tensor(v)]
        if len(c) >= 1:
            return c[0]
        raise ValueError("tuple/list output contains no Tensor.")

    raise TypeError(f"Cannot extract logits from type={type(out)}")