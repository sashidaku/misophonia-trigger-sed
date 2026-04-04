import os
import numpy as np

from src.evaluation.pipeline import downsample_labels_to_T, tune_median_and_threshold
from src.models.encorders.common_audio_encorder import EncodeSpec
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy


@torch.no_grad()
def eval_psds1_on_val_cached_feats(
    *,
    CLASS_NAMES,
    rnn_head: nn.Module,
    feats_cpu: torch.Tensor,     # (N,T,D) CPU
    labels_cpu: torch.Tensor,    # (N,C,T_lab) CPU
    meta: list,
    device: torch.device,
    batch_size: int = 64,
):
    rnn_head.eval()
    N, T, D = feats_cpu.shape
    C = labels_cpu.shape[1]

    ds = TensorDataset(feats_cpu, labels_cpu)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    pred_all = np.empty((N, C, T), dtype=np.float32)
    Y_all    = np.empty((N, C, T), dtype=np.float32)

    offset = 0
    for X_b_cpu, Y_b_cpu in loader:
        B = X_b_cpu.size(0)
        X_b = X_b_cpu.to(device).float()
        Y_b = Y_b_cpu.to(device).float()

        logits = rnn_head(X_b)  # (B,C,T)
        probs  = torch.sigmoid(logits)

        Y_ds = downsample_labels_to_T(Y_b, T)  # (B,C,T)

        pred_all[offset:offset+B] = probs.detach().cpu().numpy()
        Y_all[offset:offset+B]    = Y_ds.detach().cpu().numpy()
        offset += B

    pred_NTc = np.transpose(pred_all, (0, 2, 1))
    Y_NTc    = np.transpose(Y_all,    (0, 2, 1))

    best_win, best_psds1 = tune_median_and_threshold(
        pred_NTc, Y_NTc, meta, class_names=CLASS_NAMES, WIN_SIZE=[1],
    )
    return float(best_psds1), int(best_win)