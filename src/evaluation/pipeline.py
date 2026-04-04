import numpy as np
import scipy
import torch

from src.evaluation.psds import psds1_from_arrays
from src.evaluation.report import get_psds_meta
from config_loader import data_cfg


def downsample_labels_to_T(labels: torch.Tensor, T_res: int) -> torch.Tensor:
    B, C, T_lab = labels.shape
    assert T_lab % T_res == 0, f"Cannot downsample: T_lab={T_lab}, T_res={T_res}"
    factor = T_lab // T_res

    labels = labels.view(B, C, T_res, factor)
    labels = labels.max(dim=-1).values 
    return labels

def collect_preds_labels(
    dataloader,
    sed_model,
    device,
    frames_1s: int,
):
    sed_model.eval()
    all_probs = []
    all_labels = []
    meta = []

    with torch.no_grad():
        for audio, labels, fnames, ts in dataloader:
            audio = audio.to(device)
            labels = labels.to(device).float()

            logits = sed_model(audio)
            probs  = torch.sigmoid(logits)

            B, C, T_res = probs.shape

            labels_ds = downsample_labels_to_T(labels, T_res)
            all_probs.append(probs.cpu())
            all_labels.append(labels_ds.cpu())

            ts = ts.cpu()
            for i in range(B):
                meta.append({
                    "filename": fnames[i],
                    "t0": float(ts[i, 0].item()) if ts.ndim == 2 else 0.0,
                    "duration": 10.0,
                })

    pred_all = torch.cat(all_probs, dim=0).numpy()
    Y_all    = torch.cat(all_labels, dim=0).numpy()

    return pred_all, Y_all, meta

def tune_median_and_threshold(
    pred_va: np.ndarray,
    Y_va:   np.ndarray,
    va_meta,
    class_names,
    WIN_SIZE=[1, 3, 5, 7, 9]
):
    from collections import Counter
    gt_df, filenames, durations = get_psds_meta(Y_va, va_meta, class_names)
    print("     [DEBUG] pred_va:", pred_va.shape, "Y_va:", Y_va.shape, "len(meta):", len(va_meta), "len(filenames):", len(filenames), "len(durations):", len(durations))
    names = [m["filename"] for m in va_meta]
    dup = [n for n, c in Counter(names).items() if c > 1]
    if len(dup) != 0:
        print("     num duplicated filenames:", len(dup))
        print("     examples:", dup[:20])
    best_psds1 = -1.0
    best_median_win = 1

    for win_size in WIN_SIZE:
        print(f"    Trying median_win = {win_size}")

        if win_size > 1:
            pred_va_filtered = scipy.ndimage.median_filter(
                pred_va,
                size=(1, win_size, 1),
            )
        else:
            pred_va_filtered = pred_va

        psds1 = psds1_from_arrays(
            pred_scores=pred_va_filtered,
            filenames=filenames,
            class_names=class_names,
            frames_per_sec=data_cfg.frames_1s,
            groundtruth_df=gt_df,
            durations_sec=durations,
            scenario="psds1",
        )
        current_psds1 = psds1
        print(f"        win={win_size}, psds1={current_psds1:.4f}")

        if current_psds1 > best_psds1:
            best_psds1 = current_psds1
            best_median_win = win_size

    if len(WIN_SIZE) != 1:
        print(f"    [Best Tune] Median_win={best_median_win}, PSDS1={best_psds1:.4f}")
    return best_median_win, best_psds1