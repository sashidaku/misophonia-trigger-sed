import numpy as np
import scipy
import torch

from src.evaluation.metrics import find_best_threshold_per_class_event
from src.evaluation.pipeline import collect_preds_labels, tune_median_and_threshold
from src.evaluation.report import get_metrics

from config_loader import data_cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pred_esn(val_loader, test_loader, sed_model, CLASS_NAMES, AS_MAP=None):

    pred_va, Y_va, va_meta = collect_preds_labels(
        dataloader=val_loader,
        sed_model=sed_model,
        device=device,
        frames_1s=data_cfg.frames_1s,
    )

    print("     pred_va shape:", pred_va.shape, "Y_va shape   :", Y_va.shape)
    
    if AS_MAP != None:
        pred_order = list(AS_MAP.keys())

        perm = [pred_order.index(c) for c in data_cfg.class_names]
        pred_va = pred_va[:, perm, :]
        print("     pred_va_7 shape:", pred_va.shape)

    pred_va_NTc = np.transpose(pred_va, (0, 2, 1))  # (N, T, C)
    Y_va_NTc = np.transpose(Y_va, (0, 2, 1))      # (N, T, C)

    best_median_win, best_psds = tune_median_and_threshold(
        pred_va_NTc, Y_va_NTc, va_meta, class_names=CLASS_NAMES
    )
    pred_te, Y_te, te_meta = collect_preds_labels(
        dataloader=test_loader,
        sed_model=sed_model,
        device=device,
        frames_1s=data_cfg.frames_1s,
    )
    if AS_MAP != None:
        pred_order = list(AS_MAP.keys())

        perm = [pred_order.index(c) for c in CLASS_NAMES]
        pred_te = pred_te[:, perm, :]
    if best_median_win > 1:
        print(f"Applying median filter (width={best_median_win}) to test predictions...")
        pred_te = scipy.ndimage.median_filter(
            pred_te,
            size=(1, 1, best_median_win),
        )


    best_thresholds, f1info = find_best_threshold_per_class_event(pred_va_NTc, Y_va_NTc, data_cfg.frames_1s, data_cfg.th_grid)
    print("     Best thresholds per class:", best_thresholds)
    print("     f1_event_micro:", f1info["f1_event_micro"], "f1_event_macro:", f1info["f1_event_macro"])

    print("PRED TEST")
    auc = 0
    pred_te_NTc = np.transpose(pred_te, (0, 2, 1))
    Y_te_NTc = np.transpose(Y_te, (0, 2, 1))
    auc, psds1 = get_metrics(Y_te, pred_te, te_meta, best_thresholds, CLASS_NAMES, use_double_threshold_clock = False)

    return Y_te, pred_te, te_meta, Y_va, pred_va, va_meta, auc, psds1

