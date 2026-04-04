import csv
import importlib
import inspect
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Literal, Optional, Sequence, Union
import pandas as pd
from torch.utils.data import Subset, DataLoader
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset
import random

from src.analysis.prediction_export import _to_numpy, extract_filenames_durations
from src.evaluation.report import get_metrics

def _maybe_sigmoid_np(pred):
    pred = _to_numpy(pred)
    if pred.size == 0:
        return pred
    mn, mx = float(np.nanmin(pred)), float(np.nanmax(pred))
    if mn < -1e-6 or mx > 1 + 1e-6:
        return 1.0 / (1.0 + np.exp(-pred))
    return pred

def _align_to_NCT_np(x, C: int, expected_T=None, name="x"):
    x = _to_numpy(x)
    if x.ndim == 3:
        N, a, b = x.shape
        if a == C:  # NCT
            return x
        if b == C:  # NTC -> NCT
            return np.transpose(x, (0, 2, 1))
        if expected_T is not None:
            if a == expected_T:
                return np.transpose(x, (0, 2, 1))
            if b == expected_T:
                return x
        raise ValueError(f"{name}: cannot align shape {x.shape} to [N,{C},T]")
    elif x.ndim == 2:
        return x[:, None, :]
    elif x.ndim == 1:
        return x[:, None, None]
    else:
        raise ValueError(f"{name}: unsupported ndim={x.ndim} shape={x.shape}")

def _binary_runs(y01: np.ndarray):
    y = y01.astype(np.uint8)
    T = y.shape[0]
    runs = []
    in_run = False
    s = 0
    for t in range(T):
        if (not in_run) and y[t] == 1:
            in_run = True
            s = t
        elif in_run and y[t] == 0:
            runs.append((s, t))
            in_run = False
    if in_run:
        runs.append((s, T))
    return runs

def filter_short_events_roll(y01: np.ndarray, dt: float, min_dur_s: float):
    y = y01.astype(np.uint8).copy()
    for s, e in _binary_runs(y):
        dur = (e - s) * dt
        if dur < min_dur_s:
            y[s:e] = 0
    return y

def roll_to_event_list(y01: np.ndarray, dt: float, label: str, filename: str):
    events = []
    for s, e in _binary_runs(y01):
        onset = s * dt
        offset = e * dt
        events.append({
            "event_label": label,
            "event_onset": float(onset),
            "event_offset": float(offset),
            "file": filename,
        })
    return events

def eval_sed_eval_metrics(
    filenames,
    durations,
    Y_bin_nct,
    P_bin_nct,
    class_names,
    *,
    t_collar: float = 0.25,
    segment_res: float = 1.0,
):
    import sed_eval
    import dcase_util

    segm = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=class_names,
        time_resolution=segment_res,
    )
    evm = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=class_names,
        t_collar=t_collar,
    )

    N, C, T = Y_bin_nct.shape
    for i in range(N):
        fn = str(filenames[i]) if filenames is not None else f"clip_{i:06d}"
        dur = float(durations[i]) if durations is not None and len(durations) == N else 10.0
        dt = dur / T

        ref_events = []
        est_events = []
        for c, lab in enumerate(class_names):
            ref_events += roll_to_event_list(Y_bin_nct[i, c], dt, lab, fn)
            est_events += roll_to_event_list(P_bin_nct[i, c], dt, lab, fn)

        ref_md = dcase_util.containers.MetaDataContainer(ref_events)
        est_md = dcase_util.containers.MetaDataContainer(est_events)

        segm.evaluate(reference_event_list=ref_md, estimated_event_list=est_md)
        evm.evaluate(reference_event_list=ref_md, estimated_event_list=est_md)

    seg_overall = segm.results_overall_metrics()
    ev_overall = evm.results_overall_metrics()

    out = {
        "ev_f1": float(ev_overall["f_measure"]["f_measure"]),
        "ev_p": float(ev_overall["f_measure"]["precision"]),
        "ev_r": float(ev_overall["f_measure"]["recall"]),
        "ev_er": float(ev_overall["error_rate"]["error_rate"]),
        "sg_f1": float(seg_overall["f_measure"]["f_measure"]),
        "sg_p": float(seg_overall["f_measure"]["precision"]),
        "sg_r": float(seg_overall["f_measure"]["recall"]),
        "sg_er": float(seg_overall["error_rate"]["error_rate"]),
    }
    return out

def duration_threshold_sweep(
    *,
    Y_va,
    pred_va,
    va_meta,
    Y_te,
    pred_te,
    te_meta,
    class_names,
    chewing_min_durs=(1.0, 1.5, 2.0),
    th_grid=None,
    expected_T=250,
    t_collar=0.25,
    segment_res=1.0,
    target_class="chewing",   # これだけ最小継続時間で再定義する
    out_tsv="duration_sweep.tsv",
):
    if th_grid is None:
        th_grid = np.linspace(0.05, 0.95, 19)

    C = len(class_names)
    if target_class not in class_names:
        raise ValueError(f"target_class={target_class} not in class_names={class_names}")
    tc = class_names.index(target_class)

    va_fns, va_durs = extract_filenames_durations(va_meta)
    te_fns, te_durs = extract_filenames_durations(te_meta)

    P_va = _maybe_sigmoid_np(pred_va)
    P_te = _maybe_sigmoid_np(pred_te)

    Y_va_nct = _align_to_NCT_np(Y_va, C=C, expected_T=expected_T, name="Y_va")
    Y_te_nct = _align_to_NCT_np(Y_te, C=C, expected_T=expected_T, name="Y_te")
    P_va_nct = _align_to_NCT_np(P_va, C=C, expected_T=expected_T, name="P_va")
    P_te_nct = _align_to_NCT_np(P_te, C=C, expected_T=expected_T, name="P_te")

    rows = []
    Nva, _, T = Y_va_nct.shape
    Nte, _, T2 = Y_te_nct.shape
    assert T == T2

    for min_dur in chewing_min_durs:
        Yva_f = (Y_va_nct > 0.5).astype(np.uint8)
        Yte_f = (Y_te_nct > 0.5).astype(np.uint8)

        for i in range(Nva):
            dur = float(va_durs[i]) if len(va_durs) == Nva else 10.0
            dt = dur / T
            Yva_f[i, tc] = filter_short_events_roll(Yva_f[i, tc], dt, min_dur)

        for i in range(Nte):
            dur = float(te_durs[i]) if len(te_durs) == Nte else 10.0
            dt = dur / T
            Yte_f[i, tc] = filter_short_events_roll(Yte_f[i, tc], dt, min_dur)

        best = {"th": None, "ev_f1": -1.0, "metrics": None}
        for th in th_grid:
            Pva_bin = (P_va_nct >= th).astype(np.uint8)
            m = eval_sed_eval_metrics(
                va_fns, va_durs, Yva_f, Pva_bin, class_names,
                t_collar=t_collar, segment_res=segment_res
            )
            if m["ev_f1"] > best["ev_f1"]:
                best = {"th": float(th), "ev_f1": m["ev_f1"], "metrics": m}

        Pte_bin = (P_te_nct >= best["th"]).astype(np.uint8)
        m_te = eval_sed_eval_metrics(
            te_fns, te_durs, Yte_f, Pte_bin, class_names,
            t_collar=t_collar, segment_res=segment_res
        )

        best_thresholds = np.full((C,), float(best["th"]), dtype=np.float32)
        auc_va, psds1_va = get_metrics(Yva_f, P_va_nct, va_meta, best_thresholds, class_names)
        auc_te, psds1_te = get_metrics(Yte_f, P_te_nct, te_meta, best_thresholds, class_names)

        rows.append({
            "min_chewing_dur_s": float(min_dur),
            "best_th_on_val": float(best["th"]),
            "val_ev_f1": float(best["metrics"]["ev_f1"]),
            "test_ev_f1": float(m_te["ev_f1"]),
            "test_ev_p": float(m_te["ev_p"]),
            "test_ev_r": float(m_te["ev_r"]),
            "test_ev_er": float(m_te["ev_er"]),
            "test_sg_f1": float(m_te["sg_f1"]),
            "test_sg_p": float(m_te["sg_p"]),
            "test_sg_r": float(m_te["sg_r"]),
            "test_sg_er": float(m_te["sg_er"]),
            "val_psds1": float(psds1_va),
            "test_psds1": float(psds1_te),
            "val_auc": float(auc_va),
            "test_auc": float(auc_te),
        })

    df = pd.DataFrame(rows).sort_values("min_chewing_dur_s")
    df.to_csv(out_tsv, sep="\t", index=False)
    print(df)
    print(f"[saved] {out_tsv}")
    return df

