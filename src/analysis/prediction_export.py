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
import third_party.EfficientSED.config as cfg


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _maybe_sigmoid(pred):
    pred = _to_numpy(pred)
    if pred.size == 0:
        return pred
    mn, mx = float(np.nanmin(pred)), float(np.nanmax(pred))
    if mn < -1e-6 or mx > 1 + 1e-6:
        return 1.0 / (1.0 + np.exp(-pred))
    return pred

def align_to_NCT(x, C: int, expected_T= None, name="x"):
    x = _to_numpy(x)
    if x.ndim == 3:
        N, a, b = x.shape
        if a == C:
            return x
        if b == C:
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
    
def extract_filenames_durations(meta):
    if meta is None:
        return [], []

    if isinstance(meta, dict):
        for fk in ["filenames", "fnames", "files", "filename", "fname"]:
            if fk in meta:
                fns = meta[fk]
                break
        else:
            fns = None

        for dk in ["durations", "durs", "dur", "duration", "lens"]:
            if dk in meta:
                durs = meta[dk]
                break
        else:
            durs = None

        if fns is None:
            vals = list(meta.values())
            fns = vals[0] if len(vals) > 0 else []
            durs = vals[1] if len(vals) > 1 else []
        return list(map(str, fns)), [float(x) for x in (durs if durs is not None else [])]

    if isinstance(meta, (tuple, list)) and len(meta) == 2 and not isinstance(meta[0], dict):
        fns, durs = meta
        if len(fns) > 0 and isinstance(fns[0], (tuple, list)) and len(fns[0]) == 2:
            pairs = fns
            return [str(p[0]) for p in pairs], [float(p[1]) for p in pairs]
        return list(map(str, fns)), [float(x) for x in durs]

    if isinstance(meta, list) and len(meta) > 0:
        if isinstance(meta[0], (tuple, list)) and len(meta[0]) == 2:
            return [str(x[0]) for x in meta], [float(x[1]) for x in meta]
        if isinstance(meta[0], dict):
            fns = [str(d.get("filename", d.get("fname", d.get("file", "")))) for d in meta]
            durs = [float(d.get("dur", d.get("duration", np.nan))) for d in meta]
            return fns, durs
        
    try:
        return list(map(str, meta)), []
    except Exception:
        return [], []

def split_to_dfs(
    filenames, durations, Y, pred, class_names, expected_T=None, split_name="val"
):
    C = len(class_names)
    pred = _maybe_sigmoid(pred)
    pred_nct = align_to_NCT(pred, C=C, expected_T=expected_T, name=f"pred_{split_name}")
    Y_nct    = align_to_NCT(Y,    C=C, expected_T=expected_T, name=f"Y_{split_name}")

    N, C2, T = pred_nct.shape
    assert C2 == C
    assert len(filenames) == N, f"{split_name}: len(filenames)={len(filenames)} vs N={N}"
    if durations is None or len(durations) == 0:
        durations = [np.nan] * N
    else:
        assert len(durations) == N, f"{split_name}: len(durations)={len(durations)} vs N={N}"

    rows = []
    for i in range(N):
        fn = str(filenames[i])
        dur = float(durations[i]) if durations[i] is not None else np.nan
        for c in range(C):
            cn = class_names[c]
            for t in range(T):
                rows.append({
                    "split": split_name,
                    "file": fn,
                    "dur": dur,
                    "class": cn,
                    "t": t,
                    "pred": float(pred_nct[i, c, t]),
                    "y": int(Y_nct[i, c, t] > 0.5),
                })
    df_long = pd.DataFrame(rows)

    # file summary
    df_sum = (df_long.groupby(["split","file","class"], as_index=False)
                     .agg(pred_max=("pred","max"),
                          pred_mean=("pred","mean"),
                          y_any=("y","max"),
                          dur=("dur","max"),
                          T=("t","max")))
    df_wide = None
    if C == 1:
        pred_cols = {f"pred_{t}": pred_nct[:, 0, t] for t in range(T)}
        y_cols    = {f"y_{t}":    (Y_nct[:, 0, t] > 0.5).astype(int) for t in range(T)}
        df_wide = pd.DataFrame({
            "split": [split_name]*N,
            "file": list(map(str, filenames)),
            "dur": durations,
            **pred_cols,
            **y_cols,
        })

    return df_long, df_sum, df_wide

def write_split_tsvs(
    out_dir,
    filenames, durations, Y, pred, class_names,
    expected_T=None, split_name="val",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_long, df_sum, df_wide = split_to_dfs(
        filenames, durations, Y, pred, class_names,
        expected_T=expected_T, split_name=split_name
    )

    df_long.to_csv(out_dir / f"{split_name}_frame.tsv",
                   sep="\t", index=False, lineterminator="\n",
                   quoting=csv.QUOTE_MINIMAL)
    df_sum.to_csv(out_dir / f"{split_name}_file_summary.tsv",
                  sep="\t", index=False, lineterminator="\n",
                  quoting=csv.QUOTE_MINIMAL)

    if df_wide is not None:
        df_wide.to_csv(out_dir / f"{split_name}_wide.tsv",
                       sep="\t", index=False, lineterminator="\n",
                       quoting=csv.QUOTE_MINIMAL)

    print(f"[tsv] wrote to: {out_dir} ({split_name})")

def write_filenames_only_tsv(out_dir, filenames, split_name="val", *, dedup: bool = False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fns = [str(x) for x in (filenames or [])]

    if dedup:
        seen = set()
        fns2 = []
        for fn in fns:
            if fn in seen:
                continue
            seen.add(fn)
            fns2.append(fn)
        fns = fns2

    df = pd.DataFrame({"filename": fns})
    out_path = out_dir / f"{split_name}_filenames.tsv"
    df.to_csv(out_path, sep="\t", index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    print(f"[tsv] wrote filenames: {out_path} (n={len(fns)})")