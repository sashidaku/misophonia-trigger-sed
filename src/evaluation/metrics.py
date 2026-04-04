import numpy as np
import pandas as pd
from typing import Iterable, List, Tuple, Dict, Optional, Sequence, Union

from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

import sed_eval


def _bool_to_events_1d(b: np.ndarray, fps: float, fname: str, label: str):
    x = b.astype(np.int8)
    d = np.diff(np.r_[0, x, 0])
    on = np.where(d == 1)[0]
    off = np.where(d == -1)[0]
    rows = []
    for s, e in zip(on, off):
        if e > s:
            rows.append((fname, float(s/fps), float(e/fps), label))
    return rows

def _array_to_event_df(bin_arr: np.ndarray,
                       fps: float,
                       filenames: List[str],
                       class_names: List[str]) -> pd.DataFrame:
    N, T, C = bin_arr.shape
    rows = []
    for i in range(N):
        fname = filenames[i]
        for c, label in enumerate(class_names):
            rows += _bool_to_events_1d(bin_arr[i, :, c], fps, fname, label)
    return pd.DataFrame(rows, columns=['filename','onset','offset','event_label'])

def _event_f1(reference_df: pd.DataFrame,
              estimated_df: pd.DataFrame,
              classes: List[str],
              onset_collar: float = 0.2,
              offset_collar_rate: float = 0.2,
              evaluate_offset: bool = True,
              macro: bool = False) -> float:
    metr = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=classes,
        t_collar=onset_collar,
        percentage_of_length=offset_collar_rate,
        evaluate_onset=True,
        evaluate_offset=evaluate_offset,
        event_matching_type='optimal',
    )
    for fname in reference_df['filename'].unique():
        ref = reference_df[reference_df.filename == fname][['event_label','onset','offset']].to_dict('records')
        est = estimated_df[estimated_df.filename == fname][['event_label','onset','offset']].to_dict('records')
        metr.evaluate(reference_event_list=ref, estimated_event_list=est)
    if macro:
        return float(metr.results_class_wise_average_metrics()['f_measure']['f_measure'])
    else:
        return float(metr.results_overall_metrics()['f_measure']['f_measure'])

def f1_overall_1sec(O: np.ndarray, T: np.ndarray, block_size: int) -> float:
    return f1_overall_framewise(_block_max(O, block_size), _block_max(T, block_size))

def er_overall_1sec(O: np.ndarray, T: np.ndarray, block_size: int) -> float:
    return er_overall_framewise(_block_max(O, block_size), _block_max(T, block_size))

def _as_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        N, T, C = x.shape
        return x.reshape(N*T, C)
    raise ValueError(f"x must be 2D or 3D, got {x.ndim}D")

def _block_max(x: np.ndarray, block: int) -> np.ndarray:
    x2 = _as_2d(x)
    T, C = x2.shape
    n = int(np.ceil(T / block))
    pad = n*block - T
    if pad:
        x2 = np.pad(x2, ((0, pad), (0, 0)), mode="constant")
    return x2.reshape(n, block, C).max(axis=1)

def f1_overall_framewise(O: np.ndarray, T: np.ndarray) -> float:
    O2, T2 = _as_2d(O), _as_2d(T)
    TP = np.logical_and(T2 == 1, O2 == 1).sum()
    Nref, Nsys = T2.sum(), O2.sum()
    prec = float(TP) / float(Nsys + 1e-9)
    recall = float(TP) / float(Nref + 1e-9)
    return 2.0 * prec * recall / (prec + recall + 1e-9)

def er_overall_framewise(O: np.ndarray, T: np.ndarray) -> float:
    O2, T2 = _as_2d(O), _as_2d(T)
    FP = np.logical_and(T2 == 0, O2 == 1).sum(axis=1)
    FN = np.logical_and(T2 == 1, O2 == 0).sum(axis=1)
    S = np.minimum(FP, FN).sum()
    D = np.maximum(0, FN - FP).sum()
    I = np.maximum(0, FP - FN).sum()
    Nref = T2.sum()
    den = Nref if Nref > 0 else 1e-9
    return float((S + D + I) / den)

def compute_scores(pred: np.ndarray, y: np.ndarray, frames_in_1_sec: int = 50) -> Dict[str, float]:
    return dict(
        f1_overall_1sec  = f1_overall_1sec(pred, y, frames_in_1_sec),
        er_overall_1sec  = er_overall_1sec(pred, y, frames_in_1_sec),
        f1_overall_frame = f1_overall_framewise(pred, y),
        er_overall_frame = er_overall_framewise(pred, y),
    )

def pr_roc_auc_framewise(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """(N,T,C) or (T,C)。macro AP と macro ROC-AUC を返す。"""
    yt = _as_2d(y_true)
    yp = _as_2d(y_prob)
    C = yt.shape[1]
    aps, rocs = [], []
    for c in range(C):
        ytc, ypc = yt[:, c], yp[:, c]
        if ytc.max() == ytc.min():
            continue
        aps.append(average_precision_score(ytc, ypc))
        try:
            rocs.append(roc_auc_score(ytc, ypc))
        except ValueError:
            pass
    return dict(macro_ap=float(np.mean(aps)) if aps else float("nan"),
                macro_roc=float(np.mean(rocs)) if rocs else float("nan"))

def brier_per_class(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, float]:
    yt = _as_2d(y_true); yp = _as_2d(y_prob)
    C = yt.shape[1]
    out = np.zeros(C, dtype=np.float32)
    for c in range(C):
        out[c] = brier_score_loss(yt[:, c], yp[:, c])
    return out, float(out.mean())

def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> Tuple[np.ndarray, float]:
    yt = _as_2d(y_true); yp = _as_2d(y_prob)
    C = yt.shape[1]
    eces = np.zeros(C, dtype=np.float32)
    bins = np.linspace(0, 1, n_bins + 1)
    for c in range(C):
        conf = yp[:, c]
        labl = yt[:, c]
        ece = 0.0
        for b in range(n_bins):
            msk = (conf >= bins[b]) & (conf < bins[b+1])
            if not np.any(msk): 
                continue
            acc = labl[msk].mean()
            conf_mean = conf[msk].mean()
            w = msk.mean()
            ece += w * abs(acc - conf_mean)
        eces[c] = ece
    return eces, float(eces.mean())


def event_based_evaluation_df(
    reference_df: pd.DataFrame,
    estimated_df: pd.DataFrame,
    t_collar: float = 0.200,
    percentage_of_length: float = 0.2
) -> sed_eval.sound_event.EventBasedMetrics:
    classes = sorted(set(reference_df.event_label.dropna().unique())
                     | set(estimated_df.event_label.dropna().unique()))
    metric = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=classes,
        t_collar=t_collar,
        percentage_of_length=percentage_of_length,
        empty_system_output_handling="zero_score",
    )
    for fname in reference_df["filename"].unique():
        metric.evaluate(
            reference_event_list=_event_list_for_file(reference_df, fname),
            estimated_event_list=_event_list_for_file(estimated_df, fname),
        )
    return metric

def segment_based_evaluation_df(
    reference_df: pd.DataFrame,
    estimated_df: pd.DataFrame,
    time_resolution: float = 1.0
) -> sed_eval.sound_event.SegmentBasedMetrics:
    classes = sorted(set(reference_df.event_label.dropna().unique())
                     | set(estimated_df.event_label.dropna().unique()))
    metric = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=classes,
        time_resolution=time_resolution,
    )
    for fname in reference_df["filename"].unique():
        metric.evaluate(
            reference_event_list=_event_list_for_file(reference_df, fname),
            estimated_event_list=_event_list_for_file(estimated_df, fname),
        )
    return metric

def _event_list_for_file(df: pd.DataFrame, fname: str) -> List[dict]:
    d = df[df["filename"] == fname]
    if len(d) == 1 and pd.isna(d["event_label"].iloc[0]):
        return [{"filename": fname}]
    return d.to_dict("records")

def build_gt_event_df(
    y_NTc: np.ndarray,
    class_names: List[str],
    frames_in_1_sec: int,
    filenames: Optional[List[str]] = None,
    t0_sec: float = 0.0,
) -> pd.DataFrame:
    if y_NTc.ndim == 3:
        N = y_NTc.shape[0]
        if filenames is None:
            filenames = [f"clip_{i:06d}" for i in range(N)]
        assert len(filenames) == N, f"filenames length mismatch: {len(filenames)} vs N={N}"

        rows = []
        for i in range(N):
            df_i = build_gt_event_df(
                y_NTc[i],
                class_names,
                frames_in_1_sec,
                filenames=[filenames[i]],
                t0_sec=t0_sec,
            )
            rows.append(df_i)
        return pd.concat(rows, ignore_index=True) if len(rows) > 0 else pd.DataFrame(
            columns=["filename", "onset", "offset", "event_label"]
        )

    y = y_NTc
    if y.ndim == 1:
        y = y[:, None]

    if y.ndim == 2 and y.shape[1] != len(class_names) and y.shape[0] == len(class_names):
        y = y.T

    assert y.ndim == 2, f"y must be 2D (T,C). got {y.shape}"
    assert y.shape[1] == len(class_names), f"shape mismatch: y={y.shape}, class_names={len(class_names)}"

    if filenames is None:
        filename = "clip"
    else:
        filename = filenames[0] if isinstance(filenames, list) else str(filenames)

    T, C = y.shape
    out = []
    for c, name in enumerate(class_names):
        act = y[:, c].astype(bool)
        t = 0
        while t < T:
            if act[t]:
                t0 = t
                while t < T and act[t]:
                    t += 1
                onset  = t0 / float(frames_in_1_sec) + t0_sec
                offset = t  / float(frames_in_1_sec) + t0_sec
                out.append({"filename": filename, "onset": onset, "offset": offset, "event_label": name})
            else:
                t += 1

    return pd.DataFrame(out, columns=["filename", "onset", "offset", "event_label"])

def preds_to_event_df(
    pred_scores: np.ndarray,
    filenames: Sequence[str],
    class_names: Sequence[str],
    frames_per_sec: float,
    thresholds: Union[float, Sequence[float], np.ndarray],
    median_win: Optional[int] = None,
) -> pd.DataFrame:
    assert pred_scores.ndim == 3, "pred_scores must be (N,T,C)"
    N, T, C = pred_scores.shape
    assert len(filenames) == N, "filenames length must equal N"
    assert len(class_names) == C, "class_names length must equal C"

    if np.isscalar(thresholds):
        th = np.full(C, float(thresholds), dtype=float)
    else:
        th = np.asarray(thresholds, dtype=float)
        assert th.shape == (C,), f"thresholds must be scalar or shape (C,), got {th.shape}"

    S = pred_scores
    if median_win and median_win > 1:
        try:
            from scipy.ndimage import median_filter
        except Exception as e:
            raise RuntimeError("median_win を使うには scipy が必要です") from e
        S = np.empty_like(pred_scores)
        for i in range(N):
            S[i] = median_filter(pred_scores[i], size=(median_win, 1), mode="nearest")

    rows = []
    fps = float(frames_per_sec)

    def _bool_to_events_1d(b: np.ndarray):
        x = b.astype(np.int8)
        d = np.diff(np.r_[0, x, 0])
        on  = np.where(d == 1)[0]
        off = np.where(d == -1)[0]
        return on, off

    for i, fname in enumerate(filenames):
        for c, label in enumerate(class_names):
            b = S[i, :, c] > th[c]
            on, off = _bool_to_events_1d(b)
            if on.size == 0:
                continue
            for s, e in zip(on, off):
                if e <= s:
                    continue
                rows.append((fname, float(s / fps), float(e / fps), str(label)))

    if not rows:
        return pd.DataFrame(columns=["filename", "onset", "offset", "event_label"])

    return pd.DataFrame(rows, columns=["filename", "onset", "offset", "event_label"])

def find_best_threshold_global_event(
    pred_va: np.ndarray,
    y_va: np.ndarray,
    frames_1s: int,
    th_grid: Iterable[float],
    filenames: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    onset_collar: float = 0.2,
    offset_collar_rate: float = 0.2,
    evaluate_offset: bool = True,
    median_win: Optional[int] = None,
) -> Dict[str, float]:
    assert pred_va.shape == y_va.shape and pred_va.ndim == 3
    N, T, C = pred_va.shape
    fps = float(frames_1s)

    if filenames is None:
        filenames = [f"utt_{i:06d}" for i in range(N)]
    if class_names is None:
        class_names = [f"class_{c}" for c in range(C)]

    ref_df = _array_to_event_df((y_va > 0.5).astype(np.uint8), fps, filenames, class_names)

    best = {"th": 0.5, "f1_event_micro": -1.0, "f1_event_macro": -1.0}

    S_all = pred_va.copy()
    if median_win and median_win > 1:
        from scipy.ndimage import median_filter
        S_all = np.stack([median_filter(pred_va[i], size=(median_win,1), mode='nearest')
                          for i in range(N)], axis=0)

    for th in th_grid:
        est_bin = (S_all > th).astype(np.uint8)
        est_df  = _array_to_event_df(est_bin, fps, filenames, class_names)

        f1_micro = _event_f1(ref_df, est_df, class_names,
                             onset_collar, offset_collar_rate, evaluate_offset, macro=False)
        if f1_micro > best["f1_event_micro"]:
            f1_macro = _event_f1(ref_df, est_df, class_names,
                                 onset_collar, offset_collar_rate, evaluate_offset, macro=True)
            best = {"th": float(th),
                    "f1_event_micro": float(f1_micro),
                    "f1_event_macro": float(f1_macro)}
    return best

def find_best_threshold_per_class_event(
    pred_va: np.ndarray,
    y_va: np.ndarray,
    frames_1s: int,
    th_grid: Iterable[float],
    filenames: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    onset_collar: float = 0.2,
    offset_collar_rate: float = 0.2,
    evaluate_offset: bool = True,
    median_win: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    assert pred_va.shape == y_va.shape and pred_va.ndim == 3
    N, T, C = pred_va.shape
    fps = float(frames_1s)

    if filenames is None:
        filenames = [f"utt_{i:06d}" for i in range(N)]
    if class_names is None:
        class_names = [f"class_{c}" for c in range(C)]

    y_bin = (y_va > 0.5).astype(np.uint8)
    ref_df = _array_to_event_df(y_bin, fps, filenames, class_names)

    S = pred_va.copy()
    if median_win and median_win > 1:
        from scipy.ndimage import median_filter
        S = np.stack([median_filter(pred_va[i], size=(median_win,1), mode='nearest')
                      for i in range(N)], axis=0)

    best_th  = np.full(C, 0.5, dtype=np.float32)
    best_f1c = np.full(C, -1.0, dtype=np.float32)

    for c, label in enumerate(class_names):
        ref_c = ref_df[ref_df.event_label == label]
        for th in th_grid:
            est_bin_c = np.zeros((N, T, 1), dtype=np.uint8)
            est_bin_c[:, :, 0] = (S[:, :, c] > th).astype(np.uint8)
            est_df_c = _array_to_event_df(est_bin_c, fps, filenames, [label])

            f1_c = _event_f1(ref_c, est_df_c, [label],
                             onset_collar, offset_collar_rate, evaluate_offset, macro=False)
            if f1_c > best_f1c[c]:
                best_f1c[c] = f1_c
                best_th[c]  = float(th)

    est_all = np.zeros_like(y_bin)
    for c in range(C):
        est_all[:, :, c] = (S[:, :, c] > best_th[c]).astype(np.uint8)
    est_df_all = _array_to_event_df(est_all, fps, filenames, class_names)

    f1_micro = _event_f1(ref_df, est_df_all, class_names,
                         onset_collar, offset_collar_rate, evaluate_offset, macro=False)
    f1_macro = _event_f1(ref_df, est_df_all, class_names,
                         onset_collar, offset_collar_rate, evaluate_offset, macro=True)

    return best_th, {"f1_event_micro": float(f1_micro), "f1_event_macro": float(f1_macro)}

