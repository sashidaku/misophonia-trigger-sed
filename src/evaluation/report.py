import torch
from src.evaluation.postprocess import binary_to_event_df_single_class, double_threshold_2d, fill_short_gaps, remove_short_events
from src.evaluation.psds import psds1_from_arrays
from src.evaluation.metrics import (
    compute_scores, pr_roc_auc_framewise,
    brier_per_class, ece_score, preds_to_event_df,
    event_based_evaluation_df, segment_based_evaluation_df,
    build_gt_event_df,
)
from config_loader import data_cfg
import numpy as np
import pandas as pd


def get_metrics(
    Y_true,
    pred,
    te_meta,
    best_thresholds,
    CLASS_NAMES,
    use_double_threshold_clock: bool = False,
    clock_low_ratio: float = 0.6,
    clock_th_low: float = None,
):
  Y_te_NTc    = np.transpose(Y_true, (0, 2, 1))
  pred_te_NTc = np.transpose(pred, (0, 2, 1)) 
  min_len_frames = int(round(0.25 * data_cfg.frames_1s))
  bin_te_NTC = (pred_te_NTc >= best_thresholds.reshape(1,1,-1)).astype(np.uint8)
  if use_double_threshold_clock:
    clock_idx = CLASS_NAMES.index("clock")
    th_high = float(best_thresholds[clock_idx])
    th_low = float(clock_th_low) if clock_th_low is not None else float(th_high * clock_low_ratio)

    bin_te_NTC[:, :, clock_idx] = double_threshold_2d(
        pred_te_NTc[:, :, clock_idx],
        th_high=th_high,
        th_low=th_low,
    )

  gap_frames = int(round(0.1 * data_cfg.frames_1s))
  bin_te_NTC = fill_short_gaps(bin_te_NTC, gap_frames)
  pred_bin = remove_short_events(bin_te_NTC, min_len_frames)

  scores = compute_scores(pred_bin, Y_te_NTc, frames_in_1_sec=data_cfg.frames_1s)
  auc = pr_roc_auc_framewise(Y_te_NTc, pred_te_NTc)
  bcls, bmacro = brier_per_class(Y_te_NTc, pred_te_NTc)
  ecls, emacro = ece_score(Y_te_NTc, pred_te_NTc, n_bins=15)

  print(
        f"[TEST] F1@1s={scores['f1_overall_1sec']:.3f} "
        f"ER@1s={scores['er_overall_1sec']:.3f} "
        f"PR-AUC={auc['macro_ap']:.3f} ROC-AUC={auc['macro_roc']:.3f} "
        f"Brier={bmacro:.4f} ECE={emacro:.4f}"
  )
  te_meta = _sanitize_meta(te_meta)
  gt_df, filenames, durations = get_psds_meta(Y_te_NTc, te_meta, CLASS_NAMES)    
  filenames_list = [m["filename"] for m in te_meta]
  est_df = preds_to_event_df(
      pred_scores=pred_te_NTc,
      filenames=[m["filename"] for m in te_meta],
      class_names=CLASS_NAMES,
      frames_per_sec=data_cfg.frames_1s,
      thresholds=best_thresholds,
      median_win=None,
  )
  
  if use_double_threshold_clock:
    clock_idx = CLASS_NAMES.index("clock")

    est_df = est_df[est_df["event_label"] != "clock"].copy()

    clock_bin_NT = pred_bin[:, :, clock_idx].astype(np.uint8)

    clock_df = binary_to_event_df_single_class(
        bin_NT=clock_bin_NT,
        filenames=filenames_list,
        class_name="clock",
        frames_per_sec=data_cfg.frames_1s,
    )
    est_df = pd.concat([est_df, clock_df], ignore_index=True)

  metric_event = event_based_evaluation_df(gt_df, est_df)
  metric_segment = segment_based_evaluation_df(gt_df, est_df, 1.0)

  print_sed_eval_classwise_tsv(metric_event, metric_segment, include_overall=True)

  psds1 = psds1_from_arrays(
      pred_scores=pred_te_NTc,
      filenames=filenames,
      class_names=CLASS_NAMES,
      frames_per_sec=data_cfg.frames_1s,
      groundtruth_df=gt_df,
      durations_sec=durations,
      scenario="psds1",
  )
#   psds2 = psds1_from_arrays(
#       pred_scores=pred_te_NTc,
#       filenames=filenames,
#       class_names=CLASS_NAMES,
#       frames_per_sec=data_cfg.frames_1s,
#       groundtruth_df=gt_df,
#       durations_sec=durations,
#       scenario="psds2",
#   )
#   print(f"[PSDS] S1={psds1:.4f} S2={psds2:.3f}")
  print(f"[PSDS] S1={psds1:.4f}")
  from sklearn.metrics import f1_score, jaccard_score
  if len(CLASS_NAMES) != 1:
    print("\n" + "="*40)
    print(" [Polyphony Analysis] (Diagnosis for BiESN vs BiGRU)")
    print("="*40)
    Y_flat = Y_te_NTc.reshape(-1, Y_te_NTc.shape[-1])
    P_flat = pred_bin.reshape(-1, pred_bin.shape[-1])
    gt_polyphony = Y_flat.sum(axis=1)
    mask_single = (gt_polyphony == 1)
    mask_overlap = (gt_polyphony >= 2)
    for label, mask in [("Single  (1 sound )", mask_single), ("Overlap (>=2 sounds)", mask_overlap)]:
      if mask.sum() > 0:
        f1_macro = f1_score(Y_flat[mask], P_flat[mask], average='macro', zero_division=0)
        jaccard  = jaccard_score(Y_flat[mask], P_flat[mask], average='samples', zero_division=0)
        print(f"  {label} : Frames={mask.sum():6d} | F1-Macro={f1_macro:.3f} | Jaccard={jaccard:.3f}")
      else:
        print(f"  {label} : No frames found.")
    if mask_overlap.sum() > 0:
        pred_counts = P_flat.sum(axis=1)
        under_estimation_rate = (pred_counts[mask_overlap] < gt_polyphony[mask_overlap]).mean()
        print(f"  [Overlap Error] Under-estimation Rate: {under_estimation_rate:.1%} (Predicted < Truth)")

    df_cnt, df_pct = fp_overlap_matrix(
        Y_te_NTc,
        pred_bin,
        CLASS_NAMES,
        distribute="uniform"
    )

    print("\n" + "="*40)
    print("[Pseudo Confusion] FP overlap counts (row=pred FP class, col=GT context)")
    print("="*40)
    print(df_cnt.round(0).to_string())

    print("\n" + "="*40)
    print("[Pseudo Confusion] FP overlap ratio (row-normalized)")
    print("="*40)
    print((df_pct * 100).round(1).to_string())
    print("="*40 + "\n")
    analyze_insertions(Y_te_NTc, pred_te_NTc, CLASS_NAMES, best_thresholds)

  return auc['macro_ap'], psds1

def analyze_insertions(Y_true, Y_pred, class_names, thresholds):
    if Y_true.ndim == 3:
        Y_true = Y_true.reshape(-1, Y_true.shape[-1])
    if Y_pred.ndim == 3:
        Y_pred = Y_pred.reshape(-1, Y_pred.shape[-1])

    thresholds = np.array(thresholds)
    
    if Y_pred.shape[-1] != len(thresholds):
        raise ValueError(f"Pred: {Y_pred.shape[-1]}, Thresholds: {len(thresholds)}")


    pred_bin = (Y_pred >= thresholds).astype(int)

    false_positives = (Y_true == 0) & (pred_bin == 1)
    
    total_fp = np.sum(false_positives)
    
    print("="*40)
    print(f"[Insertion Error Analysis]")
    print(f"Total Insertions (FP): {total_fp}")

    if total_fp == 0:
        print(" -> No insertion errors detected! (Perfect Precision)")
        print("="*40)
        return 0, 0

    is_silence = (np.sum(Y_true, axis=1) == 0)
    
    fp_in_silence = np.sum(false_positives[is_silence])
    
    fp_in_activity = np.sum(false_positives[~is_silence])
    
    print(f" - In Silence (Noise):   {fp_in_silence} ({fp_in_silence/total_fp*100:.1f}%)")
    print(f" - During Activity (Crosstalk): {fp_in_activity} ({fp_in_activity/total_fp*100:.1f}%)")
    print("="*40)
    
    return fp_in_silence, fp_in_activity

def fp_overlap_matrix(Y_true_NTC: np.ndarray,
                      P_pred_NTC: np.ndarray,
                      class_names,
                      *,
                      distribute: str = "multi_count"):

    assert Y_true_NTC.shape == P_pred_NTC.shape
    N, T, C = Y_true_NTC.shape
    Y = Y_true_NTC.reshape(-1, C).astype(np.uint8)
    P = P_pred_NTC.reshape(-1, C).astype(np.uint8)

    silence_col = "__silence__"
    cols = list(class_names) + [silence_col]

    M = np.zeros((C, C + 1), dtype=np.float64)

    gt_sum = Y.sum(axis=1)
    is_silence = (gt_sum == 0)

    for c in range(C):
        fp_mask = (P[:, c] == 1) & (Y[:, c] == 0)
        if fp_mask.sum() == 0:
            continue

        M[c, C] = (fp_mask & is_silence).sum()

        active_mask = fp_mask & (~is_silence)
        if active_mask.sum() == 0:
            continue

        if distribute == "multi_count":
            for g in range(C):
                M[c, g] = (active_mask & (Y[:, g] == 1)).sum()

        elif distribute == "uniform":
            idx = np.where(active_mask)[0]
            k = gt_sum[idx].astype(np.float64)
            for g in range(C):
                hit = (Y[idx, g] == 1)
                if hit.any():
                    M[c, g] += (1.0 / k[hit]).sum()
        else:
            raise ValueError("distribute must be 'multi_count' or 'uniform'")

    row_tot = M.sum(axis=1, keepdims=True)
    M_pct = np.divide(M, row_tot, out=np.zeros_like(M), where=(row_tot > 0))

    df_cnt = pd.DataFrame(M, index=class_names, columns=cols)
    df_pct = pd.DataFrame(M_pct, index=class_names, columns=cols)

    return df_cnt, df_pct


def _sanitize_meta(meta):
    if isinstance(meta, dict):
        for k in ["filenames", "fnames", "files", "filename", "fname"]:
            if k in meta:
                meta[k] = [_to_audio_id(v) for v in meta[k]]
                break
        for k in ["durations", "durs", "dur", "duration"]:
            if k in meta:
                meta[k] = [float(_scalarize(v)) for v in meta[k]]
                break
        return meta

    if isinstance(meta, (tuple, list)) and len(meta) == 2:
        fns, durs = meta
        fns = [_to_audio_id(v) for v in fns]
        durs = [float(_scalarize(v)) for v in durs]
        return (fns, durs)

    return meta

def _scalarize(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return ""
        return x.reshape(-1)[0].item() if x.size == 1 else x.reshape(-1)[0]
    if isinstance(x, (list, tuple)):
        return _scalarize(x[0]) if len(x) > 0 else ""
    return x

def _to_audio_id(x):
    x = _scalarize(x)
    if isinstance(x, (bytes, np.bytes_)):
        x = x.decode("utf-8", errors="ignore")
    return str(x)

def get_psds_meta(Y_te_NTc, te_meta, CLASS_NAMES):
    filenames = [m["filename"] for m in te_meta]
    durations = [m.get("duration", m.get("dur")) for m in te_meta]

    if any(d is None for d in durations):
        raise KeyError("te_meta must contain 'duration' or 'dur' for every item.")

    gt_df = build_gt_event_df(
        Y_te_NTc,
        CLASS_NAMES,
        data_cfg.frames_1s,
        filenames=filenames,
        t0_sec=0.0,
    )

    return gt_df, filenames, durations

def print_sed_eval_classwise_tsv(metric_event, metric_segment, *, include_overall=True):
    ev_macro = metric_event.results_class_wise_average_metrics()
    sg_macro = metric_segment.results_class_wise_average_metrics()

    ev_cw = metric_event.results_class_wise_metrics()
    sg_cw = metric_segment.results_class_wise_metrics()

    classes = sorted(set(ev_cw.keys()) | set(sg_cw.keys()))

    header = [
        "class",
        "ev_f","ev_p","ev_r","ev_er","ev_del","ev_ins","ev_Nref","ev_Nsys",
        "sg_f","sg_p","sg_r","sg_er","sg_del","sg_ins","sg_Nref","sg_Nsys",
        "sg_acc","sg_bal_acc","sg_sens","sg_spec",
    ]
    print("\t".join(header))

    def _print_row(name, evd, sgd):
        row = [
            name,
            _g(evd,"f_measure","f_measure"), _g(evd,"f_measure","precision"), _g(evd,"f_measure","recall"),
            _g(evd,"error_rate","error_rate"), _g(evd,"error_rate","deletion_rate"), _g(evd,"error_rate","insertion_rate"),
            _g(evd,"count","Nref"), _g(evd,"count","Nsys"),
            _g(sgd,"f_measure","f_measure"), _g(sgd,"f_measure","precision"), _g(sgd,"f_measure","recall"),
            _g(sgd,"error_rate","error_rate"), _g(sgd,"error_rate","deletion_rate"), _g(sgd,"error_rate","insertion_rate"),
            _g(sgd,"count","Nref"), _g(sgd,"count","Nsys"),
            _g(sgd,"accuracy","accuracy"), _g(sgd,"accuracy","balanced_accuracy"),
            _g(sgd,"accuracy","sensitivity"), _g(sgd,"accuracy","specificity"),
        ]

        out = []
        for x in row:
            if isinstance(x, (float, np.floating)):
                out.append("" if np.isnan(x) else f"{x:.6f}")
            else:
                out.append(str(x))
        print("\t".join(out))

    if include_overall:
        _print_row(
            "__overall__",
            ev_macro,
            sg_macro,
        )

    for c in classes:
        _print_row(c, ev_cw.get(c, {}), sg_cw.get(c, {}))

def _g(d, *keys, default=np.nan):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur