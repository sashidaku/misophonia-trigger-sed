import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Sequence

from psds_eval import PSDSEval

from sed_scores_eval.intersection_based import psds as sse_psds

def create_score_dataframe(scores, timestamps, event_classes):
    T, C = scores.shape
    assert len(timestamps) == T + 1, "timestamps must be T+1"
    cols = ['onset', 'offset'] + list(event_classes)
    seg = np.c_[timestamps[:-1], timestamps[1:]]
    data = np.c_[seg, scores]
    return pd.DataFrame(data, columns=cols)

def build_scores_and_durations_dict(
    pred_scores: np.ndarray,
    filenames: Sequence[str],
    class_names: Sequence[str],
    frames_per_sec: float,
    durations_sec: Sequence[float] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, float]]:
    assert pred_scores.ndim == 3, "pred_scores must be (N, T, K)"
    N, T, K = pred_scores.shape
    assert len(filenames) == N
    assert len(class_names) == K

    scores_dict: Dict[str, pd.DataFrame] = {}
    durations_dict: Dict[str, float] = {}

    ts = np.arange(T + 1, dtype=np.float32) / float(frames_per_sec)

    for i, fname in enumerate(filenames):
        arr = pred_scores[i]  # (T, K)
        df = create_score_dataframe(
            arr, timestamps=ts, event_classes=list(class_names)
        )
        scores_dict[fname] = df
        durations_dict[fname] = float(durations_sec[i]) if durations_sec is not None else float(ts[-1])

    return scores_dict, durations_dict

def groundtruth_df_to_dict(
    gt_df: pd.DataFrame,
    all_filenames: Optional[Sequence[str]] = None,
) -> Dict[str, List[Tuple[float, float, str]]]:

    cols = {'filename', 'onset', 'offset', 'event_label'}
    if not cols.issubset(gt_df.columns):
        missing = cols - set(gt_df.columns)
        raise ValueError(f"gt_df is missing columns: {missing}")

    gt_dict: Dict[str, List[Tuple[float, float, str]]] = {}

    for fname, g in gt_df.groupby('filename'):
        g_clean = g.dropna(subset=['event_label', 'onset', 'offset'])
        if len(g_clean) == 0:
            gt_dict[fname] = []
        else:
            gt_dict[fname] = [
                (float(r.onset), float(r.offset), str(r.event_label))
                for r in g_clean.itertuples(index=False)
            ]

    if all_filenames is not None:
        for fname in all_filenames:
            gt_dict.setdefault(fname, [])

    return gt_dict

_PSDS_PARAMSETS = {
    "psds1": dict(dtc_threshold=0.7, gtc_threshold=0.7, cttc_threshold=None,
                  alpha_ct=0.0, alpha_st=1.0, e_max=100),
    "psds2": dict(dtc_threshold=0.1, gtc_threshold=0.1, cttc_threshold=0.3,
                  alpha_ct=0.5, alpha_st=1.0, e_max=100),
}

def compute_psds(
    list_operating_points: List[pd.DataFrame],
    groundtruth_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    scenario: str = "psds1"
):
    ps = _PSDS_PARAMSETS[scenario]
    psds = PSDSEval(ps["dtc_threshold"], ps["gtc_threshold"], ps["cttc_threshold"],
                    ground_truth=groundtruth_df, metadata=meta_df)
    for df in list_operating_points:
        psds.add_operating_point(df)
    score = psds.psds(alpha_ct=ps["alpha_ct"], alpha_st=ps["alpha_st"], max_efpr=ps["e_max"])
    return score

def compute_psds_sse(
    scores: Dict[str, pd.DataFrame],
    groundtruth_df: pd.DataFrame,
    audio_durations: Dict[str, float],
    scenario: str = "psds1",
    unit_of_time: str = "hour",
) -> float:
    all_filenames = list(scores.keys())
    gt_dict = groundtruth_df_to_dict(groundtruth_df, all_filenames=all_filenames)
    if scenario not in _PSDS_PARAMSETS:
        raise ValueError(f"Unknown scenario: {scenario}")
    ps = _PSDS_PARAMSETS[scenario]

    psds_value, *_ = sse_psds(
        scores=scores,
        ground_truth=gt_dict,
        audio_durations=audio_durations,
        dtc_threshold=ps["dtc_threshold"],
        gtc_threshold=ps["gtc_threshold"],
        cttc_threshold=ps["cttc_threshold"],
        alpha_ct=ps["alpha_ct"],
        alpha_st=ps["alpha_st"],
        max_efpr=ps["e_max"],
        unit_of_time=unit_of_time,
    )

    return float(getattr(psds_value, "value", psds_value))

def psds1_from_arrays(
    pred_scores: np.ndarray,
    filenames: Sequence[str],
    class_names: Sequence[str],
    frames_per_sec: float,
    groundtruth_df: pd.DataFrame,
    durations_sec: Sequence[float] = None,
    scenario: str = "psds1",
) -> float:
    scores, durations = build_scores_and_durations_dict(
        pred_scores, filenames, class_names, frames_per_sec, durations_sec
    )
    return compute_psds_sse(scores, groundtruth_df, durations, scenario=scenario)
