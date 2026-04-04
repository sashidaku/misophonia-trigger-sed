import numpy as np
import pandas as pd

def fill_short_gaps(bin_NTC: np.ndarray,
                    max_gap_frames: int) -> np.ndarray:
    N, T, C = bin_NTC.shape
    out = bin_NTC.copy()

    for n in range(N):
        for c in range(C):
            arr = out[n, :, c]
            diff = np.diff(np.concatenate([[0], arr, [0]]))
            starts = np.where(diff == 1)[0]
            ends   = np.where(diff == -1)[0]

            for (s1, e1), (s2, e2) in zip(zip(starts, ends), zip(starts[1:], ends[1:])):
                gap_start = e1
                gap_end   = s2
                gap_len   = gap_end - gap_start
                if 0 < gap_len <= max_gap_frames:
                    out[n, gap_start:gap_end, c] = 1
    return out

def remove_short_events(bin_NTC: np.ndarray,
                        min_len_frames: int) -> np.ndarray:
    N, T, C = bin_NTC.shape
    out = bin_NTC.copy()

    for n in range(N):
        for c in range(C):
            arr = out[n, :, c]
            diff = np.diff(np.concatenate([[0], arr, [0]]))
            starts = np.where(diff == 1)[0]
            ends   = np.where(diff == -1)[0]

            for s, e in zip(starts, ends):
                length = e - s
                if length < min_len_frames:
                    out[n, s:e, c] = 0
    return out

def double_threshold_2d(
    p_NT: np.ndarray,
    th_high: float,
    th_low: float,
) -> np.ndarray:
    assert 0.0 <= th_low <= th_high <= 1.0
    N, T = p_NT.shape
    out = np.zeros((N, T), dtype=np.uint8)

    for n in range(N):
        active = False
        onset = 0
        p = p_NT[n]
        for t in range(T):
            pt = p[t]
            if (not active) and (pt >= th_high):
                active = True
                onset = t
            elif active and (pt < th_low):
                out[n, onset:t] = 1
                active = False
        if active:
            out[n, onset:T] = 1

    return out

def binary_to_event_df_single_class(
    bin_NT: np.ndarray,
    filenames: list[str],
    class_name: str,
    frames_per_sec: int,
) -> pd.DataFrame:
    rows = []
    N, T = bin_NT.shape
    for n in range(N):
        b = bin_NT[n]
        fn = filenames[n]
        t = 0
        while t < T:
            if b[t] == 1:
                onset = t
                t += 1
                while t < T and b[t] == 1:
                    t += 1
                offset = t
                rows.append({
                    "filename": fn,
                    "event_label": class_name,
                    "onset": onset / frames_per_sec,
                    "offset": offset / frames_per_sec,
                })
            else:
                t += 1
    return pd.DataFrame(rows)
