#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build DESED-style foreground bank from multiple dataset directories.
Reads config.yaml or falls back to DATASET_PATHS dictionary.

Outputs files named:
  <dataset>_<orig_stem>_<startSec>_<endSec>.wav
"""
import os, sys, re, subprocess, json
from pathlib import Path
from typing import Dict, Optional, Set
import numpy as np
import librosa, soundfile as sf
import pyloudnorm as pyln
from tqdm import tqdm
import re
from typing import Iterable, Tuple, Set

try:
    import yaml
except Exception:
    yaml = None

DATASET_PATHS: Dict[str, str] = {
    "ESC-50": "/srv/datasets/ESC50",
    "FSD50K": "/srv/datasets/FSD50K",
    "VocalSound": "/srv/datasets/VocalSound/audio_44k",
    "FOAMS": "/srv/datasets/FOAMS_processed_audio_flac",
    "MATA": "/srv/datasets/MATA/audio_wav32k"
}
CUSTOM_INPUT_DIR = "./"

OUT_ROOT_DEFAULT = "/srv/datasets/misophonia_byhuman"
SR_DEFAULT = 16000
TARGET_SR = 32000
TARGET_LUFS = -26.0
MIN_EVENT_SEC = 0.25
MERGE_GAP_SEC = 0.15
MICRO_BURST_CLASSES = {"clock", "typing", "sniffing"}
CANONICAL = ["chewing","sniffing","throat_clearing","coughing","clock","breathing","typing"]

NAME2CANON = {
    "chewing, mastication": "chewing", "chewing":"chewing",
    "sniff":"sniffing","sniffing":"sniffing",
    "throat clearing":"throat_clearing","throatclearing":"throat_clearing","clearing_throat":"throat_clearing",
    "cough":"coughing","coughing":"coughing",
    "tick-tock":"clock","ticking clock":"clock","clock tick":"clock","clock":"clock",
    "breathing":"breathing","typing":"typing","keyboard typing":"typing",
    "human_breathing":"breathing",
    "sneeze": "sneeze",
    "laughter" : "laughter",
    "sigh" : "sigh"
}
ALIASES = {"chewing,mastication":"chewing, mastication","tick tock":"tick-tock"}

FILTER_RULES = {
    "chewing": {
        "mode": "only",
        "require_any": {"Chewing_and_mastication"},
        "allow_also": {"Chewing_and_mastication"},
    },
    "breathing": {
        "mode": "only",
        "require_any": {"Breathing", "Respiratory_sounds"},
        "allow_also": {"Breathing", "Respiratory_sounds"},
    },
    "clock": {
        "mode": "contains_all",
        "require_any": {"Tick-tock", "Clock", "Mechanisms"},
    },
    "typing": {
        "mode": "contains_all",
        "require_any": {"Computer_keyboard", "Typing", "Domestic_sounds_and_home_sounds"},
    },
}

# ---------------------------
# Utilities
# ---------------------------
def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def fmt_time(t: float) -> str:
    return f"{t:.3f}"

def sanitize(name: str) -> str:
    return re.sub(r'[^\w\-.]+', '-', name)

def load_config(config_path: Optional[Path]):
    cfg = {}
    if config_path and config_path.exists():
        if yaml is None:
            raise RuntimeError("pyyaml not installed but config.yaml exists. Install pyyaml or remove config file.")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    return cfg

def canonicalize(label: str):
    if not label: return None
    z = label.strip().lower()
    if z in ALIASES: z = ALIASES[z]
    return NAME2CANON.get(z, None)

# ---------------------------
# Audio helpers
# ---------------------------
def lufs_normalize(y: np.ndarray, sr: int, target_lufs: float = TARGET_LUFS):
    meter = pyln.Meter(sr)
    loud = meter.integrated_loudness(y)
    gain = target_lufs - loud
    y = y * (10 ** (gain/20))
    peak = np.max(np.abs(y))
    if peak > 0.999:
        y = y / peak * 0.999
    return y

def load_audio_any(fp: Path):
    try:
        y, s = librosa.load(str(fp), sr=None, mono=True)
        return y, s
    except Exception:
        tmp = fp.with_suffix(".tmp.wav")
        try:
            subprocess.run(["ffmpeg","-y","-i",str(fp),"-ac","1","-ar",str(tmp)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            y, s = librosa.load(str(tmp), sr=None, mono=True)
            tmp.unlink(missing_ok=True)
            return y, s
        except Exception:
            if tmp.exists(): tmp.unlink(missing_ok=True)
            raise

def trim_head_tail_with_index(y, top_db=25):
    yt, index = librosa.effects.trim(y, top_db=top_db)
    return yt, index

def intervals_from_energy(y, top_db=30):
    ivals = librosa.effects.split(y, top_db=top_db)
    return [(int(s), int(e)) for s,e in ivals]

def merge_close_intervals(intervals, sr, merge_gap_sec=MERGE_GAP_SEC):
    if not intervals: return []
    merged = [intervals[0]]
    thr = int(merge_gap_sec * sr)
    for s,e in intervals[1:]:
        ps,pe = merged[-1]
        if s - pe < thr:
            merged[-1] = (ps, e)
        else:
            merged.append((s,e))
    return merged

def pack_micro_bursts(intervals, sr, min_len_sec=MIN_EVENT_SEC, max_bridge_gap_sec=0.15):
    if not intervals: return []
    out, cur = [], None
    max_gap = int(max_bridge_gap_sec*sr)
    min_len = int(min_len_sec*sr)
    for s,e in intervals:
        if cur is None:
            cur = [s,e]
        else:
            if s - cur[1] <= max_gap:
                cur[1] = e
            else:
                if cur[1]-cur[0] >= min_len:
                    out.append(tuple(cur))
                cur = [s,e]
    if cur is not None and cur[1]-cur[0] >= min_len:
        out.append(tuple(cur))
    return out

def save_event_named(y, sr, out_dir: Path, dataset_name: str, orig_stem: str, start_sec: float, end_sec: float):
    safe_mkdir(out_dir)
    y = lufs_normalize(y, sr)
    base = f"{sanitize(dataset_name)}_{sanitize(orig_stem)}_{fmt_time(start_sec)}_{fmt_time(end_sec)}.wav"
    out_fp = out_dir / base
    if out_fp.exists():
        i = 1
        while True:
            alt = out_dir / f"{out_fp.stem}__{i}{out_fp.suffix}"
            if not alt.exists():
                out_fp = alt; break
            i += 1
    sf.write(str(out_fp), y, sr)

def save_16k(seg_y, orig_sr, out_dir: Path, dataset_name: str, orig_stem: str, start_sec: float, end_sec: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    if orig_sr != TARGET_SR:
        seg_rs = librosa.resample(seg_y, orig_sr=orig_sr, target_sr=TARGET_SR, res_type='soxr_hq')
    else:
        seg_rs = seg_y
    # ← ここでの LUFS 正規化を削除（そのまま保存）
    base = f"{sanitize(dataset_name)}_{sanitize(orig_stem)}_{fmt_time(start_sec)}_{fmt_time(end_sec)}_sr{TARGET_SR}.wav"
    out_fp = out_dir / base
    sf.write(str(out_fp), seg_rs, TARGET_SR)

# ---------------------------
# Dataset iterators (lightweight heuristics)
# ---------------------------
def iter_ESC50(root: Path):
    meta = root / "esc50.csv"
    if not meta.exists(): return
    import pandas as pd
    df = pd.read_csv(meta)
    for _,row in df.iterrows():
        cat = str(row["category"]).strip().lower()
        c = canonicalize(cat)
        if c is None:
            if "clock_tick" in cat: c="clock"
            elif "keyboard_typing" in cat: c="typing"
            elif cat in ("breathing","coughing"): c=cat
        if c in CANONICAL:
            wav = root / "audio" / "audio" / "44100" / row["filename"]
            if wav.exists():
                yield ("ESC-50", c, wav)

def _parse_labels(labels_cell):
    if isinstance(labels_cell, str):
        return set(p.strip() for p in re.split(r"[;,]", labels_cell) if p.strip())
    return set()

def _match_rule(label_set: set[str], canon_label: str) -> bool:
    rule = FILTER_RULES[canon_label]
    req = rule["require_any"]
    if rule["mode"] == "contains_all":
        return req.issubset(label_set)
    elif rule["mode"] == "only":
        if len(label_set & req) == 0:
            return False
        allow = rule["allow_also"]
        return label_set.issubset(allow)
    else:
        return False

def iter_FSD50K(root: Path, splits=("dev", "eval"), target=("clock", "typing", "breathing", "chewing",)):
    gt_dir = root / "FSD50K.ground_truth"
    import pandas as pd

    seen = set()
    gt_dir = root / "FSD50K.ground_truth"
    entries = []
    if "dev" in splits:
        entries.append(("dev", gt_dir / "dev.csv", root / "FSD50K.dev_audio"))
    if "eval" in splits:
        entries.append(("eval", gt_dir / "eval.csv", root / "FSD50K.eval_audio"))

    target = set(target)
    seen = set()

    for split_name, csvf, aud_dir in entries:
        if not csvf.exists():
            continue

        df = pd.read_csv(csvf)
        if "fname" not in df.columns or "labels" not in df.columns:
            continue
        
        for _, row in df.iterrows():
            fname = str(row["fname"])
            label_set = _parse_labels(row["labels"])
            if not fname or not label_set:
                continue

            for canon in target:
                if _match_rule(label_set, canon):
                    wav = aud_dir / f"{fname}.wav"
                    if not wav.exists() and fallback_glob:
                        cands = list(root.rglob(f"{fname}.wav"))
                        if cands: wav = cands[0]
                    key = (fname, canon)
                    if wav.exists() and key not in seen:
                        seen.add(key)
                        yield ("FSD50K", canon, wav)

def load_split_ids(csv_path: Path) -> Set[str]:
    ids: Set[str] = set()
    if not csv_path.exists():
        raise FileNotFoundError(f"Meta CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            first_col = line.split(",")[0].strip()
            if first_col:
                ids.add(first_col)
    return ids


def iter_VocalSound(
    root: Path,
    split: Optional[str] = None,
    target=("sniff", "cough", "throatclearing"),
) -> Iterable[Tuple[str, str, Path]]:
    candidates = [
        root / "data_44k",
    ]
    audio_dir = None
    for c in candidates:
        if c.exists():
            audio_dir = c
            break
    if audio_dir is None:
        return
    speaker_ids: Optional[Set[str]] = None
    if split is not None:
        meta_dir = root / "meta"
        csv_path = meta_dir / f"{split}.csv"
        speaker_ids = load_split_ids(csv_path)

    for fp in audio_dir.glob("*.wav"):
        stem_tokens = fp.stem.split("_")
        if len(stem_tokens) < 2:
            continue

        speaker_id = stem_tokens[0]
        last_token = stem_tokens[-1]

        if speaker_ids is not None and speaker_id not in speaker_ids:
            continue

        if last_token in target:
            c = canonicalize(last_token)
            if c is None:
                continue
            yield ("VocalSound", c, fp)

from pathlib import Path
from typing import Iterable, Tuple, Generator, Set

def iter_FOAMS(
    root: Path,
    target=("human_breathing", "typing", "clearing_throat"),
    fallback_glob: bool = True,
) -> Iterable[Tuple[str, str, Path]]:
    import pandas as pd
    import re

    csv_path = root / "segmentation_info.csv"
    audio_dir = root / "FOAMS_processed_audio"

    df = pd.read_csv(csv_path)
    seen_fnames: Set[str] = set()

    for _, row in df.iterrows():
        fname = str(row.get("id", "")).strip()
        if not fname or fname in seen_fnames:
            continue

        labels_raw = row.get("label", "")

        if labels_raw in target:
            wav = audio_dir / f"{fname}_processed.wav"
            if not wav.exists() and fallback_glob:
                cands = list(root.rglob(f"{fname}_processed.wav"))
                if cands:
                    wav = cands[0]
            if not wav.exists():
                continue
            if labels_raw == "human_breathing": matched = "breathing"
            elif labels_raw == "clearing_throat": matched = "throat_clearing"
            elif labels_raw == "typing": matched = "typing"
            print(matched)
            seen_fnames.add(fname)
            yield ("FOAMS", matched, wav)


def iter_MATA(
    root: Path,
    target=("chewing", "sniffing", "clearing_throat", "coughing", "breathing", "typing"),
) -> Iterable[Tuple[str, str, Path]]:
    root = Path(root)
    seen: Set[Tuple[str, str]] = set()

    def norm_path(p: Path) -> str:
        return f"/{p.as_posix().lower().strip('/')}/"

    def contains(path_lc: str, *segs: str) -> bool:
        return all(f"/{s.lower()}/" in path_lc for s in segs)

    def contains_any(path_lc: str, *segs: str) -> bool:
        return any(f"/{s.lower()}/" in path_lc for s in segs)

    def chewing_rule(path_lc: str) -> bool:
        if not contains(path_lc, "mouth sounds (eating)"):
            return False
        ok = contains_any(path_lc, "biting & chewing", "chewing")
        if not ok:
            return False
        if contains_any(path_lc, "popcorn", "gum"):
            return False
        if contains(path_lc, "mouth sounds (eating)", "biting") and not contains(path_lc, "biting & chewing"):
            return False
        return True

    def sniffing_rule(path_lc: str) -> bool:
        if not (contains(path_lc, "nasal-throat sounds") and contains(path_lc, "sniffling")):
            return False
        return not contains(path_lc, "snorting")

    def clearing_throat_rule(path_lc: str) -> bool:
        return contains(path_lc, "nasal-throat sounds") and contains(path_lc, "clearing throat")

    def coughing_rule(path_lc: str) -> bool:
        return contains(path_lc, "nasal-throat sounds") and contains(path_lc, "coughing")

    def breathing_rule(path_lc: str) -> bool:
        if not (contains(path_lc, "nasal-throat sounds") and contains(path_lc, "breathing")):
            return False
        return not contains(path_lc, "snoring")

    def typing_rule(path_lc: str) -> bool:
        return contains(path_lc, "repetitive & continuous sounds (human)") and contains(path_lc, "typing")

    RULES = [
        ("chewing", chewing_rule),
        ("sniffing", sniffing_rule),
        ("clearing_throat", clearing_throat_rule),
        ("coughing", coughing_rule),
        ("breathing", breathing_rule),
        ("typing", typing_rule),
    ]

    for wav in root.rglob("*.wav"):
        rel = wav.relative_to(root)
        path_lc = norm_path(rel)

        matched = None
        for canon, rule in RULES:
            if canon not in target:
                continue
            if rule(path_lc):
                matched = canon
                break

        if matched is None:
            continue

        key = (str(rel), matched)
        if key in seen:
            continue
        seen.add(key)
        yield ("MATA", matched, wav)

def iter_simple_directory(
    root: Path
) -> Iterable[Tuple[str, str, str, Path]]:
    exts=(".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg")
    print(f"Scanning directory: {root}")
    extset = {e.lower() for e in exts}
    
    for fp in root.rglob("*"):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in extset:
            continue
            
        label_name = fp.parent.name
        canon_label = canonicalize(label_name)
        
        if canon_label not in CANONICAL:
            continue

        filename_stem = fp.stem
        
        parts = filename_stem.split('_')
        
        if len(parts) < 2:
            print(f"[Warn] Skipping file with unexpected name format: {fp.name}")
            continue
            
        dataset_name = parts[0]
        original_stem = parts[1]
        
        yield (dataset_name, canon_label, original_stem, fp)

# ---------------------------
# Core processing
# ---------------------------
def process_file(dataset_name, canon_label, 
    src_fp: Path, out_root: Path):
    try:
        y, orig_sr = load_audio_any(src_fp)
    except Exception as e:
      import traceback
      print(f"[load_audio_any failed] {src_fp} -> {type(e).__name__}: {e}")
      traceback.print_exc()
      return 0

    yt, (idx0, idx1) = trim_head_tail_with_index(y, top_db=25)
    if yt.size == 0:
      return 0

    out_dir = out_root / canon_label
    orig_stem = src_fp.stem
    num = 0

    if canon_label == "clock":
      s_abs = idx0
      e_abs = idx1
      start_sec = s_abs / orig_sr
      end_sec = e_abs / orig_sr
      seg = y[s_abs:e_abs]
      save_16k(seg, orig_sr, out_dir, dataset_name, orig_stem, start_sec, end_sec)
      return 1
    
    ivals = intervals_from_energy(yt, top_db=30)
    ivals = merge_close_intervals(ivals, orig_sr, MERGE_GAP_SEC)
    if canon_label in MICRO_BURST_CLASSES:
        ivals = pack_micro_bursts(ivals, orig_sr, min_len_sec=MIN_EVENT_SEC, max_bridge_gap_sec=0.15)
    segs = []
    segs = [(s, e) for (s, e) in ivals if (e - s)/orig_sr >= MIN_EVENT_SEC]
    if not segs: 
      return 0

    for (s_t,e_t) in segs:
        s_abs = idx0 + s_t
        e_abs = idx0 + e_t
        start_sec = s_abs / orig_sr
        end_sec = e_abs / orig_sr
        seg = y[s_abs:e_abs]
        save_16k(seg, orig_sr, out_dir, dataset_name, orig_stem, start_sec, end_sec)
        num += 1
    return num

def process_file_perfolder(
    dataset_name,
    canon_label,
    orig_stem: str,
    src_fp: Path, out_root: Path):
    try:
        y, orig_sr = load_audio_any(src_fp)
    except Exception as e:
      import traceback
      print(f"[load_audio_any failed] {src_fp} -> {type(e).__name__}: {e}")
      traceback.print_exc()
      return 0

    yt, (idx0, idx1) = trim_head_tail_with_index(y, top_db=25)
    if yt.size == 0:
      return 0

    out_dir = out_root / canon_label
    num = 0
    
    ivals = intervals_from_energy(yt, top_db=30)
    ivals = merge_close_intervals(ivals, orig_sr, MERGE_GAP_SEC)
    if canon_label in MICRO_BURST_CLASSES:
        ivals = pack_micro_bursts(ivals, orig_sr, min_len_sec=MIN_EVENT_SEC, max_bridge_gap_sec=0.15)
    segs = []
    for (s, e) in ivals:
        if (e - s) / orig_sr >= MIN_EVENT_SEC:
            segs.append((s, e))
        else:
            if src_fp == Path("/srv/datasets/misophonia_32k/clock/clock/FSD50K_16651_0.000_14.545_sr32000.wav"):
                print(e/orig_sr, s/orig_sr)
                print(src_fp, (e - s) / orig_sr)
    if not segs: 
      return 0

    for (s_t,e_t) in segs:
        s_abs = idx0 + s_t
        e_abs = idx0 + e_t
        start_sec = s_abs / orig_sr
        end_sec = e_abs / orig_sr
        seg = y[s_abs:e_abs]
        save_16k(seg, orig_sr, out_dir, dataset_name, orig_stem, start_sec, end_sec)
        num += 1
    return num

def main_perdatasets():
    cfg = load_config(Path("config.yaml"))
    out_root = Path(cfg.get("out_root", OUT_ROOT_DEFAULT)) if isinstance(cfg, dict) else Path(OUT_ROOT_DEFAULT)
    sr = int(cfg.get("sr", SR_DEFAULT)) if isinstance(cfg, dict) else SR_DEFAULT
    datasets = cfg.get("datasets", DATASET_PATHS) if isinstance(cfg, dict) else DATASET_PATHS
    enabled = cfg.get("enabled", list(datasets.keys())) if isinstance(cfg, dict) else list(datasets.keys())

    total = 0

    for ds_name, ds_path in datasets.items():
        if ds_name not in enabled:
            continue
        root = Path(ds_path)
        if not root.exists():
            print(f"[warn] dataset path not found: {ds_name} -> {ds_path}", file=sys.stderr)
            continue

        if ds_name.lower().startswith("vocal") or "vocal" in ds_name.lower():
            for split in ("tr_meta", "val_meta", "te_meta"):
                it = iter_VocalSound(root, split=split)
                split_out_root = out_root / split

                for dataset_name, c, fp in tqdm(
                    it, desc=f"Processing {ds_name} [{split}]", unit="file"
                ):
                    total += process_file(dataset_name, c, fp, split_out_root)
            continue
        it = None
        # if ds_name.lower().startswith("esc"):
        #     it = iter_ESC50(root)
        # if ds_name.lower().startswith("fsd"):
        #     it = iter_FSD50K(root)
        # if "foam" in ds_name.lower():
        #     it = iter_FOAMS(root)
        # if "mata" in ds_name.lower():
        #     it = iter_MATA(root)

        if it is None:
            continue

        for dataset_name, c, fp in tqdm(it, desc=f"Processing {ds_name}", unit="file"):
            total += process_file(dataset_name, c, fp, out_root)

    print(f"[done] exported {total} events into {out_root}/<split>/<class>/ (VocalSound only split)")

def main_perfolder():
    cfg = load_config(Path("config.yaml"))
    
    in_root = Path(cfg.get("in_root", CUSTOM_INPUT_DIR)) if isinstance(cfg, dict) else Path(CUSTOM_INPUT_DIR)
    out_root = Path(cfg.get("out_root", OUT_ROOT_DEFAULT)) if isinstance(cfg, dict) else Path(OUT_ROOT_DEFAULT)

    if not in_root.exists():
        print(f"[ERROR] Input directory not found: {in_root}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Input directory: {in_root}")
    print(f"Output directory: {out_root}")

    total = 0

    it = iter_simple_directory(in_root) 
    
    for ds_name, c, stem, fp in tqdm(it, desc=f"Processing {in_root.name}", unit="file"):
        total += process_file_perfolder(ds_name, c, stem, fp, Path(out_root))

    print(f"[done] exported {total} events into {out_root}/<class>/")
    
if __name__ == "__main__":
    main_perdatasets()
