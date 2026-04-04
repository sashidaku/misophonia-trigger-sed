from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Optional, Union, Iterable
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import ConcatDataset, Subset, Dataset
from third_party.EfficientSED.dcase2016task2 import get_labels_for_timestamps, label_to_binary_vector, label_vocab_as_dict, label_vocab_nlabels

def tsv_to_event_dict(tsv_path: Path) -> Dict[str, List[dict]]:
    df = pd.read_csv(tsv_path, sep="\t")
    data: Dict[str, List[dict]] = {}
    for _, row in df.iterrows():
        fname = str(row["filename"])
        if fname not in data:
            data[fname] = []
        data[fname].append({
            "start": float(row["onset"]),
            "end": float(row["offset"]),
            "label": str(row["event_label"]),
        })
    return data


class TenSecondSEDDataset(Dataset):
    """
    Dataset for 10-second sound event detection clips.
    Adapted in part from the FixCropDataset design used in EfficientSED.
    """
    def __init__(
        self,
        data: Dict[str, List[dict]],
        audio_dir: Path,
        sample_rate: int,
        label_fps: int,
        label_to_idx: Dict[str, int],
        nlabels: int,
        target_classes= None,
        return_only_target: bool = False
    ):
        self.data = data
        self.clip_len = 10
        self.target_len = 10
        self.target_set = target_classes

        self.return_only_target = return_only_target
        self.pieces_per_clip = self.clip_len // self.target_len
        
        if target_classes is not None:
            self.target_list = list(dict.fromkeys(target_classes))
            self.trig_to_newidx = {c: i for i, c in enumerate(self.target_list)}
            self.out_nlabels = len(self.target_list)
        else:
            self.target_list = None
            self.trig_to_newidx = None
            self.out_nlabels = nlabels

        self.audio_dir = Path(audio_dir)
        assert self.audio_dir.is_dir(), f"{audio_dir} is not a directory"
        self.sample_rate = sample_rate
        self.label_fps = label_fps
        self.label_to_idx = label_to_idx
        self.nlabels = nlabels
        self.filenames = list(data.keys())
        self.pieces = []
        self.labels = []
        self.timestamps = []

        for filename in self.filenames:
            self.pieces += [(filename, i) for i in range(self.pieces_per_clip)]

            events = data[filename]

            frame_len = 1.0 / label_fps
            n_frames = int(label_fps * self.clip_len)
            timestamps = np.arange(n_frames) * frame_len + 0.5 * frame_len

            timestamp_labels = get_labels_for_timestamps(events, timestamps)

            ys = []
            for lbls_at_t in timestamp_labels:
                if self.trig_to_newidx is None:
                    idxs = [label_to_idx[str(ev)] for ev in lbls_at_t]
                    y_t = label_to_binary_vector(idxs, nlabels)
                else:
                    idxs = [self.trig_to_newidx[ev] for ev in lbls_at_t if ev in self.trig_to_newidx]
                    y_t = label_to_binary_vector(idxs, self.out_nlabels)
                ys.append(y_t)
            ys = torch.stack(ys)

            frames_per_clip = ys.size(0) // self.pieces_per_clip  
            self.labels += [
                ys[frames_per_clip * i: frames_per_clip * (i + 1)]
                for i in range(self.pieces_per_clip)
            ]
            self.timestamps += [
                timestamps[frames_per_clip * i: frames_per_clip * (i + 1)]
                for i in range(self.pieces_per_clip)
            ]

        assert len(self.labels) == len(self.pieces) == len(self.filenames) * self.pieces_per_clip

    def __len__(self):
        return len(self.pieces)

    def __getitem__(self, idx):
        filename, piece = self.pieces[idx]
        audio_path = self.audio_dir / filename
        audio, sr = sf.read(str(audio_path), dtype=np.float32)
        assert sr == self.sample_rate, f"sr mismatch: {sr} vs {self.sample_rate}"

        start = int(self.sample_rate * piece * self.target_len)
        end   = start + int(self.sample_rate * self.target_len)
        
        if len(audio) < end:
            pad = end - len(audio)
            audio = np.pad(audio, (0, pad))
        else:
            audio = audio[start:end]

        labels = self.labels[idx].transpose(0, 1) 
        ts = self.timestamps[idx]
        file_id = f"{self.audio_dir.name}/{filename}"
        return audio, labels, file_id, ts
    
def select_fnames_kshot(
    train_df: pd.DataFrame,
    classes: list[str],
    k_shot: int = 5,
    seed: int = 0,
    require_single_label: bool = False,
) -> set[str]:
    """
    Select k-shot filenames for each target class from the training DataFrame.

    This function samples up to ``k_shot`` unique filenames per class from
    ``train_df``. If ``require_single_label`` is True, only files containing
    events from a single class are considered. When a class has fewer than
    ``k_shot`` candidate files, sampling is done with replacement.

    Returns:
        A set of sampled filenames across all target classes.
    """
    classes = list(dict.fromkeys(classes))
    labelsbyfile = defaultdict(set)
    for r in train_df.itertuples():
        labelsbyfile[r.filename].add(str(r.event_label))

    rng = np.random.default_rng(seed)
    selected_filenames = set()

    for c in classes:
        candidate_filenames = train_df.loc[train_df["event_label"] == c, "filename"].unique().tolist()

        if require_single_label:
            candidate_filenames = [f for f in candidate_filenames if labelsbyfile[f] <= {c}]

        if len(candidate_filenames) < k_shot:
            chosen = rng.choice(candidate_filenames, size=k_shot, replace=True).tolist()
        else:
            chosen = rng.choice(candidate_filenames, size=k_shot, replace=False).tolist()

        selected_filenames.update(chosen)

    return selected_filenames

def get_training_dataset(
        audio_dir,
        metadata_dir,
        background_tsv_path='/srv/datasets/misophonia_32k/synthesis_metadata/soundscapes/onlybackground.tsv',
        background_audio_path='/srv/datasets/misophonia_32k/synthesis_audio/soundscapes_16k/onlybackground',
        sample_rate=32000,
        label_fps=25,
        
        target_classes=None,
        max_files_per_class=6000,
        seed=0,
        max_files=6000,
        with_bg_only=False,
        bg_only_ratio=0.10,

        fewshot_k = None,
        fewshot_seed: int = 0,
        fewshot_single_label: bool = False,

        include_fnames: Optional[Sequence[str]] = None,
        include_fnames_txt: Optional[Union[str, Path]] = None,
        strict_include: bool = False,
):
    """
    Build and return the training dataset.

    This function creates a training dataset from the main training annotations and audio files.
    It can optionally:
    - keep only specified target classes,
    - apply few-shot sampling,
    - restrict the dataset to a given set of filenames,
    - limit the total number of clips, and
    - mix in background-only clips at a specified ratio.

    Returns:
        The constructed training dataset.
    """
    rng = np.random.default_rng(seed)

    if fewshot_k is not None and not target_classes:
        raise ValueError("fewshot_k を使う場合は target_classes を指定してください")

    if max_files is not None and max_files <= 0:
        raise ValueError("max_files must be positive")

    if not (0.0 <= bg_only_ratio <= 1.0):
        raise ValueError("bg_only_ratio must be in [0, 1]")

    if fewshot_k is not None and fewshot_k <= 0:
        raise ValueError("fewshot_k must be positive")

    if max_files_per_class <= 0:
        raise ValueError("max_files_per_class must be positive")
    
    metadata_dir = Path(metadata_dir)
    audio_dir = Path(audio_dir)
    label_vocab_df, nlabels = label_vocab_nlabels(metadata_dir)
    
    label_to_idx = label_vocab_as_dict(label_vocab_df, key="label", value="idx")

    train_tsv_path = metadata_dir / "train.tsv"
    audio_dir_main = audio_dir / "train"
    train_event_dict = tsv_to_event_dict(train_tsv_path)
 
    if target_classes:
        target_set = list(dict.fromkeys(target_classes))
        train_df = pd.read_csv(train_tsv_path, sep="\t")
        if fewshot_k is not None:
            selected_filenames = select_fnames_kshot(
                str(train_tsv_path),
                classes=target_set,
                k_shot=fewshot_k,
                seed=fewshot_seed,
                require_single_label=fewshot_single_label,
            )
            train_event_dict = {f: evs for f, evs in train_event_dict.items() if f in selected_filenames}
            print(f"[fewshot] support files = {len(train_event_dict)} (target {len(target_set)}-way, {fewshot_k}-shot)")
        else:
            # Keep whole files that contain at least one target-class event.
            selected_filenames = set()
            for c in target_set:
                class_filenames = train_df.loc[train_df["event_label"] == c, "filename"].unique()
                if len(class_filenames) > max_files_per_class:
                    sampled_indices = rng.choice(len(class_filenames), size=max_files_per_class, replace=False)
                    class_filenames = class_filenames[sampled_indices]
                selected_filenames.update(class_filenames)
            train_event_dict = {
                f: evs for f, evs in train_event_dict.items() if f in selected_filenames
            }

    if include_fnames_txt is not None:
        p = Path(include_fnames_txt)
        lines = [ln.strip() for ln in p.read_text().splitlines()]
        lines = [ln for ln in lines if ln and not ln.startswith("#")]
        include_fnames = lines if include_fnames is None else list(include_fnames) + lines

    if include_fnames is not None:
        print(include_fnames)
        seen = set()
        include_list = []
        for f in include_fnames:
            f = str(f)
            if f not in seen:
                seen.add(f)
                include_list.append(f)

        keys = set(train_event_dict.keys())
        missing = [f for f in include_list if f not in keys]
        if missing:
            msg = f"[get_training_dataset] include_fnames: {len(missing)} files not found in train_event_dict. examples={missing[:5]}"
            if strict_include:
                raise ValueError(msg)
            else:
                print("WARNING", msg)

        train_event_dict = {f: train_event_dict[f] for f in include_list if f in train_event_dict}
        print(f"[get_training_dataset] using ONLY include_fnames: {len(train_event_dict)} files")
        
    train_ds = TenSecondSEDDataset(
        train_event_dict,
        audio_dir_main,
        sample_rate,
        label_fps,
        label_to_idx,
        nlabels,
        target_classes=target_classes,
        return_only_target=False
    )
    train_dataset = train_ds

    background_ds = None
    background_tsv_path = Path(background_tsv_path)
    background_audio_path = Path(background_audio_path)
    if with_bg_only and background_tsv_path.exists() and background_audio_path.exists():
        data_bg = tsv_to_event_dict(background_tsv_path)
        background_ds = TenSecondSEDDataset(
            data_bg,
            background_audio_path,
            sample_rate,
            label_fps,
            label_to_idx,
            nlabels
        )
    if background_ds is not None and bg_only_ratio > 0.0:
        if max_files is None:
            n_main = len(train_ds)
            n_bg_target = int(n_main * bg_only_ratio / (1.0 - bg_only_ratio))
        else:
            n_bg_target = int(max_files * bg_only_ratio)
            n_main = max_files - n_bg_target
            n_main = min(n_main, len(train_ds))
            n_bg_target = min(n_bg_target, len(background_ds))

        main_indices = rng.permutation(len(train_ds))[:n_main]
        bg_indices = rng.permutation(len(background_ds))[:n_bg_target]

        main_sub = Subset(train_ds, main_indices.tolist())
        bg_sub   = Subset(background_ds, bg_indices.tolist())

        train_dataset = ConcatDataset([main_sub, bg_sub])
    elif max_files is not None:
        num_clips = min(max_files, len(train_ds))
        indices = rng.permutation(len(train_ds))[:num_clips]
        train_dataset = Subset(train_ds, indices.tolist())

    event_dict = dataset_to_event_dict(train_dataset)
    stats_tr = summarize_event_dict(event_dict, classes=target_classes)
    print("[train] class-wise stats\n", stats_tr)
    print("[train] total_clips =", len(train_dataset), " total_events =", sum(len(v) for v in event_dict.values()))
    return train_dataset

def get_validation_dataset(
        audio_dir,
        metadata_dir,
        sample_rate=32000,
        label_fps=25,
        seed=0,
        max_files=2000,
        max_files_per_class=2000,
        target_classes = None,
        use_long_chewing_filter = False,
):
    """
    Build and return the validation dataset.

    This function creates a validation dataset from the validation annotations and audio files.
    It can optionally:
    - keep only specified target classes,
    - limit the number of files per class,
    - apply a special duration-based filter for selected classes, and
    - limit the total number of clips.

    Returns:
        The constructed validation dataset.
    """
    if max_files is not None and max_files <= 0:
        raise ValueError("max_files must be positive")

    if max_files_per_class is not None and max_files_per_class <= 0:
        raise ValueError("max_files_per_class must be positive")
    
    rng = np.random.default_rng(seed)
    metadata_dir = Path(metadata_dir)
    audio_dir = Path(audio_dir)

    label_vocab, nlabels = label_vocab_nlabels(metadata_dir)
    label_to_idx = label_vocab_as_dict(label_vocab, key="label", value="idx")

    tsv_path = metadata_dir / "test.tsv"
    audio_dir = audio_dir / "test"

    data = tsv_to_event_dict(tsv_path)

    if target_classes:
        df = pd.read_csv(tsv_path, sep="\t")
        target_set = list(dict.fromkeys(target_classes))

        rng = np.random.default_rng(seed)

        keep_fnames = set()
        added_counts = {}
        for c in target_set:
            if c == 'chewing':
                subset = df[df["event_label"] == c].copy()
                subset["duration"] = subset["offset"] - subset["onset"]
                long_event_df = subset[subset["duration"] >= 2.0]
                
                fnames_all = set(long_event_df["filename"].unique())
                fnames_new = list(fnames_all - keep_fnames)
                
                selected = fnames_new
                print(f"[Special Filter] Class '{c}': Found {len(selected)} files with duration >= 2.0s (Ignoring max_files limit)")
            else:
                fnames_all = set(df.loc[df["event_label"] == c, "filename"].unique())  # ユニークはpandasが保証 :contentReference[oaicite:2]{index=2}

                fnames_new = list(fnames_all - keep_fnames)

                if len(fnames_new) == 0:
                    added_counts[c] = 0
                    continue

                if len(fnames_new) > max_files:
                    idx = rng.choice(len(fnames_new), size=max_files, replace=False)
                    selected = [fnames_new[i] for i in idx]
                else:
                    selected = fnames_new

            keep_fnames.update(selected)
            added_counts[c] = len(selected)
            if c != 'chewing':
                if max_files is not None and len(keep_fnames) > max_files:
                    rng = np.random.default_rng(0)
                    keep_list = np.array(list(keep_fnames))
                    keep_fnames = set(rng.choice(keep_list, size=max_files, replace=False))
        print(
            f"[get_test_dataset] filter by classes={target_set}: "
            f"{len(keep_fnames)} unique files kept (max_files per class = {max_files})"
        )

        data = {fname: events for fname, events in data.items()
                if fname in keep_fnames}

    dataset = TenSecondSEDDataset(
        data, audio_dir, sample_rate, label_fps, label_to_idx, nlabels, target_classes=target_classes
    )

    is_chewing_target = (target_classes is not None) and ('chewing' in target_classes)

    if max_files is not None and not target_classes and not is_chewing_target:
        n = min(max_files, len(dataset))
        indices = np.arange(n)
        dataset = Subset(dataset, indices.tolist())
    event_dict = dataset_to_event_dict(dataset)
    stats = summarize_event_dict(event_dict, classes=target_classes)
    print("[val] class-wise stats\n", stats)
    print("[val] total_clips =", len(dataset), " total_events =", sum(len(v) for v in event_dict.values()))
    return dataset

def get_test_dataset(
        audio_dir,
        metadata_dir,
        sample_rate=32000,
        label_fps=25,
        seed=0,
        max_files=2000,
        max_files_per_class=2000,
        target_classes = None,
        use_long_chewing_filter = False,
):
    """
    Build and return the test dataset.

    This function creates a test dataset from the test annotations and audio files.
    It can optionally:
    - keep only specified target classes,
    - limit the number of files per class,
    - apply a special duration-based filter for selected classes, and
    - limit the total number of clips.

    Returns:
        The constructed test dataset.
    """
    if max_files is not None and max_files <= 0:
        raise ValueError("max_files must be positive")

    if max_files_per_class is not None and max_files_per_class <= 0:
        raise ValueError("max_files_per_class must be positive")
    
    rng = np.random.default_rng(seed)
    metadata_dir = Path(metadata_dir)
    audio_dir = Path(audio_dir)

    label_vocab, nlabels = label_vocab_nlabels(metadata_dir)
    label_to_idx = label_vocab_as_dict(label_vocab, key="label", value="idx")

    tsv_path = metadata_dir / "eval.tsv"
    audio_dir = audio_dir / "eval"

    data = tsv_to_event_dict(tsv_path)

    if target_classes:
        df = pd.read_csv(tsv_path, sep="\t")
        target_set = list(dict.fromkeys(target_classes))

        rng = np.random.default_rng(seed) 
        keep_fnames = set()
        added_counts = {}
        for c in target_set:
            if c == 'chewing':
                subset = df[df["event_label"] == c].copy()
                subset["duration"] = subset["offset"] - subset["onset"]
                long_event_df = subset[subset["duration"] >= 2.0]
                
                fnames_all = set(long_event_df["filename"].unique())
                fnames_new = list(fnames_all - keep_fnames)
                
                selected = fnames_new
                print(f"[Special Filter] Class '{c}': Found {len(selected)} files with duration >= 2.0s (Ignoring max_files limit)")
            else:
                fnames_all = set(df.loc[df["event_label"] == c, "filename"].unique())
                fnames_new = list(fnames_all - keep_fnames)

                if len(fnames_new) == 0:
                    added_counts[c] = 0
                    continue
                if len(fnames_new) > max_files:
                    idx = rng.choice(len(fnames_new), size=max_files, replace=False)
                    selected = [fnames_new[i] for i in idx]
                else:
                    selected = fnames_new

            keep_fnames.update(selected)
            added_counts[c] = len(selected)
            if c != 'chewing':
                if max_files is not None and len(keep_fnames) > max_files:
                    rng = np.random.default_rng(0)
                    keep_list = np.array(list(keep_fnames))
                    keep_fnames = set(rng.choice(keep_list, size=max_files, replace=False))
        print(
            f"[get_test_dataset] filter by classes={target_set}: "
            f"{len(keep_fnames)} unique files kept (max_files per class = {max_files})"
        )

        # data を filename ベースでフィルタ
        data = {fname: events for fname, events in data.items()
                if fname in keep_fnames}

    dataset = TenSecondSEDDataset(
        data, audio_dir, sample_rate, label_fps, label_to_idx, nlabels, target_classes=target_classes
    )

    is_chewing_target = (target_classes is not None) and ('chewing' in target_classes)

    if max_files is not None and not target_classes and not is_chewing_target:
        n = min(max_files, len(dataset))
        indices = np.arange(n)
        dataset = Subset(dataset, indices.tolist())
    event_dict = dataset_to_event_dict(dataset)
    stats = summarize_event_dict(event_dict, classes=target_classes)
    print("[test] class-wise stats\n", stats)
    print("[test] total_clips =", len(dataset), " total_events =", sum(len(v) for v in event_dict.values()))
    return dataset


def dataset_to_event_dict(ds) -> Dict[str, List[dict]]:
    """
    Convert TenSecondSEDDataset / Subset / ConcatDataset
    into an event_dict that can be passed to summarize_event_dict.
    """
    if isinstance(ds, TenSecondSEDDataset):
        return ds.data

    if isinstance(ds, Subset):
        base = ds.dataset
        if isinstance(base, TenSecondSEDDataset):
            fnames = []
            for i in ds.indices:
                fname, _piece = base.pieces[i]
                fnames.append(fname)
            fnames = list(dict.fromkeys(fnames))
            return {f: base.data[f] for f in fnames}
        return dataset_to_event_dict(base)

    if isinstance(ds, ConcatDataset):
        merged: Dict[str, List[dict]] = {}
        for sub in ds.datasets:
            merged.update(dataset_to_event_dict(sub))
        return merged

    raise TypeError(f"Unsupported dataset type for stats: {type(ds)}")

def summarize_event_dict(
    data: Dict[str, List[dict]],
    classes: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    data: dict[filename] = [{"start": float, "end": float, "label": str}, ...]
    returns: DataFrame indexed by class with columns:
      - clips_containing_class
      - event_instances
      - mean_dur_s
      - median_dur_s
    """
    class_set = set(classes) if classes is not None else None

    cls2files = defaultdict(set)     # class -> set(filename)
    cls2n_events = defaultdict(int)  # class -> count
    cls2durs = defaultdict(list)     # class -> list(duration)

    for fname, evs in data.items():
        present = set()
        for e in evs:
            lbl = str(e["label"])
            if class_set is not None and lbl not in class_set:
                continue
            cls2n_events[lbl] += 1
            present.add(lbl)
            cls2durs[lbl].append(float(e["end"]) - float(e["start"]))
        for lbl in present:
            cls2files[lbl].add(fname)

    labels = sorted(set(cls2files.keys()) | set(cls2n_events.keys()))
    rows = []
    for lbl in labels:
        durs = np.asarray(cls2durs.get(lbl, []), dtype=float)
        rows.append({
            "class": lbl,
            "clips_containing_class": len(cls2files.get(lbl, set())),
            "event_instances": int(cls2n_events.get(lbl, 0)),
            "mean_dur_s": float(durs.mean()) if durs.size else np.nan,
            "median_dur_s": float(np.median(durs)) if durs.size else np.nan,
        })

    out = pd.DataFrame(rows).set_index("class")
    out = out.sort_values("clips_containing_class", ascending=False)
    return out