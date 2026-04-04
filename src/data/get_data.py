import csv
from pathlib import Path
import random
from typing import Dict, List, Tuple
import numpy as np
import torch
import torchaudio
from config_loader import data_cfg, esn_cfg
from src.data.misophonia import get_test_dataset, get_training_dataset, get_validation_dataset, label_vocab_nlabels
from src.models.encorders.common_audio_encorder import EncodeSpec
from third_party.EfficientSED.models.efficient_cnns.fmn.utils import NAME_TO_WIDTH
from third_party.EfficientSED.models.efficient_cnns.fmn.fmn_wrapper import FrameMNWrapper
from third_party.EfficientSED.models.transformers.frame_passt.preprocess import AugmentMelSTFT
from third_party.EfficientSED.models.prediction_wrapper import PredictionsWrapper
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def encode_seq(
    *,
    encoder,
    wave: torch.Tensor,
    use_esn: bool,
    esn=None,
    detach: bool = True,
    use_amp: bool = False,
) -> torch.Tensor:
    spec = EncodeSpec(
        out="esn_seq" if use_esn else "cnn_seq",
        detach=detach,
        use_amp=use_amp,
    )
    return encoder(wave, spec=spec, esn=esn)  

def encode_seq_single(
    *,
    encoder,
    wave: torch.Tensor,
    use_esn: bool,
    esn=None,
    detach: bool = True,
    use_amp: bool = False,
) -> torch.Tensor:
    x = encode_seq(
        encoder=encoder, wave=wave,
        use_esn=use_esn, esn=esn,
        detach=detach, use_amp=use_amp,
    )
    return x.squeeze(0)

def build_mel_and_cnn(device, fmn_name="fmn10", return_wrapper=False, NUM_CLASSES=7):
    global mel_extractor, fmn

    mel_extractor = AugmentMelSTFT(
        n_mels=128,
        sr=data_cfg.sample_rate,
        win_length=data_cfg.window_len,
        hopsize=data_cfg.hopsize,
        n_fft=data_cfg.n_fft,
        freqm=0,
        timem=0,
        fmin=0.0,
        fmax=None,
        norm=1,
        fmin_aug_range=10,
        fmax_aug_range=2000,
        fast_norm=True,
        preamp=True,
        padding="center",
        periodic_window=False,
    ).to(device)
    mel_extractor.eval()

    width = NAME_TO_WIDTH(fmn_name)
    fmn = FrameMNWrapper(width).to(device)

    if esn_cfg.cnn_type == "only_pretrain":
        print("CNN: fmn10_strong (AudioSet Strong pretrain)")
        checkpoint_name = "fmn10_strong"
        embed_dim = fmn.state_dict()["fmn.features.16.1.bias"].shape[0]

        fmn_model = PredictionsWrapper(
            fmn,
            checkpoint=checkpoint_name,
            seq_model_type=None,
            seq_model_dim=256,
            embed_dim=embed_dim,
            n_classes_strong=data_cfg.num_classes,
        )
        for p in fmn.parameters():
            p.requires_grad_(False)

        for p in fmn_model.parameters():
            p.requires_grad = False
    
        fmn.eval()
    else:
        raise ValueError(f"Unknown ONLY_PRETRAIN_CNN = {esn_cfg.cnn_type}")
    return mel_extractor, fmn

def detect_events_on_file(
    wav_path: str,
    class_names: List[str],
    W: torch.Tensor,
    b: torch.Tensor,
    use_esn: bool = False,
    esn = None,
    th: float = 0.5,
):
    wave, sr = torchaudio.load(wav_path)
    assert sr == data_cfg.sample_rate
    if wave.size(0) > 1:
        wave = wave.mean(dim=0, keepdim=True)
    wave = wave.to(device)

    if use_esn:
        feat_seq = extract_cnn_esn_embeddings(wave, esn)
    else:
        feat_seq = extract_cnn_embeddings(wave)

    logits = feat_seq @ W + b
    probs = torch.sigmoid(logits)

    probs_np = probs.detach().cpu().numpy()

    events = []
    T, C = probs_np.shape
    for c_idx, cname in enumerate(class_names):
        active = probs_np[:, c_idx] >= th
        in_event = False
        onset = 0
        for t in range(T):
            if active[t] and not in_event:
                in_event = True
                onset = t
            elif not active[t] and in_event:
                in_event = False
                offset = t
                events.append({
                    "event_label": cname,
                    "onset_frame": onset,
                    "offset_frame": offset,
                })
        if in_event:
            events.append({
                "event_label": cname,
                "onset_frame": onset,
                "offset_frame": T,
            })

    print(f"=== {wav_path} ===")
    for ev in events:
        print(ev)
    return probs_np, events

def build_multiclass_support_embeddings_event(
    support_config: Dict[str, List[Dict[str, float]]],
    use_esn: bool = False,
    esn = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    class_names = sorted(support_config.keys())
    C = len(class_names)
    cname_to_idx = {c: i for i, c in enumerate(class_names)}

    Z_list: List[torch.Tensor] = []
    Y_list: List[torch.Tensor] = []

    for cname in class_names:
        c_idx = cname_to_idx[cname]
        events = support_config[cname]

        for ev in events:
            wav_path = ev["path"]
            onset_sec = float(ev["onset"])
            offset_sec = float(ev["offset"])

            wave, sr = torchaudio.load(wav_path)
            assert sr == data_cfg.sample_rate

            if wave.size(0) > 1:
                wave = wave.mean(dim=0, keepdim=True)
            wave = wave.to(device)

            if use_esn:
                feat_seq = extract_cnn_esn_embeddings(wave, esn)
            else:
                feat_seq = extract_cnn_embeddings(wave)

            T_seq = feat_seq.size(0)
            dur_sec = wave.shape[-1] / data_cfg.sample_rate

            start_idx = int(round(onset_sec  / dur_sec * T_seq))
            end_idx   = int(round(offset_sec / dur_sec * T_seq))

            start_idx = max(0, min(start_idx, T_seq - 1))
            end_idx   = max(start_idx + 1, min(end_idx, T_seq))

            feat_event = feat_seq[start_idx:end_idx].mean(dim=0)

            Z_list.append(feat_event.cpu())

            y = torch.zeros(C, dtype=torch.float32)
            y[c_idx] = 1.0
            Y_list.append(y.cpu())

    Z_all = torch.stack(Z_list, dim=0)
    Y_all = torch.stack(Y_list, dim=0)
    return Z_all, Y_all, class_names

# def build_multiclass_support_embeddings_segment(
#     support_config: Dict[str, List[Dict[str, float]]],
#     frames_per_sec: int,
#     segment_sec: float = 2.0,
#     hop_sec: float = 0.5,
#     use_esn: bool = False,
#     esn = None,
# ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
#     from esn_cnn.fewshotsed import sliding_window_segment_embeddings
#     """
#     2秒セグメント単位で pos/neg を作る版

#     return:
#       Z_all: (N_seg_total, D)
#       Y_all: (N_seg_total, C)
#     """
#     class_names = sorted(support_config.keys())
#     num_classes = len(class_names)
#     cname_to_idx = {c: i for i, c in enumerate(class_names)}

#     Z_list = []
#     Y_list = []

#     file_events: Dict[str, List[Dict[str, float]]] = {}
#     for cname, events in support_config.items():
#         c_idx = cname_to_idx[cname]
#         for ev in events:
#             ev2 = {
#                 "c_idx": c_idx,
#                 "onset": float(ev["onset"]),
#                 "offset": float(ev["offset"]),
#                 "path": ev["path"],
#             }
#             file_events.setdefault(ev["path"], []).append(ev2)

#     for wav_path, events in file_events.items():
#         c_idx = cname_to_idx[cname]

#         wave, sr = torchaudio.load(wav_path)
#         assert sr == cfg.SAMPLE_RATE

#         if wave.size(0) > 1:
#             wave = wave.mean(dim=0, keepdim=True)
#         wave = wave.to(device)

#         # 1) CNN特徴 or CNN+ESN特徴を取得
#         if use_esn:
#             feats = extract_cnn_esn_embeddings(wave, esn)  # (T_feat, D_in)
#             feats = feats.unsqueeze(0)                     # (1, T_feat, D_in)
#         else:
#             feats = extract_cnn_embeddings(wave)           # (T_feat, D_in)
#             feats = feats.unsqueeze(0)                     # (1, T_feat, D_in)

#         # 2) 2秒セグメント埋め込み (1, N_seg, D_in)
#         seg_emb = sliding_window_segment_embeddings(
#             feats,
#             frames_per_sec=frames_per_sec,
#             segment_sec=segment_sec,
#             hop_sec=hop_sec,
#         )  # (1, N_seg, D_in)
#         seg_emb = seg_emb.squeeze(0)  # (N_seg, D_in)

#         # 3) 各セグメントが「イベント区間とどのくらい重なっているか」で pos/neg を決める
#         #    ここは設計次第だけど、一番シンプルには
#         #    「その2秒窓の中に1フレームでもchewingが含まれていれば positive」
#         #    としてラベルを作ることが多い。
#         #
#         #    そのために、CNNフレーム → 秒 への変換が必要なので、
#         #    セグメント中心時刻 or 開始・終了時刻を計算して、
#         #    events の onset/offset と重なりを見る。

#         N_seg = seg_emb.size(0)
#         dur_sec = wave.shape[-1] / cfg.SAMPLE_RATE
#         T_feat = feats.size(1)  # CNN時間長

#         # CNNフレームの時間刻み (sec)
#         frame_dt = dur_sec / T_feat

#         # セグメント開始フレーム index を hop ごとに並べる
#         win_size = int(round(segment_sec * frames_per_sec))
#         hop_size = int(round(hop_sec * frames_per_sec))

#         start_indices = list(range(0, max(T_feat - win_size + 1, 1), hop_size))
#         # N_seg と start_indices の長さは合っているはず
#         assert len(start_indices) == N_seg

#         for seg_idx, start_idx in enumerate(start_indices):
#             seg_start_sec = start_idx * frame_dt
#             seg_end_sec   = seg_start_sec + segment_sec

#             y = torch.zeros(num_classes, device=device)
#             for ev in events:
#                 c_idx = ev["c_idx"]
#                 onset, offset = ev["onset"], ev["offset"]
#                 if (seg_start_sec < offset) and (seg_end_sec > onset):
#                     y[c_idx] = 1.0

#             Z_list.append(seg_emb[seg_idx:seg_idx+1])  # (1, D_in)
#             Y_list.append(y.unsqueeze(0))        

#     Z_all = torch.cat(Z_list, dim=0) 
#     Y_all = torch.cat(Y_list, dim=0) 
#     return Z_all, Y_all, class_names

def build_support_config_from_tsv(
    audio_dir: str,
    tsv_path: str,
    target_classes: List[str],
    events_per_class: Dict[str, int],
    shuffle: bool = True,
) -> Dict[str, List[Dict[str, float]]]:
    audio_dir = Path(audio_dir)
    tsv_path = Path(tsv_path)

    tmp: Dict[str, List[Dict[str, float]]] = {c: [] for c in target_classes}

    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            label = row.get("event_label")
            if label not in target_classes:
                continue

            fname = row["filename"]
            onset = float(row["onset"])
            offset = float(row["offset"])

            wav_path = audio_dir / fname
            if not wav_path.exists():
                continue

            tmp[label].append(
                {
                    "path": str(wav_path),
                    "onset": onset,
                    "offset": offset,
                }
            )

    import random
    support_config: Dict[str, List[Dict[str, float]]] = {}

    for cname in target_classes:
        cand = tmp.get(cname, [])
        if shuffle:
            random.shuffle(cand)
        k = events_per_class.get(cname, len(cand))
        support_config[cname] = cand[:k]

        print(f"[build_support_config_from_tsv] class={cname}, "
              f"available={len(cand)}, used={len(support_config[cname])}")

    for cname, events in support_config.items():
        for ev in events:
            print(
                f"class={cname}\t"
                f"path={ev['path']}\t"
                f"onset={ev['onset']:.3f}\t"
                f"offset={ev['offset']:.3f}"
            )
    return support_config

def build_multiclass_support_embeddings(
    support_config: Dict[str, List[Dict[str, float]]],
    use_esn: bool = False,
    esn = None,
    neg_ratio: float = 4.0,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    class_names = sorted(support_config.keys())
    num_classes = len(class_names)
    cname_to_idx = {c: i for i, c in enumerate(class_names)}

    Z_list = []
    Y_list = []

    file_events: Dict[Tuple[str, str], List[Dict[str, float]]] = {}
    for cname, events in support_config.items():
        for ev in events:
            key = (cname, ev["path"])
            file_events.setdefault(key, []).append(ev)

    for (cname, wav_path), events in file_events.items():
        c_idx = cname_to_idx[cname]

        wave, sr = torchaudio.load(wav_path)
        assert sr == data_cfg.sample_rate

        if wave.size(0) > 1:
            wave = wave.mean(dim=0, keepdim=True)
        wave = wave.to(device)

        if use_esn:
            feat_seq = extract_cnn_esn_embeddings(wave, esn)
        else:
            feat_seq = extract_cnn_embeddings(wave)

        T_seq = feat_seq.size(0)
        dur_sec = wave.shape[-1] / data_cfg.sample_rate
        mask_pos = torch.zeros(T_seq, dtype=torch.bool, device=device)
        for ev in events:
            onset_sec = float(ev["onset"])
            offset_sec = float(ev["offset"])

            start_idx = int(round(onset_sec / dur_sec * T_seq))
            end_idx   = int(round(offset_sec / dur_sec * T_seq))

            start_idx = max(0, min(start_idx, T_seq - 1))
            end_idx   = max(start_idx + 1, min(end_idx, T_seq))

            mask_pos[start_idx:end_idx] = True

        pos_idx = torch.where(mask_pos)[0]
        neg_idx = torch.where(~mask_pos)[0]

        if pos_idx.numel() == 0:
            continue

        if neg_idx.numel() > 0 and neg_ratio > 0:
            n_neg = int(min(neg_idx.numel(), pos_idx.numel() * neg_ratio))
            perm = torch.randperm(neg_idx.numel(), device=device)[:n_neg]
            neg_idx = neg_idx[perm]
        else:
            neg_idx = neg_idx[:0]


        feat_pos = feat_seq[pos_idx]
        y_pos = torch.zeros(feat_pos.size(0), num_classes, device=device)
        y_pos[:, c_idx] = 1.0

        Z_list.append(feat_pos)
        Y_list.append(y_pos)

        if neg_idx.numel() > 0:
            feat_neg = feat_seq[neg_idx]
            y_neg = torch.zeros(feat_neg.size(0), num_classes, device=device)
            Z_list.append(feat_neg)
            Y_list.append(y_neg)

    Z_all = torch.cat(Z_list, dim=0)
    Y_all = torch.cat(Y_list, dim=0)
    print("Z_all:", Z_all.shape)
    print("Y_all:", Y_all.shape)
    print("Y_all positive frames:", (Y_all > 0.5).sum(dim=0))
    print("Y_all mean:", Y_all.mean(dim=0))
    return Z_all, Y_all, class_names

def cnn_frontend_from_fmn10(mels: torch.Tensor) -> torch.Tensor:
    feats = fmn.fmn(mels)

    if feats.dim() != 3:
        raise ValueError(f"Unexpected fmn output shape: {feats.shape}")
    if feats.shape[1] != data_cfg.frames_1s * 10 and feats.shape[2] == data_cfg.frames_1s * 10:
        feats = feats.transpose(1, 2)
    elif feats.shape[1] > feats.shape[2]:
        feats = feats.transpose(1, 2)

    return feats

def extract_seq_features(
    wave: torch.Tensor,
    use_cnn: bool = True,
) -> torch.Tensor:
    if wave.dim() == 1:
        wave = wave.unsqueeze(0)
    elif wave.dim() == 3:
        wave = wave.squeeze(1)
    elif wave.dim() != 2:
        raise ValueError(f"Unexpected wave shape: {wave.shape}")

    with torch.no_grad():
        mels = mel_extractor(wave)

        if use_cnn:
            feats = cnn_frontend_from_fmn10(mels)
        else:
            feats = mels.squeeze(1).transpose(1, 2)

    if data_cfg.use_esn_input_std:
        mean = feats.mean(dim=(0, 1), keepdim=True)
        std  = feats.std(dim=(0, 1), keepdim=True) + 1e-8
        feats = (feats - mean) / std

    return feats

# def get_expanded_range(name: str):
#     base = data_cfg.HP_RANGES[name]
#     base_min = base["min"]
#     base_max = base["max"]
#     center = 0.5 * (base_min + base_max)
#     half_width = 0.5 * (base_max - base_min)

#     half_width_expanded = half_width * (1.0 + data_cfg.EXPAND_RATIO)

#     min2 = center - half_width_expanded
#     max2 = center + half_width_expanded

#     if name == "LEARNING_RATE":
#         min2 = max(min2, data_cfg.LR_MIN_FLOOR)

#         max2 = min(max2, data_cfg.LR_MAX_CEIL)

#         if max2 <= min2:
#             max2 = min2 * 1.0001

#     elif name == "INPUT_SCALE":
#         min2 = max(min2, data_cfg.INPUT_SCALE_MIN_FLOOR)
#         max2 = min(max2, data_cfg.INPUT_SCALE_MAX_CEIL)

#         if max2 <= min2:
#             max2 = min2 * 1.0001

#     print(name, "min", min2, "max", max2)
#     return min2, max2

def extract_cnn_embeddings(wave: torch.Tensor) -> torch.Tensor:
    """
    wave: (1, T) or (T,)
    return: (T_seq, D_cnn)
    """
    if wave.dim() == 1:
        wave = wave.unsqueeze(0)
    elif wave.dim() == 3:
        wave = wave.squeeze(0)

    with torch.no_grad():
        feats = extract_seq_features(wave, use_cnn=True)
    return feats.squeeze(0)  


def extract_cnn_esn_embeddings(wave: torch.Tensor, esn) -> torch.Tensor:
    if wave.dim() == 1:
        wave = wave.unsqueeze(0)
    elif wave.dim() == 3:
        wave = wave.squeeze(0)

    with torch.no_grad():
        feats = extract_seq_features(wave, use_cnn=True)
        h = esn(feats)
    return h.squeeze(0)

# def build_group_indices_from_map(label_to_idx: dict, head_groups: List[List[str]]) -> List[List[int]]:
#     group_indices = []
#     used = set()

#     for g in head_groups:
#         idxs = [label_to_idx[c] for c in g]  # ← CSV/TSV由来 idx を直接使う
#         assert len(set(idxs)) == len(idxs), f"dup in group: {g}"
#         group_indices.append(idxs)
#         used |= set(idxs)

#     # 全クラスをカバーしたいなら（必要に応じて）
#     # assert used == set(range(len(label_to_idx))), ...
#     return group_indices

# def build_group_indices(label_to_idx: dict, head_groups: list[list[str]]) -> list[list[int]]:
#     group_indices = []
#     used = set()
#     for g in head_groups:
#         idxs = [label_to_idx[c] for c in g]  # ← dict なのでOK
#         assert len(set(idxs)) == len(idxs), f"dup in group: {g}"
#         group_indices.append(idxs)
#         used |= set(idxs)
#     return group_indices

def build_group_indices(trigger_classes: List[str], head_groups: List[List[str]]) -> List[List[int]]:
    name_to_idx = {c: i for i, c in enumerate(trigger_classes)}
    group_indices: List[List[int]] = []
    used = set()
    for g in head_groups:
        idxs = [name_to_idx[c] for c in g]
        assert len(set(idxs)) == len(idxs), f"dup in group: {g}"
        group_indices.append(idxs)
        used |= set(idxs)
    assert used == set(range(len(trigger_classes))), f"groups do not cover all classes: used={sorted(used)}"
    return group_indices

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

def get_dataloaders(
        audio_path,
        tsv_path,
        with_background=False, 
        background_ratio=0.0,
        sample_rate=16000,
        train_num = 6000,
        eval_num = 2000,
        test_num = 2000,
        
        TRIGGER_CLASSES = None,
        K_SHOT = None,
        use_long_chewing_filter=False,
        SEED = 0,

        include_fnames=None,
):
    g = torch.Generator().manual_seed(SEED)
    g_val = torch.Generator().manual_seed(SEED + 1)
    g_test = torch.Generator().manual_seed(SEED + 2)
    
    random.seed(SEED)
    np.random.seed(SEED)

    if with_background:
        ds_train = get_training_dataset(
            audio_path,
            tsv_path,
            sample_rate=sample_rate,
            label_fps=100,
            seed=SEED,
            with_bg_only=with_background, bg_only_ratio=background_ratio,
            max_files=train_num,
            target_classes=TRIGGER_CLASSES,
            fewshot_k=K_SHOT,
            fewshot_seed=SEED,
            fewshot_single_label=True, 
            include_fnames=include_fnames,  
        )
    else:
        ds_train = get_training_dataset(audio_path, tsv_path, max_files=train_num, sample_rate=sample_rate, label_fps=100, seed=SEED, with_bg_only=with_background, target_classes=TRIGGER_CLASSES, fewshot_k=K_SHOT, include_fnames=include_fnames)
    ds_val   = get_validation_dataset(audio_path, tsv_path, sample_rate=sample_rate, label_fps=100, seed=SEED, max_files=eval_num, target_classes=TRIGGER_CLASSES, use_long_chewing_filter=use_long_chewing_filter)
    ds_test  = get_test_dataset(audio_path, tsv_path, sample_rate=sample_rate, label_fps=100, seed=SEED, max_files=test_num, target_classes=TRIGGER_CLASSES, use_long_chewing_filter=use_long_chewing_filter)
    train_loader = DataLoader(ds_train, batch_size=64, shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=g)
    val_loader   = DataLoader(ds_val,   batch_size=64, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g_val)
    test_loader  = DataLoader(ds_test,  batch_size=64, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g_test)
    
    return train_loader, val_loader, test_loader

def build_group_indices_fullspace(task_path: str, head_groups: list[list[str]]):
    label_vocab, _ = label_vocab_nlabels(task_path)
    label_vocab = label_vocab.sort_values("idx")
    label_to_idx = dict(zip(label_vocab["label"].astype(str), label_vocab["idx"].astype(int)))

    group_indices = []
    for g in head_groups:
        idxs = []
        for name in g:
            if name not in label_to_idx:
                raise ValueError(f"Head group label '{name}' not found in labelvocabulary.csv")
            idxs.append(label_to_idx[name])
        group_indices.append(idxs)
    return group_indices

@torch.no_grad()
def make_feature_loader_from_wave_loader(
    *,
    encoder,
    wave_loader,
    device,
    use_amp: bool = False,
    shuffle: bool = True,
    batch_size: int = 64,
    return_cache: bool = False,
    store_dtype=torch.float16,
    return_meta: bool = False,
    sample_rate: int = 16000,
    esn_input_from="cnn"
):
    encoder.eval()
    feats_list, labels_list = [], []
    meta = [] if return_meta else None

    if sample_rate is None:
        sample_rate = cfg.SAMPLE_RATE

    for batch in wave_loader:
        audio, labels, filenames, durations = batch
        audio  = audio.to(device)
        labels = labels.to(device).float()
        if esn_input_from == "cnn":
            feats = encoder(
                audio,
                spec=EncodeSpec(out="cnn_seq", detach=True, use_amp=use_amp, esn_input_from=esn_input_from),
                esn=None,
            )  # (B,T_feat,D)
        elif esn_input_from == "mel":
            mels = encoder.mel_extractor(audio)
            feats = encoder._mel_to_BTD(mels)
        

        feats_list.append(feats.detach().to("cpu", dtype=store_dtype))
        labels_list.append(labels.detach().to("cpu"))

        if return_meta:
            B = audio.size(0)
            for i in range(B):
                fn = filenames[i] if filenames is not None else f"clip_{len(meta):06d}"
                dur = float(audio[i].shape[-1] / sample_rate)
                meta.append({"filename": fn, "duration": dur})

    feats_cpu  = torch.cat(feats_list, dim=0)
    labels_cpu = torch.cat(labels_list, dim=0)

    if return_cache:
        if return_meta:
            return feats_cpu, labels_cpu, meta
        return feats_cpu, labels_cpu

    ds = TensorDataset(feats_cpu, labels_cpu)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

