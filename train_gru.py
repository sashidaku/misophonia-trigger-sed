from dataclasses import dataclass
import os
import time
from typing import Optional
import numpy as np

# from .get_data import build_mel_and_cnn, get_dataloaders
from src.data.get_data import build_mel_and_cnn, get_dataloaders
from src.evaluation.eval_bigru import eval_psds1_on_val_cached_feats
from src.evaluation.inference import pred_esn
from src.evaluation.pipeline import downsample_labels_to_T, tune_median_and_threshold
from src.models.encorders.common_audio_encorder import CommonAudioEncoder, EncodeSpec, compute_cnnseq_mean_std
from src.models.model_utils import get_peak_mib, list_trainable_params, reset_peak
from src.models.temporal_modules.esn import ReadoutTrainPara
from src.models.temporal_modules.gru import BiGRUFrameHead, GRU_Para
from src.models.wrappers.gru_sed import SEDModelEncoderRNN
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy

from config_loader import data_cfg, esn_cfg, gru_cfg

@torch.no_grad()
def collect_cnn_features_on_gpu(
    encoder, wave_loader, device, 
    use_amp=False, return_meta=False
):
    encoder.eval()
    print("Caching CNN features on GPU...")
    
    X_list = []
    Y_list = []
    meta = []
    
    for batch in wave_loader:
        if len(batch) == 4:
            audio, labels, filenames, durations = batch
        else:
            audio, labels = batch
            filenames, durations = None, None

        audio = audio.to(device)
        labels = labels.to(device).float()

        feats = encoder(
            audio, 
            spec=EncodeSpec(out="cnn_seq", detach=True, use_amp=use_amp), 
            esn=None
        )
        labels_ds = downsample_labels_to_T(labels, feats.shape[1])
        X_list.append(feats)
        Y_list.append(labels_ds)
        
        if return_meta and filenames is not None:
            B = audio.size(0)
            sample_rate = data_cfg.sample_rate
            
            for i in range(B):
                fn = filenames[i]
                
                dur = float(audio[i].shape[-1] / sample_rate)
                
                meta.append({"filename": fn, "duration": dur})

    X_all = torch.cat(X_list, dim=0)
    Y_all = torch.cat(Y_list, dim=0)
    
    print(f"Cached on GPU: X={X_all.shape}, Y={Y_all.shape}")
    
    if return_meta:
        return X_all, Y_all, meta
    return X_all, Y_all

def train_rnn_head_on_gpu(
    *,
    rnn_head: nn.Module,
    X_tr: torch.Tensor, 
    Y_tr: torch.Tensor,
    X_va: torch.Tensor,
    Y_va: torch.Tensor,
    val_meta: list,
    CLASS_NAMES,
    n_epochs: int,
    batch_size: int = 64,
    lr: float = 1e-4,
    pos_weight: Optional[torch.Tensor] = None,
    patience: int = 10,
):
    device = X_tr.device
    
    opt = torch.optim.Adam(rnn_head.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()

    best = {"psds1": -1.0, "epoch": 0, "state": None, "median_win": 1}
    bad = 0
    N_train = X_tr.size(0)

    print(f"Start BiGRU training on GPU (No transfer overhead)... N={N_train}")

    def _now(device):
        if device.type == "cuda":
            torch.cuda.synchronize()
            return time.perf_counter()


    train_time_total = 0.0
    val_infer_time_total = 0.0
    psds_time_total = 0.0


    EVAL_EVERY = 1
    DO_EVAL = (EVAL_EVERY is not None and EVAL_EVERY > 0)

    for epoch in range(1, n_epochs + 1):
        t_train0 = _now(device)
        rnn_head.train()
        total_loss = 0.0
        total_samples = 0
        indices = torch.randperm(N_train, device=device)
        
        for start_idx in range(0, N_train, batch_size):
            idx = indices[start_idx : start_idx + batch_size]
            
            X_batch = X_tr[idx]
            Y_batch = Y_tr[idx]
            
            T = X_batch.size(1)
            Y_ds = Y_batch

            opt.zero_grad()
            logits = rnn_head(X_batch)
            loss = crit(logits, Y_ds)
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(idx)

        train_time = _now(device) - t_train0
        train_time_total += train_time
        avg_loss = total_loss / N_train
        print(f" [RNN] Epoch {epoch}/{n_epochs}, loss={avg_loss:.4f} "
            f"(train={train_time:.2f}s)")
        
        rnn_head.eval()
        pred_list = []
        N_val = X_va.size(0)
        t_val_infer0 = _now(device)
        with torch.no_grad():
            for i in range(0, N_val, batch_size):
                X_b = X_va[i:i+batch_size]
                logits = rnn_head(X_b)
                probs = torch.sigmoid(logits)
                pred_list.append(probs.cpu().numpy())
        val_infer_time = _now(device) - t_val_infer0
        val_infer_time_total += val_infer_time

        t_psds0 = _now(device)
        pred_NTc = np.concatenate(pred_list, axis=0).transpose(0, 2, 1)
        Y_NTc = Y_va.cpu().numpy().transpose(0, 2, 1)

        best_win, psds1 = tune_median_and_threshold(
            pred_NTc, Y_NTc, val_meta, class_names=CLASS_NAMES, WIN_SIZE=[1]
        )
        psds_time = _now(device) - t_psds0
        psds_time_total += psds_time
        print(f" [Val] PSDS1={psds1:.4f} (best win={best_win}) "
            f"(infer={val_infer_time:.2f}s, psds={psds_time:.2f}s)")
        
        if psds1 > best["psds1"] + 1e-6:
            best["psds1"] = psds1
            best["epoch"] = epoch
            best["state"] = copy.deepcopy(rnn_head.state_dict())
            bad = 0
            print("        >>> Best updated!")
        else:
            bad += 1
            if patience > 0 and bad >= patience:
                print(f"    >>> Early stop at epoch {epoch}")
                break

    if best["state"] is not None:
        rnn_head.load_state_dict(best["state"])
    wall_total = train_time_total + val_infer_time_total + psds_time_total
    print(f"[TIME] train_total = {train_time_total/60:.2f} min")
    print(f"[TIME] val_infer_total = {val_infer_time_total/60:.2f} min")
    print(f"[TIME] psds_total = {psds_time_total/60:.2f} min")
    print(f"[TIME] wall_total = {wall_total/60:.2f} min")
    return best

def train_rnn_head_select_by_psds(
    *,
    rnn_head: nn.Module,
    train_feat_loader: DataLoader,
    val_feats_cpu: torch.Tensor,
    val_labels_cpu: torch.Tensor,
    val_meta: list,
    CLASS_NAMES,
    n_epochs: int,
    device: torch.device,
    lr: float = 1e-4,
    pos_weight: Optional[torch.Tensor] = None,
    eval_batch_size: int = 64,
    patience: int = 10,
):
    rnn_head.to(device)
    opt = torch.optim.Adam(rnn_head.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device)) if pos_weight is not None else nn.BCEWithLogitsLoss()

    best = {"psds1": -1.0, "epoch": 0, "state": None, "median_win": 1}
    bad = 0

    logged_once = False

    for epoch in range(1, n_epochs + 1):
        rnn_head.train()
        total, n = 0.0, 0

        for X_cpu, Y_cpu in train_feat_loader:
            X = X_cpu.to(device).float()
            Y = Y_cpu.to(device).float()

            T = X.size(1)
            Y_ds = downsample_labels_to_T(Y, T)

            opt.zero_grad()
            logits = rnn_head(X)
            loss = crit(logits, Y_ds)
            loss.backward()

            opt.step()
            if not logged_once and device.type == 'cuda':
                current_mem = torch.cuda.memory_allocated() / 1024**2
                peak_mem    = torch.cuda.max_memory_allocated() / 1024**2
                print(f"    [MEM CHECK] Step Peak: {peak_mem:.2f} MB, Current: {current_mem:.2f} MB")
                logged_once = True 

            total += float(loss.item()) * X.size(0)
            n += X.size(0)

        print(f"    [RNN] Epoch {epoch}/{n_epochs}, loss={total/max(n,1):.4f}")

        psds1, win = eval_psds1_on_val_cached_feats(
            CLASS_NAMES=CLASS_NAMES,
            rnn_head=rnn_head,
            feats_cpu=val_feats_cpu,
            labels_cpu=val_labels_cpu,
            meta=val_meta,
            device=device,
            batch_size=eval_batch_size,
        )
        print(f"        [Val] PSDS1={psds1:.4f} (best median_win={win})")

        if psds1 > best["psds1"] + 1e-6:
            best["psds1"] = psds1
            best["epoch"] = epoch
            best["median_win"] = win
            best["state"] = copy.deepcopy(rnn_head.state_dict())
            bad = 0
            print(f"        >>> Best updated! epoch={epoch}, PSDS1={psds1:.4f}")
        else:
            bad += 1
            if patience > 0 and bad >= patience:
                print(f"    >>> Early stop (patience={patience}). best_epoch={best['epoch']}")
                break

    if best["state"] is not None:
        rnn_head.load_state_dict(best["state"])
    return best

def run_gru(
    *,
    encoder,
    CLASS_NAMES,
    train_loader,
    val_loader,
    test_loader,
    gru_para: GRU_Para,
    readout_para: ReadoutTrainPara,
    device: torch.device,
    use_amp: bool = False,
):
    encoder = encoder.to(device).eval()

    peak_log = {}

    reset_peak(device)
    
    X_tr, Y_tr = collect_cnn_features_on_gpu(
        encoder, train_loader, device, use_amp=use_amp, return_meta=False
    )
    peak_log["cache_train"] = get_peak_mib(device)
    X_va, Y_va, va_meta = collect_cnn_features_on_gpu(
        encoder, val_loader, device, use_amp=use_amp, return_meta=True
    )
    peak_log["cache_val"] = get_peak_mib(device)

    D_in = X_tr.shape[-1]
    n_classes = Y_tr.shape[1]
    
    rnn_head = BiGRUFrameHead(in_dim=D_in, n_classes=n_classes, para=gru_para).to(device)
    list_trainable_params(rnn_head)

    reset_peak(device)

    best = train_rnn_head_on_gpu(
        rnn_head=rnn_head,
        X_tr=X_tr, 
        Y_tr=Y_tr,
        X_va=X_va, 
        Y_va=Y_va,
        val_meta=va_meta,
        CLASS_NAMES=CLASS_NAMES,
        n_epochs=readout_para.NUM_EPOCHS,
        batch_size=readout_para.BATCH_SIZE, 
        lr=readout_para.LEARNING_RATE,
        patience=readout_para.PATIENCE,
    )
    peak_log["train_head"] = get_peak_mib(device)

    sed_model = SEDModelEncoderRNN(encoder=encoder, rnn_head=rnn_head, use_amp=use_amp).to(device)
    reset_peak(device)
    Y_te, pred_te, te_meta, Y_va, pred_va, va_meta, auc, psds1 = pred_esn(val_loader, test_loader, sed_model, CLASS_NAMES)
    peak_log["infer"] = get_peak_mib(device)

    for k, v in peak_log.items():
        if v is None:
            continue
        alloc, resv = v
        print(f"[PEAK] {k}: allocated={alloc:.1f} MiB, reserved={resv:.1f} MiB")

    if device.type == "cuda":
        worst = max(v[1] for v in peak_log.values() if v is not None)
        print(f"[PEAK] worst reserved across stages = {worst:.1f} MiB")
    return Y_te, pred_te, te_meta, Y_va, pred_va, va_meta, auc, psds1

def main_do_rnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mel_extractor, fmn = build_mel_and_cnn(device)
    list_trainable_params(fmn) 
    encoder = CommonAudioEncoder(
        mel_extractor=mel_extractor,
        fmn=fmn,
        device=device,
        default_std_input=esn_cfg.use_esn_input_std,
        expected_T=data_cfg.frames_1s * 10,
    ).to(device)
    list_trainable_params(encoder) 
    train_loader, val_loader, test_loader = get_dataloaders(
        data_cfg.audio_path, data_cfg.tsv_path,
        with_background=data_cfg.with_background,
        background_ratio=data_cfg.background_ratio,
        TRIGGER_CLASSES=data_cfg.trigger,
        train_num=data_cfg.train_num,
        eval_num=data_cfg.val_num,
        test_num=data_cfg.test_num,
    )
    mean, std = compute_cnnseq_mean_std(encoder, train_loader, device, use_amp=False)
    encoder.set_feature_norm(mean, std)
    gru_para = GRU_Para(RNN_DIM=gru_cfg.rnn_dim, NUM_LAYERS=gru_cfg.num_layers, DROPOUT=gru_cfg.dropout, BIDIR=gru_cfg.bidir)
    readout_para = ReadoutTrainPara(
        LEARNING_RATE=gru_cfg.readout_para.learning_rate,
        NUM_EPOCHS=gru_cfg.readout_para.num_epochs,
        L2=gru_cfg.readout_para.l2,
        BATCH_SIZE=gru_cfg.readout_para.batch_size,
        PATIENCE=gru_cfg.readout_para.patience,
    )
    run_gru(
        encoder=encoder,
        CLASS_NAMES=data_cfg.class_names,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        gru_para=gru_para,
        readout_para=readout_para,
        device=device,
        use_amp=False,
    )

if __name__ == "__main__":
    main_do_rnn()