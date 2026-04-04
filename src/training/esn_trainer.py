from pathlib import Path
import time
import copy
import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

from typing import Literal, Optional


from src.evaluation.inference import pred_esn
from src.evaluation.metrics import event_based_evaluation_df, find_best_threshold_per_class_event, preds_to_event_df
from src.evaluation.pipeline import downsample_labels_to_T, tune_median_and_threshold
from src.evaluation.report import get_psds_meta
from src.models.encorders.common_audio_encorder import EncodeSpec

from config_loader import data_cfg, esn_cfg
from src.models.model_utils import get_peak_mib, list_trainable_params, reset_peak
from src.models.temporal_modules.esn import ESN, ESN_Para, ESNReadout, GroupedESNReadout, ReadoutTrainPara, make_esn, solve_ridge_grouped, solve_ridge_regression
from src.models.wrappers.esn_sed import SEDModelEncoderESNReadout, SEDModelEncoderESNRidge
from src.training.loader_utils import compute_pos_weight_from_Y, make_reservoir_loader
from third_party.EfficientSED.dcase2016task2 import label_vocab_as_dict, label_vocab_nlabels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def collect_esn_states_and_labels(
    *,
    encoder,
    esn,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    esn_input_from="cnn",
):
    H_list, Y_list = [], []

    encoder.eval()
    esn.eval()

    for audio, labels, _, _ in dataloader:
        audio  = audio.to(device)
        labels = labels.to(device).float()

        h_seq = encoder(
            audio,
            spec=EncodeSpec(out="esn_seq", detach=True, use_amp=use_amp, esn_input_from=esn_input_from),
            esn=esn,
        )

        B, T_res, Hdim = h_seq.shape
        labels = downsample_labels_to_T(labels, h_seq.shape[1])

        H_flat = h_seq.reshape(-1, h_seq.shape[-1])  
        Y_flat = labels.permute(0, 2, 1).reshape(-1, labels.size(1))

        H_list.append(H_flat)
        Y_list.append(Y_flat)

    H_all = torch.cat(H_list, 0)
    Y_all = torch.cat(Y_list, 0)
    
    print(f"Features cached on GPU: {H_all.shape}")
    return H_all, Y_all

@torch.no_grad()
def build_val_cache_esn_seq(
    *,
    encoder,
    esn,
    out: str = "esn_seq",
    val_wave_loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    store_dtype=torch.float16,
    esn_input_from="cnn",
):
    encoder.eval()
    esn.eval()

    H_list, Y_list = [], []
    meta = []

    for batch in val_wave_loader:
        audio, labels, filenames, durations = batch

        audio  = audio.to(device)
        labels = labels.to(device).float()

        h_seq = encoder(
            audio,
            spec=EncodeSpec(out=out, detach=True, use_amp=use_amp, esn_input_from=esn_input_from),
            esn=(esn if out == "esn_seq" else None),
        )
        labels = downsample_labels_to_T(labels, h_seq.shape[1])

        H_list.append(h_seq)
        Y_list.append(labels)

        B = audio.size(0)
        for i in range(B):
            fn = filenames[i] if filenames is not None else f"clip_{len(meta):06d}"
            sr = data_cfg.sample_rate
            dur = float(audio[i].shape[-1] / sr)
            meta.append({"filename": fn, "duration": dur})

    H_seq_cpu = torch.cat(H_list, dim=0) 
    Y_ds_cpu  = torch.cat(Y_list, dim=0)
    return H_seq_cpu, Y_ds_cpu, meta

def train_readout_select_by_psds(
    *,
    readout: nn.Module,
    train_res_loader: DataLoader,
    H_va_seq_cpu: torch.Tensor,
    Y_va_ds_cpu : torch.Tensor,
    va_meta: list,
    CLASS_NAMES,
    n_epochs: int,
    device: torch.device,
    lr: float = 1e-3,
    pos_weight: Optional[torch.Tensor] = None,
    eval_batch_size: int = 64,
    patience: int = 5,
    trial: Optional["optuna.trial.Trial"] = None,
):
    n_trainable = sum(p.numel() for p in readout.parameters() if p.requires_grad)
    print("trainable(readout) =", n_trainable)
    assert n_trainable > 0

    w0 = copy.deepcopy({k: v.detach().cpu().clone() for k,v in readout.state_dict().items()})

    readout.to(device)
    optimizer = torch.optim.Adam(readout.parameters(), lr=lr)

    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    best = {"psds1": -1.0, "epoch": 0, "state": None, "median_win": 1}
    bad = 0

    logged_once = False

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
        readout.train()
        total, n = 0.0, 0

        for H_batch, Y_batch in train_res_loader:
            H_batch = H_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            logits = readout(H_batch)
            loss = criterion(logits, Y_batch)
            loss.backward()
            if not logged_once and device.type == 'cuda':
                current_mem = torch.cuda.memory_allocated() / 1024**2
                peak_mem    = torch.cuda.max_memory_allocated() / 1024**2
                print(f"    [ESN MEM CHECK] Step Peak: {peak_mem:.2f} MB, Current: {current_mem:.2f} MB")
                logged_once = True
            optimizer.step()

            total += float(loss.item()) * H_batch.size(0)
            n += H_batch.size(0)

        train_time = _now(device) - t_train0
        train_time_total += train_time
        if epoch == 1:
            w1 = {k: v.detach().cpu() for k,v in readout.state_dict().items()}
            diff = max((w1[k] - w0[k]).abs().max().item() for k in w0.keys())
            print("max |Δparam| =", diff)

        print(f"[Readout] Epoch {epoch}/{n_epochs}, loss={total/max(n,1):.4f}"
              f"(train={train_time:.2f}s)")

        readout.eval()
        t_val_infer0 = _now(device)
        val_infer_time = _now(device) - t_val_infer0
        val_infer_time_total += val_infer_time
        t_psds0 = _now(device)
        psds1, win = eval_psds1_on_val_wave(
            CLASS_NAMES=CLASS_NAMES,
            readout=readout,
            H_seq_cpu=H_va_seq_cpu,
            Y_ds_cpu=Y_va_ds_cpu,
            meta=va_meta,
            device=device,
            batch_size=eval_batch_size,
        )
        psds_time = _now(device) - t_psds0
        psds_time_total += psds_time
        print(f" [Val] PSDS1={psds1:.4f} (best win={win}) "
            f"(infer={val_infer_time:.2f}s, psds={psds_time:.2f}s)")

        if trial is not None:
            trial.report(psds1, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
        if psds1 > best["psds1"] + 1e-6:
            best["psds1"] = psds1
            best["epoch"] = epoch
            best["median_win"] = win
            best["state"] = copy.deepcopy(readout.state_dict())
            bad = 0
            print(f"  >>> Best updated! epoch={epoch}, PSDS1={psds1:.4f}")
        else:
            bad += 1
            if patience > 0 and bad >= patience:
                print(f"  >>> Early stop (patience={patience}). best_epoch={best['epoch']}")
                break

    if best["state"] is not None:
        readout.load_state_dict(best["state"])
    
    wall_total = train_time_total + val_infer_time_total + psds_time_total
    print(f"[TIME] train_total = {train_time_total/60:.2f} min")
    print(f"[TIME] val_infer_total = {val_infer_time_total/60:.2f} min")
    print(f"[TIME] psds_total = {psds_time_total/60:.2f} min")
    print(f"[TIME] wall_total = {wall_total/60:.2f} min")
    return best

def train_readout_select_by_psds_on_gpu(
    readout: nn.Module,
    # Train Data (Flattened on GPU)
    H_tr_flat: torch.Tensor, 
    Y_tr_flat: torch.Tensor,
    # Val Data (Sequence on GPU)
    H_va_seq: torch.Tensor,
    Y_va_seq: torch.Tensor,
    va_meta: list,
    # Config
    n_epochs: int,
    batch_size: int,
    lr: float,
    CLASS_NAMES: list,
    patience: int = 10,
    pos_weight = None,
    trial: Optional["optuna.trial.Trial"] = None,
    wantPsds=True,
    get_answer="psds1"
):
    print(f"Start Training on GPU Cache with PSDS monitor (epochs={n_epochs})...")
    def _now(device):
        if device.type == "cuda":
            torch.cuda.synchronize()
            return time.perf_counter()
    train_time_total = 0.0
    val_infer_time_total = 0.0
    psds_time_total = 0.0
    optimizer = torch.optim.Adam(readout.parameters(), lr=lr)
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    N_train = H_tr_flat.size(0)

    if wantPsds:
        best = {"psds1": -1.0, "epoch": 0, "state": None, "median_win": 1, "F1_clock": -1.0 if get_answer == "F1_clock" else None}
    else:
        N_va, T_va, H_dim = H_va_seq.shape
        C_dim = Y_va_seq.shape[1]
        
        H_va_flat = H_va_seq.reshape(-1, H_dim)
        Y_va_flat = Y_va_seq.permute(0, 2, 1).reshape(-1, C_dim)
        N_val_total = H_va_flat.size(0)
        best = {"loss": float("inf"), "epoch": 0, "state": None}
    bad_counts = 0
    
    for epoch in range(1, n_epochs + 1):
        t_train0 = _now(device)
        readout.train()
        indices = torch.randperm(N_train, device=H_tr_flat.device)
        total_loss = 0
        
        for start_idx in range(0, N_train, batch_size):
            idx = indices[start_idx : start_idx + batch_size]
            
            H_batch = H_tr_flat[idx]
            Y_batch = Y_tr_flat[idx]
            
            optimizer.zero_grad()
            logits = readout(H_batch)
            loss = criterion(logits, Y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(idx)
        
        train_time = _now(device) - t_train0
        train_time_total += train_time
        avg_loss = total_loss / N_train
        print(f"[Readout] Epoch {epoch}/{n_epochs}, loss={avg_loss:.5f}"
              f"(train={train_time:.2f}s)")
        
        # --- 2. Validation with PSDS (GPU Sequence) ---
        if wantPsds:
            psds1, win = eval_psds_on_gpu_cache(
                readout, H_va_seq, Y_va_seq, va_meta, CLASS_NAMES
            )
            print(f"Epoch {epoch}/{n_epochs} | Loss: {avg_loss:.5f} | Val PSDS1: {psds1:.4f} (win={win})")
            # Optuna Pruning
            if trial is not None:
                trial.report(psds1, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            if psds1 > best["psds1"] + 1e-6:
                best["psds1"] = psds1
                best["epoch"] = epoch
                best["median_win"] = win
                best["state"] = copy.deepcopy(readout.state_dict())
                bad_counts = 0
                print("  >>> Best PSDS updated!")
            else:
                bad_counts += 1
                if patience > 0 and bad_counts >= patience:
                    print(f"  >>> Early stopping at epoch {epoch}. Best was {best['epoch']}")
                    break
        else: 
            readout.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for start_idx in range(0, N_val_total, batch_size):
                    end_idx = min(start_idx + batch_size, N_val_total)
                    
                    H_batch_val = H_va_flat[start_idx : end_idx]
                    Y_batch_val = Y_va_flat[start_idx : end_idx]
                    
                    logits_val = readout(H_batch_val)
                    loss_val = criterion(logits_val, Y_batch_val)
                    total_val_loss += loss_val.item() * (end_idx - start_idx)

            avg_val_loss = total_val_loss / N_val_total

            print(f"Epoch {epoch}/{n_epochs} | Tr Loss: {avg_loss:.5f} | Val Loss: {avg_val_loss:.5f}")
            
            if trial is not None:
                trial.report(avg_val_loss, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            if avg_val_loss < best["loss"]:
                best["loss"] = avg_val_loss
                best["epoch"] = epoch
                best["state"] = copy.deepcopy(readout.state_dict())
                best["median_win"] = 1
                bad_counts = 0
            else:
                bad_counts += 1
                if patience > 0 and bad_counts >= patience:
                    print(f"  >>> Early stopping at epoch {epoch}. Best Loss {best['loss']:.5f}")
                    break
    
    # 学習終了後、ベストな重みに戻す
    if best["state"] is not None:
        readout.load_state_dict(best["state"])

    if get_answer == "F1_clock":
        psds1, win, f1_clock = eval_psds_on_gpu_cache(
            readout, H_va_seq, Y_va_seq, va_meta, CLASS_NAMES, get_answer=get_answer
        )
        if get_answer == "F1_clock":
            best["F1_clock"] = f1_clock
    print(f"[TIME] train_total = {train_time_total/60} min")
    return best

@torch.no_grad()
def eval_psds1_on_val_wave(
    *,
    CLASS_NAMES,
    readout: torch.nn.Module,
    H_seq_cpu: torch.Tensor,
    Y_ds_cpu : torch.Tensor,
    meta: list,
    device: torch.device,
    batch_size: int = 64,
):
    readout.eval()
    N, T, Hdim = H_seq_cpu.shape
    C = Y_ds_cpu.shape[1]

    ds = TensorDataset(H_seq_cpu, Y_ds_cpu)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    pred_all = np.empty((N, C, T), dtype=np.float32)
    Y_all    = np.empty((N, C, T), dtype=np.float32)
    
    offset = 0
    for H_b_cpu, Y_b_cpu in loader:
        B = H_b_cpu.size(0)

        H_b = H_b_cpu.to(device, non_blocking=True).float()
        logits = readout(H_b.reshape(-1, Hdim))
        probs  = torch.sigmoid(logits).reshape(B, T, C).permute(0, 2, 1)

        pred_all[offset:offset+B] = probs.to("cpu").numpy()
        Y_all[offset:offset+B]    = Y_b_cpu.numpy()
        offset += B

    pred_NTc = np.transpose(pred_all, (0, 2, 1))
    Y_NTc    = np.transpose(Y_all,    (0, 2, 1))

    best_win, best_psds1 = tune_median_and_threshold(
        pred_NTc, Y_NTc, meta, class_names=CLASS_NAMES,  WIN_SIZE=[1],
    )
    return float(best_psds1), int(best_win)

@torch.no_grad()
def eval_psds_on_gpu_cache(
    readout: nn.Module,
    H_seq_gpu: torch.Tensor,
    Y_seq_gpu: torch.Tensor,
    meta: list,
    CLASS_NAMES: list,
    batch_size: int = 512,
    get_answer="psds1",
):
    readout.eval()
    N, T, Hdim = H_seq_gpu.shape
    C = Y_seq_gpu.shape[1]
    
    pred_list = []
    
    for i in range(0, N, batch_size):
        H_batch = H_seq_gpu[i : i + batch_size]
        
        B_curr = H_batch.size(0)
        logits = readout(H_batch.reshape(-1, Hdim))
        probs = torch.sigmoid(logits).reshape(B_curr, T, C)
        
        pred_list.append(probs.cpu().numpy())

    pred_NTc = np.concatenate(pred_list, axis=0)
    
    Y_NTc = Y_seq_gpu.cpu().numpy().transpose(0, 2, 1)
    
    best_win, best_psds1 = tune_median_and_threshold(
        pred_NTc, Y_NTc, meta, class_names=CLASS_NAMES, WIN_SIZE=[1]
    )

    if get_answer == "F1_clock":
        best_thresholds, f1info = find_best_threshold_per_class_event(pred_NTc, Y_NTc, data_cfg.frames_1s, data_cfg.th_grid)
        gt_df, filenames, durations = get_psds_meta(Y_NTc, meta, CLASS_NAMES)    
        est_df = preds_to_event_df(
            pred_scores=pred_NTc,
            filenames=[m["filename"] for m in meta],
            class_names=CLASS_NAMES,
            frames_per_sec=data_cfg.frames_1s,
            thresholds=best_thresholds,
            median_win=best_win,
        )
        metric_event = event_based_evaluation_df(gt_df, est_df)
        ev_cw = metric_event.results_class_wise_metrics()
        clock_ev_f = ev_cw["clock"]["f_measure"]["f_measure"]
        print(f"Best F1_clock (using PSDS-selected median_win={best_win}): {clock_ev_f:.4f}")
        return best_psds1, best_win, clock_ev_f
    
    return best_psds1, best_win

def build_sed_model_with_readout(
    *,
    encoder,
    esn: ESN,
    H_tr: torch.Tensor,
    Y_tr: torch.Tensor,
    H_va: torch.Tensor,
    Y_va: torch.Tensor,
    va_meta, 
    CLASS_NAMES,
    nlabels: int,
    readout_type: str,
    readout_para,
    posWeight=False,
    group_indices=None,
    SEARCH_EPOCHS_BY_PSDS=False,
    use_amp: bool = False,
    device: Optional[torch.device] = None,
    return_best_info: bool = False,
    trial: Optional["optuna.trial.Trial"] = None,
    esn_input_from: Literal["cnn","mel"]="cnn",
    wantPsds=True,
    get_answer="psds1",
) -> nn.Module:
    if device is None:
        try:
            device = next(encoder.parameters()).device
        except StopIteration:
            device = next(esn.parameters()).device

    encoder = encoder.to(device).eval()
    esn     = esn.to(device).eval()

    in_dim = H_tr.shape[1] 
    pos_weight = compute_pos_weight_from_Y(Y_tr).to(device) if posWeight else None

    if readout_type == "ridge":
        if group_indices is None:
            print("ridge")
            W, b = solve_ridge_regression(H_tr.to(device), Y_tr.to(device), l2_reg=readout_para.L2)
        else:
            print("Grouped ridge")
            W, b = solve_ridge_grouped(H_tr.to(device), Y_tr.to(device), group_indices, l2_reg=readout_para.L2)
        del H_tr, Y_tr
        import gc; gc.collect(); torch.cuda.empty_cache()
        model = SEDModelEncoderESNRidge(
                encoder=encoder, esn=esn, W=W, b=b, use_amp=use_amp, esn_input_from=esn_input_from,
            ).to(device)
        list_trainable_params(encoder) 
        list_trainable_params(model)
        return (model, None) if return_best_info else model

    if readout_type == "logistic":
        assert readout_para.LEARNING_RATE is not None

        train_res_loader = make_reservoir_loader(
            H_tr.cpu().numpy(), Y_tr.cpu().numpy(), batch_size=4096, shuffle=True
        )
        if group_indices is None:
            print("logistic")
            readout = ESNReadout(in_dim=in_dim, n_classes=nlabels).to(device)
        else:
            print("Grouped logistic")
            readout = GroupedESNReadout(in_dim=in_dim, n_classes=nlabels, group_indices=group_indices).to(device)
        list_trainable_params(encoder)
        list_trainable_params(readout)
        if SEARCH_EPOCHS_BY_PSDS:
            best = train_readout_select_by_psds_on_gpu(
                readout=readout,
                H_tr_flat=H_tr,
                Y_tr_flat=Y_tr,
                H_va_seq=H_va,
                Y_va_seq=Y_va,
                va_meta=va_meta,
                n_epochs=readout_para.NUM_EPOCHS,
                batch_size=readout_para.BATCH_SIZE,
                lr=readout_para.LEARNING_RATE,
                CLASS_NAMES=CLASS_NAMES,
                patience=readout_para.PATIENCE,
                pos_weight=pos_weight,
                wantPsds=wantPsds,
                get_answer=get_answer
            )
            # print("Selected epoch by PSDS:", best)
        else:
            val_res_loader = make_reservoir_loader(
                H_va, Y_va, batch_size=4096, shuffle=False
            )
            train_readout_on_gpu(
                readout=readout,
                train_loader=train_res_loader,
                val_loader=val_res_loader,
                n_epochs=readout_para.NUM_EPOCHS,
                device=device,
                lr=readout_para.LEARNING_RATE,
                pos_weight=pos_weight,
            )
            best = {
                "psds1": -1.0,
                "epoch": readout_para.NUM_EPOCHS,
                "state": readout.state_dict(),
                "median_win": 1,
            }
            print(f"Finished training for {readout_para.NUM_EPOCHS} epochs (No PSDS selection).")

        model = SEDModelEncoderESNReadout(encoder=encoder, esn=esn, readout=readout, use_amp=use_amp, esn_input_from=esn_input_from).to(device)
        return (model, best) if return_best_info else model
    raise ValueError(f"Unknown readout_type: {readout_type}")

def train_readout_on_gpu(
    readout: nn.Module,
    H_tr: torch.Tensor,
    Y_tr: torch.Tensor,
    n_epochs: int,
    batch_size: int,
    lr: float,
    pos_weight=None,
):
    optimizer = torch.optim.Adam(readout.parameters(), lr=lr)
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
        
    N = H_tr.size(0)
    
    for epoch in range(1, n_epochs + 1):
        readout.train()
        

        indices = torch.randperm(N, device=H_tr.device)
        
        total_loss = 0
        
        for start_idx in range(0, N, batch_size):
            idx = indices[start_idx : start_idx + batch_size]
            
            H_batch = H_tr[idx]
            Y_batch = Y_tr[idx]
            
            optimizer.zero_grad()
            logits = readout(H_batch)
            loss = criterion(logits, Y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(idx)
            
        print(f"Epoch {epoch}: Loss {total_loss/N:.5f}")

def run_esn(
    *,
    encoder,
    task_path,
    CLASS_NAMES,
    train_loader: Optional[DataLoader],
    val_loader : Optional[DataLoader],
    test_loader : Optional[DataLoader],
    readout_type="logistic",
    out="esn_seq",
    posWeight=False,
    group_indices=None,

    esn_para: Optional[ESN_Para] = None,
    readout_para: Optional[ReadoutTrainPara] = None,
    BY_EPOCH = True,
    feature_batch_size: int = 4096,
    use_amp: bool = False,
    esn_input_from: Literal["cnn","mel"] = "cnn",
    peak_log: Optional[dict] = None,
    peak_kind: str = "allocated",
):
    label_vocab, nlabels = label_vocab_nlabels(Path(task_path))
    label_to_idx = label_vocab_as_dict(label_vocab, key="label", value="idx")
    try:
        device = next(encoder.parameters()).device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder = encoder.to(device)
    peak_log = {}

    with torch.no_grad():
        dummy = torch.randn(1, 16000 * 10, device=device)

        if esn_input_from == "mel":
            mels = encoder.mel_extractor(dummy)
            mel_seq = encoder._mel_to_BTD(mels)
            D_in = mel_seq.shape[-1]
            print("mel_seq shape:", mel_seq.shape)
        else:
            feat = encoder(dummy, spec=EncodeSpec(out="cnn_seq", detach=True, use_amp=use_amp), esn=None)
            D_in = feat.shape[-1]
            print("feat shape:", feat.shape)
    
    esn, _para_used = make_esn(
        ESN_cls=ESN,
        input_dim=D_in,
        device=device,
        cfg=esn_cfg,
        esn_para=esn_para,
    )
    # esn = torch.nn.Identity().to(device)
    reset_peak(device)
    H_tr, Y_tr = collect_esn_states_and_labels(
        encoder=encoder, esn=esn, dataloader=train_loader, device=device, use_amp=use_amp, esn_input_from=esn_input_from)
    print("Train data", H_tr.shape)
    peak_log["collect_train"] = get_peak_mib(device)
    reset_peak(device)
    H_va, Y_va, va_meta = build_val_cache_esn_seq(
        encoder=encoder,
        esn=esn,
        out=out,
        val_wave_loader=val_loader,
        device=device,
        use_amp=use_amp,
        store_dtype=torch.float16,
        esn_input_from=esn_input_from
    )
    peak_log["cache_val"] = get_peak_mib(device)
    print(Y_tr.shape, Y_va.shape)
    print("val cache:", tuple(H_va.shape), tuple(Y_va.shape), "meta:", len(va_meta))
    print((Y_va == 1).sum().item())

    for name, H in [("H_tr", H_tr), ("H_va", H_va)]:
        x = H.detach().float()
        print(name, "mean", x.mean().item(), "std", x.std().item(),
            "absmax", x.abs().max().item())
    readout_para = readout_para if readout_para is not None else ReadoutTrainPara.from_cfg(esn_cfg.readout_para)
    print("ESN and Readout Parameters:")
    print(_para_used)
    print(readout_para)
    
    if readout_type == "ridge" or readout_type == "logistic":
        reset_peak(device)
        sed_model = build_sed_model_with_readout(
            encoder=encoder,
            esn=esn,
            H_tr=H_tr,
            Y_tr=Y_tr,
            H_va=H_va,
            Y_va=Y_va,
            va_meta=va_meta,
            CLASS_NAMES=CLASS_NAMES,
            SEARCH_EPOCHS_BY_PSDS=BY_EPOCH,
            nlabels=Y_tr.shape[1],
            readout_type=readout_type,
            readout_para=readout_para,
            posWeight=posWeight,
            group_indices=group_indices,
            use_amp=use_amp,
            device=device,
            esn_input_from=esn_input_from,
        )
        peak_log["train_readout"] = get_peak_mib(device)
        if peak_log is not None:
            reset_peak(device)
        
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