from typing import List, Literal, Optional
import optuna
import torch

import third_party.EfficientSED.config as cfg
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from config_loader import optuna_cfg

from src.data.get_data import make_feature_loader_from_wave_loader
from src.evaluation.pipeline import downsample_labels_to_T
from src.models.temporal_modules.esn import ESN, ESN_Para, LinearWBReadout, ReadoutTrainPara, make_esn
from src.training.esn_trainer import build_sed_model_with_readout, eval_psds1_on_val_wave

def optuna_search_esn_readout(
    *,
    encoder,
    train_loader,
    val_loader,
    CLASS_NAMES,
    nlabels: int,
    group_indices=None,
    readout_type,
    use_amp: bool = False,
    n_trials: int = 50,
    study_name: str = "esn_readout_psds",
    device: Optional[torch.device] = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device).eval()

    X_tr_cpu, Y_tr_cpu = make_feature_loader_from_wave_loader(
        encoder=encoder, wave_loader=train_loader, device=device,
        return_cache=True, store_dtype=torch.float16, esn_input_from="cnn"
    )
    print("TRAIN", X_tr_cpu.shape)
    X_va_cpu, Y_va_cpu, va_meta = make_feature_loader_from_wave_loader(
        encoder=encoder, wave_loader=val_loader, device=device,
        return_cache=True, store_dtype=torch.float16, return_meta=True, esn_input_from="cnn"
    )
    print("VAL", X_va_cpu.shape)

    def objective(trial: optuna.trial.Trial) -> float:
        esn_para = ESN_Para(
            H_ESN=1024,
            SPECTRAL_RADIUS=trial.suggest_float("SPECTRAL_RADIUS", 0.1, 1.8),
            LEAKING_RATE=trial.suggest_float("LEAKING_RATE", 0.05, 1.0),
            DENSITY=0.1,
            INPUT_SCALE=trial.suggest_float("INPUT_SCALE", 0.01, 5.0, log=True),
            TOPOLOGY="random",
            BIDIRECTIONAL=False,
            BI_MERGE="concat",
            BI_SHARE_WEIGHTS=False, 
        )

        if readout_type == "ridge":
            readout_para = ReadoutTrainPara(
                LEARNING_RATE=1e-4,
                NUM_EPOCHS=100,
                L2=trial.suggest_float("L2", 1e-6, 1e-2, log=True),
                BATCH_SIZE=1024,
                PATIENCE=10,
            )
        else:
            readout_para = ReadoutTrainPara(
                LEARNING_RATE=trial.suggest_float("LR", 1e-4, 3e-3, log=True),
                NUM_EPOCHS=100,
                L2=1e-4,
                BATCH_SIZE=1024,
                PATIENCE=100,
            )

        D_in = X_tr_cpu.shape[-1]
        esn, _ = make_esn(ESN_cls=ESN, input_dim=D_in, device=device, cfg=cfg, esn_para=esn_para)

        tr_ds = TensorDataset(X_tr_cpu, Y_tr_cpu)
        tr_feat_loader = DataLoader(tr_ds, batch_size=64, shuffle=True, num_workers=0)

        H_tr_cpu, Y_tr_cpu_flat = collect_reservoir_from_features(
            feature_loader=tr_feat_loader,
            esn=esn,
            device=device,
            class_indices=None,
            return_mode="flat",
            to_cpu=False,
            store_dtype=torch.float32,
        )

        va_ds = TensorDataset(X_va_cpu, Y_va_cpu)
        va_feat_loader = DataLoader(va_ds, batch_size=64, shuffle=False, num_workers=0)

        H_va_seq_cpu, Y_va_ds_cpu = collect_reservoir_from_features(
            feature_loader=va_feat_loader,
            esn=esn,
            device=device,
            class_indices=None,
            return_mode="seq",
            to_cpu=False,
            store_dtype=torch.float16,
        )

        model, best = build_sed_model_with_readout(
            encoder=encoder,
            esn=esn,
            H_tr=H_tr_cpu,
            Y_tr=Y_tr_cpu_flat,
            H_va=H_va_seq_cpu,
            Y_va=Y_va_ds_cpu,
            va_meta=va_meta,
            CLASS_NAMES=CLASS_NAMES,
            nlabels=nlabels,
            readout_type=readout_type,
            readout_para=readout_para,
            posWeight=False,
            group_indices=group_indices,
            SEARCH_EPOCHS_BY_PSDS=True,
            use_amp=use_amp,
            device=device,
            return_best_info=True,
            trial=trial,
            esn_input_from="cnn",
            wantPsds=True,
            get_answer=optuna_cfg.get_answer
        )

        if readout_type == "ridge":
            readout_wb = LinearWBReadout(model.W, model.b).to(device).eval()
            psds1, win = eval_psds1_on_val_wave(
                CLASS_NAMES=CLASS_NAMES,
                readout=readout_wb,
                H_seq_cpu=H_va_seq_cpu,
                Y_ds_cpu=Y_va_ds_cpu,
                meta=va_meta,
                device=device,
                batch_size=64,
            )
            trial.set_user_attr("best_epoch", 0)
            trial.set_user_attr("median_win", int(win))
            return float(psds1)
        else:
            trial.set_user_attr("best_epoch", int(best["epoch"]))
            
            if optuna_cfg.wantPsds:
                if optuna_cfg.get_answer == "psds1":
                    psds1 = float(best["psds1"])
                    trial.set_user_attr("median_win", int(best["median_win"]))
                    return psds1
                elif optuna_cfg.get_answer == "F1_clock":
                    f1_clock = float(best["F1_clock"])
                    return f1_clock
            else:
                val_loss = float(best["loss"])
                return val_loss
            
    direction = "maximize" if optuna_cfg.wantPsds else "minimize"
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner  = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials)

    print("Best Value ({direction}):", study.best_value)
    print("Best params:", study.best_params)
    return study

def esn_para_from_best_params(p: dict) -> ESN_Para:
    return ESN_Para(
        H_ESN=1024,
        SPECTRAL_RADIUS=float(p["SPECTRAL_RADIUS"]),
        LEAKING_RATE=float(p["LEAKING_RATE"]),
        DENSITY=0.1,
        INPUT_SCALE=float(p["INPUT_SCALE"]),
        TOPOLOGY="random",
        BIDIRECTIONAL=False,
        BI_MERGE="concat",       
        BI_SHARE_WEIGHTS=False,   
    )

def readout_para_from_best_params(p: dict, study, READOUT_TYPE) -> ReadoutTrainPara:
    best_epoch = study.best_trial.user_attrs.get("best_epoch", None)
    num_epochs = int(best_epoch) if best_epoch is not None else int(p["EPOCHS"])
    if READOUT_TYPE == "ridge":
        return ReadoutTrainPara(
            LEARNING_RATE=1e-4,
            NUM_EPOCHS=100,
            L2=float(p["L2"]),
            BATCH_SIZE=100,
            PATIENCE=10,
        )
    else:
        return ReadoutTrainPara(
            LEARNING_RATE=float(p["LR"]),
            NUM_EPOCHS=100,
            L2=1e-4,
            BATCH_SIZE=1024,
            PATIENCE=100,
        )

@torch.no_grad()
def collect_reservoir_from_features(
    *,
    feature_loader: DataLoader,
    esn: nn.Module,
    device: torch.device,
    class_indices: Optional[List[int]] = None,
    return_mode: Literal["flat", "seq"] = "flat",
    to_cpu: bool = False,
    store_dtype: Optional[torch.dtype] = None,
):
    esn.eval()

    H_list, Y_list = [], []

    for batch in feature_loader:
        features, labels = batch[0], batch[1]
        features = features.to(device).float()
        labels   = labels.to(device).float()

        if class_indices is not None:
            labels = labels[:, class_indices, :]

        H_seq = esn(features)
        B, T_res, Hdim = H_seq.shape

        labels_ds = downsample_labels_to_T(labels, T_res)

        if return_mode == "flat":
            H_out = H_seq.reshape(-1, Hdim)
            Y_out = labels_ds.permute(0, 2, 1).reshape(-1, labels_ds.size(1))
        else:
            H_out = H_seq
            Y_out = labels_ds

        if to_cpu:
            dt = store_dtype if store_dtype is not None else H_out.dtype
            H_out = H_out.detach().to("cpu", dtype=dt)
            Y_out = Y_out.detach().to("cpu")

        H_list.append(H_out)
        Y_list.append(Y_out)

    if return_mode == "flat":
        H_all = torch.cat(H_list, dim=0)
        Y_all = torch.cat(Y_list, dim=0)
    else:
        H_all = torch.cat(H_list, dim=0)
        Y_all = torch.cat(Y_list, dim=0)

    return H_all, Y_all
