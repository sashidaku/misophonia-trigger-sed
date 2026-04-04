from src.evaluation.inference import pred_esn
from src.data.get_data import build_mel_and_cnn, get_dataloaders
from src.models.encorders.common_audio_encorder import CommonAudioEncoder, compute_cnnseq_mean_std
from src.models.model_utils import get_peak_mib, list_trainable_params, reset_peak
from src.models.temporal_modules.esn import ReadoutTrainPara
from src.models.temporal_modules.lstm import BiLSTMFrameHead, LSTM_Para
from src.models.wrappers.gru_sed import SEDModelEncoderRNN
from train_gru import collect_cnn_features_on_gpu, train_rnn_head_on_gpu

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch

from config_loader import data_cfg, esn_cfg, lstm_cfg

def run_lstm(
    *,
    encoder,
    CLASS_NAMES,
    train_loader,
    val_loader,
    test_loader,
    lstm_para: LSTM_Para,
    readout_para: ReadoutTrainPara,
    device: torch.device,
    use_amp: bool = False,
):
    encoder = encoder.to(device).eval()
    peak_log = {}

    # 1) train/val で cnn_seq をキャッシュ（RNN学習用）
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

    rnn_head = BiLSTMFrameHead(in_dim=D_in, n_classes=n_classes, para=lstm_para).to(device)
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
    Y_te, pred_te, te_meta, Y_va2, pred_va2, va_meta2, auc, psds1 = pred_esn(
        val_loader, test_loader, sed_model, CLASS_NAMES
    )
    peak_log["infer"] = get_peak_mib(device)

    for k, v in peak_log.items():
        if v is None:
            continue
        alloc, resv = v
        print(f"[PEAK] {k}: allocated={alloc:.1f} MiB, reserved={resv:.1f} MiB")

    if device.type == "cuda":
        worst = max(v[1] for v in peak_log.values() if v is not None)
        print(f"[PEAK] worst reserved across stages = {worst:.1f} MiB")

    return Y_te, pred_te, te_meta, Y_va2, pred_va2, va_meta2, auc, psds1

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
        train_num = data_cfg.train_num,
        eval_num = data_cfg.val_num,
        test_num = data_cfg.test_num
    )

    mean, std = compute_cnnseq_mean_std(encoder, train_loader, device, use_amp=False)
    encoder.set_feature_norm(mean, std)

    lstm_para = LSTM_Para(RNN_DIM=lstm_cfg.rnn_dim, NUM_LAYERS=lstm_cfg.num_layers, DROPOUT=lstm_cfg.dropout, BIDIR=lstm_cfg.bidir)
    readout_para = ReadoutTrainPara(
                LEARNING_RATE=lstm_cfg.readout_para.learning_rate,
                NUM_EPOCHS=lstm_cfg.readout_para.num_epochs,
                L2=lstm_cfg.readout_para.l2,
                BATCH_SIZE=lstm_cfg.readout_para.batch_size,
                PATIENCE=lstm_cfg.readout_para.patience,
            )
    run_lstm(
        encoder=encoder,
        CLASS_NAMES=data_cfg.class_names,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lstm_para=lstm_para,
        readout_para=readout_para,
        device=device,
        use_amp=False,
    )

if __name__ == "__main__":
    main_do_rnn()