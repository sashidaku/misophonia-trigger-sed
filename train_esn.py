import torch

from src.data.get_data import build_mel_and_cnn, get_dataloaders
from src.models.encorders.common_audio_encorder import CommonAudioEncoder, EncodeSpec, compute_cnnseq_mean_std
import torch.nn as nn

from src.models.model_utils import list_trainable_params
from src.models.temporal_modules.esn import ESN_Para, ReadoutTrainPara
from src.training.esn_trainer import run_esn
from config_loader import data_cfg, esn_cfg

def main_do_esn():
    print(data_cfg.class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mel_extractor, fmn = build_mel_and_cnn(device)

    encoder = CommonAudioEncoder(
        mel_extractor=mel_extractor,
        fmn=fmn,
        device=device,
        default_std_input=esn_cfg.use_esn_input_std,
        expected_T=data_cfg.frames_1s * 10,
    ).to(device)

    train_loader, val_loader, test_loader = get_dataloaders(
        data_cfg.audio_path, data_cfg.tsv_path,
        with_background=data_cfg.with_background,
        background_ratio=data_cfg.background_ratio,
        TRIGGER_CLASSES=data_cfg.trigger,
        train_num = data_cfg.train_num,
        eval_num = data_cfg.val_num,
        test_num = data_cfg.test_num,
    )

    if esn_cfg.cnn_type == "batch":
        encoder.eval()

        for m in encoder.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.train()

        with torch.no_grad():
            for audio, *_ in train_loader:
                audio = audio.to(device, non_blocking=True)
                _ = encoder(
                    audio,
                    spec=EncodeSpec(out="cnn_seq", detach=True, use_amp=False, std_input=False),
                    esn=None,
                )

        encoder.eval()

    list_trainable_params(fmn)

    mean, std = compute_cnnseq_mean_std(encoder, train_loader, device, use_amp=False)
    encoder.set_feature_norm(mean, std)
    
    readout_para = ReadoutTrainPara(
                LEARNING_RATE=esn_cfg.readout_para.learning_rate,
                NUM_EPOCHS=esn_cfg.readout_para.num_epochs,
                L2=esn_cfg.readout_para.l2,
                BATCH_SIZE=esn_cfg.readout_para.batch_size,
                PATIENCE=esn_cfg.readout_para.patience,
            )
    esn_para = ESN_Para(
                H_ESN=esn_cfg.esn_para.num_reservoir,
                SPECTRAL_RADIUS=esn_cfg.esn_para.spectral_radius,
                LEAKING_RATE=esn_cfg.esn_para.leak_rate,
                DENSITY=esn_cfg.esn_para.density,
                INPUT_SCALE=esn_cfg.esn_para.input_scale,
                TOPOLOGY=esn_cfg.esn_para.topology,
                BIDIRECTIONAL=esn_cfg.esn_para.bidirectional,
                BI_MERGE=esn_cfg.esn_para.bi_merge,
                BI_SHARE_WEIGHTS=esn_cfg.esn_para.bi_share_weights,
            )
    run_esn(
        encoder=encoder,
        task_path = data_cfg.tsv_path,
        CLASS_NAMES=data_cfg.class_names,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        readout_type=esn_cfg.readout_type,
        readout_para=readout_para,
        out=esn_cfg.out,
        BY_EPOCH=True,
        posWeight=False,
        group_indices=esn_cfg.group_indices,
        esn_para=esn_para,
        esn_input_from=esn_cfg.esn_input_form,
    )

if __name__ == "__main__":
    main_do_esn()