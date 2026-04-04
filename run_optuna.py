import torch

from src.data.get_data import build_mel_and_cnn, get_dataloaders
from src.models.encorders.common_audio_encorder import CommonAudioEncoder, compute_cnnseq_mean_std
from src.search.esn_optuna import esn_para_from_best_params, optuna_search_esn_readout, readout_para_from_best_params
from src.training.esn_trainer import run_esn

from config_loader import data_cfg, esn_cfg, optuna_cfg

def run_best_from_study(
    *,
    study,
    encoder,
    task_path,
    CLASS_NAMES,
    train_loader,
    val_loader,
    test_loader,
    readout_type="logistic",
    group_indices=None,
    posWeight=False,
    use_amp=False,
):
    p = study.best_params

    best_esn_para = esn_para_from_best_params(p)
    best_readout_para = readout_para_from_best_params(p, study, readout_type)

    print("=== Best trial ===")
    print("best_value (PSDS1):", study.best_value)
    print("best_params:", p)
    print("user_attrs:", study.best_trial.user_attrs)
    print("ESN_Para:", best_esn_para)
    print("ReadoutTrainPara:", best_readout_para)

    Y_te, pred_te, te_meta, Y_va, pred_va, va_meta, auc, psds1 = run_esn(
                                                                    encoder=encoder,
                                                                    task_path=task_path,
                                                                    CLASS_NAMES=CLASS_NAMES,
                                                                    train_loader=train_loader,
                                                                    val_loader=val_loader,
                                                                    test_loader=test_loader,
                                                                    readout_type=readout_type,
                                                                    posWeight=posWeight,
                                                                    group_indices=group_indices,
                                                                    esn_para=best_esn_para,
                                                                    readout_para=best_readout_para,
                                                                    use_amp=use_amp,
                                                                    esn_input_from="cnn"
                                                                )
    print("TESTの形", Y_te.shape, "VALの形", Y_va.shape)


def main_do_esn_optuna():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mel_extractor, fmn = build_mel_and_cnn(device)

    encoder = CommonAudioEncoder(
        mel_extractor=mel_extractor,
        fmn=fmn,
        device=device,
        default_std_input=esn_cfg.use_esn_input_std,
        expected_T=data_cfg.frames_1s * 10,
    ).to(device)

    if optuna_cfg.fewshot == False:
        train_loader, val_loader, test_loader = get_dataloaders(
            data_cfg.audio_path, data_cfg.tsv_path,
            with_background=data_cfg.with_background,
            background_ratio=data_cfg.background_ratio,
            TRIGGER_CLASSES=data_cfg.trigger,
            train_num = data_cfg.train_num,
            eval_num = data_cfg.val_num,
            test_num = data_cfg.test_num
        )
    else:
        print("FEWSHOTのOptuna")
        train_loader, val_loader, test_loader = get_dataloaders(
            data_cfg.audio_path, data_cfg.tsv_path,
            WITH_BACKGROUND=False,
            BACKGROUND_RATIO=data_cfg.background_ratio,
            TRIGGER_CLASSES=data_cfg.trigger,
            TRAIN_NUM = 6000,
            VAL_NUM = 200,
            TEST_NUM = 200,
            K_SHOT = 5,
            FEWSHOT = False,
            SEED = 0,
            include_fnames=[
                '03.wav',
                '1001.wav',
                '1004.wav',
                '1017.wav',
                '1038.wav',
                '1047.wav',
                '1050.wav',
                '1270.wav',
            ]
        )

    mean, std = compute_cnnseq_mean_std(encoder, train_loader, device, use_amp=False)
    encoder.set_feature_norm(mean, std)

    nlabels = len(data_cfg.class_names)

    study = optuna_search_esn_readout(
        encoder=encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        CLASS_NAMES=data_cfg.class_names,
        readout_type=esn_cfg.readout_type,
        nlabels=nlabels,
        group_indices=esn_cfg.group_indices,
        use_amp=False,
        n_trials=optuna_cfg.n_trials,
        device=device,
    )


    run_best_from_study(
        study=study,
        encoder=encoder,
        task_path=data_cfg.tsv_path,
        CLASS_NAMES=data_cfg.class_names,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        readout_type=esn_cfg.readout_type,
        group_indices=esn_cfg.group_indices,
        posWeight=False,
        use_amp=False,
    )

if __name__ == "__main__":
    main_do_esn_optuna()