from pathlib import Path

from src.data.get_data import build_mel_and_cnn, get_dataloaders
from src.debug.dataset_debug import collect_unique_filenames, describe_dataset
from src.models.encorders.common_audio_encorder import CommonAudioEncoder, compute_cnnseq_mean_std
from src.models.temporal_modules.esn import ESN_Para, ReadoutTrainPara
from src.models.temporal_modules.gru import GRU_Para
from train_gru import run_gru

from config_loader import data_cfg, esn_cfg, fewshot_cfg
from src.training.esn_trainer import run_esn

readout_para = ReadoutTrainPara(
                LEARNING_RATE=fewshot_cfg.readout_para.learning_rate,
                NUM_EPOCHS=fewshot_cfg.readout_para.num_epochs,
                L2=fewshot_cfg.readout_para.l2,
                BATCH_SIZE=fewshot_cfg.readout_para.batch_size,
                PATIENCE=fewshot_cfg.readout_para.patience,
            )
esn_para = ESN_Para(
            H_ESN=fewshot_cfg.esn_para.H_ESN,
            SPECTRAL_RADIUS=fewshot_cfg.esn_para.SPECTRAL_RADIUS,
            LEAKING_RATE=fewshot_cfg.esn_para.LEAKING_RATE,
            DENSITY=fewshot_cfg.esn_para.DENSITY,
            INPUT_SCALE=fewshot_cfg.esn_para.INPUT_SCALE,
            TOPOLOGY=fewshot_cfg.esn_para.TOPOLOGY,
            BIDIRECTIONAL=fewshot_cfg.esn_para.BIDIRECTIONAL,
            BI_MERGE=fewshot_cfg.esn_para.BI_MERGE,
            BI_SHARE_WEIGHTS=fewshot_cfg.esn_para.BI_SHARE_WEIGHTS,
        )
def main_do_esn_fewshot(
        *,
        seed: int = 0,
        include_fnames=[
            '03.wav',
            '1001.wav',
            # # '1004.wav',
            # # '1017.wav',
            '1038.wav',
            '1047.wav',
            '1050.wav',
        ],
        k_shot: int = 7000,
        READOUT_TYPE=fewshot_cfg.readout_type,
        readout_para=readout_para,
        esn_para=esn_para
        ):
    include_fnames = list(include_fnames)
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        train_num = 6000,
        eval_num = 200,
        test_num = 120,
        SEED = seed,
        include_fnames=include_fnames
    )
    print("include_fnames =", [Path(x).name for x in include_fnames])

    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        uniq = collect_unique_filenames(loader, max_batches=50)
        print(f"[{name}] unique filenames (sample) =", uniq[:20], " ... total_unique=", len(uniq))
        miss = set([Path(x).name for x in include_fnames]) - set(uniq)
        if miss:
            print(f"[{name}] MISSING in loader:", sorted(miss))

    describe_dataset(train_loader.dataset, "train_full.dataset")


    mean, std = compute_cnnseq_mean_std(encoder, train_loader, device, use_amp=False)
    encoder.set_feature_norm(mean, std)

    if READOUT_TYPE == "logistic" or READOUT_TYPE == "ridge":
        Y_te, pred_te, te_meta, Y_va, pred_va, va_meta, auc, psds1 = run_esn(
                encoder=encoder,
                task_path=data_cfg.tsv_path,
                CLASS_NAMES=fewshot_cfg.class_names,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                readout_type=READOUT_TYPE,
                readout_para=readout_para,
                BY_EPOCH = True,
                out=esn_cfg.out,
                posWeight=False,
                group_indices=esn_cfg.group_indices,
                esn_para=esn_para,
                esn_input_from=esn_cfg.esn_input_form,
            )
    if READOUT_TYPE == "rnn":
        gru_para = GRU_Para(RNN_DIM=256, NUM_LAYERS=2, DROPOUT=0.3, BIDIR=True)
        Y_te, pred_te, te_meta, Y_va, pred_va, va_meta, auc, psds1 = run_gru(
            encoder=encoder,
            CLASS_NAMES=fewshot_cfg.class_names,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            gru_para=gru_para,
            readout_para=readout_para,
            device=device,
            use_amp=False,
        )
    return auc, psds1

if __name__ == "__main__":
    main_do_esn_fewshot()