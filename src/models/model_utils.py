from typing import Tuple
from torch.utils.data import DataLoader, TensorDataset
import torch

from src.models.encorders.common_audio_encorder import EncodeSpec
from config_loader import data_cfg

def list_trainable_params(model: torch.nn.Module, *, verbose: bool = True) -> Tuple[int, int]:
    total = 0
    trainable = 0
    rows = []
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
            rows.append((name, tuple(p.shape), n, str(p.dtype), p.device.type))

    if verbose:
        print("=== Trainable parameters (requires_grad=True) ===")
        print(f"Trainable: {trainable:,} / Total: {total:,} ({trainable/ max(total,1)*100:.2f}%)")
    return trainable, total

def get_peak_mib(device: torch.device):
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    alloc = torch.cuda.max_memory_allocated(device)
    resv  = torch.cuda.max_memory_reserved(device)
    mib = 1024**2
    return alloc / mib, resv / mib

def reset_peak(device: torch.device):
    if device.type != "cuda":
        return
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

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
        sample_rate = data_cfg.sample_rate

    for batch in wave_loader:
        audio, labels, filenames, durations = batch
        audio  = audio.to(device)
        labels = labels.to(device).float()
        if esn_input_from == "cnn":
            feats = encoder(
                audio,
                spec=EncodeSpec(out="cnn_seq", detach=True, use_amp=use_amp, esn_input_from=esn_input_from),
                esn=None,
            )
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

