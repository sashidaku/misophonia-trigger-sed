import numpy as np
import torch
from torch.utils.data import Subset, DataLoader


def make_fewshot_support_loader_from_dataset(
    dataset,
    class_names,
    k_shot: int = 5,
    batch_size: int = 16,
    seed: int = 0,
    num_workers: int = 0,
    debug_print: bool = True,
):
    rng = np.random.default_rng(seed)
    C = len(class_names)

    per_class = [[] for _ in range(C)]
    for idx in range(len(dataset)):
        audio, labels, fn, dur = dataset[idx]
        if torch.is_tensor(labels):
            present = (labels.float().sum(dim=-1) > 0)
            present = present.cpu().numpy().astype(bool)
        else:
            present = (labels.sum(axis=-1) > 0)

        for c in range(C):
            if present[c]:
                per_class[c].append(idx)

    chosen = []
    for c in range(C):
        idxs = per_class[c]
        rng.shuffle(idxs)
        chosen += idxs[:k_shot]

    chosen = sorted(set(chosen))
    support_ds = Subset(dataset, chosen)
    support_loader = DataLoader(
        support_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    print(f"[few-shot] support clips = {len(support_ds)} (k={k_shot}, C={C})")

    chosen = sorted(set(chosen))
    return support_loader, chosen