import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

def compute_pos_weight_from_Y(Y_tr: torch.Tensor, eps: float = 1e-6, clamp_max: float = 5.0):
    Y = Y_tr.float()
    pos = Y.sum(dim=0)
    neg = Y.size(0) - pos
    pw = neg / (pos + eps)
    print("positive weight", pw)
    pw = torch.clamp(pw, max=clamp_max)
    return pw

def make_reservoir_loader(H, Y, batch_size=2048, shuffle=True):
    dataset = TensorDataset(
        torch.from_numpy(H).float(),
        torch.from_numpy(Y).float()
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

