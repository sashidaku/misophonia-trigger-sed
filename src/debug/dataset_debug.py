import inspect
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, Subset

def describe_dataset(ds, name="dataset"):
    print(f"[{name}] type =", type(ds))
    try:
        print(f"[{name}] len  =", len(ds))
    except Exception as e:
        print(f"[{name}] len  = error: {e}")

    if isinstance(ds, ConcatDataset):
        print(f"[{name}] is ConcatDataset with {len(ds.datasets)} sub-datasets")
        for i, sub in enumerate(ds.datasets):
            print(f"  - sub[{i}] type={type(sub)} len={len(sub)} defined_in={inspect.getsourcefile(type(sub))}")
    if isinstance(ds, Subset):
        print(f"[{name}] is Subset of {type(ds.dataset)} (len(indices)={len(ds.indices)})")

def peek_loader(loader, name="loader"):
    print(f"\n=== peek {name} ===")
    b = next(iter(loader))
    print("batch type:", type(b))
    if isinstance(b, (tuple, list)):
        print("batch len :", len(b))
        for i, x in enumerate(b):
            shp = getattr(x, "shape", None)
            print(f"  [{i}] type={type(x)} shape={shp}")
        if len(b) >= 3:
            fn = b[2]
            print("filenames sample:", fn[:5] if isinstance(fn, (list, tuple)) else fn)
    else:
        keys = list(b.keys()) if hasattr(b, "keys") else None
        print("not tuple/list. keys:", keys)

def show_batch(loader, name):
    b = next(iter(loader))
    print(f"\n[{name}] batch type:", type(b))
    if isinstance(b, (list, tuple)):
        print(f"[{name}] len(batch):", len(b))
        for i, x in enumerate(b):
            print(f"  - batch[{i}] type={type(x)}",
                  f"shape={tuple(x.shape) if torch.is_tensor(x) else ''}")
    elif isinstance(b, dict):
        print(f"[{name}] keys:", b.keys())
        for k,v in b.items():
            print(f"  - {k}: type={type(v)}",
                  f"shape={tuple(v.shape) if torch.is_tensor(v) else ''}")
    else:
        print(f"[{name}] (single object) repr:", repr(b)[:200])

def show_dataset_src(loader, name):
    ds = loader.dataset
    print(f"\n[{name}] dataset class:", ds.__class__)
    print(f"[{name}] defined in:", inspect.getsourcefile(ds.__class__))

def collect_unique_filenames(loader, max_batches=10):
    fns = []
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        fn = batch[2]
        if isinstance(fn, (list, tuple)):
            fns += [str(x) for x in fn]
        else:
            fns.append(str(fn))
    fns = [Path(x).name for x in fns]
    return sorted(set(fns))

