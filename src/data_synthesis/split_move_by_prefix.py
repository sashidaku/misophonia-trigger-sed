import argparse
import csv
import os
import re
import shutil
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma"}

@dataclass(frozen=True)
class FileRec:
    path: Path
    dataset: str
    origin: str
    domain: str
    category: str

def parse_args():
    p = argparse.ArgumentParser(
        description="Build soundbank (train/eval/test × foreground/background × class) from archive, grouping by origin."
    )
    p.add_argument("--src", required=True, type=Path, help="Archive root (scanned recursively).")
    p.add_argument("--out_root", required=True, type=Path, help="Output root for soundbank splits.")
    p.add_argument("--ratios", default="70,15,15", help="Split ratios as 'train,eval,test' (sum to 100).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--move", action="store_true", help="Move files instead of copying.")
    mode.add_argument("--link", action="store_true", help="Create symlinks instead of copying.")
    p.add_argument("--dry_run", action="store_true", help="Plan only; do not copy/move/link.")
    p.add_argument("--per_dataset_split", action="store_true", default=True,
                   help="Split independently per dataset (recommended).")
    p.add_argument("--exts", default=",".join(sorted(AUDIO_EXTS)),
                   help="Comma-separated extensions to include (lowercase, with dot).")
    p.add_argument("--omit_from_end", default="2,3",
                   help="Tail positions to omit as variable parts for origin grouping (e.g., '2,3'). 1=last, 2=second-last ...")
    return p.parse_args()

def find_audio_files(root: Path, exts: set) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def get_dataset_prefix(stem: str) -> str:
    return stem.split("_", 1)[0]

def infer_category_from_path(p: Path) -> Tuple[str, str]:
    parts = list(p.parts)
    domain = "unknown"
    category: Optional[str] = None
    for i, seg in enumerate(parts):
        low = seg.lower()
        if low in ("foreground", "background"):
            domain = "foreground" if low == "foreground" else "background"
            if i + 1 < len(parts):
                category = parts[i + 1]
            break
    if category is None:
        category = p.parent.name
    return domain, category

def plan_split(groups: List[str], ratios: Tuple[int, int, int], seed: int) -> Dict[str, str]:
    train_pct, eval_pct, test_pct = ratios
    assert train_pct + eval_pct + test_pct == 100, "ratios must sum to 100"
    rnd = random.Random(seed)
    shuffled = groups[:]
    rnd.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * train_pct / 100.0))
    n_eval  = int(round(n * eval_pct  / 100.0))
    n_test  = n - n_train - n_eval
    if n >= 3:
        if n_train == 0: n_train = 1
        if n_eval  == 0: n_eval  = 1
        n_test = n - n_train - n_eval
        if n_test  == 0: n_test  = 1; n_eval = max(1, n_eval-1)
    split_map = {}
    for g in shuffled[:n_train]:                   split_map[g] = "train"
    for g in shuffled[n_train:n_train+n_eval]:     split_map[g] = "eval"
    for g in shuffled[n_train+n_eval:]:            split_map[g] = "test"
    return split_map

def infer_origin_by_tail_omit(stem: str, dataset_prefix: str, omit_from_end=(2, 3)) -> str:
    tokens = stem.split("_")
    tokens_wo_ds = tokens[1:] if (tokens and tokens[0] == dataset_prefix) else tokens[:]
    drop_idx = set()
    L = len(tokens_wo_ds)
    for k in omit_from_end:
        if 1 <= k <= L:
            drop_idx.add(L - k)
    kept = [t for i, t in enumerate(tokens_wo_ds) if i not in drop_idx]
    if not kept:
        kept = tokens_wo_ds if tokens_wo_ds else tokens
    return "_".join(kept)

def safe_link_or_copy(src: Path, dst: Path, *, link: bool, move: bool):
    if move:
        shutil.move(str(src), str(dst))
        return
    if link:
        try:
            os.symlink(src, dst)
            return
        except Exception:
            pass
    shutil.copy2(str(src), str(dst))

def main():
    args = parse_args()
    exts = {e.strip().lower() for e in args.exts.split(",") if e.strip()}

    files = find_audio_files(args.src, exts)
    if not files:
        print(f"No audio files found under: {args.src}", file=sys.stderr)
        sys.exit(1)

    omit_from_end = tuple(int(x) for x in args.omit_from_end.split(",") if x.strip())
    recs: List[FileRec] = []
    for f in files:
        stem = f.stem
        dataset = get_dataset_prefix(stem)
        origin = infer_origin_by_tail_omit(stem, dataset_prefix=dataset, omit_from_end=omit_from_end)
        domain, category = infer_category_from_path(f)
        recs.append(FileRec(path=f, dataset=dataset, origin=origin, domain=domain, category=category))

    pre_counts = Counter((r.dataset, r.domain, r.category) for r in recs)
    print("\n[ARCHIVE] counts by dataset × domain × category (BEFORE split):")
    for (ds, dom, cat), cnt in sorted(pre_counts.items()):
        print(f"  {ds:15s} {dom:11s} {cat:20s} : {cnt}")

    origin_to_datasets: Dict[str, set] = defaultdict(set)
    for r in recs:
        origin_to_datasets[r.origin].add(r.dataset)
    suspicious = {o: s for o, s in origin_to_datasets.items() if len(s) > 1}
    if suspicious:
        print("\n[WARN] Same 'origin' appears in multiple datasets; check naming/group rule:")
        for o, s in list(suspicious.items())[:50]:
            print(f"  origin='{o}' -> datasets={sorted(s)}")
        if len(suspicious) > 50:
            print(f"  ... and {len(suspicious)-50} more")

    ratios = tuple(int(x) for x in args.ratios.split(","))
    assert len(ratios) == 3, "ratios must be three integers: train,eval,test"

    split_of_group: Dict[Tuple[str, str], str] = {}
    if args.per_dataset_split:
        by_ds: Dict[str, List[str]] = defaultdict(list)
        for r in recs:
            by_ds[r.dataset].append(r.origin)
        for ds, origins in by_ds.items():
            uniq = sorted(set(origins))
            ds_seed = args.seed + (hash(ds) % 1000000)
            m = plan_split(uniq, ratios, ds_seed)
            for g, split in m.items():
                split_of_group[(ds, g)] = split
    else:
        all_groups = sorted({(r.dataset, r.origin) for r in recs})
        rnd = random.Random(args.seed)
        rnd.shuffle(all_groups)
        keys = [f"{ds}|{orig}" for ds, orig in all_groups]
        m = plan_split(keys, ratios, args.seed)
        for (ds, orig), key in zip(all_groups, keys):
            split_of_group[(ds, orig)] = m[key]

    rec_by_path: Dict[Path, FileRec] = {r.path: r for r in recs}
    plan: List[Tuple[Path, Path, str, str]] = []
    for r in recs:
        split = split_of_group[(r.dataset, r.origin)]
        dom = r.domain if r.domain in ("foreground", "background") else "foreground"
        dst_dir = args.out_root / split / "soundbank" / dom / r.category
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / r.path.name
        plan.append((r.path, dst_path, r.dataset, split))

    post_counts = Counter()
    for src, dst, ds, split in plan:
        r = rec_by_path[src]
        post_counts[(split, r.domain if r.domain in ("foreground","background") else "foreground", r.category)] += 1

    print("\n[PLAN] counts by split × domain × category (soundbank AFTER split):")
    for (split, dom, cat), cnt in sorted(post_counts.items()):
        print(f"  {split:5s} {dom:11s} {cat:20s} : {cnt}")

    if args.dry_run:
        print("\n[DRY-RUN] No files were linked/copied/moved.")
    else:
        for src, dst, _, _ in plan:
            safe_link_or_copy(src, dst, link=args.link, move=args.move)

        (args.out_root / "reports").mkdir(parents=True, exist_ok=True)
        pre_csv  = args.out_root / "reports" / "counts_pre_split.csv"
        post_csv = args.out_root / "reports" / "counts_by_split_soundbank.csv"

        with pre_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["dataset", "domain", "category", "n_files"])
            for (ds, dom, cat), cnt in sorted(pre_counts.items()):
                w.writerow([ds, dom, cat, cnt])

        with post_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["split", "domain", "category", "n_files"])
            for (split, dom, cat), cnt in sorted(post_counts.items()):
                w.writerow([split, dom, cat, cnt])

        print(f"\nSaved summaries:\n  {pre_csv}\n  {post_csv}")

if __name__ == "__main__":
    main()
