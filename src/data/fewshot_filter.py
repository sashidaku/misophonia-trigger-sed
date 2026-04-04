import importlib
from pathlib import Path
import re
import sys
from typing import Any, Dict, List
from torch.utils.data import ConcatDataset, Subset
from typing import Any, Dict, List, Literal, Optional, Sequence, Union
from data.misophonia import TenSecondSEDDataset


def get_filtered_dataset_from_file(
    original_dataset,
    config_file_path: Union[str, Path],
    list_name: Union[str, Sequence[str]],
    target_flags: Sequence[str],
    logic: Literal["OR", "AND"] = "OR",
    keep_if_not_match: bool = True,
    include_missing_in_original: bool = False,
    keep_event_labels: Optional[Sequence[str]] = None,
):

    base = original_dataset.dataset if isinstance(original_dataset, Subset) else original_dataset

    file_path = Path(config_file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")

    spec = importlib.util.spec_from_file_location("dynamic_config", file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dynamic_config"] = module
    spec.loader.exec_module(module)

    def _collect_lists(module, list_name) -> List[Dict[str, Any]]:
        names: List[str] = []
        if isinstance(list_name, (list, tuple)):
            names = list(list_name)
        else:
            if any(ch in list_name for ch in r".*?[]()|+^$\\"):
                pat = re.compile(list_name)
                names = [n for n in dir(module) if pat.fullmatch(n)]
            else:
                names = [list_name]

        collected: List[Dict[str, Any]] = []
        missing_vars: List[str] = []
        for n in names:
            if not hasattr(module, n):
                missing_vars.append(n)
                continue
            v = getattr(module, n)
            if not isinstance(v, list):
                continue
            collected.extend(v)
        if len(collected) == 0:
            raise AttributeError(f"No valid list found for list_name={list_name} in {file_path.name}. "
                                 f"missing_vars={missing_vars}")
        return collected

    target_configs = _collect_lists(module, list_name)

    key_by_norm = {Path(k).name: k for k in base.data.keys()}

    new_data = {}
    missing_in_original = 0
    kept_labeled = 0
    kept_unlabeled = 0

    logic = logic.upper()

    for item in target_configs:
        if "filename" not in item:
            continue
        fname_norm = Path(item["filename"]).name

        in_original = fname_norm in key_by_norm
        if not in_original and not include_missing_in_original:
            missing_in_original += 1
            continue

        fname_key = key_by_norm.get(fname_norm, fname_norm)

        flag_status = [item.get(flag, 0) == 1 for flag in target_flags]
        if len(target_flags) == 0:
            is_target = True
        else:
            flag_status = [item.get(flag, 0) == 1 for flag in target_flags]
            is_target = (all(flag_status) if logic == "AND" else any(flag_status))

        if not is_target:
            if keep_if_not_match:
                new_data[fname_key] = []
                kept_unlabeled += 1
            continue

        raw_events = base.data.get(fname_key, [])

        if len(target_flags) == 0:
            if keep_event_labels is None:
                new_data[fname_key] = raw_events
            else:
                keep_set = set(keep_event_labels)
                new_data[fname_key] = [e for e in raw_events if e.get("event_label") in keep_set]
            kept_labeled += 1
            continue

        flag_status = [item.get(flag, 0) == 1 for flag in target_flags]
        is_target = (all(flag_status) if logic == "AND" else any(flag_status))

        if not is_target:
            if keep_if_not_match:
                new_data[fname_key] = []
                kept_unlabeled += 1
            continue

        if keep_event_labels is None:
            filtered_events = raw_events
        else:
            keep_set = set(keep_event_labels)
            filtered_events = [e for e in raw_events if e.get("event_label") in keep_set]

        new_data[fname_key] = filtered_events
        kept_labeled += 1

    new_dataset = TenSecondSEDDataset(
        data=new_data,
        audio_dir=base.audio_dir,
        sample_rate=base.sample_rate,
        label_fps=base.label_fps,
        label_to_idx=base.label_to_idx,
        nlabels=base.nlabels,
        target_classes=getattr(base, "target_set", None),
        return_only_target=getattr(base, "return_only_target", False),
    )

    print(f"[Dataset Filter] loaded from {file_path.name}, list_name={list_name}")
    print(f" - logic={logic}, flags={list(target_flags)}")
    print(f" - keep_if_not_match={keep_if_not_match}, include_missing_in_original={include_missing_in_original}")
    print(f" - keep_event_labels={keep_event_labels}")
    print(f" - input items: {len(target_configs)}")
    print(f" - kept labeled   : {kept_labeled}")
    print(f" - kept unlabeled : {kept_unlabeled}")
    print(f" - missing in original (skipped): {missing_in_original}")
    print(f" - output dataset : {len(new_dataset)}")

    return new_dataset

