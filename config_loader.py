from pathlib import Path
import yaml
from types import SimpleNamespace

def to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_namespace(x) for x in obj]
    return obj

def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return to_namespace(yaml.safe_load(f))

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs"

data_cfg = load_yaml(CONFIG_DIR / "data" / "default.yaml")
esn_cfg = load_yaml(CONFIG_DIR / "experiment" / "esn.yaml")
gru_cfg = load_yaml(CONFIG_DIR / "experiment" / "gru.yaml")
linear_cfg = load_yaml(CONFIG_DIR / "experiment" / "linear.yaml")
lstm_cfg = load_yaml(CONFIG_DIR / "experiment" / "lstm.yaml")
fewshot_cfg = load_yaml(CONFIG_DIR / "experiment" / "fewshot.yaml")
optuna_cfg = load_yaml(CONFIG_DIR / "experiment" / "optuna.yaml")