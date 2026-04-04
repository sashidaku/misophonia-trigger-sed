import dataclasses
import os
import shutil

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from config_loader import fewshot_cfg, esn_cfg, data_cfg
from src.models.temporal_modules.esn import ESN_Para, ReadoutTrainPara
import run_fewshot

READOUT_PARAS = [
    ReadoutTrainPara(LEARNING_RATE=0.002045575378186123, NUM_EPOCHS=200, L2=1e-4, BATCH_SIZE=1024, PATIENCE=100),
    ReadoutTrainPara(LEARNING_RATE=0.00284360718453597, NUM_EPOCHS=200, L2=1e-4, BATCH_SIZE=1024, PATIENCE=100),
    ReadoutTrainPara(LEARNING_RATE=0.0019311395900108348, NUM_EPOCHS=200,  L2=5e-5, BATCH_SIZE=1024, PATIENCE=100),
    ReadoutTrainPara(LEARNING_RATE=0.0007833625704676601, NUM_EPOCHS=200, L2=1e-3, BATCH_SIZE=1024, PATIENCE=100),
    ReadoutTrainPara(LEARNING_RATE=0.0006842213571315586, NUM_EPOCHS=200, L2=1e-4, BATCH_SIZE=1024, PATIENCE=100),
]

ESN_PARAS = [
    ESN_Para(H_ESN=256, SPECTRAL_RADIUS=0.370726231456357, LEAKING_RATE=0.07895580952517339, DENSITY=0.1, INPUT_SCALE=0.3113707938521287,
            TOPOLOGY="random", BIDIRECTIONAL=True, BI_MERGE="concat", BI_SHARE_WEIGHTS=False),
    ESN_Para(H_ESN=256, SPECTRAL_RADIUS=0.4162439003603828, LEAKING_RATE=0.10647092779214465,  DENSITY=0.1, INPUT_SCALE=0.8611438833649729,
            TOPOLOGY="random", BIDIRECTIONAL=True, BI_MERGE="concat", BI_SHARE_WEIGHTS=False),
    ESN_Para(H_ESN=256,  SPECTRAL_RADIUS=0.5821564526511028, LEAKING_RATE=0.10082858272524339,  DENSITY=0.1, INPUT_SCALE=0.6334485791087644,
            TOPOLOGY="random", BIDIRECTIONAL=True, BI_MERGE="concat",    BI_SHARE_WEIGHTS=False),
    ESN_Para(H_ESN=256, SPECTRAL_RADIUS=0.5647251883157431, LEAKING_RATE=0.0618669837992362,  DENSITY=0.1, INPUT_SCALE=1.489546500042098,
            TOPOLOGY="random", BIDIRECTIONAL=True, BI_MERGE="concat", BI_SHARE_WEIGHTS=False),
    ESN_Para(H_ESN=256, SPECTRAL_RADIUS=0.1663710725857328, LEAKING_RATE=0.05520538425124629,  DENSITY=0.1, INPUT_SCALE=1.261901680995084,
            TOPOLOGY="random", BIDIRECTIONAL=True, BI_MERGE="concat", BI_SHARE_WEIGHTS=False),
]

def pick_files(k: int, seed: int):
    rng = np.random.default_rng(seed)
    order = rng.permutation(fewshot_cfg.all_files)
    return [str(f) for f in order[-k:]]

def move_tsv_out(dst_dir: str):
    src = "tsv_out"
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        shutil.move(src, dst_dir)

rows = []

for rt in fewshot_cfg.compare_readout_which:

    for k in fewshot_cfg.ks:
        for seed in fewshot_cfg.seeds:
            files = pick_files(k, seed)
            
            readout_para = dataclasses.replace(READOUT_PARAS[k-1])
            esn_para     = dataclasses.replace(ESN_PARAS[k-1])
            if rt == "rnn":
                readout_para = ReadoutTrainPara(
                    LEARNING_RATE=1e-4,
                    NUM_EPOCHS=200,
                    L2=0.0001,
                    BATCH_SIZE=1024,
                    PATIENCE=100,
                )
            auc, psds1 = run_fewshot.main_do_esn_fewshot(
                seed=seed,
                include_fnames=files,
                k_shot=k,
                READOUT_TYPE=rt,
                readout_para=readout_para,
                esn_para=esn_para
            )

            move_tsv_out(f"results/tsv_out/{rt}/k{k}/seed{seed}")
            if rt == "logistic":
                rows.append({
                    "readout": "BiESN",
                    "k": k,
                    "seed": seed,
                    "auc": float(auc),
                    "psds1": float(psds1),
                    "files": ",".join(files),
                })
            else:
                rows.append({
                    "readout": "BiGRU",
                    "k": k,
                    "seed": seed,
                    "auc": float(auc),
                    "psds1": float(psds1),
                    "files": ",".join(files),
                })

df = pd.DataFrame(rows)
df.to_csv("fewshot_results.csv", index=False)
print("[saved] fewshot_results.csv")

metric = "psds1"

agg = (df.groupby(["readout", "k"], as_index=False)
         .agg(mean=(metric, "mean"),
              std=(metric, "std")))

plt.figure()
for rt in fewshot_cfg.compare_readout_name:
    sub = agg[agg["readout"] == rt].sort_values("k")
    plt.errorbar(
        sub["k"], sub["mean"], yerr=sub["std"],
        capsize=3, marker="o", label=rt
    )

plt.xlabel(f"Number of support clips, K (clips)")
plt.ylabel(f"PSDS1 score")
plt.title(f"Few-shot performance ({metric}): mean±std over seeds {fewshot_cfg.seeds}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"fewshot_{metric}_fix_256_fix_ran.pdf", dpi=200)
plt.show()
print(f"[saved] fewshot_{metric}.pdf")
