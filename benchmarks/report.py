"""Run the probes and the speed test; write benchmarks/results.json (read by scripts/build_docs.py).

    python benchmarks/report.py            # after prep.py and embed.py; only missing entries are computed
    python benchmarks/report.py --fresh    # recompute everything

Schema:
    esc50[model] = {"folds": [5 accuracies], "mean": ..., "std": ...}
    fsdd[model]  = accuracy on the official split
    speed        = timing rows and support / embedding timings
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from probe import C, fit_score  # noqa: E402
from speed import softmax_attention, timeit  # noqa: E402
from ssmix import AudioEmbedder, ScaleSpaceMixer, SWTCrossScaleMixer  # noqa: E402

ESC50 = ["ssmix", "handcrafted_all", "compare", "vggish", "wav2vec2", "librosa", "scattering", "egemaps", "gfcc",
         "random_transformer", "mfcc", "logmel", "cqt"]
FSDD = ["ssmix", "efficientat_mn10", "random_transformer", "passt", "librosa", "logmel", "wav2vec2", "scattering",
        "mfcc", "gfcc", "cqt", "egemaps", "panns", "vggish"]
OUT = os.path.join(HERE, "results.json")


def r1(x):
    return round(100 * float(x), 1)


def esc(R):
    z = np.load(f"{C}/esc50_wav.npz")
    for n in ESC50:
        if n in R["esc50"]:
            continue
        X = np.load(f"{C}/emb_esc50_{n}.npy")
        acc = [fit_score(X[z["fold"] != f], z["y"][z["fold"] != f], X[z["fold"] == f], z["y"][z["fold"] == f])
               for f in range(1, 6)]
        R["esc50"][n] = {"folds": [r1(a) for a in acc], "mean": r1(np.mean(acc)), "std": r1(np.std(acc))}
        print("esc50", n, R["esc50"][n]["mean"], flush=True)
        save(R)


def fsdd(R):
    z = np.load(f"{C}/fsdd_wav.npz")
    te = z["take"] < 5
    for n in FSDD:
        if n in R["fsdd"]:
            continue
        X = np.load(f"{C}/emb_fsdd_{n}.npy")
        R["fsdd"][n] = r1(fit_score(X[~te], z["y"][~te], X[te], z["y"][te]))
        print("fsdd", n, R["fsdd"][n], flush=True)
        save(R)


@torch.no_grad()
def speed(R):
    if "rows" in R["speed"]:
        return
    rows = []
    for T in (1024, 4096, 16384, 65536):
        x = torch.randn(1, T, 64)
        a = timeit(ScaleSpaceMixer(order=4, levels=8, heads=8), x)
        b = timeit(SWTCrossScaleMixer(order=4, levels=8, heads=8, window=2), x, reps=3)
        c = timeit(softmax_attention, x, reps=2) if T <= 16384 else None
        rows.append([T, round(a, 1), round(b, 1), None if c is None else round(c, 1)])
    S = {"rows": rows}
    x = torch.randn(1, 16384, 64)
    for N in (1, 4, 16):
        S[f"db{N}"] = round(timeit(ScaleSpaceMixer(order=N, levels=8, heads=8), x), 1)
    z = np.load(f"{C}/esc50_wav.npz")
    emb = AudioEmbedder()
    emb(torch.from_numpy(z["w32"][:16]))
    t = time.perf_counter()
    for i in range(0, 2000, 16):
        emb(torch.from_numpy(z["w32"][i:i + 16]))
    S["esc50_seconds"] = round(time.perf_counter() - t)
    S["env"] = f"CPU, {torch.get_num_threads()} threads, torch {torch.__version__}"
    R["speed"] = S
    save(R)


def save(R):
    with open(OUT, "w") as f:
        json.dump(R, f, indent=1)


if __name__ == "__main__":
    R = {"esc50": {}, "fsdd": {}, "speed": {}}
    if os.path.exists(OUT) and "--fresh" not in sys.argv:
        with open(OUT) as f:
            old = json.load(f)
        if "esc50" in old:
            R = old
    speed(R)  # first, on an idle machine
    fsdd(R)
    esc(R)
    save(R)
    print(json.dumps({k: R[k] for k in ("fsdd",)}, indent=1))
