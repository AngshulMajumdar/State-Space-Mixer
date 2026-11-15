"""Identical frozen-embedding linear probe for every model.

StandardScaler + multinomial logistic regression; C chosen by 3-fold CV on the
training portion only, from {1e-3, 1e-2, 1e-1, 1}.
ESC-50: official 5-fold cross-validation (mean +/- std over folds).
FSDD:   official split (take 0-4 test, 500 clips) and speaker-held-out (leave one
        of the 6 speakers out, mean +/- std).
"""
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks", "cache")


def fit_score(Xtr, ytr, Xte, yte):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    gs = GridSearchCV(pipe, {"logisticregression__C": [1e-3, 1e-2, 1e-1, 1]},
                      cv=StratifiedKFold(3, shuffle=True, random_state=0), n_jobs=2)
    gs.fit(Xtr, ytr)
    return gs.score(Xte, yte)


def esc50(name):
    z = np.load(f"{C}/esc50_wav.npz")
    X = np.load(f"{C}/emb_esc50_{name}.npy")
    accs = [fit_score(X[z["fold"] != f], z["y"][z["fold"] != f], X[z["fold"] == f], z["y"][z["fold"] == f])
            for f in range(1, 6)]
    return np.mean(accs), np.std(accs)


def fsdd(name):
    z = np.load(f"{C}/fsdd_wav.npz")
    X = np.load(f"{C}/emb_fsdd_{name}.npy")
    te = z["take"] < 5
    official = fit_score(X[~te], z["y"][~te], X[te], z["y"][te])
    loso = [fit_score(X[z["speaker"] != s], z["y"][z["speaker"] != s], X[z["speaker"] == s], z["y"][z["speaker"] == s])
            for s in np.unique(z["speaker"])]
    return official, np.mean(loso), np.std(loso)


if __name__ == "__main__":
    ds = sys.argv[1]
    for name in sys.argv[2:]:
        if ds == "esc50":
            m, s = esc50(name)
            print(f"esc50 {name:22s} {100*m:5.1f} +/- {100*s:4.1f}", flush=True)
        else:
            o, m, s = fsdd(name)
            print(f"fsdd  {name:22s} official {100*o:5.1f}   speaker-out {100*m:5.1f} +/- {100*s:4.1f}", flush=True)
