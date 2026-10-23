"""Tiny CSV command line: ``python -m gbm.cli train|predict``."""
import argparse
import csv
import sys

import numpy as np

from . import GBMClassifier, GBMRegressor, load_model, metrics


def read_csv(path):
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    header, body = rows[0], rows[1:]
    def num(s):
        s = s.strip()
        return np.nan if s == "" or s.lower() in ("nan", "na", "null") else float(s)
    return header, body, num


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gbm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("csv"); t.add_argument("--target", required=True)
    t.add_argument("--task", choices=["regression", "classification", "poisson"], default="regression")
    t.add_argument("--out", default="model.json")
    t.add_argument("--rounds", type=int, default=200)
    t.add_argument("--lr", type=float, default=0.1)
    t.add_argument("--leaves", type=int, default=15)
    t.add_argument("--valid-frac", type=float, default=0.2)
    t.add_argument("--early-stopping", type=int, default=20)
    p = sub.add_parser("predict")
    p.add_argument("model"); p.add_argument("csv")
    a = ap.parse_args(argv)

    if a.cmd == "train":
        header, body, num = read_csv(a.csv)
        ti = header.index(a.target)
        feats = [i for i in range(len(header)) if i != ti]
        X = np.array([[num(r[i]) for i in feats] for r in body])
        raw_y = [r[ti] for r in body]
        y = np.array(raw_y) if a.task == "classification" else np.array([float(v) for v in raw_y])
        rng = np.random.RandomState(0)
        perm = rng.permutation(len(y))
        nv = int(len(y) * a.valid_frac)
        va, tr = perm[:nv], perm[nv:]
        kw = dict(n_estimators=a.rounds, learning_rate=a.lr, max_leaves=a.leaves)
        if a.task == "classification":
            m = GBMClassifier(**kw)
        else:
            m = GBMRegressor(loss="poisson" if a.task == "poisson" else "squared", **kw)
        m.fit(X[tr], y[tr], eval_set=(X[va], y[va]) if nv else None,
              early_stopping_rounds=a.early_stopping if nv else None, verbose=25)
        m.save(a.out)
        print("saved %s (%d rounds, best=%d)" % (a.out, len(m.trees_), m.best_iteration_))
        names = [header[i] for i in feats]
        for n_, v in sorted(zip(names, m.feature_importances()), key=lambda z: -z[1]):
            print("  %-20s %.3f" % (n_, v))
    else:
        m = load_model(a.model)
        header, body, num = read_csv(a.csv)
        X = np.array([[num(v) for v in r][:m.n_features_] for r in body])
        w = csv.writer(sys.stdout)
        for v in m.predict(X):
            w.writerow([v])


if __name__ == "__main__":
    main()
