"""End-to-end demo on synthetic data: accuracy, early stopping, missing values, monotone constraints.

Run from the repository root:  python examples/demo.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gbm import GBMClassifier, GBMRegressor, metrics  # noqa: E402


def make_regression(n, rng):
    X = rng.uniform(-3, 3, (n, 8))
    y = (2 * np.sin(X[:, 0]) + X[:, 1] ** 2 - X[:, 2] * (X[:, 3] > 0)
         + 0.5 * X[:, 4] * X[:, 5] + 0.2 * rng.randn(n))
    return X, y


def main():
    rng = np.random.RandomState(42)

    print("== regression with early stopping ==")
    X, y = make_regression(20000, rng)
    Xtr, ytr, Xva, yva, Xte, yte = X[:12000], y[:12000], X[12000:16000], y[12000:16000], X[16000:], y[16000:]
    t0 = time.time()
    m = GBMRegressor(n_estimators=600, learning_rate=0.1, max_leaves=31)
    m.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=25)
    print("trained %d rounds (best %d) in %.1fs" % (len(m.trees_), m.best_iteration_, time.time() - t0))
    print("test R2 = %.4f   RMSE = %.4f   (noise floor 0.20)" % (
        metrics.r2(yte, m.predict(Xte)), metrics.rmse(yte, m.predict(Xte))))
    imp = m.feature_importances()
    print("top features by gain:", ", ".join("x%d=%.2f" % (i, imp[i]) for i in np.argsort(-imp)[:4]))

    print("\n== 30% of feature values missing completely at random ==")
    Xm = X.copy()
    mask = rng.rand(*Xm.shape) < 0.3
    Xm[mask] = np.nan
    mm = GBMRegressor(n_estimators=200, max_leaves=31).fit(Xm[:12000], ytr)
    print("test R2 with NaNs = %.4f" % metrics.r2(yte, mm.predict(Xm[16000:])))

    print("\n== monotone constraint on a noisy, wiggly signal ==")
    Xc = rng.uniform(0, 1, (4000, 2))
    yc = Xc[:, 0] + 0.6 * np.sin(14 * Xc[:, 0]) + Xc[:, 1] + 0.4 * rng.randn(4000)
    grid = np.linspace(0, 1, 400)
    for name, mc in (("free", None), ("increasing", [1, 0])):
        mod = GBMRegressor(n_estimators=100, monotone_constraints=mc).fit(Xc, yc)
        bad = sum(int((np.diff(mod.predict(np.column_stack([grid, np.full_like(grid, v)]))) < -1e-9).sum())
                  for v in np.linspace(0, 1, 11))
        print("%-10s decreasing steps along x0: %d" % (name, bad))

    print("\n== multiclass (4 classes, XOR-like) ==")
    Xk = rng.randn(6000, 6)
    yk = (Xk[:, 0] > 0).astype(int) + 2 * (Xk[:, 1] * Xk[:, 2] > 0)
    clf = GBMClassifier(n_estimators=150, max_leaves=15).fit(Xk[:4500], yk[:4500])
    print("test accuracy = %.4f" % metrics.accuracy(yk[4500:], clf.predict(Xk[4500:])))


if __name__ == "__main__":
    main()
