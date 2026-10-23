import itertools
import os
import tempfile
import unittest

import numpy as np

from gbm import GBMClassifier, GBMRegressor, BinMapper, load_model, metrics
from gbm.losses import Logistic, Poisson, SquaredError, Softmax
from gbm.tree import build_hist, find_best_split, grow_tree


def numgrad(f, raw, eps=1e-6):
    g = np.zeros_like(raw)
    for i in range(raw.shape[0]):
        for k in range(raw.shape[1]):
            a, b = raw.copy(), raw.copy()
            a[i, k] += eps
            b[i, k] -= eps
            g[i, k] = (f(a) - f(b)) / (2 * eps)
    return g


class TestLosses(unittest.TestCase):
    def check(self, loss, y, raw, hess=True):
        n = len(y)
        w = np.ones(n)
        # metric is a mean, so d(metric)/d(raw_i) = grad_i / n (squared uses rmse -> use sum form below)
        g, h = loss.grad_hess(y, raw)
        g = g.reshape(raw.shape)
        if loss.name == "squared":
            f = lambda r: 0.5 * np.sum((r[:, 0] - y) ** 2)
            ng = numgrad(f, raw)
        else:
            ng = numgrad(lambda r: loss.metric(y, r, w), raw) * n
        np.testing.assert_allclose(g, ng, atol=1e-5)
        return h

    def test_squared(self):
        rng = np.random.RandomState(0)
        self.check(SquaredError(), rng.randn(8), rng.randn(8, 1))

    def test_logistic_grad_and_hess(self):
        rng = np.random.RandomState(1)
        y = (rng.rand(8) > 0.5).astype(float)
        raw = rng.randn(8, 1)
        L = Logistic()
        h = self.check(L, y, raw)
        # hessian equals numerical derivative of the gradient
        eps = 1e-6
        gp = L.grad_hess(y, raw + eps)[0]
        gm = L.grad_hess(y, raw - eps)[0]
        np.testing.assert_allclose(h, (gp - gm) / (2 * eps), atol=1e-5)

    def test_poisson_grad(self):
        rng = np.random.RandomState(2)
        self.check(Poisson(), rng.poisson(3, 8).astype(float), rng.randn(8, 1) * 0.5)

    def test_softmax_grad(self):
        rng = np.random.RandomState(3)
        self.check(Softmax(4), rng.randint(0, 4, 8).astype(float), rng.randn(8, 4))

    def test_sigmoid_is_stable(self):
        p = Logistic().transform(np.array([[-800.0], [800.0]]))
        self.assertTrue(np.all(np.isfinite(p)))
        np.testing.assert_allclose(p, [0.0, 1.0], atol=1e-12)


class TestBinning(unittest.TestCase):
    def test_few_uniques_get_own_bins(self):
        X = np.array([[1.0], [2.0], [2.0], [5.0], [np.nan]])
        m = BinMapper().fit(X)
        np.testing.assert_array_equal(m.transform(X)[:, 0], [0, 1, 1, 2, 255])

    def test_bin_order_is_monotone(self):
        rng = np.random.RandomState(0)
        X = rng.randn(2000, 1)
        codes = BinMapper(32).fit_transform(X)[:, 0]
        order = np.argsort(X[:, 0])
        self.assertTrue(np.all(np.diff(codes[order].astype(int)) >= 0))
        self.assertLessEqual(codes.max(), 31)

    def test_roundtrip_dict(self):
        X = np.random.RandomState(1).randn(300, 3)
        m = BinMapper(16).fit(X)
        m2 = BinMapper.from_dict(m.to_dict())
        np.testing.assert_array_equal(m.transform(X), m2.transform(X))


def brute_force_stump_gain(x, g, h, lam, min_leaf):
    """Exhaustive search over thresholds and missing direction (reference oracle)."""
    best = 0.0
    vals = np.unique(x[~np.isnan(x)])
    G, H = g.sum(), h.sum()
    base = G * G / (H + lam)
    for t in vals[:-1]:
        for miss_left in (False, True):
            left = np.where(np.isnan(x), miss_left, x <= t)
            if left.sum() < min_leaf or (~left).sum() < min_leaf:
                continue
            GL, HL = g[left].sum(), h[left].sum()
            GR, HR = G - GL, H - HL
            gain = 0.5 * (GL * GL / (HL + lam) + GR * GR / (HR + lam) - base)
            best = max(best, gain)
    return best


class TestSplitFinding(unittest.TestCase):
    def test_matches_brute_force_with_missing(self):
        rng = np.random.RandomState(7)
        for trial in range(15):
            n = 120
            x = rng.randint(0, 9, n).astype(float)
            x[rng.rand(n) < 0.2] = np.nan
            y = np.where(np.isnan(x), 1.5, np.sin(x)) + 0.1 * rng.randn(n)
            g, h = -y, np.ones(n)
            m = BinMapper().fit(x[:, None])
            Xb = m.transform(x[:, None])
            p = dict(reg_lambda=1.0, gamma=0.0, min_child_weight=1e-3, min_samples_leaf=3,
                     max_leaves=2, max_depth=-1)
            tree = grow_tree(Xb, g, h, np.arange(n), m.n_bins_, np.ones(1, bool), p, np.zeros(1, np.int64))
            ref = brute_force_stump_gain(x, g, h, 1.0, 3)
            self.assertAlmostEqual(tree.gain[0], ref, places=9)

    def test_histogram_subtraction_is_exact(self):
        rng = np.random.RandomState(0)
        Xb = rng.randint(0, 20, (400, 5)).astype(np.uint8)
        g, h = rng.randn(400), rng.rand(400) + 0.1
        full = build_hist(Xb, np.arange(400), g, h)
        a = np.arange(0, 400, 3)
        b = np.setdiff1d(np.arange(400), a)
        np.testing.assert_allclose(full - build_hist(Xb, a, g, h), build_hist(Xb, b, g, h), atol=1e-9)

    def test_min_samples_leaf_respected(self):
        rng = np.random.RandomState(5)
        X = rng.randn(300, 3)
        y = X[:, 0] + 0.1 * rng.randn(300)
        m = GBMRegressor(n_estimators=5, min_samples_leaf=40, max_leaves=16).fit(X, y)
        for rt in m.trees_:
            leaf_counts = rt[0].count[rt[0].feature < 0]
            self.assertTrue(np.all(leaf_counts >= 40))


class TestModels(unittest.TestCase):
    def test_regression_learns_nonlinear_function(self):
        rng = np.random.RandomState(0)
        X = rng.uniform(-3, 3, (1500, 4))
        y = np.sin(X[:, 0]) * 2 + X[:, 1] ** 2 - X[:, 2] * (X[:, 3] > 0) + 0.1 * rng.randn(1500)
        m = GBMRegressor(n_estimators=150, learning_rate=0.1).fit(X[:1200], y[:1200])
        self.assertGreater(metrics.r2(y[1200:], m.predict(X[1200:])), 0.93)

    def test_training_loss_decreases(self):
        rng = np.random.RandomState(1)
        X = rng.randn(500, 3)
        y = X[:, 0] * X[:, 1] + rng.randn(500) * 0.1
        m = GBMRegressor(n_estimators=40).fit(X, y)
        tr = [r["train"] for r in m.history_]
        self.assertTrue(all(b <= a + 1e-12 for a, b in zip(tr, tr[1:])))

    def test_binary_classification_and_labels(self):
        rng = np.random.RandomState(2)
        X = rng.randn(1000, 5)
        y = np.where(X[:, 0] + X[:, 1] ** 2 > 1.0, "yes", "no")
        m = GBMClassifier(n_estimators=80).fit(X[:800], y[:800])
        self.assertGreater(metrics.accuracy(y[800:], m.predict(X[800:])), 0.92)
        P = m.predict_proba(X[800:])
        np.testing.assert_allclose(P.sum(axis=1), 1.0)
        yb = (y[800:] == "yes")
        self.assertGreater(metrics.roc_auc(yb, P[:, list(m.classes_).index("yes")]), 0.96)

    def test_multiclass(self):
        rng = np.random.RandomState(3)
        X = rng.randn(1200, 2)
        y = (X[:, 0] > 0).astype(int) + 2 * (X[:, 1] > 0)
        m = GBMClassifier(n_estimators=60, max_leaves=8).fit(X[:900], y[:900])
        self.assertGreater(metrics.accuracy(y[900:], m.predict(X[900:])), 0.93)
        self.assertEqual(m.predict_proba(X[:5]).shape, (5, 4))

    def test_poisson_predictions_positive_and_calibrated(self):
        rng = np.random.RandomState(4)
        X = rng.uniform(0, 2, (2000, 2))
        rate = np.exp(0.5 + X[:, 0])
        y = rng.poisson(rate).astype(float)
        m = GBMRegressor(loss="poisson", n_estimators=60, learning_rate=0.1).fit(X, y)
        p = m.predict(X)
        self.assertTrue((p > 0).all())
        self.assertAlmostEqual(p.mean() / y.mean(), 1.0, delta=0.03)

    def test_missing_values_are_used_as_signal(self):
        rng = np.random.RandomState(5)
        n = 2000
        x = rng.randn(n)
        miss = rng.rand(n) < 0.3
        y = np.where(miss, 5.0, x) + 0.05 * rng.randn(n)
        X = x.copy()[:, None]
        X[miss, 0] = np.nan
        m = GBMRegressor(n_estimators=60).fit(X, y)
        pred = m.predict(X)
        self.assertLess(metrics.rmse(y[miss], pred[miss]), 0.3)

    def test_monotone_constraint_holds_on_noisy_data(self):
        rng = np.random.RandomState(6)
        X = rng.uniform(0, 1, (1500, 2))
        y = X[:, 0] + 0.8 * np.sin(12 * X[:, 0]) + X[:, 1] + 0.3 * rng.randn(1500)
        free = GBMRegressor(n_estimators=80).fit(X, y)
        con = GBMRegressor(n_estimators=80, monotone_constraints=[1, 0]).fit(X, y)
        grid = np.linspace(0, 1, 200)
        violations_free = violations_con = 0
        for x1 in np.linspace(0, 1, 15):
            Z = np.column_stack([grid, np.full_like(grid, x1)])
            violations_free += (np.diff(free.predict(Z)) < -1e-9).sum()
            violations_con += (np.diff(con.predict(Z)) < -1e-9).sum()
        self.assertGreater(violations_free, 0)  # the constraint is actually doing something
        self.assertEqual(violations_con, 0)

    def test_decreasing_monotone_constraint(self):
        rng = np.random.RandomState(7)
        X = rng.uniform(0, 1, (1000, 1))
        y = -X[:, 0] + 0.5 * np.cos(15 * X[:, 0]) + 0.2 * rng.randn(1000)
        m = GBMRegressor(n_estimators=60, monotone_constraints=[-1]).fit(X, y)
        p = m.predict(np.linspace(0, 1, 300)[:, None])
        self.assertTrue((np.diff(p) <= 1e-9).all())

    def test_early_stopping_truncates_to_best_iteration(self):
        rng = np.random.RandomState(8)
        X = rng.randn(600, 3)
        y = X[:, 0] + rng.randn(600)  # noisy -> overfits quickly
        m = GBMRegressor(n_estimators=400, learning_rate=0.3, max_leaves=31, min_samples_leaf=3)
        m.fit(X[:300], y[:300], eval_set=(X[300:], y[300:]), early_stopping_rounds=10)
        self.assertLess(len(m.trees_), 400)
        self.assertEqual(len(m.trees_), m.best_iteration_)
        best_valid = min(r["valid"] for r in m.history_)
        self.assertAlmostEqual(metrics.rmse(y[300:], m.predict(X[300:])), best_valid, places=9)

    def test_subsample_colsample_still_learn(self):
        rng = np.random.RandomState(9)
        X = rng.randn(1000, 6)
        y = X[:, 0] - 2 * X[:, 3] + 0.1 * rng.randn(1000)
        m = GBMRegressor(n_estimators=100, subsample=0.6, colsample_bytree=0.5, random_state=3).fit(X, y)
        self.assertGreater(metrics.r2(y, m.predict(X)), 0.95)

    def test_feature_importance_finds_signal(self):
        rng = np.random.RandomState(10)
        X = rng.randn(1000, 5)
        y = 3 * X[:, 2] + 0.1 * rng.randn(1000)
        imp = GBMRegressor(n_estimators=40).fit(X, y).feature_importances()
        self.assertEqual(int(np.argmax(imp)), 2)
        self.assertAlmostEqual(imp.sum(), 1.0)

    def test_save_load_roundtrip(self):
        rng = np.random.RandomState(11)
        X = rng.randn(400, 3)
        X[rng.rand(400, 3) < 0.1] = np.nan
        y = (np.nan_to_num(X[:, 0]) > 0).astype(int) + (np.nan_to_num(X[:, 1]) > 0)
        m = GBMClassifier(n_estimators=15, max_leaves=6).fit(X, y)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.json")
            m.save(path)
            m2 = load_model(path)
        np.testing.assert_allclose(m.predict_proba(X), m2.predict_proba(X), atol=1e-12)

    def test_staged_predictions_end_at_final(self):
        rng = np.random.RandomState(12)
        X = rng.randn(300, 2)
        y = X[:, 0] ** 2
        m = GBMRegressor(n_estimators=12).fit(X, y)
        stages = list(m.staged_raw(X))
        self.assertEqual(len(stages), 12)
        np.testing.assert_allclose(stages[-1][:, 0], m.predict(X))

    def test_sample_weights_equal_duplication(self):
        rng = np.random.RandomState(13)
        X = rng.randn(200, 2)
        y = X[:, 0] + 0.1 * rng.randn(200)
        w = np.ones(200)
        w[:50] = 2.0
        kw = dict(n_estimators=10, min_samples_leaf=1, min_child_weight=1e-6, max_leaves=4)
        a = GBMRegressor(**kw).fit(X, y, sample_weight=w)
        Xd, yd = np.vstack([X, X[:50]]), np.concatenate([y, y[:50]])
        b = GBMRegressor(**kw).fit(Xd, yd)
        np.testing.assert_allclose(a.predict(X), b.predict(X), atol=1e-6)

    def test_bad_inputs(self):
        with self.assertRaises(TypeError):
            GBMRegressor(bogus=1)
        with self.assertRaises(ValueError):
            GBMRegressor(loss="poisson").fit(np.zeros((5, 1)), -np.ones(5))
        with self.assertRaises(ValueError):
            GBMClassifier().fit(np.zeros((5, 1)), np.zeros(5))


if __name__ == "__main__":
    unittest.main()
