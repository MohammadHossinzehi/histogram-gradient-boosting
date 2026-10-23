"""Gradient boosting machine: ensemble driver, estimators, early stopping, persistence."""
import json
import numpy as np

from .binning import BinMapper
from .losses import make_loss
from .tree import Tree, grow_tree

DEFAULTS = dict(
    n_estimators=100, learning_rate=0.1, max_leaves=31, max_depth=-1,
    reg_lambda=1.0, gamma=0.0, min_child_weight=1e-3, min_samples_leaf=20,
    subsample=1.0, colsample_bytree=1.0, max_bins=254, random_state=0,
)


class _Base:
    def __init__(self, loss, monotone_constraints=None, **params):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise TypeError("unknown parameters: %s" % sorted(unknown))
        self.loss_name = loss
        self.monotone_constraints = monotone_constraints
        self.params = {**DEFAULTS, **params}

    # ------------------------------------------------------------------ fit
    def _fit(self, X, y, sample_weight, eval_set, early_stopping_rounds, verbose):
        P = self.params
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        y = np.asarray(y, dtype=np.float64)
        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        self.n_features_ = d
        self.mapper_ = BinMapper(P["max_bins"]).fit(X)
        Xb = self.mapper_.transform(X)
        nbins = self.mapper_.n_bins_
        mono = np.zeros(d, dtype=np.int64)
        if self.monotone_constraints is not None:
            mono = np.asarray(self.monotone_constraints, dtype=np.int64)
            if mono.shape != (d,):
                raise ValueError("monotone_constraints needs one entry per feature")
        K = self.loss_.n_outputs
        self.init_ = self.loss_.init_score(y, w)
        raw = np.tile(self.init_, (n, 1))
        if eval_set is not None:
            Xv = self.mapper_.transform(np.asarray(eval_set[0], dtype=np.float64))
            yv = self._encode_y(eval_set[1])
            wv = np.ones(len(yv))
            raw_v = np.tile(self.init_, (len(yv), 1))
        rng = np.random.RandomState(P["random_state"])
        self.trees_, self.history_ = [], []
        best, best_it = np.inf, 0
        for it in range(P["n_estimators"]):
            g, h = self.loss_.grad_hess(y, raw)
            g = g.reshape(n, K) * w[:, None]
            h = h.reshape(n, K) * w[:, None]
            rows = np.arange(n)
            if P["subsample"] < 1.0:
                rows = np.sort(rng.choice(n, max(2, int(n * P["subsample"])), replace=False))
            allowed = np.ones(d, dtype=bool)
            if P["colsample_bytree"] < 1.0:
                k = max(1, int(round(d * P["colsample_bytree"])))
                allowed[:] = False
                allowed[rng.choice(d, k, replace=False)] = True
            round_trees = []
            for k in range(K):
                t = grow_tree(Xb, g[:, k], h[:, k], rows, nbins, allowed, P, mono)
                t.value = t.value * P["learning_rate"]
                raw[:, k] += t.predict_binned(Xb)
                if eval_set is not None:
                    raw_v[:, k] += t.predict_binned(Xv)
                round_trees.append(t)
            self.trees_.append(round_trees)
            rec = {"iter": it + 1, "train": self.loss_.metric(y, raw, w)}
            if eval_set is not None:
                rec["valid"] = self.loss_.metric(yv, raw_v, wv)
                if rec["valid"] < best - 1e-12:
                    best, best_it = rec["valid"], it + 1
            self.history_.append(rec)
            if verbose and (it + 1) % verbose == 0:
                print("[%d] %s" % (it + 1, " ".join("%s=%.5f" % (k, v) for k, v in rec.items() if k != "iter")))
            if (eval_set is not None and early_stopping_rounds
                    and it + 1 - best_it >= early_stopping_rounds):
                break
        if eval_set is not None:
            self.best_iteration_ = best_it
            if early_stopping_rounds:
                self.trees_ = self.trees_[:best_it]
        else:
            self.best_iteration_ = len(self.trees_)
        return self

    # -------------------------------------------------------------- predict
    def _raw(self, X, n_trees=None):
        Xb = self.mapper_.transform(np.asarray(X, dtype=np.float64))
        raw = np.tile(self.init_, (Xb.shape[0], 1))
        for round_trees in self.trees_[:n_trees]:
            for k, t in enumerate(round_trees):
                raw[:, k] += t.predict_binned(Xb)
        return raw

    def staged_raw(self, X):
        """Yield the raw score after each boosting round."""
        Xb = self.mapper_.transform(np.asarray(X, dtype=np.float64))
        raw = np.tile(self.init_, (Xb.shape[0], 1))
        for round_trees in self.trees_:
            for k, t in enumerate(round_trees):
                raw[:, k] += t.predict_binned(Xb)
            yield raw.copy()

    # ----------------------------------------------------------- importance
    def feature_importances(self, kind="gain"):
        imp = np.zeros(self.n_features_)
        for round_trees in self.trees_:
            for t in round_trees:
                m = t.feature >= 0
                np.add.at(imp, t.feature[m], t.gain[m] if kind == "gain" else 1.0)
        total = imp.sum()
        return imp / total if total > 0 else imp

    # ---------------------------------------------------------- persistence
    def to_dict(self):
        return {"kind": type(self).__name__, "loss": self.loss_name, "params": self.params,
                "monotone": None if self.monotone_constraints is None else list(map(int, self.monotone_constraints)),
                "n_features": self.n_features_, "init": self.init_.tolist(),
                "mapper": self.mapper_.to_dict(),
                "classes": getattr(self, "classes_", np.array([])).tolist(),
                "trees": [[t.to_dict() for t in r] for r in self.trees_]}

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh)

    @staticmethod
    def load(path):
        with open(path) as fh:
            d = json.load(fh)
        cls = {"GBMRegressor": GBMRegressor, "GBMClassifier": GBMClassifier}[d["kind"]]
        m = cls.__new__(cls)
        m.loss_name, m.params = d["loss"], d["params"]
        m.monotone_constraints = d["monotone"]
        m.n_features_, m.init_ = d["n_features"], np.array(d["init"])
        m.mapper_ = BinMapper.from_dict(d["mapper"])
        m.trees_ = [[Tree.from_dict(t) for t in r] for r in d["trees"]]
        if d["classes"]:
            m.classes_ = np.array(d["classes"])
        m.loss_ = make_loss(m.loss_name, len(m.classes_) if hasattr(m, "classes_") else None)
        m.best_iteration_, m.history_ = len(m.trees_), []
        return m


class GBMRegressor(_Base):
    def __init__(self, loss="squared", monotone_constraints=None, **params):
        if loss not in ("squared", "poisson"):
            raise ValueError("regressor loss must be 'squared' or 'poisson'")
        super().__init__(loss, monotone_constraints, **params)
        self.loss_ = make_loss(loss)

    def _encode_y(self, y):
        return np.asarray(y, dtype=np.float64)

    def fit(self, X, y, sample_weight=None, eval_set=None, early_stopping_rounds=None, verbose=0):
        y = np.asarray(y, dtype=np.float64)
        if self.loss_name == "poisson" and (y < 0).any():
            raise ValueError("poisson loss needs non-negative targets")
        return self._fit(X, y, sample_weight, eval_set, early_stopping_rounds, verbose)

    def predict(self, X):
        return self.loss_.transform(self._raw(X))


class GBMClassifier(_Base):
    """Binary (logistic) or multiclass (softmax) classifier; labels may be any sortable values."""

    def __init__(self, monotone_constraints=None, **params):
        super().__init__(None, monotone_constraints, **params)

    def _encode_y(self, y):
        idx = np.searchsorted(self.classes_, np.asarray(y))
        if not np.array_equal(self.classes_[np.clip(idx, 0, len(self.classes_) - 1)], np.asarray(y)):
            raise ValueError("eval_set contains labels not seen in training")
        return idx.astype(np.float64)

    def fit(self, X, y, sample_weight=None, eval_set=None, early_stopping_rounds=None, verbose=0):
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("need at least two classes")
        self.loss_name = "logistic" if len(self.classes_) == 2 else "softmax"
        self.loss_ = make_loss(self.loss_name, len(self.classes_))
        return self._fit(X, self._encode_y(y), sample_weight, eval_set, early_stopping_rounds, verbose)

    def predict_proba(self, X):
        out = self.loss_.transform(self._raw(X))
        return np.column_stack([1 - out, out]) if out.ndim == 1 else out

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
