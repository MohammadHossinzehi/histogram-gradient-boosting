"""Quantile feature binning.

Every feature is mapped to a small integer code (uint8). Codes 0..k-1 are
ordered value bins; code 255 is reserved for missing values (NaN). A split
"at bin b" means ``x <= edges[b]`` goes left, which is exactly equivalent to
``code <= b`` after binning, so training and prediction can both run on codes.
"""
import numpy as np

MISSING = 255
MAX_BINS = 254


class BinMapper:
    def __init__(self, max_bins=254):
        if not 2 <= max_bins <= MAX_BINS:
            raise ValueError("max_bins must be in [2, %d]" % MAX_BINS)
        self.max_bins = max_bins
        self.edges_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        edges = []
        for j in range(X.shape[1]):
            col = X[:, j]
            col = col[~np.isnan(col)]
            uniq = np.unique(col)
            if len(uniq) <= 1:
                edges.append(np.empty(0))
            elif len(uniq) <= self.max_bins:
                # one bin per distinct value: thresholds are midpoints
                edges.append((uniq[:-1] + uniq[1:]) / 2.0)
            else:
                qs = np.linspace(0, 1, self.max_bins + 1)[1:-1]
                e = np.unique(np.quantile(col, qs, method="linear"))
                edges.append(e)
        self.edges_ = edges
        return self

    @property
    def n_bins_(self):
        """Number of real (non-missing) bins per feature."""
        return np.array([len(e) + 1 for e in self.edges_], dtype=np.int64)

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        if d != len(self.edges_):
            raise ValueError("expected %d features, got %d" % (len(self.edges_), d))
        out = np.empty((n, d), dtype=np.uint8)
        for j in range(d):
            col = X[:, j]
            codes = np.searchsorted(self.edges_[j], col, side="left")
            codes = codes.astype(np.uint8)
            codes[np.isnan(col)] = MISSING
            out[:, j] = codes
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def to_dict(self):
        return {"max_bins": self.max_bins, "edges": [e.tolist() for e in self.edges_]}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["max_bins"])
        m.edges_ = [np.array(e, dtype=np.float64) for e in d["edges"]]
        return m
