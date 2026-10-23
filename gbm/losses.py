"""Loss functions: per-sample gradient and hessian of the loss w.r.t. the raw score."""
import numpy as np


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


class SquaredError:
    name = "squared"
    n_outputs = 1

    def init_score(self, y, w):
        return np.array([np.average(y, weights=w)])

    def grad_hess(self, y, raw):
        return raw[:, 0] - y, np.ones_like(y)

    def transform(self, raw):
        return raw[:, 0]

    def metric(self, y, raw, w):
        return float(np.sqrt(np.average((raw[:, 0] - y) ** 2, weights=w)))

    metric_name = "rmse"


class Logistic:
    name = "logistic"
    n_outputs = 1

    def init_score(self, y, w):
        p = np.clip(np.average(y, weights=w), 1e-6, 1 - 1e-6)
        return np.array([np.log(p / (1 - p))])

    def grad_hess(self, y, raw):
        p = _sigmoid(raw[:, 0])
        return p - y, np.maximum(p * (1 - p), 1e-16)

    def transform(self, raw):
        return _sigmoid(raw[:, 0])

    def metric(self, y, raw, w):
        z = raw[:, 0]
        # stable log(1+exp(z)) - y*z
        ll = np.logaddexp(0.0, z) - y * z
        return float(np.average(ll, weights=w))

    metric_name = "logloss"


class Poisson:
    """Poisson regression with log link. Hessian uses a safeguard (as in XGBoost)."""
    name = "poisson"
    n_outputs = 1
    max_delta_step = 0.7

    def init_score(self, y, w):
        return np.array([np.log(max(np.average(y, weights=w), 1e-9))])

    def grad_hess(self, y, raw):
        mu = np.exp(np.clip(raw[:, 0], -30, 30))
        return mu - y, np.exp(np.clip(raw[:, 0] + self.max_delta_step, -30, 30))

    def transform(self, raw):
        return np.exp(np.clip(raw[:, 0], -30, 30))

    def metric(self, y, raw, w):
        z = np.clip(raw[:, 0], -30, 30)
        return float(np.average(np.exp(z) - y * z, weights=w))

    metric_name = "poisson-nll"


class Softmax:
    name = "softmax"

    def __init__(self, n_classes):
        self.n_outputs = n_classes

    def init_score(self, y, w):
        counts = np.bincount(y.astype(int), weights=w, minlength=self.n_outputs)
        p = np.clip(counts / counts.sum(), 1e-9, None)
        return np.log(p)

    @staticmethod
    def _softmax(raw):
        z = raw - raw.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def grad_hess(self, y, raw):
        p = self._softmax(raw)
        onehot = np.zeros_like(p)
        onehot[np.arange(len(y)), y.astype(int)] = 1.0
        return p - onehot, np.maximum(2.0 * p * (1.0 - p), 1e-16)

    def transform(self, raw):
        return self._softmax(raw)

    def metric(self, y, raw, w):
        z = raw - raw.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(np.average(-logp[np.arange(len(y)), y.astype(int)], weights=w))

    metric_name = "mlogloss"


def make_loss(name, n_classes=None):
    if name == "squared":
        return SquaredError()
    if name == "logistic":
        return Logistic()
    if name == "poisson":
        return Poisson()
    if name == "softmax":
        return Softmax(n_classes)
    raise ValueError("unknown loss %r" % name)
