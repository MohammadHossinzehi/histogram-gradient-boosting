"""Leaf-wise histogram tree grower (second-order, Newton boosting).

For a node with gradient sum G and hessian sum H the optimal leaf weight is
``w = -G / (H + lambda)`` and the objective improvement of a split is

    gain = score(wL) + score(wR) - score(w) - gamma,   score(w) = -(G w + (H + lambda) w^2 / 2)

evaluated with the *clamped* weights so monotone constraints are honoured
exactly. Per-node gradient histograms are built only for the smaller child;
the sibling comes from parent minus child ("histogram subtraction").
Missing values get a learned default direction at every split.
"""
import heapq
import numpy as np

from .binning import MISSING

NB = 256  # histogram width: 255 real bins max + the missing slot


class Tree:
    def __init__(self):
        self.feature, self.thr, self.default_left = [], [], []
        self.left, self.right, self.value, self.gain, self.count = [], [], [], [], []

    def add_node(self, value, count):
        self.feature.append(-1)
        self.thr.append(0)
        self.default_left.append(False)
        self.left.append(-1)
        self.right.append(-1)
        self.value.append(value)
        self.gain.append(0.0)
        self.count.append(count)
        return len(self.feature) - 1

    def freeze(self):
        self.feature = np.array(self.feature, dtype=np.int64)
        self.thr = np.array(self.thr, dtype=np.int64)
        self.default_left = np.array(self.default_left, dtype=bool)
        self.left = np.array(self.left, dtype=np.int64)
        self.right = np.array(self.right, dtype=np.int64)
        self.value = np.array(self.value, dtype=np.float64)
        self.gain = np.array(self.gain, dtype=np.float64)
        self.count = np.array(self.count, dtype=np.int64)
        return self

    @property
    def n_leaves(self):
        return int((self.feature < 0).sum())

    def predict_binned(self, Xb):
        node = np.zeros(Xb.shape[0], dtype=np.int64)
        while True:
            internal = np.nonzero(self.feature[node] >= 0)[0]
            if len(internal) == 0:
                break
            nd = node[internal]
            codes = Xb[internal, self.feature[nd]]
            go_left = np.where(codes == MISSING, self.default_left[nd], codes <= self.thr[nd])
            node[internal] = np.where(go_left, self.left[nd], self.right[nd])
        return self.value[node]

    def to_dict(self):
        return {k: getattr(self, k).tolist() for k in
                ("feature", "thr", "default_left", "left", "right", "value", "gain", "count")}

    @classmethod
    def from_dict(cls, d):
        t = cls()
        for k, v in d.items():
            setattr(t, k, v)
        return t.freeze()


def build_hist(Xb, idx, g, h):
    """Return a (3, d, 256) array of [grad sum, hess sum, count] per feature bin."""
    d = Xb.shape[1]
    keys = (Xb[idx].astype(np.int64) + (np.arange(d, dtype=np.int64) * NB)[None, :]).ravel()
    size = d * NB
    G = np.bincount(keys, weights=np.repeat(g[idx], d), minlength=size)
    H = np.bincount(keys, weights=np.repeat(h[idx], d), minlength=size)
    C = np.bincount(keys, minlength=size).astype(np.float64)
    return np.stack([G, H, C]).reshape(3, d, NB)


def _score(G, H, w, lam):
    return -(G * w + 0.5 * (H + lam) * w * w)


def find_best_split(hist, nbins, allowed, p, lb, ub, mono):
    """Best (gain, feature, bin, missing_goes_left, wL, wR) for a node, or None."""
    G, H, C = hist[0], hist[1], hist[2]
    Gt, Ht, Ct = G[0].sum(), H[0].sum(), C[0].sum()
    lam = p["reg_lambda"]
    w_par = np.clip(-Gt / (Ht + lam), lb, ub)
    parent = _score(Gt, Ht, w_par, lam)
    cG = np.cumsum(G[:, :MISSING], axis=1)
    cH = np.cumsum(H[:, :MISSING], axis=1)
    cC = np.cumsum(C[:, :MISSING], axis=1)
    Gm, Hm, Cm = G[:, MISSING:MISSING + 1], H[:, MISSING:MISSING + 1], C[:, MISSING:MISSING + 1]
    # threshold b is legal only if bins b+1.. exist for the feature
    legal = np.arange(MISSING)[None, :] < (nbins[:, None] - 1)
    legal &= allowed[:, None]
    best = None
    for miss_left in (False, True):
        GL = cG + (Gm if miss_left else 0.0)
        HL = cH + (Hm if miss_left else 0.0)
        CL = cC + (Cm if miss_left else 0.0)
        GR, HR, CR = Gt - GL, Ht - HL, Ct - CL
        ok = (legal & (CL >= p["min_samples_leaf"]) & (CR >= p["min_samples_leaf"])
              & (HL >= p["min_child_weight"]) & (HR >= p["min_child_weight"]))
        with np.errstate(divide="ignore", invalid="ignore"):
            wL = np.clip(-GL / (HL + lam), lb, ub)
            wR = np.clip(-GR / (HR + lam), lb, ub)
        m = mono[:, None]
        ok &= ~((m == 1) & (wL > wR)) & ~((m == -1) & (wL < wR))
        gain = _score(GL, HL, wL, lam) + _score(GR, HR, wR, lam) - parent - p["gamma"]
        gain = np.where(ok & np.isfinite(gain), gain, -np.inf)
        f, b = np.unravel_index(np.argmax(gain), gain.shape)
        cand = (gain[f, b], int(f), int(b), miss_left, wL[f, b], wR[f, b])
        if best is None or cand[0] > best[0]:
            best = cand
    return best if best[0] > 1e-12 else None


def grow_tree(Xb, g, h, rows, nbins, allowed, p, mono):
    """Grow one tree leaf-wise on the given rows. Leaf values are NOT yet scaled by the learning rate."""
    lam, max_leaves, max_depth = p["reg_lambda"], p["max_leaves"], p["max_depth"]
    tree = Tree()
    nodes = {}
    heap, counter = [], 0

    def make(idx, hist, depth, lb, ub):
        nonlocal counter
        Gt, Ht = hist[0][0].sum(), hist[1][0].sum()
        nid = tree.add_node(float(np.clip(-Gt / (Ht + lam), lb, ub)), len(idx))
        rec = {"idx": idx, "hist": hist, "depth": depth, "lb": lb, "ub": ub, "split": None}
        nodes[nid] = rec
        if (max_depth < 0 or depth < max_depth) and len(idx) >= 2 * p["min_samples_leaf"]:
            s = find_best_split(hist, nbins, allowed, p, lb, ub, mono)
            if s is not None:
                rec["split"] = s
                heapq.heappush(heap, (-s[0], counter, nid))
                counter += 1
        return nid

    make(rows, build_hist(Xb, rows, g, h), 0, -np.inf, np.inf)
    n_leaves = 1
    while heap and n_leaves < max_leaves:
        _, _, nid = heapq.heappop(heap)
        rec = nodes[nid]
        gain, f, b, miss_left, wL, wR = rec["split"]
        idx = rec["idx"]
        codes = Xb[idx, f]
        go_left = np.where(codes == MISSING, miss_left, codes <= b)
        li, ri = idx[go_left], idx[~go_left]
        # histogram subtraction: build the smaller side, derive the larger
        if len(li) <= len(ri):
            hl = build_hist(Xb, li, g, h)
            hr = rec["hist"] - hl
        else:
            hr = build_hist(Xb, ri, g, h)
            hl = rec["hist"] - hr
        lb_l, ub_l, lb_r, ub_r = rec["lb"], rec["ub"], rec["lb"], rec["ub"]
        if mono[f] != 0:
            mid = 0.5 * (wL + wR)
            if mono[f] == 1:
                ub_l, lb_r = min(ub_l, mid), max(lb_r, mid)
            else:
                lb_l, ub_r = max(lb_l, mid), min(ub_r, mid)
        d = rec["depth"] + 1
        lid = make(li, hl, d, lb_l, ub_l)
        rid = make(ri, hr, d, lb_r, ub_r)
        tree.feature[nid], tree.thr[nid], tree.default_left[nid] = f, b, bool(miss_left)
        tree.left[nid], tree.right[nid], tree.gain[nid] = lid, rid, float(gain)
        rec["hist"] = None  # free memory; internal nodes never need it again
        rec["idx"] = None
        n_leaves += 1
    return tree.freeze()
