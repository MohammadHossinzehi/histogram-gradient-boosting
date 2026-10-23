# histogram-gradient-boosting

A LightGBM/XGBoost style gradient boosting library written from scratch in NumPy. It uses histogram based split finding, leaf-wise tree growth, second order (Newton) leaf weights, learned default directions for missing values, exact monotone constraints, early stopping, and four losses (squared, logistic, Poisson, softmax). The only dependency is NumPy.

## Why

Gradient boosted trees are the workhorse of tabular machine learning, but most people use them as a black box. This project implements the parts that make modern boosters fast and accurate, in code short enough to read in one sitting, and tests each part against an independent oracle.

## What is inside

| Module | What it does |
| --- | --- |
| `gbm/binning.py` | Quantile binning into `uint8` codes. Code 255 is reserved for NaN. A split at bin `b` means `x <= edges[b]`, so training and prediction both work on integer codes. |
| `gbm/tree.py` | Leaf-wise tree grower with a priority queue on split gain, histogram subtraction (build the smaller child, derive the sibling), L2 regularisation, `gamma`, `min_child_weight`, `min_samples_leaf`, `max_depth`, and monotone constraints. |
| `gbm/losses.py` | Gradient and hessian for squared error, logistic, Poisson (log link, with a hessian safeguard) and softmax. |
| `gbm/booster.py` | `GBMRegressor` and `GBMClassifier` (binary and multiclass), row and column subsampling, sample weights, early stopping, staged prediction, gain importance, JSON save and load. |
| `gbm/metrics.py` | RMSE, R2, accuracy and a tie-aware rank AUC. |
| `gbm/cli.py` | Train and predict from CSV files. |

## Quick start

```bash
pip install numpy
python examples/demo.py
python -m unittest tests.test_gbm -v
```

```python
import numpy as np
from gbm import GBMRegressor, GBMClassifier, load_model

reg = GBMRegressor(n_estimators=500, learning_rate=0.1, max_leaves=31, monotone_constraints=[1, 0, 0])
reg.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=25)
pred = reg.predict(X_test)           # NaN is a valid feature value
print(reg.feature_importances())     # normalised total gain per feature

clf = GBMClassifier(n_estimators=200).fit(X, labels)   # labels can be strings; 2 or more classes
proba = clf.predict_proba(X)
clf.save("model.json"); clf2 = load_model("model.json")
```

CSV command line (empty cells and `NaN` are treated as missing):

```bash
python -m gbm.cli train data.csv --target price --task regression --out model.json
python -m gbm.cli predict model.json new_rows.csv
```

## The math in one paragraph

Each round fits a tree to the per-sample gradient `g` and hessian `h` of the loss at the current raw score. For a node with sums `G` and `H` the optimal leaf weight is `w = -G / (H + lambda)`, and a split is scored by the objective improvement `score(wL) + score(wR) - score(w) - gamma`, where `score(w) = -(G*w + (H + lambda) * w^2 / 2)`. Scoring with the clamped weights (rather than the closed form `G^2 / (H + lambda)`) is what lets monotone constraints be enforced exactly: a candidate split is rejected if the (clamped) child values violate the required order, and after a constrained split the midpoint of the two children becomes a bound for every descendant, so no leaf deeper in the tree can break monotonicity.

## Design decisions

- **Histogram subtraction.** Per feature gradient histograms are built for the smaller child only; the larger child is `parent - small`. This halves histogram work and is tested to be exact.
- **Missing values.** Every candidate threshold is evaluated twice, with the NaN bin sent left and sent right. The better direction is stored in the tree. Nothing is imputed.
- **Leaf-wise growth.** Trees grow by always splitting the leaf with the largest gain until `max_leaves`, which fits structure with fewer nodes than level-wise growth. `max_depth` is available as a guard.
- **Softmax hessian.** Uses `2 p (1 - p)` as in XGBoost, which keeps Newton steps conservative for multiclass.
- **Poisson safeguard.** The hessian is `exp(raw + 0.7)`, the usual `max_delta_step` trick, so early steps cannot explode.
- **Vectorised split search.** Cumulative sums over the histogram evaluate every bin of every feature in a handful of NumPy calls.

## Testing

`python -m unittest tests.test_gbm` runs 26 tests, including:

- finite difference checks of every loss gradient, and of the logistic hessian;
- the best stump found by the tree grower matches an exhaustive brute force oracle (all thresholds, both missing directions) to 9 decimal places on random data with missing values;
- histogram subtraction equals a directly built histogram;
- monotone constraints produce zero violations along a dense grid on noisy, wiggly data where the unconstrained model has many (increasing and decreasing constraints);
- integer sample weights give the same predictions as duplicating rows;
- early stopping truncates the ensemble to the best round and the truncated model reproduces the best validation score;
- save and load round trips predictions exactly, including NaN inputs and multiclass.

## Example output

Output of `python examples/demo.py` on the machine used to develop this (NumPy 2.5):

```
== regression with early stopping ==
trained 597 rounds (best 597) in 8.7s
test R2 = 0.9884   RMSE = 0.3873   (noise floor 0.20)

== 30% of feature values missing completely at random ==
test R2 with NaNs = 0.6196

== monotone constraint on a noisy, wiggly signal ==
free       decreasing steps along x0: 776
increasing decreasing steps along x0: 0

== multiclass (4 classes, XOR-like) ==
test accuracy = 0.9967
```

Timings and scores depend on your machine and NumPy version. This is a teaching quality implementation: it is not tuned to compete with LightGBM's C++ speed, and it does not implement categorical splits, GOSS, or feature bundling.

## License

MIT
