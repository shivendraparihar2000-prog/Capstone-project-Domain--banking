"""Shared helpers (importing this file runs nothing)."""
import csv
import numpy as np


def load():
    with open("data/records.csv") as f:
        r = list(csv.reader(f))
    a = np.array([[float(v) for v in row[1:]] for row in r[1:]])
    return a[:, :7], a[:, 7].astype(int), a[:, 8]  # X1..X7, outcome, value


def fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, str):
        return v
    return "%.4f" % v


def write_results(path, items):
    with open(path, "w") as f:
        for k, v in items:
            f.write("%s = %s\n" % (k, fmt(v)))


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(fmt(v) for v in row) + "\n")


def split_idx(y):
    tr, te = [], []
    for c in (0, 1):
        idx = [i for i in range(len(y)) if y[i] == c]
        k = (7 * len(idx)) // 10
        tr += idx[:k]
        te += idx[k:]
    return np.array(sorted(tr)), np.array(sorted(te))


def prepared():
    X, y, v = load()
    tr, te = split_idx(y)
    mu = X[tr].mean(0)
    sd = X[tr].std(0, ddof=1)
    return X, y, v, tr, te, (X[tr] - mu) / sd, (X[te] - mu) / sd


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce(p, y):
    """Return binary cross-entropy with probabilities kept away from 0 and 1."""
    eps = np.finfo(float).eps
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train_logistic(Xs, y, eta, n_upd, record=()):
    w = np.zeros(Xs.shape[1])
    b = 0.0
    n = len(y)
    losses = {}
    if 0 in record:
        losses[0] = bce(sigmoid(Xs @ w + b), y)
    for t in range(1, n_upd + 1):
        p = sigmoid(Xs @ w + b)
        e = p - y
        w = w - eta * (Xs.T @ e) / n
        b = b - eta * e.sum() / n
        if t in record:
            losses[t] = bce(sigmoid(Xs @ w + b), y)
    return w, b, losses


def cells(pred, y):
    pred = np.asarray(pred).astype(bool)
    y = np.asarray(y).astype(bool)
    return (int((pred & y).sum()), int((pred & ~y).sum()),
            int((~pred & ~y).sum()), int((~pred & y).sum()))


def metrics(tp, fp, tn, fn):
    """Return accuracy, precision, recall, F1 score and specificity."""
    def safe_divide(numerator, denominator):
        return numerator / denominator if denominator else 0.0

    accuracy = safe_divide(tp + tn, tp + fp + tn + fn)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    specificity = safe_divide(tn, tn + fp)
    return accuracy, precision, recall, f1, specificity


def knn_prob(Q, R, yR, K):
    """Share of positive neighbours for each query row."""
    out = np.zeros(len(Q))
    for i, q in enumerate(Q):
        d = ((R - q) ** 2).sum(1)
        nn = np.argsort(d, kind="stable")[:K]
        out[i] = yR[nn].sum() / K
    return out
