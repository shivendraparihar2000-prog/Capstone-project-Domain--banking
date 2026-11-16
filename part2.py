import numpy as np
from common import (
    bce,
    cells,
    knn_prob,
    metrics,
    prepared,
    sigmoid,
    train_logistic,
    write_csv,
    write_results,
)

X, y, v, tr, te, Xtr, Xte = prepared()
ytr, yte = y[tr], y[te]
R = []


def add_result(key, value):
    R.append((key, value))
add_result("train_rows", len(tr)); add_result("train_positives", int(ytr.sum()))
add_result("test_rows", len(te)); add_result("test_positives", int(yte.sum()))

w, b, L = train_logistic(Xtr, ytr, 0.1, 1000, record=(0, 1, 10, 100, 1000))
for e in (0, 1, 10, 100, 1000):
    add_result("loss_epoch_%d" % e, L[e])
for j in range(7):
    add_result("weight_X%d" % (j + 1), w[j])
add_result("bias", b)
ptest = sigmoid(Xte @ w + b); ptrain = sigmoid(Xtr @ w + b)
add_result("test_bce", bce(ptest, yte))
for eta in (0.01, 0.1, 1.0):
    _, _, l = train_logistic(Xtr, ytr, eta, 200, record=(200,))
    add_result("loss_epoch_200_eta_%s" % eta, l[200])
for j in range(7):
    add_result("odds_ratio_X%d" % (j + 1), np.exp(w[j]))

for tau in (0.5, 0.3):
    tp, fp, tn, fn = cells(ptest >= tau, yte)
    acc, pr, rc, f1, sp = metrics(tp, fp, tn, fn)
    for k, val in (("tp", tp), ("fp", fp), ("tn", tn), ("fn", fn), ("accuracy", acc),
                   ("precision", pr), ("recall", rc), ("f1", f1), ("specificity", sp)):
        add_result("tau_%s_%s" % (tau, k), val)
add_result("blind_accuracy", float((yte == 0).mean()))

rows = []
for k in range(1, 10):
    tau = k / 10
    tp, fp, tn, fn = cells(ptest >= tau - 1e-12, yte)
    rows.append([tau, tp, fp, tn, fn, tp / (tp + fn), fp / (fp + tn)])
# tau column as plain text
write_csv("outputs/part2_thresholds.csv", ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"], rows)

# KNN CV
fold = np.arange(len(tr)) % 5
cvrows = []
for K in range(1, 22, 2):
    f1s = []
    for f in range(5):
        ref, q = fold != f, fold == f
        p = knn_prob(Xtr[q], Xtr[ref], ytr[ref], K)
        pred = p * K >= (K + 1) / 2 - 1e-9
        f1s.append(metrics(*cells(pred, ytr[q]))[3])
    cvrows.append([K, float(np.mean(f1s))])
write_csv("outputs/part2_knn_cv.csv", ["K", "mean_cv_f1"], cvrows)
best = max(cvrows, key=lambda r: (r[1], -r[0]))[0]
add_result("knn_best_k", best)
pk = knn_prob(Xte, Xtr, ytr, best)
tp, fp, tn, fn = cells(pk * best >= (best + 1) / 2 - 1e-9, yte)
acc, pr, rc, f1, sp = metrics(tp, fp, tn, fn)
for k, val in (("tp", tp), ("fp", fp), ("tn", tn), ("fn", fn), ("accuracy", acc),
               ("precision", pr), ("recall", rc), ("f1", f1)):
    add_result("knn_" + k, val)

cost = lambda c: 50 * c[1] + 400 * c[3]
best_tau, best_c = None, None
for k in range(1, 20):
    t = k / 20
    c = cost(cells(ptrain >= t - 1e-12, ytr))
    if best_c is None or c < best_c:
        best_tau, best_c = t, c
add_result("cost_tau", best_tau); add_result("cost_train_at_tau", best_c)
add_result("cost_test_at_tau", cost(cells(ptest >= best_tau - 1e-12, yte)))
add_result("cost_test_at_0.5", cost(cells(ptest >= 0.5, yte)))
add_result("cost_test_flag_nobody", 400 * int(yte.sum()))
write_results("outputs/part2_results.txt", R)
