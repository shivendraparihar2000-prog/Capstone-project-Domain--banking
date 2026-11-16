import numpy as np
from common import (
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
Xo_tr, Xo_te = X[tr], X[te]
R = []


def add_result(key, value):
    R.append((key, value))


def ent(p):
    return 0.0 if p in (0.0, 1.0) else float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def gini(p):
    return 1 - p * p - (1 - p) ** 2


N = len(ytr)
P = int(ytr.sum())
add_result("root_entropy", ent(P / N)); add_result("root_gini", gini(P / N))
splits = {"S1": Xo_tr[:, 5] == 1, "S2": Xo_tr[:, 6] == 1, "S3": Xo_tr[:, 2] > 0.5, "S4": Xo_tr[:, 0] <= 24}
rows, ig, gg = [], {}, {}
for name, m in splits.items():
    ny, py = int(m.sum()), int(ytr[m].sum())
    nn, pn = N - ny, P - py
    e = (ny / N) * ent(py / ny) + (nn / N) * ent(pn / nn)
    g = (ny / N) * gini(py / ny) + (nn / N) * gini(pn / nn)
    ig[name] = ent(P / N) - e; gg[name] = gini(P / N) - g
    rows.append([name, ny, py, nn, pn, ig[name], gg[name]])
write_csv("outputs/part4_splits.csv", ["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"], rows)
bi = max(ig, key=lambda k: ig[k]); bg = max(gg, key=lambda k: gg[k])
add_result("best_split_by_info_gain", bi); add_result("best_split_by_gini_gain", bg)

m = splits[bi]
ry, rn = ytr[m].mean(), ytr[~m].mean()
add_result("stump_rate_yes", ry); add_result("stump_rate_no", rn)
cond = {"S1": lambda A: A[:, 5] == 1, "S2": lambda A: A[:, 6] == 1,
        "S3": lambda A: A[:, 2] > 0.5, "S4": lambda A: A[:, 0] <= 24}[bi]
p_stump = np.where(cond(Xo_te), ry, rn)
w, b, _ = train_logistic(Xtr, ytr, 0.1, 1000)
p_log = sigmoid(Xte @ w + b)
K = 5  # chosen in part2 (knn_best_k); read from the results file to stay consistent
with open("outputs/part2_results.txt", encoding="utf-8") as results_file:
    for line in results_file:
        if line.startswith("knn_best_k"):
            K = int(line.split("=")[1])
            break
p_knn = knn_prob(Xte, Xtr, ytr, K)
threshold_tolerance = 1e-9
pl, pk, ps = p_log >= 0.3 - threshold_tolerance, p_knn * K >= (K + 1) / 2 - threshold_tolerance, p_stump >= 0.3 - threshold_tolerance
hard = (pl.astype(int) + pk + ps) >= 2
soft = (p_log + p_knn + p_stump) / 3 >= 0.3 - threshold_tolerance
vrows = []
for name, pred in (("logistic", pl), ("knn", pk), ("stump", ps), ("hard_vote", hard), ("soft_vote", soft)):
    c = cells(pred, yte)
    acc, pr, rc, f1, sp = metrics(*c)
    vrows.append([name, *c, pr, rc, f1])
write_csv("outputs/part4_voting.csv", ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"], vrows)

# AdaBoost
yy = np.array([1, 1, 1, -1, -1, -1, 1, -1]); h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1])
h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1])
w = np.full(8, 1 / 8)
arows, alphas = [], []
for t, h in ((1, h1), (2, h2)):
    eps_t = w[h != yy].sum()
    al = 0.5 * np.log((1 - eps_t) / eps_t)
    un = w * np.exp(-al * yy * h)
    nw = un / un.sum()
    add_result("round%d_error" % t, eps_t); add_result("round%d_alpha" % t, al)
    for i in range(8):
        arows.append([t, i + 1, w[i], un[i], nw[i]])
    if t == 1:
        mis_w = nw[h != yy].sum()
    alphas.append(al)
    w = nw
write_csv("outputs/part4_adaboost.csv", ["round", "point", "weight_before", "weight_unnormalised", "weight_after"], arows)
ens = np.sign(alphas[0] * h1 + alphas[1] * h2)
add_result("ensemble_accuracy", float((ens == yy).mean()))
add_result("oob_fraction", (1 - 1 / 560) ** 560)

# architecture
mlp = [784 * 16 + 16, 16 * 16 + 16, 16 * 10 + 10]
conv1 = 6 * (5 * 5 * 1) + 6
conv2 = 16 * (5 * 5 * 6) + 16
s1 = (28 - 5 + 4) + 1  # 28
p1 = (s1 - 2) // 2 + 1  # 14
s2 = p1 - 5 + 1  # 10
p2 = (s2 - 2) // 2 + 1  # 5
flat = p2 * p2 * 16
d1 = flat * 32 + 32
d2 = 32 * 10 + 10
write_csv("outputs/part4_architecture.csv", ["layer", "parameters"],
          [["mlp_hidden1", mlp[0]], ["mlp_hidden2", mlp[1]], ["mlp_output", mlp[2]],
           ["cnn_conv1", conv1], ["cnn_conv2", conv2], ["cnn_dense1", d1], ["cnn_dense2", d2]])
add_result("mlp_total", sum(mlp)); add_result("cnn_total", conv1 + conv2 + d1 + d2)
add_result("fc_layer_params", 128 * 128 * 3 * 1024 + 1024)
add_result("conv_layer_params", 64 * (3 * 3 * 3) + 64)
add_result("rnn_total", 128 * 64 + 128 * 128 + 16 * 128 + 128 + 16)
write_results("outputs/part4_results.txt", R)
print("shapes", s1, p1, s2, p2, flat, "mis_w", mis_w)
