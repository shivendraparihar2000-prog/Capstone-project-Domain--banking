import numpy as np
from common import load, write_results, write_csv
X, y, v = load()
C = X[:, :5]
n = len(C)
Z = (C - C.mean(0)) / C.std(0, ddof=1)
R = []
def add_result(key, value):
    R.append((key, value))

def kmeans(Z, K):
    step = n // K
    cen = Z[[i * step for i in range(K)]].copy()
    lab = None
    passes = 0
    while True:
        d = ((Z[:, None, :] - cen[None, :, :]) ** 2).sum(2)
        new = np.argmin(d, axis=1)
        passes += 1

        if lab is not None and (new == lab).all():
            break
        lab = new
        for k in range(K):
            if (lab == k).any():
                cen[k] = Z[lab == k].mean(0)

    wcss = float(((Z - cen[lab]) ** 2).sum())
    return lab, cen, wcss, passes

res = {}
for K in range(2, 7):
    res[K] = kmeans(Z, K)
    add_result("wcss_k%d" % K, res[K][2])
w = {K: res[K][2] for K in res}
elbow = None
for K in (3, 4, 5):
    if w[K] - w[K + 1] < 0.5 * (w[K - 1] - w[K]):
        elbow = K
        break
add_result("elbow_k", elbow if elbow is not None else 0)
lab = res[3][0]
order = np.argsort([C[lab == k, 2].mean() for k in range(3)])
seg = np.zeros(n, dtype=int)
for rank, k in enumerate(order):
    seg[lab == k] = rank
rows = []
for s, name in enumerate("ABC"):
    m = seg == s
    add_result("size_segment_" + name, int(m.sum()))
    rows.append([name, int(m.sum())] + [C[m, j].mean() for j in range(5)] + [y[m].mean(), v[m].mean()])
write_csv("outputs/part3_segments.csv",
          ["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5", "outcome_rate", "mean_value"], rows)

# hierarchical
D = np.array([[0, 7.9, 7.4, 2.9, 2.9, 6.1], [7.9, 0, 1.4, 5.9, 5.3, 2.0], [7.4, 1.4, 0, 5.8, 4.6, 1.3],
              [2.9, 5.9, 5.8, 0, 2.8, 4.5], [2.9, 5.3, 4.6, 2.8, 0, 3.4], [6.1, 2.0, 1.3, 4.5, 3.4, 0]])
hrows = []
for link in ("single", "complete", "average"):
    cl = [[i] for i in range(6)]
    heights = []
    for step in range(1, 6):
        best = None
        for a in range(len(cl)):
            for b in range(a + 1, len(cl)):
                ds = [D[i, j] for i in cl[a] for j in cl[b]]
                h = min(ds) if link == "single" else (max(ds) if link == "complete" else sum(ds) / len(ds))
                if best is None or h < best[0]:
                    best = (h, a, b)
        h, a, b = best
        A, B = sorted(cl[a]), sorted(cl[b])
        if B[0] < A[0]:
            A, B = B, A
        nm = lambda c: "+".join("E%d" % (i + 1) for i in c)
        hrows.append([link, step, nm(A), nm(B), h])
        heights.append(h)
        cl = [c for i, c in enumerate(cl) if i not in (a, b)] + [sorted(A + B)]
    gaps = [heights[i + 1] - heights[i] for i in range(4)]
    g = int(np.argmax(gaps))  # gap between merge g+1 and g+2 -> g+1 merges below cut
    add_result(link + "_clusters_at_cut", 6 - (g + 1))
write_csv("outputs/part3_hierarchical.csv", ["linkage", "merge_no", "cluster_a", "cluster_b", "height"], hrows)
# PCA
mean_vec = Z.mean(0)
Xc = Z - mean_vec
S = Xc.T @ Xc / (n - 1)
ev, vec = np.linalg.eigh(S)
o = np.argsort(-ev)
ev, vec = ev[o], vec[:, o]
for j in range(5):
    if vec[np.argmax(np.abs(vec[:, j])), j] < 0:
        vec[:, j] = -vec[:, j]
add_result("eigenvalue_sum", ev.sum())
for j in range(5):
    add_result("eigenvalue_%d" % (j + 1), ev[j])
for j in range(5):
    add_result("explained_ratio_%d" % (j + 1), ev[j] / ev.sum())
cum = np.cumsum(ev) / ev.sum()
for j in range(5):
    add_result("cumulative_ratio_%d" % (j + 1), cum[j])
add_result("components_for_80pct", int(np.argmax(cum >= 0.8)) + 1)
for p in (0, 1):
    for j in range(5):
        add_result("pc%d_loading_X%d" % (p + 1, j + 1), vec[j, p])
P = Xc @ vec[:, :2]
for r in range(3):
    add_result("row%d_pc1" % (r + 1), P[r, 0]); add_result("row%d_pc2" % (r + 1), P[r, 1])
for s, name in enumerate("ABC"):
    add_result("mean_pc1_segment_" + name, P[seg == s, 0].mean())
write_results("outputs/part3_results.txt", R)
print("mean vector max abs:", np.abs(mean_vec).max())
