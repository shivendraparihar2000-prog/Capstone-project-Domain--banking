import numpy as np
from common import load, write_results

X, y, v = load()
n = len(v)
R = []


def add_result(key, value):
    R.append((key, value))
m = v.mean(); sd = v.std(ddof=1); se = sd / np.sqrt(n)
add_result("n", n); add_result("mean_value", m); add_result("sd_value", sd); add_result("se_value", se)
add_result("ci95_low", m - 1.96 * se); add_result("ci95_high", m + 1.96 * se)
z = (m - 8.0) / se
add_result("z_stat", z); add_result("reject_h0", abs(z) > 1.96)
add_result("n_required", int(np.ceil((1.96 * sd / 0.5) ** 2)))

x4 = X[:, 3]


def simple(x, yy):
    xm, ym = x.mean(), yy.mean()
    cov = ((x - xm) * (yy - ym)).sum() / (len(x) - 1)
    var = ((x - xm) ** 2).sum() / (len(x) - 1)
    s = cov / var
    return s, ym - s * xm


s, b0 = simple(x4, v)
fit = b0 + s * x4
tss = ((v - m) ** 2).sum(); ess = ((fit - m) ** 2).sum(); rss = ((v - fit) ** 2).sum()
add_result("reg_slope", s); add_result("reg_intercept", b0); add_result("tss", tss); add_result("ess", ess); add_result("rss", rss)
add_result("tss_equals_ess_plus_rss", abs(tss - ess - rss) < 1e-6 * tss)
add_result("r2", 1 - rss / tss); add_result("rmse", np.sqrt(rss / n)); add_result("mae", np.abs(v - fit).mean())

# gradient descent
mx, sx = x4.mean(), x4.std(ddof=1)
zz = (x4 - mx) / sx
mm = cc = 0.0
def mean_squared_error():
    predictions = mm * zz + cc
    return float(np.mean((v - predictions) ** 2))


add_result("gd_J0", mean_squared_error())
for t in range(1, 501):
    e = v - (mm * zz + cc)
    gm = -2 / n * (zz * e).sum(); gc = -2 / n * e.sum()
    mm -= 0.1 * gm; cc -= 0.1 * gc
    if t in (1, 10, 100, 500):
        add_result("gd_J%d" % t, mean_squared_error())
gs = mm / sx; gi = cc - mm * mx / sx
add_result("gd_slope", gs); add_result("gd_intercept", gi); add_result("gd_slope_gap", abs(gs - s))


def halves(fitted, resid):
    o = np.argsort(fitted, kind="stable")
    lo, hi = resid[o[:400]].std(ddof=1), resid[o[400:]].std(ddof=1)
    return lo, hi, hi / lo


lo, hi, r = halves(fit, v - fit)
add_result("resid_sd_lower_raw", lo); add_result("resid_sd_upper_raw", hi); add_result("resid_sd_ratio_raw", r)
lx, lv = np.log(x4), np.log(v)
ls, li = simple(lx, lv)
lf = li + ls * lx
lrss = ((lv - lf) ** 2).sum(); ltss = ((lv - lv.mean()) ** 2).sum()
lr2 = 1 - lrss / ltss
add_result("loglog_intercept", li); add_result("loglog_slope", ls); add_result("loglog_r2", lr2)
add_result("loglog_rmse", np.sqrt(lrss / n))
lo, hi, r = halves(lf, lv - lf)
add_result("resid_sd_lower_log", lo); add_result("resid_sd_upper_log", hi); add_result("resid_sd_ratio_log", r)

D = np.column_stack([np.ones(n), lx, X[:, 0], X[:, 1], X[:, 2], X[:, 4], X[:, 5], X[:, 6]])
beta = np.linalg.solve(D.T @ D, D.T @ lv)
for i, bv in enumerate(beta):
    add_result("mr_beta_%d" % i, bv)
res = lv - D @ beta
r2 = 1 - (res ** 2).sum() / ltss
add_result("mr_r2", r2); add_result("mr_adj_r2", 1 - (1 - r2) * (n - 1) / (n - 7 - 1))
add_result("loglog_adj_r2", 1 - (1 - lr2) * (n - 1) / (n - 1 - 1))
write_results("outputs/part1_results.txt", R)
