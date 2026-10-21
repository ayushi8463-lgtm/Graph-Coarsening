"""
How much should the neighbor block count? neigh F + pca_z hash, ratio 0.5, random merge, 5 seeds.

 raw  : F = [(1-a) X , a * nbr_avg(X)]                (current code; a scales the raw blocks)
 norm : F = [(1-a) X/||Xc|| , a * nbr_avg/||Nc||]     (each block scaled to unit total variance first,
                                                       so (1-a)^2 : a^2 really is the variance share)
Also prints the *effective* variance share of the neighbor block under the current (raw) weighting.
"""
import sys, numpy as np, scipy.sparse as sp
import compare_ahugc_cora as cc
import compare_more_datasets as cm

def blocks(A, X):
    deg = np.asarray(A.sum(1)).ravel()
    Nb = np.asarray(sp.diags(1 / np.maximum(deg, 1)) @ A @ X)
    return Nb

def build_F(kind, A, X, a):
    Nb = blocks(A, X)
    if kind == "neighN":
        X = X / np.linalg.norm(X - X.mean(0)); Nb = Nb / np.linalg.norm(Nb - Nb.mean(0))
    return np.hstack([(1 - a) * X, a * Nb]).astype(np.float32)

cc.build_F = build_F

def share(A, X, a):
    Nb = blocks(A, X)
    vx = ((1 - a) ** 2) * np.sum((X - X.mean(0)) ** 2); vn = (a ** 2) * np.sum((Nb - Nb.mean(0)) ** 2)
    return vn / (vx + vn)

def go(name, A, X, labels, train, test):
    Xn = X / np.maximum(X.sum(1, keepdims=True), 1)
    Ahat = cc.norm_adj(A); a0 = cc.heterophily_alpha(A, labels)
    print(f"\n=== {name}: computed alpha = {a0:.3f}; neighbor block's effective variance share at that alpha (raw) = "
          f"{100*share(A, X, a0):.1f}%  (nominal weight a^2/((1-a)^2+a^2) = {100*a0**2/((1-a0)**2+a0**2):.1f}%) ===", flush=True)
    print(f"{'F':<7}{'alpha':>6}{'acc %':>13}{'REE':>14}{'purity %':>13}", flush=True)
    cfg = [("neigh", 0.0), ("neigh", 0.1), ("neigh", a0), ("neigh", 0.5), ("neigh", 1.0),
           ("neighN", 0.1), ("neighN", a0), ("neighN", 0.5)]
    for kind, a in cfg:
        res = np.array([cc.run(kind, "pca_z", 0.5, s, A, X, Xn, labels, train, test, a, Ahat) for s in range(5)])
        mu, sd = res.mean(0), res.std(0)
        lab = {"neigh": "raw", "neighN": "norm"}[kind]
        print(f"{lab:<7}{a:>6.2f}{100*mu[0]:7.2f}±{100*sd[0]:4.2f} {mu[1]:7.3f}±{sd[1]:5.3f}{100*mu[2]:7.2f}±{100*sd[2]:4.2f}", flush=True)

if __name__ == "__main__":
    which = sys.argv[1:] or ["cora", "citeseer"]
    if "cora" in which:
        A, X, labels, train, test = cc.load_cora(); go("Cora", A, X, labels, train, test)
    if "citeseer" in which:
        A, X, labels, train, test = cm.load_planetoid("citeseer"); go("Citeseer", A, X, labels, train, test)
