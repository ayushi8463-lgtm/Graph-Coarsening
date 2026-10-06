"""
Compare AH-UGC feature constructions on Cora (static).

Variants of the augmented feature F:
  orig       : [(1-a) * X  ,  a * A_row]            (current AH-UGC)
  neigh      : [(1-a) * X  ,  a * mean_nbr(X)]      (proposed)
  neigh_self : [(1-a) * X  ,  a * mean_{nbr+self}(X)]

Pipeline per run: build F -> LSH score (mean of l Gaussian projections) ->
sort -> random clockwise merges until target ratio -> coarsen graph ->
train GCN on coarse graph (train-node labels only) -> test on ORIGINAL graph.

Metrics: test accuracy, REE (spectral error), label purity (diagnostic).
Deps: numpy, scipy only. Cora is downloaded automatically.

Usage: python compare_ahugc_cora.py
"""
import os, pickle, urllib.request
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, svds

RATIOS = [0.5, 0.3]      # supernodes / nodes
SEEDS = range(5)
L_PROJ = 10              # number of LSH projectors
REE_K = 10               # top-k eigenvalues for REE

# ----------------------------------------------------------------- data
URL = "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/ind.cora."


def load_cora(root="cora_data"):
    os.makedirs(root, exist_ok=True)
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph", "test.index"]
    for n in names:
        p = f"{root}/ind.cora.{n}"
        if not os.path.exists(p):
            urllib.request.urlretrieve(URL + n, p)
    objs = []
    for n in names[:-1]:
        with open(f"{root}/ind.cora.{n}", "rb") as f:
            objs.append(pickle.load(f, encoding="latin1"))
    x, y, tx, ty, allx, ally, graph = objs
    test_idx = [int(l) for l in open(f"{root}/ind.cora.test.index")]
    test_sorted = np.sort(test_idx)

    X = sp.vstack((allx, tx)).tolil()
    X[test_idx, :] = X[test_sorted, :]
    X = np.asarray(X.todense(), dtype=np.float64)
    Y = np.vstack((ally, ty))
    Y[test_idx, :] = Y[test_sorted, :]
    labels = Y.argmax(1)

    N = X.shape[0]
    rows, cols = [], []
    for i, nbrs in graph.items():
        for j in nbrs:
            if i != j:
                rows.append(i); cols.append(j)
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64)

    train = np.zeros(N, bool); train[:len(y)] = True
    test = np.zeros(N, bool); test[test_sorted] = True
    return A, X, labels, train, test


# ------------------------------------------------------------ F variants
def heterophily_alpha(A, labels):
    """alpha = 1 - edge homophily (matches Table 5 of AH-UGC: Cora ~ 0.19).
    Uses all labels, as in the paper. Same alpha for every variant, so the
    comparison stays fair."""
    r, c = A.nonzero()
    return 1.0 - (labels[r] == labels[c]).mean()


def build_F(kind, A, X, alpha):
    deg = np.asarray(A.sum(1)).ravel()
    if kind == "orig":
        second = A.toarray()
    elif kind == "neigh":
        second = sp.diags(1 / np.maximum(deg, 1)) @ A @ X
    elif kind == "neigh_self":
        second = sp.diags(1 / (deg + 1)) @ (A + sp.eye(A.shape[0])) @ X
    else:
        raise ValueError(kind)
    return np.hstack([(1 - alpha) * X, alpha * second])


# ------------------------------------------------------------ coarsening
def top_pcs(F, k=5):
    """Top-k principal directions of F via truncated SVD (cost ~ O(N*d*k))."""
    Fc = F - F.mean(0)
    U, S, Vt = svds(Fc, k=k)
    o = np.argsort(-S)
    return Fc @ Vt[o].T, S[o]            # scores (N,k), singular values


def morton(P, bits=10):
    """Z-order key over the columns of P: keeps several axes in one 1D order."""
    q = np.floor((P - P.min(0)) / (np.ptp(P, 0) + 1e-12) * (2 ** bits - 1)).astype(np.int64)
    key = np.zeros(len(P), dtype=np.int64)
    for b in range(bits):
        for c in range(P.shape[1]):
            key |= ((q[:, c] >> b) & 1) << (b * P.shape[1] + c)
    return key


def hash_scores(F, mode, rng):
    if mode == "random":                  # current AH-UGC
        W = rng.normal(size=(F.shape[1], L_PROJ))
        b = rng.uniform(0, 1, size=L_PROJ)
        return (F @ W + b).mean(1)
    P, S = top_pcs(F)
    if mode == "pca1":                    # first principal direction only
        return P[:, 0]
    if mode == "pca_w":                   # top-5 PCs, weighted by importance
        return P @ (S / S.sum())
    if mode == "pca_z":                   # Z-order over top-3 PCs
        return morton(P[:, :3]).astype(np.float64)
    raise ValueError(mode)


def coarsen(F, ratio, rng, mode="random"):
    N = F.shape[0]
    s = hash_scores(F, mode, rng)
    groups = [[int(i)] for i in np.argsort(s, kind="stable")]
    target = int(round(ratio * N))
    while len(groups) > target:
        j = int(rng.integers(len(groups) - 1))    # random supernode
        groups[j] += groups.pop(j + 1)            # merge with right neighbour
    pi = np.empty(N, int)
    for g, members in enumerate(groups):
        pi[members] = g
    return pi


# --------------------------------------------------------------- metrics
def ree(A, C, k=REE_K):
    deg = np.asarray(A.sum(1)).ravel()
    L = sp.diags(deg) - A
    cnt = np.asarray(C.sum(0)).ravel()
    Cn = C @ sp.diags(cnt ** -0.5)
    Lc = Cn.T @ L @ Cn
    lam = np.sort(eigsh(L, k=k, which="LA", return_eigenvectors=False))
    lam_c = np.sort(eigsh(Lc, k=k, which="LA", return_eigenvectors=False))
    return np.mean(np.abs(lam_c - lam) / lam)


def purity(pi, labels):
    n = pi.max() + 1
    counts = np.zeros((n, labels.max() + 1))
    np.add.at(counts, (pi, labels), 1)
    return counts.max(1).sum() / len(pi)


# ------------------------------------------------------------ numpy GCN
def norm_adj(A):
    A = sp.csr_matrix(A)
    A.setdiag(0); A.eliminate_zeros()
    A = A + sp.eye(A.shape[0])
    d = np.asarray(A.sum(1)).ravel() ** -0.5
    return sp.diags(d) @ A @ sp.diags(d)


def train_gcn(Ahat, X, y, mask, rng, hid=64, epochs=200, lr=0.01, wd=5e-4, p=0.5):
    nc = int(y.max()) + 1
    lim1, lim2 = np.sqrt(6 / (X.shape[1] + hid)), np.sqrt(6 / (hid + nc))
    W = [rng.uniform(-lim1, lim1, (X.shape[1], hid)),
         rng.uniform(-lim2, lim2, (hid, nc))]
    m = [np.zeros_like(w) for w in W]; v = [np.zeros_like(w) for w in W]
    AX = Ahat @ X
    Y = np.eye(nc)[np.where(mask, y, 0)]
    for t in range(1, epochs + 1):
        Z1 = AX @ W[0]
        dm = (rng.random(Z1.shape) < 1 - p) / (1 - p)
        H = np.maximum(Z1, 0) * dm
        AH = Ahat @ H
        out = AH @ W[1]
        e = np.exp(out - out.max(1, keepdims=True))
        P = e / e.sum(1, keepdims=True)
        go = (P - Y) * mask[:, None] / mask.sum()
        g2 = AH.T @ go
        gH = Ahat @ (go @ W[1].T)
        g1 = AX.T @ (gH * (Z1 > 0) * dm) + wd * W[0]
        for i, g in enumerate((g1, g2)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            W[i] -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
    return W


def predict(W, Ahat, X):
    H = np.maximum(Ahat @ X @ W[0], 0)
    return (Ahat @ H @ W[1]).argmax(1)


# ------------------------------------------------------------------ main
def run(kind, mode, ratio, seed, A, X, Xn, labels, train, test, alpha, Ahat_full):
    rng = np.random.default_rng(seed)
    pi = coarsen(build_F(kind, A, X, alpha), ratio, rng, mode)
    N, n = len(pi), pi.max() + 1
    C = sp.csr_matrix((np.ones(N), (np.arange(N), pi)), shape=(N, n))
    cnt = np.asarray(C.sum(0)).ravel()

    Xc = sp.diags(1 / cnt) @ C.T @ Xn               # mean features
    Ac = C.T @ A @ C
    # majority label among TRAIN nodes only (no test-label leakage)
    votes = np.zeros((n, labels.max() + 1))
    tr = np.where(train)[0]
    np.add.at(votes, (pi[tr], labels[tr]), 1)
    yc = votes.argmax(1); mask = votes.sum(1) > 0

    W = train_gcn(norm_adj(Ac), Xc, yc, mask, rng)
    acc = (predict(W, Ahat_full, Xn)[test] == labels[test]).mean()
    return acc, ree(A, C), purity(pi, labels)


if __name__ == "__main__":
    A, X, labels, train, test = load_cora()
    Xn = X / np.maximum(X.sum(1, keepdims=True), 1)   # row-normalised for GCN
    Ahat_full = norm_adj(A)
    alpha = heterophily_alpha(A, labels)
    print(f"Cora: N={A.shape[0]}, alpha={alpha:.3f}, train={train.sum()}, test={test.sum()}\n")

    print(f"{'ratio':>5} {'F':<6} {'hash':<8} {'acc %':>13} {'REE':>13} {'purity %':>13}")
    for ratio in RATIOS:
        for kind in ["orig", "neigh"]:
            for mode in ["random", "pca1", "pca_w", "pca_z"]:
                res = np.array([run(kind, mode, ratio, s_, A, X, Xn, labels, train, test,
                                    alpha, Ahat_full) for s_ in SEEDS])
                mu, sd = res.mean(0), res.std(0)
                print(f"{ratio:>5} {kind:<6} {mode:<8} "
                      f"{100*mu[0]:6.2f}±{100*sd[0]:4.2f} "
                      f"{mu[1]:6.3f}±{sd[1]:5.3f} "
                      f"{100*mu[2]:6.2f}±{100*sd[2]:4.2f}", flush=True)
        print()
