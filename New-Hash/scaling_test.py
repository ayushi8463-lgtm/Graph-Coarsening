"""
Synthetic scaling of the hash step: random projection vs PCA (neigh F), plus the random merge.
Random sparse graph (avg degree ~10), 500 binary features (2% density). F width is 1000 for every N.
The adjacency-row F would need an N x N dense block (N^2 * 4 bytes), printed for reference.
"""
import time, numpy as np, scipy.sparse as sp
import compare_ahugc_cora as cc

rng = np.random.default_rng(0); d = 500; alpha = 0.5
print(f"{'N':>7} {'build F':>8} {'random hash':>12} {'pca_z hash':>11} {'rand merge':>11} {'orig F size':>12}", flush=True)
for N in [5000, 10000, 20000, 50000, 100000]:
    r = np.repeat(np.arange(N), 5); c = rng.integers(0, N, size=5 * N)
    A = sp.coo_matrix((np.ones(len(r), np.float32), (r, c)), shape=(N, N)).tocsr()
    A = ((A + A.T) > 0).astype(np.float32)
    X = (rng.random((N, d)) < 0.02).astype(np.float32)
    t = time.time()
    deg = np.asarray(A.sum(1)).ravel()
    F = np.hstack([(1 - alpha) * X, alpha * np.asarray(sp.diags((1 / np.maximum(deg, 1)).astype(np.float32)) @ A @ X)]).astype(np.float32)
    t_build = time.time() - t
    W = rng.normal(size=(F.shape[1], 10)).astype(np.float32); b = rng.uniform(0, 1, 10).astype(np.float32)
    t = time.time(); s = (F @ W + b).mean(1); order = np.argsort(s); t_rand = time.time() - t
    t = time.time(); P, _ = cc.top_pcs(F); order = np.argsort(cc.morton(P[:, :3])); t_pca = time.time() - t
    t = time.time()
    groups = [[int(i)] for i in order]; g = np.random.default_rng(1)
    while len(groups) > N // 2:
        j = int(g.integers(len(groups) - 1)); groups[j] += groups.pop(j + 1)
    t_merge = time.time() - t
    print(f"{N:>7} {t_build:>7.2f}s {t_rand:>11.2f}s {t_pca:>10.2f}s {t_merge:>10.2f}s {N*N*4/1e9:>10.1f}GB", flush=True)
    del F, X, A, groups
