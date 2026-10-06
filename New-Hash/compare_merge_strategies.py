"""
Merge-step comparison. Hash = pca_z (neighbor-averaged F) for all, so the sorted list is identical;
only the way supernodes are chosen from it differs.

  rand : current AH-UGC step (random node, merge with right neighbour)
  gap  : cut the sorted list at the (n-1) largest feature gaps between adjacent nodes
  heap : min-heap over adjacent groups, cost = Ward cost
         n_a*n_b/(n_a+n_b) * ||mean_a - mean_b||^2, merge cheapest, update its 2 neighbours

Reports accuracy / REE / purity, largest supernode, and merge-step time.
Needs compare_ahugc_cora.py and compare_ahugc_hetero.py in the same folder.
"""
import sys, time, heapq
import numpy as np
import compare_ahugc_cora as cc
import compare_ahugc_hetero as ch

STATS = []                                   # (max supernode size, merge seconds) per call


def merge_rand(F, order, n, rng):
    groups = [[int(i)] for i in order]
    while len(groups) > n:
        j = int(rng.integers(len(groups) - 1))
        groups[j] += groups.pop(j + 1)
    pi = np.empty(len(order), int)
    for g, m in enumerate(groups):
        pi[m] = g
    return pi


def merge_gap(F, order, n, rng=None):
    Fo = F[order].astype(np.float64)
    gaps = np.linalg.norm(Fo[1:] - Fo[:-1], axis=1)
    cut = np.sort(np.argsort(-gaps, kind="stable")[: n - 1])   # n-1 biggest gaps
    seg = np.zeros(len(order), int)
    seg[cut + 1] = 1
    seg = np.cumsum(seg)                                       # segment id per sorted position
    pi = np.empty(len(order), int)
    pi[order] = seg
    return pi


def merge_heap(F, order, n, rng=None):
    N = len(order)
    S = F[order].astype(np.float64)          # running sums per group (indexed by sorted position)
    cnt = np.ones(N)
    nxt = np.arange(1, N + 1); prv = np.arange(-1, N - 1)
    alive = np.ones(N, bool); ver = np.zeros(N, int); parent = np.arange(N)

    def cost(a, b):
        d = S[a] / cnt[a] - S[b] / cnt[b]
        return cnt[a] * cnt[b] / (cnt[a] + cnt[b]) * float(d @ d)

    d0 = S[1:] - S[:-1]
    c0 = 0.5 * (d0 * d0).sum(1)              # two singletons: 1*1/2 * ||diff||^2
    heap = [(float(c0[i]), i, i + 1, 0, 0) for i in range(N - 1)]
    heapq.heapify(heap)
    groups = N
    while groups > n:
        c, a, b, va, vb = heapq.heappop(heap)
        if not (alive[a] and alive[b]) or ver[a] != va or ver[b] != vb or nxt[a] != b:
            continue                         # stale entry
        S[a] += S[b]; cnt[a] += cnt[b]
        alive[b] = False; parent[b] = a; ver[a] += 1
        nxt[a] = nxt[b]
        if nxt[a] < N:
            prv[nxt[a]] = a
        groups -= 1
        if prv[a] >= 0:
            p = prv[a]; heapq.heappush(heap, (cost(p, a), p, a, ver[p], ver[a]))
        if nxt[a] < N:
            q = nxt[a]; heapq.heappush(heap, (cost(a, q), a, q, ver[a], ver[q]))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    roots = np.array([root(i) for i in range(N)])
    _, seg = np.unique(roots, return_inverse=True)
    pi = np.empty(N, int)
    pi[order] = seg
    return pi


MERGE = {"rand": merge_rand, "gap": merge_gap, "heap": merge_heap}


def coarsen(F, ratio, rng, mode="random"):
    h, strat = mode.split("+")
    order = np.argsort(cc.hash_scores(F, h, rng), kind="stable")
    n = int(round(ratio * len(F)))
    t0 = time.time()
    pi = MERGE[strat](F, order, n, rng)
    STATS.append((np.bincount(pi).max(), time.time() - t0))
    assert pi.max() + 1 == n, (pi.max() + 1, n)
    return pi


cc.coarsen = coarsen; ch.coarsen = coarsen
MODES = ["pca_z+rand", "pca_z+gap", "pca_z+heap"]


def line(res, st):
    mu, sd = res.mean(0), res.std(0)
    return (f"{100*mu[0]:6.2f}±{100*sd[0]:4.2f}  {mu[1]:6.3f}±{sd[1]:5.3f}  {100*mu[2]:6.2f}±{100*sd[2]:4.2f}"
            f"  {st[:,0].mean():7.0f}  {st[:,1].mean():8.3f}")


HEAD = f"{'':>5} {'merge':<6}{'acc %':>13}  {'REE':>13}  {'purity %':>13}  {'max size':>8}  {'merge s':>8}"

if __name__ == "__main__":
    which = sys.argv[1:] or ["cora", "cornell"]
    if "cora" in which:
        A, X, labels, train, test = cc.load_cora()
        Xn = X / np.maximum(X.sum(1, keepdims=True), 1)
        Ahat = cc.norm_adj(A); alpha = cc.heterophily_alpha(A, labels)
        print(f"=== Cora (GCN, neigh F + pca_z hash, alpha={alpha:.3f}) ===")
        for ratio in [0.5, 0.3]:
            print(f"ratio {ratio}\n{HEAD}", flush=True)
            for mode in MODES:
                k = len(STATS)
                res = np.array([cc.run("neigh", mode, ratio, s, A, X, Xn, labels, train, test, alpha, Ahat)
                                for s in range(5)])
                print(f"{'':>5} {mode.split('+')[1]:<6}{line(res, np.array(STATS[k:]))}", flush=True)
            print()
    if "cornell" in which:
        A, X, labels = ch.load("cornell")
        Xn = X / np.maximum(X.sum(1, keepdims=True), 1)
        S_full = ch.sgc_feats(cc.norm_adj(A), Xn); alpha = cc.heterophily_alpha(A, labels)
        print(f"=== Cornell (SGC, neigh F + pca_z hash, alpha={alpha:.3f}, ratio 0.5, 10 splits) ===\n{HEAD}", flush=True)
        for mode in MODES:
            k = len(STATS)
            res = np.array([ch.run("neigh", mode, alpha, i, A, X, Xn, labels, *ch.split("cornell", i), S_full)
                            for i in range(10)])
            print(f"{'':>5} {mode.split('+')[1]:<6}{line(res, np.array(STATS[k:]))}", flush=True)
