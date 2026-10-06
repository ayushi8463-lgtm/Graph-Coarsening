"""
Fixed-hash (deterministic PCA) comparison on Citeseer and PubMed. Random merge kept for all.

Citeseer : full pipeline (GCN accuracy, REE, purity), F in {orig, neigh} x hash in {random, pca_z}.
PubMed   : coarsening-only (the numpy GCN is too slow at 20k nodes): purity, REE, hash time, merge time;
           neigh F only (the adjacency-row F would need a 20k x 20k dense block).
Needs compare_ahugc_cora.py in the same folder.   Usage: python compare_more_datasets.py [citeseer] [pubmed]
"""
import os, sys, time, pickle, urllib.request
import numpy as np
import scipy.sparse as sp
import compare_ahugc_cora as cc

URL = "https://raw.githubusercontent.com/kimiyoung/planetoid/master/data/ind."


def load_planetoid(name, root="planetoid_data"):
    os.makedirs(root, exist_ok=True)
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph", "test.index"]
    for n in names:
        p = f"{root}/ind.{name}.{n}"
        if not os.path.exists(p):
            urllib.request.urlretrieve(f"{URL}{name}.{n}", p)
    objs = [pickle.load(open(f"{root}/ind.{name}.{n}", "rb"), encoding="latin1") for n in names[:-1]]
    x, y, tx, ty, allx, ally, graph = objs
    test_idx = [int(l) for l in open(f"{root}/ind.{name}.test.index")]
    test_sorted = np.sort(test_idx)
    if name == "citeseer":                       # a few isolated test nodes are missing: pad with zeros
        full = np.arange(test_sorted.min(), test_sorted.max() + 1)
        tx_e = sp.lil_matrix((len(full), x.shape[1])); tx_e[test_sorted - test_sorted.min(), :] = tx; tx = tx_e
        ty_e = np.zeros((len(full), y.shape[1])); ty_e[test_sorted - test_sorted.min(), :] = ty; ty = ty_e
    X = sp.vstack((allx, tx)).tolil(); X[test_idx, :] = X[test_sorted, :]
    X = np.asarray(X.todense(), dtype=np.float64)
    Y = np.vstack((ally, ty)); Y[test_idx, :] = Y[test_sorted, :]
    labels = Y.argmax(1)
    N = X.shape[0]
    rows, cols = [], []
    for i, nb in graph.items():
        for j in nb:
            if i != j and i < N and j < N:
                rows.append(i); cols.append(j)
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64)
    train = np.zeros(N, bool); train[:len(y)] = True
    test = np.zeros(N, bool); test[test_sorted] = True
    return A, X, labels, train, test


def line(res):
    mu, sd = res.mean(0), res.std(0)
    return f"{100*mu[0]:6.2f}±{100*sd[0]:4.2f}  {mu[1]:6.3f}±{sd[1]:5.3f}  {100*mu[2]:6.2f}±{100*sd[2]:4.2f}"


if __name__ == "__main__":
    which = sys.argv[1:] or ["citeseer", "pubmed"]
    if "citeseer" in which:
        A, X, labels, train, test = load_planetoid("citeseer")
        Xn = X / np.maximum(X.sum(1, keepdims=True), 1)
        Ahat = cc.norm_adj(A); alpha = cc.heterophily_alpha(A, labels)
        print(f"=== Citeseer: N={A.shape[0]} train={train.sum()} test={test.sum()} alpha={alpha:.3f} ===", flush=True)
        print(f"{'ratio':>5} {'F':<6} {'hash':<8}{'acc %':>13}  {'REE':>13}  {'purity %':>13}", flush=True)
        for ratio in [0.5, 0.3]:
            for kind in ["orig", "neigh"]:
                for mode in ["random", "pca_z"]:
                    res = np.array([cc.run(kind, mode, ratio, s, A, X, Xn, labels, train, test, alpha, Ahat)
                                    for s in range(5)])
                    print(f"{ratio:>5} {kind:<6} {mode:<8}{line(res)}", flush=True)
            print(flush=True)
    if "pubmed" in which:
        A, X, labels, train, test = load_planetoid("pubmed")
        alpha = cc.heterophily_alpha(A, labels)
        print(f"=== PubMed (coarsening only): N={A.shape[0]} edges={A.nnz//2} alpha={alpha:.3f}, neigh F, 3 seeds ===", flush=True)
        t0 = time.time(); F = cc.build_F("neigh", A, X, alpha); print(f"build F: {time.time()-t0:.2f}s, F shape {F.shape}", flush=True)
        print(f"{'ratio':>5} {'hash':<8}{'purity %':>14}  {'REE':>13}  {'hash s':>7}  {'merge s':>8}", flush=True)
        for ratio in [0.5, 0.3]:
            for mode in ["random", "pca_z"]:
                out = []
                for s in range(3):
                    rng = np.random.default_rng(s)
                    t1 = time.time(); sc = cc.hash_scores(F, mode, rng); th = time.time() - t1
                    t2 = time.time(); pi = cc.coarsen(F, ratio, np.random.default_rng(s), mode); tt = time.time() - t2
                    N, n = len(pi), pi.max() + 1
                    C = sp.csr_matrix((np.ones(N), (np.arange(N), pi)), shape=(N, n))
                    out.append((cc.purity(pi, labels), cc.ree(A, C), th, tt - th))
                o = np.array(out)
                print(f"{ratio:>5} {mode:<8}{100*o[:,0].mean():7.2f}±{100*o[:,0].std():4.2f}  "
                      f"{o[:,1].mean():6.3f}±{o[:,1].std():5.3f}  {o[:,2].mean():7.2f}  {o[:,3].mean():8.2f}", flush=True)
