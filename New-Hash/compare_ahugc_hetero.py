"""
Heterophilic benchmark: AH-UGC (orig F + random hash) vs proposed (neigh F + PCA Z-order hash)
on Cornell and Film (Actor) from Geom-GCN, 50% coarsening, SGC classifier
(same dataset / ratio / model family as Table 9 of the AH-UGC paper).

Part 1: 2x2 grid {orig, neigh} x {random, pca_z} at the dataset's own alpha.
Part 2: alpha sweep for the two end configs.
Reuses coarsen / ree / purity / norm_adj / heterophily_alpha from compare_ahugc_cora.py.
Train labels only (Geom-GCN 60/20/20 splits); test on the ORIGINAL graph.

Usage: python compare_ahugc_hetero.py
"""
import os, sys, urllib.request
import numpy as np
import scipy.sparse as sp
from compare_ahugc_cora import coarsen, ree, purity, norm_adj, heterophily_alpha

BASE = "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master/"
ROOT = "hetero_data"
RATIO = 0.5


def fetch(name):
    os.makedirs(ROOT, exist_ok=True)
    files = {f"{name}_edges.txt": f"new_data/{name}/out1_graph_edges.txt",
             f"{name}_nodes.txt": f"new_data/{name}/out1_node_feature_label.txt"}
    for i in range(10):
        files[f"{name}_split_{i}.npz"] = f"splits/{name}_split_0.6_0.2_{i}.npz"
    for local, remote in files.items():
        p = f"{ROOT}/{local}"
        if not os.path.exists(p):
            urllib.request.urlretrieve(BASE + remote, p)


def load(name):
    fetch(name)
    rows = [l.split("\t") for l in open(f"{ROOT}/{name}_nodes.txt").read().strip().split("\n")[1:]]
    N = len(rows)
    labels = np.zeros(N, int)
    if name == "film":                       # sparse: comma separated feature indices
        d = 932                              # indices run 0..931 (as in PyG's Actor loader)
        X = np.zeros((N, d), np.float32)
        for nid, feat, lab in rows:
            X[int(nid), [int(i) for i in feat.split(",")]] = 1
            labels[int(nid)] = int(lab)
    else:                                    # dense comma separated values
        d = len(rows[0][1].split(","))
        X = np.zeros((N, d), np.float32)
        for nid, feat, lab in rows:
            X[int(nid)] = np.array(feat.split(","), dtype=np.float32)
            labels[int(nid)] = int(lab)
    e = np.array([l.split("\t") for l in open(f"{ROOT}/{name}_edges.txt").read().strip().split("\n")[1:]], int)
    A = sp.coo_matrix((np.ones(len(e), np.float32), (e[:, 0], e[:, 1])), shape=(N, N)).tocsr()
    A = ((A + A.T) > 0).astype(np.float32)
    A.setdiag(0); A.eliminate_zeros()
    return A, X, labels


def split(name, i):
    z = np.load(f"{ROOT}/{name}_split_{i}.npz")
    return z["train_mask"].astype(bool), z["test_mask"].astype(bool)


def build_F(kind, A, X, alpha):
    deg = np.asarray(A.sum(1)).ravel()
    if kind == "orig":
        second = A.toarray()
    elif kind == "neigh":
        second = np.asarray(sp.diags(1 / np.maximum(deg, 1)).astype(np.float32) @ A @ X)
    else:
        raise ValueError(kind)
    return np.hstack([(1 - alpha) * X, alpha * second]).astype(np.float32)


# ---- SGC: logistic regression on A_hat^2 X (numpy) ----
def sgc_feats(Ahat, X, K=2):
    for _ in range(K):
        X = Ahat @ X
    return X


def train_sgc(S, y, mask, epochs=100, lr=0.2, wd=5e-5):
    nc = int(y.max()) + 1
    W = np.zeros((S.shape[1], nc)); m = np.zeros_like(W); v = np.zeros_like(W)
    Y = np.eye(nc)[np.where(mask, y, 0)]
    for t in range(1, epochs + 1):
        out = S @ W
        e = np.exp(out - out.max(1, keepdims=True)); P = e / e.sum(1, keepdims=True)
        g = S.T @ ((P - Y) * mask[:, None] / mask.sum()) + wd * W
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        W -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return W


def run(kind, mode, alpha, seed, A, X, Xn, labels, train, test, S_full):
    rng = np.random.default_rng(seed)
    pi = coarsen(build_F(kind, A, X, alpha), RATIO, rng, mode)
    N, n = len(pi), pi.max() + 1
    C = sp.csr_matrix((np.ones(N), (np.arange(N), pi)), shape=(N, n))
    cnt = np.asarray(C.sum(0)).ravel()
    Xc = sp.diags(1 / cnt) @ C.T @ Xn
    Ac = C.T @ A @ C
    votes = np.zeros((n, labels.max() + 1))
    tr = np.where(train)[0]
    np.add.at(votes, (pi[tr], labels[tr]), 1)
    yc = votes.argmax(1); mask = votes.sum(1) > 0
    W = train_sgc(sgc_feats(norm_adj(Ac), Xc), yc, mask)
    acc = ((S_full @ W).argmax(1)[test] == labels[test]).mean()
    return acc, ree(A, C), purity(pi, labels)


def fmt(res):
    mu, sd = res.mean(0), res.std(0)
    return f"{100*mu[0]:6.2f}±{100*sd[0]:4.2f}  {mu[1]:6.3f}±{sd[1]:5.3f}  {100*mu[2]:6.2f}±{100*sd[2]:4.2f}"


if __name__ == "__main__":
    plan = {"cornell": 10, "film": 5}
    if len(sys.argv) > 1:                    # e.g. python compare_ahugc_hetero.py film
        plan = {k: v for k, v in plan.items() if k in sys.argv[1:]}
    sweep_splits = {"cornell": 10, "film": 3}
    for name, nspl in plan.items():
        A, X, labels = load(name)
        Xn = X / np.maximum(X.sum(1, keepdims=True), 1)
        Ahat = norm_adj(A); S_full = sgc_feats(Ahat, Xn)
        alpha = heterophily_alpha(A, labels)
        print(f"\n=== {name}: N={A.shape[0]} edges={A.nnz//2} alpha={alpha:.3f} (splits={nspl}) ===", flush=True)

        base = []
        for i in range(nspl):
            tr, te = split(name, i)
            W = train_sgc(S_full, labels, tr)
            base.append((S_full @ W).argmax(1)[te].__eq__(labels[te]).mean())
        print(f"Base SGC (no coarsening): {100*np.mean(base):.2f}±{100*np.std(base):.2f}", flush=True)

        print(f"\n[Part 1] {'F':<6}{'hash':<8}{'acc %':>13}  {'REE':>13}  {'purity %':>13}", flush=True)
        for kind in ["orig", "neigh"]:
            for mode in ["random", "pca_z"]:
                res = np.array([run(kind, mode, alpha, i, A, X, Xn, labels, *split(name, i), S_full)
                                for i in range(nspl)])
                print(f"         {kind:<6}{mode:<8}{fmt(res)}", flush=True)

        print(f"\n[Part 2: alpha sweep, {sweep_splits[name]} splits]  alpha  config{'':<14}{'acc %':>13}", flush=True)
        for a in [0.0, 0.3, 0.6, 0.9]:
            for kind, mode in [("orig", "random"), ("neigh", "pca_z")]:
                res = np.array([run(kind, mode, a, i, A, X, Xn, labels, *split(name, i), S_full)
                                for i in range(sweep_splits[name])])
                print(f"                      {a:<6} {kind+'+'+mode:<19}{100*res[:,0].mean():6.2f}±{100*res[:,0].std():4.2f}", flush=True)
