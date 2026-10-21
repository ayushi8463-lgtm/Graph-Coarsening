"""
ahugc_eval.py -- AH-UGC coarsening (random hash or PCA Z-order hash) + node classification
trained on the coarsened graph and tested on the ORIGINAL graph.

Datasets : cora, citeseer, pubmed (Planetoid), cornell (WebKB), film (Actor),
           chameleon, squirrel (WikipediaNetwork)                                [PyG, auto-download]
F        : feat  = X only
           orig  = [(1-a) X | a A]                     (AH-UGC / UGC augmented feature)
           neigh = [(1-a) X | a mean_nbr(X)]
hash     : random = AH-UGC Alg. 1: l=500 Gaussian projectors, s_i = mean_k (W_k.F_i + b_k),
                    sort, repeatedly pick a random super-node and merge it with its right
                    neighbour until n = ratio*N          (no bin width)
           pca_z  = same merge procedure, but the order comes from a Morton (Z-order) key over
                    the top-3 principal components of F (centred, sparse-safe)
           shingle   = same merge procedure; the order is a neighbourhood-MinHash ("shingle") order: every node gets
                    sig_k(v) = min over its closed neighbourhood N[v] of a random rank r_k(u), k = 1..3, nodes are
                    sorted lexicographically by (sig_1, sig_2, sig_3).  Structure only, O(E), ignores F and X.
           dpca      = Z-order over the top-3 PCs of DIFFUSED features  A_hat^2 X  (A_hat = D^-1/2 (A+I) D^-1/2)
           shingle_z = MinHash sig_1 of N[v] first, Z-order key over the top-3 PCs of X second
           shingle / dpca / shingle_z use the graph A and the features X directly, so F only changes what the
           classifier sees (use --F feat to avoid running identical coarsenings three times).
Coarse graph (AH-UGC paper, Sec. 3.1):  A_c = C^T A C (weighted, binary C),  F_c = mean of member rows,
           super-node label = majority vote over TRAIN nodes only, loss only on super-nodes that
           contain a train node.
Models   :
  gcn = UGC repo GCN.py  (GCN_: 3 x GCNConv  in->64->64->classes, ReLU, dropout 0.5 after the two
        hidden layers, log_softmax; Adam lr 0.003, wd 5e-4, 500 epochs, NLL loss).  Coarse edge
        weights are passed to GCNConv; the original graph is run unweighted (as in UGC.py).
  sgc = LINKX repo (CUAI/Non-Homophily-Large-Scale) models.SGC: PyG SGConv(in, classes, K=hops,
        cached=True) -- edge weights NOT used (LINKX calls conv(x, edge_index)); AdamW lr 0.01,
        weight_decay 1e-3 (their sgc_exp.sh sweeps 1e-3/1e-2/1e-1, hops is a script argument,
        default 1), 500 epochs, NLL on log_softmax.
  Both: after every epoch the model is scored on the ORIGINAL graph; test accuracy is taken at the
  epoch with the best validation accuracy (strictly greater, first occurrence), as in UGC.py / LINKX.
Metrics  : HE, RcE, REE of every coarsening, exactly as in the AH-UGC appendix / authors' spectral_properties.py
           (L_lift = C L_c C^T with binary C; HE uses raw X; RcE = log(||L-L_lift||_F^2 / N); REE over the top
           k=100 Laplacian eigenvalues, k=N/2 if N<100, capped by the coarse size). Computed exactly with sparse
           algebra on the CPU; the baseline row reports 0 for the three (nothing was coarsened). --no_metrics skips them.
Device   : --device auto (cuda if available) | cuda | cpu. Training and evaluation run on that device.
Baseline : every model/input is also trained on the UNCOARSENED graph (same split, same selection rule),
           reported as "none" with ratio 1.0.
Alpha    : --alpha_mode computed  -> 1 - edge homophily (fraction of edges joining different labels;
                                     reproduces the values in the paper's Table 5/6, e.g. Cora 0.19)
                         homophily -> edge homophily itself (the literal formula printed in Algorithm 1)
                         paper     -> the paper's tabulated value
Splits   : random 60/20/20 per seed (UGC.py "split way 2"); same split for every config of a seed.
--model auto = sgc for cornell/film (paper Table 9), gcn for cora/citeseer/pubmed (Tables 3/8).
--clf_input F : classifier consumes the same F that was hashed (what the UGC code does)
              X : classifier consumes node features only
Requires: torch, torch_geometric, scipy, numpy.

  python ahugc_eval.py                                                            # cornell film citeseer cora; ratios 0.5 0.3 0.1; 10 seeds
  python ahugc_eval.py --datasets cora --model both --clf_input X --ratios 0.5 --seeds 3
  python ahugc_eval.py --datasets cora --alpha_mode homophily
  python ahugc_eval_shingle.py --datasets cora citeseer --hash random pca_z dpca shingle shingle_z --F feat
  python ahugc_eval_shingle.py --datasets cornell pubmed --paper --seeds 10 --out results_paper.csv   # AH-UGC paper setup
"""
import argparse, csv, os, random, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, svds, eigsh
import torch
import torch.nn.functional as Fn
from torch.nn import Linear  # noqa: F401  (kept to mirror the authors' imports)
from torch_geometric.nn import GCNConv, SGConv
from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork

PAPER_ALPHA = {"cora": 0.19, "citeseer": 0.26, "pubmed": 0.20, "cornell": 0.70, "film": 0.78,
               "chameleon": 0.75, "squirrel": 0.78}
DEFAULT_MODEL = {"cornell": "sgc", "film": "sgc", "chameleon": "sgc", "squirrel": "sgc",
                 "cora": "gcn", "citeseer": "gcn", "pubmed": "gcn"}
L_PROJ = 500            # number of random projectors (paper: "l", value not stated)


# ------------------------------------------------------------------ data
def load(name, root="data"):
    if name in ("cora", "citeseer", "pubmed"):
        ds = Planetoid(f"{root}/{name}", name.capitalize() if name != "pubmed" else "PubMed")
    elif name == "cornell":
        ds = WebKB(f"{root}/webkb", "Cornell")
    elif name == "film":
        ds = Actor(f"{root}/actor")
    elif name in ("chameleon", "squirrel"):
        ds = WikipediaNetwork(f"{root}/wikipedia", name, geom_gcn_preprocess=True)
    else:
        raise ValueError(name)
    d = ds[0]
    N = d.num_nodes
    ei = d.edge_index.numpy()
    A = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(N, N))
    A = ((A + A.T) > 0).astype(np.float64).tolil()
    A.setdiag(0)
    A = A.tocsr(); A.eliminate_zeros()
    X = sp.csr_matrix(d.x.numpy().astype(np.float64))
    return A, X, d.y.numpy(), int(d.y.max()) + 1


def heterophily_alpha(A, y):
    r, c = A.nonzero()
    return 1.0 - (y[r] == y[c]).mean()


def alpha_value(mode, name, A, y):
    het = heterophily_alpha(A, y)
    return {"computed": het, "homophily": 1.0 - het, "paper": PAPER_ALPHA[name]}[mode]


def build_F(kind, A, X, alpha):
    if kind == "feat":
        return X
    deg = np.asarray(A.sum(1)).ravel()
    if kind == "orig":
        second = A
    elif kind == "neigh":
        second = sp.diags(1 / np.maximum(deg, 1)) @ A @ X
    else:
        raise ValueError(kind)
    return sp.hstack([(1 - alpha) * X, alpha * second]).tocsr()


# ------------------------------------------------------------------ hashing + coarsening
def morton(P, bits=10):
    q = np.floor((P - P.min(0)) / (np.ptp(P, 0) + 1e-12) * (2 ** bits - 1)).astype(np.int64)
    key = np.zeros(len(P), dtype=np.int64)
    for b in range(bits):
        for c in range(P.shape[1]):
            key |= ((q[:, c] >> b) & 1) << (b * P.shape[1] + c)
    return key


def top_pcs(F, k=5):
    """top-k principal scores of centred F without densifying F."""
    mu = np.asarray(F.mean(0)).ravel()
    op = LinearOperator(F.shape, dtype=np.float64,
                        matvec=lambda v: F @ v - mu @ v,
                        rmatvec=lambda u: F.T @ u - mu * np.sum(u),
                        matmat=lambda V: F @ V - (mu @ V)[None, :],
                        rmatmat=lambda U: F.T @ U - np.outer(mu, U.sum(0)))
    _, S, Vt = svds(op, k=k, random_state=0)
    o = np.argsort(-S)
    V = Vt[o].T
    return F @ V - (mu @ V)[None, :]


def hash_scores(F, mode, rng):
    if mode == "random":                       # AH-UGC Algorithm 1, lines 3-6
        W = rng.normal(size=(F.shape[1], L_PROJ))
        b = rng.uniform(0, 1, size=L_PROJ)
        return np.asarray(F @ W + b).mean(1)
    if mode == "pca_z":
        return morton(top_pcs(F)[:, :3]).astype(np.float64)
    raise ValueError(mode)


GRAPH_HASHES = ("shingle", "dpca", "shingle_z")      # graph-aware orders (use A and X, not F)


def a_hat(A):
    M = (sp.csr_matrix(A) + sp.eye(A.shape[0])).tocsr()
    d = np.asarray(M.sum(1)).ravel() ** -0.5
    return sp.diags(d) @ M @ sp.diags(d)


def minhash(A, rng, nh):
    """neighbourhood MinHash: sig_k(v) = min over the closed neighbourhood N[v] of a random rank r_k(u).
    Two nodes get the same value with probability = Jaccard overlap of their closed neighbourhoods.
    A new edge (u,v) only needs sig[u] = min(sig[u], r[v]) and sig[v] = min(sig[v], r[u])."""
    M = (sp.csr_matrix(A) + sp.eye(A.shape[0])).tocsr()          # closed neighbourhoods: no empty rows
    out = []
    for _ in range(nh):
        r = rng.permutation(A.shape[0]).astype(np.int64)         # random ranks
        out.append(np.minimum.reduceat(r[M.indices], M.indptr[:-1]))
    return out


def graph_hash_order(mode, A, X, rng):
    if mode == "shingle":
        s = minhash(A, rng, 3)
        return np.lexsort((s[2], s[1], s[0]))                    # sort by sig_1, ties by sig_2, then sig_3
    if mode == "dpca":
        Ah = a_hat(A)
        return np.argsort(morton(top_pcs(Ah @ (Ah @ X))[:, :3]), kind="stable")
    if mode == "shingle_z":
        return np.lexsort((morton(top_pcs(X)[:, :3]), minhash(A, rng, 1)[0]))
    raise ValueError(mode)


def coarsen(F, ratio, rng, mode, A=None, X=None):
    """sort by hash, then random right-neighbour merges until ratio*N super-nodes remain.
    random / pca_z hash the feature matrix F; shingle / dpca / shingle_z hash the graph A (and X)."""
    N = F.shape[0]
    if mode in GRAPH_HASHES:
        order = graph_hash_order(mode, A, X, rng)
    else:
        order = np.argsort(hash_scores(F, mode, rng), kind="stable")
    groups = [[int(i)] for i in order]
    target = int(round(ratio * N))
    while len(groups) > target:
        j = int(rng.integers(len(groups) - 1))
        groups[j] += groups.pop(j + 1)
    pi = np.empty(N, dtype=np.int64)
    for g, members in enumerate(groups):
        pi[members] = g
    return pi


def purity(pi, y, nc):
    c = np.zeros((pi.max() + 1, nc)); np.add.at(c, (pi, y), 1)
    return c.max(1).sum() / len(pi)


# ------------------------------------------------------------------ spectral metrics (as in the UGC/AH-UGC code)
def laplacian(A):
    A = sp.csr_matrix(A, dtype=np.float64)
    return (sp.diags(np.asarray(A.sum(1)).ravel()) - A).tocsr()


class SpectralRef:
    """Quantities that depend only on the ORIGINAL graph, computed once per dataset.
    Definitions follow AH-UGC Appendix F and the authors' spectral_properties.py:
      L      = D - A  (unnormalised),  L_c = Laplacian of the weighted coarse graph (self-loops dropped),
      L_lift = C L_c C^T  with the BINARY coarsening matrix C  ->  L_lift[i,j] = L_c[pi(i), pi(j)]
      HE  = arccosh( ||(L_lift-L)X||_F^2 ||X||_F^2 / (2 tr(X^T L_lift X) tr(X^T L X)) + 1 ),  X = raw node features
      RcE = log( ||L - L_lift||_F^2 / N )      (the code's version; the paper writes ||L-L_lift||_F^2)
      REE = mean_i |lam~_i - lam_i| / lam_i over the top-k Laplacian eigenvalues, k = 100 (N/2 if N < 100)
    All of it is computed exactly with sparse algebra (no N x N dense matrices)."""

    def __init__(self, A, X):
        N = A.shape[0]
        self.N = N
        self.L = laplacian(A)
        self.X = np.asarray(X.todense(), dtype=np.float64)
        self.LX = self.L @ self.X
        self.trXLX = float(np.sum(self.X * self.LX))
        self.fX2 = float(np.sum(self.X ** 2))
        self.L_sq = float(self.L.multiply(self.L).sum())
        self.k = N // 2 if N < 100 else 100
        self.eig = None                       # top eigenvalues of L, ascending, computed lazily

    def top_eig_L(self, k):
        if self.eig is None or len(self.eig) < k:
            if self.N <= 1200 or k >= self.N - 1:
                w = np.linalg.eigvalsh(self.L.toarray())
            else:
                w = eigsh(self.L, k=k, which="LM", tol=1e-3, return_eigenvectors=False)
            self.eig = np.sort(w)[-k:]
        return self.eig[-k:]


def top_eig(Lc, k):
    n = Lc.shape[0]
    if n <= 1200 or k >= n - 1:
        w = np.linalg.eigvalsh(Lc.toarray())
    else:
        w = eigsh(Lc, k=k, which="LM", tol=1e-3, return_eigenvectors=False)
    return np.sort(w)[-k:]


def spectral_metrics(ref, A, pi):
    N, n = len(pi), int(pi.max()) + 1
    C = sp.csr_matrix((np.ones(N), (np.arange(N), pi)), shape=(N, n))
    size = np.asarray(C.sum(0)).ravel()
    Ac = sp.lil_matrix(C.T @ sp.csr_matrix(A, dtype=np.float64) @ C)
    Ac.setdiag(0)
    Ac = Ac.tocsr(); Ac.eliminate_zeros()
    Lc = laplacian(Ac)
    # ---- HE
    CX = C.T @ ref.X                                   # n x d
    LcCX = Lc @ CX
    lift_X = LcCX[pi]                                  # (L_lift X) = C (L_c (C^T X))
    num = float(np.sum((lift_X - ref.LX) ** 2)) * ref.fX2
    den = 2.0 * float(np.sum(CX * LcCX)) * ref.trXLX
    he = float(np.arccosh(1.0 + num / den)) if den > 0 else float("nan")
    # ---- RcE  ||L - L_lift||_F^2 = ||L||^2 - 2<L,L_lift> + ||L_lift||^2
    Lcoo = ref.L.tocoo()
    inner = float(np.sum(Lcoo.data * np.asarray(Lc[pi[Lcoo.row], pi[Lcoo.col]]).ravel()))
    Lcc = Lc.tocoo()
    lift_sq = float(np.sum(size[Lcc.row] * size[Lcc.col] * Lcc.data ** 2))
    rce_raw = max(ref.L_sq - 2.0 * inner + lift_sq, 1e-12)
    rce = float(np.log(rce_raw / N))
    # ---- REE (top-k eigenvalues; k capped by the coarse graph size)
    k = min(ref.k, n)
    lam = ref.top_eig_L(k)
    lam_c = top_eig(Lc, k)
    ree = float(np.mean(np.abs(lam_c - lam) / lam))
    return he, rce, ree


# ------------------------------------------------------------------ models
class GCN_(torch.nn.Module):               # verbatim from UGC repo GCN.py
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 64)
        self.conv3 = GCNConv(64, num_classes)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = Fn.relu(x)
        x = Fn.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = Fn.relu(x)
        x = Fn.dropout(x, training=self.training)
        x = self.conv3(x, edge_index, edge_weight)
        return Fn.log_softmax(x, dim=1)


class SGC(torch.nn.Module):                # LINKX models.SGC; two cached convs (coarse / original) share one Linear
    def __init__(self, in_channels, out_channels, hops):
        super().__init__()
        self.conv_c = SGConv(in_channels, out_channels, hops, cached=True)
        self.conv_o = SGConv(in_channels, out_channels, hops, cached=True)
        self.conv_o.lin = self.conv_c.lin

    def reset_parameters(self):
        self.conv_c.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, original=False):
        return (self.conv_o if original else self.conv_c)(x, edge_index)   # LINKX ignores edge weights


# ------------------------------------------------------------------ training
def to_t(M):
    return torch.from_numpy(sp.csr_matrix(M).toarray()).float()


def edge_tensors(M, dev):
    M = sp.coo_matrix(M)
    return (torch.from_numpy(np.vstack((M.row, M.col))).long().to(dev),
            torch.from_numpy(M.data).float().to(dev))


def make_model(name, args, dev, d_in, nc, Xtr, ei_tr, ew_tr, Xo, ei_o, train_is_original):
    if name == "gcn":
        model = GCN_(d_in, args.hidden, nc).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=args.gcn_lr, weight_decay=args.gcn_wd)
        fwd_tr = lambda: model(Xtr, ei_tr, ew_tr)
        fwd_o = lambda: model(Xo, ei_o, None)             # original graph: unweighted (as in UGC.py)
    else:
        model = SGC(d_in, nc, args.hops).to(dev)
        model.reset_parameters()
        opt = torch.optim.AdamW(model.parameters(), lr=args.sgc_lr, weight_decay=args.sgc_wd)
        fwd_tr = lambda: model(Xtr, ei_tr, None, original=train_is_original)
        fwd_o = lambda: model(Xo, ei_o, None, original=True)
    return model, opt, fwd_tr, fwd_o


def fit(name, model, opt, fwd_tr, fwd_o, yt, mask, Y, val_t, test_t, epochs):
    """train on (yt, mask); every epoch score the ORIGINAL graph; test acc at best val (strict >)."""
    best_val, best_test = -1.0, 0.0
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        out = fwd_tr()
        if name == "sgc":
            out = Fn.log_softmax(out, dim=1)
        loss = Fn.nll_loss(out[mask], yt[mask])
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = fwd_o().argmax(1)
        v = (pred[val_t] == Y[val_t]).float().mean().item()
        if v > best_val:
            best_val = v
            best_test = (pred[test_t] == Y[test_t]).float().mean().item()
    return best_test


def original_input(cache, args, kind, Fin, dev):
    key = (args.clf_input, kind if args.clf_input == "F" else None)
    if key not in cache:
        cache.clear()
        cache[key] = to_t(Fin).to(dev)
    return cache[key]


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def run_base(name, args, dev, kind, seed, A, X, y, nc, alpha, split, cache):
    """uncoarsened baseline: train on the original graph with the train nodes."""
    train, val, test = split
    seed_all(seed)
    Fin = build_F(kind, A, X, alpha) if args.clf_input == "F" else X
    Xo = original_input(cache, args, kind, Fin, dev)
    ei_o, _ = edge_tensors(A, dev)
    Y = torch.from_numpy(y).long().to(dev)
    tr_t, val_t, test_t = (torch.from_numpy(m).to(dev) for m in (train, val, test))
    model, opt, fwd_tr, fwd_o = make_model(name, args, dev, Xo.shape[1], nc, Xo, ei_o, None, Xo, ei_o, True)
    return fit(name, model, opt, fwd_tr, fwd_o, Y, tr_t, Y, val_t, test_t, args.epochs)


def run_one(name, args, dev, kind, mode, ratio, seed, A, X, y, nc, alpha, split, cache, ref):
    train, val, test = split
    N = A.shape[0]
    seed_all(seed)
    rng = np.random.default_rng(seed)

    F = build_F(kind, A, X, alpha)
    pi = coarsen(F, ratio, rng, mode, A, X)
    n = int(pi.max()) + 1
    C = sp.csr_matrix((np.ones(N), (np.arange(N), pi)), shape=(N, n))
    size = np.asarray(C.sum(0)).ravel()

    Fin = F if args.clf_input == "F" else X
    Fc = sp.diags(1 / size) @ (C.T @ Fin)                    # mean of member rows
    Ac = (C.T @ A @ C)                                       # weighted, internal edges on the diagonal
    ei_c, ew_c = edge_tensors(Ac, dev)

    votes = np.zeros((n, nc)); tr = np.where(train)[0]
    np.add.at(votes, (pi[tr], y[tr]), 1)                     # TRAIN labels only
    yc = torch.from_numpy(votes.argmax(1)).long().to(dev)
    has_train = torch.from_numpy(votes.sum(1) > 0).to(dev)

    Xo = original_input(cache, args, kind, Fin, dev)
    ei_o, _ = edge_tensors(A, dev)
    Xc = to_t(Fc).to(dev)
    Y = torch.from_numpy(y).long().to(dev)
    val_t, test_t = torch.from_numpy(val).to(dev), torch.from_numpy(test).to(dev)

    model, opt, fwd_tr, fwd_o = make_model(name, args, dev, Xc.shape[1], nc, Xc, ei_c, ew_c, Xo, ei_o, False)
    acc = fit(name, model, opt, fwd_tr, fwd_o, yc, has_train, Y, val_t, test_t, args.epochs)
    if ref is not None:
        he, rce, r = spectral_metrics(ref, A, pi)
    else:
        he = rce = r = float("nan")
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return acc, he, rce, r, purity(pi, y, nc), n / N


def random_split(N, seed):
    perm = np.random.default_rng(1000 + seed).permutation(N)
    a, b = int(0.6 * N), int(0.8 * N)
    tr, va, te = (np.zeros(N, bool) for _ in range(3))
    tr[perm[:a]] = True; va[perm[a:b]] = True; te[perm[b:]] = True
    return tr, va, te


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["cornell", "film", "citeseer", "cora"])  # pubmed also supported
    p.add_argument("--model", default="auto", choices=["auto", "gcn", "sgc", "both"])
    p.add_argument("--paper", action="store_true",
                   help="AH-UGC paper setup: F=orig, hash=random, ratio 0.5, alpha=paper (unless overridden)")
    p.add_argument("--F", nargs="+", default=None, help="default: feat orig neigh (orig with --paper)")
    p.add_argument("--hash", nargs="+", default=None, help="default: random pca_z (random with --paper)")
    p.add_argument("--clf_input", default="F", choices=["F", "X"])
    p.add_argument("--ratios", nargs="+", type=float, default=None, help="supernodes / nodes (default 0.5 0.3 0.1; 0.5 with --paper)")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--no_baseline", action="store_true", help="skip the uncoarsened baseline")
    p.add_argument("--alpha_mode", default=None, choices=["computed", "homophily", "paper"],
                   help="default computed (paper with --paper)")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--gcn_lr", type=float, default=0.003); p.add_argument("--gcn_wd", type=float, default=0.0005)
    p.add_argument("--sgc_lr", type=float, default=0.01); p.add_argument("--sgc_wd", type=float, default=1e-3)
    p.add_argument("--hops", type=int, default=1, help="LINKX sgc_exp.sh default is 1; Wu et al. use 2")
    p.add_argument("--no_metrics", action="store_true", help="skip HE / RcE / REE")
    p.add_argument("--device", default="auto", help="auto | cuda | cuda:1 | cpu")
    p.add_argument("--out", default="results.csv")
    args = p.parse_args()
    if args.paper:
        args.F = args.F or ["orig"]
        args.hash = args.hash or ["random"]
        args.ratios = args.ratios or [0.5]
        args.alpha_mode = args.alpha_mode or "paper"
    args.F = args.F or ["feat", "orig", "neigh"]
    args.hash = args.hash or ["random", "pca_z"]
    args.ratios = args.ratios or [0.5, 0.3, 0.1]
    args.alpha_mode = args.alpha_mode or "computed"
    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but torch.cuda.is_available() is False (CPU-only torch build or no driver)")
    print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(dev)})" if dev.type == "cuda"
          else "   <-- training on CPU; install a CUDA build of torch to use the GPU"), flush=True)

    header = ["dataset", "model", "clf_input", "alpha_mode", "alpha", "F", "hash", "ratio", "seeds",
              "acc_mean", "acc_std", "he", "rce", "ree", "purity", "n_over_N", "per_seed_acc"]
    new = not os.path.exists(args.out)
    fout = open(args.out, "a", newline="")
    w = csv.writer(fout)
    if new:
        w.writerow(header)

    for name in args.datasets:
        A, X, y, nc = load(name)
        N = A.shape[0]
        het = heterophily_alpha(A, y)
        alpha = alpha_value(args.alpha_mode, name, A, y)
        models = ["gcn", "sgc"] if args.model == "both" else [DEFAULT_MODEL[name] if args.model == "auto" else args.model]
        print(f"\n=== {name}: N={N} edges={A.nnz//2} feat={X.shape[1]} classes={nc} | 1-homophily={het:.3f} "
              f"homophily={1-het:.3f} paper={PAPER_ALPHA[name]} | USING alpha={alpha:.3f} ({args.alpha_mode}) "
              f"| clf_input={args.clf_input} seeds={args.seeds}", flush=True)
        ref = None if args.no_metrics else SpectralRef(A, X)
        splits = [random_split(N, s) for s in range(args.seeds)]
        cache = {}
        for m in models:
            def report(label_f, label_h, ratio, accs, extra):
                accs = np.array(accs) * 100
                he_m, rce_m, ree_m, pur_m, nn_m = extra
                print(f"{'':<14}{label_f:<6}{label_h:<8}{accs.mean():6.2f}±{accs.std():4.2f}"
                      f"{he_m:8.3f}{rce_m:8.3f}{ree_m:8.3f}{100*pur_m:8.2f}{nn_m:7.3f}", flush=True)
                w.writerow([name, m, args.clf_input, args.alpha_mode, f"{alpha:.4f}", label_f, label_h, ratio,
                            args.seeds, f"{accs.mean():.3f}", f"{accs.std():.3f}", f"{he_m:.4f}", f"{rce_m:.4f}",
                            f"{ree_m:.4f}", f"{100*pur_m:.2f}", f"{nn_m:.4f}", ";".join(f"{a:.2f}" for a in accs)])
                fout.flush()

            if not args.no_baseline:
                print(f"[model = {m}]  uncoarsened baseline (ratio 1.0)   {'F':<6}{'hash':<8}{'acc %':>10}", flush=True)
                for kind in (args.F if args.clf_input == "F" else ["feat"]):
                    accs = [run_base(m, args, dev, kind, s, A, X, y, nc, alpha, splits[s], cache) for s in range(args.seeds)]
                    report(kind if args.clf_input == "F" else "X", "none", 1.0, accs, (0.0, 0.0, 0.0, 1.0, 1.0))
            for ratio in args.ratios:
                print(f"[model = {m}]  ratio {ratio}   {'F':<6}{'hash':<8}{'acc %':>10}{'HE':>8}{'RcE':>8}{'REE':>8}{'pur %':>8}{'n/N':>7}", flush=True)
                for kind in args.F:
                    for mode in args.hash:
                        res = np.array([run_one(m, args, dev, kind, mode, ratio, s, A, X, y, nc, alpha, splits[s], cache, ref)
                                        for s in range(args.seeds)])
                        report(kind, mode, ratio, res[:, 0], tuple(res[:, j].mean() for j in range(1, 6)))
    fout.close()
    print(f"\nresults appended to {args.out}")


if __name__ == "__main__":
    main()
