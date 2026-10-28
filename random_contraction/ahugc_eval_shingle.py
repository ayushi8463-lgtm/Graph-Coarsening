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
  appnp = paper Table 7 architecture (64-unit MLP then APPNP, K=10, alpha=0.1); dropout before
        MLP linear layers; Adam lr .003, wd .0005, 500 epochs.
  mixhop = CUAI Non-Homophily-Large-Scale implementation: 0/1/2-hop channels, batch norm,
        ReLU, output MixHop layer and final linear projection. AH-UGC points to [11] for
        heterophilic model designs; the upstream code searches hidden={8,16,32}, layers={2,3}.
        Here hidden=16/layers=2 by default; pass the other settings explicitly to reproduce a sweep.
  All models: after every epoch the model is scored on the ORIGINAL graph; test accuracy is taken at the
  epoch with the best validation accuracy (strictly greater, first occurrence), as in UGC.py / LINKX.
Metrics  : HE, RcE, REE of every coarsening, exactly as in the AH-UGC appendix / authors' spectral_properties.py
           (L_lift = C L_c C^T with binary C; HE uses raw X; RcE = log(||L-L_lift||_F^2 / N); REE over the top
           k=100 Laplacian eigenvalues, k=N/2 if N<100, capped by the coarse size). Computed exactly with sparse
           algebra on the CPU; the baseline row reports 0 for the three (nothing was coarsened). --no_metrics skips them.
           --spectral_only skips classifier training and tunes one configuration per hash/ratio for each metric,
           selecting by the mean metric over seeds. Tune random hash with --alpha_values/--projectors and shingle
           orders with --shingle_hashes; ratio stays fixed within each selection.
Device   : --device auto (cuda if available) | cuda | cpu. Training and evaluation run on that device.
Baseline : every model/input is also trained on the UNCOARSENED graph (same split, same selection rule),
           reported as "none" with ratio 1.0.
Alpha    : --alpha_mode computed  -> 1 - edge homophily (fraction of edges joining different labels;
                                     reproduces the values in the paper's Table 5/6, e.g. Cora 0.19)
                         homophily -> edge homophily itself (the literal formula printed in Algorithm 1)
                         paper     -> the paper's tabulated value
Splits   : Geom-GCN fixed 60/20/20 splits 0..9 (default, --split geom) or random 60/20/20 per seed (--split random).
Graphs   : --source authors (default) uses the authors' heterophlic_data/*.pt and film.mat with their storage
           convention; Cornell comes from the Geom-GCN source (its .pt is not in the repo). --source geom = old setting.
--model auto = sgc for cornell/film (paper Table 9), gcn for cora/citeseer/pubmed (Tables 3/8).
--clf_input F : classifier consumes the same F that was hashed (what the UGC code does)
              X : classifier consumes node features only
Requires: torch, torch_geometric, scipy, numpy.

  python ahugc_eval.py                                                            # cornell film citeseer cora; ratios 0.5 0.3 0.1; 10 seeds
  python ahugc_eval.py --datasets cora --model both --clf_input X --ratios 0.5 --seeds 3
  python ahugc_eval.py --datasets cora --alpha_mode homophily
  python ahugc_eval_shingle.py --datasets cora citeseer --hash random pca_z dpca shingle shingle_z --F feat
  python ahugc_eval_shingle.py --datasets cora citeseer pubmed physics dblp cornell texas film chameleon squirrel penn94 --model all
  python ahugc_eval_shingle.py --datasets cornell pubmed --paper --seeds 10 --out results_paper.csv   # AH-UGC paper setup
  python ahugc_eval_v2.py --datasets imdb dblp acm --hetero_models HeteroGCN2 --hetero_ratios 0.3
"""
import argparse, csv, os, random, time, urllib.request
import numpy as np
import scipy.io
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, svds, eigsh
import torch
import torch.nn.functional as Fn
from torch.nn import Linear  # noqa: F401  (kept to mirror the authors' imports)
from torch_geometric.nn import APPNP as APPNPProp, GCNConv, SGConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork

PAPER_ALPHA = {"cora": 0.19, "citeseer": 0.26, "pubmed": 0.20, "cornell": 0.70, "film": 0.78,
               "chameleon": 0.75, "squirrel": 0.78, "texas": 0.91, "physics": 0.07, "dblp": 0.18,
               "penn94": 0.53, "amherst41": float("nan"), "reed98": float("nan")}
GEOM_SPLIT = {"cora", "citeseer", "pubmed", "cornell", "film", "chameleon", "squirrel", "texas"}  # fixed splits exist
DEFAULT_MODEL = {"cornell": "sgc", "film": "sgc", "chameleon": "sgc", "squirrel": "sgc", "texas": "sgc",
                 "penn94": "sgc", "physics": "gcn", "dblp": "gcn",
                 "amherst41": "sgc", "reed98": "sgc",
                 "cora": "gcn", "citeseer": "gcn", "pubmed": "gcn"}
GG_PIN = "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/f1fc0d14b3b019c562737240d06ec83b07d16a8f"
UGC_RAW = "https://raw.githubusercontent.com/katariaMohit/UGC-Universal-Graph-Coarsening/main/heterophlic_data"
# sources PyG's Coauthor('Physics') and CitationFull('DBLP') download, and LINKX's Penn94 (facebook100)
PHYSICS_URL = "https://raw.githubusercontent.com/shchur/gnn-benchmark/master/data/npz/ms_academic_phy.npz"
DBLP_URL = "https://github.com/abojchevski/graph2gauss/raw/master/data/dblp.npz"
PENN94_URL = "https://raw.githubusercontent.com/CUAI/Non-Homophily-Large-Scale/master/data/facebook100/Penn94.mat"
# LINKX's official Facebook100 files. The LINKX/PyG preprocessing uses the raw
# adjacency and one-hot encodes metadata columns 0 and 2..end; column 1 is gender.
FB100_URL = "https://raw.githubusercontent.com/CUAI/Non-Homophily-Large-Scale/master/data/facebook100/{name}.mat"
# files the authors load for chameleon/squirrel (note the authors' 'cameleon' typo in the feature filename)
AUTHOR_PT = {"chameleon": ("edge_index_chameleon.pt", "label_chameleon.pt", "node_feat_cameleon.pt"),
             "squirrel": ("edge_index_squirrel.pt", "label_squirrel.pt", "node_feat_squirrel.pt")}
L_PROJ = 500            # number of random projectors (paper: "l", value not stated)


# ------------------------------------------------------------------ data
def fetch(url, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def _np(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def row_normalize(X):
    """Authors' heterophilic features are row-normalised (each row sums to 1); zero rows stay zero."""
    X = sp.csr_matrix(X, dtype=np.float64)
    s = np.asarray(X.sum(1)).ravel()
    return (sp.diags(1.0 / np.maximum(s, 1e-12)) @ X).tocsr()


def authors_graph(ei, X, y, normalize=True):
    """Authors' storage convention (verified against their .pt/.mat files): edges kept as stored
    (asymmetric, duplicates collapsed), existing self-loops dropped for the Laplacian, and one
    self-loop added per node in A (used by F = [(1-a)X | aA] and by the coarse graph).
    Returns A (with self-loops, for F/coarsening), A_lap (no self-loops, for L and alpha), X, y."""
    N = X.shape[0]
    s, t = ei[0], ei[1]
    keep = s != t
    A_lap = sp.csr_matrix((np.ones(int(keep.sum())), (s[keep], t[keep])), shape=(N, N))
    A_lap.data[:] = 1.0
    A_lap.eliminate_zeros()
    A = (A_lap + sp.eye(N)).tocsr()
    A.data[:] = 1.0
    return A, A_lap, (row_normalize(X) if normalize else sp.csr_matrix(X)), y


def symmetric_graph(ei, N):
    A = sp.csr_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(N, N))
    A = ((A + A.T) > 0).astype(np.float64).tolil()
    A.setdiag(0)
    A = A.tocsr(); A.eliminate_zeros()
    return A


def load(name, source="authors", root="data"):
    """Returns (A, A_lap, X, y, nc).
    source='authors': the graphs the AH-UGC/UGC authors load (heterophlic_data/*.pt, film.mat; Cornell from
                      the Geom-GCN source since its .pt file is not in the repo).
    source='geom'   : PyG loaders on the Geom-GCN source, symmetrised (our earlier setting)."""
    if name in ("cora", "citeseer", "pubmed"):
        ds = Planetoid(f"{root}/{name}", name.capitalize() if name != "pubmed" else "PubMed")
        d = ds[0]
        A = symmetric_graph(d.edge_index.numpy(), d.num_nodes)
        return A, A, sp.csr_matrix(d.x.numpy().astype(np.float64)), d.y.numpy(), int(d.y.max()) + 1

    if source == "authors" and name == "film":
        m = scipy.io.loadmat(fetch(f"{UGC_RAW}/film.mat", f"{root}/authors/film.mat"))
        ei = m["edge_index"].astype(np.int64)
        X = m["node_feat"].astype(np.float64)
        y = m["label"].ravel().astype(np.int64)
        A, A_lap, X, y = authors_graph(ei, X, y)
        return A, A_lap, X, y, int(y.max()) + 1

    if source == "authors" and name in AUTHOR_PT:
        e, l, f = AUTHOR_PT[name]
        ei = _np(torch.load(fetch(f"{UGC_RAW}/{e}", f"{root}/authors/{e}"), weights_only=False)).astype(np.int64)
        y = _np(torch.load(fetch(f"{UGC_RAW}/{l}", f"{root}/authors/{l}"), weights_only=False)).astype(np.int64).ravel()
        X = _np(torch.load(fetch(f"{UGC_RAW}/{f}", f"{root}/authors/{f}"), weights_only=False)).astype(np.float64)
        A, A_lap, X, y = authors_graph(ei, X, y)
        return A, A_lap, X, y, int(y.max()) + 1

    if name in ("physics", "dblp"):
        # same raw files PyG's Coauthor('Physics') / CitationFull('DBLP') download; graph is symmetric
        url = PHYSICS_URL if name == "physics" else DBLP_URL
        z = np.load(fetch(url, f"{root}/raw/{name}.npz"), allow_pickle=True)
        Aq = sp.csr_matrix((z["adj_data"], z["adj_indices"], z["adj_indptr"]), shape=tuple(z["adj_shape"]))
        Xq = sp.csr_matrix((z["attr_data"], z["attr_indices"], z["attr_indptr"]), shape=tuple(z["attr_shape"]))
        y = z["labels"].astype(np.int64)
        A = symmetric_graph(np.vstack(Aq.nonzero()), Aq.shape[0])
        return A, A, Xq.astype(np.float64), y, int(y.max()) + 1

    if name == "penn94":
        # LINKX fb100 loader (dataset.py load_fb100_dataset): label = gender-1 (-1 = unlabeled);
        # features = one-hot of metadata columns 0 and 2..6, one column per distinct value.
        m = scipy.io.loadmat(fetch(PENN94_URL, f"{root}/linkx/Penn94.mat"))
        A = sp.csr_matrix(m["A"]).astype(np.float64)
        A = ((A + A.T) > 0).astype(np.float64).tolil(); A.setdiag(0); A = A.tocsr(); A.eliminate_zeros()
        meta = m["local_info"].astype(np.int64)
        y = meta[:, 1] - 1
        cols = [meta[:, 0]] + [meta[:, j] for j in range(2, meta.shape[1])]
        blocks = []
        for v in cols:
            u, inv = np.unique(v, return_inverse=True)
            blocks.append(sp.csr_matrix((np.ones(len(v)), (np.arange(len(v)), inv)), shape=(len(v), len(u))))
        X = sp.hstack(blocks).tocsr()
        return A, A, X, y, int(y.max()) + 1

    if name in ("amherst41", "reed98"):
        # Exact LINKX/Facebook100 representation (Lim et al., NeurIPS 2021):
        # keep the supplied adjacency as stored; y = gender - 1 (-1 is unlabeled);
        # one-hot encode student/faculty/status/major/second-major/dorm/year columns.
        # Do not row-normalize these categorical features.
        key = "Amherst41" if name == "amherst41" else "Reed98"
        m = scipy.io.loadmat(fetch(FB100_URL.format(name=key), f"{root}/linkx/{key}.mat"))
        A = sp.csr_matrix(m["A"], dtype=np.float64)
        A.sum_duplicates(); A.data[:] = 1.0; A.eliminate_zeros()
        meta = m["local_info"].astype(np.int64)
        y = meta[:, 1] - 1
        cols = [meta[:, 0]] + [meta[:, j] for j in range(2, meta.shape[1])]
        blocks = []
        for values in cols:
            _, inv = np.unique(values, return_inverse=True)
            blocks.append(sp.csr_matrix((np.ones(len(values)), (np.arange(len(values)), inv)),
                                        shape=(len(values), int(inv.max()) + 1)))
        X = sp.hstack(blocks, format="csr", dtype=np.float64)
        return A, A, X, y, int(y.max()) + 1

    # Geom-GCN source. Cornell and Film from the pinned commit directly; chameleon/squirrel through PyG
    # (PyG's WikipediaNetwork already uses the pinned commit).
    if name in ("cornell", "film", "texas"):
        ei, X, y = load_geom_pinned(name, root)
    elif name in ("chameleon", "squirrel"):
        d = WikipediaNetwork(f"{root}/wikipedia", name, geom_gcn_preprocess=True)[0]
        ei, X, y = d.edge_index.numpy(), d.x.numpy().astype(np.float64), d.y.numpy()
    else:
        raise ValueError(name)
    if source == "authors":
        A, A_lap, X, y = authors_graph(ei, X, y)
    else:
        A = symmetric_graph(ei, X.shape[0])
        A_lap = A
        X = sp.csr_matrix(X)
    return A, A_lap, X, y, int(y.max()) + 1


def load_geom_pinned(name, root="data"):
    """Cornell / Film read directly from the pinned Geom-GCN commit (f1fc0d14, the files whose heterophily
    factor and edge counts match the paper). PyG's WebKB and Actor loaders read the 'master' branch, which
    has different Cornell labels/edges (computed alpha 0.877 instead of the paper's 0.70)."""
    base = f"{root}/geomgcn_pinned/{name}"
    nf = fetch(f"{GG_PIN}/new_data/{name}/out1_node_feature_label.txt", f"{base}/nodes.txt")
    ef = fetch(f"{GG_PIN}/new_data/{name}/out1_graph_edges.txt", f"{base}/edges.txt")
    rows = open(nf).read().split("\n")[1:-1]
    N = len(rows)
    y = np.zeros(N, dtype=np.int64)
    if name == "film":                                   # sparse index lists, Actor layout
        dim = 0; idx_rows = [None] * N
        for l in rows:
            nid, feat, lab = l.split("\t"); nid = int(nid)
            idx = [int(v) for v in feat.split(",")] if feat else []
            idx_rows[nid] = idx; y[nid] = int(lab)
            if idx: dim = max(dim, max(idx) + 1)
        X = np.zeros((N, dim), dtype=np.float64)
        for i, idx in enumerate(idx_rows):
            X[i, idx] = 1.0
    else:                                                # dense vectors, WebKB layout
        X = None
        for l in rows:
            nid, feat, lab = l.split("\t"); nid = int(nid)
            vals = np.array([float(v) for v in feat.split(",")])
            if X is None: X = np.zeros((N, len(vals)), dtype=np.float64)
            X[nid] = vals; y[nid] = int(lab)
    e = np.array([[int(x) for x in l.split("\t")] for l in open(ef).read().split("\n")[1:-1]], dtype=np.int64)
    return e.T, X, y


def load_geom_split(name, s, N, root="data"):
    """Geom-GCN fixed 60/20/20 split s (0..9) from the cited source repo (same files PyG downloads)."""
    f = fetch(f"{GG_PIN}/splits/{name}_split_0.6_0.2_{s}.npz", f"{root}/geomgcn_splits/{name}_split_0.6_0.2_{s}.npz")
    z = np.load(f)
    tr, va, te = (z[k].astype(bool) for k in ("train_mask", "val_mask", "test_mask"))
    assert len(tr) == N, (name, len(tr), N)
    return tr, va, te


def heterophily_alpha(A, y):
    """alpha = 1 - edge homophily over all edges. Unlabelled nodes (label -1, Penn94) are treated as one
    class, which reproduces LINKX Table 2 (Penn94 edge homophily 0.470 -> alpha 0.53, as AH-UGC reports)."""
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


def hash_scores(F, mode, rng, projectors=L_PROJ):
    if mode == "random":                       # AH-UGC Algorithm 1, lines 3-6
        W = rng.normal(size=(F.shape[1], projectors))
        b = rng.uniform(0, 1, size=projectors)
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


def graph_hash_order(mode, A, X, rng, shingle_hashes=3):
    if mode == "shingle":
        s = minhash(A, rng, shingle_hashes)
        return np.lexsort(tuple(reversed(s)))
    if mode == "dpca":
        Ah = a_hat(A)
        return np.argsort(morton(top_pcs(Ah @ (Ah @ X))[:, :3]), kind="stable")
    if mode == "shingle_z":
        s = minhash(A, rng, shingle_hashes)
        return np.lexsort((morton(top_pcs(X)[:, :3]), *reversed(s)))
    raise ValueError(mode)


def coarsen(F, ratio, rng, mode, A=None, X=None, projectors=L_PROJ, shingle_hashes=3):
    """Sort by hash, then randomly merge each selected super-node with its clockwise neighbour.
    random / pca_z hash the feature matrix F; shingle / dpca / shingle_z hash the graph A (and X)."""
    N = F.shape[0]
    if mode in GRAPH_HASHES:
        order = graph_hash_order(mode, A, X, rng, shingle_hashes)
    else:
        order = np.argsort(hash_scores(F, mode, rng, projectors), kind="stable")
    groups = [[int(i)] for i in order]
    target = int(round(ratio * N))
    while len(groups) > target:
        j = int(rng.integers(len(groups)))
        nxt = (j + 1) % len(groups)
        groups[j] += groups[nxt]
        del groups[nxt]
    pi = np.empty(N, dtype=np.int64)
    for g, members in enumerate(groups):
        pi[members] = g
    return pi


def purity(pi, y, nc):
    ok = y >= 0
    c = np.zeros((pi.max() + 1, nc)); np.add.at(c, (pi[ok], y[ok]), 1)
    return c.max(1).sum() / ok.sum()


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


def spectral_metrics(ref, A, pi, which=("he", "rce", "ree")):
    """Compute only the requested metrics on the partition pi. Returns (he, rce, ree); unrequested = nan."""
    which = set(which)
    N, n = len(pi), int(pi.max()) + 1
    C = sp.csr_matrix((np.ones(N), (np.arange(N), pi)), shape=(N, n))
    size = np.asarray(C.sum(0)).ravel()
    Ac = sp.lil_matrix(C.T @ sp.csr_matrix(A, dtype=np.float64) @ C)
    Ac.setdiag(0)
    Ac = Ac.tocsr(); Ac.eliminate_zeros()
    Lc = laplacian(Ac)
    he = rce = ree = float("nan")
    # ---- HE
    if "he" in which:
        CX = C.T @ ref.X                                   # n x d
        LcCX = Lc @ CX
        lift_X = LcCX[pi]                                  # (L_lift X) = C (L_c (C^T X))
        num = float(np.sum((lift_X - ref.LX) ** 2)) * ref.fX2
        den = 2.0 * float(np.sum(CX * LcCX)) * ref.trXLX
        he = float(np.arccosh(1.0 + num / den)) if den > 0 else float("nan")
    # ---- RcE  ||L - L_lift||_F^2 = ||L||^2 - 2<L,L_lift> + ||L_lift||^2
    if "rce" in which:
        Lcoo = ref.L.tocoo()
        inner = float(np.sum(Lcoo.data * np.asarray(Lc[pi[Lcoo.row], pi[Lcoo.col]]).ravel()))
        Lcc = Lc.tocoo()
        lift_sq = float(np.sum(size[Lcc.row] * size[Lcc.col] * Lcc.data ** 2))
        rce_raw = max(ref.L_sq - 2.0 * inner + lift_sq, 1e-12)
        rce = float(np.log(rce_raw / N))
    # ---- REE (top-k eigenvalues; k capped by the coarse graph size)
    if "ree" in which:
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


class APPNPNet(torch.nn.Module):
    """Two-layer MLP followed by personalized PageRank propagation."""
    def __init__(self, in_channels, hidden_channels, out_channels, k=10, alpha=0.1, dropout=0.5):
        super().__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.prop = APPNPProp(K=k, alpha=alpha, dropout=0.0, cached=False)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = Fn.dropout(x, p=self.dropout, training=self.training)
        x = Fn.relu(self.lin1(x))
        x = Fn.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return self.prop(x, edge_index, edge_weight)


def mixhop_adjacency(edge_index, edge_weight, num_nodes, dtype):
    """CUAI MixHop normalization: GCN-normalized A+I, transposed for sparse matmul."""
    edge_index, edge_weight = gcn_norm(
        edge_index, edge_weight, num_nodes, improved=False, add_self_loops=True,
        flow="source_to_target", dtype=dtype,
    )
    # PyG edge_index is source->target; the reference implementation stores A^T.
    return torch.sparse_coo_tensor(
        edge_index.flip(0), edge_weight, (num_nodes, num_nodes), device=edge_index.device
    ).coalesce()


class MixHopLayer(torch.nn.Module):
    """Adapted from CUAI/Non-Homophily-Large-Scale/models.py MixHopLayer."""
    def __init__(self, in_channels, out_channels, hops=2):
        super().__init__()
        self.hops = hops
        self.lins = torch.nn.ModuleList([Linear(in_channels, out_channels) for _ in range(hops + 1)])

    def forward(self, x, adj):
        channels = [self.lins[0](x)]
        for hop in range(1, self.hops + 1):
            x_hop = self.lins[hop](x)
            for _ in range(hop):
                x_hop = torch.sparse.mm(adj, x_hop)
            channels.append(x_hop)
        return torch.cat(channels, dim=-1)


class MixHopNet(torch.nn.Module):
    """Adapted from CUAI/Non-Homophily-Large-Scale/models.py MixHop."""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, num_layers=2, hops=2):
        super().__init__()
        self.convs = torch.nn.ModuleList([MixHopLayer(in_channels, hidden_channels, hops)])
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_channels * (hops + 1))])
        for _ in range(num_layers - 2):
            self.convs.append(MixHopLayer(hidden_channels * (hops + 1), hidden_channels, hops))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels * (hops + 1)))
        self.convs.append(MixHopLayer(hidden_channels * (hops + 1), out_channels, hops))
        self.final_project = Linear(out_channels * (hops + 1), out_channels)
        self.dropout = dropout

    def forward(self, x, adj):
        for i, conv in enumerate(self.convs[:-1]):
            x = Fn.relu(self.bns[i](conv(x, adj)))
            x = Fn.dropout(x, p=self.dropout, training=self.training)
        return self.final_project(self.convs[-1](x, adj))


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
    elif name == "sgc":
        model = SGC(d_in, nc, args.hops).to(dev)
        model.reset_parameters()
        opt = torch.optim.AdamW(model.parameters(), lr=args.sgc_lr, weight_decay=args.sgc_wd)
        fwd_tr = lambda: model(Xtr, ei_tr, None, original=train_is_original)
        fwd_o = lambda: model(Xo, ei_o, None, original=True)
    elif name == "appnp":
        model = APPNPNet(d_in, args.appnp_hidden, nc, args.appnp_k, args.appnp_alpha, args.dropout).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=args.appnp_lr, weight_decay=args.appnp_wd)
        fwd_tr = lambda: model(Xtr, ei_tr, None)
        fwd_o = lambda: model(Xo, ei_o, None)
    else:
        model = MixHopNet(d_in, args.mixhop_hidden, nc, args.dropout, args.mixhop_layers, args.mixhop_hops).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=args.mixhop_lr, weight_decay=args.mixhop_wd)
        # The cited CUAI implementation normalizes an unweighted edge_index.
        adj_tr = mixhop_adjacency(ei_tr, None, Xtr.shape[0], Xtr.dtype)
        adj_o = mixhop_adjacency(ei_o, None, Xo.shape[0], Xo.dtype)
        fwd_tr = lambda: model(Xtr, adj_tr)
        fwd_o = lambda: model(Xo, adj_o)
    return model, opt, fwd_tr, fwd_o


def fit(name, model, opt, fwd_tr, fwd_o, yt, mask, Y, val_t, test_t, epochs):
    """train on (yt, mask); every epoch score the ORIGINAL graph; test acc at best val (strict >)."""
    best_val, best_test = -1.0, 0.0
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        out = fwd_tr()
        if name != "gcn":
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
    return best_val, best_test


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
    _, test_acc = fit(name, model, opt, fwd_tr, fwd_o, Y, tr_t, Y, val_t, test_t, args.epochs)
    return test_acc


def run_one(name, args, dev, kind, mode, ratio, seed, A, X, y, nc, alpha, split, cache, ref,
            projectors=L_PROJ, shingle_hashes=3):
    train, val, test = split if split is not None else (None, None, None)
    N = A.shape[0]
    seed_all(seed)
    rng = np.random.default_rng(seed)

    F = build_F(kind, A, X, alpha)
    pi = coarsen(F, ratio, rng, mode, A, X, projectors, shingle_hashes)
    n = int(pi.max()) + 1
    C = sp.csr_matrix((np.ones(N), (np.arange(N), pi)), shape=(N, n))
    size = np.asarray(C.sum(0)).ravel()

    if args.spectral_only:
        he, rce, r = spectral_metrics(ref, A, pi, args.metrics)
        return float("nan"), float("nan"), he, rce, r, purity(pi, y, nc), n / N

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
    val_acc, test_acc = fit(name, model, opt, fwd_tr, fwd_o, yc, has_train, Y, val_t, test_t, args.epochs)
    if ref is not None:
        seed_all(seed)                                       # eigsh start vectors reproducible per seed
        he, rce, r = spectral_metrics(ref, A, pi, args.metrics)
    else:
        he = rce = r = float("nan")
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return val_acc, test_acc, he, rce, r, purity(pi, y, nc), n / N


def random_split_labeled(y, seed, train=0.6, val=0.2):
    """Random split over labelled nodes (label >= 0). Default 60/20/20 = authors' UGC.py 'split way 2';
    Penn94 uses 50/25/25 = LINKX dataset.py get_idx_split defaults (train_prop .5, valid_prop .25)."""
    N = len(y)
    idx = np.where(y >= 0)[0]
    perm = np.random.default_rng(1000 + seed).permutation(idx)
    a, b = int(train * len(idx)), int((train + val) * len(idx))
    tr, va, te = (np.zeros(N, bool) for _ in range(3))
    tr[perm[:a]] = True; va[perm[a:b]] = True; te[perm[b:]] = True
    return tr, va, te


def random_split(N, seed):
    perm = np.random.default_rng(1000 + seed).permutation(N)
    a, b = int(0.6 * N), int(0.8 * N)
    tr, va, te = (np.zeros(N, bool) for _ in range(3))
    tr[perm[:a]] = True; va[perm[a:b]] = True; te[perm[b:]] = True
    return tr, va, te


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["cornell", "film", "citeseer", "cora"])  # pubmed also supported
    p.add_argument("--model", nargs="+", default=["auto"],
                   choices=["auto", "gcn", "sgc", "appnp", "mixhop", "both", "all"],
                   help="one or more models; auto/both/all must be used alone")
    p.add_argument("--paper", action="store_true",
                   help="AH-UGC paper setup: F=orig, hash=random, ratio 0.5, alpha=paper (unless overridden)")
    p.add_argument("--F", nargs="+", default=None, help="default: feat orig neigh (orig with --paper)")
    p.add_argument("--hash", nargs="+", default=None, help="default: random pca_z (random with --paper)")
    p.add_argument("--clf_input", default="F", choices=["F", "X"])
    p.add_argument("--ratios", nargs="+", type=float, default=None, help="supernodes / nodes (default 0.5 0.3 0.1; 0.5 with --paper)")
    p.add_argument("--projectors", nargs="+", type=int, default=[L_PROJ],
                   help=f"random-hash projector counts to tune (default {L_PROJ}; ignored by graph-only hashes)")
    p.add_argument("--alpha_values", nargs="+", type=float, default=None,
                   help="feature/adjacency mixing weights to tune; defaults to the selected alpha_mode value")
    p.add_argument("--shingle_hashes", nargs="+", type=int, default=[3],
                   help="MinHash signature counts to tune for shingle/shingle_z (default 3)")
    p.add_argument("--spectral_only", action="store_true",
                   help="search coarsenings using HE/RcE/REE without training node-classification models")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--no_baseline", action="store_true", help="skip the uncoarsened baseline")
    p.add_argument("--alpha_mode", default=None, choices=["computed", "homophily", "paper"],
                   help="default computed (paper with --paper)")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--gcn_lr", type=float, default=0.003); p.add_argument("--gcn_wd", type=float, default=0.0005)
    p.add_argument("--sgc_lr", type=float, default=0.01); p.add_argument("--sgc_wd", type=float, default=1e-3)
    p.add_argument("--appnp_hidden", type=int, default=64)
    p.add_argument("--appnp_k", type=int, default=10); p.add_argument("--appnp_alpha", type=float, default=0.1)
    p.add_argument("--appnp_lr", type=float, default=0.003); p.add_argument("--appnp_wd", type=float, default=0.0005)
    p.add_argument("--mixhop_hidden", type=int, default=16, help="CUAI MixHop grid: 8, 16, 32")
    p.add_argument("--mixhop_layers", type=int, default=2, help="CUAI MixHop grid: 2 or 3")
    p.add_argument("--mixhop_hops", type=int, default=2)
    p.add_argument("--mixhop_lr", type=float, default=0.01); p.add_argument("--mixhop_wd", type=float, default=0.0005)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--hops", type=int, default=1, help="LINKX sgc_exp.sh default is 1; Wu et al. use 2")
    p.add_argument("--no_metrics", action="store_true", help="skip HE / RcE / REE")
    p.add_argument("--metrics", nargs="+", default=["he", "rce", "ree"], choices=["he", "rce", "ree"],
                   help="compute only these spectral metrics in this run (run each metric as its own job); "
                        "same candidate settings and seeds reproduce the same partitions")
    p.add_argument("--device", default="auto", help="auto | cuda | cuda:1 | cpu")
    p.add_argument("--out", default="results.csv")
    p.add_argument("--best_out", default=None,
                   help="write per-objective tuned configurations and their per-seed results here (default: <out>_best.csv)")
    p.add_argument("--source", default="authors", choices=["authors", "geom"],
                   help="authors = their .pt/.mat files + storage convention (default); geom = PyG Geom-GCN, symmetrised")
    p.add_argument("--split", default="geom", choices=["geom", "random"],
                   help="geom = Geom-GCN fixed 60/20/20 splits (default); random = our older random splits")
    args = p.parse_args()
    if any(v < 1 for v in args.projectors):
        raise SystemExit("--projectors values must be positive integers")
    if any(v < 1 for v in args.shingle_hashes):
        raise SystemExit("--shingle_hashes values must be positive integers")
    if args.alpha_values is not None and any(not 0.0 <= v <= 1.0 for v in args.alpha_values):
        raise SystemExit("--alpha_values must be between 0 and 1")
    if args.spectral_only and args.no_metrics:
        raise SystemExit("--spectral_only requires spectral metrics; do not combine it with --no_metrics")
    if args.paper:
        args.F = args.F or ["orig"]
        args.hash = args.hash or ["random"]
        args.ratios = args.ratios or [0.5]
        args.alpha_mode = args.alpha_mode or "paper"
    args.F = args.F or ["feat", "orig", "neigh"]
    args.hash = args.hash or ["random", "pca_z"]
    # Spectral tuning defaults to the paper's fixed 50% coarsening level.
    # Pass --ratios explicitly only when evaluating another target separately.
    args.ratios = args.ratios or ([0.5] if args.spectral_only else [0.5, 0.3, 0.1])
    args.alpha_mode = args.alpha_mode or "computed"
    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but torch.cuda.is_available() is False (CPU-only torch build or no driver)")
    if args.spectral_only:
        print("spectral-only tuning: classifier training is disabled", flush=True)
    else:
        print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(dev)})" if dev.type == "cuda"
              else "   <-- training on CPU; install a CUDA build of torch to use the GPU"), flush=True)

    header = ["dataset", "model", "clf_input", "alpha_mode", "alpha", "F", "hash", "ratio", "projectors", "shingle_hashes", "seeds",
              "acc_mean", "acc_std", "he", "rce", "ree", "purity", "n_over_N", "per_seed_acc"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    new = not os.path.exists(args.out)
    if not new:
        with open(args.out, newline="") as existing:
            if next(csv.reader(existing), None) != header:
                raise SystemExit(f"{args.out} has an older/different CSV schema; choose a new --out path")
    fout = open(args.out, "a", newline="")
    w = csv.writer(fout)
    if new:
        w.writerow(header)

    best_out = args.best_out or (os.path.splitext(args.out)[0] + "_best.csv")
    os.makedirs(os.path.dirname(os.path.abspath(best_out)), exist_ok=True)
    best_header = ["dataset", "model", "clf_input", "alpha_mode", "alpha", "seed", "criterion",
                   "F", "hash", "ratio", "projectors", "shingle_hashes", "val_acc", "test_acc", "he", "rce", "ree", "purity", "n_over_N"]
    best_new = not os.path.exists(best_out)
    if not best_new:
        with open(best_out, newline="") as existing:
            if next(csv.reader(existing), None) != best_header:
                raise SystemExit(f"{best_out} has an older/different CSV schema; choose a new --best_out path")
    best_fout = open(best_out, "a", newline="")
    best_w = csv.writer(best_fout)
    if best_new:
        best_w.writerow(best_header)

    for name in args.datasets:
        A, A_lap, X, y, nc = load(name, args.source)
        N = A.shape[0]
        het = heterophily_alpha(A_lap, y)
        alpha = alpha_value(args.alpha_mode, name, A_lap, y)
        alpha_values = args.alpha_values or [alpha]
        if args.spectral_only:
            models = ["spectral"]
        elif args.model == ["both"]:
            models = ["gcn", "sgc"]
        elif args.model == ["all"]:
            models = ["gcn", "sgc", "appnp", "mixhop"]
        elif args.model == ["auto"]:
            models = [DEFAULT_MODEL[name]]
        elif any(m in {"auto", "both", "all"} for m in args.model):
            raise SystemExit("--model auto, both, and all must each be used alone")
        else:
            models = args.model
        print(f"\n=== {name}: N={N} edges={A_lap.nnz if args.source == "authors" else A_lap.nnz//2} feat={X.shape[1]} classes={nc} | 1-homophily={het:.3f} "
              f"homophily={1-het:.3f} paper={PAPER_ALPHA[name]} | alpha candidates={alpha_values} ({args.alpha_mode}) "
              f"| clf_input={args.clf_input} seeds={args.seeds}", flush=True)
        ref = None if args.no_metrics else SpectralRef(A_lap, X)
        if args.spectral_only:
            splits = [None] * args.seeds
        elif args.split == "geom" and name in GEOM_SPLIT:
            splits = [load_geom_split(name, s, N) for s in range(args.seeds)]
        elif name == "penn94":
            splits = [random_split_labeled(y, s, 0.5, 0.25) for s in range(args.seeds)]   # LINKX split
        else:
            splits = [random_split_labeled(y, s) for s in range(args.seeds)]
        cache = {}
        for m in models:
            candidates_by_seed = [[] for _ in range(args.seeds)]

            def report(label_f, label_h, ratio, projectors, shingle_hashes, alpha_used, accs, extra):
                accs = np.array(accs) * 100
                he_m, rce_m, ree_m, pur_m, nn_m = extra
                print(f"{'':<14}{label_f:<6}{label_h:<8}{accs.mean():6.2f}±{accs.std():4.2f}"
                      f"{he_m:8.3f}{rce_m:8.3f}{ree_m:8.3f}{100*pur_m:8.2f}{nn_m:7.3f}", flush=True)
                w.writerow([name, m, args.clf_input, args.alpha_mode, f"{alpha_used:.4f}", label_f, label_h, ratio,
                            projectors, shingle_hashes,
                            args.seeds, f"{accs.mean():.3f}", f"{accs.std():.3f}", f"{he_m:.4f}", f"{rce_m:.4f}",
                            f"{ree_m:.4f}", f"{100*pur_m:.2f}", f"{nn_m:.4f}", ";".join(f"{a:.2f}" for a in accs)])
                fout.flush()

            if not args.no_baseline and not args.spectral_only:
                print(f"[model = {m}]  uncoarsened baseline (ratio 1.0)   {'F':<6}{'hash':<8}{'acc %':>10}", flush=True)
                for kind in (args.F if args.clf_input == "F" else ["feat"]):
                    accs = [run_base(m, args, dev, kind, s, A, X, y, nc, alpha, splits[s], cache) for s in range(args.seeds)]
                    report(kind if args.clf_input == "F" else "X", "none", 1.0, "", "", alpha, accs,
                           (0.0, 0.0, 0.0, 1.0, 1.0))
            for ratio in args.ratios:
                print(f"[model = {m}]  ratio {ratio}   {'F':<6}{'hash':<8}{'acc %':>10}{'HE':>8}{'RcE':>8}{'REE':>8}{'pur %':>8}{'n/N':>7}", flush=True)
                for kind in args.F:
                    for mode in args.hash:
                        projector_values = args.projectors if mode == "random" else [""]
                        shingle_values = args.shingle_hashes if mode in ("shingle", "shingle_z") else [""]
                        alpha_candidates = args.alpha_values or [alpha]
                        if mode not in ("random", "pca_z"):
                            alpha_candidates = [alpha]
                        for projectors in projector_values:
                            n_projectors = projectors if projectors != "" else L_PROJ
                            for shingle_hashes in shingle_values:
                                n_shingle_hashes = shingle_hashes if shingle_hashes != "" else 3
                                for alpha_used in alpha_candidates:
                                    records = [run_one(m, args, dev, kind, mode, ratio, s, A, X, y, nc,
                                                       alpha_used, splits[s], cache, ref, n_projectors,
                                                       n_shingle_hashes)
                                               for s in range(args.seeds)]
                                    res = np.asarray(records)
                                    report(kind, mode, ratio, projectors, shingle_hashes, alpha_used, res[:, 1],
                                           tuple(res[:, j].mean() for j in range(2, 7)))
                                    for s, row in enumerate(records):
                                        candidates_by_seed[s].append({
                                            "seed": s, "F": kind, "hash": mode, "ratio": ratio,
                                            "projectors": projectors, "shingle_hashes": shingle_hashes,
                                            "alpha": alpha_used,
                                            "val_acc": row[0], "test_acc": row[1], "he": row[2], "rce": row[3],
                                            "ree": row[4], "purity": row[5], "n_over_N": row[6],
                                        })

            # Tune one configuration per hash method, ratio, and objective using
            # mean validation accuracy / mean spectral metric across seeds. Then
            # report that same selected configuration for every seed. This keeps
            # the spectral objectives independent and avoids per-seed hyperparameter
            # selection. Random and shingle are never pooled together.
            by_method = {}
            for seed, candidates in enumerate(candidates_by_seed):
                for candidate in candidates:
                    by_method.setdefault((candidate["hash"], candidate["ratio"]), []).append(
                        (seed, candidate))
            for (hash_name, ratio), method_candidates in sorted(by_method.items()):
                by_config = {}
                for seed, candidate in method_candidates:
                    key = (candidate["F"], candidate["projectors"], candidate["shingle_hashes"], candidate["alpha"])
                    by_config.setdefault(key, []).append((seed, candidate))
                objectives = [("max_val_acc", "val_acc", True)] + [
                    (f"min_{metric}", metric, False) for metric in ("he", "rce", "ree")]
                for criterion, objective, maximize in objectives:
                    eligible = []
                    for config, rows in by_config.items():
                        values = [c[objective] for _, c in rows]
                        if len(rows) == args.seeds and np.all(np.isfinite(values)):
                            eligible.append((float(np.mean(values)), config, rows))
                    if not eligible:
                        continue
                    selected = (max if maximize else min)(eligible, key=lambda item: item[0])
                    for seed, c in sorted(selected[2], key=lambda item: item[0]):
                        best_w.writerow([name, m, args.clf_input, args.alpha_mode, f"{c['alpha']:.4f}", seed,
                                         criterion, c["F"], hash_name, ratio, c["projectors"], c["shingle_hashes"],
                                         f"{c['val_acc']:.4f}", f"{c['test_acc']:.4f}",
                                         f"{c['he']:.6f}", f"{c['rce']:.6f}", f"{c['ree']:.6f}",
                                         f"{100*c['purity']:.2f}", f"{c['n_over_N']:.4f}"])
            best_fout.flush()
    fout.close()
    best_fout.close()
    print(f"\nresults appended to {args.out}")
    print(f"per-seed objective selections appended to {best_out}")


if __name__ == "__main__":
    main()
