"""
ahugc_eval_v2.py -- AH-UGC coarsening (random hash or PCA Z-order hash) + node classification
trained on the coarsened graph and tested on the ORIGINAL graph.

HOMOGENEOUS datasets : cora, citeseer, pubmed (Planetoid), cornell (WebKB), film (Actor)    [PyG, auto-download]
HETEROGENEOUS        : dblp, imdb (PyG), acm (PyG HGBDataset)   -- same loaders/targets as the HGCond repo the
                       AH-UGC paper takes its hetero models from;  hetero_synth = tiny synthetic smoke test.

F        : feat  = X only
           orig  = [(1-a) X | a A]                     (AH-UGC / UGC augmented feature)
           neigh = [(1-a) X | a mean_nbr(X)]
           (hetero, per node type t:  orig = [(1-a) X_t | a R_t],  R_t = all relation biadjacency rows of t;
                                      neigh = [(1-a) X_t | a relation-wise neighbour means of X])
alpha    : --alpha_mode takes one or more of
             computed  : a = 1 - edge homophily      (reproduces AH-UGC Table 5)
             homophily : a = edge homophily          (what Sec. 3.1 of the paper literally says)
             paper     : the Table 5 value
           `feat` does not depend on alpha, so it is run once.
           Hetero graphs: the paper does not define alpha; here edge homophily of the target type is measured on
           target-target edges plus 2-hop co-neighbour pairs (nodes sharing a neighbour of another type).
hash     : random = AH-UGC Alg. 1: l=500 Gaussian projectors, s_i = mean_k (W_k.F_i + b_k),
                    sort, repeatedly pick a random super-node and merge it with its right
                    neighbour until n = ratio*N          (no bin width)
           pca_z  = same merge procedure, but the order comes from a Morton (Z-order) key over
                    the top-3 principal components of F (centred, sparse-safe, deterministic signs)
           shingle = relation-aware, type-isolated neighborhood MinHash ordering (three signatures),
                     followed by the same right-neighbour merge stage.
Coarse graph (AH-UGC paper, Sec. 3.1):  A_c = C^T A C (weighted, binary C),  F_c = mean of member rows,
           super-node label = majority vote over TRAIN nodes only, loss only on super-nodes that
           contain a train node.   Hetero: type-isolated coarsening (Alg. 2), A_e~ = C_src^T A_e C_dst.
baseline : row "base" = the same model trained directly on the ORIGINAL graph with the original features X
           (the paper's "Base" column), same split / epochs / validation-based epoch choice.
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
  HeteroSGC / HeteroGCN / HeteroGCN2 = copied from github.com/jianjianGJ/hgcond (Apache-2.0), 64 hidden, 3 layers,
        features of every node type, adjacency D_dst^-1/2 A^T D_src^-1/2 per relation.  Coarse-graph training follows
        its evalue_hgcond (Adam lr 0.01, cross-entropy, validate on the REAL graph each epoch, patience 10 then
        re-initialise, keep best-val weights); "base" follows its train_model_ealystop (Adam lr 0.01, wd 5e-4).
        HGCond uses 1000 epochs; --hetero_epochs defaults to 1000 to match the source budget.
Splits   : homogeneous: random 60/20/20 per seed (UGC.py "split way 2"); same split for every config of a seed.
           hetero (--hetero_split public): HGCond's splits (DBLP/IMDB: dataset masks; ACM: fixed 200/200/rest
           labeled-paper order, seed 0, or --acm_index file); --hetero_split random is a non-paper alternative.
Protocol: hetero defaults match HGCond evaluation: one run, 1000 epochs, hidden=64, 3 layers, Adam lr=.01.
           Coarse fit uses cross-entropy and patience-10 restarts, validates on the original graph, and tests the
           best-validation checkpoint. Full-graph baseline uses cross-entropy and weight decay 5e-4.
--model auto = sgc for cornell/film (paper Table 9), gcn for cora/citeseer/pubmed (Tables 3/8).
--clf_input F : classifier consumes the same F that was hashed (what the UGC code does)
              X : classifier consumes node features only   (hetero always uses X)
Requires: torch, torch_geometric, scipy, numpy.

  python ahugc_eval_v2.py --datasets cornell film citeseer cora pubmed            # ratios .5 .3 .1, 10 seeds, both alphas, base
  python ahugc_eval_v2.py --datasets cora --model both --clf_input X --seeds 3 --ree
  python ahugc_eval_v2.py --datasets dblp acm --hetero_models HeteroSGC HeteroGCN --hetero_ratios 0.3
  python ahugc_eval_v2.py --datasets imdb dblp acm --hetero_models HeteroGCN2 --hetero_F feat --hash shingle --hetero_ratios 0.3
  python ahugc_eval_v2.py --datasets hetero_synth --hetero_epochs 60 --hetero_seeds 2      # no download, smoke test
"""
import argparse, csv, os, random, time
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, svds, eigsh
import torch
import torch.nn.functional as Fn
from torch.nn import Linear as _TorchLinear  # noqa: F401  (kept to mirror the authors' imports)
from torch_geometric.nn import GCNConv, SGConv, Linear
from torch_geometric.datasets import Planetoid, WebKB, Actor

PAPER_ALPHA = {"cora": 0.19, "citeseer": 0.26, "pubmed": 0.20, "cornell": 0.70, "film": 0.78}
DEFAULT_MODEL = {"cornell": "sgc", "film": "sgc", "cora": "gcn", "citeseer": "gcn", "pubmed": "gcn"}
HETERO = ("dblp", "imdb", "acm", "hetero_synth")
L_PROJ = 500            # number of random projectors (paper: "l", value not stated)


# ------------------------------------------------------------------ data (homogeneous)
def load(name, root="data"):
    if name in ("cora", "citeseer", "pubmed"):
        ds = Planetoid(f"{root}/{name}", name.capitalize() if name != "pubmed" else "PubMed")
    elif name == "cornell":
        ds = WebKB(f"{root}/webkb", "Cornell")
    elif name == "film":
        ds = Actor(f"{root}/actor")
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


def edge_homophily(A, y):
    r, c = A.nonzero()
    return float((y[r] == y[c]).mean())


def get_alpha(mode, A, y, name):
    h = edge_homophily(A, y)
    return {"computed": 1.0 - h, "homophily": h, "paper": PAPER_ALPHA.get(name, float("nan"))}[mode]


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
    """top-k principal scores of centred F without densifying F.
    Deterministic: fixed solver start (random_state=0) and each PC's sign is set so its
    largest-magnitude node score is positive (SVD signs are otherwise arbitrary).
    Tiny / constant F (e.g. a 20-node type with a single constant feature) falls back to zeros."""
    N, D = F.shape
    k = min(k, min(N, D) - 1)
    if k < 1:
        return np.zeros((N, 1))
    mu = np.asarray(F.mean(0)).ravel()
    op = LinearOperator(F.shape, dtype=np.float64,
                        matvec=lambda v: F @ v - mu @ v,
                        rmatvec=lambda u: F.T @ u - mu * np.sum(u),
                        matmat=lambda V: F @ V - (mu @ V)[None, :],
                        rmatmat=lambda U: F.T @ U - np.outer(mu, U.sum(0)))
    try:
        _, S, Vt = svds(op, k=k, random_state=0)
    except Exception:
        return np.zeros((N, 1))
    o = np.argsort(-S)
    V = Vt[o].T
    P = np.asarray(F @ V - (mu @ V)[None, :])
    sign = np.sign(P[np.argmax(np.abs(P), axis=0), np.arange(P.shape[1])])
    sign[sign == 0] = 1
    return P * sign


def hash_scores(F, mode, rng):
    if mode == "random":                       # AH-UGC Algorithm 1, lines 3-6
        W = rng.normal(size=(F.shape[1], L_PROJ))
        b = rng.uniform(0, 1, size=L_PROJ)
        return np.asarray(F @ W + b).mean(1)
    if mode == "pca_z":
        return morton(top_pcs(F)[:, :3]).astype(np.float64)
    raise ValueError(mode)


def coarsen(F, ratio, rng, mode, neighborhoods=None):
    """sort by hash score/signature, then apply AH-UGC's random adjacent right-merge stage."""
    N = F.shape[0]
    if mode == "shingle":
        if neighborhoods is None:
            raise ValueError("shingle coarsening requires a graph/incidence matrix")
        order = minhash_order(neighborhoods, rng)
    else:
        order = np.argsort(hash_scores(F, mode, rng), kind="stable")
    return merge_order(order, N, ratio, rng)


def merge_order(order, N, ratio, rng):
    """Apply right-neighbor merges to a supplied node order."""
    groups = [[int(i)] for i in order]
    target = max(1, int(round(ratio * N)))
    while len(groups) > target:
        j = int(rng.integers(len(groups) - 1))
        groups[j] += groups.pop(j + 1)
    pi = np.empty(N, dtype=np.int64)
    for g, members in enumerate(groups):
        pi[members] = g
    return pi


def minhash_order(neighborhoods, rng, nh=3):
    """Three-signature closed-neighborhood MinHash order over a CSR incidence matrix."""
    M = sp.csr_matrix(neighborhoods, dtype=np.float32).copy()
    M.data[:] = 1.0
    M.eliminate_zeros(); M.sum_duplicates()
    signatures = []
    for _ in range(nh):
        ranks = rng.permutation(M.shape[1])
        signatures.append(np.minimum.reduceat(ranks[M.indices], M.indptr[:-1]))
    return np.lexsort(tuple(reversed(signatures)))


def hetero_shingle_order(hd, node_type, rng, nh=3):
    """Relation-aware neighborhood MinHash; relation types have disjoint ID blocks."""
    n = hd.N[node_type]
    blocks = [sp.eye(n, format="csr", dtype=np.float32)]
    for (src, _, dst), A in hd.A.items():
        if src == node_type:
            blocks.append(sp.csr_matrix(A, dtype=np.float32))
        if dst == node_type:
            blocks.append(sp.csr_matrix(A.T, dtype=np.float32))
    return minhash_order(sp.hstack(blocks, format="csr"), rng, nh)


def purity(pi, y, nc):
    lab = y >= 0                                # hetero targets may have unlabelled nodes (y = -1)
    c = np.zeros((pi.max() + 1, nc)); np.add.at(c, (pi[lab], y[lab]), 1)
    return c.max(1).sum() / max(lab.sum(), 1)


def ree(A, C, k=10):
    deg = np.asarray(A.sum(1)).ravel()
    L = sp.diags(deg) - A
    cnt = np.asarray(C.sum(0)).ravel()
    Cn = C @ sp.diags(cnt ** -0.5)
    Lc = Cn.T @ L @ Cn
    lam = np.sort(eigsh(L.astype(np.float64), k=k, which="LA", return_eigenvectors=False))
    lam_c = np.sort(eigsh(Lc.tocsr(), k=k, which="LA", return_eigenvectors=False))
    return float(np.mean(np.abs(lam_c - lam) / lam))


def lap_nodiag(M):
    M = sp.csr_matrix(M, dtype=np.float64); M.setdiag(0); M.eliminate_zeros()
    return sp.diags(np.asarray(M.sum(1)).ravel()) - M


def ree_ugc(A, C, k=100):
    """REE exactly as in the UGC repo (spectral_properties.eigen_error): top-k (LM) eigenvalues of the original
    Laplacian vs those of the Laplacian of the coarse weighted graph C^T A C (self-loops dropped), mean |l-lc|/l,
    eigenvalues compared position-wise.  k = 100 (as in UGC/FACH), reduced for tiny graphs."""
    n = C.shape[1]
    k = min(k, A.shape[0] // 2, n - 2)
    if k < 1:
        return float("nan")
    l = eigsh(lap_nodiag(A), k=k, which="LM", tol=1e-3, return_eigenvectors=False)
    lc = eigsh(lap_nodiag(C.T @ A @ C), k=k, which="LM", tol=1e-3, return_eigenvectors=False)
    return float(np.mean(np.abs(l - lc) / l))


# ------------------------------------------------------------------ models (homogeneous)
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


def to_t(M, dense=True):
    M = sp.csr_matrix(M)
    return torch.from_numpy(M.toarray()).float() if dense else M


def edge_index_t(M, dev):
    M = sp.coo_matrix(M)
    return torch.from_numpy(np.vstack((M.row, M.col))).long().to(dev)


def build_model(model_name, args, in_dim, nc, dev):
    """returns model, optimiser, and the two forward closures' ingredients."""
    if model_name == "gcn":
        model = GCN_(in_dim, args.hidden, nc).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=args.gcn_lr, weight_decay=args.gcn_wd)
    else:
        model = SGC(in_dim, nc, args.hops).to(dev)
        model.reset_parameters()
        opt = torch.optim.AdamW(model.parameters(), lr=args.sgc_lr, weight_decay=args.sgc_wd)
    return model, opt


def train_eval(model_name, model, opt, fwd_c, fwd_o, yc, has_train, Y, val_t, test_t, epochs):
    """train on (coarse) graph, score on the ORIGINAL graph each epoch, test acc at best validation epoch."""
    best_val, best_test = -1.0, 0.0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        out = fwd_c()
        out = Fn.log_softmax(out, dim=1) if model_name == "sgc" else out
        loss = Fn.nll_loss(out[has_train], yc[has_train])
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = fwd_o().argmax(1)
        v = (pred[val_t] == Y[val_t]).float().mean().item()
        if v > best_val:
            best_val = v
            best_test = (pred[test_t] == Y[test_t]).float().mean().item()
    return best_test


# ------------------------------------------------------------------ one homogeneous run
def orig_input(cache, key, make, dev):
    """original-graph input tensor, cached (one entry at a time to bound memory)."""
    if key not in cache:
        cache.clear()
        cache[key] = to_t(make()).to(dev)
    return cache[key]


def run_one(model_name, args, dev, kind, mode, ratio, seed, A, X, y, nc, alpha, split, graph_cache):
    train, val, test = split
    N = A.shape[0]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    F = build_F(kind, A, X, alpha)
    pi = coarsen(F, ratio, rng, mode, A + sp.eye(N, format="csr"))
    n = int(pi.max()) + 1
    C = sp.csr_matrix((np.ones(N), (np.arange(N), pi)), shape=(N, n))
    size = np.asarray(C.sum(0)).ravel()

    use_F = args.clf_input == "F" and kind != "feat"
    Fin = F if use_F else X
    Fc = sp.diags(1 / size) @ (C.T @ Fin)                    # mean of member rows
    Ac = (C.T @ A @ C).tocoo()                               # weighted, includes internal edges on the diagonal
    ei_c = torch.from_numpy(np.vstack((Ac.row, Ac.col))).long().to(dev)
    ew_c = torch.from_numpy(Ac.data).float().to(dev)

    votes = np.zeros((n, nc)); tr = np.where(train)[0]
    np.add.at(votes, (pi[tr], y[tr]), 1)                     # TRAIN labels only
    yc = torch.from_numpy(votes.argmax(1)).long().to(dev)
    has_train = torch.from_numpy(votes.sum(1) > 0).to(dev)

    Xo = orig_input(graph_cache, (kind, round(alpha, 6)) if use_F else ("X",), lambda: Fin, dev)
    ei_o = edge_index_t(A, dev)
    Xc = to_t(Fc).to(dev)
    Y = torch.from_numpy(y).long().to(dev)
    val_t, test_t = torch.from_numpy(val).to(dev), torch.from_numpy(test).to(dev)

    model, opt = build_model(model_name, args, Xc.shape[1], nc, dev)
    if model_name == "gcn":
        fwd_c = lambda: model(Xc, ei_c, ew_c)
        fwd_o = lambda: model(Xo, ei_o, None)
    else:
        fwd_c = lambda: model(Xc, ei_c, None, original=False)
        fwd_o = lambda: model(Xo, ei_o, None, original=True)
    best_test = train_eval(model_name, model, opt, fwd_c, fwd_o, yc, has_train, Y, val_t, test_t, args.epochs)
    r = (ree_ugc(A, C) if args.ree_mode == "ugc" else ree(A, C)) if args.ree else float("nan")
    return best_test, r, purity(pi, y, nc), n / N


def run_baseline(model_name, args, dev, seed, A, X, y, nc, split, graph_cache):
    """no coarsening: same model, trained on the ORIGINAL graph with the original features X."""
    train, val, test = split
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    Xo = orig_input(graph_cache, ("X",), lambda: X, dev)
    ei_o = edge_index_t(A, dev)
    Y = torch.from_numpy(y).long().to(dev)
    tr_t = torch.from_numpy(train).to(dev)
    val_t, test_t = torch.from_numpy(val).to(dev), torch.from_numpy(test).to(dev)
    model, opt = build_model(model_name, args, Xo.shape[1], nc, dev)
    if model_name == "gcn":
        fwd = lambda: model(Xo, ei_o, None)
    else:
        fwd = lambda: model(Xo, ei_o, None, original=True)
    acc = train_eval(model_name, model, opt, fwd, fwd, Y, tr_t, Y, val_t, test_t, args.epochs)
    return acc, float("nan"), float("nan"), 1.0


def random_split(N, seed):
    perm = np.random.default_rng(1000 + seed).permutation(N)
    a, b = int(0.6 * N), int(0.8 * N)
    tr, va, te = (np.zeros(N, bool) for _ in range(3))
    tr[perm[:a]] = True; va[perm[a:b]] = True; te[perm[b:]] = True
    return tr, va, te


# ================================================================== heterogeneous graphs
# ---- models: copied from github.com/jianjianGJ/hgcond (Apache-2.0); inputs are x_dict and adj_t_dict
class HeteroSGC(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers,
                 node_types, edge_types, target_node_type, num_lins=1, alpha=0.01):
        super().__init__()
        self.node_types, self.edge_types, self.alpha = node_types, edge_types, alpha
        self.num_layers, self.num_lins, self.target_node_type = num_layers, num_lins, target_node_type
        self.in_lin_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            self.in_lin_dict[node_type] = torch.nn.ModuleList()
            self.in_lin_dict[node_type].append(Linear(-1, hidden_channels))
            for _ in range(num_lins - 1):
                self.in_lin_dict[node_type].append(Linear(hidden_channels, hidden_channels))
        self.out_lin = Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for node_type in self.in_lin_dict.keys():
            for lin in self.in_lin_dict[node_type]:
                lin.reset_parameters()
        self.out_lin.reset_parameters()

    def forward(self, x_dict, adj_t_dict):
        h_dict = {}
        for node_type in x_dict.keys():
            h_dict[node_type] = self.in_lin_dict[node_type][0](x_dict[node_type]).relu_()
            for lin in self.in_lin_dict[node_type][1:]:
                h_dict[node_type] = lin(h_dict[node_type]).relu_()
        for l in range(self.num_layers):
            out_dict = {node_type: [self.alpha * x] for node_type, x in h_dict.items()}
            for edge_type, adj_t in adj_t_dict.items():
                src_type, _, dst_type = edge_type
                out_dict[dst_type].append(adj_t @ h_dict[src_type])
            for node_type in x_dict.keys():
                h_dict[node_type] = torch.sum(torch.stack(out_dict[node_type], dim=0), dim=0)
        return self.out_lin(h_dict[self.target_node_type])


class HeteroGCN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers,
                 edge_types, node_types, target_node_type, alpha=1):
        super().__init__()
        self.alpha, self.num_layers, self.target_node_type = alpha, num_layers, target_node_type
        self.lins = torch.nn.ModuleDict()
        for node_type in node_types:
            self.lins[node_type] = torch.nn.ModuleList()
            self.lins[node_type].append(Linear(-1, hidden_channels, bias=False))
            for _ in range(num_layers - 1):
                self.lins[node_type].append(Linear(hidden_channels, hidden_channels, bias=False))
        self.out_lin = Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for node_type in self.lins.keys():
            for lin in self.lins[node_type]:
                lin.reset_parameters()
        self.out_lin.reset_parameters()

    def forward(self, x_dict, adj_t_dict):
        h_dict = {}
        for l in range(self.num_layers):
            for node_type in x_dict.keys():
                if l == 0:
                    h_dict[node_type] = Fn.relu_(self.lins[node_type][l](x_dict[node_type]))
                else:
                    h_dict[node_type] = Fn.relu_(self.lins[node_type][l](h_dict[node_type]))
            out_dict = {node_type: [self.alpha * x] for node_type, x in h_dict.items()}
            for edge_type, adj_t in adj_t_dict.items():
                src_type, _, dst_type = edge_type
                out_dict[dst_type].append(adj_t @ h_dict[src_type])
            for node_type in x_dict.keys():
                h_dict[node_type] = torch.mean(torch.stack(out_dict[node_type], dim=0), dim=0)
        return self.out_lin(h_dict[self.target_node_type])


class HeteroGCN2(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers,
                 edge_types, node_types, target_node_type, alpha=1, beta=0.1):
        super().__init__()
        self.alpha, self.beta, self.num_layers, self.target_node_type = alpha, beta, num_layers, target_node_type
        self.lins = torch.nn.ModuleDict()
        for node_type in node_types:
            self.lins[node_type] = torch.nn.ModuleList()
            self.lins[node_type].append(Linear(-1, hidden_channels, bias=False))
            for _ in range(num_layers - 1):
                self.lins[node_type].append(Linear(hidden_channels, hidden_channels, bias=False))
        self.out_lin = Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for node_type in self.lins.keys():
            for lin in self.lins[node_type]:
                lin.reset_parameters()
        self.out_lin.reset_parameters()

    def forward(self, x_dict, adj_t_dict):
        h_dict, h0_dict = {}, {}
        for l in range(self.num_layers):
            for node_type in x_dict.keys():
                if l == 0:
                    h_dict[node_type] = Fn.relu_(self.lins[node_type][l](x_dict[node_type]))
                    h0_dict[node_type] = h_dict[node_type].detach()
                else:
                    h_dict[node_type] = Fn.relu_(self.lins[node_type][l](h_dict[node_type]))
            out_dict = {node_type: [self.alpha * x] for node_type, x in h_dict.items()}
            for edge_type, adj_t in adj_t_dict.items():
                src_type, _, dst_type = edge_type
                out_dict[dst_type].append(adj_t @ h_dict[src_type])
            for node_type in x_dict.keys():
                h_dict[node_type] = self.beta * h0_dict[node_type] + (1 - self.beta) * torch.mean(
                    torch.stack(out_dict[node_type], dim=0), dim=0)
        return self.out_lin(h_dict[self.target_node_type])


HGNN = {"HeteroSGC": HeteroSGC, "HeteroGCN": HeteroGCN, "HeteroGCN2": HeteroGCN2}


# ---- data container + loaders
def _csr(ei, n_src, n_dst):
    A = sp.coo_matrix((np.ones(ei.shape[1], np.float32), (ei[0], ei[1])), shape=(n_src, n_dst)).tocsr()
    A.data[:] = 1.0                                           # collapse duplicate edges
    return A


def hetero_from_pyg(d, target, name):
    """PyG HeteroData -> HData (numpy / scipy)."""
    hd = SimpleNamespace(name=name, target=target, node_types=list(d.node_types), edge_types=list(d.edge_types))
    hd.N = {t: int(d[t].num_nodes) for t in hd.node_types}
    hd.x = {}
    for t in hd.node_types:
        x = getattr(d[t], "x", None)
        hd.x[t] = np.ones((hd.N[t], 1), np.float32) if x is None else x.numpy().astype(np.float32)
    hd.A = {et: _csr(d[et].edge_index.numpy(), hd.N[et[0]], hd.N[et[2]]) for et in hd.edge_types}
    hd.y = d[target].y.numpy().astype(np.int64)
    hd.nc = int(hd.y.max()) + 1
    return hd


def load_hetero(name, root, acm_index=None):
    if name == "hetero_synth":
        return make_hetero_synth()
    if name == "dblp":
        from torch_geometric.datasets import DBLP
        d = DBLP(f"{root}/DBLP")[0]
        d["conference"].x = torch.ones(d["conference"].num_nodes, 1)           # as in HGCond utils_data.py
        hd = hetero_from_pyg(d, "author", "dblp")
        hd.public = tuple(d["author"][k].numpy() for k in ("train_mask", "val_mask", "test_mask"))
    elif name == "imdb":
        from torch_geometric.datasets import IMDB
        d = IMDB(f"{root}/IMDB")[0]
        hd = hetero_from_pyg(d, "movie", "imdb")
        hd.public = tuple(d["movie"][k].numpy() for k in ("train_mask", "val_mask", "test_mask"))
    elif name == "acm":
        from torch_geometric.datasets import HGBDataset
        d = HGBDataset(f"{root}/HGB", "ACM")[0]
        d["term"].x = torch.ones(d["term"].num_nodes, 1)                         # as in HGCond utils_data.py
        hd = hetero_from_pyg(d, "paper", "acm")
        n = hd.N["paper"]
        labelled = torch.arange(n)[d["paper"].train_mask]
        if acm_index and os.path.exists(acm_index):
            perm = torch.load(acm_index).long()
        else:                                                                    # HGCond: manual_seed(0) + randperm
            g = torch.Generator().manual_seed(0)
            perm = labelled[torch.randperm(labelled.shape[0], generator=g)]
        masks = []
        for sl in (perm[:200], perm[200:400], perm[400:]):
            m = np.zeros(n, bool); m[sl.numpy()] = True; masks.append(m)
        hd.public = tuple(masks)
    else:
        raise ValueError(name)
    return hd


def make_hetero_synth(seed=0):
    """tiny DBLP-shaped graph (author / paper / term / conference) with label signal; smoke test only."""
    rng = np.random.default_rng(seed)
    nc, Na, Np, Nt, Nc = 4, 300, 800, 200, 10
    ya = rng.integers(0, nc, Na); cp = rng.integers(0, nc, Np)
    Xa = (rng.random((Na, 24)) < 0.10).astype(np.float32)
    Xp = (rng.random((Np, 40)) < 0.05).astype(np.float32)
    for k in range(nc):
        Xa[ya == k, 6 * k:6 * k + 6] += (rng.random(((ya == k).sum(), 6)) < 0.4)
        Xp[cp == k, 10 * k:10 * k + 10] += (rng.random(((cp == k).sum(), 10)) < 0.4)
    Xt = (rng.random((Nt, 10)) < 0.2).astype(np.float32)
    ap = []
    for p in range(Np):
        pool = np.where(ya == cp[p])[0]
        for a in rng.choice(pool, 2, replace=False):
            ap.append((a, p))
    ap = np.array(ap).T
    pt = np.array([(p, t) for p in range(Np) for t in rng.choice(Nt, 4, replace=False)]).T
    pc = np.array([(p, cp[p] % Nc) for p in range(Np)]).T
    hd = SimpleNamespace(name="hetero_synth", target="author", node_types=["author", "paper", "term", "conference"])
    hd.N = {"author": Na, "paper": Np, "term": Nt, "conference": Nc}
    hd.x = {"author": Xa, "paper": Xp, "term": Xt, "conference": np.ones((Nc, 1), np.float32)}
    rel = {("author", "to", "paper"): ap, ("paper", "to", "author"): ap[::-1],
           ("paper", "to", "term"): pt, ("term", "to", "paper"): pt[::-1],
           ("paper", "to", "conference"): pc, ("conference", "to", "paper"): pc[::-1]}
    hd.edge_types = list(rel)
    hd.A = {et: _csr(np.ascontiguousarray(ei), hd.N[et[0]], hd.N[et[2]]) for et, ei in rel.items()}
    hd.y, hd.nc = ya.astype(np.int64), nc
    hd.public = None
    return hd


def hetero_splits(hd, seed, how):
    if how == "public" and hd.public is not None:
        return hd.public
    lab = np.where(hd.y >= 0)[0]
    perm = np.random.default_rng(1000 + seed).permutation(lab)
    a, b = int(0.6 * len(lab)), int(0.8 * len(lab))
    out = []
    for sl in (perm[:a], perm[a:b], perm[b:]):
        m = np.zeros(len(hd.y), bool); m[sl] = True; out.append(m)
    return tuple(out)


def hetero_edge_homophily(hd):
    """paper gives no hetero alpha: edge homophily of the target type over target-target edges plus
    2-hop co-neighbour pairs (two target nodes sharing a neighbour of another type)."""
    t, y = hd.target, hd.y
    lab = y >= 0
    same = tot = 0
    for (s, rel, d), A in hd.A.items():
        if s != t:
            continue
        P = A.copy() if d == t else (A @ A.T).tocsr()
        P = sp.csr_matrix(P); P.setdiag(0); P.eliminate_zeros()
        r, c = P.nonzero()
        keep = lab[r] & lab[c]
        same += int((y[r][keep] == y[c][keep]).sum()); tot += int(keep.sum())
    return same / max(tot, 1)


def hetero_alpha(mode, hd):
    h = hetero_edge_homophily(hd)
    return {"computed": 1.0 - h, "homophily": h}.get(mode, float("nan"))


def hetero_F(kind, hd, t, alpha):
    X = sp.csr_matrix(hd.x[t])
    if kind == "feat":
        return X
    blocks = []
    for (s, rel, d), A in hd.A.items():
        if s != t:
            continue
        if kind == "orig":
            blocks.append(A)
        elif kind == "neigh":
            deg = np.asarray(A.sum(1)).ravel()
            blocks.append(sp.diags(1 / np.maximum(deg, 1)) @ A @ sp.csr_matrix(hd.x[d]))
        else:
            raise ValueError(kind)
    if not blocks:
        return X
    return sp.hstack([(1 - alpha) * X, alpha * sp.hstack(blocks)]).tocsr()


def asym_norm(adj_t):
    """HGCond asymmetric_gcn_norm: D_dst^-1/2 adj_t D_src^-1/2, adj_t is (n_dst x n_src)."""
    adj_t = sp.csr_matrix(adj_t, dtype=np.float32)
    ds = np.asarray(adj_t.sum(0)).ravel() + 1e-5
    dd = np.asarray(adj_t.sum(1)).ravel() + 1e-5
    return (sp.diags(dd ** -0.5) @ adj_t @ sp.diags(ds ** -0.5)).tocoo()


def sp_to_torch(M, dev):
    M = sp.coo_matrix(M)
    idx = torch.from_numpy(np.vstack((M.row, M.col))).long()
    return torch.sparse_coo_tensor(idx, torch.from_numpy(M.data.astype(np.float32)), M.shape).coalesce().to(dev)


def hetero_tensors(x_np, A_dict, dev):
    x = {t: torch.from_numpy(np.asarray(v, np.float32)).to(dev) for t, v in x_np.items()}
    adj = {et: sp_to_torch(asym_norm(A.T), dev) for et, A in A_dict.items()}
    return x, adj


def make_hgnn(model_name, args, hd, nc, dev, real_x, real_adj):
    model = HGNN[model_name](hidden_channels=args.hetero_hidden, out_channels=nc, num_layers=args.hetero_layers,
                             node_types=hd.node_types, edge_types=hd.edge_types,
                             target_node_type=hd.target).to(dev)
    with torch.no_grad():                                      # lazy Linear(-1, .) shape inference on the real graph
        model(real_x, real_adj)
    return model


def hg_fit_coarse(model_name, args, hd, dev, real, syn):
    """HGCond evalue_hgcond protocol (train on the coarse graph, validate on the REAL graph, restart on stall)."""
    model = make_hgnn(model_name, args, hd, hd.nc, dev, real.x, real.adj)
    best, weights, patience, restart, opt = 0.0, None, 0, True, None
    for ep in range(args.hetero_epochs):
        if restart:
            patience, restart = 0, False
            model.reset_parameters()
            opt = torch.optim.Adam(model.parameters(), lr=args.hetero_lr)
        model.train(); opt.zero_grad()
        loss = Fn.cross_entropy(model(syn.x, syn.adj)[syn.mask], syn.y[syn.mask])
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(real.x, real.adj)[real.val].argmax(1) == real.y[real.val]).float().mean().item()
        if acc > best or weights is None:
            best, patience, weights = max(acc, best), 0, deepcopy(model.state_dict())
        else:
            patience += 1
        if patience == 10:
            restart = True
    model.load_state_dict(weights); model.eval()
    with torch.no_grad():
        return (model(real.x, real.adj)[real.test].argmax(1) == real.y[real.test]).float().mean().item()


def hg_fit_base(model_name, args, hd, dev, real):
    """HGCond train_model_ealystop protocol on the whole real graph (the 'Whole Dataset' number)."""
    model = make_hgnn(model_name, args, hd, hd.nc, dev, real.x, real.adj)
    model.reset_parameters()
    opt = torch.optim.Adam(model.parameters(), lr=args.hetero_lr, weight_decay=args.hetero_wd)
    best, weights = 0.0, None
    for ep in range(args.hetero_epochs):
        model.train(); opt.zero_grad()
        loss = Fn.cross_entropy(model(real.x, real.adj)[real.train], real.y[real.train])
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(real.x, real.adj)[real.val].argmax(1) == real.y[real.val]).float().mean().item()
        if acc > best or weights is None:
            best, weights = max(acc, best), deepcopy(model.state_dict())
    model.load_state_dict(weights); model.eval()
    with torch.no_grad():
        return (model(real.x, real.adj)[real.test].argmax(1) == real.y[real.test]).float().mean().item()


def hetero_coarsen_np(hd, kind, mode, ratio, alpha, rng, train):
    """type-isolated coarsening (Alg. 2), numpy/scipy only: per-type assignments, features, coarse relations, labels."""
    pis, Cs = {}, {}
    for t in hd.node_types:
        if mode == "shingle":
            pi = merge_order(hetero_shingle_order(hd, t, rng), hd.N[t], ratio, rng)
        else:
            pi = coarsen(hetero_F(kind, hd, t, alpha), ratio, rng, mode)
        pis[t] = pi
        Cs[t] = sp.csr_matrix((np.ones(len(pi)), (np.arange(len(pi)), pi)), shape=(len(pi), int(pi.max()) + 1))
    xc = {}
    for t in hd.node_types:                                   # super-node features = mean of member rows
        cnt = np.asarray(Cs[t].sum(0)).ravel()
        xc[t] = (sp.diags(1 / cnt) @ (Cs[t].T @ sp.csr_matrix(hd.x[t]))).toarray()
    Ac = {et: sp.csr_matrix(Cs[et[0]].T @ A @ Cs[et[2]]) for et, A in hd.A.items()}
    pi_t = pis[hd.target]
    votes = np.zeros((int(pi_t.max()) + 1, hd.nc))
    tr = np.where(train & (hd.y >= 0))[0]
    np.add.at(votes, (pi_t[tr], hd.y[tr]), 1)                 # TRAIN labels only
    return pis, Cs, xc, Ac, votes


def hetero_run(model_name, args, dev, kind, mode, ratio, seed, hd, alpha, real_base):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train, val, test = hetero_splits(hd, seed, args.hetero_split)
    real = SimpleNamespace(**vars(real_base))
    real.train, real.val, real.test = (torch.from_numpy(m).to(dev) for m in (train, val, test))

    pis, Cs, xc, Ac, votes = hetero_coarsen_np(hd, kind, mode, ratio, alpha, rng, train)
    sx, sadj = hetero_tensors(xc, Ac, dev)
    syn = SimpleNamespace(x=sx, adj=sadj,
                          y=torch.from_numpy(votes.argmax(1)).long().to(dev),
                          mask=torch.from_numpy(votes.sum(1) > 0).to(dev))
    acc = hg_fit_coarse(model_name, args, hd, dev, real, syn)
    n_tot = sum(C.shape[1] for C in Cs.values())
    return acc, float("nan"), purity(pis[hd.target], hd.y, hd.nc), n_tot / sum(hd.N.values())


# ------------------------------------------------------------------ driver
def fmt_row(ratio, kind, amode, alpha, mode, res):
    mu, sd = res.mean(0), res.std(0)
    a = "-" if amode == "-" else f"{amode[:4]} {alpha:.2f}"
    return (f"{ratio:<6}{kind:<6}{a:<11}{mode:<8}{100*mu[0]:6.2f}±{100*sd[0]:4.2f}"
            f"{mu[1]:8.3f}±{sd[1]:5.3f}{100*mu[2]:8.2f}±{100*sd[2]:4.2f}{mu[3]:7.3f}")


HEAD = f"{'ratio':<6}{'F':<6}{'alpha':<11}{'hash':<8}{'acc %':>13}{'REE':>14}{'purity %':>13}{'n/N':>7}"
CSV_HEAD = ["dataset", "model", "clf_input", "ratio", "F", "alpha_mode", "alpha", "hash", "seeds",
            "acc_mean", "acc_std", "ree", "purity", "n_over_N"]


def save_rows(path, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(CSV_HEAD)
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["cornell", "film", "citeseer", "cora", "pubmed"],
                   help="homogeneous: cora citeseer pubmed cornell film | hetero: dblp imdb acm hetero_synth")
    p.add_argument("--model", default="auto", choices=["auto", "gcn", "sgc", "both"])
    p.add_argument("--F", nargs="+", default=["feat", "orig", "neigh"])
    p.add_argument("--hash", nargs="+", default=["random", "pca_z"], choices=["random", "pca_z", "shingle"])
    p.add_argument("--clf_input", default="F", choices=["F", "X"])
    p.add_argument("--ratios", nargs="+", type=float, default=[0.5, 0.3, 0.1], help="supernodes / nodes")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--alpha_mode", nargs="+", default=["computed", "homophily"],
                   choices=["computed", "homophily", "paper"],
                   help="computed = 1-homophily (paper Table 5), homophily = homophily (paper text), paper = Table 5 values")
    p.add_argument("--no_baseline", action="store_true", help="skip the uncoarsened 'base' row")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--gcn_lr", type=float, default=0.003); p.add_argument("--gcn_wd", type=float, default=0.0005)
    p.add_argument("--sgc_lr", type=float, default=0.01); p.add_argument("--sgc_wd", type=float, default=1e-3)
    p.add_argument("--hops", type=int, default=1, help="LINKX sgc_exp.sh default is 1; Wu et al. use 2")
    p.add_argument("--ree", action="store_true", help="also compute relative eigen error (slow on big graphs)")
    p.add_argument("--ree_mode", default="ugc", choices=["ugc", "projected"],
                   help="ugc = UGC repo definition (top-100, coarse-graph Laplacian); projected = top-10 of C_n^T L C_n")
    # heterogeneous options
    p.add_argument("--hetero_models", nargs="+", default=["HeteroSGC"], choices=list(HGNN))
    p.add_argument("--hetero_F", nargs="+", default=["feat", "orig"], choices=["feat", "orig", "neigh"])
    p.add_argument("--hetero_ratios", nargs="+", type=float, default=[0.3], help="paper Table 4 uses 0.3")
    p.add_argument("--hetero_seeds", type=int, default=1,
                   help="HGCond evalue_hgcond default is one initialization/evaluation")
    p.add_argument("--hetero_epochs", type=int, default=1000, help="HGCond source default is 1000")
    p.add_argument("--hetero_lr", type=float, default=0.01); p.add_argument("--hetero_wd", type=float, default=5e-4)
    p.add_argument("--hetero_hidden", type=int, default=64); p.add_argument("--hetero_layers", type=int, default=3)
    p.add_argument("--hetero_split", default="public", choices=["public", "random"])
    p.add_argument("--acm_index", default=None, help="HGCond datahetero/acm_index_labeled_rand.tensor for exact ACM splits")
    p.add_argument("--out", default="results_v2.csv")
    args = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name in args.datasets:
        rows = []
        if name in HETERO:
            hetero_dataset(name, args, dev, rows)
        else:
            homogeneous_dataset(name, args, dev, rows)
        save_rows(args.out, rows)
        print(f"[saved {len(rows)} rows to {args.out}]", flush=True)


def homogeneous_dataset(name, args, dev, rows):
    A, X, y, nc = load(name)
    N = A.shape[0]
    alphas = {m: get_alpha(m, A, y, name) for m in args.alpha_mode}
    models = ["gcn", "sgc"] if args.model == "both" else [DEFAULT_MODEL[name] if args.model == "auto" else args.model]
    print(f"\n=== {name}: N={N} edges={A.nnz//2} feat={X.shape[1]} classes={nc} edge_homophily={edge_homophily(A, y):.3f} "
          f"alpha " + " ".join(f"{m}={a:.3f}" for m, a in alphas.items()) + f" (paper {PAPER_ALPHA[name]}) "
          f"clf_input={args.clf_input} ratios={args.ratios} seeds={args.seeds}", flush=True)
    splits = [random_split(N, s) for s in range(args.seeds)]
    cache = {}
    for m in models:
        print(f"[model = {m}]  {HEAD}", flush=True)
        if not args.no_baseline:
            res = np.array([run_baseline(m, args, dev, s, A, X, y, nc, splits[s], cache) for s in range(args.seeds)])
            mu, sd = res.mean(0), res.std(0)
            print(f"{'1.0':<6}{'base':<6}{'-':<11}{'-':<8}{100*mu[0]:6.2f}±{100*sd[0]:4.2f}", flush=True)
            rows.append([name, m, "X", 1.0, "base", "-", "", "-", args.seeds, 100 * mu[0], 100 * sd[0], "", "", 1.0])
        for ratio in args.ratios:
            for kind in args.F:
                for amode in (["-"] if kind == "feat" else args.alpha_mode):
                    alpha = 0.0 if amode == "-" else alphas[amode]
                    for mode in args.hash:
                        res = np.array([run_one(m, args, dev, kind, mode, ratio, s, A, X, y, nc, alpha, splits[s], cache)
                                        for s in range(args.seeds)])
                        mu, sd = res.mean(0), res.std(0)
                        print(fmt_row(ratio, kind, amode, alpha, mode, res), flush=True)
                        rows.append([name, m, args.clf_input, ratio, kind, amode, "" if amode == "-" else alpha, mode,
                                     args.seeds, 100 * mu[0], 100 * sd[0], mu[1], 100 * mu[2], mu[3]])
            print(flush=True)


def hetero_dataset(name, args, dev, rows):
    hd = load_hetero(name, "data", args.acm_index)
    alphas = {m: hetero_alpha(m, hd) for m in args.alpha_mode if m != "paper"}
    print(f"\n=== {name} (heterogeneous): target={hd.target} classes={hd.nc} nodes " +
          " ".join(f"{t}={n}" for t, n in hd.N.items()) + f" | hetero edge_homophily={hetero_edge_homophily(hd):.3f} "
          f"alpha " + " ".join(f"{m}={a:.3f}" for m, a in alphas.items()) +
          f" | split={args.hetero_split} ratios={args.hetero_ratios} seeds={args.hetero_seeds}", flush=True)
    rx, radj = hetero_tensors(hd.x, hd.A, dev)
    real_base = SimpleNamespace(x=rx, adj=radj, y=torch.from_numpy(hd.y).long().to(dev))
    for m in args.hetero_models:
        print(f"[model = {m}]  {HEAD}", flush=True)
        if not args.no_baseline:
            accs = []
            for s in range(args.hetero_seeds):
                random.seed(s); np.random.seed(s); torch.manual_seed(s)
                real = SimpleNamespace(**vars(real_base))
                real.train, real.val, real.test = (torch.from_numpy(mk).to(dev) for mk in hetero_splits(hd, s, args.hetero_split))
                accs.append(hg_fit_base(m, args, hd, dev, real))
            print(f"{'1.0':<6}{'base':<6}{'-':<11}{'-':<8}{100*np.mean(accs):6.2f}±{100*np.std(accs):4.2f}", flush=True)
            rows.append([name, m, "X", 1.0, "base", "-", "", "-", args.hetero_seeds, 100 * np.mean(accs), 100 * np.std(accs), "", "", 1.0])
        for ratio in args.hetero_ratios:
            for kind in args.hetero_F:
                for amode in (["-"] if kind == "feat" else list(alphas)):
                    alpha = 0.0 if amode == "-" else alphas[amode]
                    for mode in args.hash:
                        res = np.array([hetero_run(m, args, dev, kind, mode, ratio, s, hd, alpha, real_base)
                                        for s in range(args.hetero_seeds)])
                        mu, sd = res.mean(0), res.std(0)
                        print(fmt_row(ratio, kind, amode, alpha, mode, res), flush=True)
                        rows.append([name, m, "X", ratio, kind, amode, "" if amode == "-" else alpha, mode,
                                     args.hetero_seeds, 100 * mu[0], 100 * sd[0], "", 100 * mu[2], mu[3]])
            print(flush=True)


if __name__ == "__main__":
    main()
