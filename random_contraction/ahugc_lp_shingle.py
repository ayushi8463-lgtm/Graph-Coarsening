"""
ahugc_lp_shingle.py -- link prediction after graph coarsening, ConvMatch protocol + shingle (MinHash) coarsener.

EVERYTHING EXCEPT THE COARSENER follows the ConvMatch link-prediction protocol:
    [CM-paper] C. Dickens, E. Huang, A. Reganti, J. Zhu, K. Subbian, D. Koutra.
               "Graph Coarsening via Convolution Matching for Scalable Graph Neural Network Training", 2024.
               arXiv:2312.15520  (Sec. 5, Table 4a, Table 6 = GCN hyper-parameters)
    [CM-code]  https://github.com/amazon-science/convolution-matching   commit af523fb2fdf0f426424f71da6d2454e7543ca3dd
The only part that is NOT ConvMatch is the partition function (`shingle_partition`), which is the
neighbourhood-MinHash ("shingle") order + random merge of right neighbours, copied verbatim from
ahugc_eval_shingle.py (`minhash`, `graph_hash_order('shingle')`, `coarsen`).

Every block below carries a  SOURCE:  tag naming the ConvMatch file/function it reproduces and, where this file
cannot be literally identical (PyG instead of DGL, etc.), a  DEVIATION:  tag saying exactly how.

Pipeline (one "split" = one row of results), all as in [CM-code] Experiments/link_prediction_graph_summarization.py:
  1. split the undirected edges 80/10/10, sample 1 negative per held-out edge          (LinkPredictionDataset._create_splits)
  2. coarsen the TRAIN graph only (val+test edges removed)                              (summarizer.summarize(train_graph))
  3. build the summary exactly as Coarsener._to_simple_graph does
  4. train the 2-layer GCN + MLP head on the SUMMARY (edge-prediction loss, 1 neg per pos, batch 512)
  5. every epoch score validation edges on the ORIGINAL train graph, keep the best-val checkpoint (Trainer.train)
  6. test: run the best checkpoint on the ORIGINAL graph (train+val edges), AUC on held-out test edges

  python ahugc_lp_shingle.py --datasets cora citeseer --ratios 1.0 0.1 0.01 --splits 5
Requires: torch, numpy, scipy, scikit-learn, torch_geometric (only for the Planetoid loader).
"""
import argparse, copy, csv, itertools, os, time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as Fn
from sklearn.metrics import roc_auc_score

# ----------------------------------------------------------------------------------------------- constants
# SOURCE: [CM-code] Datasets/LinkPrediction/LinkPredictionDataset.py : OUTPUT_DIM, TEST_PROPORTION, NUM_NEGATIVE_EXAMPLES
OUTPUT_DIM = 256
TEST_PROPORTION = 0.1
NUM_NEGATIVE_EXAMPLES = 1
# SOURCE: [CM-code] Datasets/LinkPrediction/{Cora,Citeseer}.py : self.epochs = 250 ; Datasets/Dataset.py : compute_period = 1
EPOCHS = 250
# SOURCE: [CM-code] LinkPredictionDataset.get_training_data_loader : batch_size=512, shuffle=True, drop_last=False
BATCH_SIZE = 512
# SOURCE: [CM-code] Experiments/config/params.yaml : CoraLinkPredictionGraphSummarization / CiteseerLinkPredictionGraphSummarization
#   GCN: n_layers 2, hidden 256, use_layer_norm False, use_edge_weights True, use_input_layer False
#   MLPPredictionHead: hidden 256, use_layer_norm False ; ADAM lr 0.01, weight_decay 0.0 ; loss BinaryCrossEntropy
#   dropout: Citeseer 0.5 ; Cora 0.0 in the released yaml BUT 0.5 in [CM-paper] Table 6  -> see --dropout
HIDDEN, N_LAYERS, LR, WD, HEAD_HIDDEN = 256, 2, 1e-2, 0.0, 256
DROPOUT_YAML = {"cora": 0.0, "citeseer": 0.5}      # [CM-code] params.yaml
DROPOUT_PAPER = {"cora": 0.5, "citeseer": 0.5}     # [CM-paper] Table 6 ("Cora LP / Citeseer LP: dropout 0.5")
# SOURCE: [CM-code] Trainers/LinkPrediction/LinkPredictionTrainer.post_gradient_computation : clip_grad_norm_(.., 1.0)
GRAD_CLIP = 1.0


# ----------------------------------------------------------------------------------------------- data
def load_graph(name, root="data"):
    """DEVIATION: [CM-code] uses dgl.data.CoraGraphDataset()/CiteseerGraphDataset(); this uses PyG Planetoid, which is
    the same Planetoid graph. DGL row-normalises the features (Kipf preprocessing), reproduced with NormalizeFeatures.
    PyG's Citeseer has 9104 directed edges (no self-loops/duplicates); [CM-code] removes self-loops before splitting
    (`dgl.remove_self_loop` in _create_splits), so the edge set used for splitting is the same (not run side by side)."""
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T
    d = Planetoid(root, {"cora": "Cora", "citeseer": "Citeseer"}[name], transform=T.NormalizeFeatures())[0]
    return d.x.numpy().astype(np.float32), d.edge_index.numpy()


def edge_keys(u, v, N):
    return u.long() * N + v.long()


def sample_nonedges(forbidden_keys, N, k, gen, max_rounds=30):
    """k distinct directed pairs (u != v) that are not in `forbidden_keys`, uniform over such pairs.
    SOURCE: dgl.sampling.global_uniform_negative_sampling(g, k) with exclude_self_loops=True, replace=False, which is what
    [CM-code] calls for the val/test negatives (LinkPredictionDataset._create_splits) and, through
    dgl.dataloading.negative_sampler.GlobalUniform(1), for the training negatives.
    DEVIATION: re-implemented by rejection sampling; like DGL it returns FEWER than k pairs if k non-edges are not found."""
    chosen = torch.empty(0, dtype=torch.long)
    for _ in range(max_rounds):
        m = max(2 * (k - chosen.numel()), 256)
        u = torch.randint(N, (m,), generator=gen)
        v = torch.randint(N, (m,), generator=gen)
        key = edge_keys(u, v, N)
        ok = (u != v) & ~torch.isin(key, forbidden_keys)
        chosen = torch.unique(torch.cat([chosen, key[ok]]))
        if chosen.numel() >= k:
            break
    chosen = chosen[torch.randperm(chosen.numel(), generator=gen)[:k]]
    return torch.stack([chosen // N, chosen % N], 1)


def make_split(edge_index, N, split_idx, split_mode):
    """SOURCE: [CM-code] LinkPredictionDataset._create_splits (and Cora.py / Citeseer.py for the graphs).
    * unique undirected edges, sorted; train/val/test sizes int(0.8U)/int(0.1U)/int(0.1U); the TEST set takes the whole
      remainder of the permutation (so it can be 1-2 edges larger than val), and the number of negatives for BOTH val
      and test is `test_size` = int(0.1U)   (copied literally from the code).
    * positives are used in both directions; negatives are directed pairs not in the FULL graph.
    * train graph = all edges minus val and test; validation graph = the train graph; test graph = train + val edges.
    split_mode:
      'convmatch' : the code calls torch.manual_seed(0) INSIDE the per-split loop, so all of ConvMatch's "splits" share
                    the same positive-edge split (the comment there says the seed was meant to be the split number).
                    Reproduced literally. Negatives come from DGL's own RNG in ConvMatch (not controlled by torch's
                    seed); here they are seeded by the split index.
      'fresh'     : positive split AND negatives seeded by the split index (what the comment intends)."""
    src, dst = torch.from_numpy(edge_index[0]), torch.from_numpy(edge_index[1])
    keep = src != dst                                                       # dgl.remove_self_loop
    e = torch.sort(torch.stack([src[keep], dst[keep]], 1), dim=1)[0]
    unique_edges = torch.unique(e, dim=0)
    U = unique_edges.shape[0]
    train_size = int((1.0 - 2 * TEST_PROPORTION) * U)
    val_size = int(TEST_PROPORTION * U)
    test_size = int(TEST_PROPORTION * U)

    g_pos = torch.Generator().manual_seed(0 if split_mode == "convmatch" else split_idx)
    perm = torch.randperm(U, generator=g_pos)
    tr, va, te = perm[:train_size], perm[train_size:train_size + val_size], perm[train_size + val_size:]

    def both(ix):
        a, b = unique_edges[ix, 0], unique_edges[ix, 1]
        return torch.stack([torch.cat([a, b]), torch.cat([b, a])], 1)       # (u,v) and (v,u), as in the source

    full_dir = both(torch.arange(U))                                        # full graph, both directions
    forbidden = torch.sort(edge_keys(full_dir[:, 0], full_dir[:, 1], N))[0]
    g_neg = torch.Generator().manual_seed(10_000 + split_idx)
    val_neg = sample_nonedges(forbidden, N, NUM_NEGATIVE_EXAMPLES * test_size, g_neg)
    test_neg = sample_nonedges(forbidden, N, NUM_NEGATIVE_EXAMPLES * test_size, g_neg)
    return {"train_dir": both(tr), "val_dir": both(va),
            "val_pos": both(va), "val_neg": val_neg, "test_pos": both(te), "test_neg": test_neg}


# ----------------------------------------------------------------------------------------------- coarsening (the ONLY non-ConvMatch part)
def minhash(A, rng, nh):
    """VERBATIM from ahugc_eval_shingle.py: neighbourhood MinHash, sig_k(v) = min over closed neighbourhood N[v] of a random rank."""
    M = (sp.csr_matrix(A) + sp.eye(A.shape[0])).tocsr()
    out = []
    for _ in range(nh):
        r = rng.permutation(A.shape[0]).astype(np.int64)
        out.append(np.minimum.reduceat(r[M.indices], M.indptr[:-1]))
    return out


def shingle_partition(A, ratio, seed, shingle_hashes=3):
    """VERBATIM logic of ahugc_eval_shingle.py `graph_hash_order('shingle')` + `coarsen`: sort nodes lexicographically by
    (sig_1, sig_2, sig_3), then repeatedly pick a random super-node and merge it with its right neighbour until
    n = round(ratio*N). Uses ONLY the adjacency `A` (here: the TRAIN graph). Returns pi: node -> super-node id."""
    rng = np.random.default_rng(seed)
    N = A.shape[0]
    s = minhash(A, rng, shingle_hashes)
    order = np.lexsort(tuple(reversed(s)))
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


# ----------------------------------------------------------------------------------------------- summary graph
class Ops:
    """features + normalised adjacency of one graph, as torch tensors on `dev`."""
    def __init__(self, x, M, src, dst, n):
        self.x, self.M, self.src, self.dst, self.n = x, M, src, dst, n


def to_torch_sparse(m, dev):
    m = sp.coo_matrix(m)
    idx = torch.from_numpy(np.vstack([m.row, m.col])).long()
    return torch.sparse_coo_tensor(idx, torch.from_numpy(m.data.astype(np.float32)), m.shape).coalesce().to(dev)


def build_ops(pi, dir_edges, X, dev):
    """SOURCE: [CM-code] GraphSummarizers/Coarsener/Coarsener.py : _to_simple_graph (+ Coarsener._merge_nodes_in_graph).
    For a partition pi (node -> super-node) of the graph with directed edge list `dir_edges`:
      * super-node features  = mean of member features                       (_compute_super_node_features, size-weighted mean)
      * off-diagonal weight  = number of directed edges between members      (summed edge_weight)
      * diagonal weight      = |C_i| + (#directed edges inside C_i)          (add_self_loop weight 1 + explicit self loop of
                                                                              size + self_loop_weight - 1, summed by to_simple)
      * normalised weight    = w_ij / sqrt(rowsum_i * colsum_j)              (dgl EdgeWeightNorm(norm='both'))
    With pi = identity this is exactly _compute_normalized_edge_weights of the ORIGINAL graphs: D^-1/2 (A+I) D^-1/2.
    The training summary additionally keeps every directed edge incl. self-loops as one positive example
    (get_training_data_loader iterates over torch.arange(graph.number_of_edges()))."""
    N = X.shape[0]
    n = int(pi.max()) + 1
    size = np.bincount(pi, minlength=n).astype(np.float64)
    s, d = dir_edges[:, 0].numpy(), dir_edges[:, 1].numpy()
    W = sp.coo_matrix((np.ones(len(s)), (pi[s], pi[d])), shape=(n, n)).tocsr()   # duplicates summed; diag = internal edges
    off = W - sp.diags(W.diagonal())
    off.eliminate_zeros()
    W = (W + sp.diags(size)).tocsr()                                              # A' + C
    rs, cs = np.asarray(W.sum(1)).ravel(), np.asarray(W.sum(0)).ravel()
    Ahat = sp.diags(rs ** -0.5) @ W @ sp.diags(cs ** -0.5)                        # entry (i->j)
    M = Ahat.T.tocsr()                                                            # aggregation at dst j:  sum_i Ahat_ij h_i
    P = sp.coo_matrix((np.ones(N), (np.arange(N), pi)), shape=(N, n)).tocsr()
    Xc = sp.diags(1.0 / size) @ (P.T @ sp.csr_matrix(X))                          # mean of member rows
    Wc = sp.coo_matrix(W)                                                         # every edge incl. self loops = one positive
    summary_edges = torch.from_numpy(np.vstack([Wc.row, Wc.col]).T).long()
    return Ops(to_torch_sparse(Xc, dev), to_torch_sparse(M, dev), summary_edges[:, 0], summary_edges[:, 1], n), off.nnz


# ----------------------------------------------------------------------------------------------- models
class GCN2(torch.nn.Module):
    """SOURCE: [CM-code] Models/GNNs/GCN.py  (GCNLayer = dgl GraphConv(norm='none') with edge_weight = normalized_edge_weight;
    n_layers=2, use_layer_norm=False, use_input_layer=False). Layer 0: ReLU then dropout; layer 1 (output): linear, no dropout.
    DGL GraphConv init: xavier_uniform_ weight, zero bias.
    DEVIATION: pure torch (sparse matmul with the pre-normalised weights) instead of DGL message passing; the same
    function. ConvMatch evaluates a mini-batch on sampled blocks that carry the full-graph normalised weights; with
    full-neighbour sampling that equals running the full graph and reading out the seed nodes, which is what is done here
    (dropout masks are therefore drawn for all nodes instead of only the block's source nodes -- same distribution)."""
    def __init__(self, d_in, hidden, d_out, dropout):
        super().__init__()
        self.W1 = torch.nn.Parameter(torch.empty(d_in, hidden)); self.b1 = torch.nn.Parameter(torch.zeros(hidden))
        self.W2 = torch.nn.Parameter(torch.empty(hidden, d_out)); self.b2 = torch.nn.Parameter(torch.zeros(d_out))
        torch.nn.init.xavier_uniform_(self.W1); torch.nn.init.xavier_uniform_(self.W2)
        self.dropout = dropout

    def forward(self, ops):
        h = torch.sparse.mm(ops.x, self.W1)
        h = Fn.relu(torch.sparse.mm(ops.M, h) + self.b1)
        h = Fn.dropout(h, p=self.dropout, training=self.training)
        return torch.sparse.mm(ops.M, h @ self.W2) + self.b2


class MLPHead(torch.nn.Module):
    """SOURCE: [CM-code] Models/LinkPredictionHeads/MLPPredictionHead.py (use_layer_norm=False):
    score(u,v) = sigmoid(W3 relu(W2 relu(W1 (h_u * h_v))))  -- Hadamard product, 2 hidden layers, sigmoid output."""
    def __init__(self, d, hidden):
        super().__init__()
        self.W1, self.W2, self.W3 = (torch.nn.Linear(d, hidden), torch.nn.Linear(hidden, hidden), torch.nn.Linear(hidden, 1))

    def forward(self, hu, hv):
        z = hu * hv
        return torch.sigmoid(self.W3(Fn.relu(self.W2(Fn.relu(self.W1(z))))).squeeze(1))


def bce(pos, neg):
    """SOURCE: [CM-code] Trainers/Losses/link_prediction.py : binary_cross_entropy (scores clipped to [0,1], mean reduction)."""
    scores = torch.clip(torch.cat([pos, neg]), min=0.0, max=1.0)
    labels = torch.cat([torch.ones(pos.shape[0]), torch.zeros(neg.shape[0])]).to(scores.device)
    return Fn.binary_cross_entropy(scores, labels, reduction="mean")


@torch.no_grad()
def evaluate(model, head, ops, pos, neg):
    """SOURCE: [CM-code] Models/utils.py : link_prediction_evaluate_full + Analysis/evaluators.py : link_prediction_auc
    (eval mode, original graph, all nodes with their own features, AUC over pos-then-neg scores)."""
    model.eval(); head.eval()
    H = model(ops)
    sp_ = head(H[pos[:, 0]], H[pos[:, 1]]).cpu()
    sn_ = head(H[neg[:, 0]], H[neg[:, 1]]).cpu()
    labels = torch.cat([torch.ones(len(sp_)), torch.zeros(len(sn_))])
    return float(roc_auc_score(labels.numpy(), torch.cat([sp_, sn_]).numpy()))


def train_one(summary, val_ops, test_ops, split, dropout, args, dev):
    """SOURCE: [CM-code] Trainers/Trainer.py : train  +  LinkPredictionTrainer.compute_training_loss.
      * model init seed 0, then head init seed 0   (Experiments/utils/model_config_utils.py: configure_model / configure_prediction_head, seed=0)
      * Adam over model+head parameters
      * per epoch: shuffled batches of 512 summary edges (self-loops included), 1 uniform negative per positive drawn
        from the NON-edges of the SUMMARY graph, BCE, backward, clip_grad_norm_ 1.0 on model and head separately, step
      * every epoch: AUC on the validation edges using the ORIGINAL train graph; the checkpoint is saved when
        best_validation_score <= validation_score  (ties go to the later epoch, as in the source)
      * after training: load the best checkpoint, AUC on the test edges using the ORIGINAL train+val graph."""
    torch.manual_seed(args.model_seed)
    model = GCN2(summary.x.shape[1], HIDDEN, OUTPUT_DIM, dropout).to(dev)
    torch.manual_seed(args.model_seed)
    head = MLPHead(OUTPUT_DIM, HEAD_HIDDEN).to(dev)
    opt = torch.optim.Adam(itertools.chain(model.parameters(), head.parameters()), lr=LR, weight_decay=WD)

    E = summary.src.numel()
    pos_all = torch.stack([summary.src, summary.dst], 1)
    forbidden = torch.sort(edge_keys(summary.src, summary.dst, summary.n))[0]
    g_neg = torch.Generator().manual_seed(20_000 + split)
    best_val, best_state = -1.0, None
    for epoch in range(args.epochs):
        perm = torch.randperm(E)
        for b in range(0, E, BATCH_SIZE):
            pe = pos_all[perm[b:b + BATCH_SIZE]]
            ne = sample_nonedges(forbidden, summary.n, NUM_NEGATIVE_EXAMPLES * len(pe), g_neg)
            opt.zero_grad(set_to_none=True)
            model.train(); head.train()
            H = model(summary)
            pe, ne = pe.to(dev), ne.to(dev)
            loss = bce(head(H[pe[:, 0]], H[pe[:, 1]]), head(H[ne[:, 0]], H[ne[:, 1]]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            torch.nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP)
            opt.step()
        val = evaluate(model, head, val_ops, args.val_pos, args.val_neg)
        if best_val <= val:
            best_val = val
            best_state = (copy.deepcopy(model.state_dict()), copy.deepcopy(head.state_dict()))
        if args.verbose and (epoch % 25 == 0 or epoch == args.epochs - 1):
            print(f"      epoch {epoch:3d} loss {loss.item():.4f} val_auc {val:.4f} best {best_val:.4f}", flush=True)
    model.load_state_dict(best_state[0], strict=True); head.load_state_dict(best_state[1], strict=True)
    return best_val, evaluate(model, head, test_ops, args.test_pos, args.test_neg)


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cora", "citeseer"], choices=["cora", "citeseer"])
    ap.add_argument("--ratios", nargs="+", type=float, default=[1.0, 0.1, 0.01],
                    help="[CM-paper] Table 4a: 1%%, 10%%, 100%% (100%% = uncoarsened reference)")
    ap.add_argument("--splits", type=int, default=5, help="[CM-code] run_config.yaml: splits: 5")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--dropout", choices=["paper", "yaml"], default="paper",
                    help="paper: 0.5 for both ([CM-paper] Table 6). yaml: Cora 0.0, Citeseer 0.5 (released params.yaml).")
    ap.add_argument("--split_mode", choices=["convmatch", "fresh"], default="convmatch",
                    help="convmatch: positive split identical for every split index (literal reading of the source). fresh: new split per index.")
    ap.add_argument("--shingle_hashes", type=int, default=3)
    ap.add_argument("--model_seed", type=int, default=0, help="[CM-code] configure_model(seed=0)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="results_lp_shingle.csv")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    drop = DROPOUT_PAPER if args.dropout == "paper" else DROPOUT_YAML
    print(f"device={dev} dropout={args.dropout} split_mode={args.split_mode} epochs={args.epochs} splits={args.splits}", flush=True)

    new_file = not os.path.exists(args.out)
    fout = open(args.out, "a", newline=""); w = csv.writer(fout)
    if new_file:
        w.writerow(["dataset", "ratio", "split", "hash", "dropout", "split_mode", "n_super", "summary_edges_incl_loops",
                    "status", "val_auc", "test_auc", "test_edges_inside_one_supernode", "coarsen_s", "train_s"])
    summary_rows = {}
    for name in args.datasets:
        X, ei = load_graph(name, args.root)
        N = X.shape[0]
        for split in range(args.splits):
            sd = make_split(ei, N, split, args.split_mode)
            args.val_pos, args.val_neg = sd["val_pos"].to(dev), sd["val_neg"].to(dev)
            args.test_pos, args.test_neg = sd["test_pos"].to(dev), sd["test_neg"].to(dev)
            # original graphs used for validation / test inference ([CM-code] Validation Graph / Test Graph)
            ident = np.arange(N)
            val_ops, _ = build_ops(ident, sd["train_dir"], X, dev)                                        # train edges only
            test_ops, _ = build_ops(ident, torch.cat([sd["train_dir"], sd["val_dir"]]), X, dev)          # train + val edges
            A_train = sp.coo_matrix((np.ones(len(sd["train_dir"])), (sd["train_dir"][:, 0].numpy(), sd["train_dir"][:, 1].numpy())),
                                    shape=(N, N)).tocsr()
            for ratio in args.ratios:
                t0 = time.time()
                # ---- coarsen the TRAIN graph only ([CM-code] summarizer.summarize(dataset_split["Train Graph"], seed=split))
                pi = ident if ratio >= 1.0 else shingle_partition(A_train, ratio, split, args.shingle_hashes)
                t_coarsen = time.time() - t0
                summary, _ = build_ops(pi, sd["train_dir"], X, dev)
                n, E = summary.n, summary.src.numel()
                # SOURCE: [CM-code] link_prediction_graph_summarization.py: assert E < n(n-1) ("Summarized graph is complete") -> skip
                if E >= n * (n - 1) and ratio < 1.0:
                    status, val_auc, test_auc, train_s = "skipped_complete_summary", float("nan"), float("nan"), 0.0
                else:
                    t1 = time.time()
                    val_auc, test_auc = train_one(summary, val_ops, test_ops, split, drop[name], args, dev)
                    status, train_s = "ok", time.time() - t1
                inside = float((pi[sd["test_pos"][:, 0].numpy()] == pi[sd["test_pos"][:, 1].numpy()]).mean())
                print(f"{name:9s} r={ratio:<5} split={split} n={n:5d} E={E:6d} {status:24s} val={val_auc:.4f} test={test_auc:.4f} "
                      f"intra-supernode test edges={inside:.3f} coarsen={t_coarsen:.1f}s train={train_s:.0f}s", flush=True)
                w.writerow([name, ratio, split, "none" if ratio >= 1.0 else "shingle", args.dropout, args.split_mode, n, E, status,
                            f"{val_auc:.4f}", f"{test_auc:.4f}", f"{inside:.4f}", f"{t_coarsen:.2f}", f"{train_s:.1f}"]); fout.flush()
                summary_rows.setdefault((name, ratio), []).append(test_auc)
    fout.close()
    print("\nAUC (%)  mean±std over splits   [CM-paper Table 4a reports mean±std over its 5 splits]")
    for (name, ratio), v in summary_rows.items():
        v = np.array(v) * 100
        ok = v[~np.isnan(v)]
        print(f"  {name:9s} r={ratio:<5} " + (f"{ok.mean():6.2f} ± {ok.std():.2f}  (n={len(ok)}/{len(v)})" if len(ok) else "all skipped (complete summary)"))


if __name__ == "__main__":
    main()
