"""
ahugc_lp_convmatch.py -- link prediction after coarsening with ConvMatch / A-ConvMatch (the authors' own code).

Identical protocol to ahugc_lp_shingle.py (it IMPORTS that file: same splits, same negatives, same GCN + MLP head,
same training loop, same seeds). The ONLY difference is where the partition `pi` (node -> super-node) comes from:
here it is produced by ConvMatch's own coarsener, run on the TRAIN graph of each split.

    [CM-paper] Dickens et al., "Graph Coarsening via Convolution Matching for Scalable GNN Training", arXiv:2312.15520
    [CM-code]  https://github.com/amazon-science/convolution-matching   (commit af523fb2fdf0f426424f71da6d2454e7543ca3dd)

SETUP (needs DGL, because the coarsener is DGL code):
    git clone https://github.com/amazon-science/convolution-matching
    pip install dgl pandas tqdm scikit-learn scipy        # DGL wheel must match your torch/CUDA, see dgl.ai/pages/start.html
    python ahugc_lp_convmatch.py --cm_repo ./convolution-matching --datasets cora citeseer --ratios 0.1 0.01 --splits 5 --split_mode fresh
Run the shingle script with the same --split_mode / --dropout / --splits to get the matching shingle rows (r=1.0 is identical, so it is skipped here).

What is taken from [CM-code]:
  * Experiments/utils/summarizer_config_utils.py : LinkPredictionConvolutionMatchingSGCNCoarsener /
    LinkPredictionApproximateConvolutionMatchingSGCNCoarsener, constructed as configure_graph_summarizer does:
    cls(dataset, train_graph, **params)  (the coarsener only stores `dataset`, never reads it, so None is passed)
  * Datasets/LinkPrediction/LinkPredictionDataset.py::_compute_normalized_edge_weights : how the train graph handed to the
    summarizer is built (edge_weight=1, self-loops, EdgeWeightNorm('both'))
  * Experiments/link_prediction_graph_summarization.py : summarizer.summarize(out_dir, seed=split); r values; the
    "Summarized graph is complete" skip
  * hyper-parameters: --cm_params paper -> [CM-paper] Table 7 (Cora LP: merge batch 10, PCA dim 15; Citeseer LP: batch 1,
    PCA dim 10; SGC-K 4, top-k_nn 1, d_nn 0.01 for both).  --cm_params yaml -> released Experiments/config/params.yaml
    (batch 1 for both; PCA dim Cora 10 / Citeseer 15).  num_nearest_neighbors=25 is not in Table 7; taken from params.yaml.
TWO SUMMARIES (important): ConvMatch's own summarized graph has the same nodes, edges, off-diagonal weights and features as the
summary induced by its partition, but SMALLER self-loop (diagonal) weights than size + #internal directed edges (observed on Cora
r=0.1: 192 of 269 super-nodes; consistent with its merge step adding 2 x a binary merge-graph weight per merge). ConvMatch's
reported numbers were trained on its own summary, so --summaries theirs reproduces its pipeline; --summaries canonical rebuilds
the summary from the partition with the same rule the shingle script uses, which is the like-for-like comparison. Default: both.

NOT identical to a run inside ConvMatch's repo: the order of edges in the train DGL graph (ours is sorted by construction,
theirs comes from DGL's Cora edge order after remove_edges) -- this can only change tie-breaking inside their coarsener.
"""
import argparse, csv, os, sys, tempfile, time, warnings
import numpy as np
import scipy.sparse as sp
import torch
import ahugc_lp_shingle as S          # the verified protocol: make_split, build_ops, train_one, load_graph, ...

CLASSES = {"convmatch": "LinkPredictionConvolutionMatchingSGCNCoarsener",
           "a-convmatch": "LinkPredictionApproximateConvolutionMatchingSGCNCoarsener"}
COMMON = dict(num_hops=4, num_nearest_neighbors=25, top_k_nn=1, nearest_neighbors_keep_rate=0.01)
PARAMS = {   # pairs_per_level = "merges per level k" = merge batch size
    "paper": {"cora": dict(pairs_per_level=10, knn_embedding_space_dim=15),      # [CM-paper] Table 7
              "citeseer": dict(pairs_per_level=1, knn_embedding_space_dim=10)},
    "yaml": {"cora": dict(pairs_per_level=1, knn_embedding_space_dim=10),         # [CM-code] params.yaml
             "citeseer": dict(pairs_per_level=1, knn_embedding_space_dim=15)},
}


def import_convmatch(repo):
    repo = os.path.abspath(repo)
    if not os.path.isdir(os.path.join(repo, "GraphSummarizers")):
        raise SystemExit(f"--cm_repo {repo} does not look like the convolution-matching repo (no GraphSummarizers/ folder)")
    sys.path.insert(0, repo)
    try:
        import importlib, dgl  # noqa
        C = importlib.import_module("GraphSummarizers.Coarsener.Coarsener")
        m1 = importlib.import_module("GraphSummarizers.Coarsener.LinkPrediction.LinkPredictionConvolutionMatchingSGCNCoarsener")
        m2 = importlib.import_module("GraphSummarizers.Coarsener.LinkPrediction.LinkPredictionApproximateConvolutionMatchingSGCNCoarsener")
    except Exception as e:
        raise SystemExit(f"Could not import DGL / the ConvMatch code: {type(e).__name__}: {e}\n"
                         "Install a DGL build that matches your torch (and CUDA) version, plus pandas, tqdm, scikit-learn, scipy.")
    # saving checkpoints/logs to disk is not needed here (the partition is read from the object) -> no-op, no effect on the result
    C.Coarsener._save_summarizer_state = lambda self, *a, **k: None
    return {"dgl": dgl, "init": C.edge_weight_initializer,
            "convmatch": m1.LinkPredictionConvolutionMatchingSGCNCoarsener,
            "a-convmatch": m2.LinkPredictionApproximateConvolutionMatchingSGCNCoarsener}


def cm_train_graph(cm, dir_edges, X, N):
    """SOURCE: [CM-code] LinkPredictionDataset._compute_normalized_edge_weights -- the 'Train Graph' given to the summarizer."""
    dgl = cm["dgl"]
    g = dgl.graph((dir_edges[:, 0].long(), dir_edges[:, 1].long()), num_nodes=N)
    g.ndata["feat"] = torch.from_numpy(X)
    g.edata["edge_weight"] = torch.ones(g.num_edges(), dtype=torch.float32)
    g.set_e_initializer(cm["init"], field="edge_weight")
    g = dgl.add_self_loop(g)
    g.edata["normalized_edge_weight"] = dgl.nn.pytorch.conv.EdgeWeightNorm(norm="both")(g, g.edata["edge_weight"])
    return g


def convmatch_partition(cm, variant, params, dir_edges, X, ratio, seed):
    """Run the authors' coarsener on the train graph. Returns (pi, their summarized graph)."""
    N = X.shape[0]
    g = cm_train_graph(cm, dir_edges, X, N)
    summarizer = cm[variant](None, g, r=ratio, **params)
    # DEVIATION (reproducibility only): [CM-code] seeds torch (summarize(seed=)) but sklearn's randomised PCA inside the
    # coarsener draws from numpy's global RNG, so two runs of their code differ. Seeding numpy/python with the split index
    # makes this script repeatable; it does not change what the coarsener computes.
    import random; random.seed(seed); np.random.seed(seed)
    with tempfile.TemporaryDirectory() as out_dir:
        summarizer.summarize(out_dir, seed=seed)                                   # same call as the experiment script
    pi = np.asarray(summarizer.original_to_supernode_id, dtype=np.int64)
    sg = summarizer.summarized_graph
    assert pi.shape[0] == N and pi.min() == 0 and pi.max() + 1 == sg.num_nodes() and len(np.unique(pi)) == sg.num_nodes(), \
        "partition returned by ConvMatch is not a contiguous node->super-node map"
    return pi, sg


def ops_from_cm_graph(sg, dev):
    """Training summary = ConvMatch's OWN summarized graph (their features, their normalised edge weights, every edge
    incl. self-loops is one positive) -- what [CM-code] trains on. Same Ops container as ahugc_lp_shingle.build_ops."""
    n = sg.num_nodes()
    s, d = [t_.numpy() for t_ in sg.edges()]
    w = sg.edata["normalized_edge_weight"].numpy().astype(np.float32)
    M = sp.coo_matrix((w, (d, s)), shape=(n, n))                                  # aggregation at dst: sum_src w_(src,dst) h_src
    X = sp.csr_matrix(sg.ndata["feat"].numpy())
    return S.Ops(S.to_torch_sparse(X, dev), S.to_torch_sparse(M, dev),
                 torch.from_numpy(s).long(), torch.from_numpy(d).long(), n)


def summary_matches(ops, sg, atol=1e-5):
    """Does the summary I build from `pi` equal ConvMatch's own summarized graph (edges, normalised weights, features)?"""
    if sg.num_nodes() != ops.n:
        return False
    s, d = [t.numpy() for t in sg.edges()]
    mine = set(zip(ops.src.cpu().tolist(), ops.dst.cpu().tolist()))
    if set(zip(s.tolist(), d.tolist())) != mine:
        return False
    Ahat = ops.M.to_dense().T.cpu().numpy()
    return bool(np.allclose(sg.edata["normalized_edge_weight"].numpy(), Ahat[s, d], atol=atol)
                and np.allclose(sg.ndata["feat"].numpy(), ops.x.to_dense().cpu().numpy(), atol=atol))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cm_repo", required=True, help="path to the cloned convolution-matching repo")
    ap.add_argument("--variants", nargs="+", default=["a-convmatch", "convmatch"], choices=list(CLASSES))
    ap.add_argument("--cm_params", choices=["paper", "yaml"], default="paper")
    ap.add_argument("--summaries", nargs="+", default=["theirs", "canonical"], choices=["theirs", "canonical"],
                    help="theirs: train on ConvMatch's own summarized graph (the exact ConvMatch pipeline). canonical: rebuild the summary "
                         "from the partition with the SAME rule as ahugc_lp_shingle.py (use this one for a like-for-like comparison with shingle).")
    ap.add_argument("--datasets", nargs="+", default=["cora", "citeseer"], choices=["cora", "citeseer"])
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.1, 0.01])
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=S.EPOCHS)
    ap.add_argument("--dropout", choices=["paper", "yaml"], default="paper")
    ap.add_argument("--split_mode", choices=["convmatch", "fresh"], default="convmatch")
    ap.add_argument("--model_seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="results_lp_convmatch.csv")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    cm = import_convmatch(args.cm_repo)
    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    drop = S.DROPOUT_PAPER if args.dropout == "paper" else S.DROPOUT_YAML
    print(f"device={dev} variants={args.variants} cm_params={args.cm_params} dropout={args.dropout} "
          f"split_mode={args.split_mode} epochs={args.epochs} splits={args.splits}", flush=True)

    new_file = not os.path.exists(args.out)
    fout = open(args.out, "a", newline=""); w = csv.writer(fout)
    if new_file:
        w.writerow(["dataset", "ratio", "split", "hash", "dropout", "split_mode", "n_super", "summary_edges_incl_loops", "status",
                    "val_auc", "test_auc", "test_edges_inside_one_supernode", "coarsen_s", "train_s",
                    "realized_ratio", "canonical_summary_equals_convmatch_summary", "cm_params"])
    rows = {}
    for name in args.datasets:
        X, ei = S.load_graph(name, args.root)
        N = X.shape[0]
        for split in range(args.splits):
            sd = S.make_split(ei, N, split, args.split_mode)                    # identical to the shingle run for the same split index
            args.val_pos, args.val_neg = sd["val_pos"].to(dev), sd["val_neg"].to(dev)
            args.test_pos, args.test_neg = sd["test_pos"].to(dev), sd["test_neg"].to(dev)
            ident = np.arange(N)
            val_ops, _ = S.build_ops(ident, sd["train_dir"], X, dev)
            test_ops, _ = S.build_ops(ident, torch.cat([sd["train_dir"], sd["val_dir"]]), X, dev)
            for variant in args.variants:
                for ratio in args.ratios:
                    status, val_auc, test_auc, train_s, matches, n, E, inside = "ok", float("nan"), float("nan"), 0.0, "", 0, 0, float("nan")
                    t0 = time.time()
                    try:
                        pi, sg = convmatch_partition(cm, variant, {**COMMON, **PARAMS[args.cm_params][name]}, sd["train_dir"], X, ratio, split)
                    except AssertionError as err:                                # as in the experiment script: report and continue
                        print(f"{name} {variant} r={ratio} split={split}: ConvMatch assertion failed: {err}", flush=True)
                        status = "coarsener_assertion_failed"; pi = None
                    t_coarsen = time.time() - t0
                    modes = [] if pi is None else args.summaries
                    if pi is not None:
                        canon, _ = S.build_ops(pi, sd["train_dir"], X, dev)       # partition-induced summary (same rule as the shingle script)
                        same = summary_matches(canon, sg)
                        n = canon.n
                        inside = float((pi[sd["test_pos"][:, 0].numpy()] == pi[sd["test_pos"][:, 1].numpy()]).mean())
                    for mode in (modes or [None]):
                        label = variant if mode in (None, "theirs") else f"{variant}|canonical-summary"
                        if pi is not None:
                            status, val_auc, test_auc, train_s = "ok", float("nan"), float("nan"), 0.0
                            summary = ops_from_cm_graph(sg, dev) if mode == "theirs" else canon
                            E = summary.src.numel()
                            if E >= n * (n - 1):                                    # [CM-code] "Summarized graph is complete." -> skip
                                status = "skipped_complete_summary"
                            else:
                                t1 = time.time()
                                val_auc, test_auc = S.train_one(summary, val_ops, test_ops, split, drop[name], args, dev)
                                train_s, status = time.time() - t1, "ok"
                        print(f"{name:9s} {label:28s} r={ratio:<5} split={split} n={n:5d} (n/N={n / N:.3f}) E={E:6d} {status:26s} "
                              f"val={val_auc:.4f} test={test_auc:.4f} intra={inside:.3f} canonical==theirs:{matches if pi is None else same} "
                              f"coarsen={t_coarsen:.0f}s train={train_s:.0f}s", flush=True)
                        w.writerow([name, ratio, split, label, args.dropout, args.split_mode, n, E, status, f"{val_auc:.4f}", f"{test_auc:.4f}",
                                    f"{inside:.4f}", f"{t_coarsen:.2f}", f"{train_s:.1f}", f"{n / N:.4f}", "" if pi is None else same, args.cm_params]); fout.flush()
                        rows.setdefault((name, label, ratio), []).append(test_auc)
    fout.close()
    print("\nAUC (%)  mean±std over splits")
    for (name, variant, ratio), v in rows.items():
        v = np.array(v) * 100; ok = v[~np.isnan(v)]
        print(f"  {name:9s} {variant:11s} r={ratio:<5} " + (f"{ok.mean():6.2f} ± {ok.std():.2f}  (n={len(ok)}/{len(v)})" if len(ok) else "no valid runs"))


if __name__ == "__main__":
    main()
