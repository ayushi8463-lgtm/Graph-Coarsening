# Local results compared with AH-UGC

All accuracy differences are percentage points (pp): local accuracy minus the AH-UGC value reported in the paper. The 50% comparisons use the AH-UGC column from the paper's node-classification tables. Results from the local 10-seed runs are shown as mean ± standard deviation; paper values are point estimates.

## What the shingle coarsener does

The shingle variant builds its ordering from graph structure rather than node features. For each node it takes the **closed neighborhood** (the node itself plus its neighbors), assigns every graph node a random rank, and records the smallest rank appearing in that neighborhood. Repeating this with independent random rankings gives a short MinHash signature. We sort nodes lexicographically by these signatures, then repeatedly merge a randomly selected node group with the next group in the order (wrapping at the end) until the requested number of supernodes is reached. The coarse graph sums the original edges between groups; features are averaged, and the classifier's training labels are majority votes over labeled training members.

**Small example.** Suppose two nodes have closed neighborhoods `{a,b,c}` and `{a,b,d}`. Their overlap is `{a,b}` and their union is `{a,b,c,d}`, so their Jaccard similarity is 2/4 = 0.5. Under one uniformly random ranking, the minimum-ranked member is equally likely to come from any element of the union. The two neighborhood signatures match exactly when that minimum is `a` or `b`, hence with probability 0.5. Multiple independent minima make a compact fingerprint: similar neighborhoods are more likely to sort near one another, while unrelated neighborhoods are less likely to do so. This is a similarity cue, not a guarantee that every merged pair is structurally equivalent.

**Brief history and provenance.** MinHash was introduced by Andrei Broder in 1997 for estimating resemblance between document sets, including near-duplicate web-page detection ([Broder, *On the Resemblance and Containment of Documents*](https://doi.org/10.1109/SEQUEN.1997.666900)). Locality-sensitive hashing was developed as an algorithmic framework for approximate nearest-neighbor search soon after ([Indyk and Motwani, STOC 1998](https://doi.org/10.1145/276698.276876)). Here, a graph neighborhood is treated as a set, and MinHash supplies a lightweight structural ordering for graph coarsening. Our feature-hash/random-merge baseline is inspired by the [AH-UGC paper](https://arxiv.org/abs/2505.15842); the neighborhood-MinHash ordering is our alternative structural signal.

**Novelty assessment.** The shingle hash and the random right-merge coarsener are each established techniques. Closed-neighborhood MinHash (“shingle”) and its use for grouping or ordering graph nodes appear in SWeG (2019), SLUGGER (2022), and Chierichetti et al. (2009). AH-UGC sorts nodes by a one-dimensional hash and repeatedly merges a randomly selected group with its right neighbor. Our method uses the shingle ordering in this AH-UGC-style coarsener, replacing AH-UGC’s random-projection ordering with a graph-structure-based ordering ([SWeG](https://www.cs.cmu.edu/~kijungs/papers/swegWWW2019.pdf), [SLUGGER](https://arxiv.org/abs/2112.05374), [Chierichetti et al.](https://www.eecs.harvard.edu/~michaelm/postscripts/kdd2009.pdf), [AH-UGC](https://arxiv.org/abs/2505.15842)).

**To the best of our knowledge, this is the first graph-coarsening pipeline to order nodes by neighborhood MinHash (shingle) and merge them using AH-UGC’s random right-neighbor rule.** The cited methods cover the ingredients and the closest neighboring approaches, but our literature search found no prior work with this combination. This is a claim of likely novelty as a combination, not a new hash or a wholly new coarsening framework. The search remains incomplete: forward citations of UGC, AH-UGC, and SWeG were not checked, and the UGC TPAMI extension was only available at abstract level. A broader review could change the priority assessment. Novelty also does not by itself establish significance; the corrected, like-for-like experiments must show whether this combination provides a useful contribution.

## How comparable are the paper numbers?

The local **Shingle vs Random** comparisons are the cleanest ones: each pair uses the same local graph, feature input, split seeds, classifier, training procedure, and target ratio; the coarsening order is the main changed component. Comparing either local result to the AH-UGC paper is less controlled. The paper reports point estimates, while our homogeneous runs average ten split seeds. AH-UGC does not spell out all split and preprocessing choices in its paper, so matching dataset names, model names, and ratios does not guarantee matching experiments.

| Results in this writeup | What matches | Known or unresolved differences from AH-UGC |
|---|---|---|
| Cora, CiteSeer, PubMed | Same named datasets, model families, 50% ratio, paper alpha, and augmented `orig` classifier input. | Our runs use ten fixed Geom-GCN 60/20/20 masks. AH-UGC publishes one value and does not state its exact masks in the paper; the cited [UGC code](https://github.com/katariaMohit/UGC-Universal-Graph-Coarsening/blob/main/UGC.py) has a separate random 60/20/20 split path. Thus split identity and variance are not matched. |
| Cornell, Chameleon, Film | Same named datasets, reported models, 50% ratio, and paper alpha. | Our evaluator uses Geom-GCN masks and locally loaded/preprocessed graph data; the AH-UGC paper does not provide split IDs or enough preprocessing detail to verify identical inputs. Film is a concrete warning: the local benchmark has 29,926 stored edge entries, versus 33,544 edges in the paper's dataset table. Chameleon is closer (36,051 vs 36,101), but that alone does not prove identical preprocessing. SGC/MixHop are fixed local configurations; the cited heterophilic-model setup includes model selection/tuning, so exact hyperparameter equality is unverified. |
| Reed98, Amherst41 | Same named Facebook100 graphs, MixHop model family, and 50% ratio; local node/edge/feature counts match the paper's dataset table. | Local labels are gender with unlabeled nodes excluded, and local splits are random 60/20/20 over labeled nodes. The AH-UGC table lists three classes and gives no split details; the cited LINKX protocol uses its own Facebook100 split setup. Local alpha is also computed while the paper lists no alpha for these datasets. These are not confirmed like-for-like. The differing uncoarsened baselines (local 67.9/81.5 vs paper 62.43/72.48) reinforce that protocol or preprocessing differs. |
| ACM, heterogeneous | The local graph's node and typed-edge counts match AH-UGC Table 6; local uses the same target ratio and named HGCond model families. The ACM split follows the 200 train / 200 validation / remaining test convention. | The local result is one run, so it has no seed-variation estimate. The paper does not publish the exact split indices or full per-run setup, so exact split/model-seed matching remains unverified. Treat the values as a close reproduction attempt, not a strict replication. |
| DBLP, heterogeneous | Same named target task, model families, and 30% ratio. The local typed-edge count matches the relation counts in AH-UGC Table 6. | The paper's DBLP node-count column appears inconsistent: it lists 4,231 papers and 50 conferences, while its own relation counts imply 14,328 paper-conference links and the local PyG/HAN loader has 14,328 papers and 20 conferences. The listed 4,231 and 50 instead match paper/term feature dimensions. The table may contain a reporting error, but without the authors' exact data files we cannot verify the graph variant. Local results are one seed. |

The runtime comparison is also **not like-for-like**: AH-UGC reports total time for producing ten resolutions in one adaptive pass, while our table times one 50% coarsening; hardware and software environments differ. Those figures are context, not a speedup claim. The local memory measurements have no paper counterpart.

## Homogeneous and heterophilic datasets at 50% coarsening

| Dataset | Model | Paper AH-UGC (%) | Shingle (%) | Shingle vs paper (pp) | Random (%) | Random vs paper (pp) |
|---|---|---:|---:|---:|---:|---:|
| Cora | GCN | 77.34 | 84.225 ± 1.657 | +6.885 | 82.857 ± 1.627 | +5.517 |
| Cora | APPNP | 84.53 | 86.579 ± 2.178 | +2.049 | 85.915 ± 1.282 | +1.385 |
| CiteSeer | GCN | 65.46 | 72.591 ± 1.990 | +7.131 | 70.936 ± 1.852 | +5.476 |
| CiteSeer | APPNP | 69.06 | 73.464 ± 1.728 | +4.404 | 73.106 ± 1.967 | +4.046 |
| PubMed | GCN | 85.47 | 85.061 ± 0.503 | −0.409 | 84.744 ± 0.655 | −0.726 |
| PubMed | APPNP | 85.80 | 84.318 ± 0.748 | −1.482 | 82.967 ± 0.849 | −2.833 |
| Cornell | SGC | 76.08 | 54.595 ± 6.373 | −21.485 | 56.216 ± 5.378 | −19.864 |
| Cornell | MixHop | 71.61 | 53.243 ± 8.636 | −18.367 | 55.135 ± 7.568 | −16.475 |
| Chameleon | SGC | 58.60 | 47.412 ± 2.763 | −11.188 | 40.746 ± 2.255 | −17.854 |
| Chameleon | MixHop | 58.25 | 65.548 ± 2.138 | +7.298 | 60.307 ± 2.412 | +2.057 |
| Film | SGC | 21.68 | 22.750 ± 0.555 | +1.070 | 22.118 ± 1.428 | +0.438 |
| Film | MixHop | 21.79 | 22.612 ± 2.166 | +0.822 | 22.789 ± 1.773 | +0.999 |
| Reed98 | MixHop | 52.53 | 62.486 ± 1.872 | +9.956 | 62.543 ± 3.388 | +10.013 |
| Amherst41 | MixHop | 64.37 | 71.155 ± 2.897 | +6.785 | 70.467 ± 1.644 | +6.097 |

Reed98 and Amherst41 local numbers are from the 50% rows in the 10-seed MixHop runs. The paper's reported values are from its AH-UGC column. The local means are higher in these two comparisons, but the large differences should be read as reproduction gaps, not as evidence that the local method is superior: the data pipeline, splits, and training details may differ.

## Heterogeneous datasets at 30% coarsening

The paper reports AH-UGC accuracy at a 30% coarsening ratio for HeteroSGC and HeteroGCN2. The local values below use the computed-alpha, `F=orig` rows at the same ratio. Local heterogeneous runs used one seed, so there is no estimate of seed-to-seed variation.

| Dataset | Model | Paper AH-UGC (%) | Shingle (%) | Shingle vs paper (pp) | Random (%) | Random vs paper (pp) |
|---|---|---:|---:|---:|---:|---:|
| ACM | HeteroSGC | 59.00 | 67.258 | +8.258 | 47.929 | −11.071 |
| ACM | HeteroGCN2 | 83.47 | 78.107 | −5.363 | 68.836 | −14.634 |
| DBLP | HeteroSGC | 79.18 | 82.376 | +3.196 | 63.709 | −15.471 |
| DBLP | HeteroGCN2 | 66.00 | 46.669 | −19.331 | 54.068 | −11.932 |

These are one-seed local results and should be treated as preliminary. In the paper, ACM and DBLP refer to heterogeneous graphs for this table. They are distinct from the homogeneous DBLP benchmark used in the paper's runtime table.

## Runtime comparison with the paper

The paper reports total AH-UGC time to generate coarsened graphs at ten target ratios. The local benchmark reports the mean time for a single coarsening operation at ratio 0.5, averaged over five repetitions. The values are shown side by side because the paper provides a runtime reference, but these workloads and hardware differ, so their ratio is not a valid speedup measurement.

| Dataset | Paper AH-UGC time (s), ten ratios | Local Random time (s), ratio 0.5 | Local Shingle time (s), ratio 0.5 |
|---|---:|---:|---:|
| Cora | 7.00 | 0.0532 | 0.0067 |
| CiteSeer | 6.00 | 0.0976 | 0.0086 |
| PubMed | 29.00 | 0.4964 | 0.1078 |
| Chameleon | 6.73 | 0.0706 | 0.0061 |
| Film | 11.00 | 0.1315 | 0.0672 |

The paper reports experiments on a server with an Intel Xeon Platinum 8360Y CPU and two NVIDIA A6000 GPUs. Local measurements use this project's current CPU/software environment. The paper's Table 1 does not give a matching runtime for the heterogeneous ACM benchmark. Its DBLP runtime is for the homogeneous DBLP dataset and is not comparable to the local heterogeneous DBLP measurement.

## Local time and memory measurements

The project benchmark measures coarsening time and peak memory in an isolated worker over five repetitions at ratio 0.5. The homogeneous table reports peak RSS growth over the worker baseline. The heterogeneous table includes both absolute peak RSS and growth. The paper does not report comparable peak-RAM measurements, so no paper-versus-local memory difference can be calculated.

| Dataset | Nodes | Stored edge entries | Random time (s) | Random peak growth (MiB) | Shingle time (s) | Shingle peak growth (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| Cora | 2,708 | 10,556 | 0.0532 | 36.46 | 0.0067 | 0.59 |
| CiteSeer | 3,327 | 9,104 | 0.0976 | 51.93 | 0.0086 | 0.59 |
| PubMed | 19,717 | 88,648 | 0.4964 | 223.45 | 0.1078 | 1.08 |
| Chameleon | 2,277 | 36,051 | 0.0706 | 35.05 | 0.0061 | 0.68 |
| Film | 7,600 | 29,926 | 0.1315 | 91.04 | 0.0672 | 1.25 |
| Reed98 | 962 | 37,624 | 0.0253 | 13.75 | 0.0031 | 0.54 |
| Amherst41 | 2,235 | 181,908 | 0.1114 | 28.73 | 0.0081 | 1.23 |

| Heterogeneous dataset | Nodes | Typed edge entries | Random time (s) | Random peak RSS (MiB) | Random growth (MiB) | Shingle time (s) | Shingle peak RSS (MiB) | Shingle growth (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DBLP | 26,128 | 239,566 | 2.0104 | 1,283.98 | 202.95 | 0.6953 | 1,249.41 | 167.21 |
| ACM | 10,942 | 547,872 | 1.0650 | 1,012.93 | 72.13 | 0.3021 | 995.48 | 54.84 |

## Sources and run files

- AH-UGC paper, Tables 1, 4, 8, and 9: [OpenReview PDF](https://openreview.net/attachment?id=IQ3382JnQl&name=pdf).
- Existing homogeneous accuracy runs: `results_gcn_appnp.csv` and `results_sgc_mixhop.csv`.
- Reed98 and Amherst41 MixHop runs: `results_reed98_amherst41_mixhop.csv`.
- ACM and DBLP heterogeneous runs: `results_dblp_acm.csv` and `results_shingle_dblp_Acm.csv`.
- Runtime and memory measurements: `runtime_memory_benchmark.csv` and `hetero_runtime_memory_benchmark.csv`.
