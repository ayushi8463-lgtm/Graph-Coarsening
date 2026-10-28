# Local results compared with AH-UGC

All accuracy differences are percentage points (pp): local accuracy minus the AH-UGC value reported in the paper. The 50% comparisons use the AH-UGC column from the paper's node-classification tables. Results from the local 10-seed runs are shown as mean ± standard deviation; paper values are point estimates.

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
