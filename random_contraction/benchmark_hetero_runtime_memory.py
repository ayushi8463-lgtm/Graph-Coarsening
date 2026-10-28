"""CPU runtime and peak-RSS benchmark for type-isolated hetero coarsening.

Example:
  python random_contraction/benchmark_hetero_runtime_memory.py --datasets dblp acm --repeats 5

Times the existing AH-UGC heterogeneous coarsening routine (all node types and
typed relations), without classifier training or changes to hash/merge logic.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import psutil

import ahugc_eval_v2 as evaluator


class PeakRSS:
    def __init__(self, interval=0.002):
        self.process = psutil.Process()
        self.start = self.process.memory_info().rss
        self.peak = self.start
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop_event.is_set():
            try:
                self.peak = max(self.peak, self.process.memory_info().rss)
            except psutil.Error:
                pass
            self.stop_event.wait(0.002)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def worker(name, method, ratio, repeats):
    hd = evaluator.load_hetero(name, "data")
    alpha = evaluator.hetero_alpha("computed", hd)
    train, _, _ = evaluator.hetero_splits(hd, 0, "public")

    def run():
        for seed in range(repeats):
            evaluator.hetero_coarsen_np(
                hd, "orig", method, ratio, alpha,
                np.random.default_rng(20261028 + seed), train)

    with PeakRSS() as memory:
        start = time.perf_counter()
        run()
        elapsed = time.perf_counter() - start
    return {"peak_rss_mb": memory.peak / 2**20,
            "peak_growth_mb": max(0, memory.peak - memory.start) / 2**20,
            "seconds_for_repeats": elapsed}


def isolated_worker(name, method, ratio, repeats):
    command = [sys.executable, os.path.abspath(__file__), "--worker", name, method,
               "--ratio", str(ratio), "--repeats", str(repeats)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("WORKER_RESULT "):
            return json.loads(line[len("WORKER_RESULT "):])
    raise RuntimeError(f"No memory result for {name}/{method}: {result.stdout}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["dblp", "acm"], choices=["dblp", "acm"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--out", default="random_contraction/hetero_runtime_memory_benchmark.csv")
    parser.add_argument("--worker", nargs=2, metavar=("DATASET", "METHOD"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        print("WORKER_RESULT " + json.dumps(worker(args.worker[0], args.worker[1],
                                                     args.ratio, args.repeats)))
        return

    rows = []
    for name in args.datasets:
        start = time.perf_counter()
        hd = evaluator.load_hetero(name, "data")
        load_seconds = time.perf_counter() - start
        alpha = evaluator.hetero_alpha("computed", hd)
        train, _, _ = evaluator.hetero_splits(hd, 0, "public")
        start = time.perf_counter()
        for node_type in hd.node_types:
            evaluator.hetero_F("orig", hd, node_type, alpha)
        build_f_seconds = time.perf_counter() - start

        total_nodes = sum(hd.N.values())
        edge_entries = sum(A.nnz for A in hd.A.values())
        node_counts = ";".join(f"{t}={hd.N[t]}" for t in hd.node_types)
        feature_dims = ";".join(f"{t}={hd.x[t].shape[1]}" for t in hd.node_types)
        print(f"{name}: nodes={total_nodes} ({node_counts}), edge_entries={edge_entries}, "
              f"relations={len(hd.edge_types)}, build_F={build_f_seconds:.3f}s", flush=True)

        for method in ("random", "shingle"):
            times, coarse_nodes = [], []
            for seed in range(args.repeats):
                t0 = time.perf_counter()
                pis, _, _, _, _ = evaluator.hetero_coarsen_np(
                    hd, "orig", method, args.ratio, alpha,
                    np.random.default_rng(20261028 + seed), train)
                times.append(time.perf_counter() - t0)
                coarse_nodes.append(sum(int(pi.max()) + 1 for pi in pis.values()))
            isolated = isolated_worker(name, method, args.ratio, args.repeats)
            row = {
                "dataset": name,
                "method": method,
                "target_type": hd.target,
                "node_type_counts": node_counts,
                "total_nodes": total_nodes,
                "typed_edge_entries": edge_entries,
                "relation_types": len(hd.edge_types),
                "feature_dims_by_type": feature_dims,
                "ratio": args.ratio,
                "repeats": args.repeats,
                "alpha_1_minus_heterophily": alpha,
                "load_seconds": load_seconds,
                "build_F_seconds": build_f_seconds,
                "coarsen_seconds_mean": float(np.mean(times)),
                "coarsen_seconds_sd": float(np.std(times)),
                "coarsen_peak_rss_mb": isolated["peak_rss_mb"],
                "coarsen_peak_growth_mb": isolated["peak_growth_mb"],
                "supernodes_mean": float(np.mean(coarse_nodes)),
            }
            rows.append(row)
            print(f"  {method:7s} coarsen={row['coarsen_seconds_mean']:.4f}±"
                  f"{row['coarsen_seconds_sd']:.4f}s, peak growth="
                  f"{row['coarsen_peak_growth_mb']:.1f}MB", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} measurements to {args.out}", flush=True)


if __name__ == "__main__":
    main()
