"""Measure current AH-UGC coarsening runtime and process memory on previously run datasets.

Example: python random_contraction/benchmark_runtime_memory.py --repeats 5

This times the actual evaluator's F construction and coarsen() calls. It does not
train models or alter hashing/merging. Peak RSS is sampled during each stage and
reported both as process RSS and as growth above that stage's starting RSS.
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

import ahugc_eval_shingle as evaluator


DATASETS_RUN = ("cora", "citeseer", "pubmed", "cornell", "chameleon", "film", "reed98", "amherst41")


class PeakRSS:
    def __init__(self, interval=0.002):
        self.process = psutil.Process()
        self.interval = interval
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
            self.stop_event.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def measure_stage(fn):
    with PeakRSS() as memory:
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
    return result, elapsed, memory.peak / 2**20, max(0, memory.peak - memory.start) / 2**20


def isolated_memory(dataset, mode, ratio, repeats):
    """Measure each method in a fresh process so one method's allocator cache
    cannot mask the next method's peak memory."""
    command = [sys.executable, os.path.abspath(__file__), "--memory-worker", dataset, mode,
               "--ratio", str(ratio), "--repeats", str(repeats)]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("MEMORY_RESULT "):
            return json.loads(line[len("MEMORY_RESULT "):])
    raise RuntimeError(f"memory worker returned no result for {dataset}/{mode}: {completed.stdout}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS_RUN))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--out", default="random_contraction/runtime_memory_benchmark.csv")
    parser.add_argument("--memory-worker", nargs=2, metavar=("DATASET", "METHOD"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.memory_worker:
        name, mode = args.memory_worker
        A, A_lap, X, y, _ = evaluator.load(name, source="authors", root="data")
        alpha = evaluator.heterophily_alpha(A_lap, y)
        F = evaluator.build_F("orig", A, X, alpha)
        def repeat_coarsenings():
            for repeat in range(args.repeats):
                evaluator.coarsen(F, args.ratio, np.random.default_rng(20261028 + repeat),
                                  mode, A=A, X=X)
        with PeakRSS() as memory:
            repeat_coarsenings()
        print("MEMORY_RESULT " + json.dumps({
            "peak_rss_mb": memory.peak / 2**20,
            "peak_growth_mb": max(0, memory.peak - memory.start) / 2**20,
        }))
        return

    rows = []
    skipped = []

    for name in args.datasets:
        load_start = time.perf_counter()
        try:
            A, A_lap, X, y, _ = evaluator.load(name, source="authors", root="data")
        except Exception as exc:
            skipped.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"{name}: SKIPPED; graph loader failed: {skipped[-1][1]}", flush=True)
            continue
        load_sec = time.perf_counter() - load_start
        alpha = evaluator.heterophily_alpha(A_lap, y)
        (F, build_sec, build_peak, build_growth) = measure_stage(
            lambda: evaluator.build_F("orig", A, X, alpha))
        print(f"{name}: N={A.shape[0]} edges={A_lap.nnz} feat={X.shape[1]} load={load_sec:.3f}s "
              f"F={F.shape} build_F={build_sec:.3f}s", flush=True)

        for mode in ("random", "shingle"):
            times, peaks, growths, coarse_counts = [], [], [], []
            for repeat in range(args.repeats):
                rng = np.random.default_rng(20261028 + repeat)
                pi, elapsed, peak, growth = measure_stage(
                    lambda: evaluator.coarsen(F, args.ratio, rng, mode, A=A, X=X))
                times.append(elapsed)
                peaks.append(peak)
                growths.append(growth)
                coarse_counts.append(int(pi.max()) + 1)
                del pi
            isolated_peak = isolated_memory(name, mode, args.ratio, args.repeats)
            row = {
                "dataset": name,
                "method": mode,
                "nodes": A.shape[0],
                "directed_edges_stored": A_lap.nnz,
                "feature_dim": X.shape[1],
                "alpha_1_minus_homophily": float(alpha),
                "ratio": args.ratio,
                "repeats": args.repeats,
                "load_seconds": load_sec,
                "build_F_seconds": build_sec,
                "build_F_peak_rss_mb": build_peak,
                "build_F_peak_growth_mb": build_growth,
                "coarsen_seconds_mean": float(np.mean(times)),
                "coarsen_seconds_sd": float(np.std(times)),
                "coarsen_peak_rss_mb": isolated_peak["peak_rss_mb"],
                "coarsen_peak_growth_mb": isolated_peak["peak_growth_mb"],
                "supernodes_mean": float(np.mean(coarse_counts)),
            }
            rows.append(row)
            print(f"  {mode:7s} coarsen={row['coarsen_seconds_mean']:.4f}±{row['coarsen_seconds_sd']:.4f}s "
                  f"peakRSS={row['coarsen_peak_rss_mb']:.1f}MB (+{row['coarsen_peak_growth_mb']:.1f}MB)",
                  flush=True)
        del F, A, A_lap, X, y

        # Keep completed dataset measurements if a later graph is unavailable.
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    if rows:
        print(f"Saved {len(rows)} measurements to {args.out}", flush=True)
    if skipped:
        print("Skipped datasets (their original graph files are not cached and downloads failed):", flush=True)
        for name, reason in skipped:
            print(f"  {name}: {reason}", flush=True)


if __name__ == "__main__":
    main()
