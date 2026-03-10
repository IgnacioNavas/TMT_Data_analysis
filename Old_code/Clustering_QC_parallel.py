#!/usr/bin/env python3
"""
Quality control check for clusters (parallelized).

"""

import argparse
import os
import sys
import tempfile
import time
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from tslearn.metrics import cdist_dtw

from utils import (
    filter_replicates,
    filter_site_localizations,
    filter_dynamics_extremes,
    tslearn_clustering_KMeans,
)

# ---------------------------------------------------------------------------
# Worker-process globals
# ---------------------------------------------------------------------------
GLOBAL_DF = None # This variable will be loaded once into each CPU/core, so it doesn't have to be loaded for every task
GLOBAL_DATA_TYPE = None
GLOBAL_K_START = None
GLOBAL_K_END = None
GLOBAL_CONDITIONS = None
GLOBAL_TS_LENGTH = None
GLOBAL_TS_DIMENSIONS = None
GLOBAL_EXCLUDE_FULL = None
GLOBAL_N_SEEDS = None
GLOBAL_METRIC = None

def worker_init(filtered_pickle_path: str,
                data_type: str,
                k_start: int,
                k_end: int,
                conditions: list,
                ts_length: int,
                ts_dimensions: int,
                exclude_full: bool,
                metric: str,
                n_seeds: int,
                ) -> None:
    """
    Run once per worker process (CPU // core) before any tasks are dispatched.
    Sets BLAS/OpenMP thread counts to 1 (prevents oversubscription when
    multiple worker processes share a machine) then loads the pre-filtered
    DataFrame (GLOBAL_DF) from disk into a module-level global.
    """
    for var in ("OMP_NUM_THREADS", # OpenMP (used by many C/Fortran math libs)
                "MKL_NUM_THREADS", # Intel Math Kernel Library
                "OPENBLAS_NUM_THREADS", # OpenBLAS linear algebra
                "NUMEXPR_NUM_THREADS"): # NumExpr (fast numpy expression evaluator)
        os.environ.setdefault(var, "1")

    global GLOBAL_DF, GLOBAL_DATA_TYPE, GLOBAL_K_START, GLOBAL_K_END
    global GLOBAL_CONDITIONS, GLOBAL_TS_LENGTH, GLOBAL_TS_DIMENSIONS
    global GLOBAL_EXCLUDE_FULL, GLOBAL_METRIC, GLOBAL_N_SEEDS

    GLOBAL_DF = pd.read_pickle(filtered_pickle_path)
    GLOBAL_DATA_TYPE = data_type
    GLOBAL_K_START = k_start
    GLOBAL_K_END = k_end
    GLOBAL_CONDITIONS = conditions
    GLOBAL_TS_LENGTH = ts_length
    GLOBAL_TS_DIMENSIONS = ts_dimensions
    GLOBAL_EXCLUDE_FULL = exclude_full
    GLOBAL_METRIC = metric
    GLOBAL_N_SEEDS = n_seeds


def cluster_worker(k_seed_tuple: tuple) -> dict:
    """
    Run KMeans clustering for a single (k, seed) combination.
    All clustering parameters are read from worker-process globals set by worker_init.

    Returns
    -------
    dict with keys:
        k        – number of clusters requested
        seed     – random seed used
        labels   – cluster label array (shape: n_samples,)
        inertia  – final inertia of the fitted model
        multivariate – raw multivariate array needed for silhouette (seed 0 only, None otherwise, to avoid
                       shipping large arrays back for every seed)
    """
    k, seed = k_seed_tuple

    # Build a descriptive name for logging; keep it outside tslearn_clustering_KMeans so we don't reconstruct the
    # string inside the library call.
    conditions_str = "+".join(c.strip("_") for c in GLOBAL_CONDITIONS)
    cluster_name = (f"KMeans_{k}_cluster_seed{seed}_on_{GLOBAL_DATA_TYPE}_exludeFull{GLOBAL_EXCLUDE_FULL}_nrep>1_locSite=True_log2FC>0.5_conditions({conditions_str})_metric{GLOBAL_METRIC}")

    print(f"[worker pid={os.getpid()}] Starting {cluster_name}", flush=True)

    # GLOBAL_DF is already a copy-on-write isolated in this process; no .copy() needed.
    df_clustered, model, multivariate = tslearn_clustering_KMeans(
        df_to_cluster=GLOBAL_DF,
        data_type=GLOBAL_DATA_TYPE,
        condition_for_clustering=GLOBAL_CONDITIONS,
        exclude_full=GLOBAL_EXCLUDE_FULL,
        cluster_column_name=cluster_name,
        number_of_clusters=k,
        max_iterations=200,
        n_init=5,
        metric=GLOBAL_METRIC,
        df_dimensions=GLOBAL_TS_DIMENSIONS,
        time_series_length=GLOBAL_TS_LENGTH,
        random_state=seed,
        transpose=True,
        verbose=False,
        testing=True,
        barycenter_calculations=False
    )

    return {
        "k": k,
        "seed": seed,
        "labels": model.labels_,
        "inertia": model.inertia_,
        # Only ship multivariate back for seed 0; it can be large and is only needed once per k value for the
        # silhouette computation.
        "multivariate": multivariate if seed == 0 else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser( description="Quality control check for clusters (parallelized)")

    parser.add_argument("--input_file_path", help="Path to input TSV file")
    parser.add_argument("--output_file_path", help="Prefix for output PNG files")
    parser.add_argument("--data_type", type=str, default="log2_FC", help="Type of data used for the clustering (default: log2_FC)")
    parser.add_argument("--k_start", type=int, default=2, help="Number of clusters to start (default: 2)")
    parser.add_argument("--k_end", type=int, default=100, help="Number of clusters to end (default: 100)")
    parser.add_argument("--conditions",nargs="+", choices=["_EGF_", "_INS_", "_EGFnINS_"], default=["_EGF_"], help="Conditions to use for the clustering (default:_EGF_, _INS_, EGFnINS)")
    parser.add_argument("--ts_length", type=int, default=6, help="Length of time series to use (default: 6)")
    parser.add_argument("--ts_dimensions", type=int, default=3, help="Dimensions of time series to use (default: 3)")
    parser.add_argument("--exclude_full", type=bool, default=True, help="Exclude full for clustering (default: False)")
    parser.add_argument("--metric", type=str, default="euclidean", help="Metric to use for clustering (default: euclidean) )")
    parser.add_argument("--n_seeds", type=int, default=10, help="Seed for random number generation (default: 10)")
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel worker processes (default: 6)")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load & filter data
    # ------------------------------------------------------------------
    df_raw = pd.read_csv(args.input_file_path, sep="\t").fillna(0)

    print( f"Run config: data_type={args.data_type} | k={args.k_start}–{args.k_end} "
           f"| conditions={args.conditions} | ts_length={args.ts_length} "
           f"| ts_dimensions={args.ts_dimensions} | exclude_full={args.exclude_full} "
           f"| metric={args.metric} | n_seeds={args.n_seeds} \n")

    print(f"Raw DataFrame shape: {df_raw.shape}")

    # Filtering the dataframe
    nreps_df = filter_replicates(df_raw, n_reps=2)
    localized_sites_df = filter_site_localizations(nreps_df, loc_sites=True)
    df_filtered = filter_dynamics_extremes(
        df=localized_sites_df,   # <-- was df_raw in the original
        data_type=args.data_type,
        threshold=0.5,
        exclude_full=args.exclude_full,
    )

    print(
        f"nreps_df:           {nreps_df.shape}\n"
        f"localized_sites_df: {localized_sites_df.shape}\n"
        f"df_filtered:        {df_filtered.shape}\n"
    )

    # Serialize filtered DataFrame once; workers load it via their initializer.
    tmpdir = tempfile.mkdtemp(prefix="cluster_par_") # Adding human readable tag to the temporary directory name
    filtered_pickle_path = os.path.join(tmpdir, "df_filtered.pkl")
    df_filtered.to_pickle(filtered_pickle_path) # Store the filtered dataframe in the temporary directory as a pickle file

    # ------------------------------------------------------------------
    # Parallel clustering
    # Submit ALL (k, seed) tasks at once so all cores stay busy the entire time
    # ------------------------------------------------------------------
    ks = range(args.k_start, args.k_end)
    seeds = list(range(args.n_seeds))

    # Build the full task list upfront: all k × seed combinations
    all_tasks = [(k, seed) for k in ks for seed in seeds]
    print(f"Total tasks to run: {len(all_tasks)} ({len(list(ks))} k-values × {len(seeds)} seeds)\n", flush=True)

    # results_by_k collects all returned dicts grouped by k value, so we can compute metrics per k after all futures are done.
    results_by_k = defaultdict(list)

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=worker_init,
        initargs=(filtered_pickle_path,
                  args.data_type,
                  args.k_start,
                  args.k_end,
                  args.conditions,
                  args.ts_length,
                  args.ts_dimensions,
                  args.exclude_full,
                  args.metric,
                  args.n_seeds,),
    ) as executor:

        # Submit ALL tasks at once — executor fills cores from the queue automatically
        futures = {executor.submit(cluster_worker, t): t for t in all_tasks}

        # Collect results as they finish (in arbitrary order)
        completed = 0
        for fut in as_completed(futures):
            res = fut.result()
            results_by_k[res["k"]].append(res)
            completed += 1
            if completed % 50 == 0:
                print(f"Progress: {completed}/{len(all_tasks)} tasks completed", flush=True)

        print(f"\nAll {len(all_tasks)} tasks completed. Computing metrics...\n", flush=True)

        # ------------------------------------------------------------------
        # Compute metrics per k (done after all futures finish)
        # ------------------------------------------------------------------

        inertias: list = []
        silhouettes: list = []
        stabilities: list = []

        for k in ks:
            results = results_by_k[k]
            # Reconstruct seed → labels mapping
            seed_to_labels = {r["seed"]: r["labels"] for r in results}
            # Get inertia and multivariate from seed 0
            seed0_results = [r for r in results if r["seed"] == 0]

            if seed0_results:
                inertia_for_k = seed0_results[0]["inertia"]
                multivariate_seed0 = seed0_results[0]["multivariate"]
            else:
                print(f"WARNING: seed 0 result missing for k={k}; inertia=NaN", flush=True)
                inertia_for_k = np.nan
                multivariate_seed0 = None

            # ----------------------------------------------------------
            # Silhouette score (seed 0 labels + seed 0 multivariate). Double check which metric to use.
            # I can flatten the dataframe (reshaping to (n_samples, time_steps)) (.reshape(n_samples, -1))
            # or
            # I can use DTW (more computationally expensive) to respect timporal axis
            # ----------------------------------------------------------
            sil = np.nan
            if multivariate_seed0 is not None and 0 in seed_to_labels:
                labels_seed0 = seed_to_labels[0]
                # n_samples = multivariate_seed0.shape[0]
                # If univariate: reshape (n_samples, time_steps, 1) → (n_samples, time_steps) for sklearn (Optional)
                # if multivariate_seed0.shape[2] == 1:
                #     X_2d = multivariate_seed0.reshape(n_samples, -1)
                #     try:
                #         sil = silhouette_score(X_2d, labels_seed0, metric="euclidean")
                #     except ValueError as exc:
                #         # Can happen when k >= n_samples
                #         print(f"Silhouette skipped for k={k}: {exc}", flush=True)
                # else:
                #     # Multivariate: use DTW to respect temporal + dimensional structure
                D = cdist_dtw(multivariate_seed0)
                try:
                    sil = silhouette_score(D, labels_seed0, metric="precomputed")
                except ValueError as exc:
                    # Can happen when k >= n_samples
                    print(f"Silhouette skipped for k={k}: {exc}", flush=True)

            # ----------------------------------------------------------
            # Cluster stability: mean pairwise ARI across all seeds
            # ----------------------------------------------------------
            labels_list = list(seed_to_labels.values())
            ari_scores = [
                adjusted_rand_score(labels_list[i], labels_list[j])
                for i in range(len(labels_list))
                for j in range(i + 1, len(labels_list))
            ]
            stability = np.mean(ari_scores) if ari_scores else np.nan

            inertias.append(inertia_for_k)
            silhouettes.append(sil)
            stabilities.append(stability)

            print(f"k={k:3d} | inertia={inertia_for_k:.4f} | silhouette={sil:.4f} | stability={stability:.4f}",flush=True)

    # ------------------------------------------------------------------
    # Save results to TSV alongside the PNG files (handy for downstream use)
    # ------------------------------------------------------------------
    results_df = pd.DataFrame({
        "k":          list(ks),
        "inertia":    inertias,
        "silhouette": silhouettes,
        "stability":  stabilities,
    })

    conditions_label = "+".join(c.strip("_") for c in args.conditions)

    results_tsv = f"{args.output_file_path}_qc_metrics_{conditions_label}.tsv"
    results_df.to_csv(results_tsv, sep="\t", index=False)
    print(f"\nMetrics saved to: {results_tsv}")

    # ------------------------------------------------------------------
    # Plot results
    # ------------------------------------------------------------------
    k_list = list(ks)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Cluster QC Metrics | {conditions_label}", fontsize=14, fontweight="bold")

    for metric_name, values, ylabel, subplot in [("inertia",    inertias,    "Inertia", 0),
                                                 ("silhouette", silhouettes, "Silhouette Score", 1),
                                                 ("stability",  stabilities, "Mean Pairwise ARI (stability)", 2)]:
        ax[subplot].scatter(k_list, values, s=20)
        ax[subplot].set_title(f"{ylabel} vs Number of Clusters")
        ax[subplot].set_xlabel("Number of Clusters (k)")
        ax[subplot].set_ylabel(ylabel)
        ax[subplot].grid(True, alpha=0.3)

    fig.tight_layout()
    out_png = f"{args.output_file_path}_metrics_{conditions_label}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Imaged saved: {out_png}")

    # Clean up temp pickle
    os.remove(filtered_pickle_path)
    os.rmdir(tmpdir)
    print("Done.\n")
    print("Hand written notes: \n")


if __name__ == "__main__":

    _t0 = time.perf_counter()
    main()
    _t1 = time.perf_counter()
    elapsed = _t1 - _t0
    print(f"\nTotal time: {elapsed:.2f} s  ({elapsed / 60:.2f} min)")

