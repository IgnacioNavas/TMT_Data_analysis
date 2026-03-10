#!/usr/bin/env python3
"""
Quality control check for clusters (parallelized).

Parallelization strategy:
  - Stage 1: ALL (k x seed) clustering tasks submitted at once so all cores
              stay busy the entire time (no idle cores between k-values).
              Results are streamed to disk as they arrive to avoid holding
              all results in RAM simultaneously.
  - Stage 2: Metrics (silhouette via DTW, ARI stability) computed in parallel
              — one k-value per worker. Controlled by --metrics_workers.

Robustness:
  - Checkpoint saved after Stage 1. If Stage 2 crashes, rerun the same
    command and Stage 1 will be skipped automatically.
  - Silhouette skipped for k-values where it stays below --sil_min for
    --sil_patience consecutive k-values (optional, disabled by default).

CLI arguments:
  --input_file_path   : path to input TSV file
  --output_file_path  : prefix for all output files
  --data_type         : data type for clustering (default: log2_FC)
  --k_start           : minimum k to test (default: 2)
  --k_end             : maximum k to test, exclusive (default: 100)
  --conditions        : one or more of _EGF_ _INS_ _EGFnINS_ (default: _EGF_)
  --ts_length         : time series length (default: 6)
  --ts_dimensions     : number of dimensions (default: 3)
  --exclude_full      : exclude full data (default: True)
  --metric            : clustering distance metric (default: euclidean)
  --n_seeds           : number of seeds for stability (default: 10)
  --n_init            : KMeans initialisations per run (default: 5)
  --max_iterations    : max KMeans iterations (default: 100)
  --workers           : worker processes for Stage 1 (default: 6)
  --metrics_workers   : worker processes for Stage 2 (default: min(workers, 30))
  --sil_min           : silhouette threshold for early stopping (default: disabled)
  --sil_patience      : consecutive k-values below sil_min before stopping (default: 5)
"""

import argparse
import os
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display needed on server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score
from tslearn.metrics import cdist_dtw

from utils import (
    filter_replicates,
    filter_site_localizations,
    filter_dynamics_extremes,
    # tslearn_clustering_KMeans,
    kernnel_clustering,
)

# ---------------------------------------------------------------------------
# Worker-process globals (set once per worker by worker_init, Stage 1 only)
# ---------------------------------------------------------------------------
GLOBAL_DF            = None  # Loaded once per worker, not once per task
GLOBAL_DATA_TYPE     = None
GLOBAL_K_START       = None
GLOBAL_K_END         = None
GLOBAL_CONDITIONS    = None
GLOBAL_TS_LENGTH     = None
GLOBAL_TS_DIMENSIONS = None
GLOBAL_EXCLUDE_FULL  = None
GLOBAL_N_SEEDS       = None
GLOBAL_METRIC        = None
GLOBAL_N_INIT        = None
# GLOBAL_MAX_ITER      = None
GLOBAL_KERNEL_METRIC = None


def worker_init(
    filtered_pickle_path: str,
    data_type: str,
    k_start: int,
    k_end: int,
    conditions: list,
    ts_length: int,
    ts_dimensions: int,
    exclude_full: bool,
    metric: str,
    n_seeds: int,
    n_init: int,
    # max_iterations: int,
    kernel_metric: str,
) -> None:
    """
    Run once per worker process (CPU/core) before any tasks are dispatched.
    Sets BLAS/OpenMP thread counts to 1 (prevents oversubscription when
    multiple worker processes share a machine) then loads the pre-filtered
    DataFrame (GLOBAL_DF) from disk into a module-level global.
    """
    for var in (
        "OMP_NUM_THREADS",       # OpenMP (used by many C/Fortran math libs)
        "MKL_NUM_THREADS",       # Intel Math Kernel Library
        "OPENBLAS_NUM_THREADS",  # OpenBLAS linear algebra
        "NUMEXPR_NUM_THREADS",   # NumExpr (fast numpy expression evaluator)
    ):
        os.environ.setdefault(var, "1")

    global GLOBAL_DF, GLOBAL_DATA_TYPE, GLOBAL_K_START, GLOBAL_K_END
    global GLOBAL_CONDITIONS, GLOBAL_TS_LENGTH, GLOBAL_TS_DIMENSIONS
    global GLOBAL_EXCLUDE_FULL, GLOBAL_METRIC, GLOBAL_N_SEEDS
    global GLOBAL_N_INIT, GLOBAL_KERNEL_METRIC #, GLOBAL_MAX_ITER

    GLOBAL_DF            = pd.read_pickle(filtered_pickle_path)
    GLOBAL_DATA_TYPE     = data_type
    GLOBAL_K_START       = k_start
    GLOBAL_K_END         = k_end
    GLOBAL_CONDITIONS    = conditions
    GLOBAL_TS_LENGTH     = ts_length
    GLOBAL_TS_DIMENSIONS = ts_dimensions
    GLOBAL_EXCLUDE_FULL  = exclude_full
    GLOBAL_METRIC        = metric
    GLOBAL_N_SEEDS       = n_seeds
    GLOBAL_N_INIT        = n_init
    # GLOBAL_MAX_ITER      = max_iterations
    GLOBAL_KERNEL_METRIC = kernel_metric


# ---------------------------------------------------------------------------
# Stage 1 worker: clustering
# ---------------------------------------------------------------------------

def cluster_worker(k_seed_tuple: tuple) -> dict:
    """
    Run KMeans clustering for a single (k, seed) combination.
    All clustering parameters are read from worker-process globals set by worker_init.

    Returns
    -------
    dict with keys:
        k            - number of clusters requested
        seed         - random seed used
        labels       - cluster label array (shape: n_samples,)
        inertia      - final inertia of the fitted model
        multivariate - raw multivariate array for silhouette (seed 0 only,
                       None otherwise to avoid shipping large arrays for every seed)
    """
    k, seed = k_seed_tuple

    conditions_str = "+".join(c.strip("_") for c in GLOBAL_CONDITIONS)
    cluster_name = (
        f"KMeans_{k}_cluster_seed{seed}_on_{GLOBAL_DATA_TYPE}"
        f"_excludeFull{GLOBAL_EXCLUDE_FULL}_nrep>1_locSite=True"
        f"_log2FC>0.5_conditions({conditions_str})_metric{GLOBAL_METRIC}"
    )
    print(f"[clustering pid={os.getpid()}] k={k} seed={seed}", flush=True)

    # GLOBAL_DF is already copy-on-write isolated in this process; no .copy() needed.
    df_clustered, model, multivariate = kernnel_clustering(
        df_to_cluster=GLOBAL_DF,
        transpose=True,
        data_type=GLOBAL_DATA_TYPE,
        exclude_full=GLOBAL_EXCLUDE_FULL,
        condition_for_clustering=GLOBAL_CONDITIONS,
        df_dimensions=GLOBAL_TS_DIMENSIONS,
        time_series_length=GLOBAL_TS_LENGTH,
        seed=seed,
        n_clusters=k,
        n_init=GLOBAL_N_INIT,
        verbose=False,
        kernel=GLOBAL_KERNEL_METRIC,
        kernel_params={"sigma":"auto"},
        cluster_column_name=cluster_name,
        testing=True
    )

    return {
        "k": k,
        "seed": seed,
        "labels": model.labels_,
        "inertia": model.inertia_,
        # Only ship multivariate back for seed 0 — needed once per k for silhouette.
        "multivariate": multivariate if seed == 0 else None,
    }


# ---------------------------------------------------------------------------
# Stage 2 worker: metrics
# ---------------------------------------------------------------------------

def metrics_worker(k_results_dir_tuple: tuple) -> dict:
    """
    Compute inertia, silhouette (DTW) and ARI stability for a single k value.
    Runs in a separate process so all k-values are computed in parallel.

    Loads per-k result files from disk (written during Stage 1) rather than
    receiving large arrays via the process queue, avoiding serialization
    overhead and keeping RAM usage bounded.

    No GLOBAL_DF needed — all inputs loaded from per-k result files on disk.

    Parameters
    ----------
    k_results_dir_tuple : (k, results_dir, seeds)

    Returns
    -------
    dict with keys: k, inertia, silhouette, stability
    """
    k, results_dir, seeds = k_results_dir_tuple

    # Load all seed results for this k from disk
    results = []
    for seed in seeds:
        result_path = os.path.join(results_dir, f"k{k}_seed{seed}.pkl")
        if os.path.exists(result_path):
            results.append(pd.read_pickle(result_path))
        else:
            print(f"WARNING: missing result file for k={k} seed={seed}", flush=True)

    if not results:
        print(f"ERROR: no results found for k={k}", flush=True)
        return {"k": k, "inertia": np.nan, "silhouette": np.nan, "stability": np.nan}

    # Reconstruct seed -> labels mapping
    seed_to_labels = {r["seed"]: r["labels"] for r in results}

    # --- Inertia (from seed 0) ---
    seed0 = [r for r in results if r["seed"] == 0]
    if seed0:
        inertia_for_k      = seed0[0]["inertia"]
        multivariate_seed0 = seed0[0]["multivariate"]
    else:
        print(f"WARNING: seed 0 result missing for k={k}; inertia=NaN", flush=True)
        inertia_for_k      = np.nan
        multivariate_seed0 = None

    # --- Silhouette score via DTW distance matrix ---
    # cdist_dtw respects temporal structure for both univariate and multivariate.
    # Each worker builds its own DTW matrix (~270MB for n=5825) independently,
    # which is freed from memory as soon as this function returns.
    sil = np.nan
    if multivariate_seed0 is not None and 0 in seed_to_labels:
        D = cdist_dtw(multivariate_seed0)
        try:
            sil = silhouette_score(D, seed_to_labels[0], metric="precomputed")
        except ValueError as exc:
            # Can happen when k >= n_samples
            print(f"Silhouette skipped for k={k}: {exc}", flush=True)

    # --- Stability: mean pairwise ARI across all seeds ---
    labels_list = list(seed_to_labels.values())
    ari_scores = [
        adjusted_rand_score(labels_list[i], labels_list[j])
        for i in range(len(labels_list))
        for j in range(i + 1, len(labels_list))
    ]
    stability = np.mean(ari_scores) if ari_scores else np.nan

    print(
        f"[metrics pid={os.getpid()}] "
        f"k={k:3d} | inertia={inertia_for_k:.4f} | silhouette={sil:.4f} | stability={stability:.4f}",
        flush=True,
    )

    return {"k": k, "inertia": inertia_for_k, "silhouette": sil, "stability": stability}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Quality control check for clusters (parallelized)")

    parser.add_argument("--input_file_path",  help="Path to input TSV file")
    parser.add_argument("--output_file_path", help="Prefix for all output files")
    parser.add_argument("--data_type",        type=str,   default="log2_FC",
                        help="Type of data used for clustering (default: log2_FC)")
    parser.add_argument("--k_start",          type=int,   default=2,
                        help="Minimum number of clusters to test (default: 2)")
    parser.add_argument("--k_end",            type=int,   default=100,
                        help="Maximum number of clusters to test, exclusive (default: 100)")
    parser.add_argument("--conditions",       nargs="+",  choices=["_EGF_", "_INS_", "_EGFnINS_"],
                        default=["_EGF_"],
                        help="Conditions to cluster on (default: _EGF_)")
    parser.add_argument("--ts_length",        type=int,   default=6,
                        help="Time series length (default: 6)")
    parser.add_argument("--ts_dimensions",    type=int,   default=3,
                        help="Number of time series dimensions (default: 3)")
    parser.add_argument("--exclude_full",     type=bool,  default=True,
                        help="Exclude full data (default: True)")
    parser.add_argument("--metric",           type=str,   default="euclidean",
                        help="Distance metric for clustering (default: euclidean)")
    parser.add_argument("--n_seeds",          type=int,   default=10,
                        help="Number of seeds for stability calculation (default: 10)")
    parser.add_argument("--n_init",           type=int,   default=5,
                        help="KMeans initialisations per run (default: 5). "
                             "Higher values give better convergence but are slower.")
    # parser.add_argument("--max_iterations",   type=int,   default=100,
    #                     help="Max KMeans iterations per run (default: 100). "
    #                          "Use 300 for final clustering runs, 100 is enough for QC sweeps.")
    parser.add_argument("--workers",          type=int,   default=6,
                        help="Worker processes for Stage 1 clustering (default: 6)")
    parser.add_argument("--metrics_workers",  type=int,   default=None,
                        help="Worker processes for Stage 2 metrics (default: min(workers, 30)). "
                             "Each metrics worker builds a ~270MB DTW matrix, so this controls "
                             "peak RAM usage in Stage 2.")
    parser.add_argument("--sil_min",          type=float, default=None,
                        help="Optional silhouette early-stop threshold. If silhouette stays "
                             "below this value for --sil_patience consecutive k-values, "
                             "silhouette is set to NaN for remaining k-values (default: disabled).")
    parser.add_argument("--sil_patience",     type=int,   default=5,
                        help="Consecutive k-values below --sil_min before silhouette is skipped "
                             "(default: 5). Only used if --sil_min is set.")
    parser.add_argument("--kernel_metric", type=str, default="gak",
                        help = "Kernel metric for clustering (default: gak)")

    args = parser.parse_args()

    # Resolve metrics_workers default: min(workers, 30) as a safe RAM-aware default
    metrics_workers = args.metrics_workers if args.metrics_workers is not None else min(args.workers, 30)

    # ------------------------------------------------------------------
    # Load & filter data
    # ------------------------------------------------------------------
    df_raw = pd.read_csv(args.input_file_path, sep="\t").fillna(0)

    print(
        f"Run config:\n"
        f"  data_type={args.data_type} | k={args.k_start}–{args.k_end}\n"
        f"  conditions={args.conditions} | ts_length={args.ts_length} | ts_dimensions={args.ts_dimensions}\n"
        f"  exclude_full={args.exclude_full} | metric={args.metric} | kernel_metric={args.kernel_metric}\n"
        f"  n_seeds={args.n_seeds} | n_init={args.n_init}\n" # | max_iterations={args.max_iterations}\n"
        f"  workers={args.workers} | metrics_workers={metrics_workers}\n"
        f"  sil_min={args.sil_min} | sil_patience={args.sil_patience}\n"
    )
    print(f"Raw DataFrame shape: {df_raw.shape}")

    nreps_df           = filter_replicates(df_raw, n_reps=2)
    localized_sites_df = filter_site_localizations(nreps_df, loc_sites=True)
    df_filtered        = filter_dynamics_extremes(
        df=localized_sites_df,
        data_type=args.data_type,
        threshold=0.5,
        exclude_full=args.exclude_full,
    )

    print(
        f"nreps_df:           {nreps_df.shape}\n"
        f"localized_sites_df: {localized_sites_df.shape}\n"
        f"df_filtered:        {df_filtered.shape}\n"
    )

    # Temp directory for pickle files (filtered df + per-result files)
    tmpdir = tempfile.mkdtemp(prefix="cluster_par_")
    filtered_pickle_path = os.path.join(tmpdir, "df_filtered.pkl")
    results_dir          = os.path.join(tmpdir, "results")
    os.makedirs(results_dir)
    df_filtered.to_pickle(filtered_pickle_path)

    ks    = range(args.k_start, args.k_end)
    seeds = list(range(args.n_seeds))

    # ------------------------------------------------------------------
    # Checkpoint: skip Stage 1 if a checkpoint already exists
    # If Stage 2 crashed previously, rerun the same command to resume.
    # ------------------------------------------------------------------
    checkpoint_path = f"{args.output_file_path}_checkpoint.pkl"

    if os.path.exists(checkpoint_path):
        print(
            f"Checkpoint found at {checkpoint_path}\n"
            f"Skipping Stage 1 and loading results from checkpoint...\n",
            flush=True,
        )
        # Restore individual result files from checkpoint into results_dir
        checkpoint_data = pd.read_pickle(checkpoint_path)
        for (k, seed), result in checkpoint_data.items():
            result_path = os.path.join(results_dir, f"k{k}_seed{seed}.pkl")
            pd.to_pickle(result, result_path)
        print(f"Checkpoint loaded: {len(checkpoint_data)} results restored.\n", flush=True)

    else:
        # --------------------------------------------------------------
        # Stage 1: parallel clustering
        # ALL (k x seed) tasks submitted at once so all cores stay busy.
        # Results streamed to disk as they arrive — avoids holding all
        # results in RAM simultaneously.
        # --------------------------------------------------------------
        all_tasks = [(k, seed) for k in ks for seed in seeds]
        n_total   = len(all_tasks)
        print(
            f"Stage 1 — Clustering: {n_total} tasks "
            f"({len(list(ks))} k-values × {len(seeds)} seeds)\n",
            flush=True,
        )

        checkpoint_data = {}  # (k, seed) -> result dict, for checkpoint saving

        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=worker_init,
            initargs=(
                filtered_pickle_path,
                args.data_type,
                args.k_start,
                args.k_end,
                args.conditions,
                args.ts_length,
                args.ts_dimensions,
                args.exclude_full,
                args.metric,
                args.n_seeds,
                args.n_init,
                # args.max_iterations,
                args.kernel_metric,
            ),
        ) as executor:
            futures = {executor.submit(cluster_worker, t): t for t in all_tasks}

            completed = 0
            for fut in as_completed(futures):
                res = fut.result()
                k, seed = res["k"], res["seed"]

                # Stream result to disk immediately — don't accumulate in RAM
                result_path = os.path.join(results_dir, f"k{k}_seed{seed}.pkl")
                pd.to_pickle(res, result_path)

                # Also keep a lightweight reference for checkpoint saving
                checkpoint_data[(k, seed)] = res

                completed += 1
                if completed % 50 == 0 or completed == n_total:
                    print(f"  Clustering progress: {completed}/{n_total} tasks done", flush=True)

        # Save checkpoint after Stage 1 completes successfully
        pd.to_pickle(checkpoint_data, checkpoint_path)
        print(
            f"\nStage 1 complete. Checkpoint saved to: {checkpoint_path}\n"
            f"If Stage 2 crashes, rerun the same command to resume from here.\n",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Stage 2: parallel metrics
    # One task per k-value. Each worker loads its own result files from
    # disk — no GLOBAL variables needed, no large arrays passed via queue.
    # ------------------------------------------------------------------
    metrics_input = [(k, results_dir, seeds) for k in ks]
    n_metrics     = len(metrics_input)
    print(
        f"Stage 2 — Metrics: {n_metrics} k-values | "
        f"metrics_workers={metrics_workers}\n",
        flush=True,
    )

    # Silhouette early-stop state (rolling patience counter)
    sil_below_threshold_count = 0
    sil_stopped               = False
    metrics_by_k              = {}

    # We process metrics in k order so the patience counter is meaningful.
    # This means we can't submit all futures at once if sil_min is set —
    # we need to check the result of k before deciding whether to skip k+1.
    if args.sil_min is not None:
        # Sequential submission with early-stop check between k-values.
        # Still parallel within a batch of metrics_workers k-values at a time.
        print(
            f"Silhouette early-stop enabled: sil_min={args.sil_min}, "
            f"patience={args.sil_patience}\n",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=metrics_workers) as metrics_executor:
            for k, results_dir_arg, seeds_arg in metrics_input:
                if sil_stopped:
                    # Fill remaining k-values with NaN
                    metrics_by_k[k] = {
                        "k": k, "inertia": np.nan,
                        "silhouette": np.nan, "stability": np.nan,
                    }
                    continue

                res = metrics_executor.submit(
                    metrics_worker, (k, results_dir_arg, seeds_arg)
                ).result()
                metrics_by_k[res["k"]] = res

                # Update patience counter
                if res["silhouette"] is not np.nan and not np.isnan(res["silhouette"]):
                    if res["silhouette"] < args.sil_min:
                        sil_below_threshold_count += 1
                        if sil_below_threshold_count >= args.sil_patience:
                            print(
                                f"  Silhouette below {args.sil_min} for {args.sil_patience} "
                                f"consecutive k-values. Skipping silhouette for remaining k.\n",
                                flush=True,
                            )
                            sil_stopped = True
                    else:
                        # Reset counter — silhouette recovered
                        sil_below_threshold_count = 0
    else:
        # No early-stop: submit all metrics tasks at once for maximum parallelism
        with ProcessPoolExecutor(max_workers=metrics_workers) as metrics_executor:
            metrics_futures = {
                metrics_executor.submit(metrics_worker, t): t for t in metrics_input
            }
            completed = 0
            for fut in as_completed(metrics_futures):
                res = fut.result()
                metrics_by_k[res["k"]] = res
                completed += 1
                if completed % 10 == 0 or completed == n_metrics:
                    print(f"  Metrics progress: {completed}/{n_metrics} k-values done", flush=True)

    print(f"\nStage 2 complete.\n", flush=True)

    # Reassemble results in correct k order
    inertias    = [metrics_by_k[k]["inertia"]    for k in ks]
    silhouettes = [metrics_by_k[k]["silhouette"] for k in ks]
    stabilities = [metrics_by_k[k]["stability"]  for k in ks]

    # ------------------------------------------------------------------
    # Save results to TSV
    # ------------------------------------------------------------------
    results_df = pd.DataFrame({
        "k":          list(ks),
        "inertia":    inertias,
        "silhouette": silhouettes,
        "stability":  stabilities,
    })

    conditions_label = "+".join(c.strip("_") for c in args.conditions)
    results_tsv      = f"{args.output_file_path}_qc_metrics_{conditions_label}.tsv"
    results_df.to_csv(results_tsv, sep="\t", index=False)
    print(f"Metrics saved to: {results_tsv}")

    # ------------------------------------------------------------------
    # Plot results
    # ------------------------------------------------------------------
    k_list = list(ks)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Kernel Kmeans Cluster QC Metrics | {conditions_label}", fontsize=14, fontweight="bold")

    for metric_name, values, ylabel, subplot in [
        ("inertia",    inertias,    "Inertia",                       0),
        ("silhouette", silhouettes, "Silhouette Score",              1),
        ("stability",  stabilities, "Mean Pairwise ARI (stability)", 2),
    ]:
        ax[subplot].scatter(k_list, values, s=20)
        ax[subplot].set_title(f"{ylabel} vs Number of Clusters")
        ax[subplot].set_xlabel("Number of Clusters (k)")
        ax[subplot].set_ylabel(ylabel)
        ax[subplot].grid(True, alpha=0.3)

    fig.tight_layout()
    out_png = f"{args.output_file_path}_metrics_{conditions_label}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Image saved: {out_png}")

    # ------------------------------------------------------------------
    # Cleanup: remove temp directory (pickle files)
    # Checkpoint is kept intentionally for potential re-runs
    # ------------------------------------------------------------------
    import shutil
    shutil.rmtree(tmpdir)
    print(
        f"\nTemp files cleaned up.\n"
        f"Checkpoint kept at: {checkpoint_path}\n"
        f"  (Delete manually if no longer needed)\n"
    )
    print("Done.\n")


if __name__ == "__main__":
    _t0 = time.perf_counter()
    main()
    _t1 = time.perf_counter()
    elapsed = _t1 - _t0
    print(f"\nTotal time: {elapsed:.2f} s  ({elapsed / 60:.2f} min)")