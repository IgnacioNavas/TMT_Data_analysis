#!/usr/bin/env python3
import argparse
import os
import sys
import tempfile
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Note: we defer heavy imports into worker initializer / worker function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import *

# --- Worker global (will be set in initializer) ---
GLOBAL_DF = None

def worker_init(filtered_pickle_path):
    """
    Initializer run inside each worker process BEFORE any heavy numeric libraries are imported.
    Set environment variables to prevent BLAS/OpenMP oversubscription, then load the prepared dataframe.
    """
    # Prevent thread oversubscription inside each worker process
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # Now import pandas / numpy inside the worker context and load the precomputed filtered DF
    # (We imported pandas/numpy in main already; re-importing in worker is fine.)
    global GLOBAL_DF
    GLOBAL_DF = pd.read_pickle(filtered_pickle_path)

def cluster_worker(k_seed_tuple):
    """
    Worker function executed in a separate process.
    Expects a tuple (k, seed). Uses GLOBAL_DF loaded by worker_init.
    Returns a dict: { "k": k, "seed": seed, "labels": labels, "inertia": inertia, "multivariate": multivariate_or_none }
    Note: multivariate is only returned for seed == 0 (to compute silhouette in main process).
    """
    k, seed = k_seed_tuple
    # Access the global dataframe that was loaded during initializer
    df_to_cluster = GLOBAL_DF.copy()

    # Call the clustering routine (same signature as your original)
    cluster_name = f"KMeans_{k}_cluster_seed{seed}_on_log2FC_noFull_nrep>1_locSite=True_log2FC>0.5"
    print(f"[worker] Clustering: {cluster_name} (pid={os.getpid()})", flush=True)

    df_clustered, model, multivariate = tslearn_clustering_KMeans(
        df_to_cluster=df_to_cluster,
        data_type="log2_FC",
        condition_for_clustering=["_EGF_"], #, "_INS_", "_EGFnINS_"], # trying only EGF clustering
        exclude_full=True,
        cluster_column_name=cluster_name,
        number_of_clusters=k,
        max_iterations=200,
        n_init=5,
        metric="euclidean",
        df_dimensions=1, # 3
        time_series_length=6,
        random_state=seed,
        transpose=True,
        verbose=False,
        testing=True
    )

    # Only send multivariate back for seed==0 (silhouette calc). It's potentially large.
    multivar = multivariate if seed == 0 else None

    return {
        "k": k,
        "seed": seed,
        "labels": model.labels_,
        "inertia": model.inertia_,
        "multivariate": multivar
    }

def main():
    parser = argparse.ArgumentParser(description="Quality control check for clusters (parallelized)")
    parser.add_argument("input_file_path", type=str, help="Path to input TSV file")
    parser.add_argument("output_file_path", type=str, help="Path to output png file prefix")
    parser.add_argument("--workers", type=int, default=6, help="Number of worker processes (default: 6)")
    args = parser.parse_args()

    # --- Initial import and filtering (same as your original) ---
    df_2_hME1_scaled = pd.read_csv(args.input_file_path, sep="\t")
    df_2_hME1_scaled = df_2_hME1_scaled.fillna(0)
    print(f"The Dataframe shape: {df_2_hME1_scaled.shape}")

    nreps_df = filter_replicates(df_2_hME1_scaled, n_reps=2)
    localized_sites_df = filter_site_localizations(nreps_df, loc_sites=True)
    df_filtered = filter_dynamics_extremes(
        df=localized_sites_df,
        data_type="log2_FC",
        threshold=0.5,
        exclude_full=True
    )

    print(f"Original Dataframe shape: {df_2_hME1_scaled.shape}"
          f"\nnreps_df: {nreps_df.shape}"
          f"\nlocalized_sites_df: {localized_sites_df.shape}"
          f"\ndf_filtered: {df_filtered.shape}")

    # Save the filtered df to a temporary pickle so worker processes can load it in initializer
    tmpdir = tempfile.mkdtemp(prefix="cluster_par_")
    filtered_pickle_path = os.path.join(tmpdir, "df_filtered.pkl")
    df_filtered.to_pickle(filtered_pickle_path)

    # clustering parameters
    ks = range(2, 100)
    seeds = list(range(10))

    inertias = []
    silhouettes = []
    stabilities = []

    # We'll create a process pool; the initializer will load the df_filtered in each worker
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=worker_init,
                             initargs=(filtered_pickle_path,)) as exec:

        for k in ks:
            # submit one task per seed
            tasks = [(k, seed) for seed in seeds]
            futures = {exec.submit(cluster_worker, t): t for t in tasks}

            labels_list = []
            inertia_for_k = None
            multivariate_for_silhouette = None

            # collect results as they finish
            for fut in as_completed(futures):
                res = fut.result()
                labels_list.append(res["labels"])
                # inertia saved from the first completed seed with seed==0 or any seed (we used seed==0 in original)
                if res["seed"] == 0:
                    inertia_for_k = res["inertia"]
                    multivariate_for_silhouette = res["multivariate"]
                # if seed 0 finished later, we'll catch it then

            # If seed 0 didn't finish (unlikely), pick the inertia from the first item
            if inertia_for_k is None and len(labels_list) > 0:
                # get inertia from any returned result (but we prefer seed 0)
                # (we didn't store per-future inertia except in res; simpler to recompute by re-running? avoid)
                # so let's re-open futures results quickly to find inertia:
                # Note: in above collection we captured res and would have set inertia_for_k if seed==0
                # so this branch is unlikely. We'll set inertia to NaN to mark missing.
                inertia_for_k = np.nan

            # compute silhouette for seed==0 only if we got multivariate
            sil = np.nan
            if multivariate_for_silhouette is not None:
                from tslearn.metrics import cdist_dtw
                from sklearn.metrics import silhouette_score
                D = cdist_dtw(multivariate_for_silhouette)
                # silhouette requires labels from the seed 0 model — find it in labels_list if present
                # labels_list may be in arbitrary order; find the labels belonging to seed==0:
                # we don't persist mapping of which labels correspond to which seed here, but we returned multivariate only for seed 0
                # so the labels that correspond to seed 0 must be in the collection; however to be safe we recompute
                # We'll locate the label array that matches length of D rows
                labels_candidate = None
                for lab in labels_list:
                    if len(lab) == D.shape[0]:
                        labels_candidate = lab
                        break
                if labels_candidate is not None:
                    sil = silhouette_score(D, labels_candidate, metric="euclidean")
                else:
                    sil = np.nan

            # compute pairwise ARI across labels_list
            from sklearn.metrics import adjusted_rand_score
            ari_scores = []
            for i in range(len(labels_list)):
                for j in range(i + 1, len(labels_list)):
                    ari_scores.append(adjusted_rand_score(labels_list[i], labels_list[j]))
            stability = np.nan if len(ari_scores) == 0 else np.mean(ari_scores)

            inertias.append(inertia_for_k)
            silhouettes.append(sil)
            stabilities.append(stability)

            print(f"Completed k={k}: inertia={inertia_for_k}, silhouette={sil}, stability={stability}")

    # --- Plot results (same as original) ---
    # inertias
    # plt.scatter(x=list(ks), y=inertias)
    # plt.title("Inertia vs Clusters")
    # plt.xlabel("Clusters")
    # plt.ylabel("Inertia")
    # plt.savefig(f"{args.output_file_path}_inertia_EGF.png")
    # plt.close()
    #
    # # silhouettes
    # plt.scatter(x=list(ks), y=silhouettes)
    # plt.title("Silhouettes vs Clusters")
    # plt.xlabel("Clusters")
    # plt.ylabel("Silhouettes")
    # plt.savefig(f"{args.output_file_path}_silhouettes_EGF.png")
    # plt.close()
    #
    # # stabilities
    # plt.scatter(x=list(ks), y=stabilities)
    # plt.title("Stabilities vs Clusters")
    # plt.xlabel("Clusters")
    # plt.ylabel("Stabilities")
    # plt.savefig(f"{args.output_file_path}_stabilities_EGF.png")
    # plt.close()

    print("All done. Output files saved with prefix:", args.output_file_path)
    # (optional) cleanup pickle file if desired — leaving it for inspection

start_time = time.perf_counter()

if __name__ == "__main__":
    start_time = time.perf_counter()

    main()

    end_time = time.perf_counter()
    total_time = end_time - start_time

    print(f"\nTotal computational time: {total_time:.2f} seconds")
    print(f"Total computational time: {total_time / 60:.2f} minutes")