"""
Adaptive (divisive + agglomerative) time-series clustering.

This module implements a single higher-level clustering strategy,
`adaptive_kmeans_clustering`, built on top of tslearn's TimeSeriesKMeans (the
same engine used by `tslearn_clustering_KMeans` in src/clustering.py). Instead
of asking the user to fix the number of clusters up front, it adapts the number
of clusters to the data in two phases:

  PHASE 1 — DIVISIVE (split loose clusters)
    1. Run KMeans with `initial_n_clusters` on the whole dataset.
    2. For every cluster, measure its inertia (a compactness score — see below).
    3. Any cluster whose inertia is above `inertia_threshold` is considered too
       loose and is split into `n_subclusters` sub-clusters with another KMeans.
    4. Each new sub-cluster is measured again and split again if still too loose.
    5. This repeats at most `max_subdivision_rounds` times (default 3) along any
       branch. When that depth is reached a cluster is kept even if it is still
       above the threshold ("stop anyway").

  PHASE 2 — AGGLOMERATIVE (merge redundant clusters)
    6. Compute the centroid of every cluster produced by phase 1.
    7. Find the closest pair of centroids. If their distance is below
       `merge_distance_threshold`, tentatively merge them.
    8. Accept the merge ONLY IF the merged cluster's inertia stays at or below
       `inertia_threshold` (the same ceiling used to split). Otherwise the pair
       is rejected and left separate.
    9. Repeat until no remaining pair of centroids is closer than the threshold
       (or every close-enough pair has been rejected by the inertia guard).

The net effect: dense regions of the data get resolved into several tight
clusters, sparse regions are not over-split, and near-duplicate clusters that
KMeans happened to create are collapsed back together — all controlled by two
interpretable knobs (`inertia_threshold`, `merge_distance_threshold`).

--------------------------------------------------------------------------------
A NOTE ON "INERTIA" AND UNITS
--------------------------------------------------------------------------------
For one cluster, let d_i be the Euclidean distance from point i to the cluster
centroid (computed on the flattened multivariate time-series vector). We offer
three ways to summarise a cluster's inertia via `inertia_mode`:

  * "total" — sum_i d_i**2            (classic k-means inertia; grows with cluster size)
  * "mean"  — mean_i d_i**2           (average squared distance; size-independent) [default]
  * "rms"   — sqrt(mean_i d_i**2)     (root-mean-square distance; same units as the data)

"mean" is the default because a size-independent score makes a single
`inertia_threshold` behave consistently for both small and large clusters, and
because merging (which always adds points) then stays comparable to splitting.
Note that `merge_distance_threshold` is a plain Euclidean distance between two
centroid vectors, so it is in the *same units as the data*; if you use
inertia_mode="rms" the two thresholds are directly comparable, whereas "mean"
and "total" are in squared units.

--------------------------------------------------------------------------------
SCOPE / LIMITATIONS
--------------------------------------------------------------------------------
All inertia, centroid and merge computations are done with Euclidean distance in
the flattened feature space, so `metric` must be "euclidean" (the same default
as `tslearn_clustering_KMeans`). DTW-based inertia is a planned extension and is
intentionally not supported yet, to keep the split/merge maths self-consistent.
"""

from collections import deque
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from tslearn.clustering import TimeSeriesKMeans

from src.column_spec import ColumnSpec
from src.clustering import reshape_df, consensus_stability


# =============================================================================
# Result container
# =============================================================================
@dataclass
class AdaptiveClusteringResult:
    """
    Everything produced by a single `adaptive_kmeans_clustering` run.

    Attributes:
        labels: np.ndarray of shape (n_sites,) — final integer cluster label per
            site, aligned with the rows of the input DataFrame.
        X: np.ndarray — the reshaped multivariate array that was clustered,
            shape (n_sites, dim_a, dim_b) as produced by reshape_df().
        names: pd.Series — site identifiers aligned with `labels` / rows of X.
        centroids: np.ndarray of shape (n_final_clusters, n_features) — centroid
            of each final cluster in the flattened feature space.
        cluster_inertias: dict mapping final cluster label -> inertia value
            (computed with `inertia_mode`).
        cluster_sizes: dict mapping final cluster label -> number of sites.
        n_clusters_initial: number of clusters from the very first KMeans fit.
        n_clusters_after_split: number of clusters after the divisive phase.
        n_clusters_final: number of clusters after the merge phase.
        split_log: list of dicts describing every split decision (phase 1).
        merge_log: list of dicts describing every merge decision (phase 2).
        params: dict of the parameters the run was called with.
    """
    labels: np.ndarray
    X: np.ndarray
    names: pd.Series
    centroids: np.ndarray
    cluster_inertias: dict
    cluster_sizes: dict
    n_clusters_initial: int
    n_clusters_after_split: int
    n_clusters_final: int
    split_log: list = field(default_factory=list)
    merge_log: list = field(default_factory=list)
    params: dict = field(default_factory=dict)


# =============================================================================
# Low-level inertia / centroid helpers (Euclidean, flattened feature space)
# =============================================================================
def _flatten(X):
    """
    Flatten a multivariate time-series array to one feature vector per sample.

    Args:
        X: np.ndarray of shape (n_samples, dim_a, dim_b) (e.g. the output of
           reshape_df, either (n, conditions, timepoints) or transposed).

    Returns:
        np.ndarray of shape (n_samples, dim_a * dim_b).
    """
    return X.reshape(X.shape[0], -1)
    # Question: isn't this flattening a problem if I have a multivariate dataset? then instead of one time series with many dimensions I would have one long time series.
    # Answer: For EUCLIDEAN k-means it is NOT a problem — it is mathematically exact.
    #   Squared Euclidean distance between two samples is the sum of squared
    #   differences over ALL elements. Flattening only reorders those elements; it
    #   does not change the sum. So the pairwise distances, the centroids (means)
    #   and the inertia are identical whether computed on the (dims, timepoints)
    #   matrix or on the flattened (dims*timepoints,) vector. Each element keeps its
    #   own coordinate — dimensions are NOT blended into "one long time series" in
    #   any way that affects the maths.
    #   The ONE case where flattening WOULD be wrong is DTW: DTW warps along the time
    #   axis, and concatenating conditions end-to-end would let it warp ACROSS the
    #   boundary between conditions (e.g. align EGF's tail to INS's head), which is
    #   meaningless. That is exactly why this module is Euclidean-only (see the
    #   SCOPE note in the module docstring). To weight a condition more heavily, use
    #   the stoichiometry repeat (it adds columns for that condition, increasing its
    #   share of the distance) rather than changing the flattening.


def _centroid(X_flat):
    """
    Euclidean centroid (mean vector) of a set of flattened samples.

    Args:
        X_flat: np.ndarray of shape (n_samples, n_features).

    Returns:
        np.ndarray of shape (n_features,) — the mean over the samples axis.
    """
    return X_flat.mean(axis=0)
    # Question: what is this for? what is the use of the centroid? am I not losing the information of the shape of the time course?
    # Answer: The centroid is the MEAN PROFILE of every site in a cluster — the
    #   per-feature (per condition x timepoint) average. It is used in three places:
    #     1. inertia / compactness — how far members sit from it (phase 1 splitting),
    #     2. the merge test — distance between two clusters' centroids (phase 2),
    #     3. the representative "cluster shape" for plotting.
    #   You do NOT lose the shape. Because the mean is taken element-by-element on the
    #   flattened vector, the centroid is itself a full time course: reshape it back
    #   with _vector_to_series() and you recover the mean curve per condition. It is
    #   the AVERAGE shape of the cluster, not a single number.
    #   What the centroid alone does not capture is the SPREAD of shapes around that
    #   mean (the within-cluster variability). That spread is exactly what inertia
    #   measures, and what diagnose_cluster_substructure() inspects (close-half vs
    #   far-half) to check whether one cluster is really hiding two different shapes.


def _squared_distances_to_centroid(X_flat,
                                   centroid,):
    """
    Squared Euclidean distance from each sample to a centroid.

    Args:
        X_flat: np.ndarray of shape (n_samples, n_features).
        centroid: np.ndarray of shape (n_features,).

    Returns:
        np.ndarray of shape (n_samples,) — squared distance d_i**2 for each sample.
    """
    diff = X_flat - centroid
    return np.einsum("ij,ij->i", diff, diff)


def _inertia_from_squared(sq_distances,
                          mode,):
    """
    Reduce per-sample squared distances to a single cluster inertia value.

    Args:
        sq_distances: np.ndarray of squared distances d_i**2 (one per sample).
        mode: "total" (sum of squared distances), "mean" (mean squared distance),
              or "rms" (root-mean-square distance, same units as the data).

    Returns:
        float — the cluster inertia under the requested mode. An empty input
        returns 0.0.
    """
    n = len(sq_distances)
    if n == 0:
        return 0.0
    total = float(sq_distances.sum())
    if mode == "total":
        return total
    if mode == "mean":
        return total / n
    if mode == "rms":
        return float(np.sqrt(total / n))
    raise ValueError(f"inertia_mode must be 'total', 'mean' or 'rms', got {mode!r}")


def cluster_inertia(X_flat,
                    centroid=None,
                    mode="mean",):
    """
    Inertia (compactness score) of a single cluster.

    Args:
        X_flat: np.ndarray of shape (n_samples, n_features) — the cluster's
                samples in flattened feature space.
        centroid: optional precomputed centroid; if None it is computed as the
                  mean of X_flat.
        mode: "total", "mean" (default) or "rms" — see module docstring.

    Returns:
        float — the cluster's inertia value.
    """
    if centroid is None:
        centroid = _centroid(X_flat)
    sq = _squared_distances_to_centroid(X_flat, centroid)
    return _inertia_from_squared(sq, mode)


def compute_cluster_inertias(X,
                             labels,
                             mode="mean",):
    """
    Inertia of every cluster in a labelling.

    Useful for choosing `inertia_threshold`: run an initial KMeans, then inspect
    the distribution of per-cluster inertias to pick a sensible cut-off.

    Args:
        X: np.ndarray of shape (n_samples, dim_a, dim_b) — the multivariate data.
        labels: array-like of shape (n_samples,) — cluster label per sample.
        mode: "total", "mean" (default) or "rms".

    Returns:
        dict mapping cluster label -> inertia value.
    """
    X_flat = _flatten(np.asarray(X))
    labels = np.asarray(labels)
    out = {}
    for lab in np.unique(labels):
        out[int(lab)] = cluster_inertia(X_flat[labels == lab], mode=mode)
    return out


# =============================================================================
# KMeans fit wrapper
# =============================================================================
def _fit_kmeans(X,
                n_clusters,
                metric,
                max_iterations,
                n_init,
                random_state,):
    """
    Fit a TimeSeriesKMeans and return the integer labels.

    Thin wrapper so the divisive phase and the initial fit share identical
    KMeans settings. Runs silently (verbose=False) because the adaptive
    algorithm prints its own progress.

    Args:
        X: np.ndarray of shape (n_samples, dim_a, dim_b) to cluster.
        n_clusters: number of clusters to fit.
        metric: distance metric for TimeSeriesKMeans (only "euclidean" supported
                by the adaptive algorithm).
        max_iterations: max_iter for TimeSeriesKMeans.
        n_init: number of KMeans initialisations.
        random_state: random seed.

    Returns:
        np.ndarray of shape (n_samples,) — integer cluster labels 0..n_clusters-1.
    """
    model = TimeSeriesKMeans(n_clusters=n_clusters,
                             metric=metric,
                             max_iter=max_iterations,
                             n_init=n_init,
                             max_iter_barycenter=1000,
                             verbose=False,
                             random_state=random_state,).fit(X)
    return model.labels_


# =============================================================================
# Phase 1 — divisive splitting
# =============================================================================
def _divisive_split(X,
                    initial_labels,
                    inertia_threshold,
                    inertia_mode,
                    n_subclusters,
                    max_subdivision_rounds,
                    min_cluster_size,
                    metric,
                    max_iterations,
                    n_init,
                    random_state,
                    verbose,):
    """
    Recursively split clusters whose inertia exceeds the threshold.

    Starting from an initial labelling, each cluster is examined; if it is too
    loose (inertia > `inertia_threshold`) and has not yet been split
    `max_subdivision_rounds` times, it is divided into `n_subclusters` pieces
    with KMeans and each piece is re-examined. The traversal is breadth-first
    over a work queue of (sample indices, depth) items.

    Args:
        X: np.ndarray of shape (n_samples, dim_a, dim_b) — the full dataset.
        initial_labels: np.ndarray of shape (n_samples,) — labels from the first
                        KMeans fit; each distinct label seeds the queue at depth 0.
        inertia_threshold: clusters with inertia above this are candidates to split.
        inertia_mode: "total", "mean" or "rms" (see module docstring).
        n_subclusters: how many pieces to split a loose cluster into (>= 2).
        max_subdivision_rounds: maximum number of successive splits along any
                                branch (the "stop anyway" depth limit).
        min_cluster_size: clusters with fewer samples than this are never split.
        metric: KMeans metric ("euclidean").
        max_iterations, n_init, random_state: KMeans settings.
        verbose: if True, print each split/stop decision.

    Returns:
        final_labels: np.ndarray of shape (n_samples,) — contiguous leaf-cluster
                      labels 0..K-1.
        split_log: list of dicts, one per examined cluster, describing the action
                   taken ("split" or "stop") and why.
    """
    X_flat = _flatten(X)
    initial_labels = np.asarray(initial_labels)

    queue = deque()
    for lab in np.unique(initial_labels):
        queue.append((np.where(initial_labels == lab)[0], 0))

    leaves = []       # list of index arrays, one per final leaf cluster
    split_log = []

    while queue:
        idx, depth = queue.popleft()
        inertia = cluster_inertia(X_flat[idx], mode=inertia_mode)

        too_loose = inertia > inertia_threshold
        can_split = (depth < max_subdivision_rounds
                     and len(idx) >= max(min_cluster_size, 2)
                     and len(idx) >= n_subclusters)

        # Decide whether to split this cluster or keep it as a leaf.
        if not (too_loose and can_split):
            reason = ("compact" if not too_loose
                      else "max_rounds" if depth >= max_subdivision_rounds
                      else "too_small")
            leaves.append(idx)
            split_log.append({"depth": depth,
                              "n_sites": int(len(idx)),
                              "inertia": inertia,
                              "action": "stop",
                              "reason": reason,})
            if verbose:
                print(f"  [depth {depth}] keep cluster: {len(idx):5d} sites, "
                      f"inertia={inertia:.4f}  ({reason})")
            continue

        # Split this cluster with KMeans.
        k = min(n_subclusters, len(idx))
        sub_labels = _fit_kmeans(X[idx], k, metric, max_iterations, n_init, random_state)
        produced = [idx[sub_labels == s] for s in np.unique(sub_labels)]
        produced = [p for p in produced if len(p) > 0]

        # If KMeans failed to actually divide the cluster, keep it as a leaf.
        if len(produced) < 2:
            leaves.append(idx)
            split_log.append({"depth": depth,
                              "n_sites": int(len(idx)),
                              "inertia": inertia,
                              "action": "stop",
                              "reason": "split_collapsed",})
            if verbose:
                print(f"  [depth {depth}] keep cluster: {len(idx):5d} sites, "
                      f"inertia={inertia:.4f}  (split produced <2 groups)")
            continue

        split_log.append({"depth": depth,
                          "n_sites": int(len(idx)),
                          "inertia": inertia,
                          "action": "split",
                          "into": [int(len(p)) for p in produced],})
        if verbose:
            sizes = ", ".join(str(len(p)) for p in produced)
            print(f"  [depth {depth}] SPLIT cluster: {len(idx):5d} sites, "
                  f"inertia={inertia:.4f} -> sub-sizes [{sizes}]")

        for p in produced:
            queue.append((p, depth + 1))

    # Assign contiguous 0..K-1 labels to the leaf clusters.
    final_labels = np.empty(len(X_flat), dtype=int)
    for new_lab, idx in enumerate(leaves):
        final_labels[idx] = new_lab

    return final_labels, split_log


# =============================================================================
# Phase 2 — agglomerative merging
# =============================================================================
def _merge_close_clusters(X,
                          labels,
                          merge_distance_threshold,
                          inertia_threshold,
                          inertia_mode,
                          verbose,):
    """
    Merge clusters whose centroids are close, guarded by the inertia ceiling.

    Repeatedly finds the closest pair of centroids; if they are within
    `merge_distance_threshold` it merges them, but only if the resulting cluster
    keeps its inertia at or below `inertia_threshold`. Pairs that fail the
    inertia guard are remembered and not retried unless one of the two clusters
    changes through a later merge (in which case all rejections are cleared,
    since moved centroids may now merge acceptably).

    Args:
        X: np.ndarray of shape (n_samples, dim_a, dim_b) — the full dataset.
        labels: np.ndarray of shape (n_samples,) — labels from the divisive phase.
        merge_distance_threshold: max Euclidean distance between two centroids for
                                  them to be considered mergeable.
        inertia_threshold: merged clusters may not exceed this inertia.
        inertia_mode: "total", "mean" or "rms".
        verbose: if True, print each accepted/rejected merge.

    Returns:
        merged_labels: np.ndarray of shape (n_samples,) — contiguous labels
                       0..K'-1 after merging.
        merge_log: list of dicts describing every merge decision.
    """
    X_flat = _flatten(X)
    labels = np.asarray(labels).copy()

    active = [int(l) for l in np.unique(labels)]
    centroids = {l: _centroid(X_flat[labels == l]) for l in active}
    rejected = set()   # frozenset({a, b}) pairs rejected by the inertia guard
    merge_log = []

    while True:
        # Find the closest still-mergeable pair of centroids.
        best = None   # (distance, a, b)
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                a, b = active[i], active[j]
                if frozenset((a, b)) in rejected:
                    continue
                dist = float(np.linalg.norm(centroids[a] - centroids[b]))
                if dist < merge_distance_threshold and (best is None or dist < best[0]):
                    best = (dist, a, b)

        if best is None:
            break   # no pair close enough (or all such pairs rejected)

        dist, a, b = best
        merged_mask = (labels == a) | (labels == b)
        merged_inertia = cluster_inertia(X_flat[merged_mask], mode=inertia_mode)

        if merged_inertia <= inertia_threshold:
            # Accept: fold b into a and refresh a's centroid.
            labels[labels == b] = a
            centroids[a] = _centroid(X_flat[labels == a])
            del centroids[b]
            active.remove(b)
            rejected.clear()   # centroids moved; previously-rejected pairs may now qualify
            merge_log.append({"merged": (a, b),
                              "distance": dist,
                              "merged_inertia": merged_inertia,
                              "action": "accept",})
            if verbose:
                print(f"  MERGE clusters {a} + {b}: dist={dist:.4f}, "
                      f"merged inertia={merged_inertia:.4f}  (accepted)")
        else:
            rejected.add(frozenset((a, b)))
            merge_log.append({"merged": (a, b),
                              "distance": dist,
                              "merged_inertia": merged_inertia,
                              "action": "reject",
                              "reason": "inertia_exceeds_threshold",})
            if verbose:
                print(f"  keep clusters {a} + {b} separate: dist={dist:.4f}, "
                      f"merged inertia={merged_inertia:.4f} > {inertia_threshold}  (rejected)")

    # Relabel remaining clusters to contiguous 0..K'-1.
    remap = {old: new for new, old in enumerate(sorted(set(labels.tolist())))}
    merged_labels = np.array([remap[l] for l in labels], dtype=int)

    return merged_labels, merge_log


# =============================================================================
# Main entry point
# =============================================================================
def adaptive_kmeans_clustering(df_to_cluster,
                               data_type,
                               inertia_threshold,
                               merge_distance_threshold,
                               condition_for_clustering=None,
                               cell_lines=None,
                               exclude_full=False,
                               stoichiometry=None,
                               cluster_column_name="adaptive_cluster",
                               initial_n_clusters=10,
                               inertia_mode="mean",
                               n_subclusters=2,
                               max_subdivision_rounds=3,
                               min_cluster_size=5,
                               metric="euclidean",
                               max_iterations=1000,
                               n_init=5,
                               df_dimensions=None,
                               time_series_length=None,
                               random_state=0,
                               transpose=False,
                               verbose=True,
                               testing=False,):
    """
    Cluster phosphosite time-series with adaptive divisive-then-agglomerative KMeans.

    The data are selected and reshaped exactly like `tslearn_clustering_KMeans`
    (via ColumnSpec.select + reshape_df), then clustered with the two-phase
    algorithm described in this module's docstring: loose clusters are split
    until compact (or a depth limit is hit), then near-duplicate clusters are
    merged back together under an inertia guard. Final labels are written into
    `df_to_cluster[cluster_column_name]`.

    Args:
        df_to_cluster: DataFrame following the project naming convention. A new
            column `cluster_column_name` is added in place with the labels.
        data_type: data-type string for column selection, e.g. "log2:FC".
        inertia_threshold: compactness ceiling. In phase 1 a cluster is split if
            its inertia is ABOVE this; in phase 2 a merge is rejected if it would
            push inertia above this. Units depend on `inertia_mode` (see module
            docstring). Choose it by inspecting `compute_cluster_inertias()` on an
            initial KMeans fit.
        merge_distance_threshold: maximum Euclidean distance between two cluster
            centroids (in the flattened feature space, i.e. the same units as the
            data) for them to be considered for merging.
        condition_for_clustering: list of condition substrings, e.g.
            ["_EGF_", "_INS_", "_EGFnINS_"].
        cell_lines: list of cell-line prefixes, e.g. ["WT"].
        exclude_full: if True, drop the 'full' timepoint before clustering.
        stoichiometry: optional per-condition reweighting dict passed to
            reshape_df (requires transpose=True).
        cluster_column_name: name of the output label column (default
            "adaptive_cluster").
        initial_n_clusters: number of clusters for the first KMeans fit (default 10).
        inertia_mode: "total", "mean" (default) or "rms" — how a cluster's inertia
            is summarised. See module docstring.
        n_subclusters: number of pieces a loose cluster is split into each round
            (default 2, i.e. binary division).
        max_subdivision_rounds: maximum successive splits along any branch
            (default 3). After this a cluster is kept even if still too loose.
        min_cluster_size: clusters with fewer sites than this are never split
            (default 5).
        metric: distance metric for the KMeans fits. Only "euclidean" is
            supported (raises NotImplementedError otherwise), because the inertia
            and merge maths are Euclidean.
        max_iterations: max_iter for each KMeans fit (default 1000).
        n_init: KMeans initialisations per fit (default 5).
        df_dimensions: number of (cell_line x condition) dimensions of the
            multivariate series. REQUIRED (as in tslearn_clustering_KMeans).
        time_series_length: number of timepoints per dimension. REQUIRED.
        random_state: random seed for reproducibility (default 0).
        transpose: passed to reshape_df; if True the reshaped array is
            (n_sites, timepoints, dimensions).
        verbose: if True, print column selection and every split/merge decision.
        testing: if True, also return the AdaptiveClusteringResult object.

    Returns:
        If testing is False: df_to_cluster with the new label column.
        If testing is True: (df_to_cluster, AdaptiveClusteringResult).
    """
    if condition_for_clustering is None:
        condition_for_clustering = []
    if cell_lines is None:
        cell_lines = []
    if df_dimensions is None:
        raise ValueError("df_dimensions is required")
    if time_series_length is None:
        raise ValueError("time_series_length is required")
    if metric != "euclidean":
        raise NotImplementedError(
            "adaptive_kmeans_clustering currently supports only metric='euclidean'; "
            "DTW-based inertia is a planned extension."
        )
    if inertia_mode not in ("total", "mean", "rms"):
        raise ValueError(f"inertia_mode must be 'total', 'mean' or 'rms', got {inertia_mode!r}")

    # --- Column selection + reshape (identical to tslearn_clustering_KMeans) ---
    column_selection = ColumnSpec.select(df_to_cluster,
                                         cell_lines=cell_lines,
                                         data_type=data_type,
                                         conditions=condition_for_clustering,
                                         exclude_full=exclude_full,)
    if verbose:
        print(f"Column selection ({len(column_selection)}): {column_selection}\n")

    X, names = reshape_df(df=df_to_cluster,
                          time_series=column_selection,
                          dimensions=df_dimensions,
                          len_time_serie=time_series_length,
                          stoichiometry=stoichiometry,
                          transpose=transpose,
                          labels="site",
                          verbose=verbose,)
    if verbose:
        print(f"\nDataset shape: {X.shape}")

    # --- Initial KMeans fit ---
    if verbose:
        print(f"\n[Phase 0] Initial KMeans with {initial_n_clusters} clusters")
    initial_labels = _fit_kmeans(X, initial_n_clusters, metric,
                                 max_iterations, n_init, random_state)
    n_clusters_initial = int(len(np.unique(initial_labels)))

    # --- Phase 1: divisive splitting ---
    if verbose:
        print(f"\n[Phase 1] Divisive splitting "
              f"(inertia_threshold={inertia_threshold}, mode={inertia_mode}, "
              f"n_subclusters={n_subclusters}, max_rounds={max_subdivision_rounds})")
    split_labels, split_log = _divisive_split(X=X,
                                              initial_labels=initial_labels,
                                              inertia_threshold=inertia_threshold,
                                              inertia_mode=inertia_mode,
                                              n_subclusters=n_subclusters,
                                              max_subdivision_rounds=max_subdivision_rounds,
                                              min_cluster_size=min_cluster_size,
                                              metric=metric,
                                              max_iterations=max_iterations,
                                              n_init=n_init,
                                              random_state=random_state,
                                              verbose=verbose,)
    n_clusters_after_split = int(len(np.unique(split_labels)))

    # --- Phase 2: agglomerative merging ---
    if verbose:
        print(f"\n[Phase 2] Merging close centroids "
              f"(merge_distance_threshold={merge_distance_threshold})")
    final_labels, merge_log = _merge_close_clusters(X=X,
                                                    labels=split_labels,
                                                    merge_distance_threshold=merge_distance_threshold,
                                                    inertia_threshold=inertia_threshold,
                                                    inertia_mode=inertia_mode,
                                                    verbose=verbose,)
    n_clusters_final = int(len(np.unique(final_labels)))

    # --- Write labels and assemble result ---
    df_to_cluster[cluster_column_name] = final_labels

    X_flat = _flatten(X)
    centroids = np.vstack([_centroid(X_flat[final_labels == l])
                           for l in sorted(np.unique(final_labels))])
    cluster_inertias = {int(l): cluster_inertia(X_flat[final_labels == l], mode=inertia_mode)
                        for l in np.unique(final_labels)}
    cluster_sizes = {int(l): int((final_labels == l).sum())
                     for l in np.unique(final_labels)}

    if verbose:
        print(f"\nDone. Clusters: {n_clusters_initial} (initial) -> "
              f"{n_clusters_after_split} (after split) -> {n_clusters_final} (final)")

    result = AdaptiveClusteringResult(labels=final_labels,
                                      X=X,
                                      names=names,
                                      centroids=centroids,
                                      cluster_inertias=cluster_inertias,
                                      cluster_sizes=cluster_sizes,
                                      n_clusters_initial=n_clusters_initial,
                                      n_clusters_after_split=n_clusters_after_split,
                                      n_clusters_final=n_clusters_final,
                                      split_log=split_log,
                                      merge_log=merge_log,
                                      params={"data_type": data_type,
                                              "conditions": condition_for_clustering,
                                              "cell_lines": cell_lines,
                                              "inertia_threshold": inertia_threshold,
                                              "merge_distance_threshold": merge_distance_threshold,
                                              "inertia_mode": inertia_mode,
                                              "initial_n_clusters": initial_n_clusters,
                                              "n_subclusters": n_subclusters,
                                              "max_subdivision_rounds": max_subdivision_rounds,
                                              "min_cluster_size": min_cluster_size,
                                              "cluster_column_name": cluster_column_name,
                                              # Reshape metadata — lets a result reconstruct its own
                                              # per-condition time series (see reconstruct_series_space).
                                              # NOTE: the pipeline applies NO per-site normalization
                                              # (no z-scoring); centroids and member curves are both in
                                              # the raw `data_type` magnitude space.
                                              "column_selection": column_selection,
                                              "df_dimensions": df_dimensions,
                                              "time_series_length": time_series_length,
                                              "transpose": transpose,
                                              "stoichiometry": stoichiometry,
                                              "exclude_full": exclude_full,},)

    if testing:
        return df_to_cluster, result
    return df_to_cluster


def adaptive_clustering_report(result,):
    """
    Print a human-readable summary of an AdaptiveClusteringResult.

    Shows the cluster-count trajectory, how many splits/merges happened, and a
    per-cluster table of size and inertia. Handy for understanding what the two
    phases did without digging through the raw logs.

    Args:
        result: an AdaptiveClusteringResult returned with testing=True.

    Returns:
        pd.DataFrame with one row per final cluster (columns: cluster, n_sites,
        inertia), also printed to stdout.
    """
    n_splits = sum(1 for e in result.split_log if e["action"] == "split")
    n_merges = sum(1 for e in result.merge_log if e["action"] == "accept")
    n_rejected = sum(1 for e in result.merge_log if e["action"] == "reject")

    print("Adaptive clustering summary")
    print("-" * 40)
    print(f"Initial clusters      : {result.n_clusters_initial}")
    print(f"Splits performed      : {n_splits}")
    print(f"Clusters after split  : {result.n_clusters_after_split}")
    print(f"Merges accepted       : {n_merges}  (rejected by inertia guard: {n_rejected})")
    print(f"Final clusters        : {result.n_clusters_final}")
    print(f"Inertia mode          : {result.params.get('inertia_mode')}")
    print("-" * 40)

    table = pd.DataFrame({"cluster": sorted(result.cluster_sizes),
                          "n_sites": [result.cluster_sizes[c] for c in sorted(result.cluster_sizes)],
                          "inertia": [result.cluster_inertias[c] for c in sorted(result.cluster_sizes)],})
    print(table.to_string(index=False))
    return table


# =============================================================================
# Assignment-confidence helpers (Euclidean distance to centroids)
# =============================================================================
def distance_to_centroids(result,):
    """
    Euclidean distance from every site to every final cluster centroid.

    All distances are in the same flattened feature space the clustering used
    (no normalization is applied by the pipeline), so they are directly
    comparable to `inertia_threshold` / `merge_distance_threshold`.

    Args:
        result: an AdaptiveClusteringResult (from testing=True).

    Returns:
        np.ndarray of shape (n_sites, n_clusters) — distance to each centroid.
    """
    X_flat = _flatten(result.X)
    return cdist(X_flat, result.centroids, metric="euclidean")


def assignment_margins(result,):
    """
    Per-site assignment margin = (distance to 2nd-nearest centroid) - (nearest).

    A large margin means the site sits clearly inside its cluster; a margin near
    zero means it is on a boundary between two clusters and its label is
    uncertain. With fewer than two clusters the margin is undefined (NaN).

    Args:
        result: an AdaptiveClusteringResult (from testing=True).

    Returns:
        np.ndarray of shape (n_sites,) — the margin for each site.
    """
    D = distance_to_centroids(result)
    if D.shape[1] < 2:
        return np.full(D.shape[0], np.nan)
    D_sorted = np.sort(D, axis=1)
    return D_sorted[:, 1] - D_sorted[:, 0]


# =============================================================================
# Time-series reconstruction (for plotting in the clustered space)
# =============================================================================
def _series_layout(result,):
    """
    Work out how a flattened feature vector maps back to per-condition time series.

    Uses the reshape metadata stored in result.params to recover the timepoint
    axis, the condition/dimension axis, and (when stoichiometry reweighting
    repeated some condition slices) the index of the first, representative copy
    of each unique condition so repeats are collapsed for plotting.

    Args:
        result: an AdaptiveClusteringResult (from testing=True).

    Returns:
        dict with keys T (timepoints), transpose (bool), D_expanded (columns in
        the clustered array), condition_start (list of representative column
        indices, one per unique condition), dim_labels, timepoint_labels.
    """
    p = result.params
    T = p["time_series_length"]
    cols = p["column_selection"]
    transpose = bool(p.get("transpose", False))
    stoich = p.get("stoichiometry")
    data_type = p["data_type"]

    # One representative column per un-repeated condition-dimension (reshape_df
    # builds its multiplier list from exactly these columns).
    representative_cols = list(cols[::T])
    multipliers = []
    for col in representative_cols:
        m = 1
        if stoich:
            for substring, val in stoich.items():
                if substring in col:
                    m = val
                    break
        multipliers.append(m)
    condition_start = list(np.cumsum([0] + multipliers)[:-1])  # index of first copy
    D_expanded = int(np.sum(multipliers))

    # Dimension labels: strip the data-type token and the trailing timepoint token.
    dim_labels = []
    for col in representative_cols:
        lab = col.replace(f"{data_type}_", "")
        lab = lab.rsplit("_", 1)[0]
        dim_labels.append(lab)

    # Timepoint labels: the trailing token of each column in the first dimension.
    timepoint_labels = [c.rsplit("_", 1)[-1] for c in cols[:T]]

    return {"T": T,
            "transpose": transpose,
            "D_expanded": D_expanded,
            "condition_start": condition_start,
            "dim_labels": dim_labels,
            "timepoint_labels": timepoint_labels,}
    # Question: if this is reconstructing the time series, does it reconstruct it 100%? or does it do an approximation?
    # Answer: It is EXACT (100%), not an approximation — with one deliberate, lossless
    #   exception. _series_layout() + _vector_to_series() only reshape the flat vector
    #   back to (conditions, timepoints); a reshape is a pure reindexing, so every
    #   value comes back untouched.
    #   The one intentional reduction: when `stoichiometry` repeated a condition's
    #   columns, those copies are IDENTICAL, so reconstruction keeps only the first
    #   copy of each condition (`condition_start` points at that first copy). It drops
    #   exact duplicates, not real information.
    #   Caveat on scope: this reconstructs the SPACE THAT WAS CLUSTERED (the `data_type`
    #   values, e.g. log2:FC). It does not — and should not — undo earlier preprocessing
    #   (log2, fold-change, scaling). That is exactly what we want for plotting, so the
    #   centroid and the member curves are shown in the same space.

def _vector_to_series(vec, layout,):
    """
    Reshape one flattened feature vector into (n_unique_conditions, n_timepoints).

    Args:
        vec: 1-D np.ndarray in the flattened feature space (centroid or mean curve).
        layout: dict from _series_layout().

    Returns:
        np.ndarray of shape (n_unique_conditions, T) — one time series per condition.
    """
    T = layout["T"]
    D_expanded = layout["D_expanded"]
    if layout["transpose"]:
        arr = vec.reshape(T, D_expanded)                       # (T, D_expanded)
        series = np.stack([arr[:, s] for s in layout["condition_start"]], axis=0)
    else:
        arr = vec.reshape(D_expanded, T)                       # (D_expanded, T)
        series = np.stack([arr[s, :] for s in layout["condition_start"]], axis=0)
    return series


# =============================================================================
# Cluster substructure diagnostic
# =============================================================================
def diagnose_cluster_substructure(result,
                                  cluster_label,
                                  bins=40,
                                  figsize=None,
                                  verbose=True,):
    """
    Check whether a single cluster hides sub-structure (two shapes in one cluster).

    For the chosen cluster it:
      1. Computes the Euclidean distance of every member to the cluster's own
         centroid (the model's actual centroid in result.centroids — NOT a
         re-derived mean), and plots their histogram.
      2. Fits a 1-component and a 2-component Gaussian mixture to those distances
         and reports the BIC of each (lower BIC = better; a much lower 2-component
         BIC suggests the cluster is really two shells / sub-groups).
      3. Splits the members into the "close" half and the "far" half by the median
         distance and plots the mean time series of each half next to the cluster
         centroid, per condition, so their shapes can be compared visually.

    Interpreting bic_1 vs bic_2:
      * bic_1 = BIC of a 1-Gaussian fit to the distances = "members form a single
        shell around the centroid" (one shape).
      * bic_2 = BIC of a 2-Gaussian fit = "there are two populations — a tight inner
        group and a looser/farther outer group" = a hint the cluster is two shapes.
      * BIC (Bayesian Information Criterion) rewards fit but penalises complexity, so
        the extra parameters of the 2-component model must earn their place. LOWER
        BIC wins; `preferred_n_components` is just argmin(bic_1, bic_2).
      * Only the COMPARISON within one cluster is meaningful — never compare BIC
        across clusters. Absolute BIC scales with member count and distance units,
        so a 4000-member cluster and a 30-member cluster have incomparable magnitudes.
      * Gap rule of thumb (delta = bic_1 - bic_2, positive favours 2 components):
        <2 negligible, 2-6 weak, 6-10 strong, >10 very strong evidence for 2 groups.
      * IMPORTANT: this tests the distance-to-centroid distribution, not shapes
        directly. "2 components preferred" often just means a tight core plus a
        diffuse halo of outliers (amplitude/tightness), not two distinct temporal
        shapes. Only treat it as real sub-structure worth splitting when the
        close-half vs far-half mean CURVES (panels 2..N) also differ in SHAPE.

    Normalization note: the pipeline applies NO per-site normalization, so the
    centroid and the half-means live in the same raw `data_type` magnitude space;
    axes are labelled with that data_type and nothing is mixed across spaces.

    Args:
        result: an AdaptiveClusteringResult (from testing=True).
        cluster_label: integer label of the cluster to inspect (e.g. the largest,
                       max(result.cluster_sizes, key=result.cluster_sizes.get)).
        bins: histogram bin count (default 40).
        figsize: (width, height); auto-sized if None.
        verbose: if True, print the BIC comparison.

    Returns:
        dict with keys: bic_1, bic_2, preferred_n_components, n_members,
        median_distance, distances (np.ndarray), and fig.
    """
    labels = np.asarray(result.labels)
    if cluster_label not in set(labels.tolist()):
        raise ValueError(f"cluster_label {cluster_label} not present in result.labels")

    X_flat = _flatten(result.X)
    mask = labels == cluster_label
    members = X_flat[mask]
    centroid = result.centroids[cluster_label]   # model's actual centroid

    # 1. distances to own centroid
    distances = np.sqrt(_squared_distances_to_centroid(members, centroid))

    # 2. 1- vs 2-component GMM on the distance distribution
    dd = distances.reshape(-1, 1)
    gmm1 = GaussianMixture(n_components=1, random_state=0).fit(dd)
    gmm2 = GaussianMixture(n_components=2, random_state=0).fit(dd)
    bic_1 = float(gmm1.bic(dd))
    bic_2 = float(gmm2.bic(dd))
    preferred = 2 if bic_2 < bic_1 else 1

    if verbose:
        print(f"Cluster {cluster_label}: {mask.sum()} members")
        print(f"  BIC (1 component): {bic_1:.1f}")
        print(f"  BIC (2 components): {bic_2:.1f}")
        print(f"  Preferred: {preferred} component(s) "
              f"({'sub-structure likely' if preferred == 2 else 'looks unimodal'})")

    # 3. close vs far halves by median distance
    median_d = float(np.median(distances))
    close_mask = distances <= median_d
    far_mask = distances > median_d
    close_mean = members[close_mask].mean(axis=0)
    far_mean = members[far_mask].mean(axis=0)

    layout = _series_layout(result)
    cen_series = _vector_to_series(centroid, layout)
    close_series = _vector_to_series(close_mean, layout)
    far_series = _vector_to_series(far_mean, layout)

    dim_labels = layout["dim_labels"]
    tp = layout["timepoint_labels"]
    data_type = result.params["data_type"]
    n_dims = len(dim_labels)

    if figsize is None:
        figsize = (5 * (n_dims + 1), 4.2)
    fig, axes = plt.subplots(1, n_dims + 1, figsize=figsize, squeeze=False)
    axes = axes[0]

    # Histogram panel + fitted GMM densities
    ax = axes[0]
    ax.hist(distances, bins=bins, density=True, color="steelblue",
            edgecolor="white", alpha=0.7)
    xs = np.linspace(distances.min(), distances.max(), 200).reshape(-1, 1)
    ax.plot(xs, np.exp(gmm1.score_samples(xs)), color="black", lw=1.6, label="1-comp")
    ax.plot(xs, np.exp(gmm2.score_samples(xs)), color="crimson", lw=1.6, ls="--", label="2-comp")
    ax.axvline(median_d, color="grey", ls=":", lw=1.2, label="median")
    ax.set_xlabel("distance to centroid (Euclidean)")
    ax.set_ylabel("density")
    ax.set_title(f"Cluster {cluster_label} (n={int(mask.sum())})\n"
                 f"BIC 1={bic_1:.0f}, 2={bic_2:.0f} -> {preferred} comp")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Per-condition shape comparison: centroid vs close-half vs far-half
    x = range(layout["T"])
    for c in range(n_dims):
        ax = axes[c + 1]
        ax.plot(x, cen_series[c], color="black", lw=2.4, label="centroid")
        ax.plot(x, close_series[c], color="tab:blue", lw=2.0, label="close half")
        ax.plot(x, far_series[c], color="tab:red", lw=2.0, label="far half")
        ax.axhline(0, color="grey", ls="--", lw=1)
        ax.set_xticks(list(x))
        ax.set_xticklabels(tp, rotation=90, fontsize=7)
        ax.set_xlabel("timepoint")
        ax.set_ylabel(f"{data_type} (clustered space)" if c == 0 else "")
        ax.set_title(dim_labels[c])
        ax.grid(alpha=0.3)
        if c == 0:
            ax.legend(fontsize=8)

    fig.tight_layout()

    return {"bic_1": bic_1,
            "bic_2": bic_2,
            "preferred_n_components": preferred,
            "n_members": int(mask.sum()),
            "median_distance": median_d,
            "distances": distances,
            "fig": fig,}


# =============================================================================
# Hyperparameter sweep (2D grid over inertia_threshold x merge_distance_threshold)
# =============================================================================
def sweep_thresholds(df,
                     inertia_thresholds,
                     merge_distance_thresholds,
                     data_type,
                     condition_for_clustering,
                     cell_lines,
                     df_dimensions,
                     time_series_length,
                     exclude_full=False,
                     stoichiometry=None,
                     transpose=False,
                     inertia_mode="mean",
                     initial_n_clusters=10,
                     n_subclusters=2,
                     max_subdivision_rounds=3,
                     min_cluster_size=5,
                     max_iterations=1000,
                     n_init=5,
                     random_state=0,
                     margin_threshold=0.3,
                     compute_stability=True,
                     n_bootstrap=10,
                     bootstrap_frac=0.8,
                     silhouette_sample_size=2000,
                     verbose=True,):
    """
    Run the adaptive pipeline over the full 2D grid of the two thresholds.

    For every (inertia_threshold, merge_distance_threshold) pair in the Cartesian
    product of the two supplied ranges, the full pipeline is run once and scored.
    Everything is time-series-shape only; no site metadata is used.

    Metrics recorded per pair:
      * n_clusters                     - final number of clusters.
      * size_min/median/max/std        - cluster-size distribution.
      * median_margin                  - median assignment margin (see assignment_margins).
      * pct_low_margin                 - % of sites with margin below `margin_threshold`.
      * silhouette                     - sklearn silhouette_score on the Euclidean
                                         feature matrix (sub-sampled for speed).
      * stability                      - mean pairwise Adjusted Rand Index across
                                         `n_bootstrap` random `bootstrap_frac`
                                         subsamples (via consensus_stability, shared
                                         sites only). NaN if compute_stability=False.

    Runtime warning: stability re-runs the whole pipeline `n_bootstrap` times per
    grid cell, so total fits ~= len(grid) * (1 + n_bootstrap). Start with coarse
    ranges and/or compute_stability=False for a first pass.

    Args:
        df: DataFrame to cluster (already filtered).
        inertia_thresholds: iterable of inertia_threshold values to test.
        merge_distance_thresholds: iterable of merge_distance_threshold values.
        data_type, condition_for_clustering, cell_lines, df_dimensions,
        time_series_length, exclude_full, stoichiometry, transpose, inertia_mode,
        initial_n_clusters, n_subclusters, max_subdivision_rounds, min_cluster_size,
        max_iterations, n_init, random_state: passed straight to
        adaptive_kmeans_clustering (held fixed across the grid).
        margin_threshold: margin below which a site counts as "low confidence"
                          (default 0.3).
        compute_stability: if False, skip the (expensive) bootstrap ARI.
        n_bootstrap: number of subsample runs for stability (default 10).
        bootstrap_frac: fraction of sites per subsample (default 0.8).
        silhouette_sample_size: sites sub-sampled for silhouette_score (default 2000).
        verbose: if True, print progress per grid cell.

    Returns:
        pd.DataFrame with one row per threshold pair and the metric columns above.
    """
    base_kwargs = dict(data_type=data_type,
                       condition_for_clustering=condition_for_clustering,
                       cell_lines=cell_lines,
                       exclude_full=exclude_full,
                       stoichiometry=stoichiometry,
                       transpose=transpose,
                       inertia_mode=inertia_mode,
                       initial_n_clusters=initial_n_clusters,
                       n_subclusters=n_subclusters,
                       max_subdivision_rounds=max_subdivision_rounds,
                       min_cluster_size=min_cluster_size,
                       max_iterations=max_iterations,
                       n_init=n_init,
                       df_dimensions=df_dimensions,
                       time_series_length=time_series_length,)

    grid = list(product(list(inertia_thresholds), list(merge_distance_thresholds)))
    rows = []

    for gi, (it, mt) in enumerate(grid):
        if verbose:
            print(f"[{gi + 1}/{len(grid)}] inertia_threshold={it}, "
                  f"merge_distance_threshold={mt}")

        _, res = adaptive_kmeans_clustering(df.copy(),
                                            inertia_threshold=it,
                                            merge_distance_threshold=mt,
                                            random_state=random_state,
                                            cluster_column_name="_sweep",
                                            verbose=False,
                                            testing=True,
                                            **base_kwargs,)

        sizes = np.array(list(res.cluster_sizes.values()), dtype=float)
        margins = assignment_margins(res)
        X_flat = _flatten(res.X)
        k = res.n_clusters_final
        n = X_flat.shape[0]

        # Silhouette (sub-sampled). Undefined for <2 clusters.
        if 2 <= k <= n - 1:
            silhouette = float(silhouette_score(X_flat,
                                                res.labels,
                                                metric="euclidean",
                                                sample_size=min(silhouette_sample_size, n),
                                                random_state=random_state,))
        else:
            silhouette = float("nan")

        # Stability via bootstrap ARI (reuses consensus_stability from src.clustering).
        if compute_stability:
            def _cluster_fn(sub_df, seed, _kw=base_kwargs, _it=it, _mt=mt):
                _, r = adaptive_kmeans_clustering(sub_df.copy(),
                                                  inertia_threshold=_it,
                                                  merge_distance_threshold=_mt,
                                                  random_state=seed,
                                                  cluster_column_name="_boot",
                                                  verbose=False,
                                                  testing=True,
                                                  **_kw,)
                return r.labels

            stability = consensus_stability(df,
                                            _cluster_fn,
                                            n_runs=n_bootstrap,
                                            bootstrap=True,
                                            bootstrap_frac=bootstrap_frac,
                                            compute_coassociation=False,
                                            random_state=random_state,
                                            verbose=False,)["mean_ari"]
        else:
            stability = float("nan")

        rows.append({"inertia_threshold": it,
                     "merge_distance_threshold": mt,
                     "n_clusters": int(k),
                     "size_min": int(sizes.min()),
                     "size_median": float(np.median(sizes)),
                     "size_max": int(sizes.max()),
                     "size_std": float(sizes.std()),
                     "median_margin": float(np.nanmedian(margins)),
                     "pct_low_margin": float(100.0 * np.mean(margins < margin_threshold)),
                     "silhouette": silhouette,
                     "stability": float(stability),})

    return pd.DataFrame(rows)


def filter_and_rank_sweep(results_df,
                          min_clusters=15,
                          max_clusters=30,
                          min_smallest_cluster=20,
                          rank_by=("silhouette", "stability"),
                          top_n=None,):
    """
    Filter a sweep result to acceptable threshold pairs and rank the survivors.

    Keeps only pairs whose final cluster count is within [min_clusters,
    max_clusters] (inclusive) AND whose smallest cluster has at least
    `min_smallest_cluster` members, then ranks the survivors by the given metrics
    (higher is better for both silhouette and stability) using the average of the
    per-metric ranks, so neither metric dominates.

    Args:
        results_df: DataFrame from sweep_thresholds().
        min_clusters: minimum acceptable final cluster count (default 15).
        max_clusters: maximum acceptable final cluster count (default 30).
        min_smallest_cluster: the smallest cluster must have at least this many
                              members (default 20).
        rank_by: metric columns to rank by, higher-is-better (default silhouette
                 and stability).
        top_n: if given, return only the top N rows.

    Returns:
        DataFrame of surviving pairs, sorted best-first, with added
        <metric>_rank and mean_rank columns. Empty if nothing survives.
    """
    mask = (results_df["n_clusters"].between(min_clusters, max_clusters)
            & (results_df["size_min"] >= min_smallest_cluster))
    surviving = results_df[mask].copy()
    if surviving.empty:
        return surviving

    for m in rank_by:
        surviving[f"{m}_rank"] = surviving[m].rank(ascending=False, method="min")
    surviving["mean_rank"] = surviving[[f"{m}_rank" for m in rank_by]].mean(axis=1)
    surviving = surviving.sort_values(["mean_rank"] + list(rank_by),
                                      ascending=[True] + [False] * len(rank_by))
    return surviving if top_n is None else surviving.head(top_n)


def plot_sweep_heatmaps(results_df,
                        metrics=("n_clusters", "median_margin", "silhouette", "stability"),
                        figsize=None,
                        cmap="viridis",
                        annotate=True,):
    """
    Heatmap of each metric over the full unfiltered threshold grid.

    inertia_threshold on the y-axis, merge_distance_threshold on the x-axis; one
    panel per metric. Shows the whole landscape (not only the filtered winners),
    so plateaus, cliffs and sweet spots are visible.

    Args:
        results_df: DataFrame from sweep_thresholds().
        metrics: which metric columns to draw (default the four headline ones).
        figsize: (width, height); auto-sized if None.
        cmap: matplotlib colormap name.
        annotate: if True, write each cell's value on the heatmap.

    Returns:
        fig, axes
    """
    n = len(metrics)
    if figsize is None:
        figsize = (5 * n, 4.5)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes = axes[0]

    for ax, metric in zip(axes, metrics):
        grid = results_df.pivot(index="inertia_threshold",
                                columns="merge_distance_threshold",
                                values=metric)
        im = ax.imshow(grid.values, origin="lower", aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(grid.columns)))
        ax.set_xticklabels([f"{c:g}" for c in grid.columns], rotation=90, fontsize=8)
        ax.set_yticks(range(len(grid.index)))
        ax.set_yticklabels([f"{r:g}" for r in grid.index], fontsize=8)
        ax.set_xlabel("merge_distance_threshold")
        ax.set_ylabel("inertia_threshold")
        ax.set_title(metric)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if annotate:
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    v = grid.values[i, j]
                    if not np.isnan(v):
                        text = f"{int(v)}" if metric == "n_clusters" else f"{v:.2f}"
                        ax.text(j, i, text, ha="center", va="center",
                                fontsize=7, color="white")

    fig.tight_layout()
    return fig, axes


# =============================================================================
# Next step — post-hoc enrichment (NOT part of clustering; runs AFTER, on metadata)
# =============================================================================
def posthoc_enrichment_fisher(*args, **kwargs):
    """
    PLACEHOLDER / Next step — post-hoc enrichment of cluster membership vs. site metadata.

    Intentionally NOT implemented yet. Clustering and threshold selection are kept
    strictly time-series-shape only; site metadata (e.g. ERK vs. non-ERK motif,
    functional score, known regulatory role) must NOT influence the clustering or
    the sweep scoring.

    Once metadata is available this should, for a chosen categorical annotation:
      * build, per cluster, the 2x2 contingency table
        [[in-cluster & annotated,     in-cluster & not-annotated],
         [out-of-cluster & annotated, out-of-cluster & not-annotated]],
      * run scipy.stats.fisher_exact on each,
      * correct across clusters (e.g. Benjamini-Hochberg via
        statsmodels.stats.multitest.multipletests),
      * return a tidy DataFrame (cluster, odds_ratio, p_value, q_value, counts).

    See the "post-hoc enrichment" next step cell in the sweep notebook.
    """
    raise NotImplementedError(
        "posthoc_enrichment_fisher is a placeholder; implement once site metadata "
        "is available. Clustering/scoring stays metadata-free until then."
    )
