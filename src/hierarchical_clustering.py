"""
Agglomerative hierarchical clustering of phosphosite temporal profiles.

WHY A SEPARATE MODULE
---------------------
`src/clustering.py` holds the partitional methods (TimeSeriesKMeans, KShape,
KernelKMeans): you fix k up-front, the algorithm is stochastic, and the only
structure returned is a flat label vector. Hierarchical clustering is a
different object: it builds the *whole* merge tree once (deterministically), and
"how many clusters" becomes a question you answer afterwards by cutting that
tree. Two consequences drive the design of this module:

  1. **QC is tree-shaped.** The useful diagnostics are not "does the seed change
     the answer" (nothing is random here) but "does the tree faithfully represent
     the real distances" (cophenetic correlation), "where are the big jumps in
     merge distance" (the hierarchical elbow), and "does one giant cluster
     swallow everything" (the classic failure mode of average/single linkage).

  2. **Merging is a first-class operation.** Because every cut cluster *is* a
     node of the tree, two clusters that sit under a common parent can be merged
     without re-running anything — the merged cluster is exactly the parent node.
     `mergeable_cluster_groups()` lists which merges are legal and what they
     cost; `merge_clusters()` applies them. This is the "look at the clusters,
     then constrain the result" loop the project needs, and it is *not* something
     KMeans can do (its centroid-linkage dendrogram in
     `clustering.compute_centroid_linkage` is a post-hoc approximation, whereas
     here the tree is the model itself).

DATA FLOW
---------
    build_feature_matrix()      select columns -> (n_sites, T, C) -> flat (n_sites, T*C)
    compute_linkage()           flat matrix    -> linkage matrix Z
    hierarchical_clustering()   both of the above + a cut -> HierarchicalResult

    QC:            cophenetic_correlation, compare_linkage_methods,
                   hierarchical_kscan, plot_hierarchical_scan,
                   plot_merge_distances, silhouette_per_cluster
    Merging:       cluster_level_linkage, mergeable_cluster_groups,
                   suggest_merges, merge_clusters, merge_clusters_by_height,
                   plot_cluster_tree

DISTANCE / LINKAGE CHOICE
-------------------------
All timepoints are shared and aligned across sites, so plain Euclidean distance
on the flattened profile is appropriate (this is the same argument that made
`metric="euclidean"` the default in the KMeans path — DTW buys nothing when the
series are already aligned, and costs a lot). "ward" linkage minimises the same
within-cluster sum of squares that KMeans optimises, which makes the two methods
directly comparable; it also strongly resists the chaining that ruins "single".
Ward requires the Euclidean metric — this is enforced.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import (cophenet,
                                     dendrogram,
                                     fcluster,
                                     linkage,
                                     to_tree,)
from scipy.spatial.distance import pdist

from sklearn.metrics import (adjusted_rand_score,
                             calinski_harabasz_score,
                             davies_bouldin_score,
                             silhouette_samples,
                             silhouette_score,)

from src.column_spec import ColumnSpec
from src.clustering import reshape_df


# =============================================================================
# Result container
# =============================================================================

@dataclass
class HierarchicalResult:
    """
    Everything produced by one hierarchical clustering run.

    A dataclass rather than a tuple because the linkage matrix `Z` must travel
    with the labels — every merge / QC function in this module needs it, and
    losing it means re-running the linkage.

    Attributes:
        df: the clustered DataFrame (rows aligned with `labels`), carrying the
            new cluster-label column.
        labels: np.ndarray of shape (n_sites,) — 0-based integer cluster labels.
        Z: np.ndarray of shape (n_sites - 1, 4) — the scipy linkage matrix
           describing the full merge tree. Cutting it differently is free.
        X: np.ndarray of shape (n_sites, n_timepoints, n_conditions) — the
           multivariate profiles used (transpose=True layout).
        X_flat: np.ndarray of shape (n_sites, n_timepoints * n_conditions) —
                the flattened matrix the linkage was actually computed on.
        centroids: np.ndarray of shape (n_clusters, n_timepoints, n_conditions) —
                   per-cluster mean profile. Shaped so it can be passed straight
                   to plotting_functions.plot_cluster_hierarchy().
        distances_to_centroids: np.ndarray of shape (n_sites, n_clusters) —
                   Euclidean distance from each site to each centroid. Same role
                   as the `barycenters` matrix of the KMeans path, so it can be
                   passed straight to plotting_functions.plot_cluster_assignment_qc().
        cluster_sizes: dict mapping cluster label -> number of sites.
        cluster_column: name of the label column written into `df`.
        params: dict recording the configuration of the run (column selection,
                linkage method, metric, cut criterion, ...). Consumed by
                adaptive_clustering.diagnose_cluster_substructure() among others.
    """

    df: pd.DataFrame
    labels: np.ndarray
    Z: np.ndarray
    X: np.ndarray
    X_flat: np.ndarray
    centroids: np.ndarray
    distances_to_centroids: np.ndarray
    cluster_sizes: Dict[int, int]
    cluster_column: str
    params: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Feature matrix and linkage
# =============================================================================

def build_feature_matrix(df,
                         cell_lines,
                         conditions,
                         data_type,
                         exclude_full=True,
                         time_series_length=None,
                         df_dimensions=None,
                         stoichiometry=None,
                         transpose=True,
                         nan_policy="drop",
                         verbose=True,):
    """
    Select the data columns, reshape them to multivariate time series and flatten
    them into the 2-D matrix that scipy's linkage needs.

    Column selection goes through ColumnSpec.select() (project rule: never build
    column lists by hand) and the reshape reuses clustering.reshape_df(), so the
    feature matrix is byte-for-byte the same one the KMeans path would cluster.
    This matters: it is what makes a hierarchical-vs-KMeans comparison fair.

    Missing values are handled explicitly rather than filled with 0 — scipy's
    pdist silently propagates NaN into every distance touching that row, which
    would corrupt the whole tree, and filling with 0 would invent a "no change
    from starve" observation that was never measured.

    Args:
        df: DataFrame following the project naming convention.
        cell_lines: list of cell-line prefixes, e.g. ["WT"].
        conditions: list of condition substrings, e.g. ["_EGF_", "_INS_"].
        data_type: data-type string, e.g. "log2:zscore".
        exclude_full: if True (default), drop the 'full' timepoint.
        time_series_length: number of timepoints per condition. Inferred from the
                            column selection when None.
        df_dimensions: number of (cell_line x condition) combinations. Computed as
                       len(cell_lines) * len(conditions) when None.
        stoichiometry: optional dict re-weighting conditions, passed to reshape_df
                       (requires transpose=True).
        transpose: if True (default), X has shape (n_sites, n_timepoints,
                   n_conditions). Euclidean distance is invariant to this, but the
                   centroid shape downstream is not.
        nan_policy: "drop" (default) removes rows with any NaN in the selected
                    columns and reports how many; "raise" raises instead.
        verbose: if True, print the selection and the shapes.

    Returns:
        df_used: the (possibly NaN-filtered) DataFrame whose rows align with X.
        X: np.ndarray of shape (n_sites, n_timepoints, n_conditions).
        X_flat: np.ndarray of shape (n_sites, n_timepoints * n_conditions).
        cols: list of the selected column names.
        n_timepoints: number of timepoints per condition.
        n_dims: number of (cell_line x condition) combinations.
    """
    cols = ColumnSpec.select(df,
                             cell_lines=cell_lines,
                             data_type=data_type,
                             conditions=conditions,
                             exclude_full=exclude_full,
                             exclude_replicate_cols=True,)
    if not cols:
        raise ValueError(f"No columns found for cell_lines={cell_lines}, "
                         f"conditions={conditions}, data_type={data_type!r}.")

    if df_dimensions is None:
        df_dimensions = len(cell_lines) * len(conditions)
    if time_series_length is None:
        time_series_length = len(cols) // df_dimensions

    if len(cols) != df_dimensions * time_series_length:
        raise ValueError(f"Column count {len(cols)} is not "
                         f"df_dimensions ({df_dimensions}) x "
                         f"time_series_length ({time_series_length}). "
                         f"Check that every condition has the same timepoints.")

    # --- Missing values -----------------------------------------------------
    nan_mask = df[cols].isna().any(axis=1)
    n_nan = int(nan_mask.sum())
    if n_nan:
        if nan_policy == "raise":
            raise ValueError(f"{n_nan} rows have NaN in the selected columns. "
                             f"Filter them first, or use nan_policy='drop'.")
        elif nan_policy == "drop":
            df_used = df.loc[~nan_mask].copy()
            if verbose:
                print(f"Dropped {n_nan} rows with NaN in the clustering columns "
                      f"({len(df_used)} sites remain).")
        else:
            raise ValueError(f"nan_policy must be 'drop' or 'raise', got {nan_policy!r}")
    else:
        df_used = df.copy()

    X, _ = reshape_df(df=df_used,
                      time_series=cols,
                      dimensions=df_dimensions,
                      len_time_serie=time_series_length,
                      stoichiometry=stoichiometry,
                      labels="site",
                      transpose=transpose,
                      verbose=False,)
    X = np.asarray(X, dtype=float)
    X_flat = X.reshape(X.shape[0], -1)

    if verbose:
        print(f"Column selection ({len(cols)}): {cols}")
        print(f"Dimensions: {df_dimensions} | Timepoints per condition: {time_series_length}")
        print(f"Feature matrix: X {X.shape} -> flat {X_flat.shape}")

    return df_used, X, X_flat, cols, time_series_length, df_dimensions


def compute_linkage(X_flat,
                    method="ward",
                    metric="euclidean",
                    verbose=True,):
    """
    Build the agglomerative merge tree over the sites.

    Memory note: scipy computes the condensed pairwise-distance vector, i.e.
    n*(n-1)/2 float64 values. For ~9000 sites that is ~320 MB — fine on a laptop,
    but it grows quadratically, so filter before clustering rather than after.

    Args:
        X_flat: np.ndarray of shape (n_sites, n_features) — flattened profiles.
        method: linkage method: "ward" (default), "average", "complete",
                "weighted", "centroid", "median", "single".
                * ward     — minimises within-cluster variance; the direct
                             analogue of KMeans and the sensible default.
                * average  — UPGMA; tolerant of elongated clusters, but prone to
                             producing one dominant cluster on noisy data.
                * complete — compact, similar-diameter clusters; sensitive to
                             outliers.
                * single   — chaining; almost always produces a giant cluster plus
                             singletons on this kind of data. Kept for completeness.
        metric: distance metric passed to pdist (default "euclidean").
                "ward", "centroid" and "median" are only defined for Euclidean
                distance and this is enforced.
        verbose: if True, print the resulting tree height range.

    Returns:
        Z: np.ndarray of shape (n_sites - 1, 4) — the scipy linkage matrix.
    """
    if method in ("ward", "centroid", "median") and metric != "euclidean":
        raise ValueError(f"method={method!r} is only defined for metric='euclidean', "
                         f"got metric={metric!r}.")

    X_flat = np.asarray(X_flat, dtype=float)
    if np.isnan(X_flat).any():
        raise ValueError("X_flat contains NaN — pdist would propagate it into every "
                         "distance involving that row. Filter or drop those sites first.")

    Z = linkage(X_flat,
                method=method,
                metric=metric,)

    if verbose:
        print(f"Linkage built: method={method}, metric={metric}, "
              f"{Z.shape[0] + 1} sites, merge heights {Z[:, 2].min():.3f} - {Z[:, 2].max():.3f}")
    return Z


def cluster_centroids(X,
                      labels,):
    """
    Mean profile of each cluster, keeping the multivariate shape.

    Args:
        X: np.ndarray of shape (n_sites, n_timepoints, n_conditions).
        labels: array-like of length n_sites with 0-based integer cluster labels.

    Returns:
        centroids: np.ndarray of shape (n_clusters, n_timepoints, n_conditions),
                   indexed by sorted unique label.
    """
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    centroids = np.stack([X[labels == c].mean(axis=0) for c in uniq], axis=0)
    return centroids


def distances_to_centroids(X_flat,
                           centroids,):
    """
    Euclidean distance from every site to every cluster centroid.

    Mirrors the `barycenters` matrix returned by the KMeans path, so the output
    can be handed directly to plotting_functions.plot_cluster_assignment_qc()
    for the margin / heatmap confidence panels.

    Note that, unlike KMeans, hierarchical clustering does not assign sites to
    their nearest centroid — a site may legitimately sit closer to another
    cluster's centroid than to its own. That mismatch is itself informative and
    is quantified by assignment_agreement().

    Args:
        X_flat: np.ndarray of shape (n_sites, n_features).
        centroids: np.ndarray of shape (n_clusters, n_timepoints, n_conditions)
                   or (n_clusters, n_features).

    Returns:
        D: np.ndarray of shape (n_sites, n_clusters) of Euclidean distances.
    """
    C_flat = np.asarray(centroids, dtype=float)
    C_flat = C_flat.reshape(C_flat.shape[0], -1)
    diff = X_flat[:, None, :] - C_flat[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


def hierarchical_clustering(df_to_cluster,
                            data_type,
                            condition_for_clustering=None,
                            cell_lines=None,
                            exclude_full=True,
                            df_dimensions=None,
                            time_series_length=None,
                            stoichiometry=None,
                            cluster_column_name="Hierarchical_cluster",
                            number_of_clusters=None,
                            distance_threshold=None,
                            linkage_method="ward",
                            metric="euclidean",
                            transpose=True,
                            precomputed_Z=None,
                            nan_policy="drop",
                            verbose=True,):
    """
    Cluster phosphosite temporal profiles by agglomerative hierarchical clustering.

    The tree is built once and then cut, either to a fixed number of clusters
    (`number_of_clusters`) or at a fixed merge distance (`distance_threshold`).
    Pass `precomputed_Z` to re-cut a tree you already built — this is free,
    whereas re-running the linkage is not.

    Args:
        df_to_cluster: DataFrame following the project naming convention.
        data_type: data-type string used for clustering, e.g. "log2:zscore".
        condition_for_clustering: list of condition substrings, e.g. ["_EGF_"].
        cell_lines: list of cell-line prefixes, e.g. ["WT"].
        exclude_full: if True (default), exclude the 'full' timepoint.
        df_dimensions: number of (cell_line x condition) combinations; inferred when None.
        time_series_length: timepoints per condition; inferred when None.
        stoichiometry: optional condition re-weighting dict (needs transpose=True).
        cluster_column_name: name of the label column written into the returned DataFrame.
        number_of_clusters: k for a 'maxclust' cut. Mutually exclusive with
                            distance_threshold; exactly one must be given.
        distance_threshold: merge distance for a 'distance' cut — every cluster
                            below this height in the tree becomes one cluster.
        linkage_method: see compute_linkage() (default "ward").
        metric: distance metric (default "euclidean").
        transpose: if True (default), X is (n_sites, n_timepoints, n_conditions).
        precomputed_Z: an existing linkage matrix to re-cut instead of recomputing.
                       Must have been built from the same rows in the same order.
        nan_policy: "drop" (default) or "raise" — see build_feature_matrix().
        verbose: if True, print selection, shapes and the resulting cluster sizes.

    Returns:
        HierarchicalResult — see the dataclass docstring. `.df` carries the new
        label column, `.Z` the full tree (needed by every merge function here).
    """
    if condition_for_clustering is None:
        condition_for_clustering = []
    if cell_lines is None:
        cell_lines = []

    if (number_of_clusters is None) == (distance_threshold is None):
        raise ValueError("Give exactly one of number_of_clusters or distance_threshold.")

    df_used, X, X_flat, cols, n_timepoints, n_dims = build_feature_matrix(
        df=df_to_cluster,
        cell_lines=cell_lines,
        conditions=condition_for_clustering,
        data_type=data_type,
        exclude_full=exclude_full,
        time_series_length=time_series_length,
        df_dimensions=df_dimensions,
        stoichiometry=stoichiometry,
        transpose=transpose,
        nan_policy=nan_policy,
        verbose=verbose,)

    if precomputed_Z is not None:
        if precomputed_Z.shape[0] != X_flat.shape[0] - 1:
            raise ValueError(f"precomputed_Z was built on {precomputed_Z.shape[0] + 1} sites "
                             f"but the feature matrix has {X_flat.shape[0]}. "
                             f"Rebuild the linkage or use the same filtered DataFrame.")
        Z = precomputed_Z
    else:
        Z = compute_linkage(X_flat,
                            method=linkage_method,
                            metric=metric,
                            verbose=verbose,)

    if number_of_clusters is not None:
        labels_1based = fcluster(Z,
                                 t=number_of_clusters,
                                 criterion="maxclust",)
        criterion = "maxclust"
        cut_value = number_of_clusters
    else:
        labels_1based = fcluster(Z,
                                 t=distance_threshold,
                                 criterion="distance",)
        criterion = "distance"
        cut_value = distance_threshold

    # fcluster labels are 1-based; the rest of the project uses 0-based labels
    # (KMeans convention), so relabel by descending cluster size for readability:
    # cluster 0 is always the largest.
    labels = relabel_by_size(labels_1based - 1)

    df_used[cluster_column_name] = labels
    centroids = cluster_centroids(X, labels)
    D = distances_to_centroids(X_flat, centroids)
    sizes = pd.Series(labels).value_counts().sort_index().to_dict()

    params = {"column_selection": cols,
              "data_type": data_type,
              "cell_lines": cell_lines,
              "conditions": condition_for_clustering,
              "exclude_full": exclude_full,
              "time_series_length": n_timepoints,
              "df_dimensions": n_dims,
              "transpose": transpose,
              "stoichiometry": stoichiometry,
              "linkage_method": linkage_method,
              "metric": metric,
              "cut_criterion": criterion,
              "cut_value": cut_value,}

    if verbose:
        print(f"\nCut: criterion={criterion}, value={cut_value} -> {len(sizes)} clusters")
        print(f"Cluster sizes:\n{pd.Series(sizes)}")

    return HierarchicalResult(df=df_used,
                              labels=labels,
                              Z=Z,
                              X=X,
                              X_flat=X_flat,
                              centroids=centroids,
                              distances_to_centroids=D,
                              cluster_sizes=sizes,
                              cluster_column=cluster_column_name,
                              params=params,)


def relabel_by_size(labels,):
    """
    Renumber cluster labels 0..k-1 in order of decreasing cluster size.

    Cluster identity from fcluster is arbitrary and changes with the cut, which
    makes cluster numbers hard to talk about. Sorting by size gives a stable,
    readable convention: cluster 0 is always the biggest.

    Args:
        labels: array-like of integer cluster labels.

    Returns:
        np.ndarray of relabelled 0-based integer labels, same length and order.
    """
    labels = np.asarray(labels)
    order = pd.Series(labels).value_counts().index.tolist()  # already size-sorted
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[l] for l in labels], dtype=int)


# =============================================================================
# Quality control — how good is the tree, and where should it be cut?
# =============================================================================

def cophenetic_correlation(Z,
                           X_flat=None,
                           condensed_distances=None,):
    """
    How faithfully the dendrogram reproduces the original pairwise distances.

    The cophenetic distance between two sites is the height at which they first
    end up in the same cluster. The cophenetic correlation is the Pearson
    correlation between those tree distances and the true pairwise distances.
    It is the one QC number that judges the *tree itself* rather than a
    particular cut, and it is the right way to choose a linkage method:

        > 0.8   the tree is a good summary of the distance structure
        0.6-0.8 usable, but the tree distorts some relationships
        < 0.6   the hierarchy is misleading; prefer a partitional method

    Caveat: it rewards linkage methods that compress distances ("average" often
    scores highest almost by construction), so read it together with the cluster
    size distribution — a tree with one giant cluster can still score well.

    Args:
        Z: linkage matrix from compute_linkage().
        X_flat: np.ndarray of shape (n_sites, n_features). Used to compute the
                pairwise distances when `condensed_distances` is not given.
        condensed_distances: precomputed condensed distance vector from pdist();
                             pass this to avoid recomputing it for every method.

    Returns:
        c: float — the cophenetic correlation coefficient.
    """
    if condensed_distances is None:
        if X_flat is None:
            raise ValueError("Give either X_flat or condensed_distances.")
        condensed_distances = pdist(np.asarray(X_flat, dtype=float),
                                    metric="euclidean",)
    c, _ = cophenet(Z, condensed_distances)
    return float(c)


def compare_linkage_methods(X_flat,
                            methods=("ward", "average", "complete", "weighted"),
                            metric="euclidean",
                            k_reference=10,
                            silhouette_sample=4000,
                            random_state=0,
                            verbose=True,):
    """
    Build one tree per linkage method and score them side by side.

    Reports, for each method:
      * cophenetic correlation — faithfulness of the tree (see above);
      * silhouette at a reference k — separation of the resulting partition;
      * largest-cluster fraction — the giant-cluster failure mode. A method that
        puts 90% of sites in one cluster is useless here regardless of how good
        its cophenetic correlation looks;
      * number of singleton clusters at the reference k.

    The pairwise distance matrix is computed once and reused across methods.

    Args:
        X_flat: np.ndarray of shape (n_sites, n_features).
        methods: iterable of linkage methods to compare.
        metric: distance metric (default "euclidean"; ward requires it).
        k_reference: number of clusters used for the partition-level metrics.
        silhouette_sample: subsample size for the silhouette (None = use all
                           sites; the full computation is O(n^2)).
        random_state: seed for the silhouette subsampling.
        verbose: if True, print the table.

    Returns:
        results_df: DataFrame with one row per method, sorted by cophenetic
                    correlation (descending).
        trees: dict mapping method name -> linkage matrix, so a chosen tree can
               be reused without recomputing.
    """
    X_flat = np.asarray(X_flat, dtype=float)
    condensed = pdist(X_flat, metric=metric)

    rows = []
    trees = {}
    for method in methods:
        if method in ("ward", "centroid", "median") and metric != "euclidean":
            if verbose:
                print(f"Skipping {method!r}: requires metric='euclidean'.")
            continue

        Z = linkage(condensed, method=method)
        trees[method] = Z

        labels = fcluster(Z, t=k_reference, criterion="maxclust")
        sizes = pd.Series(labels).value_counts()

        n = len(labels)
        sample = None if silhouette_sample is None else min(silhouette_sample, n)
        sil = (silhouette_score(X_flat,
                                labels,
                                metric="euclidean",
                                sample_size=sample,
                                random_state=random_state,)
               if sizes.size > 1 else float("nan"))

        rows.append({"method": method,
                     "cophenetic_corr": cophenetic_correlation(Z, condensed_distances=condensed),
                     f"silhouette_k{k_reference}": float(sil),
                     "largest_cluster_frac": float(sizes.max() / n),
                     "n_singletons": int((sizes == 1).sum()),
                     "n_clusters_obtained": int(sizes.size),})

    results_df = pd.DataFrame(rows).sort_values("cophenetic_corr", ascending=False)
    results_df = results_df.reset_index(drop=True)

    if verbose:
        print(results_df.to_string(index=False))

    return results_df, trees


def cluster_inertia(X_flat,
                    labels,):
    """
    Total within-cluster sum of squared distances to the cluster centroid.

    The same quantity KMeans minimises, so it puts a hierarchical cut and a
    KMeans solution on one scale.

    Args:
        X_flat: np.ndarray of shape (n_sites, n_features).
        labels: array-like of cluster labels, length n_sites.

    Returns:
        float — total within-cluster sum of squares.
    """
    labels = np.asarray(labels)
    total = 0.0
    for c in np.unique(labels):
        sub = X_flat[labels == c]
        total += float(((sub - sub.mean(axis=0)) ** 2).sum())
    return total


def hierarchical_kscan(Z,
                       X_flat,
                       k_range,
                       silhouette_sample=4000,
                       random_state=0,
                       verbose=True,):
    """
    Cut one tree at many k values and score every resulting partition.

    Cheap by design: the tree is built once outside this function, and each k is
    just an fcluster() call. That is the practical advantage over the KMeans
    k-scan, which has to refit the model (times several seeds) for every k.

    Metrics per k:
      * inertia — within-cluster sum of squares; monotonically decreasing, look
        for the elbow.
      * silhouette — mean (b - a) / max(a, b) over sites; higher is better.
        Subsampled by default because the exact computation is O(n^2).
        0.2-0.4 is a normal range for phosphoproteomics; do not expect 0.7.
      * calinski_harabasz — between/within variance ratio; higher is better.
      * davies_bouldin — mean worst-case cluster overlap; LOWER is better.
      * merge_height — the tree distance at which k would collapse to k-1. A
        large value means the k clusters are well separated; see also
        plot_merge_distances().
      * largest_cluster_frac / min_size / n_singletons — the size distribution.
        Watch these: hierarchical methods often reach a good silhouette by
        shaving off tiny outlier clusters while one cluster keeps everything.

    Args:
        Z: linkage matrix from compute_linkage().
        X_flat: np.ndarray of shape (n_sites, n_features) — the matrix Z was built on.
        k_range: iterable of k values to test, e.g. range(4, 26, 2).
        silhouette_sample: subsample size for the silhouette (None = all sites).
        random_state: seed for the silhouette subsampling.
        verbose: if True, print one line per k.

    Returns:
        scan_df: DataFrame with one row per k and the columns described above.
    """
    X_flat = np.asarray(X_flat, dtype=float)
    n = X_flat.shape[0]
    rows = []

    for k in k_range:
        labels = fcluster(Z, t=k, criterion="maxclust")
        sizes = pd.Series(labels).value_counts()
        n_obtained = int(sizes.size)

        # Height at which this partition would lose one cluster. Z rows are in
        # increasing order, so the merge taking k -> k-1 is row n-1-k.
        merge_height = float(Z[n - 1 - k, 2]) if 1 <= k < n else float("nan")

        if n_obtained > 1:
            sample = None if silhouette_sample is None else min(silhouette_sample, n)
            sil = float(silhouette_score(X_flat,
                                         labels,
                                         metric="euclidean",
                                         sample_size=sample,
                                         random_state=random_state,))
            ch = float(calinski_harabasz_score(X_flat, labels))
            db = float(davies_bouldin_score(X_flat, labels))
        else:
            sil = ch = db = float("nan")

        rows.append({"k": k,
                     "n_clusters_obtained": n_obtained,
                     "inertia": cluster_inertia(X_flat, labels),
                     "silhouette": sil,
                     "calinski_harabasz": ch,
                     "davies_bouldin": db,
                     "merge_height": merge_height,
                     "largest_cluster_frac": float(sizes.max() / n),
                     "min_cluster_size": int(sizes.min()),
                     "n_singletons": int((sizes == 1).sum()),})

        if verbose:
            r = rows[-1]
            print(f"k={k:3d}  inertia={r['inertia']:10.2f}  sil={r['silhouette']:.4f}  "
                  f"CH={r['calinski_harabasz']:9.1f}  DB={r['davies_bouldin']:.3f}  "
                  f"largest={r['largest_cluster_frac']:.2%}  min_size={r['min_cluster_size']}")

    return pd.DataFrame(rows)


def plot_hierarchical_scan(scan_df,
                           figsize=(16, 8),):
    """
    Six-panel summary of a hierarchical_kscan() result.

    Panels (top row = "are the clusters good", bottom row = "are they usable"):
      inertia, silhouette, Calinski-Harabasz /
      Davies-Bouldin, merge height, cluster-size distribution.

    Read the bottom-right panel first. A k whose largest cluster holds most of
    the dataset is not a real partition, no matter what the silhouette says.

    Args:
        scan_df: DataFrame returned by hierarchical_kscan().
        figsize: (width, height) of the figure.

    Returns:
        fig, axes — the Figure and a (2, 3) array of Axes.
    """
    fig, axes = plt.subplots(nrows=2,
                             ncols=3,
                             figsize=figsize,)
    k = scan_df["k"]

    panels = [(axes[0, 0], "inertia", "Inertia (lower = tighter)", "tab:blue"),
              (axes[0, 1], "silhouette", "Silhouette (higher = better)", "tab:green"),
              (axes[0, 2], "calinski_harabasz", "Calinski-Harabasz (higher = better)", "tab:purple"),
              (axes[1, 0], "davies_bouldin", "Davies-Bouldin (LOWER = better)", "tab:red"),
              (axes[1, 1], "merge_height", "Merge height k -> k-1 (higher = better separated)", "tab:orange"),]

    for ax, col, title, color in panels:
        ax.plot(k, scan_df[col], marker="o", color=color)
        ax.set_xlabel("Number of clusters (k)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.plot(k, scan_df["largest_cluster_frac"], marker="o",
            color="tab:brown", label="largest cluster (fraction of sites)")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Fraction of sites")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax_twin = ax.twinx()
    ax_twin.plot(k, scan_df["n_singletons"], marker="s", linestyle="--",
                 color="grey", label="singleton clusters")
    ax_twin.set_ylabel("Singleton clusters")
    ax.set_title("Size distribution — watch for a giant cluster", fontsize=10)
    lines = ax.get_lines() + ax_twin.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper right")

    fig.tight_layout()
    return fig, axes


def plot_merge_distances(Z,
                         last_n=40,
                         figsize=(14, 4),):
    """
    The hierarchical elbow: merge distance of the final agglomeration steps.

    Reading right to left, each step merges two clusters. A large jump in merge
    distance means the two things joined at that step were far apart — i.e. the
    partition *before* that merge was capturing real separation. The number of
    clusters just before the largest jump is the natural k suggested by the tree.

    The second panel shows the acceleration (successive differences), which makes
    that jump easier to see than eyeballing the raw curve.

    Args:
        Z: linkage matrix from compute_linkage().
        last_n: how many of the final merges to show (default 40).
        figsize: (width, height) of the figure.

    Returns:
        fig, axes, suggested_k — the Figure, the (1, 2) Axes array, and the k
        implied by the largest gap in the plotted range.
    """
    heights = Z[-last_n:, 2]
    ks = np.arange(len(heights), 0, -1)  # merge i leaves this many clusters after it
    gaps = np.diff(heights)              # gap between consecutive merges
    # A gap at position i sits between merge i and i+1; the partition that
    # survives that jump has ks[i + 1] + 1 clusters.
    suggested_k = int(ks[int(np.argmax(gaps)) + 1] + 1)

    fig, axes = plt.subplots(nrows=1,
                             ncols=2,
                             figsize=figsize,)

    axes[0].plot(ks, heights, marker="o", color="tab:blue")
    axes[0].invert_xaxis()
    axes[0].axvline(suggested_k, color="crimson", linestyle="--",
                    label=f"largest gap -> k = {suggested_k}")
    axes[0].set_xlabel("Clusters remaining after the merge")
    axes[0].set_ylabel("Merge distance")
    axes[0].set_title(f"Merge distance of the last {last_n} agglomerations", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].bar(ks[1:], gaps, color="tab:orange")
    axes[1].invert_xaxis()
    axes[1].axvline(suggested_k, color="crimson", linestyle="--")
    axes[1].set_xlabel("Clusters remaining after the merge")
    axes[1].set_ylabel("Jump in merge distance")
    axes[1].set_title("Acceleration — the biggest bar is the natural cut", fontsize=10)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig, axes, suggested_k


def plot_site_dendrogram(Z,
                         color_threshold=None,
                         truncate_p=30,
                         figsize=(16, 5),
                         title="Site dendrogram",):
    """
    Dendrogram of the full site-level tree, truncated for readability.

    Plotting ~9000 leaves is unreadable, so by default the tree is truncated to
    its top `truncate_p` nodes ("lastp" mode); leaf labels then show how many
    original sites each collapsed branch contains.

    Args:
        Z: linkage matrix from compute_linkage().
        color_threshold: merge distance below which branches are coloured as
                         separate clusters. None (default) uses scipy's 0.7 * max
                         heuristic. Set it to the height of your intended cut to
                         preview that partition.
        truncate_p: number of top-level nodes to show; None plots every leaf.
        figsize: (width, height) of the figure.
        title: axes title.

    Returns:
        fig, ax
    """
    fig, ax = plt.subplots(figsize=figsize)
    kwargs = dict(Z=Z,
                  ax=ax,
                  color_threshold=color_threshold,
                  show_leaf_counts=True,)
    if truncate_p is not None:
        kwargs.update(truncate_mode="lastp", p=truncate_p)

    dendrogram(**kwargs)
    ax.set_title(title)
    ax.set_xlabel("Site (or collapsed branch, with site count in brackets)")
    ax.set_ylabel("Merge distance")
    if color_threshold is not None:
        ax.axhline(color_threshold, color="crimson", linestyle="--", linewidth=1)
    fig.tight_layout()
    return fig, ax


def silhouette_per_cluster(X_flat,
                           labels,
                           sample_size=None,
                           random_state=0,):
    """
    Per-cluster silhouette, to find which specific clusters are weak.

    The global silhouette is one number for the whole partition and hides the
    interesting case: a few clean clusters plus one incoherent one. A cluster
    with a mean silhouette near 0 (or negative) overlaps its neighbours and is a
    prime candidate for merging with the sibling it cannot be told apart from.

    Args:
        X_flat: np.ndarray of shape (n_sites, n_features).
        labels: array-like of cluster labels, length n_sites.
        sample_size: if given, compute on a random subsample of that many sites
                     (the exact computation is O(n^2)).
        random_state: seed for the subsampling.

    Returns:
        DataFrame with one row per cluster: cluster, n_sites, mean_silhouette,
        median_silhouette, frac_negative (fraction of sites that would be better
        placed in another cluster), sorted by mean_silhouette ascending (worst first).
    """
    X_flat = np.asarray(X_flat, dtype=float)
    labels = np.asarray(labels)

    if sample_size is not None and sample_size < len(labels):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(labels), size=sample_size, replace=False)
        X_flat, labels = X_flat[idx], labels[idx]

    values = silhouette_samples(X_flat, labels, metric="euclidean")

    rows = []
    for c in np.unique(labels):
        v = values[labels == c]
        rows.append({"cluster": int(c),
                     "n_sites": int(v.size),
                     "mean_silhouette": float(np.mean(v)),
                     "median_silhouette": float(np.median(v)),
                     "frac_negative": float(np.mean(v < 0)),})

    out = pd.DataFrame(rows).sort_values("mean_silhouette")
    return out.reset_index(drop=True)


def assignment_agreement(distances,
                         labels,):
    """
    Fraction of sites whose assigned cluster is also their nearest centroid.

    Hierarchical clustering never checks this — a site is placed by the merge
    history of its neighbourhood, not by centroid proximity. Low agreement means
    the clusters are not compact blobs (which may be fine, or may mean the tree
    is chaining). It is also exactly the disagreement a nearest-centroid transfer
    to the mutant datasets would inherit, so it is worth knowing before relying
    on that route.

    Args:
        distances: np.ndarray of shape (n_sites, n_clusters) from
                   distances_to_centroids().
        labels: array-like of 0-based cluster labels, length n_sites, indexing
                the columns of `distances`.

    Returns:
        agreement: float — overall fraction agreeing.
        per_cluster: pd.Series mapping cluster -> agreement fraction within it.
    """
    labels = np.asarray(labels)
    nearest = np.argmin(distances, axis=1)
    agree = nearest == labels
    per_cluster = pd.Series(agree).groupby(labels).mean()
    per_cluster.index.name = "cluster"
    return float(agree.mean()), per_cluster


# =============================================================================
# Cluster-level tree — the basis for constrained merging
# =============================================================================

def cluster_tree_nodes(Z,
                       labels,):
    """
    Map every cluster to the tree node that produced it.

    A cut of a hierarchical tree is only valid if each cluster is exactly the set
    of leaves under one node. This function finds that node for each cluster and
    verifies the property, which is what makes the merge operations below sound:
    if two clusters are nodes, their union is a node too whenever they share a
    parent, so merging them requires no recomputation at all.

    Args:
        Z: linkage matrix from compute_linkage().
        labels: array-like of 0-based cluster labels, one per site, in the same
                row order the linkage was built from.

    Returns:
        node_of_cluster: dict mapping cluster label -> node id in Z
                         (ids < n are leaves; id n + i is the node created by row
                         i of Z).

    Raises:
        ValueError: if a cluster is not a single subtree — which happens after a
                    non-sibling manual merge. In that case the cluster-level tree
                    is undefined and you must re-cut the tree instead.
    """
    labels = np.asarray(labels)
    n = len(labels)

    parent = np.full(2 * n - 1, -1, dtype=int)
    counts = np.ones(2 * n - 1, dtype=int)
    for i in range(n - 1):
        a, b = int(Z[i, 0]), int(Z[i, 1])
        parent[a] = n + i
        parent[b] = n + i
        counts[n + i] = int(Z[i, 3])

    tree_nodes = to_tree(Z, rd=True)[1]

    node_of_cluster = {}
    for c in np.unique(labels):
        members = np.where(labels == c)[0]
        size = len(members)

        # Walk up from any member while the ancestor still fits inside the cluster.
        v = int(members[0])
        while parent[v] != -1 and counts[parent[v]] <= size:
            v = parent[v]

        if counts[v] != size or set(tree_nodes[v].pre_order()) != set(members.tolist()):
            raise ValueError(f"Cluster {c} is not a single subtree of the linkage tree, "
                             f"so it has no parent node. This happens when clusters were "
                             f"merged across the hierarchy (allow_non_sibling=True) or when "
                             f"the labels do not come from this tree. Re-cut the tree with "
                             f"hierarchical_clustering(precomputed_Z=...) to restore a valid cut.")
        node_of_cluster[int(c)] = int(v)

    return node_of_cluster


def cluster_level_linkage(Z,
                          labels,):
    """
    Reduce the site-level tree to a tree over the clusters.

    This is the part of the hierarchy that survives above the cut: k leaves (the
    clusters) and k-1 merges, with the *exact* heights from the original tree.
    It is what you read to decide which clusters to merge — unlike
    clustering.compute_centroid_linkage(), which re-clusters the centroids and
    therefore invents a hierarchy that the model never had.

    The returned matrix is a valid scipy linkage matrix, so it can be passed to
    dendrogram(), fcluster() and plotting_functions.plot_cluster_hierarchy().

    Args:
        Z: linkage matrix from compute_linkage().
        labels: array-like of 0-based cluster labels, one per site.

    Returns:
        Zc: np.ndarray of shape (n_clusters - 1, 4) — linkage over the clusters.
            Leaf i of Zc corresponds to `cluster_order[i]`.
        cluster_order: list of cluster labels giving the leaf order of Zc.
        node_of_cluster: dict cluster label -> node id in the original tree.
    """
    labels = np.asarray(labels)
    n = len(labels)
    node_of_cluster = cluster_tree_nodes(Z, labels)

    cluster_order = sorted(node_of_cluster)
    k = len(cluster_order)
    if k < 2:
        raise ValueError("Need at least 2 clusters to build a cluster-level tree.")

    sizes_by_cluster = pd.Series(labels).value_counts().to_dict()

    active = {node_of_cluster[c]: i for i, c in enumerate(cluster_order)}
    sizes = {i: int(sizes_by_cluster[c]) for i, c in enumerate(cluster_order)}

    rows = []
    next_id = k
    for i in range(n - 1):
        a, b = int(Z[i, 0]), int(Z[i, 1])
        a_in, b_in = a in active, b in active
        if a_in and b_in:
            ra, rb = active.pop(a), active.pop(b)
            merged_size = sizes[ra] + sizes[rb]
            rows.append([ra, rb, float(Z[i, 2]), merged_size])
            sizes[next_id] = merged_size
            active[n + i] = next_id
            next_id += 1
        elif a_in or b_in:
            # Impossible for a valid cut: a cluster node's sibling is always
            # either another cluster node or a node above the cut.
            raise ValueError("Inconsistent cut: a cluster node was merged with a node "
                             "below the cut. The labels do not come from this tree.")

    Zc = np.array(rows, dtype=float)
    if Zc.shape[0] != k - 1:
        raise ValueError(f"Built {Zc.shape[0]} cluster merges for {k} clusters "
                         f"(expected {k - 1}).")
    return Zc, cluster_order, node_of_cluster


def mergeable_cluster_groups(Z,
                             labels,
                             pairs_only=False,
                             centroids=None,):
    """
    List every set of clusters that can legally be merged, and what it costs.

    A group is legal when its clusters are exactly the leaves under one node of
    the cluster-level tree, i.e. they share a parental node. Merging such a group
    is exact: the result is that node, and no site changes cluster except by
    joining the union.

    Read the table bottom-up in `merge_height`: the cheapest merges (smallest
    height) join clusters that the tree could barely tell apart in the first
    place, which are the ones worth collapsing when a cut produced more clusters
    than you can interpret.

    Args:
        Z: linkage matrix from compute_linkage().
        labels: array-like of 0-based cluster labels, one per site.
        pairs_only: if True, keep only 2-cluster groups (direct siblings).
                    If False (default), also list the larger groups formed higher
                    up the tree, e.g. merging a cluster with an already-merged pair.
        centroids: optional (n_clusters, n_timepoints, n_conditions) array. When
                   given, a `centroid_distance` column is added for pairs — the
                   Euclidean distance between the two cluster mean profiles, which
                   answers "would the merged mean profile even look different?".

    Returns:
        DataFrame sorted by merge_height ascending, with columns:
        clusters (tuple of labels), n_clusters, merge_height, merged_size,
        is_sibling_pair, and centroid_distance when available.
    """
    Zc, cluster_order, _ = cluster_level_linkage(Z, labels)
    k = len(cluster_order)

    # Leaves under each reduced node, so a group can be reported as cluster labels.
    members = {i: [cluster_order[i]] for i in range(k)}

    C_flat = None
    if centroids is not None:
        C_flat = np.asarray(centroids, dtype=float)
        C_flat = C_flat.reshape(C_flat.shape[0], -1)

    rows = []
    for i in range(Zc.shape[0]):
        ra, rb = int(Zc[i, 0]), int(Zc[i, 1])
        group = sorted(members[ra] + members[rb])
        members[k + i] = group

        is_pair = (ra < k) and (rb < k)
        row = {"clusters": tuple(group),
               "n_clusters": len(group),
               "merge_height": float(Zc[i, 2]),
               "merged_size": int(Zc[i, 3]),
               "is_sibling_pair": bool(is_pair),}

        if C_flat is not None and is_pair:
            ca, cb = cluster_order[ra], cluster_order[rb]
            row["centroid_distance"] = float(np.linalg.norm(C_flat[ca] - C_flat[cb]))
        elif C_flat is not None:
            row["centroid_distance"] = float("nan")

        rows.append(row)

    out = pd.DataFrame(rows)
    if pairs_only:
        out = out[out["is_sibling_pair"]]
    return out.sort_values("merge_height").reset_index(drop=True)


def suggest_merges(Z,
                   labels,
                   max_height=None,
                   min_cluster_size=None,
                   verbose=True,):
    """
    Propose merges from two rules you can actually justify in writing.

    Rule 1 (`max_height`) — merge sibling clusters whose parent node sits below
    this merge distance. These are clusters the tree separated only marginally,
    so keeping them apart claims a distinction the data does not support.

    Rule 2 (`min_cluster_size`) — merge any cluster smaller than this into its
    sibling group. Small clusters are rarely interpretable, cannot support
    enrichment testing, and are the usual by-product of hierarchical methods
    shaving off outliers.

    Overlapping proposals are resolved with union-find, so the returned groups
    are disjoint and can all be applied in one merge_clusters() call.

    Args:
        Z: linkage matrix from compute_linkage().
        labels: array-like of 0-based cluster labels, one per site.
        max_height: merge distance below which sibling pairs are proposed.
                    None disables rule 1.
        min_cluster_size: clusters smaller than this are proposed for merging.
                          None disables rule 2.
        verbose: if True, print the proposed groups with their justification.

    Returns:
        groups: list of lists of cluster labels — the proposed merge groups.
                Pass directly as the `merges` argument of merge_clusters().
    """
    if max_height is None and min_cluster_size is None:
        raise ValueError("Give at least one of max_height or min_cluster_size.")

    pairs = mergeable_cluster_groups(Z, labels, pairs_only=True)
    sizes = pd.Series(labels).value_counts().to_dict()

    selected = []
    for _, row in pairs.iterrows():
        a, b = row["clusters"]
        reasons = []
        if max_height is not None and row["merge_height"] <= max_height:
            reasons.append(f"merge height {row['merge_height']:.3f} <= {max_height}")
        if min_cluster_size is not None and min(sizes[a], sizes[b]) < min_cluster_size:
            reasons.append(f"cluster of {min(sizes[a], sizes[b])} sites < {min_cluster_size}")
        if reasons:
            selected.append(((a, b), "; ".join(reasons)))

    # Union-find so that overlapping pairs collapse into single disjoint groups.
    parent = {c: c for c in sizes}

    def find(x,):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), _ in selected:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    grouped = {}
    for c in sizes:
        grouped.setdefault(find(c), []).append(c)
    groups = [sorted(g) for g in grouped.values() if len(g) > 1]
    groups.sort(key=lambda g: g[0])

    if verbose:
        if not groups:
            print("No merges proposed under these thresholds.")
        else:
            print(f"{len(groups)} merge group(s) proposed "
                  f"({len(sizes)} -> {len(sizes) - sum(len(g) - 1 for g in groups)} clusters):")
            for g in groups:
                total = sum(sizes[c] for c in g)
                print(f"  merge {g} -> {total} sites")
            print("\nReasons per sibling pair:")
            for (a, b), reason in selected:
                print(f"  ({a}, {b}): {reason}")

    return groups


def merge_clusters(df,
                   cluster_column,
                   merges,
                   Z=None,
                   new_column=None,
                   allow_non_sibling=False,
                   relabel=True,
                   verbose=True,):
    """
    Apply a set of cluster merges to a clustered DataFrame.

    By default every proposed group is validated against the tree: the clusters
    in a group must share a parental node, so the merged cluster is still a node
    of the hierarchy and every downstream tree operation keeps working. Merging
    clusters that do not share a parent breaks that property — it is allowed with
    `allow_non_sibling=True`, but the resulting labels can no longer be fed back
    into cluster_level_linkage() / mergeable_cluster_groups().

    Args:
        df: DataFrame carrying `cluster_column` (typically HierarchicalResult.df).
        cluster_column: name of the existing cluster label column.
        merges: list of lists of cluster labels to merge, e.g. [[3, 7], [1, 2, 5]].
                A group of one is ignored.
        Z: linkage matrix used to validate the merges. Skipped when None, which
           also skips validation — pass it unless you know what you are doing.
        new_column: name of the output column. Defaults to
                    f"{cluster_column}_merged", so the original labels survive
                    for comparison.
        allow_non_sibling: if True, downgrade the shared-parent check to a warning.
        relabel: if True (default), renumber the surviving clusters 0..k'-1 by
                 decreasing size. If False, each merged group keeps the smallest
                 label of its members and numbering stays sparse.
        verbose: if True, print the mapping and the resulting sizes.

    Returns:
        df: the same DataFrame with the new label column added.
        mapping: dict mapping old cluster label -> new cluster label.
    """
    if new_column is None:
        new_column = f"{cluster_column}_merged"

    labels = df[cluster_column].to_numpy()
    groups = [sorted(set(g)) for g in merges if len(set(g)) > 1]

    seen = set()
    for g in groups:
        overlap = seen & set(g)
        if overlap:
            raise ValueError(f"Cluster(s) {sorted(overlap)} appear in more than one merge "
                             f"group. Combine them into a single group instead.")
        seen |= set(g)

    # --- Validate that each group shares a parental node --------------------
    if Z is not None and groups:
        legal = mergeable_cluster_groups(Z, labels, pairs_only=False)
        legal_sets = {frozenset(t) for t in legal["clusters"]}
        for g in groups:
            if frozenset(g) not in legal_sets:
                message = (f"Clusters {g} do not share a parental node — merging them "
                           f"would create a cluster that is not a subtree of the "
                           f"hierarchy. Check mergeable_cluster_groups() for the legal "
                           f"groups, or pass allow_non_sibling=True to force it.")
                if allow_non_sibling:
                    print(f"WARNING: {message}")
                else:
                    raise ValueError(message)

    # --- Build the mapping --------------------------------------------------
    mapping = {c: c for c in np.unique(labels)}
    for g in groups:
        target = min(g)
        for c in g:
            mapping[c] = target

    merged = np.array([mapping[l] for l in labels])

    if relabel:
        final = relabel_by_size(merged)
        # Compose the two steps so the returned mapping is old -> final label.
        intermediate_to_final = {}
        for old_intermediate, new_final in zip(merged, final):
            intermediate_to_final[old_intermediate] = new_final
        mapping = {old: intermediate_to_final[mid] for old, mid in mapping.items()}
        merged = final

    df[new_column] = merged

    if verbose:
        n_before = len(np.unique(labels))
        n_after = len(np.unique(merged))
        print(f"{n_before} -> {n_after} clusters, written to {new_column!r}")
        for g in groups:
            print(f"  merged {g} -> cluster {mapping[g[0]]}")
        print(f"\nNew cluster sizes:\n{pd.Series(merged).value_counts().sort_index()}")

    return df, mapping


def merge_clusters_by_height(Z,
                             labels,
                             height,
                             verbose=True,):
    """
    Collapse the cluster-level tree at a given merge distance, in one step.

    Equivalent to applying every sibling merge that occurs below `height`, and
    the fastest way to answer "what if I only keep the structure that separates
    at distance > h". Every resulting cluster is still a node of the original
    tree, so the result stays fully mergeable.

    Args:
        Z: linkage matrix from compute_linkage().
        labels: array-like of 0-based cluster labels, one per site.
        height: merge distance at which to cut the cluster-level tree.
        verbose: if True, print how many clusters survive and the grouping.

    Returns:
        new_labels: np.ndarray of 0-based labels, relabelled by decreasing size.
        mapping: dict mapping old cluster label -> new cluster label.
    """
    labels = np.asarray(labels)
    Zc, cluster_order, _ = cluster_level_linkage(Z, labels)

    reduced = fcluster(Zc, t=height, criterion="distance")   # one label per cluster leaf
    mapping_raw = {cluster_order[i]: int(reduced[i]) for i in range(len(cluster_order))}

    intermediate = np.array([mapping_raw[l] for l in labels])
    new_labels = relabel_by_size(intermediate)

    intermediate_to_final = {}
    for mid, fin in zip(intermediate, new_labels):
        intermediate_to_final[mid] = fin
    mapping = {old: intermediate_to_final[mid] for old, mid in mapping_raw.items()}

    if verbose:
        groups = {}
        for old, new in mapping.items():
            groups.setdefault(new, []).append(old)
        print(f"Cut at height {height}: {len(cluster_order)} -> {len(groups)} clusters")
        for new in sorted(groups):
            old_list = sorted(groups[new])
            if len(old_list) > 1:
                print(f"  new cluster {new} <- old {old_list}")

    return new_labels, mapping


def plot_cluster_tree(Z,
                      labels,
                      centroids=None,
                      merge_height_line=None,
                      figsize=(14, 5),
                      title="Cluster-level dendrogram",):
    """
    Dendrogram over the clusters — the figure you read before deciding a merge.

    Each leaf is one cluster, annotated with its site count; each internal node
    is drawn at the true merge distance from the site-level tree. Two leaves
    joined low down are clusters the data barely separates: exactly the pairs
    that mergeable_cluster_groups() lists first.

    Args:
        Z: linkage matrix from compute_linkage().
        labels: array-like of 0-based cluster labels, one per site.
        centroids: unused placeholder kept so callers can pass the result's
                   centroids without branching; profile shapes are better read
                   with plotting_functions.plot_cluster_hierarchy().
        merge_height_line: if given, draw a horizontal line at this merge
                           distance to preview merge_clusters_by_height().
        figsize: (width, height) of the figure.
        title: axes title.

    Returns:
        fig, ax, Zc — the Figure, the Axes, and the cluster-level linkage matrix.
    """
    labels = np.asarray(labels)
    Zc, cluster_order, _ = cluster_level_linkage(Z, labels)
    sizes = pd.Series(labels).value_counts().to_dict()

    leaf_labels = [f"C{c} (n={sizes[c]})" for c in cluster_order]

    fig, ax = plt.subplots(figsize=figsize)
    dendrogram(Zc,
               labels=leaf_labels,
               ax=ax,
               color_threshold=merge_height_line,)
    ax.set_title(title)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Merge distance in the original tree")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=9)

    if merge_height_line is not None:
        ax.axhline(merge_height_line, color="crimson", linestyle="--", linewidth=1.2,
                   label=f"merge below {merge_height_line}")
        ax.legend(fontsize=8)

    fig.tight_layout()
    return fig, ax, Zc
