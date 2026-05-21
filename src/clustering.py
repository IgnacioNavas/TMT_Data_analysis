
from src.column_spec import ColumnSpec

import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from src.transformations import parse_columns

from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, KShape
from tslearn.metrics import cdist_dtw
import tslearn as tsl

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

#---------------------
# Helper fucntions
#---------------------
def reshape_df(df,
               time_series,
               dimensions,
               len_time_serie,
               stoichiometry=None,
               verbose = None,
               labels="site",
               transpose=False,
               ):
    '''
    Reshape dataframe so it is multivariate format. Return the dataframe in numpy format so can be used, and list with the names of myseries
    '''
    sub_df = df[time_series].copy()
    mySeries = sub_df.to_numpy()
    namesofMySeries = df[labels]

    multivariate_shape = (len(df), dimensions, len_time_serie)
    if verbose == True and transpose == False:
        print(f"Reshaping dataframe to shape {multivariate_shape}")

    multivariate_df = np.reshape(mySeries, multivariate_shape)
    if transpose == True:
        multivariate_df = multivariate_df.transpose(0, 2, 1)
        if verbose == True:
            print(f"Reshaping dataframe to shape {multivariate_df.shape}")
            # After transpose: (n_samples, len_time_serie, dimensions) → (5000, 6, 3)

    # --- Stoichiometry reweighting ---
    # by defaulft : {"_EGF_": 1, "_INS_": 1, "_EGFnINS_": 1}
    # Requires transpose=True so that axis 2 is the conditions dimension.
    if stoichiometry is not None:
        if not transpose:
            raise ValueError("stoichiometry requires transpose=True (axis 2 must be the conditions dimension)")
        representative_cols = time_series[::len_time_serie]

        # Build multiplier list: one integer per condition (axis=2 slice)
        condition_multipliers = []
        for col in representative_cols:
            multiplier = 1  # default if no key matches
            for substring, mult in stoichiometry.items():
                if substring in col:
                    multiplier = mult
                    break
            condition_multipliers.append(multiplier)

        if verbose:
            print(f"Conditions detected:   {list(representative_cols)}")
            print(f"Stoichiometry applied: {condition_multipliers}")

        # Repeat each condition slice along axis=2 according to its multiplier
        # multivariate_df shape after transpose: (n_samples, len_time_serie, n_conditions)
        repeated_slices = []
        for cond_idx, multiplier in enumerate(condition_multipliers):
            slice_ = multivariate_df[:, :, cond_idx]  # shape: (n_samples, len_time_serie)
            slice_ = slice_[:, :, np.newaxis]  # shape: (n_samples, len_time_serie, 1)
            repeated_slices.append(np.repeat(slice_, multiplier, axis=2))

        multivariate_df = np.concatenate(repeated_slices, axis=2)
        # Final shape: (5000, 6, sum(multipliers)) e.g. (5000, 6, 4)

        if verbose:
            print(f"Shape after stoichiometry reweighting: {multivariate_df.shape}")


    return multivariate_df, namesofMySeries

#----------------------
# Clustering
#----------------------
def tslearn_clustering_KMeans(df_to_cluster,
                              data_type,
                              condition_for_clustering=None,
                              cell_lines=None,
                              exclude_full=False,
                              stoichiometry=None,
                              cluster_column_name="",
                              number_of_clusters=10,
                              max_iterations=1000,
                              n_init=5,
                              metric='euclidean',
                              df_dimensions=None,
                              random_state=0,
                              time_series_length=None,
                              transpose=False,
                              verbose=True,
                              testing=False,
                              barycenter_calculations=False,
                              ):

    if condition_for_clustering is None:
        condition_for_clustering = []
    if cell_lines is None:
        cell_lines = []
    if df_dimensions is None:
        raise ValueError("df_dimensions is required")
    if time_series_length is None:
        raise ValueError("time_series_length is required")

    column_selection = ColumnSpec.select(df_to_cluster, cell_lines=cell_lines, data_type=data_type, conditions=condition_for_clustering, exclude_full=exclude_full)

    if verbose == True:
        print(f"Column selection: {column_selection}\n")

    multivariate_df, names_of_myseries = reshape_df(df=df_to_cluster,
                                                    time_series=column_selection,
                                                    dimensions=df_dimensions,
                                                    len_time_serie=time_series_length,
                                                    stoichiometry=stoichiometry,
                                                    transpose=transpose,
                                                    labels="site",
                                                    verbose=verbose,
                                                    )

    if verbose == True:
        print(f"\nThe size of the dataset is {multivariate_df.shape}")
        print(f"Example:\n{multivariate_df[0]}")

    clustering = TimeSeriesKMeans(n_clusters=number_of_clusters,
                                  max_iter=max_iterations,
                                  n_init=n_init,
                                  metric=metric,
                                  max_iter_barycenter=1000,
                                  verbose=verbose,
                                  random_state=random_state).fit(multivariate_df)
    df_to_cluster[f"{cluster_column_name}"] = clustering.labels_

    if testing == True:
        if barycenter_calculations == True:
            barycenters_distances = TimeSeriesKMeans(n_clusters=number_of_clusters,
                                                     max_iter=max_iterations,
                                                     n_init=n_init,
                                                     metric=metric,
                                                     max_iter_barycenter=1000,
                                                     verbose=verbose,
                                                     random_state=random_state).fit_transform(multivariate_df)
            return df_to_cluster, clustering, multivariate_df, barycenters_distances
        else:
            return df_to_cluster, clustering, multivariate_df
    else:
        return df_to_cluster


def tslearn_clustering_KShape(df_to_cluster,
                              data_type,
                              condition_for_clustering=None,
                              cell_lines=None,
                              stoichiometry=None,
                              exclude_full=False,
                              number_of_clusters=10,
                              cluster_column_name="",
                              n_init=5,
                              max_iterations=1000,
                              df_dimensions=None,
                              time_series_length=None,
                              random_state=0,
                              transpose=False,
                              verbose=True,
                              testing=False):

    if condition_for_clustering is None:
        condition_for_clustering = []
    if cell_lines is None:
        cell_lines = []
    if df_dimensions is None:
        raise ValueError("df_dimensions is required")
    if time_series_length is None:
        raise ValueError("time_series_length is required")

    column_selection = ColumnSpec.select(df_to_cluster, cell_lines=cell_lines, data_type=data_type, conditions=condition_for_clustering, exclude_full=exclude_full)
    if verbose == True:
        print(f"Column selection: {column_selection}\n")

    multivariate_df, names_of_myseries = reshape_df(df=df_to_cluster,
                                                    time_series=column_selection,
                                                    dimensions=df_dimensions,
                                                    len_time_serie=time_series_length,
                                                    stoichiometry=stoichiometry,
                                                    transpose=transpose,
                                                    labels="site",
                                                    verbose=verbose)

    if verbose == True:
        print(f"\nThe size of the dataset is {multivariate_df.shape}")

    clustering = KShape(n_clusters=number_of_clusters,
                        max_iter=max_iterations,
                        n_init=n_init,
                        verbose=verbose,
                        random_state=random_state).fit(multivariate_df)

    df_to_cluster[f"{cluster_column_name}"] = clustering.labels_

    if testing == True:
        return df_to_cluster, clustering, multivariate_df
    else:
        return df_to_cluster


def kernnel_clustering(df_to_cluster,
                       transpose=True,
                       data_type="log2:FC",
                       cell_lines=None,
                       stoichiometry=None,
                       exclude_full=True,
                       condition_for_clustering=None,
                       df_dimensions=None,
                       time_series_length=None,
                       seed=0,
                       n_clusters=25,
                       n_init=20,
                       verbose=True,
                       kernel="gak",
                       kernel_params=None,
                       cluster_column_name="",
                       testing=False):

    if condition_for_clustering is None:
        condition_for_clustering = []
    if cell_lines is None:
        cell_lines = []
    if kernel_params is None:
        kernel_params = {"sigma": "auto"}
    if df_dimensions is None:
        raise ValueError("df_dimensions is required")
    if time_series_length is None:
        raise ValueError("time_series_length is required")

    column_selection = ColumnSpec.select(df_to_cluster, cell_lines=cell_lines, data_type=data_type, conditions=condition_for_clustering, exclude_full=exclude_full)

    if verbose == True:
        print(f"Column selection: {column_selection}\n")

    multivariate_df, names_of_myseries = reshape_df(df=df_to_cluster,
                                                    time_series=column_selection,
                                                    labels="site",
                                                    dimensions=df_dimensions,
                                                    len_time_serie=time_series_length,
                                                    stoichiometry=stoichiometry,
                                                    transpose=transpose,
                                                    verbose=verbose)

    if verbose == True:
        print(f"\nThe size of the dataset is {multivariate_df.shape}")
        print(f"Example:\n{multivariate_df[0]}")

    clustering_gak_km = KernelKMeans(n_clusters=n_clusters,
                                     kernel=kernel,
                                     kernel_params=kernel_params,
                                     n_init=n_init,
                                     verbose=verbose,
                                     random_state=seed).fit(multivariate_df)

    df_to_cluster[f"{cluster_column_name}"] = clustering_gak_km.labels_

    if testing == True:
        return df_to_cluster, clustering_gak_km, multivariate_df
    else:
        return df_to_cluster


def hdbscan_clustering(df_to_cluster,
                       data_type,
                       condition_for_clustering=None,
                       cell_lines=None,
                       exclude_full=True,
                       cluster_column_name="HDBSCAN",
                       min_cluster_size=10,
                       min_samples=5,
                       metric="euclidean",
                       scale=False,
                       verbose=True,
                       ):
    """
    NOT A GOOD CLUSTERING METHOD

    Cluster phosphosites using HDBSCAN on their flat time-series feature vector.

    Unlike the tslearn-based functions, HDBSCAN does not require specifying the
    number of clusters in advance.  It finds dense regions automatically and marks
    low-density sites as noise (label -1).

    The feature matrix is built by selecting columns with ColumnSpec.select() and
    filling missing values with 0 (no change from starve).  Optionally, features
    can be standardised with StandardScaler before clustering.

    Note — limitations relevant to this project:
      * HDBSCAN operates on the flat feature vector with euclidean distance.
        DTW (used by KMeans) is not supported natively and would require a
        precomputed distance matrix, which cannot then be used to assign new sites.
      * Noise sites (label -1) cannot be used for classifier training.
      * There are no cluster_centers_, so the hierarchy and assignment-confidence
        analyses available for KMeans are not applicable here.

    Best used as an exploratory/validation tool alongside KMeans, not as the
    primary clustering method when a downstream classifier is needed.

    Args:
        df_to_cluster: DataFrame following the project naming convention.
        data_type: data-type string, e.g. "log2:FC".
        condition_for_clustering: list of condition substrings, e.g. ["_EGF_", "_INS_"].
        cell_lines: list of cell-line prefixes, e.g. ["WT"].
        exclude_full: if True, exclude the 'full' timepoint (default True).
        cluster_column_name: column name to store cluster labels in df (default "HDBSCAN").
        min_cluster_size: minimum number of sites to form a cluster (default 10).
                          Acts as an implicit control on the number of clusters —
                          larger values produce fewer, broader clusters.
        min_samples: number of samples in a neighbourhood to be considered a core point
                     (default 5).  Higher values make the algorithm more conservative,
                     classifying more sites as noise.
        metric: distance metric passed to HDBSCAN (default "euclidean").
        scale: if True, standardise features with StandardScaler before clustering
               (default False, since log2:FC values are already on a comparable scale).
        verbose: if True, print column selection, cluster count, and noise count.

    Returns:
        df_to_cluster: input DataFrame with two new columns:
            * cluster_column_name — integer cluster labels (-1 = noise).
            * cluster_column_name + ":prob" — soft membership probability (0–1)
              from HDBSCAN; noise sites have probability 0.
        clusterer: fitted hdbscan.HDBSCAN object.
    """
    import hdbscan
    from sklearn.preprocessing import StandardScaler

    if condition_for_clustering is None:
        condition_for_clustering = []
    if cell_lines is None:
        cell_lines = []

    cols = ColumnSpec.select(
        df_to_cluster,
        cell_lines=cell_lines,
        data_type=data_type,
        conditions=condition_for_clustering,
        exclude_full=exclude_full,
        exclude_replicate_cols=True,
    )
    if not cols:
        raise ValueError(
            f"No columns found for cell_lines={cell_lines}, "
            f"conditions={condition_for_clustering}, data_type={data_type!r}."
        )

    if verbose:
        print(f"Column selection ({len(cols)}): {cols}\n")

    X = df_to_cluster[cols].fillna(0).values

    if scale:
        X = StandardScaler().fit_transform(X)
        if verbose:
            print("Features standardised with StandardScaler.")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size = min_cluster_size,
        min_samples      = min_samples,
        metric           = metric,
    ).fit(X)

    df_to_cluster[cluster_column_name]          = clusterer.labels_
    df_to_cluster[cluster_column_name + ":prob"] = clusterer.probabilities_

    if verbose:
        n_noise    = (clusterer.labels_ == -1).sum()
        n_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
        print(f"Clusters found : {n_clusters}")
        print(f"Noise sites    : {n_noise} ({n_noise / len(clusterer.labels_) * 100:.1f}%)")

    return df_to_cluster, clusterer


# =============================================================================
# Cluster quality / similarity metrics
# =============================================================================

def _resolve_clustering_columns(df,
                                cell_lines,
                                conditions,
                                data_type,
                                exclude_full):
    """
    Return (col_selection, n_timepoints, n_dims) derived from ColumnSpec.

    n_dims  = number of (cell_line × condition) combinations present in col_selection.
    n_timepoints = number of timepoints per combination (assumed equal across all).
    """
    cols = ColumnSpec.select(
        df,
        cell_lines=cell_lines,
        data_type=data_type,
        conditions=conditions,
        exclude_full=exclude_full,
        exclude_replicate_cols=True,
    )
    if not cols:
        raise ValueError(
            f"No columns found for cell_lines={cell_lines}, conditions={conditions}, "
            f"data_type={data_type!r}. Check naming convention and available columns."
        )
    n_dims = len(cell_lines) * len(conditions)
    n_timepoints = len(cols) // n_dims
    return cols, n_timepoints, n_dims


def cluster_similarity_cdist_dtw(
    df,
    cell_lines,
    conditions,
    data_type="log2:FC",
    cluster_column_name="",
    exclude_full=True,
    summary="mean",
    transpose=True, # True by default
    verbose=False,
):
    """
    Mean or median DTW distance between sites within each cluster (full multivariate time series).

    A lower value means sites within a cluster are more similar to each other.

    Args:
        df: DataFrame following the project naming convention, with a cluster column.
        cell_lines: list of cell-line prefixes, e.g. ["WT"].
        conditions: list of condition substrings, e.g. ["_EGF_", "_INS_", "_EGFnINS_"].
        data_type: data-type string, e.g. "log2:FC" (default "log2:FC").
        cluster_column_name: name of the column holding cluster labels.
        exclude_full: if True, exclude the 'full' timepoint (default True).
        summary: 'mean' (default) or 'median' applied to the upper-triangle of the distance matrix.
        transpose: if True (default), reshape to (n_samples, n_timepoints, n_conditions).
        verbose: if True, print reshape info.

    Returns:
        dict mapping cluster label → scalar DTW distance.
    """
    if summary not in ("mean", "median"):
        raise ValueError(f"summary must be 'mean' or 'median', got {summary!r}")

    cols, n_timepoints, n_dims = _resolve_clustering_columns(
        df, cell_lines, conditions, data_type, exclude_full
    )

    cluster_metric = {}
    for cluster in sorted(df[cluster_column_name].unique()):
        if cluster == 999:
            continue
        sub = df.loc[df[cluster_column_name] == cluster]
        X, _ = reshape_df(df=sub,
                          time_series=cols,
                          dimensions=n_dims,
                          len_time_serie=n_timepoints,
                          transpose=transpose, # True by default
                          verbose=verbose,
        )
        D = cdist_dtw(dataset1=X)
        tri = D[np.triu_indices_from(D, k=1)]
        cluster_metric[cluster] = float(np.mean(tri) if summary == "mean" else np.median(tri))

    return cluster_metric


def _mean_dtw_per_condition(X_time_cond, conditions):
    """
    Compute mean DTW distance per condition for a single cluster.

    Args:
        X_time_cond: array of shape (n_sites, n_timepoints, n_conditions) with transpose=True.
        conditions: list of condition labels matching the last axis of X_time_cond.

    Returns:
        dict mapping condition label → mean DTW scalar.
    """
    out = {}
    for i, cond in enumerate(conditions):
        Xc = X_time_cond[:, :, i][:, :, None]
        D = cdist_dtw(Xc)
        out[cond] = float(np.mean(D[np.triu_indices_from(D, k=1)]))
    return out


def cluster_similarity_per_condition(df,
                                     cell_lines,
                                     conditions,
                                     data_type="log2:FC",
                                     cluster_column_name="",
                                     exclude_full=True,
                                     transpose=True,
                                     verbose=False,):
    """
    Mean DTW distance per condition for each cluster.

    Args:
        df: DataFrame following the project naming convention, with a cluster column.
        cell_lines: list of cell-line prefixes, e.g. ["WT"].
        conditions: list of condition substrings, e.g. ["_EGF_", "_INS_", "_EGFnINS_"].
        data_type: data-type string, e.g. "log2:FC" (default "log2:FC").
        cluster_column_name: name of the column holding cluster labels.
        exclude_full: if True, exclude the 'full' timepoint (default True).
        transpose: if True (default), reshape to (n_samples, n_timepoints, n_conditions).
        verbose: if True, print reshape info.

    Returns:
        dict mapping cluster label → {condition_label: mean_dtw_scalar}.
    """
    cols, n_timepoints, n_dims = _resolve_clustering_columns(
        df, cell_lines, conditions, data_type, exclude_full
    )
    cond_labels = [c.strip("_") for c in conditions]

    cluster_metric = {}
    for cluster in sorted(df[cluster_column_name].unique()):
        if cluster == 999:
            continue
        sub = df.loc[df[cluster_column_name] == cluster]
        X, _ = reshape_df(df=sub,
                          time_series=cols,
                          dimensions=n_dims,
                          len_time_serie=n_timepoints,
                          transpose=transpose,
                          verbose=verbose,)
        cluster_metric[cluster] = _mean_dtw_per_condition(X, cond_labels)

    return cluster_metric


def timepoint_pairwise_distances_within_condition(X_time_cond,
                                                  conditions,
                                                  summary="mean"):
    """
    Mean or median pairwise distance between sites at each individual timepoint, per condition.

    Args:
        X_time_cond: array of shape (n_sites, n_timepoints, n_conditions) with transpose=True.
        conditions: list of condition labels matching the last axis of X_time_cond.
        summary: 'mean' (default) or 'median' applied to upper-triangle distances per timepoint.

    Returns:
        dict mapping condition label → np.ndarray of shape (n_timepoints,).
    """
    if summary not in ("mean", "median"):
        raise ValueError(f"summary must be 'mean' or 'median', got {summary!r}")

    X = np.asarray(X_time_cond)
    n, T, C = X.shape
    results = {}

    for c, cond in enumerate(conditions):
        M = X[:, :, c]
        summary_t = np.full(T, np.nan, dtype=float)
        if n >= 2:
            iu = np.triu_indices(n, k=1)
            for t in range(T):
                v = M[:, t][:, None]
                Dt = np.abs(v - v.T)
                tri = Dt[iu]
                summary_t[t] = float(np.mean(tri) if summary == "mean" else np.median(tri))
        results[cond] = summary_t

    return results


def cluster_similarity_per_condition_per_timepoint(df,
                                                   cell_lines,
                                                   conditions,
                                                   data_type="log2:FC",
                                                   cluster_column_name="",
                                                   exclude_full=True,
                                                   summary="mean",
                                                   transpose=True,
                                                   verbose=False,):
    """
    Mean pairwise distance per condition and per timepoint for each cluster.

    Args:
        df: DataFrame following the project naming convention, with a cluster column.
        cell_lines: list of cell-line prefixes, e.g. ["WT"].
        conditions: list of condition substrings, e.g. ["_EGF_", "_INS_", "_EGFnINS_"].
        data_type: data-type string, e.g. "log2:FC" (default "log2:FC").
        cluster_column_name: name of the column holding cluster labels.
        exclude_full: if True, exclude the 'full' timepoint (default True).
        summary: 'mean' (default) or 'median' applied at each timepoint.
        transpose: if True (default), reshape to (n_samples, n_timepoints, n_conditions).
        verbose: if True, print reshape info.

    Returns:
        dict mapping cluster label → {condition_label: np.ndarray of shape (n_timepoints,)}.
    """
    cols, n_timepoints, n_dims = _resolve_clustering_columns(
        df, cell_lines, conditions, data_type, exclude_full
    )
    cond_labels = [c.strip("_") for c in conditions]

    cluster_metric = {}
    for cluster in sorted(df[cluster_column_name].unique()):
        if cluster == 999:
            continue
        sub = df.loc[df[cluster_column_name] == cluster]
        X, _ = reshape_df(df=sub,
                          time_series=cols,
                          dimensions=n_dims,
                          len_time_serie=n_timepoints,
                          transpose=transpose,
                          verbose=verbose,)
        cluster_metric[cluster] = timepoint_pairwise_distances_within_condition(X,
                                                                                cond_labels,
                                                                                summary=summary)

    return cluster_metric


def combine_conditions(scores_per_cluster,
                       how="mean",
                       cond_order=("EGF", "INS", "EGFnINS")):
    """
    Collapse per-condition cluster scores into a single scalar per cluster.

    Args:
        scores_per_cluster: dict mapping cluster → {condition: scalar}, as returned by
                            cluster_similarity_per_condition.
        how: 'mean' (default), 'median', or 'max'.
        cond_order: tuple of condition labels to include (default all three stimulations).

    Returns:
        dict mapping cluster label → scalar.
    """
    if how not in ("mean", "median", "max"):
        raise ValueError(f"how must be 'mean', 'median', or 'max', got {how!r}")
    combined = {}
    for k, d in scores_per_cluster.items():
        vals = np.array([d[c] for c in cond_order if c in d], dtype=float)
        if how == "mean":
            combined[k] = float(np.nanmean(vals))
        elif how == "max":
            combined[k] = float(np.nanmax(vals))
        else:
            combined[k] = float(np.nanmedian(vals))
    return combined

def compute_centroid_linkage(centers,
                             method="weighted",):
    """
    Compute pairwise DTW distances between KMeans cluster centers and
    run hierarchical clustering on those distances.

    Takes the cluster_centers_ array from a fitted TimeSeriesKMeans model
    (shape: n_clusters × n_timepoints × n_conditions) and returns a linkage
    matrix that can be passed directly to scipy dendrogram or
    plot_cluster_hierarchy().

    Args:
        centers: np.ndarray of shape (n_clusters, n_timepoints, n_conditions) —
                 the cluster_centers_ attribute of a fitted TimeSeriesKMeans model.
        method: linkage method passed to scipy.cluster.hierarchy.linkage.
                Options: 'single', 'complete', 'average', 'weighted' (default),
                'centroid', 'median', 'ward'.

    Returns:
        Z: np.ndarray — linkage matrix of shape (n_clusters-1, 4), as returned
           by scipy.cluster.hierarchy.linkage.
        dist_matrix: np.ndarray of shape (n_clusters, n_clusters) — full
                     pairwise DTW distance matrix between cluster centers.
    """
    dist_matrix = cdist_dtw(centers)
    condensed   = squareform(dist_matrix)
    Z           = linkage(condensed, method=method)
    return Z, dist_matrix


def get_site_centroid_distances(df_clustered,
                                barycenters,
                                site,
                                cluster_col,
                                site_col="site",):
    """
    Return the distance-to-centroid vector and assigned cluster for a single phosphosite.

    Looks up the site by name in `site_col`, finds its integer position in df_clustered,
    and retrieves the corresponding row from the barycenters matrix.  Use this to
    understand why a site was assigned to its cluster — the assigned cluster should have
    the lowest distance in the returned array.

    Args:
        df_clustered: DataFrame with a site identifier column and a cluster label column,
                      as returned by tslearn_clustering_KMeans(..., testing=True).
        barycenters: np.ndarray of shape (n_sites, n_clusters) — distance-to-centroid
                     matrix from fit_transform(), as returned when barycenter_calculations=True.
        site: site identifier string to look up (e.g. 'EGFR_HUMAN-Y1068y').
        cluster_col: name of the cluster label column in df_clustered
                     (e.g. 'KMeans_15clusters').
        site_col: name of the site identifier column (default 'site').

    Returns:
        distances: np.ndarray of shape (n_clusters,) — DTW distance from this site to
                   each cluster centroid.
        assigned_cluster: the cluster label assigned to this site.
        row_position: integer row position in df_clustered (and barycenters).
    """
    mask = df_clustered[site_col] == site
    if not mask.any():
        raise ValueError(f"Site '{site}' not found in column '{site_col}'.")

    row_position  = df_clustered.index.get_loc(df_clustered.index[mask][0])
    distances     = barycenters[row_position]
    assigned_cluster = df_clustered.loc[df_clustered.index[mask][0], cluster_col]
    return distances, assigned_cluster, row_position


# moved to src/plotting_functions.py
# def clusters_shared_sites(cluster_df,
#                           clustering_1,
#                           clustering_2,
#                           site=None,
#                           clusters=None,):
#     """
#     Plot a Venn diagram of phosphosites shared between two cluster assignments.
#
#     Provide either `site` (to look up which clusters it belongs to in each assignment)
#     or explicit `clusters=[cluster1_id, cluster2_id]`.
#
#     Args:
#         cluster_df: DataFrame with a 'site' column and at least two cluster label columns.
#         clustering_1: name of the first cluster label column.
#         clustering_2: name of the second cluster label column.
#         site: site identifier string; if given, cluster IDs are inferred automatically.
#         clusters: list [cluster1_id, cluster2_id]; used when `site` is not provided.
#
#     Returns:
#         fig, ax
#     """
#     if clusters is None:
#         clusters = [None, None]
#
#     if site is not None:
#         row = cluster_df.loc[cluster_df["site"] == site, [clustering_1, clustering_2]]
#         if row.empty:
#             raise ValueError(f"Site '{site}' not found in cluster_df.")
#         cluster1_id, cluster2_id = row.iloc[0][clustering_1], row.iloc[0][clustering_2]
#     else:
#         if len(clusters) != 2 or None in clusters:
#             raise ValueError("Provide `site` or both cluster IDs via clusters=[id1, id2].")
#         cluster1_id, cluster2_id = clusters
#
#     set_1 = set(cluster_df.loc[cluster_df[clustering_1] == cluster1_id, "site"])
#     set_2 = set(cluster_df.loc[cluster_df[clustering_2] == cluster2_id, "site"])
#
#     fig, ax = plt.subplots(figsize=(6, 4))
#     venn2(
#         [set_1, set_2],
#         set_labels=(f"{clustering_1}\nCluster {cluster1_id}", f"{clustering_2}\nCluster {cluster2_id}"),
#         ax=ax,
#     )
#     ax.set_title("Shared phosphosites between cluster assignments")
#     return fig, ax

