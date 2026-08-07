from typing import Any

from numpy import ndarray, dtype

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
from scipy.spatial.distance import squareform, pdist

#---------------------
# Helper fucntions
#---------------------
def reshape_df(df,
               time_series,
               dimensions,
               len_time_serie,
               stoichiometry=None,
               labels="site",
               transpose=False,
               verbose=None,
               ):
    '''
    Reshape dataframe so it is multivariate format. Return the dataframe in numpy format so can be used, and list with the names of myseries
    Args:
        df: dataframe
        time_series: list with the column names for the time series to be selected
        dimensions: list with the dimensions of the time series to be selected (if conditions EGF and INS  are used then these will be 2 dimensions)
        len_time_serie: length of the time series. Used for transposing the matrix if needed
        stoichiometry: dictionary with the stoichiometry of the time series to be used. this can be used to modify weights of the conditons or cell lines when clustering
        labels: column name for the labels of the time series
        transpose: false or true. For some algorithms changes the perception the clustering method sees the time series data. Needs to be true to change stoichometry
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

    # --- Stoichiometry reweighting --- by defaulft : {"_EGF_": 1, "_INS_": 1, "_EGFnINS_": 1}
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

        # Repeat each condition slice along axis=2 according to its multiplier.
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
def tslearn_clustering_KMeans(df_to_cluster: object,
                              data_type: object,
                              condition_for_clustering: object = None,
                              cell_lines: object = None,
                              exclude_full: object = False,
                              df_dimensions: object = None,
                              time_series_length: object = None,
                              stoichiometry: object = None,
                              cluster_column_name: object = "",
                              number_of_clusters: object = 10,
                              max_iterations: object = 1000,
                              n_init: object = 5,
                              metric: object = 'euclidean',
                              random_state: object = 0,
                              transpose: object = False,
                              testing: object = False,
                              barycenter_calculations: object = False,
                              verbose: object = True,
                              ):
    """
    Does unsupervised clustering of a time series dataset using KMeans algorithm from tslearn.

    Args:
        df_to_cluster (object): dataframe to cluster
        data_type: type of data used to cluster - log2:FC, raw:mean ( it hase to be data not dependent on replicates)
        condition_for_clustering: experimental condition of the data to use for the clustering
        cell_lines: list of cell lines to cluster
        exclude_full: whether to exclude full cell lines from clustering
        stoichiometry: by default it is 1. I alows to change weight of cell lines or conditions in the clustering
        cluster_column_name: name the column holding the cluster labels in the results will take
        number_of_clusters: number of clusters to generate
        max_iterations: maximum number of iterations
        n_init: number of iterations to run
        metric: metric used to calculate distances between time series. Use suclidean since our time series are already aligned
        random_state : random seed for reproducibility
        df_dimensions: list with the dimensions of the time series to be selected (if conditions EGF and INS  are used then these will be 2 dimensions)
        time_series_length: length of the time series to use
        transpose: false or true. For some algorithms changes the perception the clustering method sees the time series data. Needs to be true to change stoichometry
        testing: option that allows to return more parameters of the clusters
        barycenter_calculations:
    """

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
                              exclude_full=False,
                              df_dimensions=None,
                              time_series_length=None,
                              stoichiometry=None,
                              cluster_column_name="",
                              number_of_clusters=10,
                              n_init=5,
                              max_iterations=1000,
                              random_state=0,
                              transpose=False,
                              testing=False,
                              verbose=True,
                              ):
    """
    Does unsupervised clustering of a time series dataset using KShape algorithm from tslearn. Doesn't have amplitude into account

    Args:
        df_to_cluster (object): dataframe to cluster
        data_type: type of data used to cluster - log2:FC, raw:mean ( it hase to be data not dependent on replicates)
        condition_for_clustering: experimental condition of the data to use for the clustering
        cell_lines: list of cell lines to cluster
        exclude_full: whether to exclude full cell lines from clustering
        stoichiometry: by default it is 1. I alows to change weight of cell lines or conditions in the clustering
        cluster_column_name: name the column holding the cluster labels in the results will take
        number_of_clusters: number of clusters to generate
        max_iterations: maximum number of iterations
        n_init: number of iterations to run
        metric: metric used to calculate distances between time series. Use suclidean since our time series are already aligned
        random_state : random seed for reproducibility
        df_dimensions: list with the dimensions of the time series to be selected (if conditions EGF and INS  are used then these will be 2 dimensions)
        time_series_length: length of the time series to use
        transpose: false or true. For some algorithms changes the perception the clustering method sees the time series data. Needs to be true to change stoichometry
        testing: option that allows to return more parameters of the clusters
    """

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
                       data_type="log2:FC",
                       condition_for_clustering=None,
                       cell_lines=None,
                       exclude_full=True,
                       stoichiometry=None,
                       transpose=True,
                       df_dimensions=None,
                       time_series_length=None,
                       cluster_column_name="",
                       n_clusters=25,
                       seed=0,
                       n_init=20,
                       kernel="gak",
                       kernel_params=None,
                       testing=False,
                       verbose=True,
                       ):
    """
    Does unsupervised clustering of a time series dataset using Kernel algorithm from tslearn.

    Args:
        df_to_cluster (object): dataframe to cluster
        data_type: type of data used to cluster - log2:FC, raw:mean ( it hase to be data not dependent on replicates)
        condition_for_clustering: experimental condition of the data to use for the clustering
        cell_lines: list of cell lines to cluster
        exclude_full: whether to exclude full cell lines from clustering
        stoichiometry: by default it is 1. I alows to change weight of cell lines or conditions in the clustering
        cluster_column_name: name the column holding the cluster labels in the results will take
        number_of_clusters: number of clusters to generate
        max_iterations: maximum number of iterations
        n_init: number of iterations to run
        metric: metric used to calculate distances between time series. Use suclidean since our time series are already aligned
        random_state : random seed for reproducibility
        df_dimensions: list with the dimensions of the time series to be selected (if conditions EGF and INS  are used then these will be 2 dimensions)
        time_series_length: length of the time series to use
        transpose: false or true. For some algorithms changes the perception the clustering method sees the time series data. Needs to be true to change stoichometry
        testing: option that allows to return more parameters of the clusters
    """

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


def cluster_similarity_cdist_dtw( # Time series are aligned, could use euclidean distances: lighter computation
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


def _mean_dtw_per_condition( # Time series are aligned, could use euclidean distances: lighter computation
        X_time_cond,
        conditions
        ):
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


def _mean_distance_per_condition( # Time series are aligned, could use euclidean distances: lighter computation
        X_time_cond,
        conditions,
        metric="euclidean",
        ):
    """
    Compute the mean pairwise distance between sites, per condition, for one cluster.

    Generalises _mean_dtw_per_condition() to support either DTW (time-warping,
    sensitive to profile shape with misaligned timing) or plain Euclidean distance
    (point-by-point; appropriate when all sites share the same fixed timepoints,
    which is the case in this project). Euclidean is also far cheaper to compute.

    Args:
        X_time_cond: array of shape (n_sites, n_timepoints, n_conditions) with transpose=True.
        conditions: list of condition labels matching the last axis of X_time_cond.
        metric: "dtw" (default) or "euclidean".

    Returns:
        dict mapping condition label → mean pairwise distance (NaN if <2 sites).
    """
    out = {}
    for i, cond in enumerate(conditions):
        Xc = X_time_cond[:, :, i]  # (n_sites, n_timepoints)
        if metric == "dtw":
            D = cdist_dtw(Xc[:, :, None])
            tri = D[np.triu_indices_from(D, k=1)]
        elif metric == "euclidean":
            # pdist returns the condensed upper-triangle of pairwise distances directly
            tri = pdist(Xc, metric="euclidean")
        else:
            raise ValueError(f"metric must be 'dtw' or 'euclidean', got {metric!r}")
        out[cond] = float(np.mean(tri)) if len(tri) else float("nan")
    return out


def cluster_similarity_per_condition(df,
                                     cell_lines,
                                     conditions,
                                     data_type="log2:FC",
                                     cluster_column_name="",
                                     exclude_full=True,
                                     transpose=True,
                                     metric="euclidean",
                                     verbose=False,):
    """
    Mean within-cluster pairwise distance per condition for each cluster.

    Lower values mean a cluster's sites follow more similar temporal profiles for
    that condition. Use `metric` to choose how distance is measured.

    Args:
        df: DataFrame following the project naming convention, with a cluster column.
        cell_lines: list of cell-line prefixes, e.g. ["WT"].
        conditions: list of condition substrings, e.g. ["_EGF_", "_INS_", "_EGFnINS_"].
        data_type: data-type string, e.g. "log2:FC" (default "log2:FC").
        cluster_column_name: name of the column holding cluster labels.
        exclude_full: if True, exclude the 'full' timepoint (default True).
        transpose: if True (default), reshape to (n_samples, n_timepoints, n_conditions).
        metric: "dtw" (default, time-warping) or "euclidean" (point-by-point;
                appropriate here since all sites share the same fixed timepoints,
                and much faster).
        verbose: if True, print reshape info.

    Returns:
        dict mapping cluster label → {condition_label: mean_distance_scalar}.
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
        cluster_metric[cluster] = _mean_distance_per_condition(X, cond_labels, metric=metric)

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


# =============================================================================
# Consensus / stability of a clustering
# =============================================================================
def consensus_stability(df,
                        cluster_fn,
                        n_runs=20,
                        bootstrap=False,
                        bootstrap_frac=0.8,
                        replace=False,
                        compute_coassociation=True,
                        max_sites_for_coassociation=6000,
                        reference_run=0,
                        reference_labels=None,
                        random_state=0,
                        verbose=True,):
    """
    Measure how reproducible a clustering is, and score each site's reliability.

    WHY THIS EXISTS
    ---------------
    Every clustering method here (KMeans, KShape, KernelKMeans, HDBSCAN, the
    adaptive divisive+agglomerative method) is stochastic: change the random seed
    (or perturb the input slightly) and some sites move between clusters. Before
    those cluster labels are used as ground truth to train a classifier and
    transfer it to the mutant datasets, we need to know WHICH assignments are
    trustworthy and which are coin-flips. Inertia / silhouette tell you how tight
    the clusters look in one run; they do NOT tell you whether you'd get the same
    clusters again. Stability does — and it is the single most useful reliability
    check for this project's workflow, because a label that flips between runs
    will only inject noise into the downstream classifier.

    WHAT IT DOES
    ------------
    Runs `cluster_fn` `n_runs` times with different random seeds (optionally on
    bootstrap resamples of the sites), then quantifies agreement two ways:

      * GLOBAL stability — the mean Adjusted Rand Index (ARI) between every pair
        of runs. ARI is 1.0 for identical partitions and ~0 for random ones, and
        is invariant to cluster-label permutation, so it is a clean "how
        reproducible is the whole clustering" number. This never needs the
        O(n^2) co-association matrix, so it always runs.

      * PER-SITE stability — from the co-association (consensus) matrix: the
        fraction of runs in which each pair of sites lands in the same cluster.
        Each site's stability is the average co-association with the other
        members of a reference cluster (Monti et al. "cluster consensus"). A site
        near 1.0 reliably travels with the same neighbours; a site near 0 is on a
        cluster boundary and its label should be treated with low confidence.
        This is O(n^2) in memory, so it is only computed when the number of sites
        is <= `max_sites_for_coassociation` (cluster on the *filtered* set).

    HOW TO USE IT (method-agnostic)
    -------------------------------
    Pass a small wrapper that runs your chosen method for a given seed and
    returns a 1-D label array aligned with the rows of `df`:

        def cluster_fn(d, seed):
            d2, model, X = tslearn_clustering_KMeans(
                d.copy(), data_type="log2:FC", condition_for_clustering=["_EGF_"],
                cell_lines=["WT"], number_of_clusters=15, df_dimensions=1,
                time_series_length=6, cluster_column_name="_tmp",
                random_state=seed, verbose=False, testing=True)
            return d2["_tmp"].values

        stab = consensus_stability(df_filtered, cluster_fn, n_runs=20)
        print(stab["mean_ari"])                       # global reproducibility
        df_filtered["stability"] = stab["per_site_stability"]

    The exact same call works for any method (adaptive, kernel, KShape, ...): only
    the wrapper changes. Compare methods by their `mean_ari` and per-site scores
    on identical data — this is the "same footing" comparison.

    Args:
        df: DataFrame to cluster (already filtered). Sites are its rows.
        cluster_fn: callable(sub_df, seed) -> 1-D array of integer labels with one
            entry per row of `sub_df`. Must be deterministic given the seed.
        n_runs: number of clustering repetitions (default 20).
        bootstrap: if True, each run clusters a random subset of the sites
            (assesses robustness to the sample, not just the seed). If False
            (default), every run uses all sites and only the seed varies.
        bootstrap_frac: fraction of sites sampled per run when bootstrap=True
            (default 0.8).
        replace: whether bootstrap sampling is with replacement (default False;
            note that with replacement, duplicate rows are collapsed here so each
            site appears at most once per run).
        compute_coassociation: if True (default), compute the per-site stability
            (needs the O(n^2) matrix; skipped with a message if n is too large).
        max_sites_for_coassociation: skip the co-association matrix above this many
            sites to avoid large memory use (default 6000 -> ~2 x 144 MB).
        reference_run: index of the run whose partition defines each site's
            "cluster" for the per-site score (default 0). Ignored if
            `reference_labels` is given.
        reference_labels: optional full-length array of labels (one per row of df)
            to use as the reference partition instead of a run — useful in
            bootstrap mode, where a single deterministic clustering covers all
            sites while individual runs do not.
        random_state: seed for the seed-generator and the bootstrap sampler
            (default 0) so the whole stability analysis is itself reproducible.
        verbose: if True, print progress and the headline numbers.

    Returns:
        dict with keys:
            "mean_ari": float — mean pairwise ARI across runs (global stability).
            "std_ari": float — standard deviation of the pairwise ARIs.
            "ari_pairwise": np.ndarray — all pairwise ARI values.
            "per_site_stability": pd.Series aligned to df.index, or None if the
                co-association matrix was skipped. Values in [0, 1] (NaN for sites
                absent from the reference partition).
            "mean_site_stability": float or None — mean of per_site_stability.
            "consensus_matrix": np.ndarray (n x n) of co-association fractions, or
                None if skipped. Mostly for plotting / consensus reclustering.
            "labels_per_run": np.ndarray (n_runs x n), with -1 marking sites absent
                from a run (only possible under bootstrap).
    """
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(random_state)
    n = len(df)
    idx_all = np.arange(n)

    # -1 marks "this site was not part of this run" (only happens with bootstrap)
    full_labels = np.full((n_runs, n), -1, dtype=int)

    for r in range(n_runs):
        seed = int(rng.integers(0, 2 ** 31 - 1))
        if bootstrap:
            m = int(round(bootstrap_frac * n))
            sample_idx = np.unique(rng.choice(idx_all, size=m, replace=replace))
            sub = df.iloc[sample_idx]
            labs = np.asarray(cluster_fn(sub, seed))
            full_labels[r, sample_idx] = labs
        else:
            labs = np.asarray(cluster_fn(df, seed))
            full_labels[r, :] = labs
        if verbose:
            print(f"  run {r + 1}/{n_runs} done (seed={seed})")

    # --- Global stability: mean pairwise ARI over the shared sites of each pair ---
    ari_values = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            mask = (full_labels[i] >= 0) & (full_labels[j] >= 0)
            if mask.sum() >= 2:
                ari_values.append(adjusted_rand_score(full_labels[i][mask],
                                                      full_labels[j][mask]))
    ari_values = np.asarray(ari_values, dtype=float)
    mean_ari = float(np.mean(ari_values)) if len(ari_values) else float("nan")
    std_ari = float(np.std(ari_values)) if len(ari_values) else float("nan")

    # --- Per-site stability via the co-association (consensus) matrix ---
    per_site = None
    mean_site_stability = None
    consensus_matrix = None

    if compute_coassociation and n <= max_sites_for_coassociation:
        co = np.zeros((n, n), dtype=np.int32)    # times pair co-clustered
        pair = np.zeros((n, n), dtype=np.int32)  # times pair both present
        for r in range(n_runs):
            present = full_labels[r] >= 0
            pmask = present[:, None] & present[None, :]
            pair += pmask
            same = (full_labels[r][:, None] == full_labels[r][None, :]) & pmask
            co += same
        with np.errstate(invalid="ignore", divide="ignore"):
            consensus_matrix = np.where(pair > 0, co / pair, np.nan)

        # Reference partition: an explicit labelling, else a chosen run.
        if reference_labels is not None:
            ref = np.asarray(reference_labels)
        else:
            ref = full_labels[reference_run]

        per_site_arr = np.full(n, np.nan)
        for i in range(n):
            if ref[i] < 0:
                continue
            mates = np.where(ref == ref[i])[0]
            mates = mates[mates != i]
            if len(mates) == 0:
                continue
            per_site_arr[i] = np.nanmean(consensus_matrix[i, mates])
        per_site = pd.Series(per_site_arr, index=df.index)
        mean_site_stability = float(np.nanmean(per_site_arr))
    elif compute_coassociation and verbose:
        print(f"  Skipping co-association matrix: {n} sites > "
              f"max_sites_for_coassociation={max_sites_for_coassociation}. "
              f"Global ARI still computed.")

    if verbose:
        print(f"\nMean pairwise ARI: {mean_ari:.3f} +/- {std_ari:.3f}")
        if mean_site_stability is not None:
            print(f"Mean per-site stability: {mean_site_stability:.3f}")

    return {"mean_ari": mean_ari,
            "std_ari": std_ari,
            "ari_pairwise": ari_values,
            "per_site_stability": per_site,
            "mean_site_stability": mean_site_stability,
            "consensus_matrix": consensus_matrix,
            "labels_per_run": full_labels,}

