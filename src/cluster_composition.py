"""
Cluster composition: cross-tabulate cluster membership against per-site annotations.

Once phosphosites carry both a clustering label (one column per clustering method, e.g.
"KMeans_adaptive_cluster_WT_EGF_log2_FC") and per-site annotations (predicted kinases from
src.kinase_prediction, protein identity, ...), these helpers answer:

  * Which clusters contain sites predicted to be phosphorylated by kinases X, Y, Z?
      kinase_cluster_table()  -> counts, plot_cluster_composition() -> figure
  * In which clusters do proteins X, Y, Z appear?
      protein_cluster_table() -> counts, plot_cluster_composition() -> figure

Every function takes the clustering column name explicitly, so the same call works for any
clustering method present in the dataframe.

Both *_table() functions return a DataFrame indexed by cluster (columns = the queried
kinases / proteins), so a single generic plotter, plot_cluster_composition(), renders either.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def kinase_cluster_table(df,
                         cluster_col,
                         kinases,
                         kinase_prefix="predicted_kinase",
                         top_n=5,
                         ranks=None,
                         min_percentile=None,
                         normalize=False,
                         groups=None,):
    """
    Count, per cluster, how many phosphosites each given kinase is predicted to phosphorylate.

    A site counts for kinase K if K appears among its predicted-kinase columns (ranks 1..top_n
    by default). Restrict to specific ranks with `ranks` (e.g. [1] for the top prediction only)
    and/or require a minimum percentile with `min_percentile`.

    Paralogs can be merged manually: list the merged name in `kinases` and map it in `groups`,
    e.g. kinases=["ERK1/2", "AKT1"], groups={"ERK1/2": ["ERK1", "ERK2"]}. A site then counts
    for "ERK1/2" if ANY member is predicted (logical OR — a site with both is counted once,
    never double-counted).

    Args:
        df: dataframe with cluster labels and predicted-kinase columns.
        cluster_col: name of the clustering column to group by (e.g.
            "KMeans_adaptive_cluster_WT_EGF_log2_FC").
        kinases: kinase/group name or list of names to look up (matched case-insensitively);
            a name present in `groups` is expanded to its members.
        kinase_prefix: prefix of the predicted-kinase columns (default "predicted_kinase",
            i.e. columns "{prefix}_{rank}" and "{prefix}_{rank}_prob").
        top_n: highest rank to scan when `ranks` is None (default 5).
        ranks: explicit list of ranks to scan (overrides top_n), e.g. [1, 2, 3].
        min_percentile: if given, only count a hit when the matching rank's *_prob >= this.
        normalize: if True, return each count as a fraction of the cluster's site total.
        groups: optional manual paralog merge, e.g. {"ERK1/2": ["ERK1", "ERK2"]}.

    Returns:
        DataFrame indexed by cluster (sorted), one column per kinase/group, values = site
        counts (or per-cluster fractions if normalize=True).
    """
    if isinstance(kinases, str):
        kinases = [kinases]
    groups = groups or {}
    scan_ranks = ranks if ranks is not None else range(1, top_n + 1)

    valid = df[df[cluster_col].notna()].copy()
    clusters = np.sort(valid[cluster_col].unique())

    table = {}
    for kinase in kinases:
        members = groups.get(kinase, [kinase])          # expand a merged group to its members
        targets = {m.upper() for m in members}
        hit = pd.Series(False, index=valid.index)
        for rank in scan_ranks:
            name_col = f"{kinase_prefix}_{rank}"
            if name_col not in valid.columns:
                continue
            rank_hit = valid[name_col].astype("string").str.upper().isin(targets)
            if min_percentile is not None:
                prob_col = f"{kinase_prefix}_{rank}_prob"
                rank_hit &= pd.to_numeric(valid[prob_col], errors="coerce") >= min_percentile
            hit |= rank_hit.fillna(False)
        table[kinase] = valid.loc[hit, cluster_col].value_counts()

    result = pd.DataFrame(table).reindex(clusters).fillna(0).astype(int)
    result.index.name = cluster_col
    if normalize:
        sizes = valid[cluster_col].value_counts().reindex(clusters)
        result = result.div(sizes, axis=0).fillna(0.0)
    return result


def protein_cluster_table(df,
                          cluster_col,
                          proteins,
                          protein_col="protein_name",
                          normalize=False,):
    """
    Count, per cluster, how many phosphosites belong to each given protein.

    Args:
        df: dataframe with cluster labels and a protein-identity column.
        cluster_col: name of the clustering column to group by.
        proteins: protein name/ID or list thereof (matched case-insensitively).
        protein_col: column holding the protein identity (default "protein_name"; use
            "protein_Id" to query by UniProt accession).
        normalize: if True, return each count as a fraction of the cluster's site total.

    Returns:
        DataFrame indexed by cluster (sorted), one column per protein, values = site counts
        (or per-cluster fractions if normalize=True).
    """
    if isinstance(proteins, str):
        proteins = [proteins]

    valid = df[df[cluster_col].notna()].copy()
    clusters = np.sort(valid[cluster_col].unique())
    protein_upper = valid[protein_col].astype("string").str.upper()

    table = {}
    for protein in proteins:
        hit = protein_upper == protein.upper()
        table[protein] = valid.loc[hit.fillna(False), cluster_col].value_counts()

    result = pd.DataFrame(table).reindex(clusters).fillna(0).astype(int)
    result.index.name = cluster_col
    if normalize:
        sizes = valid[cluster_col].value_counts().reindex(clusters)
        result = result.div(sizes, axis=0).fillna(0.0)
    return result


def plot_cluster_composition(table,
                             kind="bar",
                             title=None,
                             ylabel=None,
                             figsize=None,
                             cmap="viridis",
                             annotate=True,
                             ax=None,):
    """
    Plot a cluster-composition table (from kinase_cluster_table / protein_cluster_table).

    Args:
        table: DataFrame indexed by cluster, columns = queried kinases/proteins.
        kind: "bar" for grouped bars (clusters on the x-axis, one series per column) or
            "heatmap" for an entities x clusters image (better for many entities).
        title: plot title.
        ylabel: y-axis label (defaults to "fraction of cluster" or "number of sites").
        figsize: figure size; auto-sized if None.
        cmap: colormap for the heatmap.
        annotate: if True, write the value inside each heatmap cell.
        ax: existing Axes to draw on; a new figure is created if None.

    Returns:
        (fig, ax).
    """
    is_fraction = np.issubdtype(table.to_numpy().dtype, np.floating)
    if ylabel is None:
        ylabel = "fraction of cluster" if is_fraction else "number of sites"

    if kind == "bar":
        if figsize is None:
            figsize = (max(6, 1.1 * len(table.index)), 4.5)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize,)
        else:
            fig = ax.figure
        table.plot(kind="bar", ax=ax,)
        ax.set_xlabel(table.index.name or "cluster")
        ax.set_ylabel(ylabel)
        ax.legend(title=None, fontsize=8,)
        ax.grid(axis="y", alpha=0.3,)

    elif kind == "heatmap":
        if figsize is None:
            figsize = (max(6, 0.8 * len(table.index)), max(3, 0.5 * len(table.columns)))
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize,)
        else:
            fig = ax.figure
        data = table.T  # entities on rows, clusters on columns
        im = ax.imshow(data.to_numpy(), aspect="auto", cmap=cmap,)
        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels(data.columns, rotation=90, fontsize=8,)
        ax.set_yticks(range(len(data.index)))
        ax.set_yticklabels(data.index, fontsize=8,)
        ax.set_xlabel(table.index.name or "cluster")
        fig.colorbar(im, ax=ax, label=ylabel,)
        if annotate:
            fmt = "{:.2f}" if is_fraction else "{:.0f}"
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    ax.text(j, i, fmt.format(data.iat[i, j]),
                            ha="center", va="center", fontsize=7,
                            color="white",)
    else:
        raise ValueError(f"Unknown kind {kind!r} (expected 'bar' or 'heatmap').")

    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax
