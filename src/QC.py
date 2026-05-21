"""
QC functions for phosphoproteomics datasets.

Sections follow the priority order in notebooks/02_qc/General_QC.ipynb:
  1. Raw data / peptide detection quality
  2. Quantification quality (sample-level)
  3. Normalization QC
  4. Comparing datasets (Venn diagrams, overlap statistics)

All functions accept DataFrames that follow the project naming convention:
    {CellLine}_{DataType}:{subtype}_{Condition}_{Timepoint}_{Replicate}
e.g.  WT_raw:abs_EGF_full_r1,  WT_log2:FC_EGF_2,  WT_raw:cv_EGF_10

Column selection is handled by ColumnSpec from src/column_spec.py.
"""

import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib_venn import venn2, venn3
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.column_spec import ColumnSpec

# Replicate-suffix pattern (e.g. _r1, _r2)
_REP_RE = re.compile(r"_r\d+$")


def _is_replicate_col(col: str) -> bool:
    return bool(_REP_RE.search(col))


##############################################################################
# Section 1: Raw Data / Peptide Detection Quality
##############################################################################

def missing_values_per_sample(df,
                              cell_lines=None,
                              conditions=None,
                              data_type="raw:abs",
                              missing_threshold=0.30,
                              figsize=None,
                              title="Missing values per sample",
                              ax=None,
                              ):
    """
    Bar chart of percentage missing (zero or NaN) values per replicate sample.

    A dashed red line marks `missing_threshold` (fraction, default 0.30 = 30 %).
    Samples above the threshold may need to be flagged or dropped.

    Args:
        df: DataFrame following the project naming convention.
        cell_lines: list of cell-line prefixes, e.g. ["WT"].
        conditions: list of condition substrings, e.g. ["_EGF_", "_INS_"].
        data_type: data-type string to select per-replicate columns, default "raw:abs".
        missing_threshold: fraction (0–1) drawn as a warning line.
        figsize: (width, height); auto-sized if None.
        title: figure title.
        ax: optional existing Axes; a new figure is created if None.

    Returns:
        fig, ax
    """
    if cell_lines is None:
        cell_lines = []
    if conditions is None:
        conditions = []

    cols = ColumnSpec.select(df, cell_lines=cell_lines, data_type=data_type, conditions=conditions)
    cols = [c for c in cols if _is_replicate_col(c)] # Double checking the columns are from replicates
    if not cols:
        raise ValueError(
            f"No replicate columns found for cell_lines={cell_lines}, conditions={conditions}, data_type={data_type!r}"
        )

    pct_missing = (df[cols].apply(lambda s: (s.isna() | (s == 0)).sum()) / len(df) * 100) # Percentage missing values per column

    if figsize is None:
        figsize = (max(8, len(cols) * 0.5), 5)
    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.bar(range(len(cols)), pct_missing.values, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axhline(missing_threshold * 100, color="crimson", linestyle="--", linewidth=1.2, label=f"Threshold {missing_threshold * 100:.0f}%")
    ax.legend(fontsize=8)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([re.sub(r"_(?:raw|log2):[a-z]+_", "\n", c) for c in cols], rotation=90, fontsize=7 )
    ax.set_ylabel("% missing // zero values")
    ax.set_xlabel("Sample")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    if create_fig:
        fig.tight_layout()
    return fig, ax


def peptide_count_per_sample(df,
                             cell_lines=None,
                             conditions=None,
                             data_type="raw:abs",
                             missing_value=0.0,
                             figsize=None,
                             title="Detected peptides per sample",
                             ax=None,
                             ):
    """
    Bar chart of detected (non-missing) peptides per replicate sample.

    Large drops in one replicate compared to the others suggest a technical
    problem (bad injection, failed enrichment, etc.).

    Args:
        df: DataFrame following the project naming convention.
        cell_lines: list of cell-line prefixes.
        conditions: list of condition substrings.
        data_type: default "raw:abs".
        missing_value: value treated as undetected (default 0.0).
        figsize: (width, height).
        title: figure title.
        ax: optional Axes.

    Returns:
        fig, ax
    """
    if cell_lines is None:
        cell_lines = []
    if conditions is None:
        conditions = []

    cols = ColumnSpec.select(df, cell_lines=cell_lines, data_type=data_type, conditions=conditions)
    cols = [c for c in cols if _is_replicate_col(c)]
    if not cols:
        raise ValueError("No replicate columns found.")

    counts = (df[cols] != missing_value).sum() # Assuming 0.0 as missing value, not sure if it recognices NaN as missing value

    if figsize is None:
        figsize = (max(8, len(cols) * 0.5), 5)
    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.bar(range(len(cols)), counts.values, color="teal", edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([re.sub(r"_(?:raw|log2):[a-z]+_", "\n", c) for c in cols], rotation=90, fontsize=7 )
    ax.set_ylabel("Number of detected peptides")
    ax.set_xlabel("Sample")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    if create_fig:
        fig.tight_layout()
    return fig, ax


def nreps_distribution(df,
                       nreps_col="n:reps",
                       figsize=(6, 4),
                       title="Replicate count distribution",
                       ax=None,
                       ):
    """
    Bar chart of the n:reps column — how many peptides were detected in 1, 2, 3… replicates.

    Peptides detected in only 1 replicate carry much less confidence.

    Args:
        df: DataFrame with an 'n:reps' column (or similar).
        nreps_col: name of the replicate-count column, default 'n:reps'.
        figsize: (width, height).
        title: figure title.
        ax: optional Axes.

    Returns:
        fig, ax
    """
    if nreps_col not in df.columns:
        raise ValueError(f"Column '{nreps_col}' not found in DataFrame.")

    counts = df[nreps_col].value_counts().sort_index()

    n_total = counts.sum()
    print(f"Replicate detection summary ({n_total} sites):")
    for n_rep, n_sites in counts.items():
        print(f"  {n_rep} replicate{'s' if n_rep > 1 else ' '}: {n_sites / n_total * 100:.1f}%")

    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.bar(counts.index.astype(str), counts.values, color="mediumpurple", edgecolor="white")
    ax.set_xlabel("Number of replicates detected")
    ax.set_ylabel("Number of peptides")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    if create_fig:
        fig.tight_layout()
    return fig, ax


def impute_missing_replicates(
    df,
    cell_lines=None,
    conditions=None,
    data_type="raw:abs",
    method="mean",
    inplace=False,
):
    """
    Impute missing replicate values using the mean (or median) of the detected
    replicates for the same site at the same timepoint.

    For each site (row) and each timepoint group (all replicates of one
    CellLine × Condition × Timepoint combination), any zero or NaN value is
    replaced by the mean/median of the non-zero, non-NaN values in that same
    row within the group. Sites where every replicate in the group is missing
    remain NaN and should be removed before PCA with .dropna().

    This reproduces the imputation strategy from notebooks/02_qc/PCA.ipynb
    adapted to the project naming convention.

    Args:
        df: DataFrame following the project naming convention.
        cell_lines: list of cell-line prefixes, e.g. ["WT"].
        conditions: list of condition substrings, e.g. ["_EGF_", "_INS_"].
        data_type: data-type string for per-replicate columns, default "raw:abs".
        method: "mean" (default) or "median" — statistic used to fill missing values.
        inplace: if True modify df in place; if False (default) return a copy.

    Returns:
        DataFrame with imputed replicate values.
    """
    if cell_lines is None:
        cell_lines = []
    if conditions is None:
        conditions = []

    cols = ColumnSpec.select(df, cell_lines=cell_lines, data_type=data_type, conditions=conditions)
    cols = [c for c in cols if _is_replicate_col(c)]
    if not cols:
        raise ValueError(
            f"No replicate columns found for cell_lines={cell_lines}, "
            f"conditions={conditions}, data_type={data_type!r}"
        )

    # Group replicate columns by their base name (strip _r\d+ suffix).
    # All columns sharing a base belong to the same timepoint group.
    groups = defaultdict(list)
    for col in cols:
        base = _REP_RE.sub("", col)   # e.g. WT_raw:abs_EGF_full
        groups[base].append(col)

    result = df if inplace else df.copy()

    for group_cols in groups.values():
        # Replace 0 with NaN so pandas statistics ignore them
        sub = result[group_cols].replace(0.0, np.nan)

        if method == "mean":
            fill_vals = sub.mean(axis=1)   # row-wise mean of detected replicates
        elif method == "median":
            fill_vals = sub.median(axis=1)
        else:
            raise ValueError(f"method must be 'mean' or 'median', got {method!r}")

        # Fill only the missing (0 or NaN) positions
        for col in group_cols:
            missing = result[col].isna() | (result[col] == 0.0)
            result.loc[missing, col] = fill_vals[missing]

    return result


def replicate_detection_map(
    df,
    cell_lines=None,
    conditions=None,
    data_type="raw:abs",
    missing_value=0.0,
    figsize=None,
    title="Replicate detection map",
    ax=None,
):
    """
    Binary heatmap showing which replicates each phosphosite was detected in.

    Rows are sites, columns are replicate samples. Green = detected, red = missing.
    Sites are sorted by their binary detection pattern so rows with the same
    combination of detected/missing replicates form visible horizontal bands.
    Row labels are omitted because individual rows are very thin.

    Args:
        df: DataFrame following the project naming convention.
        cell_lines: list of cell-line prefixes, e.g. ["WT"].
        conditions: list of condition substrings, e.g. ["_EGF_"].
        data_type: data-type string for per-replicate columns, default "raw:abs".
        missing_value: value treated as undetected (default 0.0); NaN is always missing.
        figsize: (width, height); auto-sized if None.
        title: figure title.
        ax: optional Axes.

    Returns:
        fig, ax
    """
    if cell_lines is None:
        cell_lines = []
    if conditions is None:
        conditions = []

    cols = ColumnSpec.select(df, cell_lines=cell_lines, data_type=data_type, conditions=conditions)
    cols = [c for c in cols if _is_replicate_col(c)]
    if not cols:
        raise ValueError(
            f"No replicate columns found for cell_lines={cell_lines}, "
            f"conditions={conditions}, data_type={data_type!r}"
        )

    # Binary presence matrix: 1 = detected, 0 = missing
    presence = df[cols].copy()
    presence = (presence.notna() & (presence != missing_value)).astype(int)

    # Sort rows by detection pattern (treat each row's binary vector as a key)
    # Convert each row to a tuple for stable sorting: patterns with more detections first,
    # then lexicographically so similar patterns cluster together
    pattern_keys = presence.apply(lambda row: tuple(row), axis=1)
    sorted_idx = pattern_keys.sort_values(ascending=False).index
    presence_sorted = presence.loc[sorted_idx]

    n_sites, n_cols = presence_sorted.shape
    if figsize is None:
        figsize = (max(4, n_cols * 0.5), min(10, max(4, n_sites * 0.004)))

    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Draw as image: map 0 → red, 1 → green
    cmap = plt.matplotlib.colors.ListedColormap(["#d62728", "#2ca02c"])  # red, green
    ax.imshow(presence_sorted.values, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")

    # Column labels (replicate names, shortened)
    short_cols = [re.sub(r"_(?:raw|log2):[a-z]+_", "_", c) for c in cols]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(short_cols, rotation=45, ha="right", fontsize=8)

    ax.set_yticks([])
    ax.set_ylabel(f"{n_sites} sites")
    ax.set_title(title)

    # Colour legend
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(color="#2ca02c", label="Detected"), Patch(color="#d62728", label="Missing")],
        loc="upper right", fontsize=8, framealpha=0.8,
    )

    if create_fig:
        fig.tight_layout()
    return fig, ax


##############################################################################
# Section 2: Quantification Quality (Sample-level)
##############################################################################

def intensity_distribution(df,
                           cell_lines=None,
                           conditions=None,
                           data_type="raw:abs",
                           log_transform=False,
                           bins=80,
                           figsize=(8, 5),
                           title="Intensity distribution per sample",
                           ax=None,
                           ):
    """
    Overlaid step histograms of intensity values — one outline per replicate sample.

    Each sample is drawn as a step (outline-only) histogram so all distributions
    are visible simultaneously. Samples with similar medians and shapes are
    well-normalised; outlier replicates stand out immediately.

    Args:
        df: DataFrame following the project naming convention.
        cell_lines: list of cell-line prefixes.
        conditions: list of condition substrings.
        data_type: data-type string, e.g. "raw:abs" or "log2:abs".
        log_transform: if True apply log2(x+1) before plotting (useful for raw:abs).
        bins: number of histogram bins (default 80).
        figsize: (width, height).
        title: figure title.
        ax: optional Axes.

    Returns:
        fig, ax
    """
    if cell_lines is None:
        cell_lines = []
    if conditions is None:
        conditions = []

    cols = ColumnSpec.select(df, cell_lines=cell_lines, data_type=data_type, conditions=conditions)
    cols = [c for c in cols if _is_replicate_col(c)]
    if not cols:
        raise ValueError(f"No replicate columns found for data_type={data_type!r}")

    values = df[cols].copy().replace(0, np.nan)
    if log_transform:
        values = np.log2(values + 1)

    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    palette = plt.cm.tab20.colors
    short_labels = [re.sub(r"_(?:raw|log2):[a-z]+_", "_", c) for c in cols]

    for i, (col, label) in enumerate(zip(cols, short_labels)):
        data = values[col].dropna().values
        ax.hist(data, bins=bins, histtype="step", linewidth=1.2,
                color=palette[i % len(palette)], label=label, density=True)

    ax.set_xlabel("log2 intensity" if (log_transform or "log2" in data_type) else "Intensity")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=6, ncol=max(1, len(cols) // 20), loc="upper right")
    ax.grid(axis="both", alpha=0.3)

    if create_fig:
        fig.tight_layout()
    return fig, ax


def cv_distribution(
    df,
    cell_lines=None,
    conditions=None,
    data_type="raw:cv",
    cv_threshold=30.0,
    figsize=None,
    title="CV distribution per timepoint",
    ax=None,
):
    """
    Violin + inner box plots of CV (%) across all matching timepoint columns.

    CVs >20–30% in many peptides are a warning sign for poor replicate agreement.
    A dashed red line marks the `cv_threshold`.

    Args:
        df: DataFrame with CV columns (data_type e.g. "raw:cv").
        cell_lines: list of cell-line prefixes.
        conditions: list of condition substrings.
        data_type: data-type string for CV columns, default "raw:cv".
        cv_threshold: CV % warning level drawn as a dashed line.
        figsize: (width, height).
        title: figure title.
        ax: optional Axes.

    Returns:
        fig, ax
    """
    if cell_lines is None:
        cell_lines = []
    if conditions is None:
        conditions = []

    cols = ColumnSpec.select(df, cell_lines=cell_lines, data_type=data_type, conditions=conditions)
    if not cols:
        raise ValueError(f"No CV columns found for data_type={data_type!r}")

    # Shift from a wide format to a long format so seaborn violinplot can work
    melted = (df[cols].replace(0, np.nan).melt(var_name="column", value_name="CV").dropna())

    # Extract condition_timepoint as label that will be used to group CV and see their distribution
    melted["label"] = melted["column"].apply(lambda c: "_".join(c.split("_")[2:]) ) # everything after CellLine_datatype

    if figsize is None:
        figsize = (max(8, len(cols) * 0.55), 5)
    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    order = list(dict.fromkeys(melted["label"]))  # preserve column order, deduplicate
    palette = sns.color_palette("Set2", len(order))
    sns.violinplot(data=melted, x="label", y="CV", hue="label", ax=ax, palette=palette, order=order, inner="box", cut=0, legend=False)
    ax.axhline(cv_threshold, color="crimson", linestyle="--", linewidth=1.2, label=f"CV threshold {cv_threshold:.0f}%")
    ax.legend(fontsize=8)
    ax.set_xlabel("Condition / timepoint")
    ax.set_ylabel("CV (%)")
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    if create_fig:
        fig.tight_layout()
    return fig, ax


def sample_correlation_heatmap( # I dont see the use of this function at the moment
    df,
    cell_lines=None,
    conditions=None,
    data_type="raw:abs",
    method="pearson",
    figsize=None,
    title="Sample correlation heatmap",
    cmap="viridis",
    ax=None,
):
    """
    Pearson (or Spearman) correlation heatmap across replicate samples.

    Replicates from the same condition/timepoint should cluster together.
    If a replicate clusters with a different condition, something went wrong.

    Args:
        df: DataFrame following the project naming convention.
        cell_lines: list of cell-line prefixes.
        conditions: list of condition substrings.
        data_type: default "raw:abs".
        method: "pearson" or "spearman".
        figsize: (width, height).
        title: figure title.
        cmap: colormap.
        ax: optional Axes.

    Returns:
        fig, ax
    """
    if cell_lines is None:
        cell_lines = []
    if conditions is None:
        conditions = []

    cols = ColumnSpec.select(df, cell_lines=cell_lines, data_type=data_type,
                             conditions=conditions)
    cols = [c for c in cols if _is_replicate_col(c)]
    if not cols:
        raise ValueError(f"No replicate columns found for data_type={data_type!r}")

    mat = df[cols].replace(0, np.nan)
    corr = mat.corr(method=method)
    short_labels = [re.sub(r"_(?:raw|log2):[a-z]+_", "\n", c) for c in cols]

    if figsize is None:
        n = len(cols)
        figsize = (max(7, n * 0.5), max(6, n * 0.5))
    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    sns.heatmap(
        corr, ax=ax, cmap=cmap, vmin=0, vmax=1,
        xticklabels=short_labels, yticklabels=short_labels,
        annot=(len(cols) <= 20), fmt=".2f", annot_kws={"size": 6},
        linewidths=0.3,
    )
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)

    if create_fig:
        fig.tight_layout()
    return fig, ax


def pca_plot(df,
             cell_lines=None,
             conditions=None,
             data_type="log2:abs",
             color_by="condition",
             figsize=(7, 5),
             title="PCA of replicate samples",
             ax=None,
             ):
    """
    PCA scatter plot of replicate samples (samples are observations; peptides are features).

    PC1/PC2 should separate conditions or timepoints in a biologically meaningful way.
    Replicates from the same condition should group tightly; outlier replicates
    become immediately visible.

    Args:
        df: DataFrame following the project naming convention.
        cell_lines: list of cell-line prefixes.
        conditions: list of condition substrings.
        data_type: data-type string for replicate columns, default "log2:abs".
        color_by: "condition" — colour by condition substring;
                  "cell_line" — colour by cell line prefix.
        figsize: (width, height).
        title: figure title.
        ax: optional Axes.

    Returns:
        fig, ax, pca_df  (DataFrame with PC1, PC2, sample, group columns)
    """
    if cell_lines is None:
        cell_lines = []
    if conditions is None:
        conditions = []

    cols = ColumnSpec.select(df, cell_lines=cell_lines, data_type=data_type, conditions=conditions)
    cols = [c for c in cols if _is_replicate_col(c)]
    if not cols:
        raise ValueError(f"No replicate columns found for data_type={data_type!r}")

    # rows = peptides, columns = samples  →  transpose for PCA (samples × peptides)
    mat = df[cols].replace(0, np.nan).fillna(df[cols].replace(0, np.nan).median())
    X = mat.T.values

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "sample": cols})

    if color_by == "condition":
        def _group(col):
            for cond in conditions:
                if cond in col:
                    return cond.strip("_")
            return "other"
    else:
        def _group(col):
            for cell in cell_lines:
                if col.startswith(cell):
                    return cell
            return "other"

    pca_df["group"] = pca_df["sample"].apply(_group)

    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    palette = sns.color_palette("tab10", pca_df["group"].nunique())
    for idx, (grp, sub) in enumerate(pca_df.groupby("group")):
        ax.scatter(sub["PC1"], sub["PC2"], label=grp, color=palette[idx],
                   s=65, edgecolors="black", linewidths=0.5)
        for _, row in sub.iterrows():
            short = re.sub(r"_(?:raw|log2):[a-z]+_", "_", row["sample"])
            ax.annotate(short, (row["PC1"], row["PC2"]), fontsize=6, alpha=0.75,
                        xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.axhline(0, color="grey", linewidth=0.5, zorder=0)
    ax.axvline(0, color="grey", linewidth=0.5, zorder=0)

    if create_fig:
        fig.tight_layout()
    return fig, ax, pca_df


def pca_plot_interactive(
    df,
    cell_lines=None,
    conditions=None,
    data_type="raw:abs",
    n_components=10,
    impute=True,
    impute_method="mean",
    color_by="condition",
    figsize=(1000, 800),
    title="PCA of replicate samples",
):
    """
    Interactive Plotly PCA scatter plot of replicate samples.

    Samples (replicates) are the observations; phosphosites are the features.
    Points are coloured by condition group and shaped by stimulation type
    (EGF = diamond, INS = square, EGFnINS = cross, controls = circle).

    A separate bar chart of variance explained per PC is also returned.

    Based on the PCA analysis in notebooks/02_qc/PCA.ipynb, adapted to the
    project naming convention.

    Args:
        df: DataFrame following the project naming convention.
        cell_lines: list of cell-line prefixes, e.g. ["WT"].
        conditions: list of condition substrings, e.g. ["_EGF_", "_INS_", "_EGFnINS_"].
        data_type: data-type string for replicate columns, default "raw:abs".
        n_components: number of PCs to compute (default 10); only PC1/PC2 are plotted.
        impute: if True run impute_missing_replicates() before PCA (default True).
        impute_method: "mean" or "median", passed to impute_missing_replicates().
        color_by: "condition" (default) — colour by condition×timepoint group;
                  "cell_line" — colour by cell-line prefix.
        figsize: (width, height) in pixels for the Plotly figure.
        title: figure title.

    Returns:
        fig_scatter: Plotly Figure — PC1 vs PC2 scatter.
        fig_variance: Plotly Figure — variance explained bar chart.
        pca_df: DataFrame with PC coordinates and sample metadata.
    """
    if cell_lines is None:
        cell_lines = []
    if conditions is None:
        conditions = []

    # --- Column selection ---
    cols = ColumnSpec.select(df, cell_lines=cell_lines, data_type=data_type,
                             conditions=conditions)
    cols = [c for c in cols if _is_replicate_col(c)]
    if not cols:
        raise ValueError(f"No replicate columns found for data_type={data_type!r}")

    # --- Optional imputation ---
    if impute:
        df = impute_missing_replicates(df, cell_lines=cell_lines, conditions=conditions,
                                       data_type=data_type, method=impute_method)

    # --- Build (samples × sites) matrix; drop sites still fully missing ---
    mat = df[cols].replace(0, np.nan).T      # rows = samples, columns = sites
    mat = mat.dropna(axis=1)                 # drop sites missing in any sample

    # --- StandardScaler + PCA ---
    X_scaled = StandardScaler().fit_transform(mat.values)
    n_comp = min(n_components, mat.shape[0], mat.shape[1])
    pca = PCA(n_components=n_comp, random_state=0)
    coords = pca.fit_transform(X_scaled)

    # --- Build metadata DataFrame ---
    pca_df = pd.DataFrame(
        coords,
        columns=[f"PC{i + 1}" for i in range(n_comp)],
    )
    pca_df["sample"] = mat.index.tolist()

    # Derive condition label: strip replicate suffix, drop CellLine + DataType prefix
    def _cond_label(col):
        base = _REP_RE.sub("", col)           # e.g. WT_raw:abs_EGF_full
        return "_".join(base.split("_")[2:])  # e.g. EGF_full

    # Derive symbol from condition label
    def _symbol(label):
        if "EGFnINS" in label:
            return "cross"
        elif "EGF" in label:
            return "diamond"
        elif "INS" in label:
            return "square"
        return "circle"

    pca_df["condition"] = pca_df["sample"].apply(_cond_label)

    if color_by == "cell_line":
        def _cell_label(col):
            for cell in cell_lines:
                if col.startswith(cell):
                    return cell
            return "unknown"
        pca_df["color_group"] = pca_df["sample"].apply(_cell_label)
    else:
        pca_df["color_group"] = pca_df["condition"]

    pca_df["symbol"] = pca_df["condition"].apply(_symbol)

    # --- Scatter: PC1 vs PC2 ---
    var_pct = pca.explained_variance_ratio_ * 100
    fig_scatter = px.scatter(
        pca_df,
        x="PC1", y="PC2",
        color="color_group",
        symbol="symbol",
        hover_name="sample",
        hover_data={"condition": True, "PC1": ":.2f", "PC2": ":.2f",
                    "color_group": False, "symbol": False},
        title=title,
        labels={
            "PC1": f"PC1 ({var_pct[0]:.1f}% var)",
            "PC2": f"PC2 ({var_pct[1]:.1f}% var)",
            "color_group": color_by,
        },
        width=figsize[0], height=figsize[1],
    )
    fig_scatter.update_traces(marker=dict(size=12, line=dict(width=1.2, color="DarkSlateGrey")))
    fig_scatter.update_layout(plot_bgcolor="white", paper_bgcolor="white")

    # --- Variance explained bar chart ---
    var_df = pd.DataFrame({
        "Principal Component": [f"PC{i + 1}" for i in range(n_comp)],
        "Variance Explained (%)": var_pct,
    })
    fig_variance = px.bar(
        var_df,
        x="Principal Component",
        y="Variance Explained (%)",
        title="Variance explained per PC",
        text=var_df["Variance Explained (%)"].round(1).astype(str) + "%",
        color="Variance Explained (%)",
        color_continuous_scale="Blues",
        width=800, height=450,
    )
    fig_variance.update_traces(textposition="outside")
    fig_variance.update_layout(coloraxis_showscale=False)

    return fig_scatter, fig_variance, pca_df


def umap_plot_interactive(
    df,
    cell_lines=None,
    conditions=None,
    data_type="raw:abs",
    impute=True,
    impute_method="mean",
    n_neighbors=15,
    min_dist=0.1,
    random_state=42,
    color_by="condition",
    figsize=(1000, 800),
    title="UMAP of replicate samples",
):
    """
    Interactive Plotly UMAP scatter plot of replicate samples.

    Samples (replicates) are the observations; phosphosites are the features.
    This is the UMAP equivalent of pca_plot_interactive(): the same matrix is
    embedded with UMAP instead of PCA.

    Points are coloured by condition group and shaped by stimulation type
    (EGF = diamond, INS = square, EGFnINS = cross, controls = circle).

    Args:
        df: DataFrame following the project naming convention.
        cell_lines: list of cell-line prefixes, e.g. ["WT"].
        conditions: list of condition substrings, e.g. ["_EGF_", "_INS_", "_EGFnINS_"].
        data_type: data-type string for replicate columns (default "raw:abs").
        impute: if True run impute_missing_replicates() before UMAP (default True).
        impute_method: "mean" or "median", passed to impute_missing_replicates().
        n_neighbors: UMAP n_neighbors parameter — controls local vs global structure
                     (default 15; lower = more local detail).
        min_dist: UMAP min_dist parameter — controls point packing in the embedding
                  (default 0.1; lower = tighter clusters).
        random_state: random seed for reproducibility (default 42).
        color_by: "condition" (default) — colour by condition×timepoint group;
                  "cell_line" — colour by cell-line prefix.
        figsize: (width, height) in pixels for the Plotly figure (default (1000, 800)).
        title: figure title.

    Returns:
        fig_scatter: Plotly Figure — UMAP1 vs UMAP2 interactive scatter.
        umap_df: DataFrame with UMAP coordinates and sample metadata.
    """
    import umap as umap_lib

    if cell_lines is None:
        cell_lines = []
    if conditions is None:
        conditions = []

    # --- Column selection ---
    cols = ColumnSpec.select(df, cell_lines=cell_lines, data_type=data_type,
                             conditions=conditions)
    cols = [c for c in cols if _is_replicate_col(c)]
    if not cols:
        raise ValueError(f"No replicate columns found for data_type={data_type!r}")

    # --- Optional imputation ---
    if impute:
        df = impute_missing_replicates(df, cell_lines=cell_lines, conditions=conditions,
                                       data_type=data_type, method=impute_method)

    # --- Build (samples × sites) matrix; drop sites still fully missing ---
    mat = df[cols].replace(0, np.nan).T      # rows = samples, columns = sites
    mat = mat.dropna(axis=1)

    # --- StandardScaler + UMAP ---
    X_scaled = StandardScaler().fit_transform(mat.values)
    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    coords = reducer.fit_transform(X_scaled)

    # --- Build metadata DataFrame ---
    umap_df = pd.DataFrame(coords, columns=["UMAP1", "UMAP2"])
    umap_df["sample"] = mat.index.tolist()

    def _cond_label(col):
        base = _REP_RE.sub("", col)
        return "_".join(base.split("_")[2:])

    def _symbol(label):
        if "EGFnINS" in label:
            return "cross"
        elif "EGF" in label:
            return "diamond"
        elif "INS" in label:
            return "square"
        return "circle"

    umap_df["condition"] = umap_df["sample"].apply(_cond_label)

    if color_by == "cell_line":
        def _cell_label(col):
            for cell in cell_lines:
                if col.startswith(cell):
                    return cell
            return "unknown"
        umap_df["color_group"] = umap_df["sample"].apply(_cell_label)
    else:
        umap_df["color_group"] = umap_df["condition"]

    umap_df["symbol"] = umap_df["condition"].apply(_symbol)

    # --- Scatter: UMAP1 vs UMAP2 ---
    fig_scatter = px.scatter(
        umap_df,
        x="UMAP1", y="UMAP2",
        color="color_group",
        symbol="symbol",
        hover_name="sample",
        hover_data={"condition": True, "UMAP1": ":.2f", "UMAP2": ":.2f",
                    "color_group": False, "symbol": False},
        title=title,
        labels={"color_group": color_by},
        width=figsize[0], height=figsize[1],
    )
    fig_scatter.update_traces(marker=dict(size=12, line=dict(width=1.2, color="DarkSlateGrey")))
    fig_scatter.update_layout(plot_bgcolor="white", paper_bgcolor="white")

    return fig_scatter, umap_df


def pca_distance_heatmap(
    df,
    cell_lines=None,
    conditions=None,
    data_type="raw:abs",
    impute=True,
    impute_method="mean",
    n_pca_components=3,
    centroid_stat="median",
    zmax=None,
    figsize=(600, 600),
    title="Distance between conditions (PCA centroid)",
):
    """
    Heatmap of pairwise Euclidean distances between condition centroids in PCA space.

    For each condition×timepoint group (e.g. EGF_2, INS_10), the median (or mean)
    of the replicate positions in 3D PCA space is computed as the centroid.
    The distance matrix between all centroids is displayed as a colour-scaled
    heatmap, ordered biologically (full → starve → stimulation timepoints per
    condition). Tight clustering of full/starve and increasing distance to
    stimulated timepoints is the expected result.

    Based on the "PLOT HEAT MAPS" blocks in notebooks/02_qc/PCA.ipynb,
    adapted to the project naming convention.

    Args:
        df: DataFrame following the project naming convention.
        cell_lines: list of cell-line prefixes, e.g. ["WT"].
        conditions: list of condition substrings, e.g. ["_EGF_", "_INS_", "_EGFnINS_"].
        data_type: data-type string for replicate columns, default "raw:abs".
        impute: if True run impute_missing_replicates() before PCA (default True).
        impute_method: "mean" or "median", passed to impute_missing_replicates().
        n_pca_components: number of PCs used to compute centroid distances (default 3).
        centroid_stat: "median" (default) or "mean" — statistic for centroid computation.
        zmax: colour scale ceiling; None auto-scales.
        figsize: (width, height) in pixels for the Plotly figure.
        title: figure title.

    Returns:
        fig: Plotly Figure — distance heatmap.
        dist_df: DataFrame — the ordered distance matrix.
        pca_df: DataFrame — raw PC coordinates with condition metadata.
    """
    if cell_lines is None:
        cell_lines = []
    if conditions is None:
        conditions = []

    # --- Column selection ---
    cols = ColumnSpec.select(df, cell_lines=cell_lines, data_type=data_type,
                             conditions=conditions)
    cols = [c for c in cols if _is_replicate_col(c)]
    if not cols:
        raise ValueError(f"No replicate columns found for data_type={data_type!r}")

    # --- Optional imputation ---
    if impute:
        df = impute_missing_replicates(df, cell_lines=cell_lines, conditions=conditions,
                                       data_type=data_type, method=impute_method)

    # --- Build (samples × sites) matrix; drop sites still missing ---
    mat = df[cols].replace(0, np.nan).T
    mat = mat.dropna(axis=1)

    # --- StandardScaler + PCA ---
    X_scaled = StandardScaler().fit_transform(mat.values)
    n_comp = min(n_pca_components, mat.shape[0], mat.shape[1])
    pca = PCA(n_components=n_comp, random_state=0)
    coords = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        coords,
        columns=[f"PC{i + 1}" for i in range(n_comp)],
    )
    pca_df["sample"] = mat.index.tolist()

    # Condition label: strip replicate suffix and data-type field (index 1),
    # but keep the cell-line prefix (index 0) so that multiple cell lines
    # produce distinct labels (e.g. WT_EGF_full vs BRAFS151A_EGF_full).
    def _make_label(col):
        parts = _REP_RE.sub("", col).split("_")
        # parts[0] = cell_line, parts[1] = data_type:subtype, parts[2:] = condition+timepoint
        return "_".join([parts[0]] + parts[2:])

    pca_df["condition"] = pca_df["sample"].apply(_make_label)

    # --- Centroids per condition group ---
    pc_cols = [f"PC{i + 1}" for i in range(n_comp)]
    if centroid_stat == "median":
        centroids = pca_df.groupby("condition")[pc_cols].median()
    else:
        centroids = pca_df.groupby("condition")[pc_cols].mean()

    # --- Pairwise Euclidean distances ---
    dist_values = squareform(pdist(centroids.values, metric="euclidean"))
    dist_df_full = pd.DataFrame(dist_values, index=centroids.index, columns=centroids.index)

    # --- Biological order: preserve original column order, deduplicate ---
    seen = {}
    for col in cols:
        label = _make_label(col)
        seen[label] = None
    # Keep only labels that ended up in centroids (some may be dropped if all-NaN)
    custom_order = [lbl for lbl in seen if lbl in dist_df_full.index]

    dist_df = dist_df_full.loc[custom_order, custom_order]

    # --- Plot ---
    fig = px.imshow(
        dist_df,
        text_auto=".0f",
        zmin=0,
        zmax=zmax,
        color_continuous_scale="Blues",
        title=title,
        width=figsize[0],
        height=figsize[1],
    )
    fig.update_layout(
        xaxis_title="Condition",
        yaxis_title="Condition",
    )

    return fig, dist_df, pca_df


##############################################################################
# Section 3: Normalization QC
##############################################################################

def normalization_boxplots(
    df,
    cell_lines=None,
    conditions=None,
    raw_data_type="raw:abs",
    norm_data_type="log2:abs",
    bins=80,
    figsize=(12, 5),
    title="Before vs after normalization",
):
    """
    Side-by-side overlaid step histograms comparing intensity distributions before and after
    normalization. One outline histogram per sample; all samples overlaid on each panel.

    After normalization the histogram peaks should align across samples.
    Raw intensities are log2-transformed for visual comparison.

    Args:
        df: DataFrame containing both raw and normalized columns.
        cell_lines: list of cell-line prefixes.
        conditions: list of condition substrings.
        raw_data_type: data-type string for pre-normalization intensities (default "raw:abs").
        norm_data_type: data-type string for post-normalization intensities (default "log2:abs").
        bins: number of histogram bins (default 80).
        figsize: (width, height).
        title: suptitle.

    Returns:
        fig, (ax_raw, ax_norm)
    """
    if cell_lines is None:
        cell_lines = []
    if conditions is None:
        conditions = []

    def _get_rep_cols(dtype):
        c = ColumnSpec.select(df, cell_lines=cell_lines, data_type=dtype, conditions=conditions)
        return [x for x in c if _is_replicate_col(x)]

    raw_cols = _get_rep_cols(raw_data_type)
    norm_cols = _get_rep_cols(norm_data_type)

    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=figsize)
    palette = plt.cm.tab20.colors

    def _draw(ax, cols, label, log_transform=False):
        if not cols:
            ax.set_title(f"No columns for {label}")
            return
        vals = df[cols].copy().replace(0, np.nan)
        if log_transform:
            vals = np.log2(vals + 1)
        short_labels = [re.sub(r"_(?:raw|log2):[a-z]+_", "_", c) for c in cols]
        for i, (col, slabel) in enumerate(zip(cols, short_labels)):
            data = vals[col].dropna().values
            ax.hist(data, bins=bins, histtype="step", linewidth=1.2,
                    color=palette[i % len(palette)], label=slabel, density=True)
        ax.set_xlabel("log2 intensity" if (log_transform or "log2" in label) else "Intensity")
        ax.set_ylabel("Density")
        ax.set_title(f"{label}{'  [log2-transformed]' if log_transform else ''}")
        ax.legend(fontsize=6, ncol=max(1, len(cols) // 20), loc="upper right")
        ax.grid(axis="both", alpha=0.3)

    _draw(ax_raw, raw_cols, raw_data_type, log_transform=("log2" not in raw_data_type))
    _draw(ax_norm, norm_cols, norm_data_type)

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig, (ax_raw, ax_norm)


def cv_before_after_normalization(
    df,
    cell_lines=None,
    conditions=None,
    raw_cv_type="raw:cv",
    norm_sd_type="log2:sd",
    cv_threshold=30.0,
    figsize=None,
    title="CV / SD before vs after normalization",
):
    """
    Violin plots comparing dispersion before (raw:cv) and after (log2:sd) normalization.

    Normalization should reduce inter-replicate variability without collapsing
    biologically meaningful differences between conditions/timepoints.

    Args:
        df: DataFrame with CV and SD columns.
        cell_lines: list of cell-line prefixes.
        conditions: list of condition substrings.
        raw_cv_type: data-type string for raw CV (default "raw:cv").
        norm_sd_type: data-type string for post-normalization SD (default "log2:sd").
        cv_threshold: reference line value.
        figsize: (width, height).
        title: suptitle.

    Returns:
        fig, (ax_before, ax_after)
    """
    if cell_lines is None:
        cell_lines = []
    if conditions is None:
        conditions = []

    raw_cols = ColumnSpec.select(df, cell_lines=cell_lines, data_type=raw_cv_type,
                                 conditions=conditions)
    norm_cols = ColumnSpec.select(df, cell_lines=cell_lines, data_type=norm_sd_type,
                                  conditions=conditions)

    n = max(len(raw_cols), len(norm_cols), 1)
    if figsize is None:
        figsize = (max(10, n * 0.55 * 2), 5)

    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=figsize)

    def _draw(ax, cols, label, threshold):
        if not cols:
            ax.set_title(f"No columns for {label}")
            return
        melted = (
            df[cols].replace(0, np.nan)
            .melt(var_name="column", value_name="value")
            .dropna()
        )
        melted["tp"] = melted["column"].apply(lambda c: c.rsplit("_", 1)[-1])
        tp_order = list(dict.fromkeys(melted["tp"]))
        sns.violinplot(data=melted, x="tp", y="value", hue="tp", ax=ax, inner="box",
                       palette="Set2", cut=0, order=tp_order, legend=False)
        ax.axhline(threshold, color="crimson", linestyle="--", linewidth=1.2,
                   label=f"Threshold {threshold:.0f}")
        ax.legend(fontsize=8)
        ax.set_title(label)
        ax.set_xlabel("Timepoint")
        ax.set_ylabel("Value")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    _draw(ax_before, raw_cols, f"Before normalization ({raw_cv_type})", cv_threshold)
    _draw(ax_after, norm_cols, f"After normalization ({norm_sd_type})", cv_threshold)

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig, (ax_before, ax_after)


##############################################################################
# Section 4: Comparing datasets
##############################################################################

def venn_diagrams(
    data_frames: list,
    labels: list,
    colors: list,
    title: str,
    column_to_compare: str = "site",
):
    """
    Venn diagram of identifier overlap across 2 or 3 DataFrames.

    Args:
        data_frames: list of 2 or 3 DataFrames.
        labels: list of set labels matching data_frames.
        colors: list of colours matching data_frames.
        title: figure title.
        column_to_compare: column whose unique values define set membership (default "site").

    Returns:
        fig, ax
    """
    n = len(data_frames)
    if n not in (2, 3):
        raise ValueError("venn_diagrams supports exactly 2 or 3 DataFrames.")

    sets = [set(df[column_to_compare].dropna().tolist()) for df in data_frames]

    fig, ax = plt.subplots(figsize=(6, 4))
    if n == 2:
        venn2(subsets=[sets[0], sets[1]], set_labels=labels, set_colors=colors, ax=ax)
    else:
        venn3(subsets=[sets[0], sets[1], sets[2]], set_labels=labels, set_colors=colors, ax=ax)

    ax.set_title(title)
    fig.tight_layout()
    plt.show()
    return fig, ax


def overlap_summary(
    data_frames: list,
    labels: list,
    column_to_compare: str = "site",
) -> pd.DataFrame:
    """
    Pairwise overlap statistics between all dataset pairs.

    Args:
        data_frames: list of DataFrames.
        labels: list of names matching data_frames.
        column_to_compare: column whose unique values define set membership (default "site").

    Returns:
        DataFrame with columns:
            dataset_A, dataset_B, n_A, n_B, n_overlap,
            pct_A_in_B (% of A found in B),
            pct_B_in_A (% of B found in A)
    """
    sets = [set(df[column_to_compare].dropna().tolist()) for df in data_frames]
    rows = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            a, b = sets[i], sets[j]
            overlap = a & b
            rows.append({
                "dataset_A": labels[i],
                "dataset_B": labels[j],
                "n_A": len(a),
                "n_B": len(b),
                "n_overlap": len(overlap),
                "pct_A_in_B": round(len(overlap) / len(a) * 100, 1) if a else 0.0,
                "pct_B_in_A": round(len(overlap) / len(b) * 100, 1) if b else 0.0,
            })
    return pd.DataFrame(rows)
