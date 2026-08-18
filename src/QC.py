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
import plotly.io as pio
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
        missing_value: value treated as undetected (default 0.0); NaN is always missing.
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

    # A site is detected only if it is neither NaN nor the missing sentinel (default 0.0).
    # Handles both NaN-based and 0-based missing encodings, matching replicate_detection_map.
    counts = (df[cols].notna() & (df[cols] != missing_value)).sum()

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


def localization_completeness(df,
                              numphos_col="NumPhos",
                              localized_col="LocalizedNumPhos",
                              nreps_col="n:reps",
                              figsize=(12, 4.5),
                              title="Phosphosite localization completeness",
                              axes=None,
                              ):
    """
    How many phospho-peptides had *every* phosphorylation localized, and how that splits by
    replicate count.

    A peptide is fully localized when `NumPhos == LocalizedNumPhos`: the search engine both
    detected n phosphorylations and could assign all n to specific S/T/Y residues. Everything
    else is incomplete, in one of two ways — nothing at all was localized
    (`LocalizedNumPhos == 0`), or some but not all of the detected phosphorylations were
    (`0 < LocalizedNumPhos < NumPhos`), so the site-level identity of that row is partly
    ambiguous. This matters beyond identification quality: `src/kinase_prediction.py` only
    scores single-localized sites, so incompletely localized peptides silently drop out of
    every kinase-based analysis downstream.

    The report is deliberately two-level: the overall split (left panel) says how much of the
    dataset is usable at site level, and the per-replicate-count breakdown (right panel) says
    whether the fully localized peptides are also the well-replicated ones — the two failure
    modes compound, and a peptide that is both ambiguously localized and seen in one replicate
    of four carries almost no usable information.

    Args:
        df: DataFrame carrying the FragPipe-derived localization columns.
        numphos_col: column with the number of phosphorylations detected on the peptide.
        localized_col: column with the number that were confidently localized.
        nreps_col: column with the replicate detection count.
        figsize: (width, height) of the figure.
        title: figure suptitle.
        axes: optional array of two Axes to draw into; created if None.

    Returns:
        Tuple (localization_table, reps_table, fig, axes):
            localization_table: one row per localization category ('all localized',
                'some not localized', 'none localized'), with peptide counts and percentages
                of the dataset.
            reps_table: one row per replicate count, with the number of peptides, how many of
                them were fully localized, and that percentage.
            fig, axes: the figure and its two Axes.

    """
    for col in [numphos_col, localized_col, nreps_col,]:
        if col not in df.columns:
            raise ValueError(f"localization_completeness: column '{col}' not found. Available "
                             f"localization columns: "
                             f"{[c for c in df.columns if 'Phos' in c or 'reps' in c]}")

    n_phos = df[numphos_col]
    n_loc = df[localized_col]

    # Guard against a column pair that cannot mean what the check assumes.
    if (n_loc > n_phos).any():
        n_bad = int((n_loc > n_phos).sum())
        raise ValueError(f"localization_completeness: '{localized_col}' exceeds '{numphos_col}' "
                         f"in {n_bad} rows. More phosphorylations cannot be localized than were "
                         f"detected — check that the two columns were not swapped.")

    fully = (n_phos == n_loc)
    none_loc = (n_loc == 0) & ~fully
    partial = ~fully & ~none_loc

    n_total = len(df)
    n_fully = int(fully.sum())
    n_none = int(none_loc.sum())
    n_partial = int(partial.sum())
    n_incomplete = n_none + n_partial

    print(f"Localization completeness ({n_total} phospho-peptides):")
    print(f"  all phosphosites localized (NumPhos == LocalizedNumPhos) : "
          f"{n_fully} ({100 * n_fully / n_total:.1f}%)")
    print(f"  not all phosphosites localized                           : "
          f"{n_incomplete} ({100 * n_incomplete / n_total:.1f}%)")
    print(f"    no phosphosite localized (LocalizedNumPhos == 0)       : "
          f"{n_none} ({100 * n_none / n_total:.1f}%)")
    print(f"    some phosphorylations could not be localized           : "
          f"{n_partial} ({100 * n_partial / n_total:.1f}%)")

    # --- Table 1: the overall split, exactly as printed above ---
    localization_table = pd.DataFrame({"n_peptides": [n_fully, n_partial, n_none,],},
                                      index=["all localized",
                                             "some not localized",
                                             "none localized",],)
    localization_table["pct_peptides"] = 100 * localization_table["n_peptides"] / n_total
    localization_table.index.name = "localization"

    # --- Table 2: fully localized peptides per replicate count ---
    reps_all = df[nreps_col].value_counts().sort_index()
    reps_fully = df.loc[fully, nreps_col].value_counts().sort_index()
    reps_index = sorted(set(reps_all.index) | set(reps_fully.index),)

    reps_table = pd.DataFrame({"n_peptides": reps_all.reindex(reps_index,).fillna(0,).astype(int,),
                               "n_fully_localized": reps_fully.reindex(reps_index,).fillna(0,).astype(int,),},
                              index=reps_index,)
    reps_table["pct_fully_localized"] = (100 * reps_table["n_fully_localized"]
                                         / reps_table["n_peptides"].replace(0, np.nan,))
    reps_table.index.name = nreps_col

    print(f"\nPeptides with all phosphosites localized, by number of replicates detected:")
    print(reps_table.round(1,).to_string())

    # --- Figure ---
    create_fig = axes is None
    if create_fig:
        fig, axes = plt.subplots(1, 2, figsize=figsize,)
    else:
        axes = np.atleast_1d(axes,)
        fig = axes[0].get_figure()

    # Left panel: one bar for the fully localized peptides, one for the incomplete ones,
    # the latter stacked so the two ways of being incomplete stay visible.
    axes[0].bar(0,
                n_fully,
                color="mediumseagreen",
                edgecolor="white",
                label="all phosphosites localized",)
    axes[0].bar(1,
                n_partial,
                color="lightcoral",
                edgecolor="white",
                label="some phosphorylations not localized",)
    axes[0].bar(1,
                n_none,
                bottom=n_partial,
                color="indianred",
                edgecolor="white",
                label="no phosphosite localized",)
    axes[0].text(0,
                 n_fully,
                 f"{n_fully}\n({100 * n_fully / n_total:.1f}%)",
                 ha="center",
                 va="bottom",
                 fontsize=8,)
    axes[0].text(1,
                 n_incomplete,
                 f"{n_incomplete}\n({100 * n_incomplete / n_total:.1f}%)",
                 ha="center",
                 va="bottom",
                 fontsize=8,)
    axes[0].set_xticks([0, 1,],)
    axes[0].set_xticklabels(["all localized", "not all localized",],)
    axes[0].set_xlim(-0.7, 1.7,)
    axes[0].set_ylim(0, n_total * 1.15,)
    axes[0].set_ylabel("Number of phospho-peptides")
    axes[0].set_title(f"Localization completeness\n({n_total} phospho-peptides)",
                      fontsize=10,)
    axes[0].legend(frameon=False, fontsize=8,)
    axes[0].grid(axis="y", alpha=0.3,)

    # Right panel: the same fully-localized count, split by how many replicates saw the
    # peptide. The faint bar behind is the group total, so the green bar can be read as a
    # fraction rather than only as a count.
    xr = np.arange(len(reps_table),)
    axes[1].bar(xr,
                reps_table["n_peptides"],
                color="lightgrey",
                edgecolor="white",
                label="all peptides",)
    axes[1].bar(xr,
                reps_table["n_fully_localized"],
                color="mediumseagreen",
                edgecolor="white",
                label="all phosphosites localized",)
    for xi, (_, row) in zip(xr, reps_table.iterrows(),):
        axes[1].text(xi,
                     row["n_peptides"],
                     f"{int(row['n_fully_localized'])}\n({row['pct_fully_localized']:.0f}%)",
                     ha="center",
                     va="bottom",
                     fontsize=8,)
    axes[1].set_xticks(xr,)
    axes[1].set_xticklabels(reps_table.index.astype(str,),)
    axes[1].set_ylim(0, reps_table["n_peptides"].max() * 1.2,)
    axes[1].set_xlabel(f"Number of replicates detected ({nreps_col})")
    axes[1].set_ylabel("Number of phospho-peptides")
    axes[1].set_title("Fully localized peptides per replicate count", fontsize=10,)
    axes[1].legend(frameon=False, fontsize=8,)
    axes[1].grid(axis="y", alpha=0.3,)

    if create_fig:
        fig.suptitle(title, weight="bold",)
        fig.tight_layout()
    return localization_table, reps_table, fig, axes


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
# Section 1b: Replicate coverage across timepoints
##############################################################################

def _parse_data_column(col,):
    """
    Split a project data-column name into its naming-convention fields.

    Args:
        col: column name, e.g. "WT_raw:abs_EGF_2_r1".

    Returns:
        Dict with keys cell_line, data_type, condition, timepoint, replicate, or  None if the name does not have the five replicate-level fields.

    """
    parts = col.split("_")
    if len(parts) < 5:
        return None
    return {"cell_line": parts[0],
            "data_type": parts[1],
            "condition": parts[2],
            "timepoint": parts[3],
            "replicate": parts[4],}


def _resolve_site_index(df,
                        site_col=None,
                        ):
    """
    Pick the index used to label sites in the coverage tables.

    The returned Index carries the source column in its `.name`, which is what lets
    filter_by_coverage match the coverage tables back onto the original DataFrame
    without the caller having to remember which key was used.

    Args:
        df: DataFrame following the project naming convention.
        site_col: column to use as the site key. If None, the first of the project
            keys 'site', 'site_index', 'peptide_index' present in df is used, then
            the raw FragPipe 'Index' as a fallback for tables that have not been
            renamed yet, and finally df.index if none of them exist.

    Returns:
        pandas Index of site keys, aligned with df's rows, named after the column
        it was taken from (None if df.index was used).

    """
    if site_col is not None:
        if site_col not in df.columns:
            raise ValueError(f"site_col={site_col!r} not found in DataFrame.")
        return pd.Index(df[site_col], name=site_col,)

    # Project keys first, raw FragPipe column names last: 'Index' is what the dia-quant output ships with before the preprocessing rename to 'peptide_index'.
    for candidate in ("site", "site_index", "peptide_index", "Index",):
        if candidate in df.columns:
            return pd.Index(df[candidate], name=candidate,)
    return df.index


def replicate_coverage_matrix(df,
                              cell_lines=None,
                              conditions=None,
                              data_type="raw:abs",
                              timepoints=None,
                              exclude_full=False,
                              exclude_starve=False,
                              missing_value=0.0,
                              site_col=None,
                              ):
    """
    Count, per site, in how many replicates it was detected at each
    cell line × condition × timepoint.

    This is the raw material for every other function in this section: the
    per-timepoint counts, before any minimum is taken across timepoints.

    Args:
        df: DataFrame following the project naming convention.
        cell_lines: list of cell-line prefixes, e.g. ["WT", "EGFRT693A"]. If None,every cell line found among the `data_type` replicate columns is used.
        conditions: list of condition substrings, e.g. ["_EGF_"]. If None, every condition found among those columns is used.
        data_type: data-type string for per-replicate columns, default "raw:abs".
        timepoints: optional list of timepoint labels to restrict to, e.g. ["2", "5", "10"]. If None, all timepoints present are used.
        exclude_full: if True, drop the 'full' timepoint.
        exclude_starve: if True, drop the 'starve' timepoint.
        missing_value: value treated as undetected (default 0.0); NaN is always missing.
        site_col: column to use as site key (see _resolve_site_index).

    Returns:
        DataFrame with one row per site and a MultiIndex column (cell_line, condition, timepoint) holding the integer number of replicates in which that site was detected.

    """
    parsed = []
    for col in df.columns:
        if not _is_replicate_col(col,): #the column is not a replicate column (raw:abs)
            continue
        info = _parse_data_column(col,) # extrac Cell_, data_type, Condition, Timepoint and replicate informaiton from the column name
        if info is None or info["data_type"] != data_type:
            continue
        info["column"] = col
        parsed.append(info,)

    if not parsed:
        raise ValueError(f"No replicate columns found for data_type={data_type!r}.")

    # Cell lines and conditions are matched on the exact parsed field rather than by substring, because ColumnSpec.matches uses startswith: a prefix such as "BRAFS151A" would otherwise also pull in BRAFS151A1 and BRAFS151A2.
    if cell_lines is None:
        cell_lines = list(dict.fromkeys(p["cell_line"] for p in parsed),)
    if conditions is None:
        conditions = list(dict.fromkeys(p["condition"] for p in parsed),)
    conditions = [c.strip("_",) for c in conditions]   # accept "_EGF_" or "EGF"

    drop_tps = set()
    if exclude_full:
        drop_tps.add("full",)
    if exclude_starve:
        drop_tps.add("starve",)

    parsed = [p for p in parsed
              if p["cell_line"] in cell_lines
              and p["condition"] in conditions
              and p["timepoint"] not in drop_tps
              and (timepoints is None or p["timepoint"] in timepoints)]
    if not parsed:
        raise ValueError(f"No replicate columns left after filtering on cell_lines={cell_lines}, conditions={conditions}, timepoints={timepoints}.")

    # Group the replicate columns of each cell line × condition × timepoint together
    groups = defaultdict(list,)
    for p in parsed:
        groups[(p["cell_line"], p["condition"], p["timepoint"],)].append(p["column"],)

    site_index = _resolve_site_index(df, site_col=site_col,)

    counts = {}
    for key, cols in groups.items(): # For each cell_line, condition, time_point cit counts individual detections and then sums it up to know in how many replicates it  is pressent
        detected = df[cols].notna() & (df[cols] != missing_value)
        counts[key] = detected.sum(axis=1,).to_numpy()

    # Order the columns: cell line and condition as requested, timepoints experimentally
    ordered_keys = []
    for cell in cell_lines:
        for cond in conditions:
            tps = _order_timepoints([k[2] for k in groups if k[0] == cell and k[1] == cond],)
            ordered_keys.extend([(cell, cond, tp,) for tp in tps],)

    matrix = pd.DataFrame({k: counts[k] for k in ordered_keys},
                          index=site_index,)
    matrix.columns = pd.MultiIndex.from_tuples(ordered_keys,
                                               names=["cell_line", "condition", "timepoint",],)
    return matrix


def replicate_coverage_depth(counts,
                             groups=None,
                             include_all=False,
                             all_label="ALL",
                             ):
    """
    Reduce a coverage matrix to one depth value per site per sample group.

    The depth of a site is the smallest per-timepoint replicate count over every (cell line × condition × timepoint) in the group.

    Args:
        counts: output of replicate_coverage_matrix.
        groups: how to group cell lines. None (default) gives one group per cell line; a dict maps a group name to a list of cell lines, e.g. {"WT+EGFR": ["WT", "EGFRT693A"]}
        include_all: if True, append a group containing every cell line in `counts`.
        all_label: name of that group, default "ALL".

    Returns:
        DataFrame with one row per site and one integer column per group.

    """
    cell_lines = list(dict.fromkeys(counts.columns.get_level_values("cell_line",),),)

    # Always create a grouping dictionary, if NO group is defined, each group is its individual cell line
    if groups is None:
        group_map = {cell: [cell] for cell in cell_lines}
    elif isinstance(groups, dict,):
        group_map = {name: ([members] if isinstance(members, str,) else list(members,))
                     for name, members in groups.items()}
    else:
        members = list(groups,)
        group_map = {"+".join(members,): members}

    if include_all:
        group_map[all_label] = cell_lines

    depth = {}
    for name, members in group_map.items():
        unknown = [m for m in members if m not in cell_lines]
        if unknown:
            raise ValueError(f"Group {name!r} refers to cell lines not present in the "
                             f"coverage matrix: {unknown}. Available: {cell_lines}")
        cols = [c for c in counts.columns if c[0] in members]
        depth[name] = counts[cols].min(axis=1,)

    return pd.DataFrame(depth, index=counts.index,)


def sites_with_coverage(depth, #I think this and the fucntion below shold be merged.
                        group,
                        min_reps=1,
                        ):
    """
    List the sites covered in at least `min_reps` replicates at every timepoint.

    The keys are whichever column _resolve_site_index picked when the coverage matrix
    was built ('site' by default on the current tables, 'peptide_index' or 'Index' on
    tables that have no composite site key). Read `depth.index.name` to see which one,
    or use filter_by_coverage to subset a DataFrame without handling the key at all.

    Args:
        depth: output of replicate_coverage_depth.
        group: column name of the group to query, e.g. "WT" or "WT + EGFRT693A".
        min_reps: minimum coverage depth required, default 1.

    Returns:
        pandas Index of the site keys meeting the requirement.

    """
    if group not in depth.columns:
        raise ValueError(f"Group {group!r} not found. Available: {list(depth.columns,)}")
    return depth.index[depth[group] >= min_reps]


def filter_by_coverage(df,
                       depth,
                       group,
                       min_reps=1,
                       site_col=None,
                       ):
    """
    Subset a DataFrame to the sites covered in at least `min_reps` replicates at every timepoint.

    Does what sites_with_coverage + .isin() does, but resolves the key column itself from
    `depth.index.name`, so the filter cannot silently mismatch the key the coverage matrix
    was built on (e.g. filtering on 'peptide_index' a table indexed by 'site').

    Args:
        df: the DataFrame the coverage matrix was computed from.
        depth: output of replicate_coverage_depth.
        group: column name of the group to query, e.g. "WT + EGFRT693A".
        min_reps: minimum coverage depth required, default 1.
        site_col: override for the key column in df; by default the column the coverage
            index was taken from, or df.index when it has no name.

    Returns:
        Filtered copy-free view of df containing only the covered sites.

    """
    keys = sites_with_coverage(depth,
                               group=group,
                               min_reps=min_reps,)

    key_col = site_col if site_col is not None else depth.index.name
    if key_col is None:
        return df.loc[df.index.isin(keys,)]
    if key_col not in df.columns:
        raise ValueError(f"Key column {key_col!r} not found in the DataFrame. Pass site_col= "
                         f"explicitly, or rebuild the coverage matrix from this DataFrame.")
    return df.loc[df[key_col].isin(keys,)]


def subset_by_coverage(df,
                       cell_lines,
                       min_reps=1,
                       conditions=None,
                       data_type="raw:abs",
                       timepoints=None,
                       exclude_full=False,
                       exclude_starve=False,
                       missing_value=0.0,
                       group_label=None,
                       verbose=True,
                       ):
    """
    Subset a DataFrame to the sites one set of cell lines measured deeply enough, in one call.

    Composition of the three steps that were previously written out by hand —
    replicate_coverage_matrix -> replicate_coverage_depth -> filter_by_coverage — for the
    common case of a single ad-hoc group: "give me the sites that these cell lines all
    detected in at least `min_reps` replicates at every timepoint". Building the coverage
    matrix from `df` itself also means the filter cannot be applied to a table other than
    the one it was computed on, which is the failure mode of carrying a `depth` table
    between notebook sections.

    Use the three functions separately when several groups are being compared (the coverage
    matrix is then computed once and reduced many times) or when the depth table itself is
    wanted for a plot; use this when the goal is just the filtered rows.

    Args:
        df: DataFrame following the project naming convention, carrying per-replicate columns.
        cell_lines: list of cell-line prefixes that must ALL cover a site, e.g. ["WT"] or
            ["WT", "EGFRT693A"]. Adding a cell line can only remove sites, never add them.
        min_reps: minimum number of replicates the site must be present in at *every*
            timepoint (its coverage depth), default 1.
        conditions: list of condition substrings, e.g. ["_EGF_"]; None auto-detects.
        data_type: data-type string of the per-replicate columns, default "raw:abs". This is
            the raw detection evidence, so it stays "raw:abs" even when the PCA or the
            clustering that follows runs on log2:abs.
        timepoints: optional list of timepoint labels to restrict the requirement to.
        exclude_full: if True, the 'full' timepoint does not constrain the depth.
        exclude_starve: if True, the 'starve' timepoint does not constrain the depth.
        missing_value: value treated as undetected (default 0.0); NaN is always missing.
        group_label: name of the group in the intermediate depth table; defaults to the
            cell lines joined with '+'.
        verbose: print how many sites were kept.

    Returns:
        Filtered DataFrame containing only the covered sites.

    """
    label = group_label if group_label is not None else "+".join(cell_lines,)

    counts = replicate_coverage_matrix(df,
                                       cell_lines=cell_lines,
                                       conditions=conditions,
                                       data_type=data_type,
                                       timepoints=timepoints,
                                       exclude_full=exclude_full,
                                       exclude_starve=exclude_starve,
                                       missing_value=missing_value,)
    depth = replicate_coverage_depth(counts,
                                     groups={label: cell_lines},)
    subset = filter_by_coverage(df,
                                depth,
                                group=label,
                                min_reps=min_reps,)

    if verbose:
        print(f"{label}: {len(subset)} of {len(df)} sites covered in >= {min_reps} "
              f"replicate(s) at every timepoint ({len(subset) / max(len(df), 1):.1%})")
    return subset


def add_nreps_columns(df,
                      cell_lines=None,
                      conditions=None,
                      data_type="raw:abs",
                      timepoints=None,
                      exclude_full=False,
                      exclude_starve=False,
                      missing_value=0.0,
                      add_combined=False,
                      combined_col="n:reps",
                      ):
    """
    Write the coverage depth onto the DataFrame as n:reps columns.

    Same number as replicate_coverage_depth — replicates the site is present in at
    *every* timepoint — but stored as a column per cell line × condition instead of a
    separate table. That matters for two reasons: the criterion survives a save to
    disk, so downstream notebooks can filter without recomputing the coverage matrix,
    and the column is then a normal replicate count, so the existing
    filters.filter_by_nreps(df, min_reps, nreps_col=...) applies unchanged.

    This is also the definition the project already documents for LFQ n:reps: a site
    detected in (r1, r3) at one timepoint and (r2, r3) at another has n:reps 2, and a
    site detected in one replicate at any single timepoint has n:reps 1.

    Output column names:
        {cell_line}_n:reps_{treatment}_all    ('all' in the timepoint field because the
                                               value spans the whole time course; keeps
                                               the column selectable with ColumnSpec)
        n:reps                                (only if add_combined=True — the minimum
                                               across every selected cell line, i.e. the
                                               depth at which a site is usable everywhere)

    Args:
        df: DataFrame following the project naming convention.
        cell_lines: list of cell-line prefixes; None (default) auto-detects all of them.
        conditions: list of condition substrings, e.g. ["_EGF_"]; None auto-detects.
        data_type: data-type string for per-replicate columns, default "raw:abs".
        timepoints: optional list of timepoint labels to restrict the minimum to.
        exclude_full: if True, the 'full' timepoint does not constrain the count.
        exclude_starve: if True, the 'starve' timepoint does not constrain the count.
        missing_value: value treated as undetected (default 0.0); NaN is always missing.
        add_combined: if True, also write a single `combined_col` holding the minimum
            across all selected cell lines.
        combined_col: name of that column, default "n:reps".

    Returns:
        Copy of df with the new count columns appended.

    """
    counts = replicate_coverage_matrix(df,
                                       cell_lines=cell_lines,
                                       conditions=conditions,
                                       data_type=data_type,
                                       timepoints=timepoints,
                                       exclude_full=exclude_full,
                                       exclude_starve=exclude_starve,
                                       missing_value=missing_value,)

    # replicate_coverage_matrix keeps df's row order, so the values line up positionally.
    # Assigning .to_numpy() rather than the Series avoids reindexing on the site keys,
    # which need not be unique.
    new_cols = {}
    keys = list(dict.fromkeys((c[0], c[1],) for c in counts.columns),)
    for cell_line, condition in keys:
        cols = [c for c in counts.columns if c[0] == cell_line and c[1] == condition]
        new_cols[f"{cell_line}_n:reps_{condition}_all"] = counts[cols].min(axis=1,).to_numpy()

    if add_combined:
        new_cols[combined_col] = counts.min(axis=1,).to_numpy()

    return pd.concat([df.copy(), pd.DataFrame(new_cols, index=df.index,)], axis=1,)


def coverage_summary(depth,
                     max_depth=None,
                     print_summary=True,
                     ):
    """
    Count how many sites reach each coverage depth, per group.

    Args:
        depth: output of replicate_coverage_depth.
        max_depth: highest depth threshold to report. If None, the largest depth  observed in `depth` is used (i.e. the replicate count of the design).
        print_summary: if True, print the table as counts and percentages.

    Returns:
        DataFrame indexed by group with columns n_sites, 'n_min1', 'n_min2', …
        and the matching 'pct_min1', 'pct_min2', … percentages of all sites.

    """
    if max_depth is None:
        max_depth = int(depth.to_numpy().max(),) if len(depth,) else 0
    max_depth = max(int(max_depth,), 1,)

    n_sites = len(depth,)
    rows = {}
    for group in depth.columns:
        row = {"n_sites": n_sites}
        for k in range(1, max_depth + 1,):
            n_k = int((depth[group] >= k).sum(),)
            row[f"n_min{k}"] = n_k
            row[f"pct_min{k}"] = round(n_k / n_sites * 100, 1,) if n_sites else 0.0
        rows[group] = row

    table = pd.DataFrame(rows,).T
    int_cols = [c for c in table.columns if c.startswith("n_",)]
    table[int_cols] = table[int_cols].astype(int,)

    if print_summary:
        print(f"Replicate coverage per timepoint ({n_sites} sites total):")
        for group, row in table.iterrows():
            parts = [f"≥{k} rep{'s' if k > 1 else ' '}: {int(row[f'n_min{k}'],)} "
                     f"({row[f'pct_min{k}']:.1f}%)"
                     for k in range(1, max_depth + 1,)]
            print(f"  {group:<20} " + " | ".join(parts,))

    return table


def plot_replicate_coverage(depth,
                            max_depth=None,
                            figsize=None,
                            title="Sites covered in n replicates at every timepoint",
                            annotate=True,
                            as_percent=False,
                            ax=None,
                            ):
    """
    Grouped bar chart of coverage depth per sample group.

    One bar group per cell line (or cell-line combination), one bar per depth
    threshold: sites present in at least 1, 2, 3 … replicates at *every* timepoint.

    Args:
        depth: output of replicate_coverage_depth.
        max_depth: highest depth threshold to draw; defaults to the largest observed.
        figsize: (width, height); auto-sized if None.
        title: figure title.
        annotate: if True, write the count above each bar.
        as_percent: if True, plot percentages of all sites instead of counts.
        ax: optional Axes.

    Returns:
        summary_table, fig, ax — the table is the coverage_summary output used to draw.

    """
    table = coverage_summary(depth,
                             max_depth=max_depth,
                             print_summary=False,)
    depths = sorted(int(c.replace("n_min", "",),)
                    for c in table.columns if c.startswith("n_min",))

    groups = list(table.index,)
    x = np.arange(len(groups,),)
    width = 0.8 / len(depths,)
    colors = plt.cm.viridis(np.linspace(0.15, 0.8, len(depths,),),)

    if figsize is None:
        figsize = (max(7, len(groups,) * 1.4,), 5,)
    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(figsize=figsize,)
    else:
        fig = ax.get_figure()

    for i, k in enumerate(depths,):
        values = table[f"pct_min{k}"] if as_percent else table[f"n_min{k}"]
        offset = (i - (len(depths,) - 1) / 2) * width
        bars = ax.bar(x + offset,
                      values.to_numpy(),
                      width=width,
                      color=colors[i],
                      edgecolor="white",
                      label=f"≥ {k} replicate{'s' if k > 1 else ''}",)
        if annotate:
            for bar, value in zip(bars, values.to_numpy(),):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{value:.0f}" if not as_percent else f"{value:.0f}%",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=90,)

    ax.set_xticks(x,)
    ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=9,)
    ax.set_ylabel("% of sites" if as_percent else "Number of sites")
    ax.set_xlabel("Cell line / combination")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8,)
    ax.grid(axis="y", alpha=0.3,)
    if annotate:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15,)

    if create_fig:
        fig.tight_layout()
    return table, fig, ax


def coverage_per_timepoint(counts,
                           max_depth=None,
                           print_summary=False,
                           ):
    """
    Count sites reaching each replicate depth at each individual timepoint.

    Unlike replicate_coverage_depth this takes no minimum across timepoints, so it shows *which* timepoint limits the coverage of a cell line — the bottleneck behind the depth numbers.

    Args:
        counts: output of replicate_coverage_matrix.
        max_depth: highest depth threshold to report; defaults to the largest observed.
        print_summary: if True, print the table.

    Returns:
        Tidy DataFrame with columns cell_line, condition, timepoint, n_min1, n_min2, …

    """
    if max_depth is None:
        max_depth = int(counts.to_numpy().max(),) if counts.size else 0
    max_depth = max(int(max_depth,), 1,)

    rows = []
    for cell, cond, tp in counts.columns:
        col = counts[(cell, cond, tp,)]
        row = {"cell_line": cell, "condition": cond, "timepoint": tp}
        for k in range(1, max_depth + 1,):
            row[f"n_min{k}"] = int((col >= k).sum(),)
        rows.append(row,)

    table = pd.DataFrame(rows,)
    if print_summary:
        print(table.to_string(index=False,),)
    return table


def plot_coverage_per_timepoint(counts,
                                min_reps=1,
                                figsize=(10, 5),
                                title=None,
                                ax=None,
                                ):
    """
    Line plot of detected sites per timepoint, one line per cell line.

    Reads the per-timepoint counts (no minimum across timepoints), so a dip in one
    line identifies the timepoint that limits that cell line's coverage depth.

    Args:
        counts: output of replicate_coverage_matrix.
        min_reps: replicate depth required at the timepoint to count a site, default 1.
        figsize: (width, height).
        title: figure title; auto-generated from min_reps if None.
        ax: optional Axes.

    Returns:
        table, fig, ax — table is the plotted counts, timepoints × (cell_line, condition).

    """
    per_tp = coverage_per_timepoint(counts,
                                    max_depth=min_reps,
                                    print_summary=False,)
    col = f"n_min{min_reps}"
    if col not in per_tp.columns:
        raise ValueError(f"min_reps={min_reps} exceeds the replicate count of the design.")

    # pivot_table sorts its column index; reindex to keep the cell-line order the coverage matrix was built with, so every plot in this section is ordered alike.
    tp_order = _order_timepoints(per_tp["timepoint"],)
    key_order = list(dict.fromkeys((c[0], c[1],) for c in counts.columns),)
    table = (per_tp.pivot_table(index="timepoint",
                                columns=["cell_line", "condition",],
                                values=col,)
                   .reindex(index=tp_order,
                            columns=pd.MultiIndex.from_tuples(key_order,
                                                              names=["cell_line", "condition",],),))

    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(figsize=figsize,)
    else:
        fig = ax.get_figure()

    x = np.arange(len(table,),)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(table.columns,), 1,),),)
    for i, key in enumerate(table.columns,):
        label = f"{key[0]} ({key[1]})" if len(set(table.columns.get_level_values(1,),),) > 1 else key[0]
        ax.plot(x,
                table[key].to_numpy(),
                marker="o",
                markersize=4,
                linewidth=1.5,
                color=colors[i],
                label=label,)

    ax.set_xticks(x,)
    ax.set_xticklabels(table.index, rotation=45, ha="right",)
    ax.set_xlabel("Timepoint")
    ax.set_ylabel(f"Sites detected in ≥ {min_reps} replicate{'s' if min_reps > 1 else ''}")
    ax.set_title(title or f"Detected sites per timepoint (≥ {min_reps} replicate {'s' if min_reps > 1 else ''})")
    ax.legend(frameon=False, fontsize=8, ncol=2,)
    ax.grid(axis="y", alpha=0.3,)

    if create_fig:
        fig.tight_layout()
    return table, fig, ax


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
    symbol_by="stimulation",
    figsize=(1000, 800),
    title="PCA of replicate samples",
):
    """
    Interactive Plotly PCA scatter plot of replicate samples.

    Samples (replicates) are the observations; phosphosites are the features.
    Points are coloured according to `color_by` and shaped according to
    `symbol_by`. By default colour = condition×timepoint group and marker =
    stimulation type (EGF = diamond, INS = square, EGFnINS = cross,
    controls = circle). For the mutant datasets (only EGF stimulation, several
    cell lines) the useful combination is color_by="cell_line",
    symbol_by="timepoint": every point coloured by its cell line and shaped by
    its timepoint. With the default symbol_by="stimulation" the mutant samples
    would all share one marker (all EGF), hiding the timepoints.

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
        symbol_by: what the marker shape encodes:
                   "stimulation" (default) — EGF/INS/EGFnINS/control marker;
                   "timepoint" — one marker per timepoint (full, starve, 2, ...);
                   "condition" — one marker per condition×timepoint group.
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

    def _timepoint_label(col):
        # Timepoint is the last token of the base name, e.g. WT_raw:abs_EGF_2 -> "2"
        base = _REP_RE.sub("", col)
        return base.split("_")[-1]

    # Derive marker from stimulation type in the condition label
    def _stimulation_symbol(label):
        if "EGFnINS" in label:
            return "cross"
        elif "EGF" in label:
            return "diamond"
        elif "INS" in label:
            return "square"
        return "circle"

    pca_df["condition"] = pca_df["sample"].apply(_cond_label)
    pca_df["timepoint"] = pca_df["sample"].apply(_timepoint_label)

    if color_by == "cell_line":
        def _cell_label(col):
            for cell in cell_lines:
                if col.startswith(cell):
                    return cell
            return "unknown"
        pca_df["color_group"] = pca_df["sample"].apply(_cell_label)
    else:
        pca_df["color_group"] = pca_df["condition"]

    # Marker shape encodes whatever symbol_by selects
    if symbol_by == "timepoint":
        pca_df["symbol"] = pca_df["timepoint"]
    elif symbol_by == "condition":
        pca_df["symbol"] = pca_df["condition"]
    else:  # "stimulation" (default)
        pca_df["symbol"] = pca_df["condition"].apply(_stimulation_symbol)

    # --- Scatter: PC1 vs PC2 ---
    var_pct = pca.explained_variance_ratio_ * 100
    fig_scatter = px.scatter(
        pca_df,
        x="PC1", y="PC2",
        color="color_group",
        symbol="symbol",
        hover_name="sample",
        hover_data={"condition": True, "timepoint": True,
                    "PC1": ":.2f", "PC2": ":.2f",
                    "color_group": False, "symbol": False},
        title=title,
        labels={
            "PC1": f"PC1 ({var_pct[0]:.1f}% var)",
            "PC2": f"PC2 ({var_pct[1]:.1f}% var)",
            "color_group": color_by,
            "symbol": symbol_by,
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
    symbol_by="stimulation",
    figsize=(1000, 800),
    title="UMAP of replicate samples",
):
    """
    Interactive Plotly UMAP scatter plot of replicate samples.

    Samples (replicates) are the observations; phosphosites are the features.
    This is the UMAP equivalent of pca_plot_interactive(): the same matrix is
    embedded with UMAP instead of PCA.

    Points are coloured according to `color_by` and shaped according to
    `symbol_by`. By default colour = condition×timepoint group and marker =
    stimulation type (EGF = diamond, INS = square, EGFnINS = cross,
    controls = circle). For the mutant datasets (only EGF stimulation, several
    cell lines) the useful combination is color_by="cell_line",
    symbol_by="timepoint": every point coloured by its cell line and shaped by
    its timepoint. With the default symbol_by="stimulation" the mutant samples
    would all share one marker (all EGF), hiding the timepoints.

    NOTE — identical/duplicated samples do NOT co-locate in UMAP.
    UMAP is a stochastic, graph-based force-directed layout: every sample is a
    separate node, and negative sampling (repulsion) can push even byte-for-byte
    identical points apart, regardless of `random_state`. In this project the
    `full` and `starve` timepoints are duplicated across the EGF/INS/EGFnINS
    conditions (they are the same physical samples), so they will appear
    SEPARATED here even though their feature vectors are identical. This is an
    artifact of the algorithm, not a data or imputation problem. To check that
    replicates of a timepoint agree (i.e. that identical samples overlap), use
    pca_plot_interactive() instead — PCA is a deterministic linear projection
    that maps identical vectors to the exact same coordinate. Use UMAP only for
    exploring non-linear grouping structure, not for exact-overlap QC.

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
        symbol_by: what the marker shape encodes:
                   "stimulation" (default) — EGF/INS/EGFnINS/control marker;
                   "timepoint" — one marker per timepoint (full, starve, 2, ...);
                   "condition" — one marker per condition×timepoint group.
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

    def _timepoint_label(col):
        # Timepoint is the last token of the base name, e.g. WT_raw:abs_EGF_2 -> "2"
        base = _REP_RE.sub("", col)
        return base.split("_")[-1]

    def _stimulation_symbol(label):
        if "EGFnINS" in label:
            return "cross"
        elif "EGF" in label:
            return "diamond"
        elif "INS" in label:
            return "square"
        return "circle"

    umap_df["condition"] = umap_df["sample"].apply(_cond_label)
    umap_df["timepoint"] = umap_df["sample"].apply(_timepoint_label)

    if color_by == "cell_line":
        def _cell_label(col):
            for cell in cell_lines:
                if col.startswith(cell):
                    return cell
            return "unknown"
        umap_df["color_group"] = umap_df["sample"].apply(_cell_label)
    else:
        umap_df["color_group"] = umap_df["condition"]

    # Marker shape encodes whatever symbol_by selects
    if symbol_by == "timepoint":
        umap_df["symbol"] = umap_df["timepoint"]
    elif symbol_by == "condition":
        umap_df["symbol"] = umap_df["condition"]
    else:  # "stimulation" (default)
        umap_df["symbol"] = umap_df["condition"].apply(_stimulation_symbol)

    # --- Scatter: UMAP1 vs UMAP2 ---
    fig_scatter = px.scatter(
        umap_df,
        x="UMAP1", y="UMAP2",
        color="color_group",
        symbol="symbol",
        hover_name="sample",
        hover_data={"condition": True, "timepoint": True,
                    "UMAP1": ":.2f", "UMAP2": ":.2f",
                    "color_group": False, "symbol": False},
        title=title,
        labels={"color_group": color_by, "symbol": symbol_by},
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


##############################################################################
# Section 5: Differential statistics (limma results)
##############################################################################

# Default colour per stimulation condition, so the same condition keeps the same
# colour across every plot in this section.
_CONDITION_COLORS = {"EGF": "tab:blue",
                     "INS": "tab:orange",
                     "EGFnINS": "tab:green",}


def _order_timepoints(labels,):
    """
    Sort timepoint labels into the experimental order used everywhere in the project.

    'full' and 'starve' are qualitative and come first (in that order); every remaining
    label that parses as a number is sorted numerically, so 90 lands after 15 instead of
    between 15 and 2 as a string sort would put it. Labels that are neither are appended
    alphabetically rather than dropped.

    Args:
        labels: iterable of timepoint labels, e.g. ["90", "2", "full", "10"].

    Returns:
        List of the unique labels in experimental order.

    """
    labels = list(dict.fromkeys(labels,))
    head = [t for t in ("full", "starve",) if t in labels]
    numeric = sorted([t for t in labels if t not in ("full", "starve",)
                      and str(t).replace(".", "", 1,).isdigit()],
                     key=float,)
    other = sorted([t for t in labels if t not in head and t not in numeric],)
    return head + numeric + other


def _limma_columns(df,
                   cell_line,
                   condition,
                   data_type,
                   exclude_full,
                   ):
    """
    Select the limma columns of one data type for one cell line × condition.

    Thin wrapper over `ColumnSpec.select` that also returns the timepoint label of each
    column, so callers never have to parse column names themselves.

    Args:
        df: DataFrame carrying the merged limma columns.
        cell_line: cell-line prefix, e.g. "WT".
        condition: condition token without underscores, e.g. "EGF".
        data_type: limma data type, e.g. "log2:limmaFC" or "log2:FDR".
        exclude_full: if True, drop the 'full' timepoint.

    Returns:
        Tuple (columns, timepoints): the matching column names and their timepoint labels,
        both ordered by `_order_timepoints`.

    """
    cols = ColumnSpec.select(df,
                             cell_lines=[cell_line],
                             data_type=data_type,
                             conditions=[f"_{condition}_"],
                             exclude_full=exclude_full,)
    if not cols:
        available = sorted({c.split("_")[1] for c in df.columns if "_" in c and ":" in c},)
        raise ValueError(f"No '{data_type}' columns found for {cell_line} / {condition}. "
                         f"Data types present in the DataFrame: {available}. "
                         f"Have the limma results been merged in "
                         f"(merge_limma_results in src/transformations.py)?")

    by_tp = {c.split("_")[3]: c for c in cols}
    timepoints = _order_timepoints(by_tp.keys(),)
    return [by_tp[t] for t in timepoints], timepoints


def _responsive_mask(df,
                     cell_line,
                     condition,
                     method="omnibus",
                     fdr_threshold=0.05,
                     min_fc=None,
                     fc_type="log2:limmaFC",
                     fdr_type="log2:FDR",
                     omnibus_type="log2:FFDR",
                     exclude_full=True,
                     ):
    """
    Flag the sites that limma calls responsive in one stimulation condition.

    Two definitions are available. Both ask "does this site respond to this stimulation",
    and differ only in how the timepoints are combined into one call — neither set contains
    the other.

    `method="omnibus"` reads the moderated *F*-test column
    `{cell_line}_{omnibus_type}_{condition}_omnibus`. Upstream, limma fits one linear model
    per site (`log2:abs ~ group + plex`) and tests that condition's timepoint contrasts
    *jointly* against starve — H0 is that the whole time course is flat. Being one test per
    site, BH runs over sites only and the FDR means what it says. It pools evidence across
    timepoints, so a consistent moderate change at several timepoints can pass while every
    individual *t*-test fails; the flip side is that a lone sharp spike is diluted across
    the contrasts and is *less* likely to pass than its own single-timepoint *t*-test. Note
    that `limma_for_pvalues.rmd` builds this F over the stimulation timepoints only, so
    `exclude_full` has no effect on this branch (it still applies to `min_fc`).

    `method="any_timepoint"` calls a site responsive if any single-timepoint moderated
    *t*-test passes. Each contrast is BH-corrected across sites separately, so the OR over
    timepoints has no joint error control: for m timepoints the union's false-positive rate
    lies between the nominal cutoff and ~1-(1-cutoff)^m — closer to the low end, since
    consecutive timepoints are strongly correlated, but not the nominal value. The count
    therefore grows with the size of the timepoint grid and is not comparable across
    datasets with different schedules. What it buys is sensitivity to single-timepoint
    transients. Use it as a diagnostic against the omnibus, not as the reported number.

    `min_fc` adds an effect-size requirement on top of significance: the largest |limmaFC|
    across the included timepoints must reach the threshold. Significance alone says the
    change is reproducible, not that it is large.

    Sites limma did not test (too few plexes) carry NaN and are never counted as
    responsive; they are reported separately through the `tested` mask so that percentages
    can be given against the tested set rather than against rows that never had a chance.

    Args:
        df: DataFrame carrying the merged limma columns.
        cell_line: cell-line prefix, e.g. "WT".
        condition: condition token without underscores, e.g. "EGF".
        method: "omnibus" or "any_timepoint" (see above).
        fdr_threshold: FDR cutoff applied to the chosen statistic.
        min_fc: optional minimum |log2 fold change| at the site's strongest timepoint.
        fc_type: data type of the limma fold-change columns.
        fdr_type: data type of the per-timepoint FDR columns.
        omnibus_type: data type of the omnibus F-test FDR columns.
        exclude_full: if True, ignore the 'full' timepoint everywhere in this call.

    Returns:
        Tuple (responsive, tested): two boolean Series aligned to `df.index`.

    """
    if method not in ("omnibus", "any_timepoint",):
        raise ValueError(f"_responsive_mask: method must be 'omnibus' or 'any_timepoint', "
                         f"got '{method}'.")

    if method == "omnibus":
        col = f"{cell_line}_{omnibus_type}_{condition}_omnibus"
        if col not in df.columns:
            omnibus_cols = [c for c in df.columns if "omnibus" in c]
            raise ValueError(f"Omnibus column '{col}' not found. Omnibus columns present: "
                             f"{omnibus_cols}. Use method='any_timepoint' to call "
                             f"responsiveness from the per-timepoint FDR columns instead.")
        tested = df[col].notna()
        responsive = df[col] < fdr_threshold
    else:
        fdr_cols, _ = _limma_columns(df, cell_line, condition, fdr_type, exclude_full,)
        tested = df[fdr_cols].notna().any(axis=1,)
        responsive = (df[fdr_cols] < fdr_threshold).any(axis=1,)

    if min_fc is not None:
        fc_cols, _ = _limma_columns(df, cell_line, condition, fc_type, exclude_full,)
        responsive = responsive & (df[fc_cols].abs().max(axis=1,) >= min_fc)

    return responsive.fillna(False,).astype(bool,), tested


def limma_peak_timepoints(df,
                          cell_line="WT",
                          conditions=("EGF", "INS", "EGFnINS",),
                          fdr_threshold=0.05,
                          responsive="omnibus",
                          min_fc=None,
                          exclude_full=True,
                          split_direction=False,
                          fc_type="log2:limmaFC",
                          fdr_type="log2:FDR",
                          omnibus_type="log2:FFDR",
                          figsize=(9, 4.5),
                          title="Timepoint of peak response (limma)",
                          ax=None,
                          ):
    """
    When do the responsive sites peak? One bar per timepoint, per stimulation condition.

    For every site the peak is the timepoint with the largest **absolute** limma fold
    change, so a strong dephosphorylation counts as a peak exactly like a strong
    phosphorylation — the question is where the response is strongest, not in which
    direction it goes. `split_direction=True` then splits each bar into up- and
    down-regulated peaks, which is the cheap way to see whether a timepoint is dominated
    by one direction.

    Only sites limma calls responsive in that condition enter the count
    (`responsive=None` counts every tested site instead). This matters: a non-responding
    site still has a largest-|FC| timepoint, driven entirely by noise, and including those
    fills the plot with a flat background that hides the real timing distribution.

    The peak is read off the limma moderated fold changes rather than `log2:FC`, so it is
    the same quantity the p-values were computed on. Ties — exactly equal |FC| at two
    timepoints — go to the earlier timepoint; they are vanishingly rare on continuous data.

    Args:
        df: DataFrame carrying the merged limma columns.
        cell_line: cell-line prefix, e.g. "WT".
        conditions: condition token or list of tokens to plot, e.g. "EGF" or
            ("EGF", "INS", "EGFnINS"). Plotted as grouped bars in the given order.
        fdr_threshold: FDR cutoff defining responsiveness.
        responsive: "omnibus" (one moderated F-test per condition), "any_timepoint" (OR of
            the per-timepoint moderated t-tests), or None to use every site limma tested.
            See `_responsive_mask` for what separates the two — the omnibus is stricter on
            single-timepoint transients, which is exactly the population this plot is
            about, so it is worth checking the timing distribution under both.
        min_fc: optional minimum |log2 fold change| for a site to be counted.
        exclude_full: if True, the 'full' timepoint is excluded both from the peak search
            and from the plot — normally what you want, since 'full' is a separate
            culture condition and not a point on the stimulation time course.
        split_direction: if True, stack each bar into peaks with a positive vs negative
            fold change.
        fc_type: data type of the limma fold-change columns.
        fdr_type: data type of the per-timepoint FDR columns.
        omnibus_type: data type of the omnibus F-test FDR columns.
        figsize: (width, height) of the figure.
        title: figure title.
        ax: optional Axes to draw into; created if None.

    Returns:
        Tuple (peak_table, fig, ax):
            peak_table: timepoints (rows) × conditions (columns) counts of sites peaking
                there, with `{condition}_pct` columns giving the percentage within that
                condition's counted set, and `{condition}_up` / `{condition}_down`
                columns when `split_direction=True`.
            fig, ax: the figure and its Axes.

    """
    if isinstance(conditions, str,):
        conditions = [conditions]
    conditions = list(conditions,)

    counts = {}
    up_counts = {}
    down_counts = {}
    totals = {}
    all_timepoints = []

    for cond in conditions:
        fc_cols, timepoints = _limma_columns(df, cell_line, cond, fc_type, exclude_full,)
        all_timepoints.extend(timepoints,)

        if responsive is None:
            _, keep = _responsive_mask(df,
                                       cell_line,
                                       cond,
                                       method="any_timepoint",
                                       fdr_threshold=fdr_threshold,
                                       fdr_type=fdr_type,
                                       exclude_full=exclude_full,)
        else:
            keep, _ = _responsive_mask(df,
                                       cell_line,
                                       cond,
                                       method=responsive,
                                       fdr_threshold=fdr_threshold,
                                       min_fc=min_fc,
                                       fc_type=fc_type,
                                       fdr_type=fdr_type,
                                       omnibus_type=omnibus_type,
                                       exclude_full=exclude_full,)

        # Rows with no fold change at all have no peak to report.
        sub = df.loc[keep, fc_cols].dropna(how="all",)
        if sub.empty:
            counts[cond] = pd.Series(0, index=timepoints,)
            up_counts[cond] = pd.Series(0, index=timepoints,)
            down_counts[cond] = pd.Series(0, index=timepoints,)
            totals[cond] = 0
            continue

        peak_col = sub.abs().idxmax(axis=1,)
        peak_tp = peak_col.str.split("_",).str[3]
        peak_value = pd.Series([sub.at[i, c] for i, c in peak_col.items()],
                               index=peak_col.index,)

        counts[cond] = peak_tp.value_counts().reindex(timepoints,).fillna(0,).astype(int,)
        up_counts[cond] = (peak_tp[peak_value > 0].value_counts()
                           .reindex(timepoints,).fillna(0,).astype(int,))
        down_counts[cond] = (peak_tp[peak_value <= 0].value_counts()
                             .reindex(timepoints,).fillna(0,).astype(int,))
        totals[cond] = int(len(sub,))

    timepoints = _order_timepoints(all_timepoints,)
    peak_table = pd.DataFrame({c: counts[c].reindex(timepoints,).fillna(0,).astype(int,)
                               for c in conditions},
                              index=timepoints,)
    peak_table.index.name = "timepoint"
    for cond in conditions:
        peak_table[f"{cond}_pct"] = (100 * peak_table[cond] / max(totals[cond], 1,))
        if split_direction:
            peak_table[f"{cond}_up"] = up_counts[cond].reindex(timepoints,).fillna(0,).astype(int,)
            peak_table[f"{cond}_down"] = down_counts[cond].reindex(timepoints,).fillna(0,).astype(int,)

    label = ("all limma-tested sites" if responsive is None
             else f"sites responsive by {responsive} FDR < {fdr_threshold}"
                  + (f" and |log2FC| >= {min_fc}" if min_fc is not None else ""))
    print(f"Timepoint of peak |limma fold change| — {label}:")
    for cond in conditions:
        print(f"  {cond}: {totals[cond]} sites counted")
    print(peak_table.round(1,).to_string())

    # --- Figure ---
    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(figsize=figsize,)
    else:
        fig = ax.get_figure()

    x = np.arange(len(timepoints),)
    width = 0.8 / len(conditions,)
    for i, cond in enumerate(conditions,):
        offset = (i - (len(conditions,) - 1) / 2) * width
        color = _CONDITION_COLORS.get(cond, f"C{i}",)
        if split_direction:
            ax.bar(x + offset,
                   peak_table[f"{cond}_up"],
                   width=width,
                   color=color,
                   edgecolor="white",
                   label=f"{cond} (up)",)
            ax.bar(x + offset,
                   peak_table[f"{cond}_down"],
                   width=width,
                   bottom=peak_table[f"{cond}_up"],
                   color=color,
                   alpha=0.45,
                   edgecolor="white",
                   hatch="//",
                   label=f"{cond} (down)",)
        else:
            ax.bar(x + offset,
                   peak_table[cond],
                   width=width,
                   color=color,
                   edgecolor="white",
                   label=f"{cond} (n={totals[cond]})",)
        for xi, (n, pct) in zip(x + offset,
                                zip(peak_table[cond], peak_table[f"{cond}_pct"],),):
            if n > 0:
                ax.text(xi,
                        n,
                        f"{pct:.0f}%",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=90 if len(conditions,) > 2 else 0,)

    ax.set_xticks(x,)
    ax.set_xticklabels(timepoints,)
    ax.set_xlabel("Timepoint of peak |limma log2 fold change| (min)")
    ax.set_ylabel("Number of sites")
    ax.set_ylim(0, peak_table[conditions].to_numpy().max() * 1.18 or 1,)
    ax.set_title(f"{title}\n{label}", fontsize=10,)
    ax.legend(frameon=False, fontsize=8,)
    ax.grid(axis="y", alpha=0.3,)

    if create_fig:
        fig.tight_layout()
    return peak_table, fig, ax


def limma_responsive_sites(df,
                           cell_line="WT",
                           conditions=("EGF", "INS", "EGFnINS",),
                           fdr_threshold=0.05,
                           method="omnibus",
                           min_fc=None,
                           exclude_full=True,
                           fc_type="log2:limmaFC",
                           fdr_type="log2:FDR",
                           omnibus_type="log2:FFDR",
                           figsize=(11, 4.5),
                           title="Responsive sites (limma)",
                           axes=None,
                           ):
    """
    How many sites respond to each stimulation, and how much of the dataset responds at all.

    The left panel counts the responsive sites per condition plus an 'any condition' bar;
    the right panel counts how many conditions each site responds in (0, 1, 2, 3), so the
    height of the '0' bar is the unresponsive remainder of the dataset and everything to
    its right is the responsive fraction, split by how specific the response is.

    Two denominators are reported and they are not interchangeable. Sites limma could not
    test — detected in too few plexes — carry NaN and can never be called responsive, so a
    percentage taken over all rows understates the response rate by however large that
    untested group is. Both are printed; the plot labels use the percentage of all sites
    and the panel titles state the tested count.

    Args:
        df: DataFrame carrying the merged limma columns.
        cell_line: cell-line prefix, e.g. "WT".
        conditions: condition token or list of tokens, e.g. ("EGF", "INS", "EGFnINS").
        fdr_threshold: FDR cutoff defining responsiveness.
        method: "omnibus" (one moderated F-test per condition, recommended) or
            "any_timepoint" (OR of the per-timepoint moderated t-tests). See
            `_responsive_mask` for what separates them — the choice changes the size of the
            responsive set and, for "any_timepoint", makes it depend on how many timepoints
            the experiment has.
        min_fc: optional minimum |log2 fold change| at the site's strongest timepoint.
        exclude_full: if True, the 'full' timepoint is ignored. This matters for
            method="any_timepoint" (where a 'full' FDR column exists and would count a
            culture-condition difference as a stimulation response) and for `min_fc`; the
            omnibus F was already fitted on the stimulation timepoints only.
        fc_type: data type of the limma fold-change columns.
        fdr_type: data type of the per-timepoint FDR columns.
        omnibus_type: data type of the omnibus F-test FDR columns.
        figsize: (width, height) of the figure.
        title: figure suptitle.
        axes: optional array of two Axes to draw into; created if None.

    Returns:
        Tuple (responsive_table, specificity_table, fig, axes):
            responsive_table: one row per condition plus an 'any condition' row, with the
                number of sites tested, the number responsive, and that number as a
                percentage of all sites and of the tested sites.
            specificity_table: one row per number of conditions a site responds in, with
                counts and percentages of all sites.
            fig, axes: the figure and its two Axes.

    """
    if isinstance(conditions, str,):
        conditions = [conditions]
    conditions = list(conditions,)

    n_total = len(df,)
    masks = {}
    tested = {}
    for cond in conditions:
        masks[cond], tested[cond] = _responsive_mask(df,
                                                     cell_line,
                                                     cond,
                                                     method=method,
                                                     fdr_threshold=fdr_threshold,
                                                     min_fc=min_fc,
                                                     fc_type=fc_type,
                                                     fdr_type=fdr_type,
                                                     omnibus_type=omnibus_type,
                                                     exclude_full=exclude_full,)

    mask_df = pd.DataFrame(masks,)
    tested_df = pd.DataFrame(tested,)
    any_responsive = mask_df.any(axis=1,)
    any_tested = tested_df.any(axis=1,)

    rows = []
    for cond in conditions:
        n_tested = int(tested[cond].sum(),)
        n_resp = int(masks[cond].sum(),)
        rows.append({"condition": cond,
                     "n_tested": n_tested,
                     "n_responsive": n_resp,
                     "pct_of_all": 100 * n_resp / max(n_total, 1,),
                     "pct_of_tested": 100 * n_resp / max(n_tested, 1,),})
    rows.append({"condition": "any condition",
                 "n_tested": int(any_tested.sum(),),
                 "n_responsive": int(any_responsive.sum(),),
                 "pct_of_all": 100 * int(any_responsive.sum(),) / max(n_total, 1,),
                 "pct_of_tested": 100 * int(any_responsive.sum(),)
                                  / max(int(any_tested.sum(),), 1,),})
    responsive_table = pd.DataFrame(rows,).set_index("condition",)

    n_cond_responsive = mask_df.sum(axis=1,)
    spec_counts = n_cond_responsive.value_counts().reindex(range(len(conditions,) + 1),
                                                           ).fillna(0,).astype(int,)
    specificity_table = pd.DataFrame({"n_sites": spec_counts,},)
    specificity_table["pct_of_all"] = 100 * specificity_table["n_sites"] / max(n_total, 1,)
    specificity_table.index.name = "n_conditions_responsive"

    label = (f"{method} FDR < {fdr_threshold}"
             + (f", |log2FC| >= {min_fc}" if min_fc is not None else ""))
    print(f"Responsive sites ({label}) — {n_total} sites in the dataset, "
          f"{int(any_tested.sum(),)} tested by limma:")
    print(responsive_table.round(1,).to_string())
    print(f"\nIn how many conditions is a site responsive:")
    print(specificity_table.round(1,).to_string())

    # --- Figure ---
    create_fig = axes is None
    if create_fig:
        fig, axes = plt.subplots(1, 2, figsize=figsize,)
    else:
        axes = np.atleast_1d(axes,)
        fig = axes[0].get_figure()

    bar_labels = conditions + ["any condition"]
    colors = [_CONDITION_COLORS.get(c, f"C{i}",) for i, c in enumerate(conditions,)] + ["dimgrey"]
    x = np.arange(len(bar_labels,),)
    axes[0].bar(x,
                responsive_table["n_responsive"],
                color=colors,
                edgecolor="white",)
    for xi, (_, row) in zip(x, responsive_table.iterrows(),):
        axes[0].text(xi,
                     row["n_responsive"],
                     f"{int(row['n_responsive'])}\n({row['pct_of_all']:.1f}%)",
                     ha="center",
                     va="bottom",
                     fontsize=8,)
    axes[0].set_xticks(x,)
    axes[0].set_xticklabels(bar_labels,)
    axes[0].set_ylabel("Number of responsive sites")
    axes[0].set_ylim(0, max(responsive_table["n_responsive"].max(), 1,) * 1.25,)
    axes[0].set_title(f"Responsive per condition\n({label}; % of all {n_total} sites)",
                      fontsize=10,)
    axes[0].grid(axis="y", alpha=0.3,)

    xs = np.arange(len(specificity_table,),)
    spec_colors = ["lightgrey"] + ["mediumseagreen"] * (len(specificity_table,) - 1)
    axes[1].bar(xs,
                specificity_table["n_sites"],
                color=spec_colors,
                edgecolor="white",)
    for xi, (_, row) in zip(xs, specificity_table.iterrows(),):
        axes[1].text(xi,
                     row["n_sites"],
                     f"{int(row['n_sites'])}\n({row['pct_of_all']:.1f}%)",
                     ha="center",
                     va="bottom",
                     fontsize=8,)
    axes[1].set_xticks(xs,)
    axes[1].set_xticklabels(specificity_table.index.astype(str,),)
    axes[1].set_xlabel("Number of conditions the site responds in")
    axes[1].set_ylabel("Number of sites")
    axes[1].set_ylim(0, max(specificity_table["n_sites"].max(), 1,) * 1.25,)
    axes[1].set_title(f"Response breadth\n({int(any_responsive.sum(),)} of {n_total} sites "
                      f"= {100 * int(any_responsive.sum(),) / max(n_total, 1,):.1f}% respond "
                      f"to something)",
                      fontsize=10,)
    axes[1].grid(axis="y", alpha=0.3,)

    if create_fig:
        fig.suptitle(title, weight="bold",)
        fig.tight_layout()
    return responsive_table, specificity_table, fig, axes
