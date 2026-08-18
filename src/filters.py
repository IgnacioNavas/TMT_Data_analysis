"""
Filtering functions for phosphoproteomics DataFrames following the project
naming convention: {CellLine}_{DataType}:{subtype}_{Condition}_{TimePoint}_{Replicate}.

All functions are compatible with *_transformed.tsv files in data/ and accept
the standard metadata columns (site, n:reps, CON, REV, protein_name, protein_Id).

Use ColumnSpec.select() for column selection — never build column lists by hand.
"""

import pandas as pd

from src.column_spec import ColumnSpec


def filter_by_nreps(df: pd.DataFrame,
                    min_reps: int,
                    nreps_col: str = "n:reps") -> pd.DataFrame:
    """
    Keep rows detected in at least `min_reps` replicates.

    Args:
        df: DataFrame with an n:reps column (or equivalent).
        min_reps: minimum number of replicates required.
        nreps_col: name of the replicate-count column (default 'n:reps').

    Returns:
        Filtered DataFrame copy.
    """
    if nreps_col not in df.columns:
        raise ValueError(f"Column '{nreps_col}' not found in DataFrame.")
    return df.loc[df[nreps_col] >= min_reps].copy()


def filter_contaminants(
    df: pd.DataFrame,
    remove_contaminants: bool = True,
    remove_reverse: bool = True,
    con_col: str = "CON",
    rev_col: str = "REV",
) -> pd.DataFrame:
    """
    Remove contaminant and reverse-sequence decoy rows.

    Rows are dropped when the flag column is truthy (non-zero, non-NaN, non-empty string).
    Columns absent from df are silently ignored so the function works on both TMT
    and LFQ datasets regardless of which flag columns they carry.

    Args:
        df: DataFrame with CON and/or REV flag columns.
        remove_contaminants: drop contaminant rows when True (default True).
        remove_reverse: drop reverse/decoy rows when True (default True).
        con_col: contaminant flag column name (default 'CON').
        rev_col: reverse/decoy flag column name (default 'REV').

    Returns:
        Filtered DataFrame copy.
    """
    mask = pd.Series(True, index=df.index)
    for flag_col, should_remove in [(con_col, remove_contaminants), (rev_col, remove_reverse)]:
        if should_remove and flag_col in df.columns:
            is_flagged = df[flag_col].notna() & (df[flag_col] != 0) & (df[flag_col] != "")
            mask &= ~is_flagged
    return df.loc[mask].copy()


def filter_dynamics(
    df: pd.DataFrame,
    cell_lines: list,
    conditions: list,
    data_type: str = "log2:FC",
    threshold: float = 0.5,
    mode: str = "extremes",
    exclude_full: bool = True,
) -> pd.DataFrame:
    """
    Filter rows by the maximum absolute value across selected time-series columns.

    Uses ColumnSpec.select() to resolve columns, so it works with the standard
    {CellLine}_{DataType}_{Condition}_{TimePoint} naming convention.

    Args:
        df: DataFrame following the project naming convention.
        cell_lines: list of cell-line prefixes, e.g. ["WT"].
        conditions: list of condition substrings, e.g. ["_EGF_", "_INS_", "_EGFnINS_"].
        data_type: data-type string, e.g. "log2:FC" or "log2:scaled" (default "log2:FC").
        threshold: absolute value cutoff.
        mode: 'extremes' keeps rows where max|value| >= threshold (dynamic sites);
              'within' keeps rows where max|value| <= threshold (stable sites).
        exclude_full: if True, exclude the 'full' timepoint from the comparison (default True).

    Returns:
        Filtered DataFrame copy.
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

    max_abs = df[cols].abs().max(axis=1)
    if mode == "extremes":
        mask = max_abs >= threshold
    elif mode == "within":
        mask = max_abs <= threshold
    else:
        raise ValueError(f"mode must be 'extremes' or 'within', got {mode!r}")
    return df.loc[mask].copy()


def get_dynamics_columns(
    df: pd.DataFrame,
    cell_lines: list,
    conditions: list,
    data_type: str = "log2:FC",
    exclude_full: bool = True,
) -> pd.DataFrame:
    """
    Return a DataFrame containing only the time-series value columns for the given spec.

    Useful for passing data directly to clustering or dimensionality-reduction functions.

    Args:
        df: DataFrame following the project naming convention.
        cell_lines: list of cell-line prefixes, e.g. ["WT"].
        conditions: list of condition substrings, e.g. ["_EGF_", "_INS_", "_EGFnINS_"].
        data_type: data-type string, e.g. "log2:FC" (default "log2:FC").
        exclude_full: if True, exclude the 'full' timepoint (default True).

    Returns:
        DataFrame with only the selected data columns (index preserved from df).
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
            f"data_type={data_type!r}."
        )
    return df[cols].copy()


def filter_incomplete_timeseries(
    df: pd.DataFrame,
    cell_lines: list,
    conditions: list,
    data_type: str = "log2:FC",
    exclude_full: bool = True,
    max_missing: int = 0,
) -> pd.DataFrame:
    """
    Remove sites with missing (NaN) values in the time-series columns used for clustering.

    Clustering (e.g. KMeans) needs a complete numeric vector per site: a single NaN in any
    timepoint of the selected series makes the Euclidean distance undefined and raises
    "Input contains NaN". Such holes come from sites that were not quantified at some
    timepoint (typically the sparse n:reps==1 tail of TMT data, where a peptide has no
    reporter intensity in any replicate at a given timepoint). Columns are resolved with
    ColumnSpec.select(), so the check targets exactly the columns the clustering will use.

    This is the alternative to the `.fillna(0)` convention: instead of imputing the missing
    timepoints as "no change vs starve", drop the affected sites entirely.

    Args:
        df: DataFrame following the project naming convention.
        cell_lines: list of cell-line prefixes, e.g. ["WT"].
        conditions: list of condition substrings, e.g. ["_EGF_", "_INS_", "_EGFnINS_"].
        data_type: data-type string of the clustering values (default "log2:FC").
        exclude_full: if True, exclude the 'full' timepoint from the check (default True);
            set this to match how the clustering selects its columns.
        max_missing: maximum number of missing timepoints tolerated per row (default 0,
            i.e. keep only rows with a fully complete series). Rows with more than
            `max_missing` NaNs across the selected columns are dropped.

    Returns:
        Filtered DataFrame copy (rows whose selected time series has <= max_missing NaNs).
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
    n_missing = df[cols].isna().sum(axis=1)
    return df.loc[n_missing <= max_missing].copy()


def filter_by_ffdr(df: pd.DataFrame,
                   cell_lines: list,
                   conditions: list,
                   max_ffdr: float = 0.05,
                   omnibus_type: str = "log2:FFDR",
                   combine: str = "any",
                   verbose: bool = False,
                   ) -> pd.DataFrame:
    """
    Keep the sites limma's omnibus F-test calls responsive.

    The omnibus F, computed in `notebooks/01_preprocessing/limma_for_pvalues.rmd` and merged
    onto the processed tables, tests one condition's timepoint contrasts *jointly* against
    starve: H0 is that the whole time course is flat. It is one test per site per condition,
    so the FDR needs no further correction across timepoints.

    This is the reproducibility criterion `filter_dynamics` cannot give. An amplitude cutoff
    asks only "did this site move"; the F-test asks "did it move more than its own replicate
    noise", which is a different question and keeps different sites — a 0.6 log2FC seen once
    is not the evidence a 0.6 log2FC seen in four plexes is.

    Columns are built from the project naming convention as
    `{cell_line}_{omnibus_type}_{condition}_omnibus`, e.g. `WT_log2:FFDR_EGF_omnibus`
    (`ColumnSpec.select` is not used here: the omnibus columns carry the literal timepoint
    field 'omnibus' rather than a real timepoint, so there is nothing to select over).
    Passing `conditions=["_ALL_"]` resolves to `{cell_line}_log2:FFDR_ALL_omnibus`, the single
    F over every stimulation contrast of every condition — the stricter alternative to OR-ing
    the per-condition tests when several conditions are clustered jointly.

    ⚠ Sites limma did not test (detected in too few plexes) carry NaN and are always dropped,
    since `NaN < max_ffdr` is False. This is only true if the statistics columns really are
    NaN: a `.fillna(0)` applied to the whole DataFrame on load turns an untested site into
    FFDR = 0.0, i.e. perfectly significant, and every untested site then survives this filter.
    Fill the data columns only.

    Args:
        df: DataFrame with the merged limma omnibus columns.
        cell_lines: list of cell-line prefixes, e.g. ["WT"].
        conditions: list of condition substrings, e.g. ["_EGF_", "_INS_"]; surrounding
            underscores are optional. Use ["_ALL_"] for the joint omnibus column.
        max_ffdr: FDR cutoff; None returns the DataFrame unfiltered, so a notebook can switch
            the step off by setting its parameter to None without an `if` around the call.
        omnibus_type: data type of the omnibus FDR columns (default "log2:FFDR"). Use
            "log2:Fpvalue" to cut on the raw p-value instead of the corrected one.
        combine: 'any' keeps a site responding in at least one of the requested conditions
            (the union — what you want when clustering several conditions together);
            'all' keeps only sites responding in every one of them.
        verbose: print how many sites were kept, dropped, and never tested.

    Returns:
        Filtered DataFrame copy.

    """
    if max_ffdr is None:
        return df.copy()

    if combine not in ("any", "all",):
        raise ValueError(f"combine must be 'any' or 'all', got {combine!r}")

    cols = [f"{cell_line}_{omnibus_type}_{condition.strip('_')}_omnibus"
            for cell_line in cell_lines
            for condition in conditions]

    missing = [c for c in cols if c not in df.columns]
    if missing:
        present = [c for c in df.columns if "omnibus" in c]
        raise KeyError(f"limma omnibus columns not found in the dataset: {missing}. "
                       f"Omnibus columns present: {present}. Run limma_for_pvalues.rmd and "
                       f"the merge cell in TMT_dataset_preprocessing.ipynb.")

    significant = df[cols] < max_ffdr
    mask = significant.any(axis=1,) if combine == "any" else significant.all(axis=1,)

    if verbose:
        n_untested = int(df[cols].isna().all(axis=1,).sum(),)
        print(f"FFDR < {max_ffdr} ({combine} of {', '.join(cols)}): "
              f"{int(mask.sum(),)} of {len(df,)} sites kept "
              f"({int((~mask).sum(),)} dropped, of which {n_untested} never tested by limma)")

    return df.loc[mask].copy()


def filter_by_localization(
    df: pd.DataFrame,
    min_localized: int = 1,
    loc_col: str = "n_localized",
) -> pd.DataFrame:
    """
    Filter rows by phosphosite localization confidence (LFQ/mutant datasets).

    Args:
        df: DataFrame with a localization count column.
        min_localized: minimum number of localized sites required (default 1,
                       i.e. keep only rows with at least one confidently localized site).
        loc_col: localization count column name (default 'n_localized', 'LocalizedNumPhos',
                 as produced by the LFQ preprocessing pipeline).

    Returns:
        Filtered DataFrame copy.
    """
    if loc_col not in df.columns:
        raise ValueError(
            f"Column '{loc_col}' not found. LFQ/mutant datasets use 'n_localized'; "
            f"check the column names in your dataset."
        )
    return df.loc[df[loc_col] >= min_localized].copy()


def filter_by_protein(
    df: pd.DataFrame,
    proteins,
    match_col: str = None,
) -> pd.DataFrame:
    """
    Keep rows belonging to one or more proteins.

    Matches against protein_name and protein_Id by default; restrict to one column
    via match_col if needed.

    Args:
        df: DataFrame with protein_name and/or protein_Id columns.
        proteins: string or list of strings matched against protein identifiers.
        match_col: if provided, restrict matching to this single column name.

    Returns:
        Filtered DataFrame copy.
    """
    if isinstance(proteins, str):
        proteins = [proteins]
    proteins = list(proteins)

    if match_col is not None:
        if match_col not in df.columns:
            raise ValueError(f"Column '{match_col}' not found in DataFrame.")
        mask = df[match_col].isin(proteins)
    else:
        mask = pd.Series(False, index=df.index)
        for col in ("protein_name", "protein_Id"):
            if col in df.columns:
                mask |= df[col].isin(proteins)

    found = set()
    for col in ("protein_name", "protein_Id"):
        if col in df.columns:
            found |= set(df.loc[mask, col].dropna())
    missing = set(proteins) - found
    if missing:
        print(f"Warning: proteins not found in dataset: {sorted(missing)}")

    return df.loc[mask].copy()


def filter_by_site(df: pd.DataFrame, sites, site_col: str = "site") -> pd.DataFrame:
    """
    Keep rows matching one or more phosphosite identifiers.

    Args:
        df: DataFrame with a site identifier column.
        sites: string or list of site identifiers (e.g. 'EGFR_HUMAN-Y1068y').
        site_col: name of the site column (default 'site').

    Returns:
        Filtered DataFrame copy.
    """
    if site_col not in df.columns:
        raise ValueError(f"Column '{site_col}' not found in DataFrame.")
    if isinstance(sites, str):
        sites = [sites]
    sites = list(sites)
    mask = df[site_col].isin(sites)
    missing = set(sites) - set(df.loc[mask, site_col])
    if missing:
        print(f"Warning: sites not found in dataset: {sorted(missing)}")
    return df.loc[mask].copy()
