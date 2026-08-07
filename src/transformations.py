"""
Adapted code from claude code

This code will struggle with more than 1 cell line.
"""

from src.column_spec import *

import re
import warnings

import numpy as np
import pandas as pd
# Statistics (p-values / FDR) are no longer computed here — they are done downstream
# with the limma package in R. Imports kept commented for reference.
# from scipy.stats import ttest_ind
# from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
# Canonical timepoint sort order.
_TP_ORDER = ["full", "starve", "1", "2", "5", "10", "15", "90"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sort_timepoints(timepoints: list) -> list:
    """
    Sort a list of timepoint strings in biological order.

    Named timepoints (full, starve) come first according to _TP_ORDER; any numeric timepoints not explicitly listed are
    sorted by integer value and appended.

    Args:
        timepoints: list of timepoint strings (e.g. ['10', 'full', '2', 'starve'])

    Returns:
        Sorted list of timepoint strings.
    """
    known = {tp: i for i, tp in enumerate(_TP_ORDER)} # time points known (defined in _TP_ORDER)
    named = [tp for tp in timepoints if tp in known] # from the time points selected, which ones appear in the known dic
    numeric_extra = [tp for tp in timepoints if tp not in known] # from the time points selected, which ones don't appear in the known dic
    named_sorted = sorted(named, key=lambda x: known[x])
    numeric_sorted = sorted(numeric_extra, key=lambda x: int(x))
    return named_sorted + numeric_sorted

def _new_columns_group(columns_list : list,
                       match : str,
                       replace : str,
                       replicates : bool = True) -> list:
    """
    Takes the column list from ColumnSpec.select and replace the data type by the new one to create the new data type columns

    Args:
        columns_list: list of columns to create new columns for.
        match: what is the data type to replace, e.g. 'raw', 'raw:abs'
        replace: new data type to replace with, e.g. 'log2', 'log2:abs'
        replicates: whether to replicate information or not, e.g. for log2:FC there is no need of replicates information
    Returns:
        list of new columns
    """
    new_columns = []
    for column in columns_list:
        replaced_name = column.replace(match, replace)
        if replicates == False:
            replaced_name = re.sub("_r(\d+)$", "", replaced_name)
        new_columns.append(replaced_name)
    return list(dict.fromkeys(new_columns))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse_columns(df: pd.DataFrame,
                  cell_lines: list = ['WT'],
                  data_type: str = "raw:abs",
                  conditions: list = ['_EGF_', '_INS_', '_EGFnINS_'],
                  replicates: bool = False) -> dict:
    """
    Group columns by cell line and condition, with optional grouping by time point.

    Args:
        df: DataFrame with columns following the naming convention {CellLine}_{DataType}_{Treatment}_{TimePoint}_{Replicate}.
        cell_lines: list of cell line prefixes to match, e.g. ['WT', 'BRAFS151A'].
        data_type: the DataType field to match, e.g. 'raw:abs' or 'log2:abs'.
        conditions: list of condition substrings to match, e.g. ['_EGF_', '_INS_'].
        replicates: if True, adds an extra nesting level grouping columns by time point.

    Returns:
        If replicates=False:
            dict mapping cell -> condition -> list of column names.
            e.g. {'WT': {'_EGF_': ['WT_raw:abs_EGF_starve_r1', ...]}}
        If replicates=True:
            dict mapping cell -> condition -> timepoint -> list of column names.
            e.g. {'WT': {'_EGF_': {'starve': ['WT_raw:abs_EGF_starve_r1', ...], '1': [...]}}}
    """
    if type(df) == pd.DataFrame:
        column_names = df.columns.tolist()
    else:
        column_names = df.index.tolist()
    columns_dir = {cell: {condition: {} for condition in conditions} for cell in cell_lines}
    warned = False  # flag to avoid repeating the warning for every column

    for cell in cell_lines:
        for condition in conditions:
            matched_cols = [
                col for col in column_names
                if col.startswith(cell)
                and condition in col
                and data_type in col
                and "cluster" not in col
            ]

            if replicates:
                timepoint_dict = {}
                for col in matched_cols:
                    parts = col.split('_')
                    if parts[-1].startswith('r') and parts[-1][1:].isdigit():  # checks for pattern like r1, r2
                        timepoint = parts[-2]
                        timepoint_dict.setdefault(timepoint, []).append(col)
                    elif not warned:
                        warnings.warn(
                            f"The selected data_type '{data_type}' does not have individual replicate columns "
                        )
                        warned = True
                columns_dir[cell][condition] = timepoint_dict
            else:
                columns_dir[cell][condition] = matched_cols

    return columns_dir

def check_replicates(df: pd.DataFrame,
                     groups: dict,
                     cell_line: str = "WT",
                     time_point_to_check: str = "starve",
                     reference_condition: str = "_EGF_") -> pd.DataFrame:
    """
    Check amount of replicates in which the site is detected. If needed, the code checks number of replicates per cell line per condition per time point.

    Args:
        df: DataFrame with raw:abs replicate columns.
        replicates: nested dictionary with cell lines, conditions, and time points
        time_point_to_check: where to check the number of replicates. If "all" is given as an input it will check the amount of replicates for all timepoints and conditions.
    Returns:
        Copy of df with the new n_rep columns appended.
        If "all" is given, the data information has the format {CellLine}_{reps}_{Treatment}_{TimePoint}.
    """
    result = df.copy()
    rep_cols = {}

    if time_point_to_check == "all":
        for condition, timepoints in groups[cell_line].items():
            treatment = condition.strip('_')
            for timepoint, cols in timepoints.items():
                rep_col = f"{cell_line}_n:reps_{treatment}_{timepoint}"
                rep_cols[rep_col] = result[cols].notna().sum(axis=1)

    else:
        if reference_condition not in groups[cell_line]:
            warnings.warn(
                f"check_replicates: reference_condition '{reference_condition}' not found "
                f"for cell line '{cell_line}' — skipping."
            )
            return result

        timepoints = groups[cell_line][reference_condition]

        if time_point_to_check not in timepoints:
            warnings.warn(
                f"check_replicates: timepoint '{time_point_to_check}' not found "
                f"for condition '{reference_condition}' — skipping."
            )
            return result

        cols = timepoints[time_point_to_check]
        rep_cols["n:reps"] = result[cols].notna().sum(axis=1)

    for col_name, series in rep_cols.items():
        result[col_name] = series

    return result

def compute_raw_stats(df: pd.DataFrame,
                      groups: dict,
                      cell_line: str = "WT",) -> pd.DataFrame:
    """
    Compute mean, median, standard deviation, and coefficient of variation of raw:abs values, grouped by (treatment, timepoint).

    Missing values (NaN) and zeros are treated as missing.

    Output column order: all means, then all medians, then all sds, then all cvs.
    Within each statistic, columns are ordered by condition then timepoint.

    Output column names (no replicate suffix):
        {cell_line}_raw:mean_{treatment}_{timepoint}
        {cell_line}_raw:median_{treatment}_{timepoint}
        {cell_line}_raw:sd_{treatment}_{timepoint}
        {cell_line}_raw:cv_{treatment}_{timepoint}   (sd / mean × 100; NaN where mean ≤ 0)

    Args:
        df: DataFrame with raw:abs replicate columns.
        groups: nested dict returned by parse_columns(replicates=True),
        cell_line: cell line identifier prefix, e.g. 'WT'.

    Returns:
        Copy of df with the new statistic columns appended.
    """
    result = df.copy()

    # Store computed series in separate dicts before appending
    means   = {}
    medians = {}
    sds     = {}
    cvs     = {}

    for condition, timepoints in groups[cell_line].items():
        treatment = condition.strip('_')
        for timepoint, cols in timepoints.items():
            data = result[cols].replace(0, np.nan)
            pfx  = f"{cell_line}_raw"

            mean_col   = f"{pfx}:mean_{treatment}_{timepoint}"
            median_col = f"{pfx}:median_{treatment}_{timepoint}"
            sd_col     = f"{pfx}:sd_{treatment}_{timepoint}"
            cv_col     = f"{pfx}:cv_{treatment}_{timepoint}"

            means[mean_col]     = data.mean(axis=1, skipna=True)
            medians[median_col] = data.median(axis=1, skipna=True)
            sds[sd_col]         = data.std(axis=1, skipna=True)

            cv = sds[sd_col] / means[mean_col].abs() * 100
            cv[means[mean_col] == 0] = np.nan
            cvs[cv_col] = cv

    # Append in statistic blocks: all means, then medians, then sds, then cvs
    for stat_block in [means, medians, sds, cvs]:
        for col_name, series in stat_block.items():
            result[col_name] = series

    return result

def compute_log2_abs(df: pd.DataFrame,
                     groups: dict,
                     cell_line: str = "WT") -> tuple:
    """
    Compute log2 of each raw:abs replicate column.

    Zero values are treated as missing (log2(0) is undefined) and replaced with NaN before transformation.

    Output column names (replicate suffix retained): {cell_line}_log2:abs_{treatment}_{timepoint}_{replicate}

    Args:
        df: DataFrame with raw:abs replicate columns.
        groups: nested dict returned by parse_columns(replicates=True),
        cell_line: cell line identifier prefix, e.g. 'WT'.

    Returns:
        Tuple (updated_df, log2_groups) where log2_groups has the same
        {cell_line: {condition: {timepoint: [cols]}}} structure as groups
        but maps to the newly created log2:abs column names.
        Pass log2_groups to the log2 statistics functions.
    """
    result = df.copy()
    log2_groups: dict = {cell_line: {}}

    for condition, timepoints in groups[cell_line].items():
        treatment = condition.strip('_')
        log2_groups[cell_line][condition] = {}

        for timepoint, cols in timepoints.items():
            new_cols = []
            for col in cols:
                replicate = col.split('_')[-1]  # e.g. 'r1', 'r2'
                new_col = f"{cell_line}_log2:abs_{treatment}_{timepoint}_{replicate}"
                result[new_col] = np.log2(result[col].replace(0, np.nan))
                new_cols.append(new_col)

            if new_cols:
                log2_groups[cell_line][condition][timepoint] = new_cols

    return result, log2_groups


def compute_log2_stats(df: pd.DataFrame,
                       log2_groups: dict,
                       cell_line: str = "WT",
                       ) -> pd.DataFrame:
    """
    Compute mean, median, and standard deviation of log2:abs replicate columns, grouped by (treatment, timepoint).

    Output column order: all means, then all medians, then all sds.

    Output column names (no replicate suffix): {cell_line}_log2:mean_{treatment}_{timepoint}

    Args:
        df: DataFrame with log2:abs replicate columns.
        log2_groups: nested dict returned by compute_log2_abs(), structured as {cell_line: {condition: {timepoint: [cols]}}}.
        cell_line: cell line identifier prefix, e.g. 'WT'.

    Returns:
        Copy of df with the new log2 statistic columns appended.
    """
    result = df.copy()

    means   = {}
    medians = {}
    sds     = {}

    for condition, timepoints in log2_groups[cell_line].items():
        treatment = condition.strip('_')
        for timepoint, cols in timepoints.items():
            data = result[cols] # sub dataframe with the info of each replicate
            pfx  = f"{cell_line}_log2"

            means  [f"{pfx}:mean_{treatment}_{timepoint}"]   = data.mean(axis=1, skipna=True)
            medians[f"{pfx}:median_{treatment}_{timepoint}"] = data.median(axis=1, skipna=True)
            sds    [f"{pfx}:sd_{treatment}_{timepoint}"]     = data.std(axis=1, skipna=True)

    for stat_block in [means, medians, sds]:
        for col_name, series in stat_block.items():
            result[col_name] = series

    return result


def compute_fold_change(df: pd.DataFrame,
                        log2_groups: dict,
                        cell_line: str = "WT") -> pd.DataFrame:
    """
    Compute log2 fold change relative to the starve timepoint for each treatment.
    log2:FC(timepoint) = log2:mean(timepoint) − log2:mean(starve)

    Output column order: all timepoints per condition in order, then next condition.
    Output column names:  {cell_line}_log2:FC_{treatment}_{timepoint}

    Args:
        df: DataFrame with log2:mean columns (produced by compute_log2_stats).
        log2_groups: nested dict returned by compute_log2_abs(), {cell_line: {condition: {timepoint: [cols]}}}.
        cell_line: cell line identifier prefix, e.g. 'WT'.

    Returns:
        Copy of df with the new log2:FC columns appended.
    """
    result = df.copy()
    fold_changes = {}

    for condition, timepoints in log2_groups[cell_line].items():
        treatment = condition.strip('_')

        starve_col = f"{cell_line}_log2:mean_{treatment}_starve"
        if starve_col not in result.columns:
            warnings.warn(
                f"compute_fold_change: starve reference column '{starve_col}' "
                f"not found — skipping FC for treatment '{treatment}'."
            )
            continue

        for timepoint in _sort_timepoints(timepoints.keys()):
            mean_col = f"{cell_line}_log2:mean_{treatment}_{timepoint}"
            if mean_col not in result.columns:
                continue
            fc_col = f"{cell_line}_log2:FC_{treatment}_{timepoint}"
            fold_changes[fc_col] = result[mean_col] - result[starve_col]

    for col_name, series in fold_changes.items():
        result[col_name] = series

    return result


def compute_scaled_fc(df: pd.DataFrame,
                      cell_line: str = "WT",
                      conditions: list = ['_EGF_', '_INS_', '_EGFnINS_']) -> pd.DataFrame:
    """
    Compute scaled fold change so the maximum absolute FC across all timepoints AND all conditions equals 1 per peptide (and the starve baseline is 0).

    For each peptide:
        log2:scaled(condition, timepoint) = log2:FC(condition, timepoint) / max(|log2:FC|)

    Output column names:
        {cell_line}_log2:scaled_{treatment}_{timepoint}

    Args:
        df: DataFrame with log2:FC columns (produced by compute_fold_change).
        cell_line: cell line identifier prefix, e.g. 'WT'.
        conditions: list of condition substrings to match, e.g. ['_EGF_', '_INS_'].

    Returns:
        Copy of df with the new log2:scaled columns appended.
    """
    result = df.copy()
    scaled_cols = {}

    # Collect ALL FC columns across all conditions at once
    all_fc_cols = ColumnSpec.select(result, cell_lines=[cell_line], data_type="log2:FC", conditions=conditions, exclude_full=False,)
    if not all_fc_cols:
        warnings.warn(f"compute_scaled_fc: no log2:FC columns found for '{cell_line}'.")
        return result

    # Max absolute FC across ALL conditions and timepoints per peptide
    max_abs_fc = result[all_fc_cols].abs().max(axis=1).replace(0, np.nan)

    # Generate scaled column names from FC column names
    all_scaled_names = _new_columns_group(all_fc_cols, match="FC", replace="scaled", replicates=False)

    for fc_col, scaled_col in zip(all_fc_cols, all_scaled_names):
        scaled_cols[scaled_col] = result[fc_col] / max_abs_fc

    for col_name, series in scaled_cols.items():
        result[col_name] = series

    return result


def compute_zscore_fc(df: pd.DataFrame,
                      cell_line: str = "WT",
                      conditions: list = ['_EGF_', '_INS_', '_EGFnINS_'],
                      exclude_full: bool = False) -> pd.DataFrame:
    """
    Compute a per-site z-score of the temporal profile, separately per condition and cell line.

    For each peptide (row) and each (cell_line, condition), the log2:FC values across
    the timepoints of that condition are standardised:

        log2:zscore(condition, timepoint) =
            (log2:FC(timepoint) − mean_t log2:FC) / std_t log2:FC

    where the mean and (population, ddof=0) standard deviation are taken across the
    timepoints of that condition ONLY — so every condition of every cell line is
    normalised independently. This removes amplitude differences between sites,
    leaving only the SHAPE of the temporal response, which is useful for clustering
    profiles by dynamics rather than magnitude.

    Note: z-scoring is invariant to an additive shift, so standardising log2:FC gives
    the same result as standardising log2:mean (log2:FC is log2:mean minus the
    constant starve baseline). log2:FC is used here because it is already produced
    by the pipeline.

    Sites whose profile is flat within a condition (std = 0) yield NaN for that
    condition. NaN timepoints are ignored when computing the mean/std (skipna).

    Output column names:
        {cell_line}_log2:zscore_{treatment}_{timepoint}

    Args:
        df: DataFrame with log2:FC columns (produced by compute_fold_change).
        cell_line: cell line identifier prefix, e.g. 'WT'.
        conditions: list of condition substrings to match, e.g. ['_EGF_', '_INS_'].
        exclude_full: if True, drop the 'full' timepoint from the series before
            computing the z-score (default False, matching compute_scaled_fc).

    Returns:
        Copy of df with the new log2:zscore columns appended.
    """
    result = df.copy()
    zscore_cols = {}

    # Standardise each condition independently (per condition per cell line).
    for condition in conditions:
        fc_cols = ColumnSpec.select(result, cell_lines=[cell_line], data_type="log2:FC",
                                    conditions=[condition], exclude_full=exclude_full,)
        if not fc_cols:
            warnings.warn(
                f"compute_zscore_fc: no log2:FC columns found for "
                f"cell_line='{cell_line}', condition='{condition}'."
            )
            continue

        sub = result[fc_cols]
        row_mean = sub.mean(axis=1)
        row_std = sub.std(axis=1, ddof=0).replace(0, np.nan)  # flat profiles -> NaN

        zscore_names = _new_columns_group(fc_cols, match="FC", replace="zscore", replicates=False)
        for fc_col, zscore_col in zip(fc_cols, zscore_names):
            zscore_cols[zscore_col] = (result[fc_col] - row_mean) / row_std

    for col_name, series in zscore_cols.items():
        result[col_name] = series

    return result


# ---------------------------------------------------------------------------
# DEPRECATED — statistics moved to R/limma (2026-08-06)
# ---------------------------------------------------------------------------
# compute_pvalues / compute_fdr / compute_log10_fdr are no longer part of the
# transformation pipeline. Differential statistics (moderated t / F, BH-FDR) are
# computed downstream with the limma package in R. Kept commented for reference.

# def compute_pvalues(df: pd.DataFrame,
#                     log2_groups: dict,
#                     cell_line: str = "WT") -> pd.DataFrame:
#     """
#     Compute unpaired t-test p-values comparing log2:abs replicates at each timepoint against the starve timepoint, per treatment.
#
#     Uses scipy.stats.ttest_ind with equal_var=False (Welch's t-test).
#     Returns NaN for any (peptide, timepoint) where either group has fewer
#     than 2 valid (non-NaN) replicate values.
#
#     Output column names:
#         {cell_line}_log2:pvalue_{treatment}_{timepoint}
#
#     Args:
#         df: DataFrame with log2:abs replicate columns.
#         log2_groups: nested dict returned by compute_log2_abs(), structured as
#             {cell_line: {condition: {timepoint: [cols]}}}.
#         cell_line: cell line identifier prefix, e.g. 'WT'.
#
#     Returns:
#         Copy of df with the new log2:pvalue columns appended.
#     """
#     result = df.copy()
#     n_rows = len(df)
#     pvalue_cols = {}
#
#     for condition, timepoints in log2_groups[cell_line].items():
#         treatment = condition.strip('_')
#
#         if "starve" not in timepoints:
#             warnings.warn(
#                 f"compute_pvalues: no log2:abs starve columns for treatment "
#                 f"'{treatment}' — skipping p-values."
#             )
#             continue
#
#         starve_vals = result[timepoints["starve"]].values.astype(float)
#
#         for timepoint in _sort_timepoints([tp for tp in timepoints if tp != "starve"]):
#             tp_vals = result[timepoints[timepoint]].values.astype(float)
#             pvalues = np.full(n_rows, np.nan)
#
#             for i in range(n_rows):
#                 sv = starve_vals[i][~np.isnan(starve_vals[i])]
#                 tv = tp_vals[i][~np.isnan(tp_vals[i])]
#                 if len(sv) >= 2 and len(tv) >= 2:
#                     _, p = ttest_ind(tv, sv, equal_var=False)
#                     pvalues[i] = p
#
#             result[f"{cell_line}_log2:pvalue_{treatment}_{timepoint}"] = pvalues
#
#     for col_name, pvalues in pvalue_cols.items():
#         result[col_name] = pvalues
#
#     return result
#
#
# def compute_fdr(df: pd.DataFrame,
#                 cell_line: str = "WT") -> pd.DataFrame:
#     """
#     Apply Benjamini-Hochberg FDR correction to p-values across all peptides
#     for each (treatment, timepoint) comparison.
#
#     NaN p-values (peptides with insufficient replicates) are excluded from the
#     correction and remain NaN in the output.
#
#     Output column order: mirrors the order of the pvalue columns.
#     Output column names:
#         {cell_line}_log2:FDR_{treatment}_{timepoint}
#
#     Args:
#         df: DataFrame with log2:pvalue columns (produced by compute_pvalues).
#         cell_line: cell line identifier prefix, e.g. 'WT'.
#
#     Returns:
#         Copy of df with the new log2:FDR columns appended.
#     """
#     result = df.copy()
#     fdr_cols = {}
#
#     pval_cols = [c for c in result.columns if c.startswith(f"{cell_line}_log2:pvalue_")]
#     for pval_col in pval_cols:
#         fdr_col = pval_col.replace("_log2:pvalue_", "_log2:FDR_")
#
#         pvals = result[pval_col].values.astype(float)
#         valid_mask = ~np.isnan(pvals)
#         fdr_values = np.full(len(pvals), np.nan)
#
#         if valid_mask.sum() > 0:
#             _, fdr_corrected, _, _ = multipletests(pvals[valid_mask], method="fdr_bh")
#             fdr_values[valid_mask] = fdr_corrected
#
#         fdr_cols[fdr_col] = fdr_values
#
#     for col_name, fdr_values in fdr_cols.items():
#         result[col_name] = fdr_values
#
#     return result
#
# def compute_log10_fdr(df: pd.DataFrame,
#                       cell_line: str = "WT" ) -> pd.DataFrame:
#     """
#     Compute -log10(FDR) for each FDR column produced by compute_fdr().
#
#     NaN FDR values remain NaN.
#
#     Output column order: mirrors the order of the FDR columns.
#
#     Output column names:
#         {cell_line}_log2:adjustedFDR_{treatment}_{timepoint}
#
#     Args:
#         df: DataFrame with log2:FDR columns (produced by compute_fdr).
#         cell_line: cell line identifier prefix, e.g. 'WT'.
#
#     Returns:
#         Copy of df with the new -log10(FDR) columns appended.
#     """
#     result = df.copy()
#     log10_cols = {}
#
#     fdr_col_list = [c for c in result.columns if c.startswith(f"{cell_line}_log2:FDR_")]
#     for fdr_col in fdr_col_list:
#         log10_col = fdr_col.replace("_log2:FDR_", "_log2:adjustedFDR_")
#
#         fdr_vals = result[fdr_col].values.astype(float)
#
#         zero_mask = (fdr_vals == 0)
#         if zero_mask.sum() > 0:
#             warnings.warn(
#                 f"compute_log10_fdr: {zero_mask.sum()} zero FDR value(s) in '{fdr_col}' "
#                 f"would produce +inf — setting to NaN."
#             )
#             fdr_vals[zero_mask] = np.nan
#
#         log10_values = np.full(len(fdr_vals), np.nan)
#         valid_mask = ~np.isnan(fdr_vals)
#         log10_values[valid_mask] = -np.log10(fdr_vals[valid_mask])
#
#         log10_cols[log10_col] = log10_values
#
#     for col_name, log10_values in log10_cols.items():
#         result[col_name] = log10_values
#
#     return result


def run_all_transformations(df: pd.DataFrame,
                            cell_lines: list = ['WT'],
                            data_type: str = "raw:abs",
                            conditions: list = ['_EGF_', '_INS_', '_EGFnINS_']) -> pd.DataFrame:
    """
    Run the complete transformation pipeline on a phosphoproteomics dataset,
    iterating over all provided cell lines and accumulating results into a
    single DataFrame.

    Transformations applied per cell line, in order:
        1. n:reps
        2. raw:mean, raw:median, raw:sd, raw:cv
        3. log2:abs (per replicate, zeros treated as NaN)
        4. log2:mean, log2:median, log2:sd
        5. log2:FC  (fold change vs. starve, per treatment)
        6. log2:scaled (max-normalized fold change)
        7. log2:zscore (per-site z-score of the temporal profile, per condition per cell line)

    No statistics (p-values / FDR) are computed here — differential testing is done
    downstream with the limma package in R.

    All cell lines share the same output DataFrame — their derived columns are
    simply appended side-by-side.  Cell lines that have no matching columns are
    skipped with a warning rather than raising an error.

    Args:
        df: Input DataFrame with raw:abs replicate columns following the
            naming convention {CellLine}_{DataType}_{Treatment}_{TimePoint}_{Replicate}.
        cell_lines: List of cell line identifiers used as column prefixes,
            e.g. ['WT', 'BRAFS151A'].
        data_type: DataType field for the raw abundance columns, default 'raw:abs'.
        conditions: List of condition substrings to match, e.g. ['_EGF_', '_INS_'].

    Returns:
        Transformed DataFrame with all new columns for every cell line appended.
        The original DataFrame is not modified.
    """
    result = df.copy()
    original_ncols = df.shape[1]

    for cell_line in cell_lines:
        groups = parse_columns(result, cell_lines=[cell_line], data_type=data_type, conditions=conditions,
                               replicates=True)

        if not any(groups[cell_line].values()):
            warnings.warn(
                f"run_all_transformations: no '{data_type}' columns found for "
                f"cell_line='{cell_line}' — skipping."
            )
            continue

        n_groups = sum(len(tps) for tps in groups[cell_line].values())
        print(f"[{cell_line}] Found {len(groups[cell_line])} conditions, "
              f"{n_groups} (treatment, timepoint) groups.")

        cols_before = result.shape[1]

        result = check_replicates(result, groups, cell_line=cell_line, time_point_to_check="starve", reference_condition="_EGF_")
        result = compute_raw_stats(result, groups, cell_line=cell_line)
        result, log2_groups = compute_log2_abs(result, groups, cell_line=cell_line)
        result = compute_log2_stats(result, log2_groups, cell_line=cell_line)
        result = compute_fold_change(result, log2_groups, cell_line=cell_line)
        result = compute_scaled_fc(result, cell_line=cell_line, conditions=conditions)
        result = compute_zscore_fc(result, cell_line=cell_line, conditions=conditions)
        # Statistics removed — p-values / FDR are computed downstream with limma (R):
        # result = compute_pvalues(result, log2_groups, cell_line=cell_line)
        # result = compute_fdr(result, cell_line=cell_line)
        # result = compute_log10_fdr(result, cell_line=cell_line)

        print(f"[{cell_line}] Done. Added {result.shape[1] - cols_before} columns.")

    print(f"\nAll cell lines processed. Total columns: {original_ncols} -> {result.shape[1]}")
    return result

#----------------------
# Merging PhosphoSitePlus data
#----------------------

def get_average_score(protein_id: str,
                      localized_phsopho: str,
                      phospho_lookup: dict):
    """
    Extract all individual sites, look up their functional scores,
    and return the average (ignoring NaN/NA values).
    Args:
        protein_id : str (uniprot ID)
        localized_phsopho : str (residues with localized phosphorylations)
        phospho_lookup : dict
    Returns:
        funtional_score value if 1 phosphorylated site was localized
        Average functional_score if more than one phosphorylated site was localized
        NaN if no phsopho was localized

    """
    # Extract all individual sites from the string e.g. "S23T26" -> ["S23", "T26"]
    sites = re.findall(r'[A-Z]\d+', localized_phsopho) if isinstance(localized_phsopho, str) else []

    if not sites:
        return float("nan")

    scores = []
    for site in sites:
        score = phospho_lookup.get((protein_id, site), float("nan"))
        try:
            scores.append(float(score))
        except (ValueError, TypeError):
            pass  # skip "NA" strings and actual NaNs

    return float("nan") if not scores else sum(scores) / len(scores)

def merge_functional_score(df: pd.DataFrame,
                           phosphosite_df: pd.DataFrame,
                           ph_residue: str,
                           ph_position: str,) -> pd.DataFrame:
    """
    Match phosphosites from experimental data (df) with phosphoPlus annotations (phosphosite_df).

    Args:
        df : pd.DataFrame
        phosphosite_df : pd.DataFrame
        ph_residue : str
            Column name in phosphosite_df for the amino acid residue (e.g. 'aa').
        ph_position : str
            Column name in phosphosite_df for the sequence position (e.g. 'prot_seq_position').

    Returns
        pd.DataFrame: with functional score added
    """
    result = df.copy()
    info_df = phosphosite_df.copy()

    # Build the "info_column" key in phosphoPlus: combine residue + position (e.g. "S12")
    if ph_residue == ph_position:
        info_df["info_column"] = info_df[ph_residue].astype(str)
    else:
        # Fix: use + operator for Series string concatenation, not "".join()
        info_df["info_column"] = info_df[ph_residue].astype(str) + info_df[ph_position].astype(str)

    # Build a lookup dictionary from PhosphoPlus dataset
    phospho_lookup = info_df.set_index(["protein_Id", "info_column"])["functional_score"].to_dict()

    # Extract the localization label part form the site column
    if "~" in result["site"].iloc[0]:
        result["_loc_label"] = (result["site"].str.split("~").str[0].str.split("_").str[-1])
    else:
        result["_loc_label"] = (result["site"].str.split("_").str[-1])

    # Apply the lookup dictionary and the averaging fucntion
    result["functional_score"] = result.apply(lambda row: get_average_score(row["protein_Id"], row["_loc_label"], phospho_lookup), axis=1)

    result = result.drop(columns=["_loc_label"])

    return result

def get_column_infos(protein_id: str,
                     site_str: str,
                     phospho_lookup: dict) -> str:
    """
        Extract all individual sites, look up their info values,
        and return them concatenated with '|'.
        Args:
            protein_id : str (uniprot ID)
            site_str : str (residues with localized phosphorylations)
            phospho_lookup : dict {(protein_id, site): info_value}
        Returns:
            str with info values concatenated with '|'
            NaN if no phospho was localized
        """

    # Extract all individual sites from the string e.g. "S23T26" -> ["S23", "T26"]
    sites = re.findall(r'[A-Z]\d+', site_str) if isinstance(site_str, str) else []

    if not sites:
        return float("nan")

    infos = []
    for site in sites:
        value = phospho_lookup.get((protein_id, site), None)
        if value is not None and not (isinstance(value, float) and pd.isna(value)):
            infos.append(str(value))

    return float("nan") if not infos else "|".join(infos)


def merge_phosphoplus_info(df: pd.DataFrame,
                           phosphosite_df: pd.DataFrame,
                           ph_residue: str,
                           ph_position: str,
                           adding_info: list,
                           regulatory_sites: bool = False) -> pd.DataFrame:
    """
    Add extra info columns from phosphoPlus to experimental data.
    If a site has multiple localizations, values are concatenated with '|'.
    Args:
        df : pd.DataFrame
        phosphosite_df : pd.DataFrame
        ph_residue : str
        ph_position : str
        adding_info : list of column names to add from phosphosite_df
    Returns:
        pd.DataFrame with new info columns added
    """
    result = df.copy()
    info_df = phosphosite_df.copy()

    # Build the "info_column" key in phosphoPlus: combine residue + position (e.g. "S12")
    if regulatory_sites == True:
        info_df["info_column"] = info_df["MOD_RSD"].str.split("-").str[0]
    else:
        if ph_residue == ph_position:
            info_df["info_column"] = info_df[ph_residue].astype(str)
        else:
            # Fix: use + operator for Series string concatenation, not "".join()
            info_df["info_column"] = info_df[ph_residue].astype(str) + info_df[ph_position].astype(str)

    # Extract the localization label part form the site column
    if "~" in result["site"].iloc[0]:
        result["_loc_label"] = (result["site"].str.split("~").str[0].str.split("_").str[-1])
    else:
        result["_loc_label"] = (result["site"].str.split("_").str[-1])

    for info in adding_info:
        # Build a lookup dictionary from PhosphoPlus dataset
        phospho_lookup = info_df.set_index(["protein_Id", "info_column"])[info].to_dict()

        result[info] = result.apply(lambda row: get_column_infos(row["protein_Id"], row["_loc_label"], phospho_lookup), axis=1)

    result = result.drop(columns=["_loc_label"])
    return result


#----------------------
# Merging limma statistics
#----------------------

def merge_limma_results(df: pd.DataFrame,
                        limma_path: str,
                        key: str = "site",
                        verbose: bool = True) -> pd.DataFrame:
    """
    Merge the limma statistics table into a transformed dataset.

    The statistics themselves are computed in R by
    notebooks/01_preprocessing/limma_for_pvalues.rmd, which writes one
    {dataset}_limma_pvalues.tsv per dataset keyed by `site`. This function joins that
    table back onto the Python frame so the rest of the pipeline can use the p-values.

    The merge is a left join on `key`, so every row of df is kept. Sites that limma did not
    test (those below its MIN_PLEX threshold, i.e. detected in too few TMT plexes to have a
    variance estimate) receive NaN rather than being dropped.

    Re-running the merge is safe: any limma column already present in df is removed first,
    so the join never produces _x / _y suffixed duplicates.

    Args:
        df: transformed dataset containing the key column.
        limma_path: path to the {dataset}_limma_pvalues.tsv written by the R notebook.
        key: column to join on, default 'site'.
        verbose: if True, print how many sites carry statistics after the merge.

    Returns:
        Copy of df with the limma columns appended. The original DataFrame is not modified.
    """
    limma_df = pd.read_csv(limma_path, sep="\t", low_memory=False)

    if key not in df.columns:
        raise KeyError(f"merge_limma_results: '{key}' not found in the dataset columns.")
    if key not in limma_df.columns:
        raise KeyError(f"merge_limma_results: '{key}' not found in '{limma_path}'.")

    if limma_df[key].duplicated().any():
        raise ValueError(
            f"merge_limma_results: '{key}' is not unique in '{limma_path}' — "
            f"the join would duplicate rows."
        )
    if df[key].duplicated().any():
        warnings.warn(
            f"merge_limma_results: '{key}' is not unique in the dataset — "
            f"limma statistics will be repeated across the duplicated rows."
        )

    stat_cols = [col for col in limma_df.columns if col != key]

    # Drop any previous merge so re-running does not create _x / _y suffixes.
    already_present = [col for col in stat_cols if col in df.columns]
    result = df.drop(columns=already_present) if already_present else df.copy()
    if already_present and verbose:
        print(f"  replacing {len(already_present)} limma column(s) from a previous merge")

    result = result.merge(limma_df, on=key, how="left")

    if verbose:
        matched = result[stat_cols[0]].notna().sum()
        unmatched_in_limma = (~limma_df[key].isin(df[key])).sum()
        print(f"  merged {len(stat_cols)} limma columns from {limma_path}")
        print(f"  {matched} / {len(result)} sites carry statistics "
              f"({len(result) - matched} not tested by limma)")
        if unmatched_in_limma:
            warnings.warn(
                f"merge_limma_results: {unmatched_in_limma} site(s) in '{limma_path}' "
                f"are absent from the dataset and were discarded by the left join."
            )

    return result






