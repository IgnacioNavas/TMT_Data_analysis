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


def log2_step_size(df: pd.DataFrame,
                   cell_line: str = "WT",
                   conditions: list = ['_EGF_', '_INS_', '_EGFnINS_'],
                   baseline: str = "starve",
                   exclude_full: bool = True,) -> pd.DataFrame:
    """
    Compute the step size between consecutive timepoints of the log2:FC profile.

    Where log2:FC measures each timepoint against the starve baseline, log2:step
    measures each timepoint against the timepoint immediately before it, i.e. the
    increment of phosphorylation gained (or lost) during that interval:

        log2:step(t_i) = log2:FC(t_i) − log2:FC(t_{i-1})

    with the baseline timepoint ('starve' by default) as t_0. Since
    log2:FC(starve) is identically 0, the first stimulation timepoint reproduces
    its own log2:FC value; the baseline column itself is subtracted explicitly
    rather than assumed to be 0, so the function stays correct if it is ever
    computed against a different reference.

    Step columns exist only for the timepoints AFTER the baseline — there is no
    log2:step for 'full' or for 'starve' itself, since neither has a preceding
    timepoint inside the stimulation series. 'full' is excluded from the chain by
    default (exclude_full=True); it is a separate media control, not the
    timepoint preceding starve, so including it would make the first step a
    full → starve difference rather than a stimulation step.

    Note that the steps are differences over UNEQUAL time intervals (the grid
    {2, 5, 10, 15, 90} min is roughly log-spaced), so a step is an increment per
    interval, not a rate per minute. Divide by the interval width if a rate is
    what is wanted.

    Missing values propagate: if either timepoint of a pair is NaN the step is NaN.

    Output column names:
        {cell_line}_log2:step_{treatment}_{timepoint}

    Args:
        df: DataFrame with log2:FC columns (produced by compute_fold_change).
        cell_line: cell line identifier prefix, e.g. 'WT'.
        conditions: list of condition substrings to match, e.g. ['_EGF_', '_INS_'].
        baseline: timepoint used as the first reference of the chain, default 'starve'.
        exclude_full: if True (default) the 'full' timepoint is left out of the
            chain entirely, so no step column is produced for it and it is never
            used as a previous timepoint.

    Returns:
        Copy of df with the new log2:step columns appended, one per condition per
        timepoint after the baseline, in experimental time order.
    """
    result = df.copy()
    step_cols = {}

    for condition in conditions:
        fc_cols = ColumnSpec.select(result, cell_lines=[cell_line], data_type="log2:FC",
                                    conditions=[condition], exclude_full=exclude_full,)
        if not fc_cols:
            warnings.warn(
                f"log2_step_size: no log2:FC columns found for "
                f"cell_line='{cell_line}', condition='{condition}'."
            )
            continue

        treatment = condition.strip('_')
        timepoints = _sort_timepoints_numeric(
            ColumnSpec.timepoints_from(result, cell_line=cell_line, data_type="log2:FC",
                                       condition=condition,)
        )
        if exclude_full:
            timepoints = [tp for tp in timepoints if tp != "full"]

        if baseline not in timepoints:
            warnings.warn(
                f"log2_step_size: baseline timepoint '{baseline}' not found for "
                f"treatment '{treatment}' — the first available timepoint "
                f"('{timepoints[0]}') is used as the chain reference instead and "
                f"gets no step column."
            )
        else:
            # Chain always starts at the baseline, whatever its position in the sort order.
            timepoints = [baseline] + [tp for tp in timepoints if tp != baseline]

        for previous_tp, timepoint in zip(timepoints[:-1], timepoints[1:]):
            previous_col = f"{cell_line}_log2:FC_{treatment}_{previous_tp}"
            current_col = f"{cell_line}_log2:FC_{treatment}_{timepoint}"
            if previous_col not in result.columns or current_col not in result.columns:
                continue
            step_col = f"{cell_line}_log2:step_{treatment}_{timepoint}"
            step_cols[step_col] = result[current_col] - result[previous_col]

    for col_name, series in step_cols.items():
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


# ---------------------------------------------------------------------------
# diaPASEF (LFQ) transformations
# ---------------------------------------------------------------------------
# Parallel implementation of the transformation chain for the hme1_lfq diaPASEF
# dataset. The functions above were written for a single TMT cell line and are
# kept untouched; these differ in three ways that matter for this dataset:
#
#   1. Timepoint order. _sort_timepoints() sorts against the fixed _TP_ORDER list
#      and appends anything unknown afterwards, so the diaPASEF grid
#      {full, starve, 2, 5, 10, 15, 20, 30, 90} comes out as ... 15, 90, 20, 30.
#      _sort_timepoints_numeric() below sorts the numeric labels by value instead.
#   2. All 8 cell lines are handled in one pass, and each column is assigned to a
#      cell line by its exact first field rather than by str.startswith, so a name
#      that prefixes another (BRAFS151A vs BRAFS151A1/2) cannot pull in both.
#   3. New columns are attached with a single pd.concat per statistic block rather
#      than one insertion at a time — this dataset adds ~1500 columns, where
#      repeated single-column insertion both fragments the frame and is slow.
#
# Chain produced:
#     raw:mean, raw:median, raw:sd, raw:cv
#     log2:abs (zeros treated as NaN)
#     log2:mean, log2:median, log2:sd
#     log2:FC     (vs starve; sites with no starve abundance stay NaN)
#     log2:scaled (amplitude normalisation, per cell line across conditions)
#     log2:zscore (shape normalisation, per cell line per condition)
#
# The two normalisations mirror compute_scaled_fc / compute_zscore_fc but differ in
# one deliberate way: which timepoints define the normalisation basis. In the TMT
# functions the basis is every timepoint present, `full` included — and `full` is a
# different media state, not a response to stimulation, so it inflates the scaling
# denominator and, for the z-score, was measured to carry ~26% of the clustering
# variance on hme1_2 (see clustering_method_decision.md §1). Here the basis is
# explicit and excludes `full` by default (and `starve` as well for the z-score, where
# it is a structural zero). The columns are still written for every timepoint, so
# nothing is lost — only the basis over which they are standardised changes.
# ---------------------------------------------------------------------------

def _split_data_column(col: str) -> dict:
    """
    Split a data-column name into its naming-convention fields.

    Args:
        col: column name, e.g. 'WT_raw:abs_EGF_2_r1' or 'WT_log2:FC_EGF_2'.

    Returns:
        Dict with keys cell_line, data_type, condition, timepoint, replicate
        (replicate is '' for columns without a replicate suffix), or None if the
        name has fewer than the four mandatory fields.
    """
    parts = col.split("_")
    if len(parts) < 4:
        return None
    return {"cell_line": parts[0],
            "data_type": parts[1],
            "condition": parts[2],
            "timepoint": parts[3],
            "replicate": parts[4] if len(parts) > 4 else "",}


def _sort_timepoints_numeric(timepoints) -> list:
    """
    Sort timepoint labels as full, starve, then every numeric label by value.

    Unlike _sort_timepoints() this does not rely on a hard-coded list of known
    timepoints, so timepoint grids that differ between datasets (the diaPASEF grid
    adds 20 and 30 min) still come out in experimental order. Labels that are
    neither named nor numeric are appended alphabetically rather than dropped.

    Args:
        timepoints: iterable of timepoint labels, e.g. ['90', '2', 'full', '20'].

    Returns:
        List of the unique labels in experimental order.
    """
    labels = list(dict.fromkeys(timepoints))
    named = [tp for tp in ("full", "starve",) if tp in labels]
    numeric = sorted([tp for tp in labels
                      if tp not in ("full", "starve",) and str(tp).replace(".", "", 1).isdigit()],
                     key=float)
    other = sorted([tp for tp in labels if tp not in named and tp not in numeric])
    return named + numeric + other


def dia_parse_groups(df: pd.DataFrame,
                     cell_lines: list = None,
                     conditions: list = None,
                     data_type: str = "raw:abs") -> dict:
    """
    Group the replicate columns of every cell line by condition and timepoint.

    Equivalent to parse_columns(replicates=True) but for all cell lines at once,
    matching the cell line on the exact first field of the column name and ordering
    timepoints with _sort_timepoints_numeric.

    Args:
        df: DataFrame following the project naming convention.
        cell_lines: list of cell-line prefixes, e.g. ['WT', 'EGFRT693A']. If None,
            every cell line found among the `data_type` replicate columns is used.
        conditions: list of condition substrings, e.g. ['_EGF_']; the surrounding
            underscores are optional. If None, every condition found is used.
        data_type: DataType field of the input columns, default 'raw:abs'.

    Returns:
        Nested dict {cell_line: {condition: {timepoint: [replicate columns]}}},
        with conditions keyed in the '_EGF_' delimited form used by the rest of
        this module, and timepoints in experimental order.
    """
    # ColumnSpec does the naming-convention selection; the exact-field check below
    # then guards against its startswith() matching of the cell-line prefix.
    all_cells = list(dict.fromkeys(
        info["cell_line"] for info in (_split_data_column(c) for c in df.columns)
        if info is not None and info["data_type"] == data_type and info["replicate"]))
    all_conds = list(dict.fromkeys(
        info["condition"] for info in (_split_data_column(c) for c in df.columns)
        if info is not None and info["data_type"] == data_type and info["replicate"]))

    if cell_lines is None:
        cell_lines = all_cells
    if conditions is None:
        conditions = all_conds
    conditions = [f"_{c.strip('_')}_" for c in conditions]

    groups: dict = {}
    for cell_line in cell_lines:
        cell_groups: dict = {}
        for condition in conditions:
            cols = ColumnSpec.select(df,
                                     cell_lines=[cell_line],
                                     data_type=data_type,
                                     conditions=[condition],)
            timepoint_cols: dict = {}
            for col in cols:
                info = _split_data_column(col)
                if info is None or not info["replicate"]:
                    continue
                if info["cell_line"] != cell_line or info["condition"] != condition.strip("_"):
                    continue
                timepoint_cols.setdefault(info["timepoint"], []).append(col)

            if timepoint_cols:
                cell_groups[condition] = {tp: timepoint_cols[tp]
                                          for tp in _sort_timepoints_numeric(timepoint_cols.keys())}

        if cell_groups:
            groups[cell_line] = cell_groups
        else:
            warnings.warn(f"dia_parse_groups: no '{data_type}' replicate columns found for "
                          f"cell_line='{cell_line}' — skipped.")

    return groups


def dia_compute_raw_stats(df: pd.DataFrame,
                          groups: dict,
                          min_reps: int = 1) -> pd.DataFrame:
    """
    Compute raw:mean, raw:median, raw:sd and raw:cv per cell line × condition × timepoint.

    Zeros are treated as missing alongside NaN, so a site that was not detected in a
    replicate does not drag its mean towards zero. The statistics are taken over the
    detected replicates only.

    Output column order: all means, then all medians, then all sds, then all cvs;
    within each block, cell line, then condition, then timepoint in experimental order.

    Output column names (no replicate suffix):
        {cell_line}_raw:mean_{treatment}_{timepoint}
        {cell_line}_raw:median_{treatment}_{timepoint}
        {cell_line}_raw:sd_{treatment}_{timepoint}     (sample sd, ddof=1 — NaN for 1 replicate)
        {cell_line}_raw:cv_{treatment}_{timepoint}     (sd / |mean| × 100; NaN where mean = 0)

    Args:
        df: DataFrame with raw:abs replicate columns.
        groups: nested dict returned by dia_parse_groups().
        min_reps: minimum number of detected replicates required for a statistic to
            be reported; groups with fewer detections give NaN. Default 1 (report
            whatever was detected).

    Returns:
        Copy of df with the new statistic columns appended.
    """
    means, medians, sds, cvs = {}, {}, {}, {}

    for cell_line, cell_groups in groups.items():
        for condition, timepoints in cell_groups.items():
            treatment = condition.strip("_")
            for timepoint in _sort_timepoints_numeric(timepoints.keys()):
                cols = timepoints[timepoint]
                data = df[cols].replace(0, np.nan)
                enough = data.notna().sum(axis=1) >= min_reps
                pfx = f"{cell_line}_raw"

                mean = data.mean(axis=1, skipna=True).where(enough)
                sd = data.std(axis=1, skipna=True).where(enough)

                cv = sd / mean.abs() * 100
                cv[mean == 0] = np.nan

                means[f"{pfx}:mean_{treatment}_{timepoint}"] = mean
                medians[f"{pfx}:median_{treatment}_{timepoint}"] = data.median(axis=1, skipna=True).where(enough)
                sds[f"{pfx}:sd_{treatment}_{timepoint}"] = sd
                cvs[f"{pfx}:cv_{treatment}_{timepoint}"] = cv

    new_cols = {**means, **medians, **sds, **cvs}
    return pd.concat([df.copy(), pd.DataFrame(new_cols, index=df.index)], axis=1)


def dia_compute_log2_abs(df: pd.DataFrame,
                         groups: dict) -> tuple:
    """
    Compute log2 of every raw:abs replicate column.

    Zeros are replaced with NaN before the transformation: log2(0) is undefined, and a
    zero here means "not detected", not "abundance zero". NaN stays NaN.

    Output column names (replicate suffix retained):
        {cell_line}_log2:abs_{treatment}_{timepoint}_{replicate}

    Args:
        df: DataFrame with raw:abs replicate columns.
        groups: nested dict returned by dia_parse_groups().

    Returns:
        Tuple (updated_df, log2_groups), where log2_groups mirrors the structure of
        `groups` but maps to the new log2:abs column names — pass it to
        dia_compute_log2_stats() and dia_compute_fold_change().
    """
    new_cols = {}
    log2_groups: dict = {}

    for cell_line, cell_groups in groups.items():
        log2_groups[cell_line] = {}
        for condition, timepoints in cell_groups.items():
            treatment = condition.strip("_")
            log2_groups[cell_line][condition] = {}
            for timepoint in _sort_timepoints_numeric(timepoints.keys()):
                tp_cols = []
                for col in timepoints[timepoint]:
                    replicate = col.split("_")[-1]
                    new_col = f"{cell_line}_log2:abs_{treatment}_{timepoint}_{replicate}"
                    new_cols[new_col] = np.log2(df[col].replace(0, np.nan))
                    tp_cols.append(new_col)
                if tp_cols:
                    log2_groups[cell_line][condition][timepoint] = tp_cols

    result = pd.concat([df.copy(), pd.DataFrame(new_cols, index=df.index)], axis=1)
    return result, log2_groups


def dia_compute_log2_stats(df: pd.DataFrame,
                           log2_groups: dict,
                           min_reps: int = 1) -> pd.DataFrame:
    """
    Compute log2:mean, log2:median and log2:sd per cell line × condition × timepoint.

    These are statistics *of the log2 values*, not the log2 of the raw statistics: the
    mean of log2:abs is the log2 geometric mean of the intensities, which is the scale
    the fold changes and all downstream modelling work on.

    Output column order: all means, then all medians, then all sds.

    Output column names (no replicate suffix):
        {cell_line}_log2:mean_{treatment}_{timepoint}
        {cell_line}_log2:median_{treatment}_{timepoint}
        {cell_line}_log2:sd_{treatment}_{timepoint}    (sample sd, ddof=1)

    Args:
        df: DataFrame with log2:abs replicate columns.
        log2_groups: nested dict returned by dia_compute_log2_abs().
        min_reps: minimum number of detected replicates required for a statistic to be
            reported; groups with fewer detections give NaN. Default 1.

    Returns:
        Copy of df with the new log2 statistic columns appended.
    """
    means, medians, sds = {}, {}, {}

    for cell_line, cell_groups in log2_groups.items():
        for condition, timepoints in cell_groups.items():
            treatment = condition.strip("_")
            for timepoint in _sort_timepoints_numeric(timepoints.keys()):
                data = df[timepoints[timepoint]]
                enough = data.notna().sum(axis=1) >= min_reps
                pfx = f"{cell_line}_log2"

                means[f"{pfx}:mean_{treatment}_{timepoint}"] = data.mean(axis=1, skipna=True).where(enough)
                medians[f"{pfx}:median_{treatment}_{timepoint}"] = data.median(axis=1, skipna=True).where(enough)
                sds[f"{pfx}:sd_{treatment}_{timepoint}"] = data.std(axis=1, skipna=True).where(enough)

    new_cols = {**means, **medians, **sds}
    return pd.concat([df.copy(), pd.DataFrame(new_cols, index=df.index)], axis=1)


def dia_compute_fold_change(df: pd.DataFrame,
                            log2_groups: dict,
                            reference: str = "starve",
                            verbose: bool = True) -> pd.DataFrame:
    """
    Compute log2 fold change relative to the starve timepoint, per cell line × condition.

        log2:FC(timepoint) = log2:mean(timepoint) − log2:mean(starve)

    A site with no abundance detected in starve has no reference to divide by, so its
    fold change is undefined: the site is skipped and every FC of that cell line ×
    condition stays NaN for it. This is a per-site, per-cell-line decision — a site can
    have a usable FC in one cell line and none in another — and the counts are reported
    so the loss is visible rather than silent. NaN is used rather than 0 on purpose: 0
    would read as "no change" everywhere downstream.

    log2:FC_{treatment}_starve is computed too and is identically 0 by construction; the
    rest of the project relies on that structural zero.

    Output column names:
        {cell_line}_log2:FC_{treatment}_{timepoint}

    Args:
        df: DataFrame with log2:mean columns (produced by dia_compute_log2_stats).
        log2_groups: nested dict returned by dia_compute_log2_abs().
        reference: timepoint used as the baseline, default 'starve'.
        verbose: if True, print how many sites were skipped per cell line × condition.

    Returns:
        Copy of df with the new log2:FC columns appended.
    """
    fold_changes = {}
    skipped = []

    for cell_line, cell_groups in log2_groups.items():
        for condition, timepoints in cell_groups.items():
            treatment = condition.strip("_")

            ref_col = f"{cell_line}_log2:mean_{treatment}_{reference}"
            if ref_col not in df.columns:
                warnings.warn(f"dia_compute_fold_change: reference column '{ref_col}' not found "
                              f"— skipping FC for {cell_line} / {treatment}.")
                continue

            ref = df[ref_col]
            # Sites with no reference abundance: the subtraction below already leaves
            # them NaN, this only records how many they are.
            skipped.append({"cell_line": cell_line,
                            "condition": treatment,
                            "n_sites": len(df),
                            "n_no_starve": int(ref.isna().sum()),
                            "pct_no_starve": round(ref.isna().sum() / len(df) * 100, 1) if len(df) else 0.0,})

            for timepoint in _sort_timepoints_numeric(timepoints.keys()):
                mean_col = f"{cell_line}_log2:mean_{treatment}_{timepoint}"
                if mean_col not in df.columns:
                    continue
                fold_changes[f"{cell_line}_log2:FC_{treatment}_{timepoint}"] = df[mean_col] - ref

    if verbose and skipped:
        print(f"Sites without a '{reference}' reference (log2:FC left as NaN):")
        for row in skipped:
            print(f"  {row['cell_line']:<14} {row['condition']:<10} "
                  f"{row['n_no_starve']:>7} / {row['n_sites']} ({row['pct_no_starve']:.1f}%)")

    return pd.concat([df.copy(), pd.DataFrame(fold_changes, index=df.index)], axis=1)


def _dia_fc_columns(df: pd.DataFrame,
                    cell_line: str,
                    timepoints: dict,
                    treatment: str,
                    exclude: tuple) -> tuple:
    """
    List the log2:FC columns of one cell line × condition, split into all columns and
    the subset that may define a normalisation basis.

    Args:
        df: DataFrame holding the log2:FC columns.
        cell_line: cell-line prefix, e.g. 'BRAFS151A1'.
        timepoints: {timepoint: [log2:abs columns]} for this cell line × condition,
            i.e. one leaf of the dict returned by dia_compute_log2_abs().
        treatment: condition label without delimiters, e.g. 'EGF'.
        exclude: timepoint labels kept out of the basis, e.g. ('full', 'starve').

    Returns:
        Tuple (all_cols, basis_cols) of column names in experimental order; basis_cols
        is all_cols minus the excluded timepoints. Columns absent from df are dropped
        from both.
    """
    all_cols, basis_cols = [], []
    for timepoint in _sort_timepoints_numeric(timepoints.keys()):
        col = f"{cell_line}_log2:FC_{treatment}_{timepoint}"
        if col not in df.columns:
            continue
        all_cols.append(col)
        if str(timepoint) not in exclude:
            basis_cols.append(col)
    return all_cols, basis_cols


def dia_compute_scaled_fc(df: pd.DataFrame,
                          log2_groups: dict,
                          exclude_from_scale: tuple = ("full",),
                          verbose: bool = True) -> pd.DataFrame:
    """
    Scale the fold changes of each site so its largest stimulation response is ±1.

        log2:scaled(condition, timepoint) = log2:FC(condition, timepoint) / max(|log2:FC|)

    The denominator is taken per site **per cell line, jointly across that cell line's
    conditions**, so the relative amplitude between conditions is preserved (an INS arm
    that responds half as strongly as the EGF arm still reads as half). Only the
    timepoints outside `exclude_from_scale` enter the maximum.

    Why `full` is excluded by default: cells in full media are a different media state,
    not a response to the stimulation, and their |log2:FC| vs starve is routinely the
    largest value in the row. Including it makes the denominator "how different is full
    media from starvation", which squashes the actual response towards zero. The
    log2:scaled column for `full` is still written — it is simply allowed to exceed 1.

    Sites whose basis is all-NaN, or flat at exactly 0, give NaN rather than an infinite
    scale factor.

    Output column names:
        {cell_line}_log2:scaled_{treatment}_{timepoint}

    Args:
        df: DataFrame with log2:FC columns (produced by dia_compute_fold_change).
        log2_groups: nested dict returned by dia_compute_log2_abs().
        exclude_from_scale: timepoint labels kept out of the maximum, default ('full',).
            Pass () to use every timepoint, i.e. the compute_scaled_fc behaviour.
        verbose: if True, print per cell line how many sites got a usable scale factor.

    Returns:
        Copy of df with the new log2:scaled columns appended.
    """
    scaled_cols = {}
    report = []

    for cell_line, cell_groups in log2_groups.items():
        all_cols, basis_cols = [], []
        for condition, timepoints in cell_groups.items():
            cond_all, cond_basis = _dia_fc_columns(df,
                                                   cell_line=cell_line,
                                                   timepoints=timepoints,
                                                   treatment=condition.strip("_"),
                                                   exclude=tuple(exclude_from_scale),)
            all_cols.extend(cond_all)
            basis_cols.extend(cond_basis)

        if not all_cols:
            warnings.warn(f"dia_compute_scaled_fc: no log2:FC columns found for "
                          f"cell_line='{cell_line}' — skipped.")
            continue
        if not basis_cols:
            warnings.warn(f"dia_compute_scaled_fc: every timepoint of '{cell_line}' is in "
                          f"exclude_from_scale={exclude_from_scale} — skipped.")
            continue

        # replace(0, nan): a site flat at exactly 0 has no amplitude to scale by.
        max_abs_fc = df[basis_cols].abs().max(axis=1).replace(0, np.nan)

        for fc_col in all_cols:
            scaled_cols[fc_col.replace("log2:FC", "log2:scaled")] = df[fc_col] / max_abs_fc

        report.append({"cell_line": cell_line,
                       "n_scaled": int(max_abs_fc.notna().sum()),
                       "n_sites": len(df),})

    if verbose and report:
        print(f"log2:scaled — scale factor from timepoints excluding {tuple(exclude_from_scale)}:")
        for row in report:
            print(f"  {row['cell_line']:<14} {row['n_scaled']:>7} / {row['n_sites']} sites scaled")

    return pd.concat([df.copy(), pd.DataFrame(scaled_cols, index=df.index)], axis=1)


def dia_compute_zscore_fc(df: pd.DataFrame,
                          log2_groups: dict,
                          exclude_from_basis: tuple = ("full", "starve",),
                          min_timepoints: int = 3,
                          verbose: bool = True) -> pd.DataFrame:
    """
    Standardise each site's temporal profile, separately per cell line × condition.

        log2:zscore(timepoint) = (log2:FC(timepoint) − mean_t) / sd_t

    where mean_t and the population sd (ddof=0) are taken across the **stimulation**
    timepoints of that condition only — every condition of every cell line standardised
    independently. This removes amplitude, leaving the SHAPE of the response, which is
    what profile clustering should see.

    Two timepoints are excluded from the basis by default:
      - `full`, because full media is a different state rather than a response, and
        standardising over it makes a quarter of the resulting geometry "how different
        is this site in full media" (measured on hme1_2, clustering_method_decision.md §1);
      - `starve`, which is identically 0 in FC space by construction and so contributes
        a constant, not information.
    Both columns are still written — they are just expressed in the units the
    stimulation timepoints define, so log2:zscore at starve reads as "how far the
    baseline sits below the mean response", in SDs.

    Sites with fewer than `min_timepoints` measured basis timepoints, or a flat basis
    (sd = 0), give NaN for that cell line × condition. NaN timepoints are skipped when
    computing the mean and sd.

    Note: z-scoring is invariant to an additive shift, so standardising log2:FC gives the
    same answer as standardising log2:mean — log2:FC is used because the pipeline already
    produced it.

    Output column names:
        {cell_line}_log2:zscore_{treatment}_{timepoint}

    Args:
        df: DataFrame with log2:FC columns (produced by dia_compute_fold_change).
        log2_groups: nested dict returned by dia_compute_log2_abs().
        exclude_from_basis: timepoint labels kept out of the mean/sd, default
            ('full', 'starve'). Pass () for the compute_zscore_fc behaviour.
        min_timepoints: minimum number of non-NaN basis timepoints required, default 3.
            A sd over 1–2 points is not a shape, and the z-scores it produces are ±1 by
            construction rather than by biology.
        verbose: if True, print per cell line × condition how many sites were standardised.

    Returns:
        Copy of df with the new log2:zscore columns appended.
    """
    zscore_cols = {}
    report = []

    for cell_line, cell_groups in log2_groups.items():
        for condition, timepoints in cell_groups.items():
            treatment = condition.strip("_")
            all_cols, basis_cols = _dia_fc_columns(df,
                                                   cell_line=cell_line,
                                                   timepoints=timepoints,
                                                   treatment=treatment,
                                                   exclude=tuple(exclude_from_basis),)

            if not all_cols:
                warnings.warn(f"dia_compute_zscore_fc: no log2:FC columns found for "
                              f"cell_line='{cell_line}', condition='{treatment}' — skipped.")
                continue
            if not basis_cols:
                warnings.warn(f"dia_compute_zscore_fc: every timepoint of '{cell_line}' / "
                              f"'{treatment}' is in exclude_from_basis={exclude_from_basis} "
                              f"— skipped.")
                continue

            basis = df[basis_cols]
            enough = basis.notna().sum(axis=1) >= min_timepoints
            row_mean = basis.mean(axis=1).where(enough)
            row_std = basis.std(axis=1, ddof=0).replace(0, np.nan).where(enough)  # flat -> NaN

            for fc_col in all_cols:
                zscore_cols[fc_col.replace("log2:FC", "log2:zscore")] = (df[fc_col] - row_mean) / row_std

            report.append({"cell_line": cell_line,
                           "condition": treatment,
                           "n_zscored": int(row_std.notna().sum()),
                           "n_sites": len(df),})

    if verbose and report:
        print(f"log2:zscore — basis excludes {tuple(exclude_from_basis)}, "
              f"min_timepoints={min_timepoints}:")
        for row in report:
            print(f"  {row['cell_line']:<14} {row['condition']:<10} "
                  f"{row['n_zscored']:>7} / {row['n_sites']} sites standardised")

    return pd.concat([df.copy(), pd.DataFrame(zscore_cols, index=df.index)], axis=1)


def run_diapasef_transformations(df: pd.DataFrame,
                                 cell_lines: list = None,
                                 conditions: list = None,
                                 data_type: str = "raw:abs",
                                 min_reps: int = 1,
                                 reference: str = "starve",
                                 exclude_from_scale: tuple = ("full",),
                                 exclude_from_zscore_basis: tuple = ("full", "starve",),
                                 min_zscore_timepoints: int = 3,
                                 verbose: bool = True) -> pd.DataFrame:
    """
    Run the diaPASEF transformation chain on all cell lines at once.

    Transformations applied, in order:
        1. raw:mean, raw:median, raw:sd, raw:cv   (zeros treated as missing)
        2. log2:abs                               (per replicate, zeros treated as NaN)
        3. log2:mean, log2:median, log2:sd
        4. log2:FC                                (vs starve; no starve -> site skipped, NaN)
        5. log2:scaled                            (amplitude, per cell line across conditions)
        6. log2:zscore                            (shape, per cell line per condition)

    Steps 5 and 6 normalise the same log2:FC values in two different ways and neither
    reads the other, so both are written and the downstream analysis picks one — they are
    alternative representations, not a sequence. Their normalisation basis excludes `full`
    (and `starve` for the z-score) by default; see dia_compute_scaled_fc /
    dia_compute_zscore_fc for why, and pass () to reproduce the TMT-side behaviour.

    No differential statistics are computed here — those are done downstream in R/limma.

    Args:
        df: DataFrame with raw:abs replicate columns following the naming convention.
        cell_lines: list of cell-line prefixes; None (default) auto-detects all of them.
        conditions: list of condition substrings, e.g. ['_EGF_']; None auto-detects.
        data_type: DataType field of the input columns, default 'raw:abs'.
        min_reps: minimum detected replicates for a statistic to be reported, default 1.
        reference: baseline timepoint for the fold change, default 'starve'.
        exclude_from_scale: timepoints kept out of the log2:scaled denominator,
            default ('full',).
        exclude_from_zscore_basis: timepoints kept out of the log2:zscore mean/sd,
            default ('full', 'starve').
        min_zscore_timepoints: minimum non-NaN basis timepoints for a z-score, default 3.
        verbose: if True, print the per-step column counts and the FC skip report.

    Returns:
        Transformed DataFrame with all new columns appended. The input is not modified.
    """
    groups = dia_parse_groups(df,
                              cell_lines=cell_lines,
                              conditions=conditions,
                              data_type=data_type,)
    if not groups:
        raise ValueError(f"run_diapasef_transformations: no '{data_type}' replicate columns found.")

    if verbose:
        n_groups = sum(len(tps) for cg in groups.values() for tps in cg.values())
        print(f"Found {len(groups)} cell lines, "
              f"{n_groups} (cell line, condition, timepoint) groups.")

    n_before = df.shape[1]

    result = dia_compute_raw_stats(df, groups, min_reps=min_reps,)
    result, log2_groups = dia_compute_log2_abs(result, groups,)
    result = dia_compute_log2_stats(result, log2_groups, min_reps=min_reps,)
    result = dia_compute_fold_change(result, log2_groups, reference=reference, verbose=verbose,)
    result = dia_compute_scaled_fc(result,
                                   log2_groups,
                                   exclude_from_scale=exclude_from_scale,
                                   verbose=verbose,)
    result = dia_compute_zscore_fc(result,
                                   log2_groups,
                                   exclude_from_basis=exclude_from_zscore_basis,
                                   min_timepoints=min_zscore_timepoints,
                                   verbose=verbose,)

    if verbose:
        print(f"\nDone. Columns: {n_before} -> {result.shape[1]} "
              f"(+{result.shape[1] - n_before})")

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

    The statistics themselves are computed in R, which writes one
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
            f"merge_limma_results: '{key}' is not unique in '{limma_path}' — the join would duplicate rows."
        )
    if df[key].duplicated().any():
        warnings.warn(
            f"merge_limma_results: '{key}' is not unique in the dataset — limma statistics will be repeated across the duplicated rows."
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
                f"merge_limma_results: {unmatched_in_limma} site(s) in '{limma_path}' are absent from the dataset and were discarded by the left join."
            )

    return result






