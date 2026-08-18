"""
LFQ pre-treatment: modification annotation, quality filtering, and cross-dataset matching.

Module layout
─────────────
1. Internal helpers      — private utilities used by the public functions
2. LFQ annotation        — add_modification_metadata()
3. LFQ filtering         — filter_no_ptm(), filter_not_detected_in_starve(),
                           filter_missing_replicates()
4. LFQ pipeline          — run_all_pretreatment()  ← main entry point for LFQ data
5. Cross-dataset matching — add_new_site_column()  (LFQ simplified key)
                            add_site_matching_column()  (TMT site reconstruction)

Typical workflow
────────────────
    df = pd.read_csv("data/my_lfq_dataset.tsv", sep="\\t")

    # Steps 1–4: annotate + filter
    df = run_all_pretreatment(df, cell_lines=["WT", "BRAFS151A"], conditions=["_EGF_"])

    # Log2 transformations (handled by src.transformations)
    df = run_all_transformations(df, cell_lines=["WT", "BRAFS151A"], conditions=["_EGF_"])

    # Step 5 (optional): add simplified key for TMT/LFQ cross-dataset comparison
    df = add_new_site_column(df)
"""
import re
import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Internal helpers
# ---------------------------------------------------------------------------
def _count_sty_mods(mod_str: str) -> int:
    """Count comma-separated modification items that contain any S, T, or Y character."""
    if not isinstance(mod_str, str) or not mod_str:
        return 0
    return sum(any(c in item for c in "STY") for item in mod_str.split(","))


def _extract_sty_positions(mod_str: str) -> str: #mod_str = peptide_index
    """
    Return only the S, T, and Y residue-position pairs from a compact modification string.

    Filters out non-phosphorylatable modifications such as oxidised methionine (M),
    N-terminal acetylation (N-term), and cysteine carbamidomethylation (C), keeping only
    the residues that carry phosphorylation-relevant information.

    Args:
        mod_str: compact modification string produced by _offset_mod_positions,
                 e.g. 'N-termM15S261T300'.

    Returns:
        Concatenated STY pairs, e.g. 'S261T300', or None when no STY residues are present.

    Examples:
        'S479S484'       → 'S479S484'  (all STY, unchanged)
        'N-termM15S261'  → 'S261'
        'N-termM15M21'   → None        (no STY modifications)
    """
    if not isinstance(mod_str, str) or not mod_str:
        return None
    # matches = matches = re.findall(r"[STY]\d+", mod_str.split("_")[-1])
    matches = re.findall(r"[STY]\d+", mod_str)
    return "".join(matches) if matches else None

def _get_phosphosite_sequene(mod_str: str):
    if not isinstance(mod_str, str) or not mod_str:
        return 0
    sequence = mod_str.replace("S(UniMod:21)", "s")
    sequence = sequence.replace("T(UniMod:21)", "t")
    sequence = sequence.replace("Y(UniMod:21)", "y")
    sequence = sequence.replace("M(UniMod:35)", "M")
    sequence = sequence[:-1] # remove last number of the se quence
    return sequence

def _get_raw_columns(df: pd.DataFrame, cell_lines: list, conditions: list, data_type: str) -> list:
    """Return all raw intensity columns matching the given cell lines, conditions, and data type."""
    cols = df.columns.tolist()
    return [c for c in cols
            if any(c.startswith(cl) for cl in cell_lines)
            and any(cond in c for cond in conditions)
            and data_type in c
    ]

# ---------------------------------------------------------------------------
# 2. LFQ annotation
# ---------------------------------------------------------------------------
def add_site_identificator(
    df: pd.DataFrame,
    mod_col: str = "Best Precursor for Quant",
    peptide_index_col: str = "peptide_index",
) -> pd.DataFrame: # Valid only for the LFQ test datafrae
    """
    Adds the peptide sequence with the phosphorylatied residue in small letter to the peptide_index
    identification to have the complete site naming
    """
    result = df.copy()
    mods = result[mod_col]

    # Build absolute-position modification strings; NaN for rows without any modification
    site_seq = mods.apply(_get_phosphosite_sequene)
    result["phosphosite_seq"] = site_seq  #

    result["site"] = result[peptide_index_col].astype(str) + "~" +  result["phosphosite_seq"].astype(str)
    result = result.drop(columns="phosphosite_seq")
    return result

# ---------------------------------------------------------------------------
# 4. LFQ pipeline
# ---------------------------------------------------------------------------
def add_zscore_normalization(
    df: pd.DataFrame,
    cell_lines: list,
    conditions: list = ["_EGF_", "_INS_", "_EGFnINS_"],
    exclude_full: bool = False,
) -> pd.DataFrame:
    """
    Add per-site z-score normalization of the temporal profile to an LFQ dataset.

    Convenience entry point for LFQ data. It delegates to
    src.transformations.compute_zscore_fc(), applying it once per cell line, so the
    LFQ workflow gets log2:zscore columns without importing from src.transformations
    directly. For each phosphosite (row), each (cell_line, condition) time series of
    log2:FC values is standardised across its timepoints:

        log2:zscore(condition, timepoint) =
            (log2:FC(timepoint) − mean_t log2:FC) / std_t log2:FC

    Note: run_all_transformations() already applies this same step, so calling this
    is only needed to add z-scores to a DataFrame that was transformed WITHOUT it
    (or to re-apply with a different `exclude_full`). Requires log2:FC columns to be
    present already (i.e. run after run_all_transformations).

    Args:
        df: DataFrame that already has log2:FC columns (post run_all_transformations).
        cell_lines: list of cell line identifiers used as column prefixes,
            e.g. ['WT', 'BRAFS151A', 'GAB1Y259A'].
        conditions: list of condition substrings to match, e.g. ['_EGF_'].
        exclude_full: if True, drop the 'full' timepoint before computing the z-score
            (default False).

    Returns:
        Copy of df with the new log2:zscore columns appended. The original is not modified.
    """
    from src.transformations import compute_zscore_fc

    result = df.copy()
    for cell_line in cell_lines:
        result = compute_zscore_fc(result, cell_line=cell_line, conditions=conditions,
                                   exclude_full=exclude_full)
    return result


# ---------------------------------------------------------------------------
# 5. LFQ filtering
# ---------------------------------------------------------------------------
