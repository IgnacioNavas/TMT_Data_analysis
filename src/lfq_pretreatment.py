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

def _count_other_mods(mod_str: str) -> int:
    """Count comma-separated modification items that contain no S, T, or Y character."""
    if not isinstance(mod_str, str) or not mod_str:
        return 0
    return sum(not any(c in item for c in "STY") for item in mod_str.split(","))

def _clean_mod_string(mod_str: str) -> str:
    """
    Reformat a raw modification string to compact letter+position notation.

    Steps applied:
      1. Strip mass annotations in parentheses, e.g. 'S(79.9663)' → 'S'
      2. Swap digit-first notation to letter-first, e.g. '17S' → 'S17'
      3. Remove commas between entries

    Example: '17S(79.9663),25S(79.9663)' → 'S17S25'
    """
    if not isinstance(mod_str, str) or not mod_str:
        return ""
    cleaned = re.sub(r"\s*\([^)]*\)", "", mod_str)             # strip mass values e.g., (79.9633)
    cleaned = re.sub(r"\b(\d+)([A-Za-z])\b", r"\2\1", cleaned) # digit+letter → letter+digit
    return cleaned.replace(",", "")

def _offset_mod_positions(mod_str: str, site_start) -> str:
    """
    Shift each numeric position in a compact modification string by (site_start - 1).

    Converts peptide-relative positions to absolute protein positions.
    Example: 'S17S25' with site_start=462 → 'S478S486'
    """
    if not mod_str or pd.isna(site_start):
        return mod_str
    return re.sub(r"([A-Za-z])(\d+)", lambda m: f"{m.group(1)}{int(m.group(2)) + int(site_start) - 1}", mod_str,)

def _extract_sty_positions(mod_str: str) -> str:
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
    matches = re.findall(r"[STY]\d+", mod_str)
    return "".join(matches) if matches else None


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
def add_modification_metadata(
    df: pd.DataFrame,
    mod_col: str = "assigned_modifications",
    charges_col: str = "charges",
    site_start_col: str = "site_start",
    site_end_col: str = "site_end",
    protein_id_col: str = "protein_Id",
) -> pd.DataFrame:
    """
    Add modification metadata and a composite site identifier column to the DataFrame.

    New columns appended:
        n_localized:                 total number of localized modifications
        STY_localized:               number of S/T/Y (phosphorylation) modifications
        other_localized:             number of non-S/T/Y modifications (e.g. oxidation)
        all_localized:               True when all modifications equal the first charge state count
        assigned_modifications_clean: compact string of residue+absolute_position, e.g. 'S142M148S261'
        sty_positions:               only the S/T/Y pairs from assigned_modifications_clean, e.g. 'S142S261'
        site:                  composite row identifier using only the STY positions,
                                     e.g. 'Q9Y2U8_256_273_1_1_S261'

    Args:
        df: DataFrame with LFQ phosphoproteomics columns.
        mod_col: column with assigned modifications (comma-separated, e.g. '17S(79.9663),25S(79.9663)').
        charges_col: column with charge states (comma-separated for multiple, e.g. '2,3').
        site_start_col: column with the peptide's start residue position in the protein.
        site_end_col: column with the peptide's end residue position in the protein.
        protein_id_col: column with the UniProt protein identifier.

    Returns:
        Copy of df with the six annotation columns appended.
    """
    result = df.copy()
    mods = result[mod_col]

    result["n_localized"] = mods.str.split(",").str.len().fillna(0).astype(int)
    result["STY_localized"] = mods.apply(_count_sty_mods)
    result["other_localized"] = mods.apply(_count_other_mods)

    # first_charge = result[charges_col].astype(str).str.split(",").str[0].astype(int) # sometimes the charge column has 2 integers, here we are selecting one
    # result["all_localized"] = (result["STY_localized"] + result["other_localized"]) == first_charge # If the sum of types of charges is equal to the amount of charges, then all modifications were localized

    # Build absolute-position modification strings; NaN for rows without any modification
    clean_mods = mods.apply(_clean_mod_string)
    abs_mods = [_offset_mod_positions(mod, start) if mod else np.nan for mod, start in zip(clean_mods, result[site_start_col])]
    result["assigned_modifications_clean"] = abs_mods  #
    result.loc[mods.isna(), "assigned_modifications_clean"] = np.nan

    # Extract only STY positions (filters out oxidation, N-term acetylation, etc.)
    result["sty_positions"] = result["assigned_modifications_clean"].apply(_extract_sty_positions)

    # Composite site identifier; when no STY modification is present, sty_positions is "0"
    sty_label = result["sty_positions"].fillna("0")
    result["site_index"] = (
        result[protein_id_col].astype(str) + "_"
        + result[site_start_col].astype(str) + "_"
        + result[site_end_col].astype(str) + "_"
        + result["n_localized"].astype(str) + "_"
        + result["STY_localized"].astype(str) + "_"
        + sty_label.astype(str)
    )
    result["site"] = result["site_index"].astype(str) + "~" + result["Modified_Sequence"].str.replace(r'([A-Z])(\[\d+\.?\d*\])', lambda m: m.group(1).lower(), regex=True)
    return result


# ---------------------------------------------------------------------------
# 4. LFQ pipeline
# ---------------------------------------------------------------------------
def run_all_pretreatment( # This function its applying pre-filtering
    df: pd.DataFrame,
    cell_lines: list,
    conditions: list = ["_EGF_", "_INS_", "_EGFnINS_"],
    data_type: str = "raw:abs",
    mod_col: str = "assigned_modifications",
    charges_col: str = "charges",
    site_start_col: str = "site_start",
    site_end_col: str = "site_end",
    protein_id_col: str = "protein_Id",
) -> pd.DataFrame:
    """
    Run the full LFQ pre-treatment pipeline in a single call.

    Steps applied in order:
        1. add_modification_metadata — annotates n_localized, STY_localized, other_localized,
           all_localized, assigned_modifications_clean, and site columns.
        2. filter_no_ptm — removes rows with no identified PTM.
        3. filter_not_detected_in_starve — removes rows absent from any starvation control replicate.
        4. filter_missing_replicates — removes rows undetected in all replicates of any time point.

    After this pipeline, the DataFrame is ready for src.transformations.run_all_transformations().

    Args:
        df: Input DataFrame with raw LFQ phosphoproteomics columns following the naming convention
            {CellLine}_{DataType}_{Treatment}_{TimePoint}_{Replicate}.
        cell_lines: list of cell line identifiers used as column prefixes,
            e.g. ['WT', 'BRAFS151A', 'GAB1Y259A'].
        conditions: list of condition substrings to match, e.g. ['_EGF_', '_INS_'].
        data_type: DataType field for raw intensity columns, default 'raw:abs'.
        mod_col: column with the raw assigned modifications string.
        charges_col: column with charge state information.
        site_start_col: column with the peptide start position.
        site_end_col: column with the peptide end position.
        protein_id_col: column with the protein identifier.

    Returns:
        Pre-treated DataFrame with annotation columns added and low-quality rows removed.
        The original DataFrame is not modified.
    """
    print(f"Starting LFQ pre-treatment: {len(df)} rows, {len(df.columns)} columns.")

    result = add_modification_metadata(
        df,
        mod_col=mod_col,
        charges_col=charges_col,
        site_start_col=site_start_col,
        site_end_col=site_end_col,
        protein_id_col=protein_id_col,
    )
    result = filter_no_ptm(result)
    result = filter_not_detected_in_starve(result, cell_lines=cell_lines, conditions=conditions, data_type=data_type)
    result = filter_missing_replicates(result, cell_lines=cell_lines, conditions=conditions, data_type=data_type)

    print(f"Pre-treatment complete: {len(result)} rows remaining "
          f"({len(df) - len(result)} removed), {len(result.columns)} columns.")
    return result



# ---------------------------------------------------------------------------
# 5. Cross-dataset matching
# ---------------------------------------------------------------------------
def add_site_matching_column(
    df: pd.DataFrame,
    protein_id_col: str = "protein_Id",
    start_mod_col: str = "site_start",
    end_mod_col: str = "site_end",
    num_phos_col: str = "NumPhos",
    local_num_phos_col: str = "LocalizedNumPhos",
    phos_sites_col: str = "PhosSites",
    peptide_seq_col: str = "peptide_seq",
    new_col: str = "site_matching",
) -> pd.DataFrame:
    """
    Add a site_matching column to a TMT dataset, replicating the format of the 'site' column.

    The TMT site identifier format is:
        {protein_Id}_{startModSite}_{endModSite}_{NumPhos}_{LocalizedNumPhos}~{peptide_seq}
    When at least one site is localized (LocalizedNumPhos > 0), the localized residue positions
    are inserted before the peptide sequence:
        {protein_Id}_{startModSite}_{endModSite}_{NumPhos}_{LocalizedNumPhos}_{PhosSites}~{peptide_seq}

    where PhosSites is the semicolon-separated list of localized sites with semicolons removed,
    e.g. 'T205;T207' → 'T205T207'.

    Examples:
        LocalizedNumPhos=0 → 'Q9P1Y5_362_368_1_0~HPLLSSGGPQSPLR'
        LocalizedNumPhos=2 → 'Q04637_202_223_2_2_T205T207~TAStPtPPQTGGGLEPQANGETPQVAVIVRPDDR'

    Args:
        df: TMT DataFrame with phosphosite annotation columns.
        protein_id_col: column with the UniProt protein identifier.
        start_mod_col: column with the first modified residue position in the protein.
        end_mod_col: column with the last modified residue position in the protein.
        num_phos_col: column with the total number of STY phosphorylations on the peptide.
        local_num_phos_col: column with the number of STY phosphorylations that were localized.
        phos_sites_col: column with localized site positions (semicolon-separated, e.g. 'T205;T207').
        peptide_seq_col: column with the peptide amino acid sequence.
        new_col: name for the new site identifier column.

    Returns:
        Copy of df with the site_matching column appended.
    """
    result = df.copy()

    base = (result[protein_id_col].astype(str) + "_"
            + result[start_mod_col].astype(str) + "_"
            + result[end_mod_col].astype(str) + "_"
            + result[num_phos_col].astype(str) + "_"
            + result[local_num_phos_col].astype(str))

    # Append localized site positions only for rows with at least one localized phosphosite
    has_localized = result[local_num_phos_col] > 0
    sites_suffix = pd.Series("", index=result.index)
    sites_suffix[has_localized] = (
        "_" + result.loc[has_localized, phos_sites_col]
                    .astype(str)
                    .str.replace(";", "", regex=False)
    )

    result[new_col] = base + sites_suffix + "~" + result[peptide_seq_col].astype(str)
    return result


# ---------------------------------------------------------------------------
# 5. LFQ filtering
# ---------------------------------------------------------------------------

def filter_no_ptm(
    df: pd.DataFrame,
    mod_clean_col: str = "assigned_modifications_clean",
) -> pd.DataFrame:
    """
    Remove rows with no identified post-translational modification.

    Rows where assigned_modifications_clean is NaN carry no PTM information
    and cannot be used in phosphoproteomics analysis.

    Args:
        df: DataFrame containing the assigned_modifications_clean column.
        mod_clean_col: name of the cleaned modification column.

    Returns:
        Filtered copy of df.
    """
    before = len(df)
    result = df[df[mod_clean_col].notna()].copy()
    removed = before - len(result)
    if removed:
        print(f"filter_no_ptm: removed {removed} rows without PTM.")
    return result


def filter_not_detected_in_starve(
    df: pd.DataFrame,
    cell_lines: list,
    conditions: list,
    data_type: str = "raw:abs",
) -> pd.DataFrame:
    """
    Remove rows where any starvation control replicate has intensity 0.

    The starvation time point is the fold-change reference; sites absent
    at starvation cannot produce a meaningful fold-change profile.

    Args:
        df: DataFrame with raw intensity columns.
        cell_lines: list of cell line prefixes, e.g. ['WT', 'BRAFS151A', 'GAB1Y259A'].
        conditions: list of condition substrings, e.g. ['_EGF_', '_INS_'].
        data_type: data type substring to match, default 'raw:abs'.

    Returns:
        Filtered copy of df.
    """
    raw_cols = _get_raw_columns(df, cell_lines, conditions, data_type)
    starve_cols = [c for c in raw_cols if "starve" in c]

    if not starve_cols:
        warnings.warn("filter_not_detected_in_starve: no starve columns found — skipping.")
        return df.copy()

    before = len(df)
    result = df.copy()
    for col in starve_cols:
        result = result[result[col] != 0]

    removed = before - len(result)
    if removed:
        print(f"filter_not_detected_in_starve: removed {removed} rows absent from starvation control.")
    return result.copy()


def filter_missing_replicates(
    df: pd.DataFrame,
    cell_lines: list,
    conditions: list,
    data_type: str = "raw:abs",
) -> pd.DataFrame:
    """
    Remove rows where all replicates are 0 for any non-starve time point.

    A site undetected in every replicate at a given time point contributes
    no signal for that time point and is excluded from the dataset.

    Args:
        df: DataFrame with raw intensity columns.
        cell_lines: list of cell line prefixes, e.g. ['WT', 'BRAFS151A', 'GAB1Y259A'].
        conditions: list of condition substrings, e.g. ['_EGF_', '_INS_'].
        data_type: data type substring to match, default 'raw:abs'.

    Returns:
        Filtered copy of df.
    """
    raw_cols = _get_raw_columns(df, cell_lines, conditions, data_type)
    non_starve = [c for c in raw_cols if "starve" not in c]

    # Derive unique base names by stripping replicate suffixes (e.g. _r1, _r2, _r3)
    base_names = list(dict.fromkeys(re.sub(r"_r\d+$", "", c) for c in non_starve))

    before = len(df)
    result = df.copy()

    for base in base_names:
        rep_cols = [c for c in non_starve if re.sub(r"_r\d+$", "", c) == base]
        if not rep_cols:
            continue
        all_zero = (result[rep_cols] == 0).all(axis=1)
        result = result[~all_zero]

    removed = before - len(result)
    if removed:
        print(f"filter_missing_replicates: removed {removed} rows undetected across all replicates of a time point.")
    return result.copy()