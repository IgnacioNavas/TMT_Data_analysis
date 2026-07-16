"""
Kinase prediction: annotate phosphosites with their most likely upstream kinases.

Uses the `kinase_library` package (Johnson et al. 2023 position-weight matrices) to
score each phosphosite and return, per site, the top-N predicted kinases together with
their percentile scores (0-100; the site's score ranked against a reference
phosphoproteome — higher = more likely).

Module layout
─────────────
1. UniProt sequences     — load_uniprot_sequences()
2. Site / window helpers — parse_single_phosphosite(), extract_window(),
                           validate_against_uniprot()
3. Window building       — default_columns(), add_kinase_windows()
4. Kinase prediction     — predict_top_kinases()  ← main scoring entry point

Design notes
────────────
* The kinase_library PWMs read a 15-mer centered on the phospho-acceptor (index 7).
  Windows are cut from the full UniProt protein sequence using each site's ABSOLUTE
  residue position, so the same code path serves TMT and LFQ datasets and always
  yields true protein context (padded with '_' only at protein termini).
* The package routes each window to the correct kinome by its central residue
  (S/T -> 'ser_thr', Y -> 'tyrosine'); predict_top_kinases() scores both and combines.
* Only single-localized sites should be passed in (TMT LocalizedNumPhos == 1;
  LFQ n_localized == 1 & other_localized == 0) — filter upstream in the notebook.

Typical workflow
────────────────
    seq_lookup = load_uniprot_sequences("External_Data/Metadata/uniprotkb_...tsv")
    cols       = default_columns(df, dataset_type="TMT")
    windows_df, report = add_kinase_windows(df_single_localized, seq_lookup, cols)
    predicted  = predict_top_kinases(windows_df, top_n=5)
"""
import re

import pandas as pd
import kinase_library as kl


# ---------------------------------------------------------------------------
# 1. UniProt sequences
# ---------------------------------------------------------------------------
def load_uniprot_sequences(uniprot_tsv_path,
                           id_col="Entry",
                           seq_col="Sequence",):
    """
    Build an accession -> protein-sequence lookup from a UniProt TSV export.

    Only the two needed columns are read, so this is fast even on the full
    reviewed-human proteome (~20k rows, ~0.2 s).

    Args:
        uniprot_tsv_path: path to the UniProt TSV (must contain id_col and seq_col).
        id_col: column holding the UniProt accession (default "Entry").
        seq_col: column holding the full protein sequence (default "Sequence").

    Returns:
        dict mapping {accession: sequence}, with missing rows dropped.
    """
    table = pd.read_csv(uniprot_tsv_path,
                        sep="\t",
                        usecols=[id_col, seq_col],)
    table = table.dropna(subset=[id_col, seq_col],)
    return dict(zip(table[id_col],
                    table[seq_col],))


# ---------------------------------------------------------------------------
# 2. Site / window helpers
# ---------------------------------------------------------------------------
def parse_single_phosphosite(site_field,):
    """
    Parse a single-site position string into a (residue, position) pair.

    Accepts values such as "S199" (TMT PhosSites) or "S676" (LFQ sty_positions).
    Rejects empty values and any field containing more than one S/T/Y site (e.g.
    "T205;T207" or "T23S33"), returning None so multiply-localized rows are skipped.

    Args:
        site_field: the localized-site string for one row.

    Returns:
        (residue, position) with residue in {"S","T","Y"} and position as an int
        (1-based), or None if the field is not a clean single S/T/Y site.
    """
    if not isinstance(site_field, str):
        return None
    matches = re.findall(r"([STY])(\d+)",
                         site_field.upper(),)
    if len(matches) != 1:
        return None
    residue, position = matches[0]
    return (residue, int(position),)


def extract_window(sequence,
                   position,
                   half_width=7,
                   pad_char="_",):
    """
    Cut a (2*half_width+1)-mer window centered on a 1-based residue position.

    The phospho-acceptor sits at the center (index half_width). Positions within
    half_width of a protein terminus are padded with pad_char so the window is always
    exactly 2*half_width+1 characters long (the length the kinase_library PWMs expect).

    Args:
        sequence: the full protein sequence.
        position: 1-based position of the phospho-acceptor in `sequence`.
        half_width: residues to include on each side (default 7 -> 15-mer).
        pad_char: character used to pad at protein termini (default "_").

    Returns:
        The centered window string, or None if `sequence` is invalid or `position`
        is out of range.
    """
    if not isinstance(sequence, str) or position < 1 or position > len(sequence):
        return None
    center = position - 1  # 0-based
    start = center - half_width
    end = center + half_width + 1
    left_pad = pad_char * max(0, -start)
    right_pad = pad_char * max(0, end - len(sequence))
    core = sequence[max(0, start):min(len(sequence), end)]
    return left_pad + core + right_pad


def validate_against_uniprot(peptide_seq,
                             start,
                             sequence,):
    """
    Check whether a detected peptide matches the UniProt sequence at its start position.

    Compares the (uppercased) peptide against sequence[start-1 : start-1+len(peptide)].
    A mismatch flags an isoform difference, a wrong protein mapping, an off-by-one
    position, or an obsolete accession. Some datasets store several equivalent peptide
    variants joined by ";" (e.g. "RSsVAQEDSK;SsVAQEDSK"); the peptide is considered a
    match if ANY variant matches the UniProt sequence at `start`.

    Args:
        peptide_seq: the peptide sequence (may carry a lowercase phospho marker; it is
                     uppercased before comparison). May be a ";"-joined list of variants.
        start: 1-based start position of the peptide in the protein.
        sequence: the full UniProt protein sequence.

    Returns:
        True/False whether the peptide matches, or None if inputs are missing/invalid.
    """
    if not isinstance(peptide_seq, str) or not isinstance(sequence, str) or pd.isna(start):
        return None
    start = int(start)
    for variant in peptide_seq.upper().split(";"):
        variant = variant.strip()
        if not variant:
            continue
        subsequence = sequence[start - 1: start - 1 + len(variant)]
        if subsequence == variant:
            return True
    return False


# ---------------------------------------------------------------------------
# 3. Window building
# ---------------------------------------------------------------------------
def default_columns(df,
                    dataset_type,):
    """
    Resolve the dataset-specific column names used for window building and validation.

    Handles the two LFQ casings (e.g. "peptide_sequence" vs "Peptide_Sequence") by
    picking whichever variant is present in `df`.

    Args:
        df: the site-level dataframe (used only to resolve column casing).
        dataset_type: "TMT" (hme1_1/hme1_2/hek_1) or "LFQ" (mutant datasets).

    Returns:
        dict with keys 'protein_id', 'position', 'peptide', 'start' naming the
        columns to use.

    Raises:
        ValueError: if dataset_type is unknown or an expected column is absent.
    """
    def _first_present(candidates,):
        for name in candidates:
            if name in df.columns:
                return name
        raise ValueError(f"None of {candidates} found in dataframe columns.")

    if dataset_type == "TMT":
        return {"protein_id": _first_present(["protein_Id", "protein_ID"]),
                "position":   "PhosSites",
                "peptide":    "peptide_seq",
                "start":      "Start",}
    if dataset_type == "LFQ":
        return {"protein_id": _first_present(["protein_Id", "protein_ID"]),
                "position":   _first_present(["sty_positions"]),
                "peptide":    _first_present(["peptide_sequence", "Peptide_Sequence"]),
                "start":      _first_present(["site_start"]),}
    raise ValueError(f"Unknown dataset_type {dataset_type!r} (expected 'TMT' or 'LFQ').")


def add_kinase_windows(df,
                       seq_lookup,
                       columns,
                       half_width=7,):
    """
    Add UniProt-derived kinase windows (and a validation flag) to single-localized sites.

    For each row it parses the single localized S/T/Y site, looks up the protein
    sequence, cuts the centered window, and checks that the UniProt residue at that
    position matches the claimed residue. Rows without a clean single site, without a
    known protein, with an out-of-range position, or with a residue disagreement get no
    window and are excluded from the returned frame (but counted in the report).

    Pass a dataframe already filtered to single-localized sites (TMT
    LocalizedNumPhos == 1; LFQ n_localized == 1 & other_localized == 0).

    Args:
        df: single-localized site-level dataframe.
        seq_lookup: {accession: sequence} dict from load_uniprot_sequences().
        columns: column-name bundle from default_columns() (keys 'protein_id',
                 'position', 'peptide', 'start').
        half_width: window half width (default 7 -> 15-mer).

    Returns:
        (windows_df, report):
            windows_df — copy of the input rows that yielded a valid window, with added
                columns 'kinase_window', 'kinase_residue' (S/T/Y) and
                'uniprot_seq_match' (bool/None peptide-vs-UniProt check).
            report — dict of counts: input_rows, valid_windows, bad_or_multi_site,
                protein_not_found, position_out_of_range, residue_mismatch, seq_mismatch.
    """
    prot_col = columns["protein_id"]
    pos_col = columns["position"]
    pep_col = columns["peptide"]
    start_col = columns["start"]

    windows, residues, seq_matches = [], [], []
    n_bad_site = n_no_protein = n_out_of_range = n_residue_mismatch = 0

    for _, row in df.iterrows():
        parsed = parse_single_phosphosite(row[pos_col],)
        if parsed is None:
            n_bad_site += 1
            windows.append(None); residues.append(None); seq_matches.append(None)
            continue
        residue, position = parsed
        sequence = seq_lookup.get(row[prot_col])
        if not isinstance(sequence, str):
            n_no_protein += 1
            windows.append(None); residues.append(residue); seq_matches.append(None)
            continue
        if position > len(sequence):
            n_out_of_range += 1
            windows.append(None); residues.append(residue); seq_matches.append(None)
            continue
        if sequence[position - 1] != residue:
            n_residue_mismatch += 1
            windows.append(None); residues.append(residue); seq_matches.append(False)
            continue
        windows.append(extract_window(sequence,
                                      position,
                                      half_width=half_width,))
        residues.append(residue)
        seq_matches.append(validate_against_uniprot(row[pep_col],
                                                    row[start_col],
                                                    sequence,))

    annotated = df.copy()
    annotated["kinase_window"] = windows
    annotated["kinase_residue"] = residues
    annotated["uniprot_seq_match"] = seq_matches

    windows_df = annotated[annotated["kinase_window"].notna()].copy()
    report = {"input_rows":            len(df),
              "valid_windows":         len(windows_df),
              "bad_or_multi_site":     n_bad_site,
              "protein_not_found":     n_no_protein,
              "position_out_of_range": n_out_of_range,
              "residue_mismatch":      n_residue_mismatch,
              "seq_mismatch":          int((windows_df["uniprot_seq_match"] == False).sum()),}
    return windows_df, report


# ---------------------------------------------------------------------------
# 4. Kinase prediction
# ---------------------------------------------------------------------------
def predict_top_kinases(df,
                        seq_col="kinase_window",
                        top_n=5,
                        kinase_col_prefix="predicted_kinase",):
    """
    Add per-site top-N predicted kinases and their percentile scores.

    Scores S/T sites with the ser_thr kinome and Y sites with the tyrosine kinome
    (the package routes each window by its central residue), combines both percentile
    matrices, and for every site keeps the top_n kinases by percentile. Predictions are
    mapped back to the original rows through the PhosphoProteomics 'Sequence' column, so
    rows the package drops as invalid substrates simply receive NaN.

    Args:
        df: dataframe with a window column (typically from add_kinase_windows()).
        seq_col: name of the 15-mer window column (default "kinase_window").
        top_n: number of ranked kinases to keep per site (default 5).
        kinase_col_prefix: prefix for the new columns (default "predicted_kinase").

    Returns:
        Copy of df with 2*top_n new columns:
            {prefix}_1 .. {prefix}_top_n           kinase names (rank 1 = most likely)
            {prefix}_1_prob .. {prefix}_top_n_prob percentile 0-100 (non-increasing)
        Rows without a valid prediction hold NaN in these columns.
    """
    result = df.copy()
    pps = kl.PhosphoProteomics(df,
                               seq_col=seq_col,
                               pp=True,)

    percentile_frames = []
    for kin_type in ("ser_thr", "tyrosine",):
        percentiles = pps.percentile(kin_type=kin_type,
                                     values_only=True,)
        if len(percentiles):
            percentile_frames.append(percentiles)

    # Pre-create the output columns (NaN by default).
    kinase_cols = [f"{kinase_col_prefix}_{i}" for i in range(1, top_n + 1)]
    prob_cols = [f"{kinase_col_prefix}_{i}_prob" for i in range(1, top_n + 1)]
    for col in kinase_cols + prob_cols:
        result[col] = pd.NA
    if not percentile_frames:
        return result

    # One combined (substrate x kinase) percentile matrix; rows are disjoint by residue
    # type, so the "other" kinome's columns are NaN and ignored by nlargest.
    percentile_matrix = pd.concat(percentile_frames,
                                  axis=0,)
    percentile_matrix = percentile_matrix[~percentile_matrix.index.duplicated(keep="first")]

    # Per-substrate top-N lookup: {substrate_sequence: (names, probs)}.
    top_by_substrate = {}
    for substrate, row in percentile_matrix.iterrows():
        top = row.dropna().nlargest(top_n)
        names = list(top.index) + [pd.NA] * (top_n - len(top))
        probs = list(top.values) + [pd.NA] * (top_n - len(top))
        top_by_substrate[substrate] = (names, probs)

    # Bridge substrate -> original row index via the retained 'Sequence' column.
    for orig_idx, substrate in pps.data["Sequence"].items():
        record = top_by_substrate.get(substrate)
        if record is None:
            continue
        names, probs = record
        for i in range(top_n):
            result.at[orig_idx, kinase_cols[i]] = names[i]
            result.at[orig_idx, prob_cols[i]] = probs[i]

    return result
