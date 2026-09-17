"""
Dynamic EGF signalling network, version 2 — site table construction.

This module backs `notebooks/07_dynamics_RPCST/dynamic_network_v2.ipynb`. It is the successor to
`src/dynamic_rpcst.py`, which is left untouched so the two runs can be compared; functions that
did not need to change are imported from there rather than copied.

The full specification, including the reasoning behind every rule implemented here, is
`Claude_promts/dynamic_network_instrucitons.md`. Section numbers in the docstrings refer to it.

Implemented so far (step 1 of §9):
  - loading the processed hme1_2 table,
  - the limma FFDR responsiveness gate that replaces the amplitude-only top-fraction cut,
  - the site activation time, dispatched on TIME_DEFINITION (only 'peak_fc' for now),
  - the signed peak response, so a dephosphorylation is no longer indistinguishable from a
    phosphorylation,
  - explosion of multi-site peptides into one candidate row per residue, which is what makes a
    site addressable in the kinase-substrate table,
  - collapse to one row per phosphosite, and the filter ledger that reports every step.

Implemented in step 2 of §9:
  - the kinase-substrate edge loader, with the two stringency parameters that keep the graph
    small (§4.0: cut edges, not sites),
  - the sign rule deciding which sites may pass activation on to their own kinase (§4.1),
  - the candidate graph, in which any site may be a leaf,
  - the ILP, rooted by a single-commodity flow rather than the MTZ ordering of version 1 (§5),
  - the objective decomposition, the feedback tagging and the validation of a selection.

Not yet implemented: the PPI layer, the ensemble, the drawing (steps 4-6 of §9).
"""

from __future__ import annotations

import pickle
from itertools import islice
from pathlib import Path

import cvxpy as cp
import networkx as nx
import numpy as np
import pandas as pd

from src.column_spec import ColumnSpec
from src.filters import filter_by_ffdr

# ── Defaults ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CELL_LINE = "WT"
DEFAULT_CONDITION = "EGF"

# The measured timepoint grid of hme1_2, excluding the two controls. A site's activation time is
# always one of these labels; 'full' and 'starve' are not responses and are never a valid peak.
STIMULATION_TIMEPOINTS = ("2", "5", "10", "15", "90",)
CONTROL_TIMEPOINTS = ("full", "starve",)

DEFAULT_MAX_FFDR = 0.05

# The EGF receptor: the one node the network is rooted at in step 2.
EGFR_UNIPROT = "P00533"
ROOT = "ROOT"

# §4.0. The two parameters that keep the candidate graph small. The kinase-substrate table is
# dominated by lenient motif predictions, so without them a site carries ~43 candidate kinases.
DEFAULT_RESOURCE_REGEX = "Strict|literature"
DEFAULT_MAX_KINASES_PER_SITE = 5

# The grid a latent kinase activation time is drawn from. 0 lets a kinase sit upstream of the
# earliest measured site.
ALLOWED_ACTIVATION_TIMES = (0, 2, 5, 10, 15, 90,)

# Charged on every edge so that a zero-cost cycle can never be free; see build_ilp.
EDGE_EPSILON = 1e-4

# §4.0. Edge price by evidence class. The kinase-substrate table's `score` column is NOT a
# confidence on a common scale: measured medians are kinlib-Strict 0.9951, combined-Strict 0.9935,
# phosformer-Strict 0.8570 and literature **0.0039**. Pricing an edge as 1 - score therefore
# charges ~0.996 for a curated, experimentally documented interaction and ~0.005 for a motif guess,
# i.e. it penalises real knowledge hardest. Cost is set from the evidence class instead, and the
# score is used only to rank edges *within* a class.
RESOURCE_CLASS_COSTS = {"literature": 0.05,
                        "strict": 0.20,
                        "moderate": 0.40,
                        "lenient": 0.60,}
RESOURCE_CLASS_ORDER = ("literature", "strict", "moderate", "lenient",)
WITHIN_CLASS_SPAN = 0.15

# §3. 'onset' is the default since 2026-09-16, measured against 'peak_fc' on this dataset:
# both activation-loop pairs agree under onset and only one under peak_fc; every marker kinase
# moves from t=5-10 to t=2; ERK2 stops escaping the temporal constraint through an orphan edge;
# and the ILP solves in 37 s instead of 98 s while collecting slightly more prize. 'max_step' is
# specified in the instructions but not built.
TIME_DEFINITIONS = ("peak_fc", "max_step", "onset",)
DEFAULT_TIME_DEFINITION = "onset"

# Pairs of residues that are phosphorylated by the same event and must therefore receive the same
# activation time. Used by activation_loop_report() as a check on the time definition (§3).
ACTIVATION_LOOP_PAIRS = (
    ("P27361", "T202", "Y204", "ERK1 TEY activation loop",),
    ("P28482", "T185", "Y187", "ERK2 TEY activation loop",),
    ("Q02750", "S218", "S222", "MEK1 activation loop",),
    ("P36507", "S222", "S226", "MEK2 activation loop",),
    ("P31749", "T308", "S473", "AKT1 activation, PDK1 and mTORC2",),
)


# ── 1. Loading ───────────────────────────────────────────────────────────────────────────


def load_processed_table(path: Path,
                         usecols: list | None = None,) -> pd.DataFrame:
    """
    Read a processed phosphoproteomics table without touching its missing values.

    Missing values stay NaN. This matters twice over: QC functions need to see them, and the
    limma statistics columns carry NaN for every site limma could not test, where 0.0 would read
    as perfectly significant and let untested sites through the responsiveness gate.

    Args:
      path: Path to the processed .tsv table.
      usecols: Optional list of columns to read, passed to pandas. None reads every column.

    Returns:
      DataFrame as stored on disk, with NaN preserved.
    """
    return pd.read_csv(path,
                       sep="\t",
                       usecols=usecols,
                       low_memory=False,)


# ── 2. Filtering ─────────────────────────────────────────────────────────────────────────


def filter_localized(df: pd.DataFrame,) -> pd.DataFrame:
    """
    Keep only peptides whose phosphosite positions were confidently localized.

    A row with no PhosSites has a phosphorylation somewhere on the peptide but no assigned
    residue, so it cannot be matched to a kinase-substrate edge or drawn as a site node.

    Args:
      df: Peptide-level DataFrame carrying the FragPipe PhosSites column.

    Returns:
      Filtered DataFrame copy holding only rows with a non-null PhosSites value.
    """
    return df[df["PhosSites"].notna()].copy()


def filter_responsive(df: pd.DataFrame,
                      max_ffdr: float | None = DEFAULT_MAX_FFDR,
                      cell_line: str = DEFAULT_CELL_LINE,
                      condition: str = DEFAULT_CONDITION,) -> pd.DataFrame:
    """
    Keep the sites limma's omnibus F-test calls responsive to the stimulus.

    Thin wrapper over filters.filter_by_ffdr, which reads
    {cell_line}_log2:FFDR_{condition}_omnibus. This replaces the amplitude-only top-fraction cut
    of version 1 (§2.1): an amplitude cutoff asks "did this site move", the F-test asks "did it
    move more than its own replicate noise". Sites limma never tested carry NaN and are dropped.

    Args:
      df: DataFrame carrying the merged limma omnibus columns.
      max_ffdr: FDR cutoff on the omnibus F. None switches the filter off and returns a copy.
      cell_line: Cell-line prefix of the omnibus column, e.g. 'WT'.
      condition: Stimulation condition of the omnibus column, e.g. 'EGF'.

    Returns:
      Filtered DataFrame copy.
    """
    return filter_by_ffdr(df,
                          cell_lines=[cell_line],
                          conditions=[condition],
                          max_ffdr=max_ffdr,
                          combine="any",)


# ── 3. Activation time and peak response ─────────────────────────────────────────────────


def filter_by_prize_floor(site_table: pd.DataFrame,
                          min_abs_log2_fc: float | None = None,) -> pd.DataFrame:
    """
    Keep only the sites whose response is large enough to be worth explaining (§4.0).

    The significance gate of §2.1 asks whether a site moved reproducibly; it says nothing about how
    far. On this dataset 14797 sites pass it with a median |log2FC| of 0.42, and explaining all of
    them produces a network of thousands of nodes. Raising NODE_COST shrinks that network but by
    discarding coverage: the readable 59-node result of step 3 explains 1.9% of the response.

    A prize floor attacks the same problem from the other side, which is the "mixture of both" idea
    from the original notes: filter first to the sites worth explaining, then let a *lower* node
    cost explain most of that set. 'Most of the strong responders' is a defensible claim in a way
    that '1.9% of everything' is not.

    Args:
      site_table: Site table as returned by build_site_table.
      min_abs_log2_fc: Minimum |peak log2 fold change|. None applies no floor, so a notebook can
        switch the step off without an `if` around the call.

    Returns:
      Filtered DataFrame copy.
    """
    if min_abs_log2_fc is None:
        return site_table.copy()
    return site_table[site_table["peak_abs_log2_fc_vs_starve"] >= min_abs_log2_fc].copy()


def add_activation_time(df: pd.DataFrame,
                        time_definition: str = DEFAULT_TIME_DEFINITION,
                        cell_line: str = DEFAULT_CELL_LINE,
                        condition: str = DEFAULT_CONDITION,) -> pd.DataFrame:
    """
    Assign each site the single timepoint used as its activation time by the ILP.

    Dispatches on time_definition so the definition can be swapped from the notebook's parameter
    cell without touching anything downstream (§3). Only 'peak_fc' is implemented:

      peak_fc  timepoint of largest |log2:FC|, read from the pre-computed {cell_line}_peak:FC_
               {condition} column. The current default, and the version 1 behaviour.
      max_step timepoint of the largest signed step, from the log2:step columns. Specified, not
               built.
      onset    first timepoint reaching 50% of the peak, interpolated in log-time. Specified, not
               built.

    Rows whose peak time is missing, or is one of the control timepoints, are given NaN rather
    than being silently coerced: version 1 crashed on exactly these rows, because .astype(str)
    turned NaN into the string 'nan' and the fold-change lookup then failed (§6.1).

    Args:
      df: Site-level DataFrame carrying the peak-timing column.
      time_definition: One of TIME_DEFINITIONS; see above.
      cell_line: Cell-line prefix, e.g. 'WT'.
      condition: Stimulation condition, e.g. 'EGF'.

    Returns:
      DataFrame copy with an added 'activation_time' column holding the timepoint label as a
      string, NaN where no valid timepoint could be assigned. Raises NotImplementedError for the
      two definitions that are specified but not built, and ValueError for an unknown one.
    """
    if time_definition not in TIME_DEFINITIONS:
        raise ValueError(f"time_definition must be one of {TIME_DEFINITIONS}, got {time_definition!r}")
    if time_definition == "max_step":
        raise NotImplementedError(
            "time_definition='max_step' is specified in section 3 of "
            "Claude_promts/dynamic_network_instrucitons.md but not implemented yet"
        )
    if time_definition == "onset":
        return add_onset_time(df,
                              cell_line=cell_line,
                              condition=condition,)

    out = df.copy()
    peak_column = f"{cell_line}_peak:FC_{condition}"
    if peak_column not in out.columns:
        raise KeyError(f"{peak_column!r} not found; add_peak_timepoints has not been run on this table")

    peak_time = out[peak_column]
    peak_time = peak_time.where(peak_time.notna(), np.nan)
    peak_label = peak_time.astype("string").str.replace(r"\.0$", "", regex=True)
    valid = peak_label.isin(list(STIMULATION_TIMEPOINTS))
    out["activation_time"] = peak_label.where(valid, pd.NA)
    out["time_definition"] = time_definition
    return out


def add_onset_time(df: pd.DataFrame,
                   cell_line: str = DEFAULT_CELL_LINE,
                   condition: str = DEFAULT_CONDITION,
                   fraction: float = 0.5,
                   snap_to_grid: bool = True,) -> pd.DataFrame:
    """
    Time a site by when its response *starts*, not by when it peaks (§3).

    Measured consequence of the peak definition, found in step 4: a kinase's own activation-loop
    site peaks *after* the kinase became active, because phosphorylation accumulates while the
    kinase is already working. ERK2's Y187 and T185 both peak at 10 min, so entering ERK2 through
    its own site forces t(ERK2) >= 10 under the temporal constraint, and ERK2 can then no longer
    explain the 23 substrates it has at 2 and 5 min. The optimiser escaped by paying the orphan
    price instead, which put ERK2 at t = 0 — also wrong, and it makes the orphan edges
    uninterpretable as "kinases with no upstream explanation".

    Onset time is the first moment the profile reaches `fraction` of its own peak, interpolated on
    the log10(t + 1) axis the project uses elsewhere for time. The profile is anchored at the
    starve control, which is identically 0 in fold-change space, and is read in the direction of
    its own response, so a down-regulated site is timed by when it starts falling.

    Args:
      df: Site-level DataFrame carrying the log2:FC columns of this condition.
      cell_line: Cell-line prefix, e.g. 'WT'.
      condition: Stimulation condition, e.g. 'EGF'.
      fraction: Fraction of the peak that defines onset. 0.5 is half-maximal.
      snap_to_grid: Snap the interpolated time to the nearest measured timepoint, so site times
        stay on the same grid as the latent kinase times. False keeps the continuous value.

    Returns:
      DataFrame copy with 'activation_time' (string label when snapped, else the continuous value),
      'onset_time_continuous' and 'time_definition'. Sites whose profile is entirely missing get NA.
    """
    out = df.copy()
    fc_columns = ColumnSpec.select(out,
                                   cell_lines=[cell_line],
                                   data_type="log2:FC",
                                   conditions=[f"_{condition}_"],)
    by_timepoint = {column.split("_")[3]: column for column in fc_columns}
    usable = [timepoint for timepoint in STIMULATION_TIMEPOINTS if timepoint in by_timepoint]
    times = np.array([0.0] + [float(timepoint) for timepoint in usable])
    axis = np.log10(times + 1.0)

    values = out[[by_timepoint[timepoint] for timepoint in usable]].to_numpy(dtype=float)
    profile = np.column_stack([np.zeros(len(out)), values])

    # Read every profile in the direction of its own response, so a falling site is timed the same
    # way as a rising one.
    finite = np.where(np.isfinite(profile), profile, 0.0)
    peak_index = np.abs(finite).argmax(axis=1)
    rows = np.arange(len(out))
    peak_value = finite[rows, peak_index]
    direction = np.where(peak_value < 0, -1.0, 1.0)
    oriented = finite * direction[:, None]
    threshold = fraction * np.abs(peak_value)

    onset = np.full(len(out), np.nan)
    for row in rows:
        if not np.isfinite(peak_value[row]) or peak_value[row] == 0:
            continue
        reached = np.where(oriented[row, 1:] >= threshold[row])[0]
        if len(reached) == 0:
            continue
        index = reached[0] + 1
        previous, current = oriented[row, index - 1], oriented[row, index]
        if current == previous:
            onset[row] = times[index]
            continue
        weight = (threshold[row] - previous) / (current - previous)
        onset[row] = 10.0 ** (axis[index - 1] + weight * (axis[index] - axis[index - 1])) - 1.0

    out["onset_time_continuous"] = onset
    if snap_to_grid:
        grid = np.array([float(timepoint) for timepoint in usable])
        snapped = np.full(len(out), np.nan)
        known = np.isfinite(onset)
        snapped[known] = grid[np.abs(onset[known, None] - grid[None, :]).argmin(axis=1)]
        labels = pd.Series(snapped, index=out.index).map(
            lambda value: str(int(value)) if np.isfinite(value) else pd.NA)
        out["activation_time"] = labels.astype("string")
    else:
        out["activation_time"] = pd.Series(onset, index=out.index)
    out["time_definition"] = "onset"
    return out


def add_peak_response(df: pd.DataFrame,
                      cell_line: str = DEFAULT_CELL_LINE,
                      condition: str = DEFAULT_CONDITION,) -> pd.DataFrame:
    """
    Read each site's fold change at its activation time, keeping the sign.

    The sign is what distinguishes a phosphorylation from a dephosphorylation. Version 1 kept
    only the absolute value, so a site losing phosphorylation was rewarded exactly like a site
    gaining it and was allowed to switch its kinase on (§2.3, §4.1).

    The fold-change columns are selected with ColumnSpec rather than by string formatting, and
    the value is taken per row with a vectorised lookup rather than a Python loop.

    Args:
      df: DataFrame carrying an 'activation_time' column as added by add_activation_time.
      cell_line: Cell-line prefix, e.g. 'WT'.
      condition: Stimulation condition, e.g. 'EGF'.

    Returns:
      DataFrame copy with three added columns: peak_log2_fc_vs_starve (signed),
      peak_abs_log2_fc_vs_starve (the prize) and peak_direction ('up' / 'down', NA where the
      activation time is missing).
    """
    out = df.copy()
    fc_columns = ColumnSpec.select(out,
                                   cell_lines=[cell_line],
                                   data_type="log2:FC",
                                   conditions=[f"_{condition}_"],)
    by_timepoint = {column.split("_")[3]: column for column in fc_columns}
    usable = [timepoint for timepoint in STIMULATION_TIMEPOINTS if timepoint in by_timepoint]
    missing = set(STIMULATION_TIMEPOINTS) - set(usable)
    if missing:
        raise KeyError(f"No log2:FC column for timepoint(s) {sorted(missing)} of condition {condition!r}")

    values = out[[by_timepoint[timepoint] for timepoint in usable]].to_numpy(dtype=float)

    # The prize is the size of the response, which is the largest |log2FC| over the stimulation
    # timepoints — deliberately independent of activation_time. Under TIME_DEFINITION='peak_fc'
    # the two coincide by construction, but under 'onset' the site is *timed* by when its response
    # starts while still being *prized* by how far it eventually moves.
    finite = np.where(np.isfinite(values), values, 0.0)
    peak_index = np.abs(finite).argmax(axis=1)
    picked = finite[np.arange(len(out)), peak_index]
    picked[~np.isfinite(values).any(axis=1)] = np.nan
    picked[picked == 0] = np.nan

    out["peak_log2_fc_vs_starve"] = picked
    out["peak_abs_log2_fc_vs_starve"] = np.abs(picked)
    out["peak_direction"] = pd.Series(np.where(picked > 0, "up", "down"),
                                      index=out.index,).where(~np.isnan(picked), pd.NA)
    return out


# ── 4. Site identifiers ──────────────────────────────────────────────────────────────────


def explode_multisite_peptides(df: pd.DataFrame,) -> pd.DataFrame:
    """
    Split a multi-phosphorylated peptide into one candidate row per residue.

    PhosSites holds ';'-separated residues, e.g. 'S416;S417'. Version 1 built its lookup key as
    protein_Id + '_' + PhosSites, producing 'Q00000_S416;S417', which can never match a
    kinase-substrate target of the form 'Q00000_S416'. Every multi-site peptide was therefore
    silently unmatchable — on hme1_2 that is 5959 of 34675 localized rows (§2.4).

    Exploding makes each residue addressable. The ambiguity does not disappear: the fold change
    of an exploded row is the fold change of the whole peptide, so the same measurement is
    attributed to each of its residues. n_sites_on_peptide records how many residues shared it.

    Args:
      df: Peptide-level DataFrame carrying protein_Id, protein_name and PhosSites.

    Returns:
      DataFrame copy with one row per (peptide, residue), carrying the added columns phosphosite
      (e.g. 'S416'), n_sites_on_peptide, kinsub_site_id ('{protein_Id}_{residue}', the key used
      to match the kinase-substrate table) and protein_site_id ('{protein_name}_{residue}', the
      human-readable node label).
    """
    out = df.copy()
    out["n_sites_on_peptide"] = out["PhosSites"].astype(str).str.count(";") + 1
    out["phosphosite"] = out["PhosSites"].astype(str).str.split(";")
    out = out.explode("phosphosite", ignore_index=True)
    out["phosphosite"] = out["phosphosite"].str.strip()

    protein_name = out["protein_name"].fillna(out["protein_Id"]).astype(str)
    out["kinsub_site_id"] = out["protein_Id"].astype(str) + "_" + out["phosphosite"]
    out["protein_site_id"] = protein_name + "_" + out["phosphosite"]
    return out


def collapse_to_sites(df: pd.DataFrame,) -> pd.DataFrame:
    """
    Reduce several peptide measurements of the same phosphosite to one row.

    A site can be measured on more than one peptide (different missed cleavages, different
    co-modifications). The row kept is the one with the largest absolute peak fold change, with
    replicate count and reference intensity as tie-breakers, matching the version 1 rule so the
    two site tables stay comparable.

    Args:
      df: Exploded DataFrame carrying kinsub_site_id, peak_abs_log2_fc_vs_starve, n:reps and
        ReferenceIntensity.

    Returns:
      DataFrame copy with one row per kinsub_site_id, with an added n_source_rows column counting
      the peptide rows that mapped to that site, sorted by descending prize.
    """
    out = df.copy()
    out = out.join(out.groupby("kinsub_site_id").size().rename("n_source_rows"),
                   on="kinsub_site_id",)
    out = out.sort_values(["kinsub_site_id",
                           "peak_abs_log2_fc_vs_starve",
                           "n:reps",
                           "ReferenceIntensity"],
                          ascending=[True, False, False, False],
                          na_position="last",)
    out = out.drop_duplicates("kinsub_site_id", keep="first")
    return out.sort_values("peak_abs_log2_fc_vs_starve", ascending=False).reset_index(drop=True)


# ── 5. The step-1 pipeline ───────────────────────────────────────────────────────────────


def build_site_table(path: Path,
                     max_ffdr: float | None = DEFAULT_MAX_FFDR,
                     time_definition: str = DEFAULT_TIME_DEFINITION,
                     cell_line: str = DEFAULT_CELL_LINE,
                     condition: str = DEFAULT_CONDITION,
                     min_reps: int | None = None,) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the per-phosphosite table the network optimisation selects from, with a filter ledger.

    Runs the whole of step 1 in order: load, drop contaminants and unlocalized peptides, apply the
    limma responsiveness gate, assign the activation time, read the signed peak fold change,
    explode multi-site peptides and collapse to one row per site.

    Args:
      path: Path to the processed .tsv table.
      max_ffdr: Omnibus-F FDR cutoff. None switches the responsiveness gate off.
      time_definition: Activation-time definition, see add_activation_time.
      cell_line: Cell-line prefix, e.g. 'WT'.
      condition: Stimulation condition, e.g. 'EGF'.
      min_reps: Optional minimum n:reps. None applies no replicate filter, since the FFDR gate
        already drops sites limma could not test.

    Returns:
      Tuple of (site table, ledger). The site table has one row per kinsub_site_id, sorted by
      descending peak_abs_log2_fc_vs_starve. The ledger is a DataFrame with one row per filter
      step holding the step name, the rows remaining and the rows dropped, so the notebook can
      print exactly where the data went.
    """
    ledger = []

    def record(step: str,
               frame: pd.DataFrame,
               previous: int | None,) -> int:
        """
        Append one row to the filter ledger and return the current row count.

        Args:
          step: Human-readable name of the step just applied.
          frame: DataFrame as it stands after that step.
          previous: Row count before the step, or None for the first entry.

        Returns:
          The number of rows in frame.
        """
        n = len(frame)
        ledger.append({"step": step,
                       "rows": n,
                       "dropped": None if previous is None else previous - n,})
        return n

    df = load_processed_table(path)
    n = record("loaded", df, None)

    for flag in ("CON", "REV",):
        if flag in df.columns:
            df = df[~df[flag].astype(bool)].copy()
    n = record("contaminants and decoys removed", df, n)

    df = filter_localized(df)
    n = record("localized phosphosites only", df, n)

    if min_reps is not None:
        df = df[df["n:reps"] >= min_reps].copy()
        n = record(f"n:reps >= {min_reps}", df, n)

    df = filter_responsive(df,
                           max_ffdr=max_ffdr,
                           cell_line=cell_line,
                           condition=condition,)
    n = record(f"limma omnibus F, FFDR <= {max_ffdr}" if max_ffdr is not None else "responsiveness gate skipped",
               df,
               n,)

    df = add_activation_time(df,
                             time_definition=time_definition,
                             cell_line=cell_line,
                             condition=condition,)
    df = df[df["activation_time"].notna()].copy()
    n = record(f"valid activation time ({time_definition})", df, n)

    df = add_peak_response(df,
                           cell_line=cell_line,
                           condition=condition,)
    df = df[df["peak_log2_fc_vs_starve"].notna()].copy()
    n = record("peak fold change present", df, n)

    df = explode_multisite_peptides(df)
    n = record("multi-site peptides exploded to residues", df, n)

    site_table = collapse_to_sites(df)
    record("collapsed to one row per phosphosite", site_table, n)

    return site_table, pd.DataFrame(ledger)


# ── 6. Checks on the site table ──────────────────────────────────────────────────────────


def trim_site_table(site_table: pd.DataFrame,
                    cell_line: str = DEFAULT_CELL_LINE,
                    condition: str = DEFAULT_CONDITION,
                    keep_profiles: bool = True,) -> pd.DataFrame:
    """
    Reduce the site table to the columns the network work actually needs.

    The processed source table carries ~500 columns, nearly all of them other conditions,
    replicates and intermediate statistics. Writing all of them to disk produces a ~117 MB file
    per run for ~15000 sites. This keeps the identifiers, the response, the annotation columns the
    sign rule and the drawing use, and optionally the fold-change and step profiles of the
    condition being modelled.

    Args:
      site_table: Site table as returned by build_site_table.
      cell_line: Cell-line prefix, e.g. 'WT'.
      condition: Stimulation condition, e.g. 'EGF'.
      keep_profiles: When True, keep the log2:FC and log2:step columns of this condition, so the
        temporal profile of a node can be plotted without re-reading the source table.

    Returns:
      DataFrame copy holding only the retained columns, in a stable order.
    """
    identifiers = ["kinsub_site_id", "protein_site_id", "protein_Id", "protein_name",
                   "phosphosite", "PhosSites", "site", "description"]
    response = ["activation_time", "time_definition", "onset_time_continuous",
                "peak_log2_fc_vs_starve", "peak_abs_log2_fc_vs_starve", "peak_direction"]
    quality = ["n:reps", "n_sites_on_peptide", "n_source_rows", "NumPhos", "LocalizedNumPhos",
               "MaxPepProb", "ReferenceIntensity", "peptide_seq", "SequenceWindow"]
    annotation = ["functional_score", "ERK_motif", "ON_FUNCTION", "ON_PROCESS",
                  "ON_PROT_INTERACT"]
    statistics = [f"{cell_line}_log2:FFDR_{condition}_omnibus",
                  f"{cell_line}_log2:Fpvalue_{condition}_omnibus",
                  f"{cell_line}_peak:FC_{condition}",
                  f"{cell_line}_peak:step_{condition}"]

    profiles = []
    if keep_profiles:
        for data_type in ("log2:FC", "log2:step",):
            profiles += ColumnSpec.select(site_table,
                                          cell_lines=[cell_line],
                                          data_type=data_type,
                                          conditions=[f"_{condition}_"],)

    keep = identifiers + response + quality + annotation + statistics + profiles
    return site_table[[column for column in keep if column in site_table.columns]].copy()


def activation_loop_report(site_table: pd.DataFrame,
                           pairs: tuple = ACTIVATION_LOOP_PAIRS,) -> pd.DataFrame:
    """
    Test the activation-time definition against residues known to be phosphorylated together.

    Both residues of a kinase activation loop are modified by the same event, so any sound time
    definition must give them the same activation time. This is the objective test proposed in
    §3 for choosing between peak_fc, max_step and onset. Version 1 fails it: ERK1 Y204 is placed
    at 5 min and T202 at 90 min, which then drags ERK1's latent activation time to 90.

    Args:
      site_table: Site table as returned by build_site_table.
      pairs: Iterable of (protein_Id, residue_a, residue_b, description) tuples.

    Returns:
      DataFrame with one row per pair, reporting the activation time and signed peak fold change
      found for each residue, whether both were detected, and whether their times agree.
    """
    indexed = site_table.set_index("kinsub_site_id")
    rows = []
    for protein_id, residue_a, residue_b, description in pairs:
        record = {"protein_Id": protein_id,
                  "description": description,
                  "site_a": f"{protein_id}_{residue_a}",
                  "site_b": f"{protein_id}_{residue_b}",}
        for label, residue in (("a", residue_a,), ("b", residue_b,),):
            key = f"{protein_id}_{residue}"
            if key in indexed.index:
                row = indexed.loc[key]
                record[f"time_{label}"] = row["activation_time"]
                record[f"log2fc_{label}"] = round(float(row["peak_log2_fc_vs_starve"]), 3)
            else:
                record[f"time_{label}"] = pd.NA
                record[f"log2fc_{label}"] = pd.NA
        record["both_detected"] = pd.notna(record["time_a"]) and pd.notna(record["time_b"])
        record["times_agree"] = record["both_detected"] and record["time_a"] == record["time_b"]
        rows.append(record)
    return pd.DataFrame(rows)


def site_table_summary(site_table: pd.DataFrame,) -> pd.DataFrame:
    """
    Describe the composition of the site table, one metric per row.

    Reports the counts a reader needs to judge what the network will be built from: how many
    sites survive, how they split by direction and by activation time, how many come from
    multi-site peptides, and how the prize is distributed.

    Args:
      site_table: Site table as returned by build_site_table.

    Returns:
      Two-column DataFrame of metric name and value.
    """
    prize = site_table["peak_abs_log2_fc_vs_starve"]
    metrics = [("sites", len(site_table),),
               ("proteins", site_table["protein_Id"].nunique(),),
               ("up-regulated", int((site_table["peak_direction"] == "up").sum()),),
               ("down-regulated", int((site_table["peak_direction"] == "down").sum()),),
               ("from multi-site peptides", int((site_table["n_sites_on_peptide"] > 1).sum()),),
               ("measured on >1 peptide", int((site_table["n_source_rows"] > 1).sum()),),
               ("median |log2FC|", round(float(prize.median()), 3),),
               ("max |log2FC|", round(float(prize.max()), 3),),
               ("|log2FC| >= 1", int((prize >= 1).sum()),),]
    for timepoint in STIMULATION_TIMEPOINTS:
        metrics.append((f"peak at {timepoint} min",
                        int((site_table["activation_time"] == timepoint).sum()),))
    return pd.DataFrame(metrics, columns=["metric", "value"])


# ── 7. Kinase-substrate edges ────────────────────────────────────────────────────────────


def load_kinase_substrate_edges(path: Path,
                                site_ids: set,
                                resource_regex: str | None = DEFAULT_RESOURCE_REGEX,
                                max_kinases_per_site: int | None = DEFAULT_MAX_KINASES_PER_SITE,
                                min_score: float = 0.0,) -> tuple[pd.DataFrame, dict]:
    """
    Load the kinase-substrate edges pointing at our sites, and cut the prediction fan-out.

    The kinase-substrate table is dominated by lenient motif predictions, so without filtering a
    site carries ~43 candidate kinases and the ILP has ~570000 binary edge variables (§4.0). The
    two parameters below are what keep the graph small; they cut edges rather than sites, which
    is the cheap direction — a 43-way prediction was never evidence.

    Args:
      path: Path to the pickled kinase-substrate DataFrame with columns source (kinase UniProt),
        target (site as '{protein_Id}_{residue}'), score and resource.
      site_ids: Site identifiers to keep edges for, i.e. the kinsub_site_id column of the site
        table.
      resource_regex: Regular expression matched against the resource column; only matching rows
        are kept. None keeps every resource.
      max_kinases_per_site: Keep only this many highest-scoring kinases per site. None keeps all.
      min_score: Drop edges scoring below this value.

    Returns:
      Tuple of (edge DataFrame with columns source, target, score and resources, statistics dict
      reporting the row and edge counts at each stage).
    """
    with Path(path).open("rb") as handle:
        kinsub = pickle.load(handle)

    stats = {"kinsub_rows": len(kinsub),
             "kinsub_kinases": int(kinsub["source"].nunique()),}

    edges = kinsub[kinsub["target"].isin(site_ids)].copy()
    stats["rows_targeting_our_sites"] = len(edges)

    if resource_regex:
        keep = edges["resource"].astype(str).str.contains(resource_regex, regex=True, na=False)
        edges = edges.loc[keep].copy()
        stats["rows_after_resource_filter"] = len(edges)

    edges = (edges.groupby(["source", "target"], as_index=False)
                  .agg(score=("score", "max"),
                       resources=("resource", lambda values: ";".join(sorted(set(map(str, values)))),),))
    stats["unique_edges_before_topk"] = len(edges)

    if min_score > 0:
        edges = edges[edges["score"] >= min_score].copy()
    if max_kinases_per_site is not None:
        edges = (edges.sort_values("score", ascending=False)
                      .groupby("target", sort=False)
                      .head(max_kinases_per_site)
                      .copy())

    edges = add_resource_costs(edges)
    stats["unique_edges"] = len(edges)
    stats["edges_by_class"] = edges["resource_class"].value_counts().to_dict()
    stats["sites_with_a_kinase"] = int(edges["target"].nunique())
    stats["kinases_used"] = int(edges["source"].nunique())
    stats["edges_per_site"] = round(len(edges) / max(edges["target"].nunique(), 1), 2)
    return edges.reset_index(drop=True), stats


def classify_resource(resources: str,) -> str:
    """
    Reduce a ';'-joined resource string to the single strongest evidence class it contains.

    An edge is usually supported by several resources at once, e.g.
    'combined-Lenient;phosformer-Moderate'. The edge is priced by the best evidence available for
    it, so the strongest class present wins: literature, then Strict, Moderate, Lenient.

    Args:
      resources: The resource string carried by the edge, as aggregated by
        load_kinase_substrate_edges.

    Returns:
      One of 'literature', 'strict', 'moderate', 'lenient'. Falls back to 'lenient' when nothing
      recognisable is present, which is the most expensive class and therefore the safe default.
    """
    text = str(resources).lower()
    for label in RESOURCE_CLASS_ORDER:
        if label in text:
            return label
    return "lenient"


def add_resource_costs(edges: pd.DataFrame,
                       cost_by_class: dict | None = None,
                       within_class_span: float = WITHIN_CLASS_SPAN,) -> pd.DataFrame:
    """
    Price each kinase-substrate edge by its evidence class rather than by its raw score.

    Two things are wrong with costing an edge as 1 - score. The score means different things in
    different resources — kinlib never scores below 0.90, phosformer sits near 0.61, and literature
    rows are fixed at 0.0039 — so the scales are not comparable, and the curated rows come out as
    the most expensive edges in the graph. Here the class sets the base price and the score only
    orders edges inside their own class, where the scale is at least self-consistent.

    Args:
      edges: Edge DataFrame with source, target, score and resources columns.
      cost_by_class: Base cost per evidence class. None uses RESOURCE_CLASS_COSTS.
      within_class_span: Width of the within-class term added on top of the base cost. The best
        edge of a class pays the base, the worst pays base + span, so classes never overlap as
        long as the span is smaller than the gap between base costs.

    Returns:
      DataFrame copy with two added columns: resource_class and cost.
    """
    cost_by_class = RESOURCE_CLASS_COSTS if cost_by_class is None else cost_by_class
    out = edges.copy()
    out["resource_class"] = out["resources"].map(classify_resource)
    out["cost"] = out["resource_class"].map(cost_by_class).astype(float)

    # Rank within the class, so incomparable score scales are never compared across classes.
    # 'literature' is deliberately left flat: its score column is a constant 0.0039 for most rows
    # and reflects nothing about how well established the interaction is, so ranking on it would
    # order curated edges by noise. A curated edge is a curated edge.
    for label, group in out.groupby("resource_class"):
        if label == "literature":
            continue
        scores = group["score"].astype(float)
        span = scores.max() - scores.min()
        if span <= 0 or len(group) < 2:
            continue
        normalised = (scores - scores.min()) / span
        out.loc[group.index, "cost"] = cost_by_class[label] + within_class_span * (1.0 - normalised)
    return out


def add_propagation_rule(site_table: pd.DataFrame,
                         kinase_ids: set,) -> pd.DataFrame:
    """
    Decide, per site, whether it may pass activation on to the kinase it sits on.

    Version 1 added a site to kinase edge whenever the modified protein was a kinase, whatever the
    site does, so a known inhibitory site propagated activation and a dephosphorylation was
    treated like a phosphorylation. The rule here (§4.1) uses the PhosphoSitePlus ON_FUNCTION
    annotation together with the measured direction:

      up   + activity induced (or unknown)  -> may propagate
      up   + activity inhibited             -> may not; the site can still be a feedback leaf
      down + activity inhibited             -> may propagate, this is loss of inhibition
      down + activity induced               -> may not, the kinase is being switched off

    Only 1174 of 14797 sites carry any ON_FUNCTION annotation, so most edges end up flagged
    'none'. That is the honest state of the evidence and the figure should show it.

    Args:
      site_table: Site table carrying protein_Id, peak_direction and ON_FUNCTION.
      kinase_ids: UniProt accessions that appear as kinases in the kinase-substrate table.

    Returns:
      DataFrame copy with three added columns: is_kinase_protein, sign_evidence ('induced',
      'inhibited', 'ambiguous' or 'none') and can_propagate (bool).
    """
    out = site_table.copy()
    annotation = out["ON_FUNCTION"].astype(str).str.lower()
    induced = annotation.str.contains("activity, induced", na=False)
    inhibited = annotation.str.contains("activity, inhibited", na=False)

    evidence = np.where(induced & inhibited, "ambiguous",
                        np.where(induced, "induced",
                                 np.where(inhibited, "inhibited", "none",),),)
    out["sign_evidence"] = evidence
    out["is_kinase_protein"] = out["protein_Id"].isin(kinase_ids)

    going_up = out["peak_direction"] == "up"
    activating = ((going_up & (out["sign_evidence"].isin(["induced", "none", "ambiguous"])))
                  | (~going_up & (out["sign_evidence"].isin(["inhibited", "none", "ambiguous"]))))
    out["can_propagate"] = out["is_kinase_protein"] & activating
    return out


# ── 8. The candidate graph ───────────────────────────────────────────────────────────────


def build_candidate_graph(site_table: pd.DataFrame,
                          edges: pd.DataFrame,
                          root_kinase: str = EGFR_UNIPROT,
                          root_name: str = ROOT,
                          tf_protein_ids: set | None = None,
                          tf_bonus: float = 0.0,) -> tuple[nx.DiGraph, dict]:
    """
    Assemble the graph the ILP selects a subnetwork from.

    Three kinds of edge: ROOT to the receptor kinase, kinase to phosphosite from the
    kinase-substrate table, and phosphosite to kinase where the site sits on a kinase protein and
    the sign rule lets it propagate. There is deliberately **no SINK**: since version 2 allows any
    site to be a leaf (§4.1), a path may stop anywhere, so transcription-factor sites are rewarded
    with a prize bonus instead of being the only permitted terminal.

    Unlike version 1 the graph is **not** pruned to nodes reachable from ROOT, because step 3 will
    add a virtual root for orphan kinases; the reachable set is reported instead, so the effect of
    that change is visible.

    Args:
      site_table: Site table after add_propagation_rule.
      edges: Kinase-substrate edges as returned by load_kinase_substrate_edges.
      root_kinase: UniProt accession of the receptor the network is rooted at.
      root_name: Key of the artificial root node.
      tf_protein_ids: Optional set of transcription-factor accessions receiving the prize bonus.
      tf_bonus: Prize added to a site sitting on a transcription factor.

    Returns:
      Tuple of (directed graph, statistics dict). Phosphosite nodes carry node_type, prize,
      activation_time, the signed fold change, direction, sign_evidence and the display label;
      kinase nodes carry node_type and latent_time.
    """
    tf_protein_ids = set(tf_protein_ids or set())
    graph = nx.DiGraph()
    graph.add_node(root_name, node_type="root", dynamic_exempt=True)

    kinase_ids = set(edges["source"].astype(str))
    kinase_ids.add(root_kinase)
    for kinase_id in kinase_ids:
        graph.add_node(kinase_id,
                       node_type="kinase",
                       latent_time=kinase_id != root_kinase,
                       activation_time=0.0 if kinase_id == root_kinase else None,
                       prize=0.0,)
    graph.add_edge(root_name, root_kinase, edge_type="root_to_kinase", weight=1.0, cost=0.0)

    sites = site_table.set_index("kinsub_site_id")
    n_tf_bonus = 0
    for site_id, row in sites.iterrows():
        prize = float(row["peak_abs_log2_fc_vs_starve"])
        if tf_bonus and str(row["protein_Id"]) in tf_protein_ids:
            prize += tf_bonus
            n_tf_bonus += 1
        graph.add_node(site_id,
                       node_type="phosphosite",
                       prize=prize,
                       activation_time=float(row["activation_time"]),
                       protein_site_id=row["protein_site_id"],
                       protein_id=row["protein_Id"],
                       peak_log2_fc_vs_starve=float(row["peak_log2_fc_vs_starve"]),
                       peak_direction=row["peak_direction"],
                       sign_evidence=row["sign_evidence"],
                       latent_time=False,)

    for row in edges.itertuples(index=False):
        if row.source in graph and row.target in graph:
            graph.add_edge(row.source,
                           row.target,
                           edge_type="kinase_to_phosphosite",
                           weight=float(row.score),
                           cost=float(getattr(row, "cost", 1.0 - float(row.score))),
                           resource_class=getattr(row, "resource_class", None),
                           resources=row.resources,)

    n_propagating = 0
    for site_id, row in sites.iterrows():
        if bool(row["can_propagate"]) and str(row["protein_Id"]) in graph:
            graph.add_edge(site_id,
                           str(row["protein_Id"]),
                           edge_type="phosphosite_to_kinase",
                           weight=1.0,
                           cost=0.0,
                           sign_evidence=row["sign_evidence"],)
            n_propagating += 1

    reachable = nx.descendants(graph, root_name) | {root_name}
    stats = {"nodes": graph.number_of_nodes(),
             "edges": graph.number_of_edges(),
             "kinases": len(kinase_ids),
             "phosphosites": len(sites),
             "sites_that_can_propagate": n_propagating,
             "sites_blocked_by_sign_rule": int((site_table["is_kinase_protein"]
                                                & ~site_table["can_propagate"]).sum()),
             "sites_with_tf_bonus": n_tf_bonus,
             "reachable_from_root": len(reachable),
             "unreachable_from_root": graph.number_of_nodes() - len(reachable),}
    return graph, stats


# ── 9. The ILP ───────────────────────────────────────────────────────────────────────────


def _incidence_matrices(n_nodes: int,
                        source_indices: np.ndarray,
                        target_indices: np.ndarray,):
    """
    Build the sparse in- and out-incidence matrices used by the flow constraints.

    Args:
      n_nodes: Number of nodes in the candidate graph.
      source_indices: Node index of the source of each edge.
      target_indices: Node index of the target of each edge.

    Returns:
      Tuple of (A_in, A_out) scipy CSR matrices of shape (n_nodes, n_edges), where A_in[v, e] is 1
      when edge e enters node v and A_out[v, e] is 1 when edge e leaves it.
    """
    from scipy.sparse import csr_matrix

    n_edges = len(source_indices)
    ones = np.ones(n_edges)
    columns = np.arange(n_edges)
    a_in = csr_matrix((ones, (target_indices, columns),), shape=(n_nodes, n_edges,),)
    a_out = csr_matrix((ones, (source_indices, columns),), shape=(n_nodes, n_edges,),)
    return a_in, a_out


def select_network(graph: nx.DiGraph,
                   node_penalty: float,
                   root_name: str = ROOT,
                   allowed_times: tuple = ALLOWED_ACTIVATION_TIMES,
                   allow_equal_time: bool = True,
                   latent_time_penalty: float = 1e-6,
                   solver: str | None = None,
                   mip_gap: float = 0.05,
                   time_limit: int = 300,
                   max_cycle_rounds: int = 5,
                   max_cycles_per_round: int = 200,
                   verbose: bool = False,) -> dict:
    """
    Select a temporally coherent, rooted subnetwork by solving a prize-collecting ILP.

    Two things differ from version 1's formulation, both from §5:

    **Leaves are allowed.** Version 1 required every selected node to have a selected outgoing
    edge, so every path had to reach a transcription factor. Since a site only has an out-edge
    when its protein is a kinase or a TF, every other site was structurally unselectable however
    large its prize, and ~98% of the prize on offer was a constant no solution could reduce. Here
    a path may stop anywhere, which also makes feedback expressible: a kinase phosphorylating a
    site on an upstream protein is an ordinary forward-in-time edge once that site need not
    continue into its own kinase node.

    **Connectivity is enforced by a single-commodity flow** rather than by MTZ distance labels.
    The root ships one unit of flow to every selected node, flow may only travel along selected
    edges, and each selected node consumes one unit. This gives the same rooted guarantee with a
    tighter LP relaxation, which matters at this graph size. Acyclicity follows because every edge
    carries a small positive cost, so a cycle is never free; it is asserted afterwards rather than
    constrained.

    Args:
      graph: Candidate graph from build_candidate_graph. Nodes carry node_type, prize,
        activation_time and latent_time; edges carry weight and edge_type.
      node_penalty: Cost charged for every selected biological node. Larger values give smaller
        networks.
      root_name: Key of the artificial root, which is always selected and pays no node cost.
      allowed_times: Discrete grid a latent kinase activation time may take.
      allow_equal_time: When True an edge may join two nodes with the same activation time; when
        False the source must be strictly earlier.
      latent_time_penalty: Small weight on the sum of latent times, breaking ties towards earlier
        activation.
      solver: CVXPY solver name. None picks GUROBI if installed, else SCIPY (HiGHS).
      mip_gap: Relative MIP optimality gap at which the solver may stop.
      time_limit: Solver time limit in seconds, applied to each solve of the cycle-elimination
        loop, so the total time can be up to (1 + max_cycle_rounds) times this.
      max_cycle_rounds: How many times a cycle found in the selection may be forbidden and the
        problem re-solved. 0 disables cycle elimination and can return a non-DAG.
      max_cycles_per_round: Cap on how many cycles are enumerated and forbidden per round, so a
        pathological selection cannot stall the loop in nx.simple_cycles.
      verbose: Passed to the solver.

    Returns:
      Dict holding the selected subgraph ('subgraph'), the solver 'status', 'objective_value',
      the achieved gap where the solver reports one, the selected node and edge lists, the chosen
      latent times, the objective decomposition and the candidate-graph counts. A status other
      than 'optimal' must be treated as a failed run, not as a small network.
    """
    if solver is None:
        installed = cp.installed_solvers()
        solver = "GUROBI" if "GUROBI" in installed else "SCIPY"

    nodes = list(graph.nodes())
    node_index = {node: i for i, node in enumerate(nodes)}
    n_nodes = len(nodes)

    latent_nodes = [node for node, data in graph.nodes(data=True) if data.get("latent_time")]
    latent_index = {node: i for i, node in enumerate(latent_nodes)}
    times = np.array([np.nan if data.get("activation_time") is None else float(data["activation_time"])
                      for _, data in graph.nodes(data=True)])

    # Edges between two nodes whose times are both known and already out of order cannot appear in
    # any feasible solution, so they are dropped before the ILP is built rather than left to it.
    kept_edges = []
    dropped_by_time = 0
    for u, v, data in graph.edges(data=True):
        u_time, v_time = times[node_index[u]], times[node_index[v]]
        both_known = not (np.isnan(u_time) or np.isnan(v_time))
        if both_known and u not in (root_name,) and v not in (root_name,):
            if (u_time > v_time) if allow_equal_time else (u_time >= v_time):
                dropped_by_time += 1
                continue
        kept_edges.append((u, v, data,))

    edges = [(u, v,) for u, v, _ in kept_edges]
    n_edges = len(edges)
    source_indices = np.array([node_index[u] for u, _ in edges])
    target_indices = np.array([node_index[v] for _, v in edges])
    # An edge may carry an explicit 'cost' — the orphan-entry price omega (§4.2) and the PPI charge
    # (§4.3) are set that way. Everything else is priced by its interaction confidence as 1 - score.
    edge_costs = np.array([float(data["cost"]) if "cost" in data
                           else 1.0 - float(data.get("weight", 1.0))
                           for _, _, data in kept_edges]) + EDGE_EPSILON

    prizes = np.array([float(data.get("prize", 0.0)) for _, data in graph.nodes(data=True)])
    root_i = node_index[root_name]
    biological = np.array([i for i in range(n_nodes) if i != root_i])
    root_selector = np.zeros(n_nodes)
    root_selector[root_i] = 1.0

    node_vars = cp.Variable(n_nodes, boolean=True)
    edge_vars = cp.Variable(n_edges, boolean=True)
    flow_vars = cp.Variable(n_edges, nonneg=True)

    # Per-edge flow capacity. The textbook bound is |V| on every edge, which is hopelessly weak
    # here: 92% of edges point into a node with no outgoing edge, and such an edge can only ever
    # carry the single unit its head consumes. Capping those at 1 tightens the LP relaxation by
    # four orders of magnitude on most of the graph, which is the difference between a solve and a
    # hang once orphan edges make every site reachable (§4.2).
    out_degree = np.zeros(n_nodes)
    np.add.at(out_degree, source_indices, 1.0)
    capacities = np.where(out_degree[target_indices] == 0, 1.0, float(n_nodes))

    a_in, a_out = _incidence_matrices(n_nodes, source_indices, target_indices)
    constraints = [node_vars[root_i] == 1,
                   edge_vars <= node_vars[source_indices],
                   edge_vars <= node_vars[target_indices],
                   flow_vars <= cp.multiply(capacities, edge_vars),
                   (a_in - a_out) @ flow_vars == node_vars - root_selector * cp.sum(node_vars),
                   (a_in @ edge_vars)[biological] >= node_vars[biological],]

    # Two-cycles have to be forbidden explicitly. A kinase that phosphorylates a site on its own
    # protein gives K -> S -> K, and with equal activation times allowed that pair is temporally
    # legal; the only thing arguing against it is the epsilon edge cost, which is far inside the
    # MIP gap, so the solver returns it. Autophosphorylation is real biology, but it is drawn as a
    # feedback annotation (tag_feedback_edges) rather than carried as a cycle in the selection.
    edge_position = {edge: i for i, edge in enumerate(edges)}
    forward, backward = [], []
    for (u, v,), i in edge_position.items():
        j = edge_position.get((v, u,))
        if j is not None and i < j:
            forward.append(i)
            backward.append(j)
    if forward:
        constraints.append(edge_vars[np.array(forward)] + edge_vars[np.array(backward)] <= 1)

    latent_time_vars = None
    if latent_nodes:
        from scipy.sparse import csr_matrix

        grid = np.array(sorted(float(value) for value in allowed_times))
        choice_vars = cp.Variable((len(latent_nodes), len(grid),), boolean=True)
        constraints.append(cp.sum(choice_vars, axis=1) == 1)
        latent_time_vars = choice_vars @ grid

        rows = [node_index[node] for node in latent_nodes]
        selector = csr_matrix((np.ones(len(latent_nodes)), (rows, np.arange(len(latent_nodes)),),),
                              shape=(n_nodes, len(latent_nodes),),)
        fixed = np.nan_to_num(times, nan=0.0)
        node_times = fixed + selector @ latent_time_vars

        timed = np.array([i for i, (u, v,) in enumerate(edges)
                          if u != root_name and v != root_name])
        if len(timed):
            time_M = float(grid.max() - grid.min()) + 1.0
            offset = 0.0 if allow_equal_time else 1.0
            constraints.append(node_times[source_indices[timed]] + offset
                               <= node_times[target_indices[timed]] + time_M * (1 - edge_vars[timed]))

    missed_prize = cp.sum(cp.multiply(prizes, 1 - node_vars))
    edge_cost = cp.sum(cp.multiply(edge_costs, edge_vars))
    node_cost = node_penalty * cp.sum(node_vars[biological])
    latent_cost = latent_time_penalty * cp.sum(latent_time_vars) if latent_time_vars is not None else 0.0

    objective = cp.Minimize(missed_prize + edge_cost + node_cost + latent_cost)
    solve_kwargs = {"solver": solver, "verbose": verbose}
    if solver == "GUROBI":
        solve_kwargs.update({"TimeLimit": time_limit, "MIPGap": mip_gap})
    elif solver == "SCIPY":
        solve_kwargs["scipy_options"] = {"time_limit": float(time_limit),
                                         "mip_rel_gap": float(mip_gap),
                                         "disp": bool(verbose),}

    # Lazy cycle elimination. The flow constraints guarantee rootedness but not acyclicity, and the
    # pairwise constraint above only blocks two-cycles. A longer cycle whose nodes all share one
    # activation time is temporally legal and costs only the epsilon edge charge, which is far
    # inside the MIP gap — so the solver returns it and the selection is not a DAG. Rather than
    # enumerate every cycle of a 15000-node graph up front, solve, look at the cycles that actually
    # appear in the selection, forbid exactly those, and re-solve. Cycles are rare, so this
    # normally converges in one or two rounds.
    cycle_constraints: list = []
    rounds = 0
    cycles_removed = 0
    selected_edges: list = []
    subgraph = nx.DiGraph()
    solver_error = None
    while True:
        problem = cp.Problem(objective, constraints + cycle_constraints)
        # A time-limited MILP can end with no incumbent at all, and CVXPY raises rather than
        # returning a status for that. Treat it as a failed run reported in the result, not as an
        # exception: a scan over parameters must not lose its completed rows because one cell was
        # too hard. Anything other than status 'optimal' is not a small network, it is no answer.
        try:
            problem.solve(**solve_kwargs)
        except cp.error.SolverError as error:
            solver_error = str(error)
            break

        if edge_vars.value is None:
            selected_edges = []
        else:
            selected_edges = [edge for edge, value in zip(edges, edge_vars.value) if value > 0.5]

        selected_set = set(selected_edges)
        subgraph = nx.DiGraph()
        for u, v, data in kept_edges:
            if (u, v,) in selected_set:
                subgraph.add_node(u, **graph.nodes[u])
                subgraph.add_node(v, **graph.nodes[v])
                subgraph.add_edge(u, v, **data)

        if rounds >= max_cycle_rounds or nx.is_directed_acyclic_graph(subgraph):
            break

        found = list(islice(nx.simple_cycles(subgraph), max_cycles_per_round,))
        if not found:
            break
        for cycle in found:
            indices = [edge_position[(cycle[i], cycle[(i + 1) % len(cycle)],)]
                       for i in range(len(cycle))
                       if (cycle[i], cycle[(i + 1) % len(cycle)],) in edge_position]
            if len(indices) > 1:
                cycle_constraints.append(cp.sum(edge_vars[np.array(indices)]) <= len(indices) - 1)
        cycles_removed += len(found)
        rounds += 1

    latent_values = {}
    if latent_time_vars is not None and latent_time_vars.value is not None:
        for node in latent_nodes:
            if node in subgraph:
                latent_values[node] = float(latent_time_vars.value[latent_index[node]])
    for node in subgraph.nodes():
        assigned = latent_values.get(node, graph.nodes[node].get("activation_time"))
        if assigned is not None:
            subgraph.nodes[node]["assigned_activation_time"] = float(assigned)

    selected_nodes = [node for node in subgraph.nodes()]
    collected = float(sum(graph.nodes[node].get("prize", 0.0) for node in selected_nodes))
    decomposition = {"total_prize_on_offer": float(prizes.sum()),
                     "collected_prize": collected,
                     "missed_prize": float(prizes.sum()) - collected,
                     "edge_cost": float(sum(1.0 - graph[u][v].get("weight", 1.0) + EDGE_EPSILON
                                            for u, v in selected_edges)),
                     "node_cost": node_penalty * max(len(selected_nodes) - 1, 0),
                     "latent_time_cost": latent_time_penalty * float(sum(latent_values.values())),}
    decomposition["fraction_of_prize_collected"] = (decomposition["collected_prize"]
                                                    / max(decomposition["total_prize_on_offer"], 1e-12))

    return {"subgraph": subgraph,
            "status": "solver_failed" if solver_error else problem.status,
            "solver_error": solver_error,
            "objective_value": problem.value,
            "solver": solver,
            "selected_edges": selected_edges,
            "selected_nodes": selected_nodes,
            "latent_times": latent_values,
            "decomposition": decomposition,
            "candidate_nodes": n_nodes,
            "candidate_edges": n_edges,
            "edges_dropped_by_time": dropped_by_time,
            "cycle_rounds": rounds,
            "cycles_removed": cycles_removed,
            "node_values": node_vars.value,
            "edge_values": edge_vars.value,
            "node_order": nodes,
            "edge_order": edges,}


# ── 10. Reading a selection ──────────────────────────────────────────────────────────────


def tag_feedback_edges(selected: nx.DiGraph,) -> tuple[nx.DiGraph, int]:
    """
    Mark the edges by which a kinase phosphorylates a site on a protein upstream of itself.

    This is the deliverable for the first point of the user's notes. No cycle is involved: the
    edge runs forward in time, and it only looks like a loop once the site is drawn next to the
    protein it belongs to. An edge is feedback when the protein carrying the target site is an
    ancestor of the kinase that phosphorylates it.

    Args:
      selected: Selected subnetwork, modified in place.

    Returns:
      Tuple of (the graph, number of feedback edges tagged). Every edge gains an is_feedback flag.
    """
    ancestors = {node: nx.ancestors(selected, node) for node in selected.nodes()}
    n_feedback = 0
    for u, v, data in selected.edges(data=True):
        target_protein = selected.nodes[v].get("protein_id")
        is_feedback = (data.get("edge_type") == "kinase_to_phosphosite"
                       and target_protein is not None
                       and target_protein in ancestors.get(u, set()))
        data["is_feedback"] = bool(is_feedback)
        n_feedback += int(is_feedback)
    return selected, n_feedback


def validate_selection(result: dict,
                       root_name: str = ROOT,
                       allow_equal_time: bool = True,) -> pd.DataFrame:
    """
    Check a selection, including the two things version 1's validation never checked.

    Version 1 asserted only properties its own constraints had already enforced, using its own
    variable values (§7). The checks that actually carry information are the solver status and
    whether every selected node really is reachable from the root; both are included here.

    Args:
      result: Result dict from select_network.
      root_name: Key of the artificial root.
      allow_equal_time: Whether equal times were permitted, so the temporal check matches the run.

    Returns:
      DataFrame with one row per check, holding the check name, its value and a passed flag.
    """
    graph = result["subgraph"]
    checks = []

    status_ok = result["status"] == "optimal"
    checks.append({"check": "solver status", "value": result["status"], "passed": status_ok})

    is_dag = nx.is_directed_acyclic_graph(graph)
    checks.append({"check": "acyclic", "value": is_dag, "passed": bool(is_dag)})

    if root_name in graph:
        reachable = nx.descendants(graph, root_name) | {root_name}
        orphans = set(graph.nodes()) - reachable
    else:
        orphans = set(graph.nodes())
    checks.append({"check": "nodes unreachable from root",
                   "value": len(orphans),
                   "passed": len(orphans) == 0,})

    violations = 0
    for u, v in graph.edges():
        u_time = graph.nodes[u].get("assigned_activation_time")
        v_time = graph.nodes[v].get("assigned_activation_time")
        if u_time is None or v_time is None or u == root_name or v == root_name:
            continue
        if (u_time > v_time) if allow_equal_time else (u_time >= v_time):
            violations += 1
    checks.append({"check": "temporal violations", "value": violations, "passed": violations == 0})

    leaves = [node for node in graph.nodes() if graph.out_degree(node) == 0]
    checks.append({"check": "leaf nodes (impossible in v1)", "value": len(leaves), "passed": True})

    return pd.DataFrame(checks)


def selection_tables(result: dict,) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Turn a selected network into the node and edge tables written to disk.

    Args:
      result: Result dict from select_network, after tag_feedback_edges has run.

    Returns:
      Tuple of (node table, edge table). The node table carries the label, type, assigned time,
      signed fold change, direction, prize and sign evidence; the edge table carries the endpoints,
      type, weight, feedback flag and the two endpoint times.
    """
    graph = result["subgraph"]
    nodes = pd.DataFrame([{"node": node,
                           "label": data.get("protein_site_id", node),
                           "node_type": data.get("node_type"),
                           "assigned_activation_time": data.get("assigned_activation_time"),
                           "peak_log2_fc_vs_starve": data.get("peak_log2_fc_vs_starve"),
                           "peak_direction": data.get("peak_direction"),
                           "prize": data.get("prize"),
                           "sign_evidence": data.get("sign_evidence"),
                           "protein_id": data.get("protein_id"),}
                          for node, data in graph.nodes(data=True)])
    edges = pd.DataFrame([{"source": u,
                           "target": v,
                           "source_label": graph.nodes[u].get("protein_site_id", u),
                           "target_label": graph.nodes[v].get("protein_site_id", v),
                           "edge_type": data.get("edge_type"),
                           "weight": data.get("weight"),
                           "is_feedback": data.get("is_feedback"),
                           "source_time": graph.nodes[u].get("assigned_activation_time"),
                           "target_time": graph.nodes[v].get("assigned_activation_time"),}
                          for u, v, data in graph.edges(data=True)])
    if len(nodes):
        nodes = nodes.sort_values(["node_type", "assigned_activation_time"], na_position="last")
    return nodes.reset_index(drop=True), edges


def add_orphan_root_edges(graph: nx.DiGraph,
                          orphan_cost: float,
                          root_name: str = ROOT,
                          root_kinase: str = EGFR_UNIPROT,) -> int:
    """
    Let any kinase enter the network without a path from the receptor, at a price (§4.2).

    The user's second objection to version 1: only EGFR seeded the network, so a kinase whose
    substrates clearly respond could not appear unless the receptor could reach it. Step 2 showed
    this is not hypothetical — ERK, RSK and p38 are unreachable from EGFR in this data, because the
    receptor's substrates are adaptors and MEK's activation sites were never measured.

    The fix is the standard prize-collecting-forest device: an edge from the artificial root to
    every kinase, charged omega. The selection becomes a forest — the EGF tree plus one small tree
    per orphan kinase — while the formulation stays a single rooted flow problem. A small omega
    degenerates the result into a kinase-substrate lookup, where every site is simply paired with
    its best-scoring kinase; a large omega reproduces the EGFR-only behaviour. It must be scanned,
    not guessed.

    An orphan kinase has no upstream, so its latent activation time is unconstrained from above and
    the tie-break pins it to the earliest allowed value, which reads as 'active from the start'.

    Args:
      graph: Candidate graph from build_candidate_graph, modified in place.
      orphan_cost: The price omega charged for entering the network without a receptor path.
      root_name: Key of the artificial root the orphan edges start from.
      root_kinase: The receptor, which keeps its free root edge and gets no orphan edge.

    Returns:
      Number of orphan edges added.
    """
    added = 0
    for node, data in list(graph.nodes(data=True)):
        if data.get("node_type") != "kinase" or node == root_kinase:
            continue
        if graph.has_edge(root_name, node):
            continue
        graph.add_edge(root_name,
                       node,
                       edge_type="orphan_to_kinase",
                       weight=1.0,
                       cost=float(orphan_cost),)
        added += 1
    return added


def orphan_report(result: dict,
                  root_name: str = ROOT,) -> pd.DataFrame:
    """
    Describe how a selection used the orphan entry points.

    Args:
      result: Result dict from select_network, run on a graph carrying orphan edges.
      root_name: Key of the artificial root.

    Returns:
      DataFrame with one row per selected orphan kinase, holding its accession, the activation time
      the solver assigned it, and how many nodes hang below it in the selection.
    """
    graph = result["subgraph"]
    rows = []
    for _, kinase, data in graph.out_edges(root_name, data=True) if root_name in graph else []:
        if data.get("edge_type") != "orphan_to_kinase":
            continue
        rows.append({"kinase": kinase,
                     "assigned_activation_time": graph.nodes[kinase].get("assigned_activation_time"),
                     "descendants": len(nx.descendants(graph, kinase)),})
    return pd.DataFrame(rows).sort_values("descendants", ascending=False) if rows else pd.DataFrame(
        columns=["kinase", "assigned_activation_time", "descendants"])


# ── 11. The SIGNOR causal layer ──────────────────────────────────────────────────────────

# §4.3. Price of a curated causal edge, by the mechanism SIGNOR records for it. A curated
# phosphorylation is evidence of the same kind as a literature kinase-substrate edge, so it is
# priced with it. A binding or a relocalisation is a weaker claim about signal flow — it says two
# proteins interact, not that activity propagates — so it costs more. SIGNOR's SCORE *is* a real
# 0.2-0.95 confidence, unlike the kinase-substrate table's score, so it is used directly here.
SIGNOR_MECHANISM_COSTS = {"phosphorylation": 0.05,
                          "dephosphorylation": 0.05,
                          "guanine nucleotide exchange factor": 0.15,
                          "gtpase-activating protein": 0.15,
                          "binding": 0.25,
                          "relocalization": 0.25,}
SIGNOR_DEFAULT_COST = 0.40
SIGNOR_SCORE_SPAN = 0.15

# Mechanisms that act on protein *amount* rather than activity. A 2-90 minute phospho response is
# far too fast for transcription or degradation to carry the signal, so they are excluded by
# default rather than silently allowed to explain a 2 minute change.
SIGNOR_SLOW_MECHANISMS = ("transcriptional regulation",
                          "post transcriptional regulation",)

THREE_LETTER_RESIDUES = {"Ser": "S", "Thr": "T", "Tyr": "Y",}


def _signor_residue(residue: object,) -> str | None:
    """
    Convert a SIGNOR residue label to the project's single-letter form.

    SIGNOR writes 'Tyr1110'; the PhosSites column and every site key in this project use 'Y1110'.

    Args:
      residue: The RESIDUE field of a SIGNOR row, which is often missing.

    Returns:
      The single-letter residue string, or None when the field is empty or not a S/T/Y site.
    """
    text = str(residue).strip()
    if len(text) < 4 or text[:3] not in THREE_LETTER_RESIDUES:
        return None
    position = text[3:]
    return f"{THREE_LETTER_RESIDUES[text[:3]]}{position}" if position.isdigit() else None


def load_signor(path: Path | list,
                direct_only: bool = True,
                drop_slow_mechanisms: bool = True,) -> tuple[pd.DataFrame, dict]:
    """
    Load a SIGNOR export as a directed, signed causal edge table (§4.3).

    This is the first resource in the project that can express a connection which is not a
    phosphorylation. The kinase-substrate table has 504 kinase sources and phosphosite targets and
    nothing else, so EGFR -> GRB2/SOS1 -> RAS -> RAF is unrepresentable with it: step 2 measured
    that the EGF signal cannot leave the receptor, because EGFR's substrates are adaptors.

    Args:
      path: Path to a SIGNOR .tsv export, or a list of paths whose rows are concatenated. Each
        export is a pathway-scoped query result, so they cover different parts of the cascade and
        are complementary: measured 2026-09-16, EGFR_16_09_26 carries the receptor-proximal layer
        while SIGNOR-EGF_14_08_26 carries GRB2 -> SOS1 -> HRAS -> BRAF and neither carries the
        other's edges.
      direct_only: Keep only rows flagged DIRECT == 'YES'.
      drop_slow_mechanisms: Drop transcriptional and post-transcriptional regulation, which change
        protein amount on a timescale far slower than a 2-90 minute phospho response.

    Returns:
      Tuple of (edge DataFrame, statistics dict). The DataFrame carries source and target UniProt
      accessions, entity names, effect ('up', 'down' or 'unknown'), mechanism, the converted
      residue where SIGNOR records one, score and the derived cost.
    """
    paths = [Path(path)] if isinstance(path, (str, Path,)) else [Path(item) for item in path]
    raw = pd.concat([pd.read_csv(item, sep="\t", low_memory=False,) for item in paths],
                    ignore_index=True,)
    stats = {"files": [item.name for item in paths], "rows": len(raw)}

    edges = raw[(raw["TYPEA"] == "protein") & (raw["TYPEB"] == "protein")].copy()
    stats["protein_to_protein"] = len(edges)
    if direct_only:
        edges = edges[edges["DIRECT"].astype(str).str.upper() == "YES"].copy()
        stats["direct"] = len(edges)
    edges = edges[edges["IDA"].astype(str) != edges["IDB"].astype(str)].copy()
    stats["after_self_loops_dropped"] = len(edges)

    mechanism = edges["MECHANISM"].astype(str).str.lower()
    if drop_slow_mechanisms:
        edges = edges[~mechanism.isin(SIGNOR_SLOW_MECHANISMS)].copy()
        mechanism = edges["MECHANISM"].astype(str).str.lower()
        stats["after_slow_mechanisms_dropped"] = len(edges)

    effect = edges["EFFECT"].astype(str).str.lower()
    out = pd.DataFrame({"source": edges["IDA"].astype(str),
                        "target": edges["IDB"].astype(str),
                        "source_name": edges["ENTITYA"].astype(str),
                        "target_name": edges["ENTITYB"].astype(str),
                        "effect": np.where(effect.str.startswith("up"), "up",
                                           np.where(effect.str.startswith("down"), "down", "unknown",),),
                        "mechanism": mechanism,
                        "residue": edges["RESIDUE"].map(_signor_residue),
                        "score": edges["SCORE"].astype(float),})

    base = out["mechanism"].map(SIGNOR_MECHANISM_COSTS).fillna(SIGNOR_DEFAULT_COST)
    out["cost"] = base + SIGNOR_SCORE_SPAN * (1.0 - out["score"].clip(0.0, 1.0))
    out = out.drop_duplicates(["source", "target", "mechanism"]).reset_index(drop=True)

    stats["unique_edges"] = len(out)
    stats["proteins"] = len(set(out["source"]) | set(out["target"]))
    stats["with_residue"] = int(out["residue"].notna().sum())
    stats["by_mechanism"] = out["mechanism"].value_counts().to_dict()
    stats["by_effect"] = out["effect"].value_counts().to_dict()
    return out, stats


def signor_gene_names(signor: pd.DataFrame,) -> dict:
    """
    Build an accession to gene-name lookup from a SIGNOR table.

    Proteins that enter the graph only through the causal layer have no row in the site table, so
    the figure would label them with a bare accession (P12931 rather than SRC).

    Args:
      signor: Causal edge table from load_signor.

    Returns:
      Dict mapping UniProt accession to the entity name SIGNOR uses.
    """
    names = dict(zip(signor["target"].astype(str), signor["target_name"].astype(str)))
    names.update(dict(zip(signor["source"].astype(str), signor["source_name"].astype(str))))
    return names


def add_signor_layer(graph: nx.DiGraph,
                     signor: pd.DataFrame,
                     site_table: pd.DataFrame,
                     kinase_ids: set,
                     add_curated_site_edges: bool = True,
                     add_causal_protein_edges: bool = True,) -> dict:
    """
    Add curated causal structure to the candidate graph, repairing the break found in step 2.

    Two things are added, and the second is what makes the canonical cascade representable:

    1. **Curated kinase to site edges** from SIGNOR rows that record a phosphorylation on a residue
       we measured. These are experimental evidence rather than motif prediction, and they also
       give an independent check on the predicted kinase-substrate table.
    2. **Causal protein to protein edges**, including to and from proteins that are not kinases.
       A protein enters as a node with a latent activation time and no prize, exactly like an
       unmeasured kinase. Crucially this **does not require the intermediate site to have been
       measured**: step 2 found MEK1's activation sites S218/S222 absent from this dataset, so a
       model that can only step through measured sites can never reconstruct RAF -> MEK -> ERK,
       however good the interaction data is.

    A protein that carries a measured, responsive site also gets that site's propagation edge, so
    the user's rule is respected where the evidence exists: a non-kinase protein can be entered
    through its own phosphorylation.

    Args:
      graph: Candidate graph from build_candidate_graph, modified in place.
      signor: Causal edge table from load_signor.
      site_table: Site table after add_propagation_rule, used to attach sites to new proteins.
      kinase_ids: Accessions that are kinases in the kinase-substrate table, so a new node is
        typed as 'kinase' rather than 'protein' where it is one.
      add_curated_site_edges: Add the kinase to measured-site edges of point 1.
      add_causal_protein_edges: Add the protein to protein edges of point 2.

    Returns:
      Statistics dict reporting what was added, including how many curated site edges were already
      present in the predicted table (`curated_edges_already_known`), which is the agreement check.
    """
    stats = {"curated_site_edges_added": 0,
             "curated_edges_already_known": 0,
             "causal_edges_added": 0,
             "protein_nodes_added": 0,
             "site_edges_to_new_proteins": 0,}

    def ensure_node(accession: str,) -> None:
        """
        Add a protein or kinase node for an accession that the graph does not carry yet.

        Args:
          accession: UniProt accession of the protein to add.

        Returns:
          None. The graph is modified in place.
        """
        if accession in graph:
            return
        graph.add_node(accession,
                       node_type="kinase" if accession in kinase_ids else "protein",
                       latent_time=True,
                       activation_time=None,
                       prize=0.0,)
        stats["protein_nodes_added"] += 1

    if add_curated_site_edges:
        phosphorylations = signor[signor["mechanism"].isin(["phosphorylation", "dephosphorylation"])
                                  & signor["residue"].notna()]
        for row in phosphorylations.itertuples(index=False):
            site_id = f"{row.target}_{row.residue}"
            if site_id not in graph:
                continue
            ensure_node(row.source)
            if graph.has_edge(row.source, site_id):
                stats["curated_edges_already_known"] += 1
                graph[row.source][site_id]["signor_confirmed"] = True
                continue
            graph.add_edge(row.source,
                           site_id,
                           edge_type="kinase_to_phosphosite",
                           weight=float(row.score),
                           cost=float(row.cost),
                           resource_class="signor",
                           effect=row.effect,
                           mechanism=row.mechanism,)
            stats["curated_site_edges_added"] += 1

    if add_causal_protein_edges:
        sites_by_protein = {}
        for row in site_table[site_table["can_propagate"]].itertuples(index=False):
            sites_by_protein.setdefault(str(row.protein_Id), []).append(row.kinsub_site_id)

        for row in signor.itertuples(index=False):
            ensure_node(row.source)
            ensure_node(row.target)
            if graph.has_edge(row.source, row.target):
                continue
            graph.add_edge(row.source,
                           row.target,
                           edge_type="causal",
                           weight=float(row.score),
                           cost=float(row.cost),
                           effect=row.effect,
                           mechanism=row.mechanism,
                           resource_class="signor",)
            stats["causal_edges_added"] += 1

        # A protein that carries a measured, responsive site can also be entered through it.
        for accession, site_ids in sites_by_protein.items():
            if accession not in graph:
                continue
            for site_id in site_ids:
                if site_id in graph and not graph.has_edge(site_id, accession):
                    graph.add_edge(site_id,
                                   accession,
                                   edge_type="phosphosite_to_protein",
                                   weight=1.0,
                                   cost=0.0,)
                    stats["site_edges_to_new_proteins"] += 1

    stats["nodes"] = graph.number_of_nodes()
    stats["edges"] = graph.number_of_edges()
    return stats


# ── 12. Stability and alternatives (§7) ──────────────────────────────────────────────────


def ensemble_selection(graph: nx.DiGraph,
                       node_penalty: float,
                       n_runs: int = 50,
                       noise: float = 0.2,
                       seed: int = 0,
                       **select_kwargs,) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Re-solve the ILP many times under perturbed edge costs and report how often each edge is kept.

    A single optimal network says what the cheapest explanation is, not how much better it is than
    the next one. On this data the difference is often tiny — ERK1 and ERK2 are near-identical
    substrates competing for the same explanation, and which one survives has already been seen to
    flip with a parameter change. An edge chosen in 49 of 50 perturbed runs is a result; an edge
    chosen in 12 of 50 is a coin toss, and drawing both with the same arrow is misleading.

    Each run multiplies every edge cost by 1 + U(-noise, +noise), which perturbs the ranking of
    near-equivalent explanations without changing the structure of the problem.

    Args:
      graph: Candidate graph, unmodified; each run works on a copy.
      node_penalty: Node cost passed to select_network.
      n_runs: Number of perturbed solves.
      noise: Relative width of the multiplicative cost perturbation.
      seed: Base random seed, so a run is reproducible.
      **select_kwargs: Passed through to select_network (mip_gap, time_limit, ...).

    Returns:
      Tuple of (edge frequency table, node frequency table, list of per-run summaries). Frequencies
      are computed over the runs that proved optimality only; each summary carries a `counted` flag
      saying whether it contributed. Raises RuntimeError if no run succeeded.
    """
    rng = np.random.default_rng(seed)
    base_costs = {(u, v,): float(data.get("cost", 1.0 - data.get("weight", 1.0)))
                  for u, v, data in graph.edges(data=True)}

    edge_counts: dict = {}
    node_counts: dict = {}
    summaries = []
    for run in range(n_runs):
        perturbed = graph.copy()
        factors = 1.0 + rng.uniform(-noise, noise, size=len(base_costs),)
        for (edge, cost,), factor in zip(base_costs.items(), factors,):
            perturbed[edge[0]][edge[1]]["cost"] = max(cost * factor, 0.0)
        result = select_network(perturbed, node_penalty=node_penalty, **select_kwargs,)
        # A run that did not prove optimality is not a smaller network, it is a worse answer, and
        # averaging it into the frequencies understates how stable the selection really is. On a
        # notebook run where the solver degraded late, contaminated runs moved the node-stability
        # figure from 84% to 68%. Failed runs are counted and reported, never averaged in.
        if result["status"] != "optimal":
            summaries.append({"run": run,
                              "status": result["status"],
                              "nodes": result["subgraph"].number_of_nodes(),
                              "edges": result["subgraph"].number_of_edges(),
                              "prize_pct": float("nan"),
                              "counted": False,})
            continue
        selected = result["subgraph"]
        for edge in selected.edges():
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
        for node in selected.nodes():
            node_counts[node] = node_counts.get(node, 0) + 1
        summaries.append({"run": run,
                          "status": result["status"],
                          "nodes": selected.number_of_nodes(),
                          "edges": selected.number_of_edges(),
                          "prize_pct": round(100 * result["decomposition"]["fraction_of_prize_collected"], 2),
                          "counted": True,})

    def label(node: str,) -> str:
        """
        Return the display label of a node, falling back to its accession.

        Args:
          node: Node key.

        Returns:
          The protein_site_id where the node is a phosphosite, otherwise the node key itself.
        """
        return graph.nodes[node].get("protein_site_id") or node

    counted = sum(1 for summary in summaries if summary["counted"])
    if counted == 0:
        raise RuntimeError("every ensemble run failed to prove optimality; nothing to summarise")
    n_runs = counted

    edge_table = pd.DataFrame([{"source": u,
                                "target": v,
                                "source_label": label(u),
                                "target_label": label(v),
                                "edge_type": graph[u][v].get("edge_type"),
                                "frequency": count / n_runs,}
                               for (u, v,), count in edge_counts.items()])
    node_table = pd.DataFrame([{"node": node,
                                "label": label(node),
                                "node_type": graph.nodes[node].get("node_type"),
                                "frequency": count / n_runs,}
                               for node, count in node_counts.items()])
    if len(edge_table):
        edge_table = edge_table.sort_values("frequency", ascending=False).reset_index(drop=True)
    if len(node_table):
        node_table = node_table.sort_values("frequency", ascending=False).reset_index(drop=True)
    return edge_table, node_table, summaries


def augmented_edges(candidate: nx.DiGraph,
                    selected: nx.DiGraph,
                    allow_equal_time: bool = True,) -> pd.DataFrame:
    """
    List the candidate edges between selected nodes that the optimiser did not take.

    The ILP returns the cheapest explanation, so a route it rejects may have been barely more
    expensive than the one it kept. These edges cost nothing to compute and answer the question the
    selection alone cannot: which parts of the network are a parsimony choice rather than the only
    option. Drawn faintly (§8), they are also where the "lateral information processing" of the
    user's fourth note becomes visible.

    Args:
      candidate: The full candidate graph the selection was made from.
      selected: The selected subnetwork.
      allow_equal_time: Whether two nodes sharing an activation time may be joined, matching the
        setting the ILP was solved with.

    Returns:
      DataFrame of the unselected, time-consistent edges between selected nodes, with endpoints,
      labels, edge type and cost.
    """
    rows = []
    nodes = set(selected.nodes())
    for u, v, data in candidate.edges(data=True):
        if u not in nodes or v not in nodes or selected.has_edge(u, v):
            continue
        u_time = selected.nodes[u].get("assigned_activation_time")
        v_time = selected.nodes[v].get("assigned_activation_time")
        if u_time is not None and v_time is not None:
            if (u_time > v_time) if allow_equal_time else (u_time >= v_time):
                continue
        rows.append({"source": u,
                     "target": v,
                     "source_label": selected.nodes[u].get("protein_site_id") or u,
                     "target_label": selected.nodes[v].get("protein_site_id") or v,
                     "edge_type": data.get("edge_type"),
                     "cost": data.get("cost"),})
    return pd.DataFrame(rows)


# ── 13. Drawing (§8) ─────────────────────────────────────────────────────────────────────

# Fill colour of a node by its activation time, in minutes.
TIME_COLORS_V2 = {0: "#475569",
                  2: "#2563eb",
                  5: "#16a34a",
                  10: "#f59e0b",
                  15: "#dc2626",
                  90: "#7c3aed",}


def readable_subnetwork(selected: nx.DiGraph,
                        seeds: list | None = None,
                        max_nodes: int = 120,
                        root_name: str = ROOT,) -> nx.DiGraph:
    """
    Cut a selected network down to something a figure can actually show.

    A selection of ~2500 nodes is a result, not a picture. This keeps the paths from the root to a
    set of seed proteins, then fills the remaining budget with the highest-prize sites hanging off
    the proteins already included, so the figure is the backbone plus its strongest readout rather
    than an arbitrary crop.

    Args:
      selected: The selected network.
      seeds: Node keys the figure must contain, e.g. the cascade members. None uses the highest
        prize nodes as seeds.
      max_nodes: Node budget for the figure.
      root_name: Key of the artificial root, kept when present.

    Returns:
      Induced subgraph of the selection, as a copy.
    """
    keep = {root_name} if root_name in selected else set()
    seeds = [seed for seed in (seeds or []) if seed in selected]
    if not seeds:
        by_prize = sorted(selected.nodes(),
                          key=lambda node: selected.nodes[node].get("prize", 0.0),
                          reverse=True,)
        seeds = by_prize[:5]

    for seed in seeds:
        keep.add(seed)
        if root_name in selected and nx.has_path(selected, root_name, seed):
            keep.update(nx.shortest_path(selected, root_name, seed))

    # Fill the budget with the best-supported readout of the proteins already in the figure.
    candidates = []
    for node in keep.copy():
        for _, site in selected.out_edges(node):
            if site not in keep and selected.nodes[site].get("node_type") == "phosphosite":
                candidates.append((selected.nodes[site].get("prize", 0.0), site,))
    for _, site in sorted(candidates, reverse=True):
        if len(keep) >= max_nodes:
            break
        keep.add(site)
    return selected.subgraph(keep).copy()


def write_protein_centred_dot(graph: nx.DiGraph,
                              path: Path,
                              title: str,
                              gene_names: dict | None = None,
                              edge_frequency: dict | None = None,
                              time_colors: dict | None = None,
                              root_name: str = ROOT,) -> Path:
    """
    Write the figure the user asked for: each protein drawn surrounded by its own phosphosites.

    Version 1 drew kinases next to the sites they phosphorylate, so a protein and its own
    regulatory sites ended up scattered across the picture. Here every protein is a Graphviz
    cluster containing its kinase/protein node **and its measured sites**, so the site that
    regulates a kinase sits with that kinase, and the long arrows between clusters are the
    phosphorylation events.

    Encoding, all of it from the user's notes plus what the analysis has since shown matters:
      - node colour  = activation time;
      - node fill    = direction, solid for a phosphorylation that increases, hollow (white fill,
                       coloured border) for one that decreases. Hollow is used rather than a low
                       alpha because white label text on a pale fill is unreadable;
      - edge opacity = ensemble frequency where given, since 79% of edges are chosen in under half
                       of perturbed runs and drawing a coin toss like a fact would mislead;
      - dashed edge  = SIGNOR causal (non-phosphorylation) step;
      - red edge     = feedback, a kinase phosphorylating a site on a protein upstream of itself.

    Args:
      graph: Network to draw, small enough to read (see readable_subnetwork).
      path: Destination .dot path.
      title: Caption placed at the top.
      gene_names: Accession to gene-name lookup for protein nodes.
      edge_frequency: Optional {(source, target): frequency} from ensemble_selection.
      time_colors: Activation-time to colour map. None uses TIME_COLORS_V2.
      root_name: Key of the artificial root.

    Returns:
      The path written.
    """
    time_colors = TIME_COLORS_V2 if time_colors is None else time_colors
    gene_names = gene_names or {}
    edge_frequency = edge_frequency or {}

    def colour_for(node: str,) -> str:
        """
        Pick the fill colour of a node from its assigned activation time.

        Args:
          node: Node key.

        Returns:
          A hex colour string, grey when the node has no time.
        """
        time = graph.nodes[node].get("assigned_activation_time")
        if time is None:
            return "#6b7280"
        return time_colors.get(int(time), "#6b7280")

    # Group every node under the protein it belongs to, which is what makes the clusters.
    groups: dict = {}
    for node, data in graph.nodes(data=True):
        if node == root_name:
            continue
        protein = data.get("protein_id") if data.get("node_type") == "phosphosite" else node
        groups.setdefault(protein, []).append(node)

    lines = ["digraph G {",
             '  graph [rankdir=TB, compound=true, splines=true, nodesep=0.25, ranksep=0.7, bgcolor="white"];',
             '  node [fontname="Helvetica", fontsize=9, margin=0.05];',
             '  edge [fontname="Helvetica", arrowsize=0.6];',
             '  labelloc="t";',
             f'  label="{title}";',
             ""]
    if root_name in graph:
        lines.append(f'  "{root_name}" [label="{root_name}", shape="doublecircle", style="filled", fillcolor="#111827", fontcolor="white"];')

    for index, (protein, members,) in enumerate(sorted(groups.items())):
        gene = gene_names.get(protein, protein)
        lines.append(f'  subgraph cluster_{index} {{')
        lines.append(f'    label="{gene}"; style="rounded"; color="#cbd5e1"; fontsize=10;')
        for node in members:
            data = graph.nodes[node]
            colour = colour_for(node)
            if data.get("node_type") == "phosphosite":
                site = str(data.get("protein_site_id", node)).split("_")[-1]
                fold = data.get("peak_log2_fc_vs_starve", 0.0) or 0.0
                label = f"{site}\\n{fold:+.1f}"
                if data.get("peak_direction") == "down":
                    style = f'shape="ellipse", style="filled", fillcolor="white", color="{colour}", penwidth="2.2", fontcolor="{colour}"'
                else:
                    style = f'shape="ellipse", style="filled", fillcolor="{colour}", color="{colour}", fontcolor="white"'
            else:
                # The assigned activation time is the one thing the optimiser decides about a
                # protein node, so it belongs on the node rather than only in the node table.
                time = graph.nodes[node].get("assigned_activation_time")
                label = gene if time is None else f"{gene}\\nt={time:g}"
                style = f'shape="box", style="rounded,filled", fillcolor="#dbeafe", color="{colour}", penwidth="2.4"'
            lines.append(f'    "{node}" [label="{label}", {style}];')
        lines.append("  }")

    lines.append("")
    for u, v, data in graph.edges(data=True):
        frequency = edge_frequency.get((u, v,))
        alpha = "FF" if frequency is None else f"{max(int(255 * frequency), 40):02X}"
        if data.get("is_feedback"):
            # Not red: #dc2626 is the 15-minute node colour, so a feedback edge was indistinguishable
            # from a legend swatch. Magenta is used by nothing else in the figure.
            colour, style, width = "#be185d", "bold", "2.4"
        elif data.get("edge_type") == "causal":
            colour, style, width = "#7c3aed", "dashed", "1.8"
        elif data.get("edge_type") in ("phosphosite_to_kinase", "phosphosite_to_protein",):
            colour, style, width = "#94a3b8", "dotted", "1.2"
        elif data.get("edge_type") == "orphan_to_kinase":
            colour, style, width = "#111827", "dashed", "1.2"
        else:
            colour, style, width = "#64748b", "solid", "1.4"
        label = ""
        if data.get("is_feedback"):
            label = ', label=" feedback", fontsize=7, fontcolor="#be185d"'
        elif frequency is not None and frequency < 0.95:
            label = f', label=" {frequency:.0%}", fontsize=7, fontcolor="#94a3b8"'
        lines.append(f'  "{u}" -> "{v}" [color="{colour}{alpha}", style="{style}", penwidth="{width}"{label}];')

    lines.extend(["", "  subgraph cluster_legend {",
                  '    label="Legend"; color="#cbd5e1"; style="rounded"; fontsize=10;',
                  '    "l_up" [label="up", shape="ellipse", style="filled", fillcolor="#2563eb", fontcolor="white"];',
                  '    "l_down" [label="down", shape="ellipse", style="filled", fillcolor="white", color="#2563eb", penwidth="2.2", fontcolor="#2563eb"];',
                  '    "l_prot" [label="protein", shape="box", style="rounded,filled", fillcolor="#dbeafe"];',
                  '    "l_up" -> "l_down" [label=" phosphorylation", color="#64748b", fontsize=8];',
                  '    "l_down" -> "l_prot" [label=" causal (SIGNOR)", color="#7c3aed", style="dashed", fontsize=8];',
                  '    "l_prot" -> "l_up" [label=" feedback", color="#be185d", style="bold", fontsize=8];'])
    previous = None
    for time, colour in sorted(time_colors.items()):
        node = f"l_t{time}"
        lines.append(f'    "{node}" [label="{time} min", shape="box", style="filled", fillcolor="{colour}", fontcolor="white", fontsize=8];')
        if previous:
            lines.append(f'    "{previous}" -> "{node}" [style="invis"];')
        previous = node
    lines.extend(["  }", "}"])
    Path(path).write_text("\n".join(lines) + "\n")
    return Path(path)
