"""
CurveCurator integration — running the Kuster-lab curve fitter on *time* instead of dose.

CurveCurator (Bayer, Gander et al., kusterlab/curve_curator) fits a 4-parameter log-logistic
model to dose-response proteomics data and, more importantly, wraps it in a statistical
framework this project does not otherwise have: a regularised F-test against a flat null, a
target-decoy FDR, and the relevance score that fuses significance with effect size.

WHAT IS BEING ADAPTED
---------------------
CurveCurator's model is (curve_curator/models.py, LogisticModel.core):

    y = (front - back) / (1 + 10 ** (slope * (x + pEC50))) + back

with `x` **already in log10 space**: x = log10(dose * dose_scale), built by
`toolbox.build_drug_log_concentrations`. Nothing in that expression is chemical. Substituting
time in minutes for concentration gives a log-time sigmoid — which is the parameterisation
`src/curve_fitting.py` argued for on this data anyway (log-time is a better *parameterisation*,
not a better fit: the 90-min point carries leverage 0.979 in linear time and 0.659 in log time).

The dictionary between the two implementations:

    CurveCurator (dose)          this project (time)
    -------------------------    ----------------------------------------------------
    x = log10(dose * scale)      x = log10(t_minutes * scale), scale = 1
    pEC50 = -log10(EC50)         pEC50 = -log10(T50)   ->  T50 = 10 ** (-pEC50) / scale
    front (x -> -inf plateau)    response ratio at t = 0, which is 1 by construction
    back  (x -> +inf plateau)    final plateau, as a ratio to the starve control
    slope                        steepness on the log-time axis (the 'k' of curve_fitting)
    y = ratio to control         2 ** (log2 FC vs starve)

Two consequences worth stating before anyone reads a T50 out of this module:

1. **CurveCurator is monotone-sigmoid only.** Its own FAQ: "CurveCurator was specifically
   optimized for dose-response data ... Other x-dimensions, such as time, can be processed ...
   but CurveCurator will miss any non-sigmoidal behaviors." On hme1_2 EGF, 69% of responsive
   sites are transient or biphasic. The optimiser does not refuse them, it returns a number.
   Gate with `src/response_shapes.py` *before* fitting, exactly as the anchored-sigmoid
   notebook does — `build_curvecurator_input` takes a `mask` for this.

2. **Ratios, not log2 fold changes.** CurveCurator divides each measurement by the mean of the
   control columns and fits in ratio space, so the input must be *intensities*
   (`raw:abs`), not `log2:FC`. Feeding it fold changes would fit a sigmoid to a logarithm of
   the thing the model expects.

WHAT THIS BUYS OVER src/curve_fitting.py
----------------------------------------
Replicates. `curve_fitting` fits the 6-point mean profile with SEM weights; CurveCurator
supports replicated doses fitted simultaneously (see
example_toml_files/minimal_parameters_with_triplicates.toml — the dose value is simply repeated
once per replicate). Handing it 4 replicates x 6 timepoints turns 6 points into 24 and gives
the slope real degrees of freedom, which is the parameter `profile_global_k` could not
identify on the aggregated profile.

It also has a `normalization` switch (median centring of log-normalised values) that addresses
the missing sample-loading normalisation recorded under "Known issues" in CLAUDE.md.

REFERENCES
----------
- Repository and documentation: https://github.com/kusterlab/curve_curator
- Model: curve_curator/models.py; dose axis: curve_curator/toolbox.py
  (`build_drug_log_concentrations`); config schema: example_toml_files/all_parameters.toml.
- CurveCurator is NOT a dependency of this project's conda env. Install it separately
  (`pip install curve-curator`, the authors recommend its own python 3.12 environment) and
  point `run_curvecurator` at that interpreter.
"""

import os
import shutil
import subprocess

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.column_spec import ColumnSpec
from src.response_shapes import timepoint_minutes


# CurveCurator refuses a design with fewer than 4 experiments (toml_parser validation).
MIN_EXPERIMENTS = 4

# Column added to the curves file by this module, in project units.
T50_COL = "t50_min"


# =============================================================================
# 1. The model, and the pEC50 <-> T50 dictionary
# =============================================================================

def logistic_response(x: np.ndarray,
                      pec50: float,
                      slope: float,
                      front: float,
                      back: float,) -> np.ndarray:
    """
    CurveCurator's 4-parameter log-logistic, reimplemented for plotting and verification.

    Mirrors `curve_curator.models.LogisticModel.core` exactly. Kept here so the notebook can
    draw fitted curves, and check round trips, without importing CurveCurator — which lives in
    a separate environment.

    Args:
        x: transformed axis, log10(time * dose_scale). NOT raw minutes.
        pec50: inflection point; -log10 of the half-response time (in scaled units).
        slope: steepness of the transition on the log axis.
        front: plateau as x -> -inf (the control level; 1.0 in ratio space).
        back: plateau as x -> +inf (the final level, as a ratio to the control).

    Returns:
        Array of model values in ratio space, same shape as x.

    """
    return (front - back) / (1.0 + 10.0 ** (slope * (x + pec50))) + back


def pec50_to_t50(pec50,
                 dose_scale: float = 1.0,):
    """
    Convert CurveCurator's pEC50 into a half-response time in minutes.

    The fit places the inflection at x = -pEC50 on the axis x = log10(t * dose_scale), so
    t50 = 10 ** (-pEC50) / dose_scale. With `dose_scale = 1.0` and doses given in minutes this
    is simply 10 ** (-pEC50) — note the negative exponent, which makes pEC50 *negative* for any
    T50 above one minute. That is expected and not a sign error: pEC50 was designed for
    sub-molar concentrations, where the same expression comes out positive.

    Args:
        pec50: fitted pEC50, scalar or array-like.
        dose_scale: the numeric value of `dose_scale` in the TOML (1.0 when doses are minutes).

    Returns:
        Half-response time in minutes, same shape as the input.

    """
    # A pEC50 at CurveCurator's lower bound (min(-x) - PEC50_DELTA) already means a transition
    # three orders of magnitude past the last timepoint; anything more extreme overflows the
    # power. Let it go to inf quietly rather than emitting a RuntimeWarning per call — such a
    # value is not a measurement either way, and the pEC50_filter is what excludes it.
    with np.errstate(over="ignore",):
        return 10.0 ** (-np.asarray(pec50, dtype=float,)) / dose_scale


def t50_to_pec50(t50_min,
                 dose_scale: float = 1.0,):
    """
    Inverse of `pec50_to_t50` — the pEC50 a given half-response time corresponds to.

    Used to translate a time window into the `pEC50_filter` bounds CurveCurator expects.

    Args:
        t50_min: half-response time in minutes, scalar or array-like.
        dose_scale: the numeric value of `dose_scale` in the TOML.

    Returns:
        The corresponding pEC50 value(s).

    """
    return -np.log10(np.asarray(t50_min, dtype=float,) * dose_scale)


def pec50_filter_for_window(times: Sequence[float],
                            dose_scale: float = 1.0,
                            pad_log10: float = 0.0,) -> List[float]:
    """
    Build a `pEC50_filter` covering the observed time window, ordered low to high.

    A T50 outside the sampling window is an extrapolation, not a measurement: the fit is
    asserting a transition it never saw. CurveCurator will still report the curve but will not
    classify it as up/down when this filter is set.

    Args:
        times: the minute axis, including or excluding the zero control.
        dose_scale: the numeric value of `dose_scale` in the TOML.
        pad_log10: widen the window by this much on each side, in log10 units (0.3 ~ a factor
            of two). Zero keeps the filter strictly inside the sampled range.

    Returns:
        [lower, upper] pEC50 bounds as plain floats, ready to write into the TOML.

    """
    positive = np.asarray([t for t in times if t > 0], dtype=float,)
    if positive.size == 0:
        raise ValueError("pec50_filter_for_window: no positive timepoints given.")

    # pEC50 runs *opposite* to time, so the earliest timepoint gives the upper bound.
    hi = float(t50_to_pec50(positive.min(), dose_scale,) + pad_log10)
    lo = float(t50_to_pec50(positive.max(), dose_scale,) - pad_log10)
    return [lo, hi]


# =============================================================================
# 2. Building the input table and the experimental design
# =============================================================================

def build_curvecurator_input(df: pd.DataFrame,
                             cell_line: str = "WT",
                             condition: str = "_EGF_",
                             data_type: str = "raw:abs",
                             name_col: str = "site",
                             mask: Optional[np.ndarray] = None,
                             aggregate_replicates: bool = False,) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Turn one condition of a project dataset into CurveCurator's input table plus its design.

    CurveCurator reads a tab-separated file with a `Name` column and one `Raw {experiment}`
    column per experiment (`data_parser.load_generic`, with search_engine/measurement_type/
    data_type all set to 'OTHER'). The experiment ids and the dose each one carries are
    declared in the TOML, not in the file, so the two must be generated together — hence the
    design table returned alongside.

    Every replicate of every timepoint becomes its own experiment, carrying the *same* dose
    (= its timepoint in minutes). This is CurveCurator's "fitted simultaneously" mode for
    replicated doses and is the reason to prefer it over `src/curve_fitting.py` here: it fits
    n_replicates x n_timepoints points rather than the aggregated profile, which is what gives
    the slope enough degrees of freedom to be estimated instead of fixed by convention.

    The `full` timepoint is dropped — it is a biological control, not a point on the
    post-stimulation clock. `starve` becomes dose 0.0, which CurveCurator treats as the control
    that every ratio is formed against.

    Args:
        df: dataset following the project column naming convention.
        cell_line: cell-line prefix, e.g. 'WT'.
        condition: condition token with underscores, e.g. '_EGF_'.
        data_type: which values to export. Must be an **intensity** type ('raw:abs' normally),
            because CurveCurator forms its own ratios against the control; a 'log2:FC'
            data_type is rejected.
        name_col: column used as CurveCurator's unique `Name`. Must be unique across the
            exported rows.
        mask: optional boolean array over `df` rows selecting which sites to export — normally
            the `sigmoid_legitimate` shape gate, since CurveCurator cannot represent the
            transient majority.
        aggregate_replicates: export the per-timepoint mean instead of the individual
            replicates. Recovers the 6-point behaviour of `src/curve_fitting.py`; off by
            default because the replicates are the point.

    Returns:
        Tuple (input_df, design):
            input_df: the table to write as CurveCurator's input file — a `Name` column
                followed by one `Raw {id}` column per experiment.
            design: one row per experiment with columns `experiment`, `source_column`,
                `timepoint`, `minutes`, `replicate` and `is_control`, in dose order. Pass it
                straight to `write_curvecurator_toml`.

    """
    if ":" in data_type and data_type.split(":")[0] == "log2":
        raise ValueError(f"build_curvecurator_input: data_type='{data_type}' is a log-space "
                         f"quantity. CurveCurator fits ratios to the control and forms them "
                         f"itself, so it needs intensities — use 'raw:abs' (or 'raw:mean' with "
                         f"aggregate_replicates=True).")
    if name_col not in df.columns:
        raise ValueError(f"build_curvecurator_input: '{name_col}' not in the DataFrame.")

    cols = ColumnSpec.select(df,
                             cell_lines=[cell_line],
                             data_type=data_type,
                             conditions=[condition],
                             exclude_full=True,
                             exclude_replicate_cols=aggregate_replicates,)
    if not cols:
        raise ValueError(f"build_curvecurator_input: no '{data_type}' columns for "
                         f"cell_line='{cell_line}', condition='{condition}'.")

    records = []
    for col in cols:
        parts = col.split("_")
        if len(parts) < 4:
            continue
        minutes = timepoint_minutes(parts[3],)
        if not np.isfinite(minutes):
            continue                                   # 'full', already excluded, belt and braces
        replicate = parts[4] if len(parts) > 4 else ""
        records.append({"source_column": col,
                        "timepoint": parts[3],
                        "minutes": minutes,
                        "replicate": replicate,})

    if not records:
        raise ValueError("build_curvecurator_input: no columns survived the timepoint parse.")

    design = pd.DataFrame(records,)
    design = design.sort_values(["minutes", "replicate",],
                                kind="stable",).reset_index(drop=True,)
    design["experiment"] = np.arange(1, len(design) + 1,)
    design["is_control"] = design["minutes"] == 0.0
    design = design[["experiment", "source_column", "timepoint", "minutes", "replicate",
                     "is_control",]]

    if len(design) < MIN_EXPERIMENTS:
        raise ValueError(f"build_curvecurator_input: only {len(design)} experiments; "
                         f"CurveCurator requires at least {MIN_EXPERIMENTS}.")
    if not design["is_control"].any():
        raise ValueError("build_curvecurator_input: no 'starve' column found, so there is no "
                         "dose-0 control for CurveCurator to form ratios against.")

    rows = df if mask is None else df.loc[np.asarray(mask, dtype=bool,)]

    names = rows[name_col].astype(str,)
    if names.duplicated().any():
        n_dup = int(names.duplicated().sum())
        raise ValueError(f"build_curvecurator_input: '{name_col}' has {n_dup} duplicate values "
                         f"among the exported rows. CurveCurator uses Name as the unique key "
                         f"and would silently collapse them.")

    input_df = pd.DataFrame({"Name": names.to_numpy(),},)
    for _, row in design.iterrows():
        input_df[f"Raw {row['experiment']}"] = rows[row["source_column"]].to_numpy()

    return input_df, design


# =============================================================================
# 3. Writing the TOML
# =============================================================================

def write_curvecurator_toml(path: str,
                            design: pd.DataFrame,
                            input_file: str,
                            meta: Optional[Dict[str, str]] = None,
                            dose_scale: str = "1e0",
                            dose_unit: str = "min",
                            alpha: float = 0.05,
                            fc_lim: float = 0.45,
                            pec50_filter: Optional[Sequence[float]] = None,
                            fixed_front: Optional[float] = 1.0,
                            fixed_slope: Optional[float] = None,
                            control_fold_change: bool = True,
                            normalization: bool = True,
                            imputation: bool = False,
                            max_missing: Optional[int] = None,
                            available_cores: int = 1,
                            speed: str = "standard",
                            fit_type: str = "OLS",
                            curves_file: str = "./curves.txt",
                            decoys_file: str = "./decoys.txt",
                            fdr_file: str = "./fdr.txt",
                            dashboard: str = "./dashboard.html",
                            extra_sections: Optional[Dict[str, Dict]] = None,) -> str:
    """
    Write the CurveCurator parameter file for a time-series design.

    Follows the schema in `example_toml_files/all_parameters.toml`: sections `['Meta']`,
    `['Experiment']`, `['Paths']`, `['Processing']`, `['Curve Fit']`, `['F Statistic']`,
    `['Dashboard']`, with all paths relative to the TOML itself. The file is written by hand
    rather than with a TOML library so the comments explaining the time adaptation survive into
    the artefact someone will read six months from now.

    Defaults chosen for this project, and why:

    - `dose_scale = '1e0'`, `dose_unit = 'min'` — doses are minutes, so x = log10(t) and
      T50 = 10 ** (-pEC50) with no further scaling.
    - `fixed_front = 1.0` — the response at the control is a ratio to the control, so it is 1
      by construction. This is the same argument that makes the model in `src/curve_fitting.py`
      anchored: the baseline is a constraint, not a parameter to spend a degree of freedom on.
    - `control_fold_change = true` — report the fold change against the control rather than
      between the lowest and highest dose, so `Curve Fold Change` is comparable to the
      `plateau` of the anchored fits.
    - `normalization = true` — median centring of the log-normalised values. The project
      pipeline applies no sample-loading normalisation (CLAUDE.md, Known issues), and the 84
      `raw:abs` column medians span 2.1-fold; letting CurveCurator normalise is the point of
      feeding it intensities rather than pre-computed ratios.

    Args:
        path: where to write the .toml.
        design: the design table from `build_curvecurator_input`.
        input_file: path to the input table, relative to `path`.
        meta: values for the ['Meta'] section (`id`, `description`, `condition`,
            `treatment_time`); missing keys are filled with placeholders.
        dose_scale: TOML `dose_scale`, as the scientific-notation *string* CurveCurator expects.
        dose_unit: TOML `dose_unit`; 'min' for a time series.
        alpha: p-value threshold for a statistically meaningful curve.
        fc_lim: absolute log2 fold-change threshold for a biologically meaningful curve.
        pec50_filter: [lower, upper] pEC50 bounds; curves outside are not classified as
            regulated. Use `pec50_filter_for_window` to derive it from the time grid.
        fixed_front: pin the front plateau to this value, or None to fit it.
        fixed_slope: pin the slope, or None (default) to fit it. Fitting it is the whole reason
            to pass replicates; fix it only to reproduce the fixed-k mode of `curve_fitting`.
        control_fold_change: report fold change relative to the control.
        normalization: apply CurveCurator's median-centring normalisation.
        imputation: impute missing values with a low constant. Off by default — this project
            forbids silently filling missing values.
        max_missing: maximum NaNs tolerated per curve, excluding controls. Defaults to a
            quarter of the non-control experiments.
        available_cores: parallel workers.
        speed: 'fast' | 'standard' | 'exhaustive' | 'basinhopping'.
        fit_type: 'OLS' or 'MLE'.
        curves_file: output path for the fitted curves.
        decoys_file: output path for decoy curves (--fdr mode).
        fdr_file: output path for the FDR estimate (--fdr mode).
        dashboard: output path for the interactive Bokeh dashboard.
        extra_sections: {section_name: {key: value}} merged in verbatim, for keys this wrapper
            does not expose.

    Returns:
        The path written.

    """
    if fit_type not in ("OLS", "MLE",):
        raise ValueError(f"write_curvecurator_toml: fit_type must be 'OLS' or 'MLE', "
                         f"got {fit_type!r}.")
    if speed not in ("fast", "standard", "exhaustive", "basinhopping",):
        raise ValueError(f"write_curvecurator_toml: speed must be one of 'fast', 'standard', "
                         f"'exhaustive', 'basinhopping', got {speed!r}.")
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"write_curvecurator_toml: alpha must be in (0, 1], got {alpha}.")
    if fc_lim < 0.0:
        raise ValueError(f"write_curvecurator_toml: fc_lim must be >= 0, got {fc_lim}.")

    meta = dict(meta or {},)
    experiments = [int(e) for e in design["experiment"]]
    doses = [float(d) for d in design["minutes"]]
    controls = [int(e) for e in design.loc[design["is_control"], "experiment"]]
    n_treated = int((~design["is_control"]).sum())

    if max_missing is None:
        max_missing = max(1, n_treated // 4,)

    def fmt(value,) -> str:
        """
        Render a python value as TOML.

        Args:
            value: string, bool, number or sequence.

        Returns:
            Its TOML representation.

        """
        if isinstance(value, bool,):
            return "true" if value else "false"
        if isinstance(value, str,):
            return f"'{value}'"
        if isinstance(value, (list, tuple,)):
            return "[" + ", ".join(fmt(v,) for v in value) + "]"
        if isinstance(value, float,) and not np.isfinite(value,):
            return "+inf" if value > 0 else "-inf"
        return str(value)

    grid = sorted(set(d for d in doses if d > 0),)
    lines = [
        "#",
        "# CurveCurator parameters — TIME SERIES adaptation.",
        "#",
        "# The x axis is time, not concentration. CurveCurator computes x = log10(dose * dose_scale)",
        "# and is indifferent to what 'dose' means, so `doses` below are MINUTES after stimulation",
        f"# and dose_scale is {dose_scale}. Read the fitted pEC50 back as a half-response time with",
        "#     T50 [min] = 10 ** (-pEC50) / dose_scale        (src.curvecurator_io.pec50_to_t50)",
        "# pEC50 is therefore NEGATIVE for any T50 above one minute. That is not a sign error.",
        "#",
        "# Dose 0.0 is the starve control: CurveCurator forms every ratio against it and anchors",
        "# the curve there. Each replicate is its own experiment carrying the same dose, which is",
        "# CurveCurator's 'fitted simultaneously' mode for replicated doses.",
        "#",
        "# CAVEAT, from CurveCurator's own FAQ: the model is a monotone sigmoid in log-x and",
        "# 'will miss any non-sigmoidal behaviors'. On this data the majority of responsive sites",
        "# are transient. Gate the input with src/response_shapes.py before fitting.",
        "#",
        f"# Written by src/curvecurator_io.py. Time grid: {grid} min, "
        f"{len(controls)} control experiment(s).",
        "#",
        "",
        "['Meta']",
        f"id = {fmt(meta.get('id', 'time_series'),)}",
        f"description = {fmt(meta.get('description', 'Phosphoproteomics time course'),)}",
        f"condition = {fmt(meta.get('condition', 'EGF'),)}",
        f"treatment_time = {fmt(meta.get('treatment_time', f'{max(grid)} min'),)}",
        "",
        "['Experiment']",
        f"experiments = {fmt(experiments,)}",
        f"doses = {fmt(doses,)}      # MINUTES, not concentration",
        f"dose_scale = {fmt(dose_scale,)}",
        f"dose_unit = {fmt(dose_unit,)}",
        f"control_experiment = {fmt(controls,)}      # the starve channels",
        "measurement_type = 'OTHER'",
        "data_type = 'OTHER'",
        "search_engine = 'OTHER'",
        "",
        "['Paths']",
        f"input_file = {fmt(input_file,)}",
        f"curves_file = {fmt(curves_file,)}",
        f"decoys_file = {fmt(decoys_file,)}",
        f"fdr_file = {fmt(fdr_file,)}",
        f"dashboard = {fmt(dashboard,)}",
        "",
        "['Processing']",
        f"available_cores = {fmt(available_cores,)}",
        f"max_missing = {fmt(int(max_missing),)}",
        f"imputation = {fmt(imputation,)}",
        f"normalization = {fmt(normalization,)}      # median centring; the project pipeline applies none",
        "",
        "['Curve Fit']",
        f"type = {fmt(fit_type,)}",
        f"speed = {fmt(speed,)}",
        f"control_fold_change = {fmt(control_fold_change,)}",
    ]
    if fixed_front is not None:
        lines.append(f"front = {fmt(float(fixed_front),)}      "
                     f"# ratio to the control at t=0 is 1 by construction")
    if fixed_slope is not None:
        lines.append(f"slope = {fmt(float(fixed_slope),)}")

    lines += [
        "",
        "['F Statistic']",
        f"alpha = {fmt(alpha,)}",
        f"fc_lim = {fmt(fc_lim,)}",
    ]
    if pec50_filter is not None:
        if len(pec50_filter) != 2:
            raise ValueError("write_curvecurator_toml: pec50_filter must have length 2.")
        lo, hi = float(pec50_filter[0]), float(pec50_filter[1])
        lines.append(f"pEC50_filter = [{lo}, {hi}]      "
                     f"# T50 within [{pec50_to_t50(hi, float(dose_scale),):.3g}, "
                     f"{pec50_to_t50(lo, float(dose_scale),):.3g}] min")

    lines += ["",
              "['Dashboard']",
              "backend = 'webgl'",]

    for section, keys in (extra_sections or {}).items():
        lines += ["", f"['{section}']"]
        lines += [f"{k} = {fmt(v,)}" for k, v in keys.items()]

    os.makedirs(os.path.dirname(os.path.abspath(path,),), exist_ok=True,)
    with open(path, "w", encoding="utf-8",) as handle:
        handle.write("\n".join(lines) + "\n")
    return path


# =============================================================================
# 4. Running it
# =============================================================================

def run_curvecurator(toml_path: str,
                     executable: Optional[str] = None,
                     fdr: bool = True,
                     mad: bool = False,
                     extra_args: Optional[Sequence[str]] = None,
                     timeout: Optional[int] = None,
                     verbose: bool = True,) -> subprocess.CompletedProcess:
    """
    Invoke the CurveCurator CLI on a parameter file.

    CurveCurator is not installed in this project's environment and the authors recommend its
    own python 3.12 environment, so `executable` normally points at that interpreter — the
    package ships a `__main__`, making `<python> -m curve_curator <toml>` equivalent to the
    `CurveCurator` console script and independent of PATH.

    Args:
        toml_path: path to the parameter file. All paths inside it are relative to it.
        executable: interpreter or console script to run. Accepts a python interpreter (used as
            `<exe> -m curve_curator`) or the `CurveCurator` script itself. Defaults to the
            `CurveCurator` script if it is on PATH.
        fdr: pass `--fdr`, estimating the false discovery rate by target-decoy.
        mad: pass `--mad`, flagging outlier experiments by median absolute deviation.
        extra_args: further CLI arguments.
        timeout: seconds before the run is abandoned.
        verbose: echo the command and its output.

    Returns:
        The completed process, with stdout and stderr captured.

    """
    if executable is None:
        executable = shutil.which("CurveCurator",)
        if executable is None:
            raise FileNotFoundError(
                "run_curvecurator: 'CurveCurator' is not on PATH and no executable was given. "
                "CurveCurator is not part of this project's conda environment — install it "
                "separately (`pip install curve-curator`, its authors recommend a dedicated "
                "python 3.12 env) and pass that environment's python as `executable`.")

    base = os.path.basename(executable,).lower()
    if base.startswith("python",):
        cmd = [executable, "-m", "curve_curator",]
    else:
        cmd = [executable]
    cmd.append(os.path.abspath(toml_path,),)
    if fdr:
        cmd.append("--fdr",)
    if mad:
        cmd.append("--mad",)
    cmd += list(extra_args or [])

    if verbose:
        print("Running:", " ".join(cmd,))

    result = subprocess.run(cmd,
                            capture_output=True,
                            text=True,
                            timeout=timeout,)
    if verbose:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"run_curvecurator: CurveCurator exited with code "
                           f"{result.returncode}.\n{result.stderr[-2000:]}")
    return result


# =============================================================================
# 5. Reading the results back into project units
# =============================================================================

def load_curvecurator_curves(curves_file: str,
                             dose_scale: float = 1.0,
                             name_col: str = "Name",
                             site_col: str = "site",
                             annotate: Optional[pd.DataFrame] = None,
                             annotate_cols: Optional[Sequence[str]] = None,) -> pd.DataFrame:
    """
    Read a CurveCurator curves file and translate it into this project's units.

    CurveCurator writes `pEC50`, `Curve Slope`, `Curve Front`, `Curve Back`,
    `Curve Fold Change`, `Curve AUC`, `Curve RMSE`, `Curve R2`, the per-parameter errors, the
    null-model statistics, `Curve F_Value` / `Curve P_Value`, and — when the relevance
    thresholds are applied — `Curve Regulation` and the relevance score. This adds:

    - `t50_min`      : the half-response time, 10 ** (-pEC50) / dose_scale.
    - `t50_lo_min` / `t50_hi_min` : the interval implied by `pEC50 Error`, if that column is
      present. The mapping is monotone decreasing, so the *upper* pEC50 gives the *lower* time.
    - `log2_plateau` : log2 of `Curve Back` / `Curve Front`, i.e. the final level expressed the
      way the rest of this project expresses fold changes.

    Args:
        curves_file: path to the curves file written by CurveCurator.
        dose_scale: the numeric value of the `dose_scale` used in the TOML.
        name_col: CurveCurator's identifier column.
        site_col: name to give that column so it joins with project tables.
        annotate: optional DataFrame to merge annotations from, keyed on `site_col`.
        annotate_cols: which columns of `annotate` to bring across; all non-key columns if None.

    Returns:
        DataFrame of curves with the added time-domain columns, indexed as read.

    """
    curves = pd.read_csv(curves_file, sep="\t", low_memory=False,)

    if name_col in curves.columns and site_col not in curves.columns:
        curves = curves.rename(columns={name_col: site_col},)

    if "pEC50" not in curves.columns:
        raise ValueError(f"load_curvecurator_curves: no 'pEC50' column in {curves_file}. "
                         f"Columns found: {list(curves.columns)[:15]}")

    curves[T50_COL] = pec50_to_t50(curves["pEC50"], dose_scale,)

    if "pEC50 Error" in curves.columns:
        # pEC50 -> time is monotone decreasing, so the bounds swap.
        curves["t50_lo_min"] = pec50_to_t50(curves["pEC50"] + curves["pEC50 Error"], dose_scale,)
        curves["t50_hi_min"] = pec50_to_t50(curves["pEC50"] - curves["pEC50 Error"], dose_scale,)

    if {"Curve Back", "Curve Front"}.issubset(curves.columns,):
        with np.errstate(divide="ignore", invalid="ignore",):
            ratio = curves["Curve Back"] / curves["Curve Front"]
            curves["log2_plateau"] = np.log2(ratio.where(ratio > 0,),)

    if annotate is not None:
        if site_col not in annotate.columns:
            raise ValueError(f"load_curvecurator_curves: '{site_col}' not in the annotation "
                             f"table, cannot merge.")
        take = ([site_col] + list(annotate_cols) if annotate_cols is not None
                else list(annotate.columns))
        curves = curves.merge(annotate[take].drop_duplicates(subset=[site_col],),
                              on=site_col,
                              how="left",)
    return curves
