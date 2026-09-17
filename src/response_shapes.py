"""
Model-free description and classification of phosphosite temporal response shapes.

WHY THIS MODULE EXISTS
----------------------
A sigmoid is strictly monotone: y'(t) = k(y_max - y_0)*sigma*(1 - sigma) never changes
sign, so the curve has **no interior extremum**. A transient response has one. This is
not an approximation issue, it is a structural impossibility — no choice of parameters
makes a logistic rise and then fall.

That matters here because EGF-induced signalling is *mostly* transient: the prior
analysis in `notebooks/03_clustering/clustering_method_decision.md` (section 9.5) measured
59.8% transient + 9.5% biphasic under EGF, leaving only ~30.3% of sites a sigmoid may
legitimately be fitted to. Fitting a sigmoid to the rest does not raise an error — the
optimiser happily returns a plausible-looking, wrong T50 (see `src/curve_fitting.py`).

So the workflow is **classify first, fit second**, and this module is the "classify"
half. Everything in it is model-free: it assumes no functional form, it works on every
site in every dataset, and it is the fallback when a fit fails.

DATA FLOW
---------
    build_profile_matrix()   df -> (values (n_sites, T), times (T,), columns)
    per_site_sem()           df -> (n_sites, T) standard error of each value
    classify_response_shape()  values + sem -> one of six shape classes per site
    shape_census()           classes -> the counts/percentages table

    Descriptors (all sites, no model):  shape_descriptors()
    QC:                                 starve_level_summary(), replicate_count_table()
    Figures:                            plot_shape_census(), plot_shape_examples()

WHICH SCALE EACH CONSUMER USES — KEEP THIS STRAIGHT
---------------------------------------------------
    shape_descriptors            log2:FC     —
    classify_response_shape      log2:FC     per_site_sem(target="FC")
    fit_sigmoid_* (curve_fitting) log2:mean  per_site_sem(target="mean")
    center_timepoint_medians     log2:FC     —

Classification stays on the fold-change scale deliberately: it is model-free, the class
definitions and the published census are FC-based, and the FC SEM is the one the noise band
delta was calibrated against. Passing a mean-scale SEM into `classify_response_shape` would
shrink delta (it carries no starve term) and silently reclassify sites — the census would move
for reasons that have nothing to do with the data.

THE STRUCTURAL ZERO — AND WHY THE FIT DOES NOT USE IT
-----------------------------------------------------
`{cell}_log2:FC_{cond}_starve` is identically 0 for every site, because FC is *defined*
relative to starve. On this scale the starve column is a **constraint, not an observation**:
it carries no information and no variance. Classification uses only the post-stimulation
points; `build_profile_matrix` keeps starve as t = 0 and warns if it is ever non-zero.

The standing objection to that — *"if I forget about the structural 0 (the baseline) I cannot
know whether the first timepoint is staying at the same level, going up, or going down"* — is
correct, and it is the reason the **fitting** happens on `log2:mean` instead. Treating starve
as an exact zero asserts the baseline is known without error; on the mean scale it is an
estimated level with its own SEM, the starve replicates stop being discarded, and "is t = 2
above baseline?" becomes a question the fit can actually answer. See the soft-anchored model in
`src/curve_fitting.py`. The structural zero is not wrong — it is simply the wrong scale to fit
on, and this module keeps it only for the model-free classification it was built for.

NORMALISATION CAVEAT
--------------------
The project pipeline applies no sample-loading normalisation (known issue 1.1 in the
decision document: the median log2:FC across *all* sites is +0.566 at EGF 5 min, i.e. the
whole distribution is shifted, not a tail). That shift inflates the apparent response and
biases this classification. `center_timepoint_medians()` provides the FC-space stand-in so
the census can be reported both ways; it is never applied automatically.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.column_spec import ColumnSpec


# The five shape classes, plus 'incomplete' for sites with missing timepoints.
# The one-line rule for each is copied here from the classify_response_shape docstring, so the
# constant is self-documenting; `delta` is the per-site noise band z * median_t(SEM), and `p` are
# the post-stimulation values only (starve is a structural zero in FC space).
SHAPE_CLASSES = ["non_responsive",  # max|p| <= delta — no excursion distinguishable from noise
                 "monotonic",       # responsive, one-signed, and the peak IS the last timepoint
                 "sustained",       # responsive, one-signed, peak before the end, |y_peak - y_last| <= delta
                 "transient",       # responsive, one-signed, peak before the end, |y_peak - y_last| >  delta
                 "biphasic",        # responsive, and some p > delta AND some p < -delta (sign changes)
                 "incomplete",]     # any post-stimulation value missing — never classified on partial data

# Classes a monotone sigmoid may legitimately be fitted to.
SIGMOID_CLASSES = ["monotonic", "sustained",]


# =============================================================================
# 1. Building the time axis and the profile matrix
# =============================================================================

def timepoint_minutes(label: str) -> float: # manually checked
    """
    Map a timepoint label from a column name onto the numeric minute axis.

    'starve' is the zero of the post-stimulation clock (the fold-change reference).
    'full' is a biological control, not a point on that clock, and is deliberately
    mapped to NaN so it can never enter a kinetic fit.

    Args:
        label: timepoint field of a column name, e.g. 'starve', 'full', '2', '90'.

    Returns:
        Minutes as a float; 0.0 for 'starve', NaN for 'full' or any unparseable label.
    """
    if label == "starve":
        return 0.0
    try:
        return float(label)
    except (TypeError, ValueError):
        return np.nan


def build_profile_matrix(df: pd.DataFrame, # manually checked, I understand the concern but the data is well ordered in the dataset, does not harm to keep this function. OVER-COMPLICATED
                         cell_line: str = "WT",
                         condition: str = "_EGF_",
                         data_type: str = "log2:FC",
                         include_starve: bool = True,) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract one condition's temporal profile matrix, ordered on the minute axis.

    Columns are selected with ColumnSpec (project rule) and then sorted **explicitly by
    time** rather than trusting the on-disk column order. The 'full' timepoint is always
    dropped: it is a control, not a kinetic point.

    Args:
        df: DataFrame with columns following {CellLine}_{DataType}_{Condition}_{TimePoint}.
        cell_line: cell-line prefix, e.g. 'WT'.
        condition: condition token with underscores, e.g. '_EGF_'.
        data_type: transform to extract. Two are in use here and they are NOT interchangeable:
            'log2:FC' (the classification / descriptor scale, starve identically 0) and
            'log2:mean' (the *fitting* scale, on which the starve level is a real measurement
            with its own error — see the soft-anchored model in src/curve_fitting.py).
        include_starve: keep the starve column as the t = 0 anchor. Set False to get the
            post-stimulation points only.

    Returns:
        Tuple (values, times, columns):
            values: (n_sites, T) float array, time-ordered, NaN preserved.
            times: (T,) float array of minutes matching the columns.
            columns: the ordered column names actually used.
    """
    cols = ColumnSpec.select(df, cell_lines=[cell_line], data_type=data_type, conditions=[condition], exclude_full=True, exclude_replicate_cols=True,)
    if not cols:
        raise ValueError(f"build_profile_matrix: no '{data_type}' columns found for cell_line='{cell_line}', condition='{condition}'.")

    labels = [c.split("_")[3] for c in cols] # the 4th element "[3]" is the time point
    minutes = np.array([timepoint_minutes(l) for l in labels], dtype=float,) # Return the time points as a float's array

    keep = np.isfinite(minutes) # round minutes
    if not include_starve:
        keep &= minutes > 0
    cols = [c for c, k in zip(cols, keep) if k]
    minutes = minutes[keep]

    order = np.argsort(minutes,)
    ordered_cols = [cols[i] for i in order]
    times = minutes[order]

    values = df[ordered_cols].to_numpy(dtype=float,)

    # The structural-zero warning applies ONLY to the fold-change scale. On 'log2:FC' the starve
    # column is identically 0 by construction and a non-zero value means something upstream is
    # broken, so warn loudly. On 'log2:mean' the starve column is a genuine intensity measurement
    # of order 15-25 log2 units — firing the same warning there would be a guaranteed false alarm
    # on every call. Use `starve_level_summary()` to report that distribution instead.
    is_fold_change = data_type.split(":")[-1] == "FC"
    if is_fold_change and include_starve and times[0] == 0.0:
        starve = values[:, 0]
        max_abs = np.nanmax(np.abs(starve)) if np.isfinite(starve).any() else 0.0
        if max_abs > 1e-9:
            warnings.warn(f"build_profile_matrix: '{ordered_cols[0]}' is not identically zero "
                          f"(max |value| = {max_abs:.3g}). The fold change is expected to be "
                          f"defined relative to starve; anchored fits assume y(0) = 0.")

    return values, times, ordered_cols


def starve_level_summary(values: np.ndarray,
                         times: np.ndarray,) -> pd.Series:
    """
    Describe the starve (t = 0) column of a profile matrix across sites.

    The mean-scale counterpart of the structural-zero warning in `build_profile_matrix`. On
    'log2:FC' the starve column is identically 0 and there is nothing to describe; on
    'log2:mean' it is the baseline level the soft-anchored fit estimates `y0` from, so its
    distribution is the thing worth looking at — it should be a plausible spread of log2
    intensities (order 15-25 on TMT), not a spike at zero and not a column of NaN.

    Args:
        values: (n_sites, T) profile matrix from build_profile_matrix.
        times: (T,) minute axis; the t = 0 column is the one summarised.

    Returns:
        Series with n_sites, n_finite, and the min / 5% / 25% / 50% / 75% / 95% / max of the
        starve level across sites.

    """
    if not np.any(times == 0.0):
        raise ValueError("starve_level_summary: the profile matrix has no t = 0 (starve) column.")

    starve = values[:, int(np.where(times == 0.0)[0][0])]
    finite = starve[np.isfinite(starve)]
    if finite.size == 0:
        raise ValueError("starve_level_summary: the starve column is entirely NaN.")

    q = np.percentile(finite, [0, 5, 25, 50, 75, 95, 100],)
    return pd.Series({"n_sites": float(starve.size),
                      "n_finite": float(finite.size),
                      "min": q[0],
                      "p05": q[1],
                      "q25": q[2],
                      "median": q[3],
                      "q75": q[4],
                      "p95": q[5],
                      "max": q[6],},)

def per_site_sem(df: pd.DataFrame,
                 times: np.ndarray,
                 cell_line: str = "WT",
                 condition: str = "_EGF_",
                 data_type: str = "log2:FC",
                 target: str = "FC",
                 moderate: bool = True,
                 prior_df: float = 2.0,
                 min_n: int = 3,
                 n_reps_col: str = "n:reps",) -> np.ndarray:
    """
    Standard error of each profile value, per site and timepoint, on the requested scale.

    TWO SCALES, TWO FORMULAE — `target` picks which, and they are not interchangeable.

    `target="FC"` (default, unchanged behaviour). A fold change is a difference of two means,
    FC(t) = mean(t) - mean(starve), so its variance is the sum of both variances:

        SEM_FC(t) = sqrt( var(t)/n_t + var(starve)/n_starve ),   SEM_FC(starve) := 0

    ⚠️ These are **marginal** SEMs of *correlated* quantities. Every timepoint carries the same
    -mean(starve) term, so Cov(FC(t_i), FC(t_j)) = Var(mean(starve)) > 0 for every pair — the
    shared term this branch adds into each marginal SEM is exactly the covariance it then
    ignores. A diagonal weight matrix (weights = 1/SEM, which is what `fit_sigmoid_site` uses)
    assumes independence, so **this branch is not the one to fit on**; handling it correctly
    would need the full covariance matrix, which is singular at the anchor. Use it for the
    model-free, threshold-based work — `classify_response_shape`, the noise band delta — where
    only the marginal scale of the noise matters.

    `target="mean"`. No propagation and no structural zero: every timepoint, starve included,
    gets the SEM of its own mean,

        SEM_mean(t) = sqrt( var(t)/n_t )

    Different replicate measurements at different timepoints, so these errors are independent
    **by construction** and the existing diagonal weighting becomes correct rather than
    approximate. This is the branch `fit_sigmoid_site(..., free_baseline=True)` fits on.
    Note the trap this avoids: patching only the starve column to a non-zero value while
    leaving the propagation sum in place would inflate *every* SEM by the starve term.

    Replicate counts are taken per timepoint from the log2:abs replicate columns when they
    are present (exact), falling back to the dataset-level 'n:reps' column.

    VARIANCE MODERATION. With only 4 replicates a per-site SD has 3 degrees of freedom and is
    itself very noisy, which makes a per-site noise band unstable. `moderate=True` shrinks each
    site's variance toward the median variance at that timepoint:

        s2_moderated = (prior_df * s2_prior + d * s2_site) / (prior_df + d),   d = n - 1

    This is the *idea* behind limma's empirical-Bayes variance moderation (Smyth 2004) with a
    fixed prior weight — it is a deliberate simplification, not limma: the prior variance is
    the observed median rather than a fitted hyperparameter. `prior_df=0` reproduces
    `moderate=False` exactly (the notebook asserts this).

    ⚠️ `prior_df` is not a cosmetic knob. It propagates into the headline shape census through
    `delta = z * median(SEM)`, which gates which sites are fitted at all, so the percentage moves
    for a reason that is invisible unless it is swept. The weight it takes is prior_df/(prior_df
    + d) with d = n - 1, so an inherited prior_df = 4.0 gives the prior **57% at n=4** and
    **80% at n=3** — and n=3 is a third of the replicate counts in hme1_2. That is the prior
    outvoting the data.

    Measured sweep (hme1_2, WT, EGF, responsive denominator, Z_BAND = 2.5; the cell that
    produces it is in `notebooks/06_sigmoids/Sigmoid_fitting.ipynb`):

        prior_df   median delta   % sustained   % transient   % sigmoid-legitimate
        0            0.272          40.4          37.1          44.0
        1            0.274          42.8          36.4          46.5
        2            0.276          43.4          35.9          47.1
        4            0.279          43.4          35.4          47.1
        8            0.281          42.8          35.6          46.4

    The census rises by 3 points from 0 to 2 and is then **flat**: 2 and 4 differ by 0.04
    points. The default is therefore **2.0** — on the plateau, but giving the prior 40% at n=4
    and 50% at n=3 instead of 57% and 80%. It is a choice, made from this table, not inherited.

    WHY NOT limma? (a standing question on this function). limma returns **one pooled residual
    variance per site** — homoscedastic across timepoints, after the plex blocking term — built
    for hypothesis testing. A single pooled sigma would weight every timepoint identically,
    which defeats the entire purpose of per-timepoint fit weights: the per-timepoint SD varies
    roughly two-fold across this series, and down-weighting the noisy points is the point.
    `per_site_sem` deliberately keeps the per-timepoint structure limma pools away, while
    borrowing limma's shrinkage idea. `per_site_sem` and limma's `s2.post` are **not** competing
    estimates of the same quantity and must not be reconciled.

    Args:
        df: DataFrame carrying the {cell}_log2:sd_{cond}_{tp} columns.
        times: minute axis returned by build_profile_matrix (0.0 = starve).
        cell_line: cell-line prefix.
        condition: condition token with underscores.
        data_type: the transform the profiles were taken from; only its prefix is used, so
            both 'log2:FC' and 'log2:mean' look up the 'log2:sd' and 'log2:abs' columns.
        target: 'FC' for the propagated fold-change SEM (starve column 0), 'mean' for the
            independent per-timepoint SEM of the mean (starve column real).
        moderate: apply the variance moderation described above.
        prior_df: weight of the prior variance, in pseudo-degrees-of-freedom.
        min_n: smallest replicate count admitted as a fit weight. An n=2 SD has 1 degree of
            freedom: usable in a limma test, where moderation carries it, but poor as a weight,
            so the SEM path defaults to 3. n=1 is excluded everywhere (no variance exists);
            n=2 may stay in the differential test but not in the weighting; n>=3 is fine.
            Measured cost on hme1_2 (WT, EGF): raising it from 2 to 3 blanks 32.6% -> 54.1% of
            SEM cells across *all* 50002 sites, which looks alarming and is not — among the
            responsive sites every per-timepoint count is 3 or 4 (exactly one cell at 2), and
            all 7816 shape-gated sites keep all six timepoints under either setting. The
            discarded cells are entirely in sites that never reach the fit.
        n_reps_col: fallback column holding the replicate count.

    Returns:
        (n_sites, T) array of standard errors, aligned with the `times` axis. With
        target='FC' the starve column is exactly 0 (a structural zero has no variance — its
        uncertainty is already carried by the post-stimulation points); with target='mean' it
        carries the real SEM of the starve mean. NaN where the underlying SD is missing or the
        replicate count is below `min_n`.
    """
    if target not in ("FC", "mean",):
        raise ValueError(f"per_site_sem: target must be 'FC' or 'mean', got {target!r}.")

    prefix = data_type.split(":")[0]                  # 'log2:FC' -> 'log2'
    n_sites = len(df)

    sd_cols = ColumnSpec.select(df,
                                cell_lines=[cell_line],
                                data_type=f"{prefix}:sd",
                                conditions=[condition],
                                exclude_full=True,
                                exclude_replicate_cols=True,)
    if not sd_cols:
        raise ValueError(f"per_site_sem: no '{prefix}:sd' columns found for "
                         f"cell_line='{cell_line}', condition='{condition}'.")
    sd_by_time = {timepoint_minutes(c.split("_")[3]): c for c in sd_cols}

    abs_cols = ColumnSpec.select(df,
                                 cell_lines=[cell_line],
                                 data_type=f"{prefix}:abs",
                                 conditions=[condition],
                                 exclude_full=True,)

    def _variance_and_n(t: float,) -> Tuple[np.ndarray, np.ndarray]:
        """
        Per-site variance and replicate count of the mean at one timepoint.

        Args:
            t: timepoint in minutes.

        Returns:
            Tuple (variance, n) of (n_sites,) arrays; NaN where undefined.
        """
        col = sd_by_time.get(t)
        if col is None:
            return np.full(n_sites, np.nan,), np.full(n_sites, np.nan,)

        var = df[col].to_numpy(dtype=float,) ** 2

        reps_here = [c for c in abs_cols
                     if timepoint_minutes(c.split("_")[3]) == t]
        if reps_here:
            n = df[reps_here].notna().sum(axis=1,).to_numpy(dtype=float,)
        elif n_reps_col in df.columns:
            n = df[n_reps_col].to_numpy(dtype=float,)
        else:
            n = np.full(n_sites, np.nan,)
        # An SD needs at least 2 replicates to exist at all; min_n additionally refuses the
        # 1-degree-of-freedom n=2 case as a fit weight (see the Args note on min_n).
        n = np.where(n >= max(2, min_n,), n, np.nan,)

        if moderate:
            prior_var = np.nanmedian(var)
            d = n - 1.0
            var = (prior_df * prior_var + d * var) / (prior_df + d)
        return var, n

    starve_var, starve_n = _variance_and_n(0.0,)

    sem = np.full((n_sites, len(times)), np.nan,)
    for j, t in enumerate(times):
        var, n = _variance_and_n(t,)
        if target == "mean":
            # No subtraction happened, so there is nothing to propagate: each timepoint carries
            # only the error of its own mean, and starve is a measurement like any other.
            sem[:, j] = np.sqrt(var / n)
            continue
        if t == 0.0:
            # The starve column of a fold change is a structural zero with no variance;
            # its uncertainty is carried by the post-stimulation points instead.
            sem[:, j] = 0.0
            continue
        sem[:, j] = np.sqrt(var / n + starve_var / starve_n)

    return sem


def replicate_count_table(df: pd.DataFrame,
                          times: np.ndarray,
                          cell_line: str = "WT",
                          condition: str = "_EGF_",
                          data_type: str = "log2:FC",
                          min_n: int = 3,) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    QC the per-timepoint replicate counts that `per_site_sem` weights on.

    WHY THIS EXISTS. The per-site `n:reps` filter is a single number for the whole time course,
    so it cannot see *where* the replicates are. A site with 4 replicates at starve and 2 at
    90 min has a trajectory whose halves differ in reliability: `per_site_sem` propagates that
    correctly into the weights, but nothing upstream ever reports it, and an uneven site can
    pass `n:reps >= 3` while contributing a badly determined tail to the fit.

    Args:
        df: DataFrame carrying the {cell}_log2:abs_{cond}_{tp}_{rep} columns.
        times: minute axis from build_profile_matrix (0.0 = starve).
        cell_line: cell-line prefix.
        condition: condition token with underscores.
        data_type: transform whose prefix locates the replicate columns ('log2:FC' -> 'log2:abs').
        min_n: the SEM weighting threshold, reported as `n_below_min` per timepoint.

    Returns:
        Tuple (per_timepoint, counts):
            per_timepoint: one row per timepoint with the mean/median/min replicate count, how
                many sites fall below `min_n`, and how many have no replicates at all.
            counts: (n_sites, T) DataFrame of the raw per-timepoint replicate counts, with an
                `n_min` / `n_max` / `uneven` triple appended — `uneven` flags sites whose count
                is not constant across the time course.

    """
    prefix = data_type.split(":")[0]
    abs_cols = ColumnSpec.select(df,
                                 cell_lines = [cell_line],
                                 data_type  = f"{prefix}:abs",
                                 conditions = [condition],
                                 exclude_full = True,)
    if not abs_cols:
        raise ValueError(f"replicate_count_table: no '{prefix}:abs' replicate columns found for "
                         f"cell_line='{cell_line}', condition='{condition}'.")

    counts = {}
    for t in times:
        reps_here = [c for c in abs_cols if timepoint_minutes(c.split("_")[3]) == t]
        counts[t] = (df[reps_here].notna().sum(axis=1,).to_numpy(dtype=float,) if reps_here
                     else np.full(len(df), np.nan,))

    counts = pd.DataFrame(counts, index=df.index,)
    per_tp = pd.DataFrame({"mean_n": counts.mean(),
                           "median_n": counts.median(),
                           "min_n": counts.min(),
                           "max_n": counts.max(),
                           "n_below_min": (counts < min_n).sum(),
                           "n_zero": (counts == 0).sum(),},)
    per_tp.index.name = "timepoint_min"

    counts["n_min"] = counts[list(times)].min(axis=1,)
    counts["n_max"] = counts[list(times)].max(axis=1,)
    counts["uneven"] = counts["n_max"] != counts["n_min"]
    return per_tp, counts


def center_timepoint_medians(values: np.ndarray,) -> np.ndarray:
    """
    Remove the across-site median at each timepoint (median centring in fold-change space).

    The pipeline applies no sample-loading normalisation: the 84 raw:abs column medians span
    2.1-fold, which propagates into every fold change and shifts the *entire* FC distribution
    (median log2:FC = +0.566 at EGF 5 min against an IQR of only ~0.33-0.61). A shared shift
    at a timepoint is a loading artefact, not biology — no plausible biology moves every
    phosphosite in the same direction at once.

    Subtracting the per-timepoint median across sites is the fold-change-space equivalent of
    median-centring each sample in log2 space. It is a stand-in, not a fix: the real repair
    belongs in `src/transformations.py` before compute_log2_stats.

    ⚠️ FOLD-CHANGE SCALE ONLY. This is written for a `log2:FC` matrix, where the shared shift is
    a *response* offset and subtracting it leaves the anchor at zero. Do not apply it to a
    `log2:mean` matrix: there the loading shift is an absolute level offset, and removing the
    across-site median would also remove the site-to-site spread in baseline abundance — which
    is real biology and is precisely what the soft-anchored fit estimates `y0` from. A
    mean-space equivalent would have to centre each *sample column* on a common reference rather
    than on its own median across sites, and that repair belongs in `src/transformations.py`.
    Neither version is ever applied automatically.

    Args:
        values: (n_sites, T) profile matrix from build_profile_matrix, on the log2:FC scale.

    Returns:
        (n_sites, T) copy with the column-wise across-site median subtracted.
    """
    centered = values.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning,)
        medians = np.nanmedian(centered, axis=0,)
    return centered - medians[None, :]


# =============================================================================
# 2. Model-free descriptors — computable for every site
# =============================================================================

def shape_descriptors(values: np.ndarray,
                      times: np.ndarray,
                      index: Optional[pd.Index] = None,) -> pd.DataFrame:
    """
    Summarise each temporal profile with descriptors that assume no functional form.

    These are the safety net: they exist for every site in every dataset, including the
    ~70% of EGF sites no sigmoid may be fitted to, and they need no optimiser. Their
    weakness is that `peak_time` is quantised to the sampling grid (under EGF, 73% of sites
    peak at either 5 or 10 min), which is exactly the limitation a parametric fit removes.

    Args:
        values: (n_sites, T) profile matrix, time-ordered, starve included as t = 0.
        times: (T,) minute axis.
        index: optional pandas index to attach (e.g. df.index) so the result joins back.

    Returns:
        DataFrame with one row per site:
            signed_peak       value of largest |FC| among post-stimulation points, sign kept
            peak_time         minute at which that peak occurs
            last_value        FC at the final timepoint
            amplitude         max - min across post-stimulation points
            retention_90      last_value / signed_peak (1 = fully sustained, 0 = fully reversed)
            transience_index  1 - retention_90
            auc_logtime       trapezoidal area on the log10(t+1) axis
            initial_slope     (first post-stimulation value - 0) / its time
            decay_halflife    ln(2)/rate from a log-linear fit of the post-peak points (min)
            n_finite          number of non-missing post-stimulation points
    """
    post = times > 0
    p = values[:, post]
    t_post = times[post]
    x_all = np.log10(times + 1.0)

    n_sites = values.shape[0]
    n_finite = np.isfinite(p).sum(axis=1,)
    valid = n_finite > 0

    signed_peak = np.full(n_sites, np.nan,)
    peak_time = np.full(n_sites, np.nan,)
    last_value = np.full(n_sites, np.nan,)
    amplitude = np.full(n_sites, np.nan,)
    auc = np.full(n_sites, np.nan,)
    initial_slope = np.full(n_sites, np.nan,)
    halflife = np.full(n_sites, np.nan,)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning,)

        # Peak = largest absolute excursion, keeping its sign.
        abs_p = np.where(np.isfinite(p), np.abs(p), -np.inf,)
        peak_idx = np.argmax(abs_p, axis=1,)
        rows = np.arange(n_sites,)
        signed_peak[valid] = p[rows[valid], peak_idx[valid]]
        peak_time[valid] = t_post[peak_idx[valid]]

        amplitude[valid] = (np.nanmax(p[valid], axis=1,)
                            - np.nanmin(p[valid], axis=1,))

        # Last observed (not necessarily last column, if the tail is missing).
        for i in np.where(valid)[0]:
            finite = np.where(np.isfinite(p[i]))[0]
            last_value[i] = p[i, finite[-1]]

        complete = np.isfinite(values).all(axis=1,)
        auc[complete] = np.trapz(values[complete], x_all, axis=1,)

        if t_post.size:
            initial_slope = p[:, 0] / t_post[0]

        # Decay half-life: log-linear regression of |y| on the post-peak points, only where
        # the site actually declines and keeps its sign.
        for i in np.where(valid)[0]:
            k = peak_idx[i]
            tail_t = t_post[k:]
            tail_y = p[i, k:]
            ok = np.isfinite(tail_y) & (np.sign(tail_y) == np.sign(signed_peak[i])) & (tail_y != 0)
            if ok.sum() < 2:
                continue
            slope = np.polyfit(tail_t[ok], np.log(np.abs(tail_y[ok])), 1,)[0]
            if slope < 0:
                halflife[i] = np.log(2.0) / (-slope)

        retention = last_value / np.where(signed_peak == 0, np.nan, signed_peak,)

    out = pd.DataFrame({"signed_peak": signed_peak,
                        "peak_time": peak_time,
                        "last_value": last_value,
                        "amplitude": amplitude,
                        "retention_90": retention,
                        "transience_index": 1.0 - retention,
                        "auc_logtime": auc,
                        "initial_slope": initial_slope,
                        "decay_halflife": halflife,
                        "n_finite": n_finite,},
                       index=index if index is not None else np.arange(n_sites,),)
    return out


# =============================================================================
# 3. Shape classification
# =============================================================================

def classify_response_shape(values: np.ndarray,
                            times: np.ndarray,
                            sem: np.ndarray,
                            z: float = 2.5,
                            index: Optional[pd.Index] = None,) -> pd.DataFrame:
    """
    Assign each site one of six response-shape classes, using a noise-aware band.

    The band is delta_i = z * median_t(SEM_i,t): an excursion smaller than delta is not
    distinguishable from measurement noise for that site. Classification runs on the
    **post-stimulation** points only, since starve is a structural zero.

        incomplete     : any post-stimulation value missing (classified on partial data
                         would be silently wrong — DIA/TMT missingness is not random)
        non_responsive : max|p| <= delta
        biphasic       : responsive AND some p > delta AND some p < -delta
        monotonic      : responsive AND not biphasic AND the peak IS the last timepoint
        sustained      : responsive AND not biphasic AND peak before the end AND
                         |y_peak - y_last| <= delta   (declines, but not detectably)
        transient      : responsive AND not biphasic AND peak before the end AND
                         |y_peak - y_last| >  delta   (detectable decline from the peak)

    Only `monotonic` and `sustained` may be fitted with a monotone sigmoid; the rule set is
    taken from clustering_method_decision.md section 9.5.

    Args:
        values: (n_sites, T) profile matrix, time-ordered, starve included.
        times: (T,) minute axis.
        sem: (n_sites, T) standard errors from per_site_sem, same ordering.
        z: width of the noise band in standard errors (2.5 by default).
        index: optional pandas index to attach.

    Returns:
        DataFrame with columns:
            shape_class         one of SHAPE_CLASSES
            sigmoid_legitimate  True for monotonic / sustained
            delta               the per-site noise band, in log2 units
            peak_time, signed_peak, last_value  (repeated here for convenience)
    """
    post = times > 0
    p = values[:, post]
    t_post = times[post]
    sem_post = sem[:, post]

    n_sites = p.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning,)
        delta = z * np.nanmedian(sem_post, axis=1,)

    complete = np.isfinite(p).all(axis=1,) & np.isfinite(delta)

    abs_p = np.where(np.isfinite(p), np.abs(p), -np.inf,)
    peak_idx = np.argmax(abs_p, axis=1,)
    rows = np.arange(n_sites,)
    signed_peak = p[rows, peak_idx]
    peak_time = t_post[peak_idx]
    last_value = p[:, -1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning,)
        max_abs = np.nanmax(np.abs(p), axis=1,)
        up = (p > delta[:, None]).any(axis=1,)
        down = (p < -delta[:, None]).any(axis=1,)

    responsive = max_abs > delta
    is_last = peak_idx == (p.shape[1] - 1)
    decline = np.abs(signed_peak - last_value) > delta

    shape = np.full(n_sites, "incomplete", dtype=object,)
    ok = complete
    shape[ok & ~responsive] = "non_responsive"
    shape[ok & responsive & up & down] = "biphasic"
    mono_family = ok & responsive & ~(up & down)
    shape[mono_family & is_last] = "monotonic"
    shape[mono_family & ~is_last & ~decline] = "sustained"
    shape[mono_family & ~is_last & decline] = "transient"

    out = pd.DataFrame({"shape_class": shape,
                        "sigmoid_legitimate": np.isin(shape, SIGMOID_CLASSES,),
                        "delta": delta,
                        "peak_time": np.where(complete, peak_time, np.nan,),
                        "signed_peak": np.where(complete, signed_peak, np.nan,),
                        "last_value": np.where(complete, last_value, np.nan,),},
                       index=index if index is not None else np.arange(n_sites,),)
    return out


def shape_census(shape_class: pd.Series,
                 groups: Optional[Dict[str, np.ndarray]] = None,) -> pd.DataFrame:
    """
    Tabulate how many sites fall in each shape class, and what fraction that is.

    The denominator matters more than the numerator here: a census over all quantified
    sites and a census over statistically responsive sites answer different questions, and
    a "% fittable to a sigmoid" number is meaningless without saying which was used. Pass
    both as groups and report them side by side.

    Args:
        shape_class: Series of class labels from classify_response_shape.
        groups: optional {group_name: boolean mask} to compute one column pair per group.
            The mask must align with `shape_class`. Default: a single 'all' group.

    Returns:
        DataFrame indexed by shape class (SHAPE_CLASSES order, plus the two summary rows
        'sigmoid_legitimate' and 'needs_transient_model'), with an `n_{group}` and a
        `pct_{group}` column per group.
    """
    if groups is None:
        groups = {"all": np.ones(len(shape_class), dtype=bool,)}

    table = {}
    for name, mask in groups.items():
        subset = shape_class[np.asarray(mask, dtype=bool,)]
        counts = subset.value_counts()
        n_total = len(subset)

        col_n = [int(counts.get(c, 0)) for c in SHAPE_CLASSES]
        # Summary rows are computed over the classified sites only — 'incomplete' sites were
        # never classified, so including them in the denominator would understate both.
        n_classified = n_total - int(counts.get("incomplete", 0))
        n_sigmoid = sum(int(counts.get(c, 0)) for c in SIGMOID_CLASSES)
        n_transient = sum(int(counts.get(c, 0)) for c in ("transient", "biphasic",))

        col_n += [n_sigmoid, n_transient]
        denom = n_classified if n_classified else np.nan
        col_pct = [100.0 * n / n_total if n_total else np.nan for n in col_n[:len(SHAPE_CLASSES)]]
        col_pct += [100.0 * n_sigmoid / denom, 100.0 * n_transient / denom]

        table[f"n_{name}"] = col_n
        table[f"pct_{name}"] = col_pct

    return pd.DataFrame(table,
                        index=SHAPE_CLASSES + ["sigmoid_legitimate", "needs_transient_model",],)


# =============================================================================
# 4. Figures
# =============================================================================

def plot_shape_census(census: pd.DataFrame,
                      figsize: Tuple[float, float] = (7, 5),
                      title: str = "Response-shape census",) -> Tuple[plt.Figure, plt.Axes]:
    """
    Stacked bar of the shape composition, one bar per group in the census table.

    Only the five real classes plus 'incomplete' are drawn; the two summary rows are
    derived quantities and would double-count.

    Args:
        census: table returned by shape_census().
        figsize: figure size in inches.
        title: axes title.

    Returns:
        Tuple (figure, axes).
    """
    pct_cols = [c for c in census.columns if c.startswith("pct_")]
    groups = [c.replace("pct_", "") for c in pct_cols]
    data = census.loc[SHAPE_CLASSES, pct_cols]

    fig, ax = plt.subplots(figsize=figsize,)
    bottom = np.zeros(len(groups),)
    colors = plt.cm.viridis(np.linspace(0, 0.92, len(SHAPE_CLASSES),),)

    for cls, color in zip(SHAPE_CLASSES, colors,):
        heights = data.loc[cls].to_numpy(dtype=float,)
        ax.bar(groups,
               heights,
               bottom=bottom,
               color=color,
               edgecolor="white",
               label=cls,)
        for xi, (h, b) in enumerate(zip(heights, bottom,)):
            if h > 3:
                ax.text(xi,
                        b + h / 2,
                        f"{h:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white",)
        bottom += np.nan_to_num(heights,)

    ax.set_ylabel("% of sites")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False,)
    fig.tight_layout()
    return fig, ax


def plot_shape_examples(values: np.ndarray,
                        times: np.ndarray,
                        shapes: pd.DataFrame,
                        sem: Optional[np.ndarray] = None,
                        n_per_class: int = 4,
                        labels: Optional[Sequence[str]] = None,
                        random_state: int = 0,
                        figsize_per_panel: Tuple[float, float] = (2.6, 2.2),):
    """
    Draw example profiles from each shape class, so the classification can be eyeballed.

    A rule-based classifier on 5 noisy points deserves visual inspection before its output
    is used to gate a fit — this is that inspection. The shaded band is the per-site noise
    band delta used to make the call.

    Args:
        values: (n_sites, T) profile matrix.
        times: (T,) minute axis.
        shapes: output of classify_response_shape (must align row-wise with `values`).
        sem: optional (n_sites, T) errors, drawn as error bars.
        n_per_class: number of example sites per class.
        labels: optional per-site labels (e.g. df['site']) used as panel titles.
        random_state: seed for choosing which examples to show.
        figsize_per_panel: size of each subplot in inches.

    Returns:
        Tuple (figure, axes array).
    """
    rng = np.random.default_rng(random_state,)
    present = [c for c in SHAPE_CLASSES if (shapes["shape_class"] == c).any()]

    fig, axes = plt.subplots(len(present),
                             n_per_class,
                             figsize=(figsize_per_panel[0] * n_per_class,
                                      figsize_per_panel[1] * len(present),),
                             squeeze=False,)
    x = np.log10(times + 1.0)

    for r, cls in enumerate(present):
        pos = np.where((shapes["shape_class"] == cls).to_numpy())[0]
        pick = rng.choice(pos, size=min(n_per_class, pos.size), replace=False,)
        for c in range(n_per_class):
            ax = axes[r, c]
            if c >= pick.size:
                ax.axis("off")
                continue
            i = pick[c]
            yerr = sem[i] if sem is not None else None
            ax.errorbar(x,
                        values[i],
                        yerr=yerr,
                        marker="o",
                        ms=4,
                        color="black",
                        capsize=2,
                        elinewidth=1.0,
                        lw=1.4,)
            d = shapes["delta"].to_numpy()[i]
            ax.axhspan(-d, d, color="tab:orange", alpha=0.15,)
            ax.axhline(0, color="grey", lw=0.7,)
            ax.set_xticks(x,)
            ax.set_xticklabels([f"{t:g}" for t in times], fontsize=7,)
            name = labels[i] if labels is not None else f"row {i}"
            ax.set_title(str(name)[:22], fontsize=7,)
            if c == 0:
                ax.set_ylabel(cls, fontsize=9, weight="bold",)

    fig.supxlabel("time (min, log-spaced)")
    fig.tight_layout()
    return fig, axes
