"""
Anchored sigmoid fitting of phosphosite temporal responses, and the T50 that follows.

THE MODEL
---------
With x = log10(t + 1) and t in minutes:

    y(x) = y0 + A * [ sigma(k * (x - x50))  -  sigma(-k * x50) ],   sigma(u) = 1/(1 + exp(-u))

The bracketed term is the **anchored** logistic: the subtracted sigma(-k*x50) makes it vanish
identically at x = 0, which is what "anchored" means. `y0` is a *soft* anchor — the level the
curve starts from — and whether it is free is the one modelling choice this module exposes:

    free_baseline=False (hard anchor, y0 == 0)   fit `log2:FC`, where
        `{cell}_log2:FC_{cond}_starve` is identically 0 because the fold change is *defined*
        relative to starve. The baseline is a constraint, not a parameter.

    free_baseline=True  (soft anchor, y0 fitted)  fit `log2:mean`, where the starve level is a
        real measurement carrying its own SEM. **This is the recommended mode.**

WHY THE SOFT ANCHOR, AND WHY IT COSTS NOTHING
---------------------------------------------
The degrees-of-freedom ledger is *neutral*. Fitting the mean adds one observation (starve stops
being a structural zero that has to be dropped from the residual vector) and one parameter:

    scale   k       observations   free params        lack-of-fit df
    FC      free    5              A, k, x50     = 3        2
    FC      fixed   5              A, x50        = 2        3
    mean    free    6              y0, A, k, x50 = 4        2
    mean    fixed   6              y0, A, x50    = 3        3

So "you do not have 6 data points, you have 5 informative points plus one structural zero" is
right about *information content* but wrong to conclude the offset is unaffordable — the
observation it buys back pays for it exactly.

What the soft anchor gains:

 1. **Uncorrelated errors — the real argument.** Every `log2:FC` timepoint contains the same
    -mean(starve) term, so Cov(FC(t_i), FC(t_j)) = Var(mean(starve)) > 0 for *every* pair. The
    weighting here is a **diagonal** matrix (weights = 1/SEM), which assumes independence — an
    assumption the FC scale violates, and `per_site_sem(target="FC")` computes the offending
    shared term explicitly while folding it into each marginal SEM. Handling that properly would
    need the full covariance matrix, which is singular at the anchor. On the mean scale the
    errors are independent by construction and the diagonal weighting becomes *correct* rather
    than approximate.
 2. **The starve replicates start doing work.** Under the hard anchor they are discarded
    entirely (the t = 0 point is dropped from the residual vector). On the mean scale they
    constrain y0 with weight 1/sqrt(var/n).
 3. **It answers the objection in `response_shapes.py`** — "if I forget about the structural 0 I
    cannot know whether the first timepoint is staying level, going up, or going down". With an
    estimated baseline carrying an uncertainty, that is a question the fit can answer.

⚠️ What it does **not** gain: accuracy. Median RMSE and the fitted k / x50 stay close to the
hard-anchored values whenever SEM(starve) is small relative to the post-stimulation SEMs. This
is a change in **statistical honesty, not fit quality** — the same category as the log-time
argument below ("not a better fit; a better parameterisation"). Report it in those terms.

MEASURED, hme1_2 (WT, EGF, 1500 shape-gated sites fitted both ways, k fixed at 10)
----------------------------------------------------------------------------------
The two fits agree, and where they disagree it is where the theory says they should:

    median |Delta T50|              0.153 min  =  **4.7% of the median per-site SE (3.25 min)**
    74% of sites agree to within 0.1 * SE
    Spearman(T50) on sites above 3 min   **0.981**   (uncensored [2,60] min: 0.917)
    Spearman(T50) over everything        0.800  <- censoring, not disagreement: 40% of sites
        sit in [2,3] min against the 2 min left-censoring bound, where ranks are noise. Below
        2 min the two agree at only 0.51 — and there T50 must be reported as "< 2 min" anyway.
    Spearman: A 0.947, plateau 0.972, RMSE 0.979, SE(T50) 0.957
    |Delta T50| correlates with SEM(starve): rho = 0.215, p = 4e-17; the noisier half of sites
        disagrees by 0.185 min against 0.129 min for the well-measured half — exactly the
        predicted pattern, and the single scatter plot that argues for the change.

One consequence to expect rather than be surprised by: **the lack-of-fit test gets stricter.**
Sites not rejected fall from 45.3% to 20.5%, because the FC-scale SEM is inflated by the shared
starve term (median 0.113 vs 0.082 on the mean scale) and inflated errors make any model look
adequate. Median chi-square rises from 8.28 to 14.43 on identical residuals — median RMSE
actually *improves* slightly, 0.139 to 0.131. The old pass rate was flattering the fit.

Everything derived from the fit carries over unchanged, because the bracketed term is untouched
and A is still the amplitude: `plateau_value` is the level reached **relative to the fitted
baseline**, and `half_response_time` is independent of both A and y0.

Degrees of freedom, which is the whole identifiability story (decision document section 9.4):

    lack-of-fit df = (distinct timepoints) - 1 - (free parameters) = 6 - 1 - 3 = 2

Two df is thin but real: enough to fit, to test the model against a flat null, and to get
finite (if wide) confidence intervals. **Replicates do not add to this** — they buy
pure-error df (a better estimate of sigma), not lack-of-fit df. A five-parameter impulse
model would have 0 lack-of-fit df on this design: it interpolates any site, including pure
noise, and is therefore not fitted here. Moving to the mean scale does not rescue it either
(6 points, 5 parameters -> 1 df, still untestable); the ~70% of EGF sites that are transient or
biphasic are served by the model-free descriptors in `src.response_shapes.shape_descriptors`.

WHY LOG-TIME
------------
The timepoints {0, 2, 5, 10, 15, 90} span a factor of 45 but are sampled densely at the
short end. In linear time the 90-min point has leverage **0.979** — one measurement
essentially dictates the fit — against 0.659 in log10(t+1); `leverage()` and
`plot_leverage()` compute this for any design. Log-time is *not* a better fit (measured
median RMSE 0.1048 linear vs 0.1051 log — indistinguishable); it is a better
**parameterisation**: k varies 30-fold with T50 in linear time versus 1.5-fold in log-time,
so only in log-time is k a shape parameter independent of timing. And 10% of linear-time
fits return a physically meaningless negative T50, against a hard floor of 0 in log-time.
`compare_time_axes()` reproduces this on your own data.

⚠️ STEEPNESS IS NOT IDENTIFIABLE ON THIS DESIGN — FIT 2 PARAMETERS, NOT 3
--------------------------------------------------------------------------
Measured on hme1_2 (WT, EGF, 7810 shape-gated sites): fitting all three parameters drives k to
whatever upper bound it is given for **92.8%** of sites, and `profile_global_k` — which fits one
shared k across sites — returns a total chi-square that decreases *monotonically* to the bound
with no interior minimum. Both say the same thing, and it is not a numerical problem:

    the first post-stimulation samples are at 2, 5 and 10 min, and most sites complete their
    rise between two of them, so the data cannot distinguish a steep sigmoid from a step.

Note the parallel: `council_note_3_temporal_curve_modelling.md` measured exactly this for the
impulse model's beta (76.5% of fits driven into step-function territory), and recommended fitting
the shared shape parameter globally and then fixing it. That is `fixed_k` here.

**Fixing k is safe for the quantities that matter**, which was checked rather than assumed
(800 sites, k in {5, 10, 15, 25}):

    plateau (amplitude)        0.213 -> 0.219      essentially invariant
    T50 median                 1.65 -> 2.83 min    shifts ~1.2 min across the whole range
    median |T50(k=10)-T50(k=25)|  0.42 min         per site
    Spearman rho of T50 across conventions  0.87-0.97   ordering preserved
    median SE(T50)             4.48 -> 2.35 min    tightens as k steepens

So: the absolute T50 carries a modelling convention and must be reported with it, but the
*ordering* of sites by T50 — which is what the cascade reconstruction and every WT-vs-mutant
comparison actually use — is robust. `DEFAULT_FIXED_K = 10` is the recommended convention.
Fixing k also *buys* a degree of freedom (3 lack-of-fit df instead of 2), so the
goodness-of-fit test gets stronger, not weaker.

WHAT T50 MEANS HERE — TWO DIFFERENT NUMBERS
-------------------------------------------
`x50` is the **inflection point** of the logistic. Because of the anchoring offset it is
not exactly the time at which the curve reaches half of its final value. The requested
quantity is the latter, so both are returned:

    plateau     the final level the site reaches, signed:  A * (1 - sigma(-k*x50))
    t50_min     back-transformed inflection:               10**x50 - 1
    t_half_min  time to reach 0.5 * plateau  <-- the headline T50

`t_half_min` has a closed form (see `half_response_time`), so no grid search is involved.
For steep early sites the two agree to within a fraction of a minute; the table carries
both so the difference is always auditable.

HOW FAR T50 CAN BE PUSHED
-------------------------
Not far, per site. At this noise level (sigma ~ 0.184) the asymptotic SE of T50 is ~2.55 min
for a true 8 min, against a biological IQR of ~3.1 min, and it *grows* with T50 (the delta
method carries a factor 10**x50). Sites below 2 min are **left-censored** — the first
post-stimulation sample is at 2 min — and must be reported as "< 2 min", never as a number.
T50 is a **set-level** statistic: "substrates of kinase K are delayed by 1.8 +/- 0.4 min" is
defensible, "site X is delayed" is not.

USE THE SHAPE GATE
------------------
A logistic is strictly monotone and has no interior extremum, so it cannot represent a
transient response — and ~70% of EGF sites are transient or biphasic. Fitting them anyway
does not fail: it returns a plausible, wrong T50. Always fit through the
`sigmoid_legitimate` mask from `src.response_shapes.classify_response_shape`, and use
`fit_quality_gates()` on top. `corruption_check()` exists to quantify what the gate is
buying on your data.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from scipy.special import expit, logit
from scipy.stats import chi2 as chi2_dist


# Parameter order used everywhere in this module: (A, k, x50).
PARAM_NAMES = ["A", "k", "x50",]

# The soft-anchored model adds the baseline in front. This is the canonical ordering of the
# *full* parameter vector; the free subset is this list filtered (y0 dropped when the anchor is
# hard, k dropped when it is fixed), which keeps the covariance matrix and the delta-method
# gradient in a known order in every mode.
PARAM_NAMES_SOFT = ["y0", "A", "k", "x50",]

# Bounds. A is a log2 fold change (|A| > 8 is not a phosphosite response, it is a fit running
# away); x50 = log10(t+1), so 2.5 is ~315 min (well past the observation window) and -0.5 allows
# an inflection nominally before t = 0 without letting the curve bend arbitrarily.
#
# The upper bound on k matters more than it looks. The whole observation window spans only
# x = 0 to 1.96 on the log10(t+1) axis, and a logistic covers its 10-90% rise in about 4.4/k.
# At k = 25 that rise takes Delta_x = 0.18 — already faster than the gap between the first two
# samples, i.e. a switch as sharp as this design can resolve. Beyond that the likelihood is
# flat: the data cannot distinguish k = 30 from k = 300, both are step functions through the
# same points. Leaving the bound wide does not buy accuracy, it buys a slow grind up a ridge
# and a meaningless "steepness" estimate. Fitted values for real responses sit around k = 9-14.
DEFAULT_BOUNDS = ((-8.0, 0.05, -0.5,),
                  (8.0, 25.0, 2.5,),)

# k within this factor of its upper bound is reported as unidentified rather than estimated.
K_BOUND_TOLERANCE = 0.98

# Recommended fixed steepness for the 2-parameter mode (see the "STEEPNESS" note in the module
# docstring). Chosen as a convention, not estimated: it sits in the k = 9-14 range that
# corresponds to a physically plausible transition on the log-time axis, it keeps the fitted
# curve a visible sigmoid rather than a step, and T50 rank ordering is stable around it.
DEFAULT_FIXED_K = 10.0


# =============================================================================
# 1. The model
# =============================================================================

def anchored_sigmoid(x: np.ndarray,
                     A: float,
                     k: float,
                     x50: float,) -> np.ndarray:
    """
    Three-parameter anchored logistic, satisfying y(0) = 0 exactly.

    Args:
        x: transformed time axis, normally log10(t + 1) with t in minutes.
        A: amplitude scale of the logistic (see `plateau_value` for the level actually reached).
        k: steepness of the transition on the x axis.
        x50: inflection point on the x axis.

    Returns:
        Array of model values, same shape as x.
    """
    return A * (expit(k * (x - x50)) - expit(-k * x50))


def soft_anchored_sigmoid(x: np.ndarray,
                          y0: float,
                          A: float,
                          k: float,
                          x50: float,) -> np.ndarray:
    """
    The anchored logistic plus a free vertical offset: y(x) = y0 + anchored_sigmoid(x, A, k, x50).

    Literally the existing three-parameter model with a fitted baseline in front. The bracketed
    term is unchanged, so y(0) = y0 exactly and A remains the amplitude — every derived quantity
    (`plateau_value`, `half_response_time`) applies verbatim, measured relative to y0.

    This is the model to use on the `log2:mean` scale, where the starve level is a measurement
    with its own error rather than a structural zero. `y0 = 0` reduces it to `anchored_sigmoid`.

    Args:
        x: transformed time axis, normally log10(t + 1) with t in minutes.
        y0: baseline level the curve starts from, in the units of the fitted profile.
        A: amplitude scale of the logistic, relative to y0.
        k: steepness of the transition on the x axis.
        x50: inflection point on the x axis.

    Returns:
        Array of model values, same shape as x.
    """
    return y0 + anchored_sigmoid(x, A, k, x50,)


def plateau_value(A: float,
                  k: float,
                  x50: float,) -> float:
    """
    Final level the anchored sigmoid approaches as t -> infinity ("the final reference").

    Signed: positive for an induced site, negative for a suppressed one.

    Under the soft anchor this is the level reached **relative to the fitted baseline y0**, not
    an absolute intensity — y0 does not enter, so the number stays a fold change and is directly
    comparable with a hard-anchored fit on log2:FC.

    Args:
        A: amplitude parameter.
        k: steepness parameter.
        x50: inflection point on the x axis.

    Returns:
        The asymptotic value of the fitted curve, in log2 fold-change units.
    """
    return A * (1.0 - expit(-k * x50))


def half_response_time(k: float,
                       x50: float,) -> float:
    """
    Time at which the anchored sigmoid reaches half of its final level.

    Solving y(x) = 0.5 * plateau gives, with c = sigma(-k * x50):

        sigma(k * (x - x50)) = 0.5 * (1 + c)
        x_half = x50 + logit(0.5 * (1 + c)) / k
        t_half = 10**x_half - 1

    The amplitude A cancels, so the answer is a pure timing quantity — it is identical for an
    up- and a down-regulated site of the same kinetics. Note x_half >= x50 whenever x50 > 0:
    the anchoring offset pushes the half-response slightly later than the inflection.

    Args:
        k: steepness parameter.
        x50: inflection point on the x axis.

    Returns:
        Half-response time in minutes (may be negative if the fitted inflection sits before
        the first sample; such fits are flagged as censored downstream).
    """
    c = expit(-k * x50)
    x_half = x50 + logit(0.5 * (1.0 + c)) / k
    return float(10.0 ** x_half - 1.0)


def leverage(times: np.ndarray,
             log_time: bool = True,) -> np.ndarray:
    """
    Diagonal of the hat matrix for a straight-line fit on the given time axis.

    h_ii measures how much observation i determines its own fitted value; h_ii -> 1 means
    that point is fitted exactly regardless of the others, i.e. the fit is a hostage to one
    measurement. This is the concrete argument for log-time: on the hme1_2 design the 90-min
    point carries leverage 0.979 in linear time and 0.659 in log10(t+1).

    Computed for the linear model because leverage is a property of the *design*, not of the
    non-linear model fitted on it; it is the standard diagnostic and transfers directly.

    Args:
        times: (T,) minute axis.
        log_time: use log10(t + 1) instead of raw minutes.

    Returns:
        (T,) array of leverages, summing to 2 (the number of linear parameters).
    """
    x = np.log10(times + 1.0) if log_time else np.asarray(times, dtype=float,)
    X = np.column_stack([np.ones_like(x), x,],)
    H = X @ np.linalg.pinv(X.T @ X) @ X.T
    return np.diag(H)


# =============================================================================
# 2. Fitting one site
# =============================================================================

def _initial_guesses(y: np.ndarray,
                     x: np.ndarray,
                     n_starts: int,
                     bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
                     rng: np.random.Generator,) -> List[Tuple[float, float, float]]:
    """
    Build the multi-start list: one data-driven seed plus random points in the bounded box.

    Args:
        y: observed profile (finite values only).
        x: matching transformed time axis.
        n_starts: total number of starts requested.
        bounds: (lower, upper) parameter bounds.
        rng: random generator.

    Returns:
        List of (A, k, x50) starting tuples, the first of which is the descriptor seed.
    """
    lo, hi = np.array(bounds[0]), np.array(bounds[1])

    A0 = y[-1] if np.isfinite(y[-1]) and y[-1] != 0 else y[np.argmax(np.abs(y))]
    A0 = float(np.clip(A0, lo[0], hi[0],))
    # First x where the profile passes half of its final value.
    half = 0.5 * A0
    crossed = np.where(np.abs(y) >= abs(half))[0]
    x50_0 = float(x[crossed[0]]) if crossed.size else float(np.median(x))
    x50_0 = float(np.clip(x50_0, lo[2], hi[2],))
    starts = [(A0, 5.0, x50_0,)]

    for _ in range(max(0, n_starts - 1,)):
        starts.append(tuple(rng.uniform(lo, hi,)))
    return starts


def _baseline_seed(y: np.ndarray,
                   x: np.ndarray,) -> float:
    """
    Data-driven starting value for the free baseline y0.

    The observed level at x = 0 (the starve measurement) is the obvious seed and is what the
    parameter means. If that point is missing the earliest finite observation is used instead.

    Args:
        y: observed profile (finite values only).
        x: matching transformed time axis.

    Returns:
        Starting value for y0.
    """
    at_zero = np.isclose(x, 0.0,)
    if at_zero.any():
        return float(y[at_zero][0])
    return float(y[np.argmin(x)])


def fit_sigmoid_site(y: np.ndarray,
                     x: np.ndarray,
                     sem: Optional[np.ndarray] = None,
                     n_starts: int = 10,
                     bounds: Tuple = DEFAULT_BOUNDS,
                     loss: str = "linear",
                     fixed_k: Optional[float] = None,
                     free_baseline: bool = False,
                     y0_bounds: Optional[Tuple[float, float]] = None,
                     random_state: int = 0,) -> Dict[str, float]:
    """
    Fit the anchored sigmoid to one site's temporal profile, hard- or soft-anchored.

    Weighted least squares with weights 1/SEM, so precisely measured timepoints pull harder —
    the cleanest single improvement available given that the per-timepoint SD varies roughly
    two-fold across the series.

    TWO MODES, AND THEY EXPECT DIFFERENT INPUT (see the module docstring):

        free_baseline=False   y == log2:FC profile,   sem from per_site_sem(target="FC")
                              y0 is held at 0, the t = 0 point is dropped from the residual
                              vector (the model satisfies it identically, so counting it as an
                              observation would inflate the degrees of freedom).

        free_baseline=True    y == log2:mean profile, sem from per_site_sem(target="mean")
                              y0 is fitted, the t = 0 point is a real observation and is kept.
                              One extra observation, one extra parameter: identical df.

    Passing a log2:FC profile with free_baseline=True is not an error — it just fits a baseline
    to a column that is identically zero, wasting a parameter. Passing a log2:mean profile with
    free_baseline=False *is* effectively an error: it forces the curve through zero at t = 0 when
    the data sit around 18 log2 units.

    Standard errors are asymptotic, from the Gauss-Newton approximation
    cov = s^2 * (J^T J)^-1 with s^2 = 2 * cost / dof. They are optimistic near a parameter
    bound and on a flat likelihood ridge, which is what `bootstrap_sigmoid` checks.

    `loss` defaults to 'linear' (ordinary weighted least squares) rather than a robust loss.
    With only five informative points, down-weighting one of them as an "outlier" discards a
    fifth of the information about the shape; misfit is better *detected* by the lack-of-fit
    test and excluded by `fit_quality_gates` than silently absorbed. Robust losses
    ('soft_l1', 'huber') remain available for sensitivity checks.

    Args:
        y: (T,) observed profile. On the FC scale this includes the structural zero at t = 0;
            on the mean scale t = 0 is the measured starve level.
        x: (T,) transformed time axis matching y (log10(t+1) by convention).
        sem: (T,) standard errors for weighting; unweighted if None. Must come from the branch
            of `per_site_sem` matching `free_baseline` (target='FC' / target='mean').
        n_starts: number of optimiser starts (1 descriptor seed + n-1 random).
        bounds: (lower, upper) tuples of (A, k, x50) bounds. The baseline has its own bounds,
            see `y0_bounds` — `bounds` keeps its three-parameter shape in both modes.
        loss: scipy least_squares loss function.
        fixed_k: hold the steepness at this value and fit only the remaining parameters. Use the
            global value from `profile_global_k`. This removes one free parameter and buys one
            lack-of-fit df, and is the recommended mode on a 5-point design where k is not
            identifiable per site.
        free_baseline: fit the vertical offset y0 instead of anchoring it at 0, and keep the
            t = 0 observation in the residual vector. Use with a `log2:mean` profile.
        y0_bounds: (lower, upper) bounds on the baseline. Default None derives them from the
            data as (min(y) - 10, max(y) + 10): a box 1000-fold wide in either direction, which
            cannot bind on any real profile, and which adapts automatically to whatever scale
            the input is on (log2 intensities ~18, or fold changes ~0) rather than hard-coding
            an assumption about it.
        random_state: seed for the random starts.

    Returns:
        Dict with the fitted parameters (`y0`, A, k, x50) and their SEs, the derived `plateau`,
        `t50_min`, `t_half_min` and `se_t_half`, fit statistics (`rss`, `rmse`, `chi2`,
        `df_lof`, `p_lof`, `n_points`), and diagnostics (`converged`, `n_starts_at_optimum`,
        `free_baseline`). With free_baseline=False, `y0` is 0.0 and `se_y0` is NaN — it is a
        structural constant, not an estimate. All-NaN with converged=False if the site cannot
        be fitted.
    """
    failure = {**{p: np.nan for p in PARAM_NAMES_SOFT},
               **{f"se_{p}": np.nan for p in PARAM_NAMES_SOFT},
               "plateau": np.nan,
               "t50_min": np.nan,
               "t_half_min": np.nan,
               "se_t_half": np.nan,
               "rss": np.nan,
               "rmse": np.nan,
               "chi2": np.nan,
               "df_lof": np.nan,
               "p_lof": np.nan,
               "n_points": 0,
               "converged": False,
               "n_starts_at_optimum": 0,
               "free_baseline": bool(free_baseline),}

    ok = np.isfinite(y) & np.isfinite(x)
    y_fit, x_fit = y[ok], x[ok]
    if sem is not None:
        w = np.asarray(sem, dtype=float,)[ok]
        # Sanitise the SEMs before inverting them. Two things land here: the structural zero the
        # FC branch writes at starve (which would divide to infinity) and genuinely missing SDs
        # (too few replicates). Both become NaN weights and are dropped by `use` below. This
        # guard and the `use` mask are deliberately independent — the second is not redundant,
        # it is what removes the non-finite weights this one creates.
        w = np.where(np.isfinite(w) & (w > 0), w, np.nan,)
        weights = 1.0 / w
    else:
        weights = np.ones_like(y_fit,)

    if free_baseline:
        # On the mean scale t = 0 is the measured starve level, not a structural zero: it stays
        # in the residual vector, where it constrains y0 with weight 1/sqrt(var/n). This is the
        # observation that pays for the extra parameter — the lack-of-fit df is unchanged.
        use = np.isfinite(weights)
    else:
        # Drop the anchored t = 0 point from the residual vector: the model satisfies it
        # identically, it contributes exactly zero residual, and counting it as an observation
        # would inflate the degrees of freedom. It is a constraint, not a measurement.
        use = (x_fit != 0) & np.isfinite(weights)
    y_fit, x_fit, weights = y_fit[use], x_fit[use], weights[use]

    # The free parameters are the canonical order with y0 dropped when the anchor is hard and k
    # dropped when the steepness is supplied. Keeping one ordering for all four modes is what
    # lets the covariance matrix, the SE map and the delta-method gradient below stay aligned
    # without a special case each.
    free_names = [name for name in PARAM_NAMES_SOFT
                  if (name != "y0" or free_baseline)
                  and (name != "k" or fixed_k is None)]
    n_free = len(free_names)

    n_points = y_fit.size
    if n_points <= n_free:
        return {**failure, "n_points": int(n_points)}

    # Baseline bounds: derived from the data unless given, so the same call works on a log2:mean
    # profile (~18) and a log2:FC profile (~0) without assuming which one it received.
    if y0_bounds is None:
        y0_lo, y0_hi = float(np.min(y_fit) - 10.0), float(np.max(y_fit) + 10.0)
    else:
        y0_lo, y0_hi = float(y0_bounds[0]), float(y0_bounds[1])

    lower = {"y0": y0_lo, "A": bounds[0][0], "k": bounds[0][1], "x50": bounds[0][2],}
    upper = {"y0": y0_hi, "A": bounds[1][0], "k": bounds[1][1], "x50": bounds[1][2],}
    bounds_used = (tuple(lower[name] for name in free_names),
                   tuple(upper[name] for name in free_names),)

    def _expand(p: np.ndarray,) -> Tuple[float, float, float, float]:
        """
        Map the free parameter vector onto the full (y0, A, k, x50) quadruple.

        Args:
            p: free parameter vector, ordered as `free_names`.

        Returns:
            Tuple (y0, A, k, x50), filling in y0 = 0 under a hard anchor and k = fixed_k when
            the steepness was supplied.
        """
        vals = dict(zip(free_names, p,))
        return (float(vals.get("y0", 0.0,)),
                float(vals["A"]),
                float(vals.get("k", fixed_k,)),
                float(vals["x50"]),)

    def residuals(p: np.ndarray,) -> np.ndarray:
        """
        Weighted residual vector for the optimiser.

        Args:
            p: free parameter vector.

        Returns:
            (n_points,) array of weighted residuals.
        """
        return (y_fit - soft_anchored_sigmoid(x_fit, *_expand(p,),)) * weights

    rng = np.random.default_rng(random_state,)
    # Seed the baseline from the observed starve level and build the shape seeds on the
    # baseline-subtracted profile, so the existing amplitude/crossing heuristics keep working
    # unchanged on data that sits at 18 log2 units rather than around 0.
    y0_seed = _baseline_seed(y_fit, x_fit,) if free_baseline else 0.0
    starts_full = _initial_guesses(y_fit - y0_seed, x_fit, n_starts, bounds, rng,)

    starts = []
    for n, (A0, k0, x50_0,) in enumerate(starts_full):
        # The descriptor start keeps the observed baseline exactly; the random restarts jitter
        # it, so a bad starve measurement cannot pin every start to the same wrong y0.
        y0_0 = y0_seed if n == 0 else y0_seed + rng.normal(0.0, 0.5,)
        vals = {"y0": float(np.clip(y0_0, y0_lo, y0_hi,)),
                "A": A0,
                "k": k0,
                "x50": x50_0,}
        starts.append(tuple(vals[name] for name in free_names))

    best = None
    costs = []
    for p0 in starts:
        try:
            # x_scale='jac' rescales the free parameters, which sit on very different scales
            # (y0 ~ 18, A ~ 1, k ~ 10, x50 ~ 0.5) — the baseline makes this more necessary, not
            # less. max_nfev is deliberately modest: with 2-4 parameters and 5-6 points a genuine
            # optimum is reached in a few dozen evaluations, so a run that needs hundreds is
            # climbing a flat ridge, not converging.
            res = least_squares(residuals,
                                x0=np.clip(p0, bounds_used[0], bounds_used[1],),
                                bounds=bounds_used,
                                loss=loss,
                                x_scale="jac",
                                max_nfev=400,)
        except (ValueError, RuntimeError):
            continue
        costs.append(res.cost)
        if best is None or res.cost < best.cost:
            best = res

    if best is None:
        return failure

    # How many starts reached (within 1%) the best solution — a per-site identifiability read.
    costs = np.array(costs,)
    n_at_optimum = int(np.sum(costs <= best.cost * 1.01 + 1e-12,))

    y0, A, k, x50 = _expand(best.x,)
    dof = n_points - n_free

    # Asymptotic covariance from the Jacobian at the optimum.
    se = np.full(n_free, np.nan,)
    cov = None
    try:
        JtJ = best.jac.T @ best.jac
        cov_unit = np.linalg.pinv(JtJ,)
        s2 = 2.0 * best.cost / dof if dof > 0 else np.nan
        cov = cov_unit * s2
        se = np.sqrt(np.clip(np.diag(cov), 0, np.inf,))
    except np.linalg.LinAlgError:
        pass

    resid_raw = y_fit - soft_anchored_sigmoid(x_fit, y0, A, k, x50,)
    rss = float(np.sum(resid_raw ** 2,))
    rmse = float(np.sqrt(rss / n_points,))
    chi2_val = float(np.sum((resid_raw * weights) ** 2,))
    p_lof = float(chi2_dist.sf(chi2_val, dof,)) if dof > 0 else np.nan

    t_half = half_response_time(k, x50,)

    # SE of t_half by the delta method, with the gradient taken numerically (the closed form
    # in k and x50 is messy and finite differences are exact enough at this precision).
    # The gradient is taken over the FREE parameters only — with k fixed it contributes no
    # variance, which is precisely why fixing it tightens SE(T50) so much. Both A and y0 get a
    # zero entry: t_half is a pure timing quantity, so neither the amplitude nor the baseline
    # contributes any variance to it, however uncertain they are.
    se_t_half = np.nan
    if cov is not None and np.all(np.isfinite(cov)):
        eps = 1e-6
        d_dk = (half_response_time(k + eps, x50,) - half_response_time(k - eps, x50,)) / (2 * eps)
        d_dx50 = (half_response_time(k, x50 + eps,) - half_response_time(k, x50 - eps,)) / (2 * eps)
        gradients = {"y0": 0.0, "A": 0.0, "k": d_dk, "x50": d_dx50,}
        grad = np.array([gradients[name] for name in free_names],)
        var = float(grad @ cov @ grad)
        se_t_half = float(np.sqrt(var)) if var >= 0 else np.nan

    se_map = dict(zip(free_names, se,))

    return {"y0": float(y0),
            "A": float(A),
            "k": float(k),
            "x50": float(x50),
            "se_y0": float(se_map.get("y0", np.nan,)),
            "se_A": float(se_map.get("A", np.nan,)),
            "se_k": float(se_map.get("k", np.nan,)),
            "se_x50": float(se_map.get("x50", np.nan,)),
            "plateau": float(plateau_value(A, k, x50,)),
            "t50_min": float(10.0 ** x50 - 1.0),
            "t_half_min": t_half,
            "se_t_half": se_t_half,
            "rss": rss,
            "rmse": rmse,
            "chi2": chi2_val,
            "df_lof": int(dof),
            "p_lof": p_lof,
            "n_points": int(n_points),
            "converged": bool(best.success),
            "n_starts_at_optimum": n_at_optimum,
            "free_baseline": bool(free_baseline),}


# =============================================================================
# 3. Fitting a dataset
# =============================================================================

def fit_sigmoid_dataset(values: np.ndarray,
                        times: np.ndarray,
                        sem: Optional[np.ndarray] = None,
                        mask: Optional[np.ndarray] = None,
                        log_time: bool = True,
                        n_starts: int = 10,
                        bounds: Tuple = DEFAULT_BOUNDS,
                        loss: str = "linear",
                        fixed_k: Optional[float] = None,
                        free_baseline: bool = False,
                        y0_bounds: Optional[Tuple[float, float]] = None,
                        index: Optional[pd.Index] = None,
                        n_jobs: int = 1,
                        progress: bool = True,) -> pd.DataFrame:
    """
    Fit the anchored sigmoid to every selected site.

    ⚠️ `values` and `sem` must be on the scale `free_baseline` expects: log2:FC with
    per_site_sem(target="FC") when False, log2:mean with per_site_sem(target="mean") when True.
    Nothing checks this — a mean-scale profile fitted with a hard anchor produces a converged,
    meaningless fit.

    Args:
        values: (n_sites, T) profile matrix from build_profile_matrix.
        times: (T,) minute axis.
        sem: (n_sites, T) standard errors, or None for an unweighted fit.
        mask: boolean (n_sites,) selecting which sites to fit — normally the
            `sigmoid_legitimate` flag. Unselected rows come back as NaN.
        log_time: fit on log10(t+1) (recommended) instead of raw minutes.
        n_starts: optimiser starts per site.
        bounds: parameter bounds passed through to fit_sigmoid_site.
        loss: scipy least_squares loss.
        fixed_k: hold the steepness at this value for every site (see profile_global_k).
        free_baseline: fit the vertical offset y0 (soft anchor) — the recommended mode, on a
            log2:mean profile.
        y0_bounds: bounds on the baseline; None derives them per site from the data.
        index: pandas index to attach to the result.
        n_jobs: parallel workers (joblib); 1 runs in-process.
        progress: show a tqdm progress bar.

    Returns:
        DataFrame with one row per site of `values` (NaN for unselected sites), carrying the
        columns documented in fit_sigmoid_site plus `log_time`.
    """
    n_sites = values.shape[0]
    if mask is None:
        mask = np.ones(n_sites, dtype=bool,)
    mask = np.asarray(mask, dtype=bool,)

    x = np.log10(times + 1.0) if log_time else np.asarray(times, dtype=float,)
    targets = np.where(mask)[0]

    def _one(i: int,) -> Dict[str, float]:
        """
        Fit a single row index.

        Args:
            i: row position in `values`.

        Returns:
            The fit dict from fit_sigmoid_site.
        """
        return fit_sigmoid_site(values[i],
                                x,
                                sem[i] if sem is not None else None,
                                n_starts=n_starts,
                                bounds=bounds,
                                loss=loss,
                                fixed_k=fixed_k,
                                free_baseline=free_baseline,
                                y0_bounds=y0_bounds,
                                random_state=i,)

    iterator = targets
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(targets, desc="sigmoid fits", unit="site",)
        except ImportError:
            pass

    if n_jobs != 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs,)(delayed(_one)(i,) for i in iterator)
    else:
        results = [_one(i,) for i in iterator]

    # Unselected sites keep an all-NaN template row, so the result always aligns row-for-row
    # with `values` and can be concatenated onto the source DataFrame without reindexing.
    template = fit_sigmoid_site(np.array([np.nan]),
                                np.array([0.0]),
                                free_baseline=free_baseline,)
    rows = [dict(template) for _ in range(n_sites)]
    for pos, res in zip(targets, results,):
        rows[pos] = res

    out = pd.DataFrame(rows,
                       index=index if index is not None else np.arange(n_sites,),)
    out["log_time"] = log_time
    return out


def profile_global_k(values: np.ndarray,
                     times: np.ndarray,
                     sem: np.ndarray,
                     mask: np.ndarray,
                     k_grid: Optional[np.ndarray] = None,
                     n_sample: int = 800,
                     log_time: bool = True,
                     random_state: int = 0,
                     **fit_kwargs,) -> pd.DataFrame:
    """
    Estimate one steepness parameter shared by all sites, by profiling the total fit.

    WHY THIS EXISTS. On a 5-point design the per-site steepness is not identifiable: the rise
    happens *between* two samples, so the data cannot distinguish a steep sigmoid from a step
    and the optimiser drives k to whatever bound it is given (measured on hme1_2 EGF: over 90%
    of sites pin k at the bound, and the resulting SE(T50) is meaningless). This is the same
    pathology `council_note_3_temporal_curve_modelling.md` measured for the impulse model's
    beta, and its recommended remedy is the same: **fit the shared shape parameter globally,
    then fix it**, leaving each site with the two parameters the data can actually determine —
    amplitude and timing.

    Method: for each candidate k, refit every sampled site with only (A, x50) free and sum the
    weighted residuals. The minimum of that profile is the global k. This is a profile
    likelihood over the shared parameter, not an average of unstable per-site estimates — it
    never asks any single site to determine k.

    Fixing k also buys a degree of freedom: 2 free parameters instead of 3 means 3 lack-of-fit
    df instead of 2, so the goodness-of-fit test gets stronger, not weaker.

    ⚠️ On hme1_2 this profile has **no interior minimum** — total chi-square falls monotonically
    to the upper bound, i.e. the data prefer a step function. When that happens there is no
    data-driven k to adopt: pick one by convention (`DEFAULT_FIXED_K`) and justify it with
    `t50_sensitivity_to_k`, rather than reporting the bound as an estimate. Always look at the
    profile before taking its argmin.

    Args:
        values: (n_sites, T) profile matrix.
        times: (T,) minute axis.
        sem: (n_sites, T) standard errors.
        mask: boolean selection of sites to profile over (normally sigmoid_legitimate).
        k_grid: candidate k values; default is 14 points log-spaced over [1, 25].
        n_sample: how many sites to sample for the profile.
        log_time: must match the axis used for the final fits.
        random_state: sampling seed.
        **fit_kwargs: forwarded to fit_sigmoid_dataset — pass `free_baseline=True` here when
            working on the log2:mean scale, so the sampled refits match the main ones.

    Returns:
        DataFrame indexed by k with total_chi2, median_rmse and the number of sites whose
        lack-of-fit test passes. The argmin is the global k **only if the profile turns over**;
        check for an interior minimum before using it.
    """
    if k_grid is None:
        k_grid = np.geomspace(1.0, DEFAULT_BOUNDS[1][1], 14,)

    rng = np.random.default_rng(random_state,)
    pos = np.where(np.asarray(mask, dtype=bool,))[0]
    pick = rng.choice(pos, size=min(n_sample, pos.size), replace=False,)
    sub_mask = np.zeros(values.shape[0], dtype=bool,)
    sub_mask[pick] = True

    rows = {}
    for k in k_grid:
        fits = fit_sigmoid_dataset(values,
                                   times,
                                   sem,
                                   mask     = sub_mask,
                                   log_time = log_time,
                                   fixed_k  = float(k),
                                   progress = False,
                                   **fit_kwargs,)
        sub = fits.iloc[pick]
        rows[float(k)] = {"total_chi2": float(sub["chi2"].sum()),
                          "median_rmse": float(sub["rmse"].median()),
                          "n_lof_ok": int((sub["p_lof"] > 0.05).sum()),
                          "n_sites": int(pick.size),}

    out = pd.DataFrame(rows,).T
    out.index.name = "k"
    return out


def t50_sensitivity_to_k(values: np.ndarray,
                         times: np.ndarray,
                         sem: np.ndarray,
                         mask: np.ndarray,
                         k_values: Sequence[float] = (5.0, 10.0, 15.0, 25.0,),
                         n_sample: int = 800,
                         log_time: bool = True,
                         random_state: int = 0,
                         **fit_kwargs,) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Measure how much the reported T50 depends on the fixed-k convention.

    Because k is not identifiable on this design (see the module docstring), it has to be
    fixed by convention — and a convention is only acceptable if the conclusions drawn from it
    survive a different choice. This refits the same sites at several k and reports both the
    marginal distributions and the **per-site rank correlation**, which is the quantity that
    matters: absolute T50 may shift with the convention, but if the ordering of sites is
    preserved then every comparative claim (this kinase's substrates are earlier than that
    one's; this site is delayed in the mutant) is unaffected.

    Args:
        values: (n_sites, T) profile matrix.
        times: (T,) minute axis.
        sem: (n_sites, T) standard errors.
        mask: boolean selection of sites (normally sigmoid_legitimate).
        k_values: fixed steepness values to compare.
        n_sample: number of sites to sample.
        log_time: must match the axis used for the final fits.
        random_state: sampling seed.
        **fit_kwargs: forwarded to fit_sigmoid_dataset — pass `free_baseline=True` here when
            working on the log2:mean scale, so the sampled refits match the main ones.

    Returns:
        Tuple (summary, rank_correlation):
            summary: one row per k with the median/IQR of T50, median SE, median RMSE, the
                fraction passing the lack-of-fit test, and the median plateau.
            rank_correlation: Spearman correlation matrix of per-site T50 between k choices.
    """
    rng = np.random.default_rng(random_state,)
    pos = np.where(np.asarray(mask, dtype=bool,))[0]
    pick = rng.choice(pos, size=min(n_sample, pos.size), replace=False,)
    sub_mask = np.zeros(values.shape[0], dtype=bool,)
    sub_mask[pick] = True

    rows = {}
    t50_by_k = {}
    for k in k_values:
        fits = fit_sigmoid_dataset(values,
                                   times,
                                   sem,
                                   mask     = sub_mask,
                                   log_time = log_time,
                                   fixed_k  = float(k),
                                   progress = False,
                                   **fit_kwargs,)
        sub = fits.iloc[pick]
        t50_by_k[float(k)] = sub["t_half_min"].to_numpy()
        rows[float(k)] = {"T50_median": float(sub["t_half_min"].median()),
                          "T50_q25": float(sub["t_half_min"].quantile(0.25,)),
                          "T50_q75": float(sub["t_half_min"].quantile(0.75,)),
                          "median_se_T50": float(sub["se_t_half"].median()),
                          "median_rmse": float(sub["rmse"].median()),
                          "pct_lof_ok": 100.0 * float((sub["p_lof"] > 0.05).mean()),
                          "median_plateau": float(sub["plateau"].median()),}

    summary = pd.DataFrame(rows,).T
    summary.index.name = "fixed_k"
    rank_corr = pd.DataFrame(t50_by_k,).corr(method="spearman",)
    return summary, rank_corr


def compare_to_flat(fit_df: pd.DataFrame,
                    values: np.ndarray,
                    times: np.ndarray,
                    sem: Optional[np.ndarray] = None,
                    free_baseline: bool = False,
                    fdr_method: str = "fdr_bh",) -> pd.DataFrame:
    """
    Test each fitted sigmoid against the flat null: "this site does not respond".

    ⚠️ THE NULL DIFFERS BETWEEN THE TWO SCALES, AND SO DOES ITS df.

        free_baseline=False, log2:FC scale — the null is the exact curve **y = 0** with
            **zero** free parameters. The fold change is anchored at the starve baseline, so
            "no response" is a fully specified curve, and chi2_null is computed about zero over
            the post-stimulation points.

        free_baseline=True, log2:mean scale — the null is a **fitted constant** y = y0, i.e.
            **one** free parameter, and chi2_null is the weighted residual sum about the
            weighted mean level over **all** timepoints, starve included. A site sitting flat at
            18.4 log2 units is not responsive; measured about zero it would look overwhelmingly
            so, which is the trap this branch exists to avoid.

    The df of the difference test follows: `n_free_sigmoid - n_free_null`, so 3 or 2 on the FC
    scale and 3 or 2 on the mean scale as well (the sigmoid gains y0 and the null gains y0).
    The BIC comparison charges the null its parameter in the same way.

    The test is a **chi-square difference test**, not an F-test:

        delta_chi2 = chi2_null - chi2_sigmoid   ~   chi2 with (p_sigmoid - p_null) df

    This is the right choice *because the measurement errors are known* — the residuals are
    already standardised by the per-site SEM, so the noise scale does not need estimating from
    the residuals. An F-test would re-estimate it from the 2-3 residual df this design leaves,
    which is so imprecise that it destroys the test: on hme1_2 the sigmoid cuts the median
    chi-square from ~32 to ~8.5, an overwhelming improvement, yet the F-test returns a median p
    of 0.37 and rejects nothing. The chi-square difference test on the same numbers gives
    p ~ 1e-5.

    A site that fails this test has no evidence of a response *of sigmoid shape*, and its T50
    is not interpretable however good the RMSE looks.

    Args:
        fit_df: output of fit_sigmoid_dataset.
        values: the (n_sites, T) profile matrix the fits were computed from.
        times: (T,) minute axis.
        sem: (n_sites, T) errors used for weighting, or None.
        free_baseline: must match the flag the fits were produced with. Selects the null
            described above; getting it wrong makes every `p_vs_flat` wrong.
        fdr_method: multiple-testing method passed to statsmodels multipletests.

    Returns:
        Copy of fit_df with `chi2_null`, `y0_null` (the fitted null level, NaN under the exact
        zero null), `df_vs_flat`, `delta_chi2`, `p_vs_flat`, `fdr_vs_flat` and
        `bic_sigmoid` / `bic_flat` appended.
    """
    from statsmodels.stats.multitest import multipletests

    # Which timepoints the null is evaluated over, matching what the fit used: the soft anchor
    # keeps starve as an observation, the hard anchor drops it.
    keep = np.ones(len(times), dtype=bool,) if free_baseline else (times > 0)
    y = values[:, keep]

    if sem is not None:
        s = sem[:, keep]
        # Same sanitisation as the fit: a zero or missing SEM must not become an infinite weight.
        w = np.where(np.isfinite(s) & (s > 0), 1.0 / np.where(s > 0, s, np.nan,), np.nan,)
    else:
        w = np.ones_like(y,)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning,)
        finite = np.isfinite(y) & np.isfinite(w)
        y_ok = np.where(finite, y, 0.0,)
        w2 = np.where(finite, w ** 2, 0.0,)

        if free_baseline:
            # The null is a fitted constant: the weighted mean is its least-squares estimate,
            # and n_free_null = 1.
            denom = w2.sum(axis=1,)
            y0_null = np.where(denom > 0, (w2 * y_ok).sum(axis=1,) / np.where(denom > 0, denom, np.nan,), np.nan,)
            chi2_null = (w2 * (y_ok - y0_null[:, None]) ** 2).sum(axis=1,)
            n_free_null = 1.0
        else:
            y0_null = np.full(y.shape[0], np.nan,)
            chi2_null = (w2 * y_ok ** 2).sum(axis=1,)
            n_free_null = 0.0

    out = fit_df.copy()
    out["chi2_null"] = chi2_null
    out["y0_null"] = y0_null

    n = out["n_points"].to_numpy(dtype=float,)
    chi2_sig = out["chi2"].to_numpy(dtype=float,)
    # Free parameters actually used — recovered from the fit rather than assumed, so the test
    # stays correct across all four (hard/soft anchor) x (free/fixed k) combinations.
    n_free = n - out["df_lof"].to_numpy(dtype=float,)
    df_test = n_free - n_free_null

    with np.errstate(invalid="ignore", divide="ignore",):
        delta_chi2 = chi2_null - chi2_sig
        p = chi2_dist.sf(delta_chi2, df_test,)
        # Gaussian BIC on the weighted residuals; the null is charged its own parameters.
        out["bic_sigmoid"] = n * np.log(chi2_sig / n) + n_free * np.log(n)
        out["bic_flat"] = n * np.log(chi2_null / n) + n_free_null * np.log(n)

    out["df_vs_flat"] = df_test
    out["delta_chi2"] = delta_chi2
    out["p_vs_flat"] = p

    fdr = np.full(len(out), np.nan,)
    valid = np.isfinite(p)
    if valid.any():
        fdr[valid] = multipletests(p[valid], method=fdr_method,)[1]
    out["fdr_vs_flat"] = fdr
    return out


def fit_quality_gates(fit_df: pd.DataFrame,
                      sigma: float,
                      alpha_lof: float = 0.05,
                      alpha_flat: float = 0.05,
                      t_censor_low: float = 2.0,
                      t_censor_high: float = 60.0,
                      max_se_x50: float = 1.0,) -> pd.DataFrame:
    """
    Turn the raw fit statistics into the explicit pass/fail gates a T50 must clear.

    The gates, and why each exists:

        converged            the optimiser returned a solution
        rmse_below_noise     RMSE < sigma — the fit tracks the data to within measurement error
        lof_ok               lack-of-fit p > alpha: the sigmoid shape is not rejected
        responsive           the sigmoid beats the flat null after FDR correction
        identifiable         SE(x50) finite and below max_se_x50, and k not pinned to its
                             upper bound — i.e. the likelihood is not flat in any direction
        t50_censored         T50 below the first sample (2 min) or beyond the useful range
                             (60 min) — report as "< 2 min" / "> 60 min", never as a number

    `fit_ok` is the conjunction of the first five. Censoring is reported *separately*: a
    censored fit is a successful fit whose T50 must be reported as an inequality, not a
    failed one.

    Args:
        fit_df: output of fit_sigmoid_dataset, ideally after compare_to_flat.
        sigma: the measurement noise scale in log2 units (median across-replicate SD).
        alpha_lof: significance threshold for the lack-of-fit test.
        alpha_flat: FDR threshold for the sigmoid-vs-flat test.
        t_censor_low: lower T50 censoring bound in minutes (the first post-stimulation sample).
        t_censor_high: upper T50 censoring bound in minutes.
        max_se_x50: largest acceptable SE on the inflection point, on the log10(t+1) axis.

    Returns:
        Copy of fit_df with the boolean gate columns and `fit_ok` appended.
    """
    out = fit_df.copy()
    out["rmse_below_noise"] = out["rmse"] < sigma
    out["lof_ok"] = out["p_lof"] > alpha_lof
    # k sitting on its upper bound means the likelihood is flat in k: the data cannot tell a
    # steep sigmoid from a step, so the "steepness" is not an estimate. T50 usually survives
    # this (it is the location, not the sharpness), but the site is not fully identified.
    out["k_at_bound"] = out["k"] >= K_BOUND_TOLERANCE * DEFAULT_BOUNDS[1][1]
    out["identifiable"] = (out["se_x50"].notna()
                           & (out["se_x50"] < max_se_x50)
                           & ~out["k_at_bound"])
    if "fdr_vs_flat" in out.columns:
        out["responsive"] = out["fdr_vs_flat"] < alpha_flat
    else:
        out["responsive"] = True
    out["t50_censored"] = (out["t_half_min"] < t_censor_low) | (out["t_half_min"] > t_censor_high)

    out["fit_ok"] = (out["converged"].astype(bool)
                     & out["rmse_below_noise"].fillna(False)
                     & out["lof_ok"].fillna(False)
                     & out["responsive"].fillna(False)
                     & out["identifiable"].fillna(False))
    return out


# =============================================================================
# 4. Diagnostics
# =============================================================================

def corruption_check(values: np.ndarray,
                     times: np.ndarray,
                     sem: np.ndarray,
                     shapes: pd.DataFrame,
                     sigma: float,
                     classes: Sequence[str] = ("transient", "biphasic",),
                     n_sample: int = 500,
                     random_state: int = 0,
                     **fit_kwargs,) -> pd.DataFrame:
    """
    Quantify what the shape gate is actually buying: fit sigmoids to sites it excludes.

    A logistic cannot represent a transient response, but the optimiser does not know that —
    it returns a number anyway. This function fits a sample of the *excluded* classes and
    reports how many would have passed the goodness-of-fit gate regardless. Those are the
    sites that would have entered a T50 table with a plausible, wrong value if the shape gate
    were dropped.

    Args:
        values: (n_sites, T) profile matrix.
        times: (T,) minute axis.
        sem: (n_sites, T) standard errors.
        shapes: output of classify_response_shape.
        sigma: measurement noise scale, for the RMSE gate.
        classes: shape classes to test (the ones normally excluded).
        n_sample: how many sites to sample per class (all of them if fewer exist).
        random_state: sampling seed.
        **fit_kwargs: forwarded to fit_sigmoid_dataset — pass `free_baseline=True` here when
            working on the log2:mean scale, so the sampled refits match the main ones.

    Returns:
        DataFrame indexed by class with n_tested, n_passing_gof, pct_passing_gof and the
        median fitted T50 of the passing sites.
    """
    rng = np.random.default_rng(random_state,)
    rows = {}

    for cls in classes:
        pos = np.where((shapes["shape_class"] == cls).to_numpy())[0]
        if pos.size == 0:
            continue
        pick = rng.choice(pos, size=min(n_sample, pos.size), replace=False,)
        mask = np.zeros(values.shape[0], dtype=bool,)
        mask[pick] = True

        fits = fit_sigmoid_dataset(values,
                                   times,
                                   sem,
                                   mask=mask,
                                   progress=False,
                                   **fit_kwargs,)
        sub = fits.iloc[pick]
        passing = (sub["converged"].astype(bool)
                   & (sub["rmse"] < sigma)
                   & (sub["p_lof"] > 0.05))
        rows[cls] = {"n_tested": int(pick.size),
                     "n_passing_gof": int(passing.sum()),
                     "pct_passing_gof": 100.0 * passing.mean(),
                     "median_t_half_min": float(sub.loc[passing, "t_half_min"].median())
                     if passing.any() else np.nan,}

    return pd.DataFrame(rows,).T


def compare_time_axes(values: np.ndarray,
                      times: np.ndarray,
                      sem: np.ndarray,
                      mask: np.ndarray,
                      n_sample: int = 400,
                      random_state: int = 0,
                      **fit_kwargs,) -> pd.DataFrame:
    """
    Fit the same sites in linear time and in log10(t+1) and compare the parameterisations.

    Reproduces the argument in decision document section 9.3 on the current data: the two
    axes fit about equally well (RMSE is nearly identical), but linear time yields negative,
    physically meaningless T50s and a steepness parameter confounded with timing.

    Args:
        values: (n_sites, T) profile matrix.
        times: (T,) minute axis.
        sem: (n_sites, T) standard errors.
        mask: boolean selection of sites to test (normally sigmoid_legitimate).
        n_sample: number of sites to sample from the mask.
        random_state: sampling seed.
        **fit_kwargs: forwarded to fit_sigmoid_dataset — pass `free_baseline=True` here when
            working on the log2:mean scale, so the sampled refits match the main ones.

    Returns:
        DataFrame with one row per axis: median RMSE, fraction of negative T50s, and the
        interquartile ratio of the fitted k (how much steepness varies across sites).
    """
    rng = np.random.default_rng(random_state,)
    pos = np.where(np.asarray(mask, dtype=bool,))[0]
    pick = rng.choice(pos, size=min(n_sample, pos.size), replace=False,)
    sub_mask = np.zeros(values.shape[0], dtype=bool,)
    sub_mask[pick] = True

    rows = {}
    for name, log_time in (("log10(t+1)", True,), ("linear t", False,),):
        fits = fit_sigmoid_dataset(values,
                                   times,
                                   sem,
                                   mask=sub_mask,
                                   log_time=log_time,
                                   progress=False,
                                   **fit_kwargs,)
        sub = fits.iloc[pick]
        k = sub["k"].dropna()
        rows[name] = {"n": int(pick.size),
                      "median_rmse": float(sub["rmse"].median()),
                      "pct_negative_T50": 100.0 * float((sub["t_half_min"] < 0).mean()),
                      "k_q90_over_q10": float(k.quantile(0.9) / k.quantile(0.1))
                      if len(k) else np.nan,}
    return pd.DataFrame(rows,).T


def bootstrap_sigmoid(values: np.ndarray,
                      times: np.ndarray,
                      sem: np.ndarray,
                      fit_df: pd.DataFrame,
                      site_positions: Sequence[int],
                      n_boot: int = 500,
                      log_time: bool = True,
                      free_baseline: bool = False,
                      random_state: int = 0,
                      **fit_kwargs,) -> pd.DataFrame:
    """
    Residual bootstrap of the sigmoid fit, as a check on the asymptotic standard errors.

    For each selected site the fitted curve is held fixed, residuals are resampled with
    replacement and added back, and the model is refitted. Percentile intervals follow.

    The asymptotic SEs come from a local quadratic approximation to the likelihood, which is
    optimistic near a parameter bound or on a flat ridge. The **ratio** of bootstrap SD to
    asymptotic SE is therefore the useful output: a ratio above ~2 means the site sits on a
    ridge and its parameters should not be used, however tight the reported SE looks.

    A plain resample of 5 residuals is coarse, which is exactly why this is run on a subset
    as a diagnostic rather than used to produce every published interval.

    Args:
        values: (n_sites, T) profile matrix.
        times: (T,) minute axis.
        sem: (n_sites, T) standard errors.
        fit_df: output of fit_sigmoid_dataset (the point fits to bootstrap around).
        site_positions: row positions of the sites to bootstrap.
        n_boot: bootstrap resamples per site.
        log_time: must match the axis the point fits used.
        free_baseline: must match the anchor the point fits used. It decides both which points
            carry a resampled residual (the soft anchor resamples the starve point too, since
            there it is a measurement) and how each resample is refitted.
        random_state: seed.
        **fit_kwargs: forwarded to fit_sigmoid_site.

    Returns:
        DataFrame indexed by the given site positions with the bootstrap median, 2.5/97.5
        percentiles and SD of t_half_min, the asymptotic SE, and their ratio.
    """
    x = np.log10(times + 1.0) if log_time else np.asarray(times, dtype=float,)
    # Which points get a resampled residual: under the hard anchor t = 0 is a structural zero
    # and must stay exactly 0 in every resample; under the soft anchor it is an observation like
    # any other and resampling it is what propagates the baseline uncertainty.
    resample = np.ones(len(times), dtype=bool,) if free_baseline else (times > 0)
    rng = np.random.default_rng(random_state,)
    rows = {}

    for i in site_positions:
        params = fit_df.iloc[i][PARAM_NAMES_SOFT if free_baseline else PARAM_NAMES].to_numpy(dtype=float,)
        if not np.all(np.isfinite(params)):
            continue
        y = values[i]
        fitted = (soft_anchored_sigmoid(x, *params,) if free_baseline
                  else anchored_sigmoid(x, *params,))
        resid = (y - fitted)[resample]
        resid = resid[np.isfinite(resid)]
        if resid.size < 2:
            continue

        draws = []
        for b in range(n_boot):
            y_star = fitted.copy()
            noise = rng.choice(resid, size=int(resample.sum()), replace=True,)
            y_star[resample] = fitted[resample] + noise
            if not free_baseline:
                y_star[~resample] = 0.0
            res = fit_sigmoid_site(y_star,
                                   x,
                                   sem[i],
                                   free_baseline=free_baseline,
                                   random_state=b,
                                   **fit_kwargs,)
            draws.append(res["t_half_min"])

        draws = np.array(draws, dtype=float,)
        draws = draws[np.isfinite(draws)]
        if draws.size == 0:
            continue
        se_asym = float(fit_df.iloc[i]["se_t_half"])
        sd_boot = float(np.std(draws, ddof=1,))
        rows[i] = {"t_half_point": float(fit_df.iloc[i]["t_half_min"]),
                   "boot_median": float(np.median(draws,)),
                   "boot_lo95": float(np.percentile(draws, 2.5,)),
                   "boot_hi95": float(np.percentile(draws, 97.5,)),
                   "boot_sd": sd_boot,
                   "se_asymptotic": se_asym,
                   "ratio_boot_over_asym": sd_boot / se_asym if se_asym and np.isfinite(se_asym) else np.nan,}

    return pd.DataFrame(rows,).T


# =============================================================================
# 5. Figures
# =============================================================================

def plot_leverage(times: np.ndarray,
                  figsize: Tuple[float, float] = (6, 4),):
    """
    Bar chart of per-timepoint leverage in linear versus log time.

    One panel, and it settles the choice of time axis: a bar approaching 1.0 is a timepoint
    that fits itself exactly and lets one noisy measurement dictate the whole curve.

    Args:
        times: (T,) minute axis.
        figsize: figure size in inches.

    Returns:
        Tuple (figure, axes).
    """
    lin = leverage(times, log_time=False,)
    log = leverage(times, log_time=True,)
    labels = [f"{t:g}" for t in times]
    pos = np.arange(len(times),)

    fig, ax = plt.subplots(figsize=figsize,)
    ax.bar(pos - 0.2, lin, width=0.4, label="linear t", color="tab:red",)
    ax.bar(pos + 0.2, log, width=0.4, label="log10(t+1)", color="tab:blue",)
    ax.axhline(1.0, color="grey", ls="--", lw=0.8,)
    ax.set_xticks(pos,)
    ax.set_xticklabels(labels,)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("leverage $h_{ii}$")
    ax.set_title("How much each timepoint determines its own fitted value")
    ax.legend(frameon=False,)
    fig.tight_layout()
    return fig, ax


def plot_fit_examples(values: np.ndarray,
                      times: np.ndarray,
                      fit_df: pd.DataFrame,
                      site_positions: Sequence[int],
                      sem: Optional[np.ndarray] = None,
                      labels: Optional[Sequence[str]] = None,
                      log_time: bool = True,
                      n_cols: int = 4,
                      figsize_per_panel: Tuple[float, float] = (3.0, 2.4),):
    """
    Draw observed profiles with their fitted sigmoid overlaid.

    Args:
        values: (n_sites, T) profile matrix.
        times: (T,) minute axis.
        fit_df: output of fit_sigmoid_dataset.
        site_positions: row positions to draw.
        sem: optional errors, drawn as error bars.
        labels: optional per-site labels for panel titles.
        log_time: must match the axis used for fitting.
        n_cols: panels per row.
        figsize_per_panel: size of each panel in inches.

    Returns:
        Tuple (figure, axes array).
    """
    site_positions = list(site_positions)
    if not site_positions:
        raise ValueError("plot_fit_examples: no sites to plot. If this came from a gate mask, "
                         "the gates rejected every site — inspect the individual gate columns "
                         "(rmse_below_noise / lof_ok / responsive / identifiable) rather than "
                         "assuming a plotting problem.")
    n_rows = int(np.ceil(len(site_positions) / n_cols,))
    fig, axes = plt.subplots(n_rows,
                             n_cols,
                             figsize=(figsize_per_panel[0] * n_cols,
                                      figsize_per_panel[1] * n_rows,),
                             squeeze=False,)

    grid_t = np.linspace(0, times.max(), 400,)
    grid_x = np.log10(grid_t + 1.0) if log_time else grid_t
    obs_x = np.log10(times + 1.0) if log_time else times

    # Soft-anchored fits are drawn on the log2:mean scale, so the y-axis label changes with them.
    soft = ("y0" in fit_df.columns
            and bool(np.any(np.isfinite(fit_df["y0"].to_numpy(dtype=float,))
                            & (fit_df["y0"].to_numpy(dtype=float,) != 0.0))))

    for n, i in enumerate(site_positions):
        ax = axes[n // n_cols, n % n_cols]
        ax.errorbar(obs_x,
                    values[i],
                    yerr=sem[i] if sem is not None else None,
                    fmt="o",
                    ms=4,
                    color="black",
                    capsize=2,
                    elinewidth=1.0,
                    label="observed",)
        row = fit_df.iloc[i]
        # A fit table written before the soft anchor existed has no y0 column; 0.0 then
        # reproduces the hard-anchored curve exactly.
        y0 = float(row["y0"]) if ("y0" in fit_df.columns and np.isfinite(row["y0"])) else 0.0
        if np.isfinite(row["A"]):
            ax.plot(grid_x,
                    soft_anchored_sigmoid(grid_x, y0, row["A"], row["k"], row["x50"],),
                    color="crimson",
                    lw=1.8,
                    label="fit",)
            # The plateau is relative to the baseline, so the level drawn is y0 + plateau.
            ax.axhline(y0 + row["plateau"], color="tab:blue", ls=":", lw=1.0,)
            t_half = row["t_half_min"]
            if np.isfinite(t_half) and t_half >= 0:
                ax.axvline(np.log10(t_half + 1.0) if log_time else t_half,
                           color="tab:green",
                           ls="--",
                           lw=1.0,)
        ax.axhline(y0, color="grey", lw=0.7,)
        ax.set_xticks(obs_x,)
        ax.set_xticklabels([f"{t:g}" for t in times], fontsize=7,)
        name = labels[i] if labels is not None else f"row {i}"
        ax.set_title(f"{str(name)[:20]}\nT50={row['t_half_min']:.1f} min  RMSE={row['rmse']:.2f}",
                     fontsize=8,)

    for n in range(len(site_positions), n_rows * n_cols,):
        axes[n // n_cols, n % n_cols].axis("off")

    fig.supxlabel("time (min)")
    fig.supylabel("log2 fold change vs starve" if not soft else "log2 mean intensity")
    fig.tight_layout()
    return fig, axes


def plot_t50_distribution(fit_df: pd.DataFrame,
                          gate_col: str = "fit_ok",
                          t_censor_low: float = 2.0,
                          bins: int = 60,
                          figsize: Tuple[float, float] = (11, 4),):
    """
    Distribution of fitted T50 and of the plateau amplitude, for the sites that passed the gates.

    The censoring bound is drawn explicitly: everything piled against it is unresolved, not
    fast, because the first post-stimulation sample is at 2 min.

    Args:
        fit_df: output of fit_quality_gates.
        gate_col: boolean column selecting the sites to show.
        t_censor_low: censoring bound to mark, in minutes.
        bins: histogram bins.
        figsize: figure size in inches.

    Returns:
        Tuple (figure, axes array).
    """
    sub = fit_df[fit_df[gate_col].fillna(False).astype(bool)] if gate_col in fit_df else fit_df

    fig, axes = plt.subplots(1, 3, figsize=figsize,)

    t50 = sub["t_half_min"].dropna()
    axes[0].hist(t50.clip(upper=90,), bins=bins, color="tab:blue",)
    axes[0].axvline(t_censor_low, color="crimson", ls="--",
                    label=f"censoring bound ({t_censor_low:g} min)",)
    axes[0].set_xlabel("T50 (min)")
    axes[0].set_ylabel("sites")
    axes[0].set_title(f"T50  (median {t50.median():.1f} min, "
                      f"IQR {t50.quantile(0.25):.1f}-{t50.quantile(0.75):.1f})", fontsize=9,)
    axes[0].legend(frameon=False, fontsize=8,)

    axes[1].hist(sub["plateau"].dropna(), bins=bins, color="tab:green",)
    axes[1].axvline(0, color="grey", lw=0.8,)
    axes[1].set_xlabel("plateau (log2 FC)")
    axes[1].set_title("Final level reached", fontsize=9,)

    axes[2].scatter(sub["t_half_min"], sub["plateau"], s=4, alpha=0.3,)
    axes[2].axhline(0, color="grey", lw=0.8,)
    axes[2].axvline(t_censor_low, color="crimson", ls="--",)
    axes[2].set_xscale("symlog",)
    axes[2].set_xlabel("T50 (min)")
    axes[2].set_ylabel("plateau (log2 FC)")
    axes[2].set_title("Timing vs amplitude", fontsize=9,)

    fig.tight_layout()
    return fig, axes


def plot_gof_summary(fit_df: pd.DataFrame,
                     sigma: float,
                     figsize: Tuple[float, float] = (11, 4),):
    """
    Goodness-of-fit overview: RMSE against the noise scale, lack-of-fit p-values, and
    how many optimiser starts reached the optimum.

    Read the RMSE panel in both directions. Bars to the right of sigma are misfits. Bars far
    to the *left* of sigma are not triumphs — a fit much tighter than the measurement error
    is interpolating noise, which is the signature of too many parameters for the design.

    Args:
        fit_df: output of fit_sigmoid_dataset (after compare_to_flat if available).
        sigma: measurement noise scale in log2 units.
        figsize: figure size in inches.

    Returns:
        Tuple (figure, axes array).
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize,)

    rmse = fit_df["rmse"].dropna()
    axes[0].hist(rmse, bins=60, color="tab:blue",)
    axes[0].axvline(sigma, color="crimson", ls="--", label=f"sigma = {sigma:.3f}",)
    axes[0].axvline(sigma / 3, color="orange", ls=":", label="sigma/3 (interpolation)",)
    axes[0].set_xlabel("RMSE (log2 units)")
    axes[0].set_ylabel("sites")
    axes[0].set_title(f"{100 * (rmse < sigma).mean():.1f}% below noise", fontsize=9,)
    axes[0].legend(frameon=False, fontsize=8,)

    p_lof = fit_df["p_lof"].dropna()
    axes[1].hist(p_lof, bins=20, color="tab:green",)
    axes[1].axvline(0.05, color="crimson", ls="--",)
    axes[1].set_xlabel("lack-of-fit p")
    axes[1].set_title(f"{100 * (p_lof > 0.05).mean():.1f}% not rejected", fontsize=9,)

    starts = fit_df["n_starts_at_optimum"].dropna()
    axes[2].hist(starts, bins=np.arange(starts.max() + 2,) - 0.5, color="tab:purple",)
    axes[2].set_xlabel("starts reaching the optimum")
    axes[2].set_title("Multi-start agreement", fontsize=9,)

    fig.tight_layout()
    return fig, axes
