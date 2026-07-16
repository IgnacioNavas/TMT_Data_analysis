"""
Kinase activity inference (KSEA) from phosphosite fold-changes.

Motivation
──────────
You cannot measure a kinase's activity directly in this data — you measure the
phosphorylation of thousands of sites. KSEA (Kinase-Substrate Enrichment Analysis, Casado
et al. 2013, Sci. Signal.) infers a kinase's activity from the COLLECTIVE behaviour of its
substrates: if a kinase is active, its substrate sites should, on average, go up. Averaging
over many substrates cancels site-specific noise and turns a noisy per-site signal into a
robust per-kinase readout — and, computed at each timepoint, an activity TRAJECTORY.

The KSEA score
──────────────
For a kinase K with substrate set S in one condition/timepoint (one column of log2 fold-
changes vs the starve reference):

    z_K = (mean_S − mean_P) * sqrt(m) / delta

  * mean_S  = mean log2FC of the m substrates of K,
  * mean_P  = mean log2FC of the whole quantified population P (the background),
  * m       = number of substrates of K quantified in this column,
  * delta   = standard deviation of log2FC across the population P.

Read it as a z-score: (mean_S − mean_P) is how far the substrate set sits above/below the
global average; dividing by delta/sqrt(m) — the standard error of a mean of m draws from a
population with spread delta — expresses that shift in standard-error units. So z_K is large
and positive when K's substrates are coordinately UP well beyond what the sample size and
background scatter would produce by chance → inferred activation; large negative → inferred
inhibition. A two-sided normal p-value follows from z, and we Benjamini-Hochberg correct
across kinases within each column.

What the substrate set is here (important caveat to state up front)
──────────────────────────────────────────────────────────────────
Classic KSEA uses curated kinase-substrate relationships (PhosphoSitePlus/NetworKIN). Here
the substrate set for K is defined by the kinase_library MOTIF prediction (K among a site's
top-N predicted kinases, optionally above a percentile threshold). This is a deliberate,
statable choice: the readout is the activity of the *motif-defined* substrate set of K, not
of experimentally validated substrates. Consequences to acknowledge when defending it:
  * Motif match ≠ physical substrate (no localisation/context/scaffold information).
  * Substrate sets of related kinases overlap (shared motifs) → correlated, non-independent
    scores; treat kinases within a family as one signal, not independent evidence.
  * z reflects net phosphorylation change, which also depends on phosphatases and on protein
    abundance — KSEA attributes it to the kinase but cannot prove causation.
Use a stringent substrate set (e.g. min_percentile≈90, min_substrates≈5) to trade a little
sensitivity for much better specificity, and validate against known biology (ERK/RSK up
early after EGF).

References: Casado et al. 2013 (Sci. Signal. 6:rs6); Wiredja, Koyutürk & Chance 2017
(KSEAapp, Bioinformatics) for the z-score implementation; Johnson et al. 2023 (Nature) for
the motif matrices behind the substrate predictions.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from src.cluster_enrichment import kinase_membership


# ---------------------------------------------------------------------------
# 1. Substrate sets
# ---------------------------------------------------------------------------
def kinase_substrate_sets(df,
                          kinases=None,
                          kinase_prefix="predicted_kinase",
                          top_n=5,
                          ranks=None,
                          min_percentile=None,):
    """
    Map each kinase to the row-index of its predicted substrate sites.

    Thin wrapper over cluster_enrichment.kinase_membership() (same substrate definition, so
    enrichment and activity analyses stay consistent).

    Args:
        df: dataframe with predicted-kinase columns.
        kinases: kinase name/list to include; if None, all kinases seen in the columns.
        kinase_prefix: predicted-kinase column prefix (default "predicted_kinase").
        top_n: highest rank to scan when `ranks` is None (default 5).
        ranks: explicit ranks to scan (overrides top_n).
        min_percentile: require the matching *_prob >= this to count a substrate.

    Returns:
        dict {kinase: pandas Index of substrate site rows}.
    """
    membership = kinase_membership(df,
                                   kinases=kinases,
                                   kinase_prefix=kinase_prefix,
                                   top_n=top_n,
                                   ranks=ranks,
                                   min_percentile=min_percentile,)
    return {kinase: membership.index[membership[kinase]] for kinase in membership.columns}


# ---------------------------------------------------------------------------
# 2. KSEA on a single fold-change vector
# ---------------------------------------------------------------------------
def ksea_from_values(fold_changes,
                     substrate_sets,
                     min_substrates=5,):
    """
    Compute KSEA z-scores for one column of log2 fold-changes.

    Args:
        fold_changes: Series of log2FC per site (the population P = its non-null entries).
        substrate_sets: dict {kinase: Index of substrate sites} from kinase_substrate_sets().
        min_substrates: minimum quantified substrates required to score a kinase; kinases
            below this are skipped (a mean of too few sites is not trustworthy). Default 5.

    Returns:
        DataFrame indexed by kinase with columns: n_substrates, mean_fc (mean_S),
        population_mean (mean_P), z_score, p_value. Kinases with fewer than min_substrates
        quantified substrates are omitted.
    """
    population = fold_changes.dropna()
    mean_p = float(population.mean())
    delta = float(population.std(ddof=1))
    pop_index = population.index

    records = []
    for kinase, sites in substrate_sets.items():
        substrate_values = population.reindex(pop_index.intersection(sites)).dropna()
        m = int(substrate_values.shape[0])
        if m < min_substrates:
            continue
        mean_s = float(substrate_values.mean())
        z = (mean_s - mean_p) * np.sqrt(m) / delta if delta > 0 else np.nan
        p = float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
        records.append({"kinase": kinase,
                        "n_substrates": m,
                        "mean_fc": mean_s,
                        "population_mean": mean_p,
                        "z_score": z,
                        "p_value": p,})

    return pd.DataFrame(records).set_index("kinase") if records else pd.DataFrame(
        columns=["n_substrates", "mean_fc", "population_mean", "z_score", "p_value"])


# ---------------------------------------------------------------------------
# 3. Activity trajectories across conditions x timepoints
# ---------------------------------------------------------------------------
def kinase_activity_profile(df,
                            conditions,
                            timepoints,
                            cell_line="WT",
                            data_type="log2:FC",
                            kinases=None,
                            kinase_prefix="predicted_kinase",
                            top_n=5,
                            ranks=None,
                            min_percentile=90,
                            min_substrates=5,
                            fdr_method="fdr_bh",):
    """
    Compute a KSEA activity trajectory for each kinase across conditions and timepoints.

    Substrate sets are built once from the predictions, then KSEA is run on each
    {cell_line}_{data_type}_{condition}_{timepoint} fold-change column. FDR correction is
    applied across kinases WITHIN each (condition, timepoint) column.

    Args:
        df: dataframe with predicted-kinase columns and log2FC columns.
        conditions: bare condition tokens present in the column names, e.g.
            ["EGF", "INS", "EGFnINS"].
        timepoints: ordered timepoint tokens to profile, e.g. ["2", "5", "10", "15", "90"]
            (omit "full"/"starve": starve is the reference so its FC is 0 by construction).
        cell_line: cell-line token in the column names (default "WT").
        data_type: fold-change data type token (default "log2:FC").
        kinases: kinases to profile; if None, all predicted kinases.
        kinase_prefix: predicted-kinase column prefix (default "predicted_kinase").
        top_n / ranks / min_percentile: substrate-set definition (default: top 5 with
            percentile >= 90 — a stringent, specificity-favouring set for activity inference).
        min_substrates: minimum quantified substrates to score a kinase (default 5).
        fdr_method: multipletests method applied per column (default BH).

    Returns:
        Tidy long DataFrame with columns: kinase, condition, timepoint, n_substrates,
        mean_fc, z_score, p_value, q_value. `timepoint` is an ordered categorical so it plots
        in the supplied order.
    """
    substrate_sets = kinase_substrate_sets(df,
                                           kinases=kinases,
                                           kinase_prefix=kinase_prefix,
                                           top_n=top_n,
                                           ranks=ranks,
                                           min_percentile=min_percentile,)
    frames = []
    for condition in conditions:
        for timepoint in timepoints:
            column = f"{cell_line}_{data_type}_{condition}_{timepoint}"
            if column not in df.columns:
                continue
            scores = ksea_from_values(df[column],
                                      substrate_sets,
                                      min_substrates=min_substrates,)
            if scores.empty:
                continue
            scores = scores.reset_index()
            scores["condition"] = condition
            scores["timepoint"] = timepoint
            scores["q_value"] = multipletests(scores["p_value"], method=fdr_method)[1]
            frames.append(scores)

    if not frames:
        return pd.DataFrame(columns=["kinase", "condition", "timepoint", "n_substrates",
                                     "mean_fc", "z_score", "p_value", "q_value"])
    profile = pd.concat(frames, ignore_index=True,)
    profile["timepoint"] = pd.Categorical(profile["timepoint"],
                                          categories=list(timepoints),
                                          ordered=True,)
    return profile[["kinase", "condition", "timepoint", "n_substrates",
                    "mean_fc", "z_score", "p_value", "q_value"]]


# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------
def plot_activity_profiles(profile_df,
                           kinases=None,
                           conditions=None,
                           value="z_score",
                           sig_col="q_value",
                           sig_threshold=0.05,
                           figsize=None,):
    """
    Plot KSEA activity trajectories: `value` vs timepoint, one panel per condition, one line
    per kinase. Points that pass the significance threshold are drawn filled, others hollow.

    Args:
        profile_df: output of kinase_activity_profile().
        kinases: subset of kinases to plot (default: all in profile_df).
        conditions: subset/order of conditions (default: order of appearance).
        value: y-axis quantity (default "z_score").
        sig_col: significance column for the filled/hollow marker (default "q_value").
        sig_threshold: threshold below which a point is filled (default 0.05).
        figsize: figure size; auto-sized if None.

    Returns:
        (fig, axes).
    """
    data = profile_df if kinases is None else profile_df[profile_df["kinase"].isin(kinases)]
    if conditions is None:
        conditions = list(dict.fromkeys(data["condition"]))
    timepoints = list(data["timepoint"].cat.categories) if hasattr(data["timepoint"], "cat") \
        else list(dict.fromkeys(data["timepoint"]))

    if figsize is None:
        figsize = (4.6 * len(conditions), 4.2)
    fig, axes = plt.subplots(1, len(conditions), figsize=figsize, sharey=True, squeeze=False,)
    axes = axes[0]

    plotted = sorted(data["kinase"].unique())
    cmap = plt.get_cmap("tab10" if len(plotted) <= 10 else "tab20")
    colours = {k: cmap(i % cmap.N) for i, k in enumerate(plotted)}
    x = range(len(timepoints))

    for ax, condition in zip(axes, conditions):
        sub = data[data["condition"] == condition]
        for kinase in plotted:
            series = (sub[sub["kinase"] == kinase]
                      .set_index("timepoint")
                      .reindex(timepoints))
            y = series[value].to_numpy(dtype=float)
            ax.plot(x, y, "-", color=colours[kinase], lw=1.8, label=kinase,)
            sig = series[sig_col].to_numpy(dtype=float) < sig_threshold
            ax.scatter(np.array(x)[sig], y[sig], color=colours[kinase], s=40, zorder=3,)
            ax.scatter(np.array(x)[~sig], y[~sig], facecolors="white",
                       edgecolors=colours[kinase], s=30, zorder=3,)
        ax.axhline(0, color="grey", ls="--", lw=1,)
        ax.set_xticks(list(x))
        ax.set_xticklabels(timepoints,)
        ax.set_xlabel("timepoint")
        ax.set_title(condition)
        ax.grid(alpha=0.3,)
    axes[0].set_ylabel(f"{value} (filled = {sig_col} < {sig_threshold})")
    axes[-1].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left",)
    fig.tight_layout()
    return fig, axes
