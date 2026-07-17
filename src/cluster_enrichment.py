"""
Cluster enrichment: test whether per-site annotations are over-represented in clusters.

Motivation
──────────
The temporal clusters are defined ONLY by phosphorylation dynamics (the log2 fold-change
time series). They carry no biological meaning by construction. This module asks, for a
cluster and an *independent* annotation (a predicted kinase, an ERK motif, a curated
process, ...): does that annotation appear in the cluster MORE OFTEN THAN EXPECTED BY
CHANCE? Because the annotation never entered the clustering, a positive answer is a genuine,
non-circular association between a dynamic behaviour and a biological property.

Statistical approach
─────────────────────
For each (cluster, annotation category) we build a 2x2 contingency table over the tested
universe of sites:

                        in this cluster   not in this cluster
    has annotation            a                  c
    lacks annotation          b                  d

and apply Fisher's exact test. Fisher's exact test computes the probability of seeing a
table at least as extreme as the observed one under the null hypothesis that cluster
membership and annotation are independent, using the hypergeometric distribution (i.e. it
is exact for any counts, unlike the chi-square approximation which is unreliable for small
a). We report:
  * odds_ratio      = (a*d)/(b*c)                — Fisher's effect size.
  * log2_enrichment = log2( (a/cluster_size) / (category_total/universe) )
                    = log2(observed fraction / expected fraction)  — intuitive effect size.
  * p_value         — Fisher's exact p.
  * q_value         — p corrected for multiple testing across ALL (cluster x category)
                      tests with Benjamini-Hochberg (controls the false discovery rate).

The "universe" (denominator population) is the set of sites the question is even defined
for, and it MUST be chosen deliberately — it is the most common point of critique:
  * Kinase enrichment: universe = sites that received a kinase prediction (single-localized
    scored sites). Unscored sites cannot be "ERK substrates or not", so including them would
    deflate the category fraction and inflate enrichment.
  * Metadata enrichment: universe defaults to sites where the annotation is DEFINED
    (non-null), so absence-of-curation is not miscounted as absence-of-property.

References: Fisher (1922); Benjamini & Hochberg (1995, JRSS-B) for FDR; the hypergeometric
"over-representation test" underlies GO/pathway enrichment tools (e.g. DAVID, gProfiler).
"""
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Membership builders — define the tested universe and the boolean matrix
# ---------------------------------------------------------------------------
def merge_kinase_groups(membership,
                        groups,
                        keep_members=False,):
    """
    Collapse paralog kinase columns of a boolean membership matrix into merged groups.

    Paralogous kinases (e.g. ERK1/ERK2) share almost the same substrate motif, so the
    matrices cannot tell them apart and it is often more honest to report them as one group.
    The merge is a logical **OR** over the member columns: a site belongs to group "ERK1/2"
    if it is a (predicted) substrate of ERK1 OR ERK2. This is the correct set-level union —
    for downstream KSEA it means the group's substrate set is the union of its members' sets,
    with shared sites counted once (never double-counted).

    Args:
        membership: boolean DataFrame (sites x kinases), from kinase_membership().
        groups: manual mapping {group_name: [member_kinase, ...]}, e.g.
            {"ERK1/2": ["ERK1", "ERK2"], "P38": ["P38A", "P38B", "P38D"]}. Matching is
            case-insensitive; members absent from `membership` are ignored (with no error).
        keep_members: if True, keep the individual member columns alongside the merged
            group column; if False (default), drop them.

    Returns:
        Boolean DataFrame with the merged group columns added and (optionally) the member
        columns removed. Ungrouped kinases are passed through unchanged.
    """
    result = membership.copy()
    upper_to_col = {c.upper(): c for c in result.columns}
    consumed = []
    for group_name, members in groups.items():
        present = [upper_to_col[m.upper()] for m in members if m.upper() in upper_to_col]
        if not present:
            continue
        result[group_name] = result[present].any(axis=1)
        consumed.extend(present)
    if not keep_members:
        drop = [c for c in set(consumed) if c not in groups]
        result = result.drop(columns=drop,)
    return result.astype(bool)


def kinase_membership(df,
                      kinases=None,
                      kinase_prefix="predicted_kinase",
                      top_n=5,
                      ranks=None,
                      min_percentile=None,
                      groups=None,
                      keep_members=False,):
    """
    Build a boolean sites x kinases membership matrix from the predicted-kinase columns.

    A site "has kinase K" if K appears among its scanned predicted-kinase ranks (optionally
    only certain `ranks`, and only when the matching percentile >= `min_percentile`). The
    universe (matrix rows) is restricted to sites that received a prediction — the only sites
    for which "substrate of K or not" is a meaningful question.

    Args:
        df: dataframe with predicted-kinase columns ("{prefix}_{rank}" and "{prefix}_{rank}_prob").
        kinases: kinase name/list to include; if None, every kinase seen in the columns.
        kinase_prefix: predicted-kinase column prefix (default "predicted_kinase").
        top_n: highest rank to scan when `ranks` is None (default 5).
        ranks: explicit ranks to scan (overrides top_n), e.g. [1] for the top call only.
        min_percentile: require the matching *_prob >= this to count a hit (default None).
        groups: optional manual paralog merge, e.g. {"ERK1/2": ["ERK1", "ERK2"]}; merged
            via merge_kinase_groups() (logical OR of member columns). Member kinases are
            included in the build even if absent from `kinases`.
        keep_members: if True, keep the individual member columns alongside merged groups.

    Returns:
        Boolean DataFrame indexed by scored site, columns = kinase names (upper-cased),
        with any `groups` collapsed into merged columns.
    """
    scan_ranks = list(ranks) if ranks is not None else list(range(1, top_n + 1))
    name_cols = [f"{kinase_prefix}_{r}" for r in scan_ranks if f"{kinase_prefix}_{r}" in df.columns]
    if not name_cols:
        raise ValueError("No predicted-kinase columns found for the requested ranks.")

    scored = df[df[name_cols].notna().any(axis=1)]
    if kinases is None:
        seen = pd.unique(scored[name_cols].values.ravel("K"))
        columns = sorted({str(k).upper() for k in seen if isinstance(k, str)})
    else:
        columns = [k.upper() for k in ([kinases] if isinstance(kinases, str) else kinases)]

    # Make sure group members are built even if not listed in `kinases`.
    if groups is not None:
        members = {m.upper() for member_list in groups.values() for m in member_list}
        columns = sorted(set(columns) | members)

    membership = pd.DataFrame(False, index=scored.index, columns=columns)
    for rank in scan_ranks:
        name_col = f"{kinase_prefix}_{rank}"
        if name_col not in scored.columns:
            continue
        names = scored[name_col].astype("string").str.upper()
        if min_percentile is not None:
            prob_col = f"{kinase_prefix}_{rank}_prob"
            if prob_col in scored.columns:
                names = names.where(pd.to_numeric(scored[prob_col], errors="coerce") >= min_percentile)
        dummies = pd.get_dummies(names).reindex(columns=columns, fill_value=False).astype(bool)
        membership = membership | dummies

    membership = membership.astype(bool)
    if groups is not None:
        membership = merge_kinase_groups(membership, groups, keep_members=keep_members,)
    return membership


def annotation_membership(df,
                          annotation_col,
                          categories=None,
                          positive_values=None,
                          multi_sep=None,
                          universe="defined",):
    """
    Build a boolean sites x categories membership matrix from a metadata column.

    Three modes, chosen by the arguments:
      * positive_values given → a single category named after the column; a site "has" it
        when its value is in positive_values (e.g. ERK_motif with positive_values=[True]).
      * multi_sep given → the cell holds several `multi_sep`-joined labels (e.g. an
        ON_PROCESS field "apoptosis|cell cycle"); each label becomes its own category.
      * otherwise → one category per distinct value of the column (one-hot).

    Args:
        df: dataframe containing the annotation column.
        annotation_col: the metadata column to test.
        categories: restrict to these categories (default: all present).
        positive_values: values that count as "has annotation" (single-category mode).
        multi_sep: separator for multi-label cells (e.g. "|").
        universe: "defined" restricts the tested rows to non-null annotation values
            (recommended for sparse curated fields); "all" keeps every row (missing counts
            as not-in-category — appropriate for a complete boolean like ERK_motif).

    Returns:
        Boolean DataFrame indexed by the universe of sites, columns = categories.
    """
    if universe == "defined":
        base = df[df[annotation_col].notna()]
    elif universe == "all":
        base = df
    else:
        raise ValueError(f"universe must be 'defined' or 'all', got {universe!r}.")

    values = base[annotation_col]
    if positive_values is not None:
        membership = pd.DataFrame({annotation_col: values.isin(positive_values).to_numpy()},
                                  index=base.index,)
    elif multi_sep is not None:
        exploded = values.astype("string").str.split(multi_sep).explode().str.strip()
        membership = pd.crosstab(exploded.index, exploded).astype(bool)
        membership = membership.reindex(base.index, fill_value=False)
    else:
        membership = pd.get_dummies(values.astype("string")).astype(bool)

    if categories is not None:
        membership = membership.reindex(columns=categories, fill_value=False)
    return membership.astype(bool)


# ---------------------------------------------------------------------------
# 2. The enrichment test
# ---------------------------------------------------------------------------
def cluster_enrichment(cluster_labels,
                       membership,
                       alternative="two-sided",
                       min_count=1,
                       fdr_method="fdr_bh",):
    """
    Fisher's exact enrichment of each membership category within each cluster.

    The tested universe is the set of sites present in BOTH `membership` (rows) and
    `cluster_labels` (non-null) — so pick the universe when you build `membership`.

    Args:
        cluster_labels: Series mapping site -> cluster id (index aligned with membership).
            Sites with a missing cluster label are dropped from the universe.
        membership: boolean DataFrame (sites x categories) from a membership builder.
        alternative: Fisher alternative — "two-sided" (enrichment OR depletion; default),
            or "greater" to test enrichment only (more powerful when the direction is
            pre-specified).
        min_count: skip categories with fewer than this many annotated sites in the whole
            universe (their odds ratios are too unstable to interpret).
        fdr_method: multipletests method for correcting across all tests (default BH).

    Returns:
        Tidy DataFrame, one row per (cluster, category), with columns: cluster, category,
        in_cluster_with (a), cluster_size (a+b), category_total (a+c), universe (N),
        expected (cluster_size*category_total/N), odds_ratio, log2_enrichment, p_value,
        q_value. Sorted by q_value.
    """
    labels = pd.Series(cluster_labels).reindex(membership.index)
    keep = labels.notna()
    labels = labels[keep]
    membership = membership.loc[keep]
    universe = len(labels)
    clusters = np.sort(labels.unique())
    label_arr = labels.to_numpy()

    records = []
    for category in membership.columns:
        has = membership[category].to_numpy(dtype=bool)
        category_total = int(has.sum())
        if category_total < min_count:
            continue
        for cluster in clusters:
            in_cluster = label_arr == cluster
            a = int(np.sum(in_cluster & has))
            b = int(np.sum(in_cluster & ~has))
            c = int(np.sum(~in_cluster & has))
            d = int(np.sum(~in_cluster & ~has))
            cluster_size = a + b
            odds_ratio, p_value = fisher_exact([[a, b], [c, d]], alternative=alternative,)
            expected = cluster_size * category_total / universe if universe else np.nan
            observed_fraction = a / cluster_size if cluster_size else np.nan
            expected_fraction = category_total / universe if universe else np.nan
            with np.errstate(divide="ignore"):
                log2_enrichment = float(np.log2(observed_fraction / expected_fraction)) \
                    if observed_fraction and expected_fraction else -np.inf
            records.append({"cluster": cluster,
                            "category": category,
                            "in_cluster_with": a,
                            "cluster_size": cluster_size,
                            "category_total": category_total,
                            "universe": universe,
                            "expected": expected,
                            "odds_ratio": odds_ratio,
                            "log2_enrichment": log2_enrichment,
                            "p_value": p_value,})

    result = pd.DataFrame(records)
    if not result.empty:
        result["q_value"] = multipletests(result["p_value"], method=fdr_method)[1]
        result = result.sort_values("q_value").reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# 3. Plot
# ---------------------------------------------------------------------------
def plot_enrichment_heatmap(enrichment_df,
                            value="log2_enrichment",
                            sig_col="q_value",
                            sig_threshold=0.05,
                            clip=None,
                            cmap="RdBu_r",
                            title=None,
                            figsize=None,
                            ax=None,):
    """
    Heatmap of enrichment effect size (categories x clusters) with significance stars.

    Cells are coloured by `value` (log2 enrichment by default, on a diverging red=enriched /
    blue=depleted scale centred at 0) and marked with "*" where `sig_col` < `sig_threshold`.

    Args:
        enrichment_df: output of cluster_enrichment().
        value: column to colour by (default "log2_enrichment").
        sig_col: column used for the significance star (default "q_value").
        sig_threshold: threshold below which a cell is starred (default 0.05).
        clip: optional (low, high) to clip the colour scale (e.g. (-3, 3)) so a few extreme
            cells do not wash out the rest.
        cmap: diverging colormap (default "RdBu_r").
        title: plot title.
        figsize: figure size; auto-sized if None.
        ax: existing Axes; a new figure is created if None.

    Returns:
        (fig, ax).
    """
    effect = enrichment_df.pivot(index="category", columns="cluster", values=value)
    sig = enrichment_df.pivot(index="category", columns="cluster", values=sig_col)
    data = effect.to_numpy(dtype=float)
    if clip is not None:
        data = np.clip(data, clip[0], clip[1])
    bound = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0

    if figsize is None:
        figsize = (max(5, 0.7 * effect.shape[1] + 2), max(3, 0.45 * effect.shape[0] + 1))
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize,)
    else:
        fig = ax.figure

    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-bound, vmax=bound,)
    ax.set_xticks(range(effect.shape[1]))
    ax.set_xticklabels(effect.columns, fontsize=8,)
    ax.set_yticks(range(effect.shape[0]))
    ax.set_yticklabels(effect.index, fontsize=8,)
    ax.set_xlabel(enrichment_df["cluster"].name or "cluster")
    fig.colorbar(im, ax=ax, label=value,)

    sig_arr = sig.to_numpy(dtype=float)
    for i in range(effect.shape[0]):
        for j in range(effect.shape[1]):
            if np.isfinite(sig_arr[i, j]) and sig_arr[i, j] < sig_threshold:
                ax.text(j, i, "*", ha="center", va="center", color="black", fontsize=11,)

    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax
