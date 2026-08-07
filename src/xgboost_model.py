"""
XGBoost cluster classifier + SHAP interpretation for phosphosite dynamic clusters.

Goal: learn to predict which dynamic cluster a phosphosite belongs to from its
*metadata* (not the raw temporal profile), and use SHAP to quantify how much each
feature contributes to the cluster assignment. This is the WT-side foundation for
the later step of transferring cluster labels to the mutant datasets.

Two engineered feature families are built and compared:
  A. build_bio_features()      — intrinsic/biological metadata: predicted kinase,
                                 residue, ERK motif, functional score, localization,
                                 peptide quality, curated-annotation flags.
  B. build_temporal_features() — temporal-SHAPE descriptors (peak, AUC, slopes,
                                 amplitude, transient score, cross-condition synergy)
                                 summarised from the FC profiles.

Leakage note
────────────
The 6 cluster-label columns were computed by KMeans on the temporal profiles.
* KMeans_adaptive_cluster_WT_EGF_* used ALL THREE conditions jointly (EGF+INS+EGFnINS)
  despite the "WT_EGF" name — so for that target the temporal descriptors (family B,
  from any condition) reconstruct the clustering input and act as a DESCRIPTIVE
  characterisation, not an independent predictor.
* KMeans_11_cluster_WT_EGF_* used EGF only — so for that target the INS/EGFnINS
  descriptors are genuinely non-leaky (a crosstalk test).
The biological metadata (family A) is always an independent predictor. Running the two
families separately and combined (see run_feature_group) keeps the two questions apart.

Module layout
─────────────
1. Feature family A — build_bio_features()
2. Feature family B — build_temporal_features()  (+ _shape_descriptors helper)
3. Target           — get_target()
4. Modeling         — train_xgb_classifier(), cross_validate_xgb()
5. Evaluation       — evaluate_classifier(), run_feature_group(), plot_confusion()
6. SHAP             — shap_explain(), plot_shap_global(), plot_shap_per_class()

Typical workflow
────────────────
    from src.xgboost_model import (build_bio_features, build_temporal_features,
                                   get_target, run_feature_group, shap_explain)
    X_bio      = build_bio_features(df)
    X_temporal = build_temporal_features(df)
    y, classes = get_target(df, "KMeans_adaptive_cluster_WT_EGF_log2_FC")
    result     = run_feature_group(X_bio, y, name="bio")           # split+train+eval
    sv, X_s    = shap_explain(result["model"], X_bio)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             classification_report,
                             confusion_matrix,)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
import shap

from src.column_spec import ColumnSpec


# ---------------------------------------------------------------------------
# 1. Feature family A — biological / intrinsic metadata
# ---------------------------------------------------------------------------
# Numeric columns fed straight through (XGBoost handles NaN natively, so
# non-single-localized sites lacking kinase probabilities keep NaN as a signal).
BIO_NUMERIC_COLS = ["functional_score",
                    "MaxPepProb",
                    "protein_length",
                    "nrPeptides",
                    "nr_tryptic_peptides",
                    "NumPhos",
                    "LocalizedNumPhos",
                    "n:reps",
                    "predicted_kinase_1_prob",
                    "predicted_kinase_2_prob",
                    "predicted_kinase_3_prob",
                    "predicted_kinase_4_prob",
                    "predicted_kinase_5_prob",]

# Sparse curated-annotation columns that mostly hold the '0' sentinel — binarised
# to a "has curated annotation" flag rather than one-hot encoding the free text.
BIO_ANNOTATION_COLS = ["ON_FUNCTION",
                       "ON_PROCESS",
                       "ON_PROT_INTERACT",]


def build_bio_features(df,
                       top_k_kinases=25,
                       numeric_cols=BIO_NUMERIC_COLS,
                       annotation_cols=BIO_ANNOTATION_COLS,):
    """
    Build the biological / intrinsic-metadata feature matrix (family A).

    Encodes, per site: numeric metadata (functional score, peptide quality,
    localization counts, kinase percentile scores, derived peptide length),
    one-hot phospho-acceptor residue (S/T/Y), a parsed ERK-motif boolean, a
    top-K one-hot of the rank-1 predicted kinase (+ an 'other' bucket), and
    binary "has curated annotation" flags. Contains NO temporal data columns
    and NO cluster labels, so it is an independent predictor of the cluster.

    Args:
        df: source DataFrame (one row per phosphosite) with the project metadata columns.
        top_k_kinases: number of most-frequent rank-1 predicted kinases to one-hot
            individually; the remainder collapse into a single 'other' column.
        numeric_cols: metadata columns passed through numerically (missing → NaN).
        annotation_cols: curated columns binarised to has_<col> presence flags.

    Returns:
        DataFrame (indexed like df) of engineered biological features, all numeric.
    """
    feats = {}

    # -- numeric metadata (coerce to float; leave missing as NaN) --------------
    for col in numeric_cols:
        if col in df.columns:
            feats[col] = pd.to_numeric(df[col],
                                       errors="coerce",)

    # derived peptide length from Start/End residue positions
    if "Start" in df.columns and "End" in df.columns:
        start = pd.to_numeric(df["Start"], errors="coerce",)
        end = pd.to_numeric(df["End"], errors="coerce",)
        feats["peptide_length"] = end - start

    X = pd.DataFrame(feats,
                     index=df.index,)

    # -- one-hot phospho-acceptor residue (S / T / Y) --------------------------
    if "kinase_residue" in df.columns:
        for res in ["S", "T", "Y"]:
            X[f"residue_{res}"] = (df["kinase_residue"] == res).astype(int)

    # -- ERK motif -> clean boolean (parse '0'/'False'/'True'/'True|False'/...) -
    if "ERK_motif" in df.columns:
        X["erk_motif_any"] = (df["ERK_motif"]
                              .astype(str)
                              .str.contains("True")
                              .astype(int))

    # -- top-K one-hot of the rank-1 predicted kinase --------------------------
    if "predicted_kinase_1" in df.columns:
        kin = df["predicted_kinase_1"]
        top = kin.value_counts().index[:top_k_kinases]
        for name in top:
            X[f"kin1_{name}"] = (kin == name).astype(int)
        # 'other' = a kinase was predicted but not among the top-K; NaN stays all-zero
        X["kin1_other"] = (kin.notna() & ~kin.isin(top)).astype(int)

    # -- curated-annotation presence flags -------------------------------------
    for col in annotation_cols:
        if col in df.columns:
            X[f"has_{col}"] = (df[col].notna()
                               & (df[col].astype(str) != "0")).astype(int)

    return X


# ---------------------------------------------------------------------------
# 2. Feature family B — temporal-shape descriptors
# ---------------------------------------------------------------------------
def _label_to_minutes(label,):
    """
    Map a timepoint label to a numeric minute value for the temporal axis.

    Args:
        label: timepoint label from a column name (e.g. 'starve', '2', '90').

    Returns:
        float minutes ('starve' -> 0.0, numeric strings -> their value), or NaN
        for anything non-numeric that is not 'starve' (e.g. 'full').
    """
    if label == "starve":
        return 0.0
    try:
        return float(label)
    except (TypeError, ValueError):
        return np.nan


def _shape_descriptors(values,
                       times,
                       prefix,):
    """
    Compute temporal-shape descriptors for one condition's time series.

    Operates row-wise (vectorised) over an (n_sites, n_timepoints) matrix whose
    columns are ordered by ascending time. NaN-robust: all-NaN rows yield NaN
    descriptors rather than raising.

    Args:
        values: (n_sites, n_timepoints) float array of profile values (time-ordered).
        times: (n_timepoints,) float array of minutes matching the columns.
        prefix: condition prefix for the output column names (e.g. 'EGF').

    Returns:
        dict {feature_name: (n_sites,) array} of descriptors, plus a private
        '_auc'/'_peak' pair reused for cross-condition synergy features.
    """
    n = values.shape[0]
    valid = np.isfinite(values).any(axis=1,)          # rows with ≥1 finite value

    with np.errstate(invalid="ignore", divide="ignore",):
        peak_value = np.full(n, np.nan,)
        trough_value = np.full(n, np.nan,)
        peak_time = np.full(n, np.nan,)
        peak_value[valid] = np.nanmax(values[valid], axis=1,)
        trough_value[valid] = np.nanmin(values[valid], axis=1,)
        peak_idx = np.nanargmax(np.where(np.isfinite(values[valid]),
                                         values[valid],
                                         -np.inf,),
                                axis=1,)
        peak_time[valid] = times[peak_idx]

        amplitude = peak_value - trough_value
        mean = np.full(n, np.nan,)
        std = np.full(n, np.nan,)
        mean[valid] = np.nanmean(values[valid], axis=1,)
        std[valid] = np.nanstd(values[valid], axis=1,)

        # AUC over the numeric time axis (trapezoidal); NaN if any gap
        auc = np.trapz(values, times, axis=1,)

        first, last = values[:, 0], values[:, -1]
        net_change = last - first
        initial_slope = (values[:, 1] - first) / (times[1] - times[0])
        # decay from peak to last point (0 when the peak IS the last point)
        dt = times[-1] - peak_time
        decay_slope = np.where(dt > 0,
                               (last - peak_value) / np.where(dt == 0, np.nan, dt),
                               0.0,)
        transient_score = np.where(amplitude > 0,
                                   (peak_value - last) / np.where(amplitude == 0,
                                                                  np.nan,
                                                                  amplitude,),
                                   np.nan,)

    out = {f"{prefix}_peak_value": peak_value,
           f"{prefix}_peak_time": peak_time,
           f"{prefix}_trough_value": trough_value,
           f"{prefix}_amplitude": amplitude,
           f"{prefix}_auc": auc,
           f"{prefix}_initial_slope": initial_slope,
           f"{prefix}_decay_slope": decay_slope,
           f"{prefix}_net_change": net_change,
           f"{prefix}_transient_score": transient_score,
           f"{prefix}_mean": mean,
           f"{prefix}_std": std,}
    return out


def build_temporal_features(df,
                            data_type="log2:FC",
                            cell_line="WT",
                            conditions=("_EGF_", "_INS_", "_EGFnINS_"),):
    """
    Build the temporal-shape descriptor feature matrix (family B).

    For each condition, selects the profile columns with ColumnSpec (excluding the
    'full' timepoint, matching the clustering input), orders them on a numeric
    minute axis, and summarises the shape into interpretable descriptors (peak,
    trough, amplitude, AUC, slopes, net change, transient score, mean, std). Adds
    cross-condition crosstalk features when all three conditions are present:
    non-additive synergy of EGF+INS co-stimulation and the EGF-vs-INS AUC gap.

    NOTE: for the adaptive (all-3-condition) cluster target these descriptors
    reconstruct the clustering input and are therefore DESCRIPTIVE, not independent
    predictors — see the module-level leakage note.

    Args:
        df: source DataFrame with the {cell_line}_{data_type}_{cond}_{time} columns.
        data_type: transform space to summarise (default 'log2:FC').
        cell_line: cell-line prefix to select (default 'WT').
        conditions: tuple of condition tokens (with underscores) to summarise.

    Returns:
        DataFrame (indexed like df) of temporal-shape descriptors, all numeric.
    """
    frames = {}
    auc_by_cond = {}
    peak_by_cond = {}

    for cond in conditions:
        cols = ColumnSpec.select(df,
                                 cell_lines=[cell_line],
                                 data_type=data_type,
                                 conditions=[cond],
                                 exclude_full=True,)
        if not cols:
            continue

        labels = [c.split("_")[3] for c in cols]
        minutes = np.array([_label_to_minutes(l) for l in labels],
                           dtype=float,)
        order = np.argsort(minutes,)
        ordered_cols = [cols[i] for i in order]
        times = minutes[order]

        values = df[ordered_cols].to_numpy(dtype=float,)
        prefix = cond.strip("_")
        desc = _shape_descriptors(values,
                                  times,
                                  prefix,)
        frames.update(desc,)
        auc_by_cond[prefix] = desc[f"{prefix}_auc"]
        peak_by_cond[prefix] = desc[f"{prefix}_peak_value"]

    X = pd.DataFrame(frames,
                     index=df.index,)

    # -- cross-condition crosstalk / synergy features --------------------------
    if all(k in auc_by_cond for k in ("EGF", "INS", "EGFnINS",)):
        X["synergy_auc"] = auc_by_cond["EGFnINS"] - (auc_by_cond["EGF"]
                                                     + auc_by_cond["INS"])
        X["synergy_peak"] = peak_by_cond["EGFnINS"] - (peak_by_cond["EGF"]
                                                       + peak_by_cond["INS"])
        X["egf_minus_ins_auc"] = auc_by_cond["EGF"] - auc_by_cond["INS"]

    return X


# ---------------------------------------------------------------------------
# 3. Target
# ---------------------------------------------------------------------------
def get_target(df,
               target_col="KMeans_adaptive_cluster_WT_EGF_log2_FC",):
    """
    Extract and label-encode the cluster-label target column.

    Args:
        df: source DataFrame containing the cluster-label column.
        target_col: name of the cluster-label column to predict.

    Returns:
        (y, class_labels) where y is an int array in 0..K-1 (XGBoost-ready) and
        class_labels are the original cluster ids in encoded order.
    """
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[target_col].to_numpy(),)
    return y, list(encoder.classes_)


# ---------------------------------------------------------------------------
# 4. Modeling
# ---------------------------------------------------------------------------
def _make_model(num_class,
                random_state,):
    """
    Construct an XGBoost multi-class classifier with the project defaults.

    Args:
        num_class: number of target classes.
        random_state: RNG seed for reproducibility.

    Returns:
        An unfitted XGBClassifier configured for multi:softprob.
    """
    return XGBClassifier(objective="multi:softprob",
                         num_class=num_class,
                         n_estimators=400,
                         max_depth=5,
                         learning_rate=0.08,
                         subsample=0.8,
                         colsample_bytree=0.8,
                         min_child_weight=2,
                         tree_method="hist",
                         eval_metric="mlogloss",
                         n_jobs=-1,
                         random_state=random_state,)


def train_xgb_classifier(X,
                         y,
                         test_size=0.2,
                         random_state=0,):
    """
    Stratified split + train an XGBoost classifier with balanced sample weights.

    Class imbalance is handled by per-sample 'balanced' weights (XGBClassifier has
    no class_weight argument). Missing values in X are left as NaN — XGBoost learns
    a default split direction for them.

    Args:
        X: feature DataFrame.
        y: int label array (0..K-1).
        test_size: held-out fraction for the stratified split.
        random_state: RNG seed for the split and the model.

    Returns:
        (model, (X_train, X_test, y_train, y_test)) — the fitted classifier and split.
    """
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=y,)
    sample_weight = compute_sample_weight(class_weight="balanced",
                                          y=y_train,)
    model = _make_model(num_class=len(np.unique(y)),
                        random_state=random_state,)
    model.fit(X_train,
              y_train,
              sample_weight=sample_weight,)
    return model, (X_train, X_test, y_train, y_test)


def cross_validate_xgb(X,
                       y,
                       n_splits=5,
                       random_state=0,):
    """
    Stratified K-fold cross-validation (accuracy + macro-F1) with balanced weights.

    Args:
        X: feature DataFrame.
        y: int label array (0..K-1).
        n_splits: number of stratified folds.
        random_state: RNG seed for the fold split and the models.

    Returns:
        dict with 'accuracy' and 'macro_f1' arrays (one score per fold).
    """
    skf = StratifiedKFold(n_splits=n_splits,
                          shuffle=True,
                          random_state=random_state,)
    accs, f1s = [], []
    X_values = X.reset_index(drop=True,)
    for train_idx, val_idx in skf.split(X_values, y,):
        X_tr, X_val = X_values.iloc[train_idx], X_values.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        sw = compute_sample_weight(class_weight="balanced",
                                   y=y_tr,)
        model = _make_model(num_class=len(np.unique(y)),
                            random_state=random_state,)
        model.fit(X_tr,
                  y_tr,
                  sample_weight=sw,)
        pred = model.predict(X_val,)
        accs.append(accuracy_score(y_val, pred,))
        f1s.append(f1_score(y_val, pred, average="macro",))
    return {"accuracy": np.array(accs,),
            "macro_f1": np.array(f1s,)}


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------
def evaluate_classifier(model,
                        X_test,
                        y_test,
                        y_train=None,
                        class_labels=None,):
    """
    Evaluate a fitted classifier against a majority-class baseline.

    Args:
        model: fitted classifier with a .predict method.
        X_test: held-out feature DataFrame.
        y_test: held-out int labels.
        y_train: training labels; if given, a DummyClassifier('most_frequent')
            baseline is fit on train and scored on test. Otherwise the baseline is
            the most-frequent-class proportion within y_test.
        class_labels: original cluster ids for the classification report.

    Returns:
        dict with accuracy, macro_f1, baseline_acc, lift (accuracy − baseline),
        y_pred, and the sklearn classification_report (as a dict).
    """
    y_pred = model.predict(X_test,)
    acc = accuracy_score(y_test, y_pred,)
    macro_f1 = f1_score(y_test, y_pred, average="macro",)

    if y_train is not None:
        dummy = DummyClassifier(strategy="most_frequent",)
        dummy.fit(np.zeros((len(y_train), 1)), y_train,)
        baseline_acc = accuracy_score(y_test,
                                      dummy.predict(np.zeros((len(y_test), 1))),)
    else:
        counts = np.bincount(y_test,)
        baseline_acc = counts.max() / counts.sum()

    target_names = ([str(c) for c in class_labels]
                    if class_labels is not None else None)
    report = classification_report(y_test,
                                   y_pred,
                                   target_names=target_names,
                                   output_dict=True,
                                   zero_division=0,)
    return {"accuracy": acc,
            "macro_f1": macro_f1,
            "baseline_acc": baseline_acc,
            "lift": acc - baseline_acc,
            "y_pred": y_pred,
            "report": report,}


def run_feature_group(X,
                      y,
                      name,
                      class_labels=None,
                      test_size=0.2,
                      random_state=0,):
    """
    Split, train, and evaluate one feature group in a single call.

    Convenience wrapper used to run families A (bio), B (temporal) and A+B combined
    and tabulate a comparison. Keeps the fitted model and the split so SHAP can be
    run afterwards on the same data.

    Args:
        X: feature DataFrame for this group.
        y: int label array (0..K-1).
        name: label for this feature group (e.g. 'bio', 'temporal', 'combined').
        class_labels: original cluster ids for the classification report.
        test_size: held-out fraction for the stratified split.
        random_state: RNG seed.

    Returns:
        dict with name, n_features, accuracy, macro_f1, baseline_acc, lift, the
        fitted 'model', the 'split' tuple, and 'metrics' (full evaluate_classifier).
    """
    model, split = train_xgb_classifier(X,
                                        y,
                                        test_size=test_size,
                                        random_state=random_state,)
    X_train, X_test, y_train, y_test = split
    metrics = evaluate_classifier(model,
                                  X_test,
                                  y_test,
                                  y_train=y_train,
                                  class_labels=class_labels,)
    return {"name": name,
            "n_features": X.shape[1],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "baseline_acc": metrics["baseline_acc"],
            "lift": metrics["lift"],
            "model": model,
            "split": split,
            "metrics": metrics,}


def plot_confusion(y_test,
                   y_pred,
                   class_labels=None,
                   normalize=True,
                   ax=None,
                   title="Confusion matrix",):
    """
    Plot a (optionally row-normalized) confusion-matrix heatmap.

    Args:
        y_test: true int labels.
        y_pred: predicted int labels.
        class_labels: tick labels (original cluster ids).
        normalize: if True, normalize each true-class row to sum to 1.
        ax: optional matplotlib Axes to draw on.
        title: plot title.

    Returns:
        The matplotlib Axes with the heatmap.
    """
    cm = confusion_matrix(y_test, y_pred,).astype(float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True,)
        cm = np.divide(cm,
                       row_sums,
                       out=np.zeros_like(cm),
                       where=row_sums != 0,)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6),)
    im = ax.imshow(cm,
                   cmap="viridis",
                   vmin=0,
                   vmax=1 if normalize else None,)
    ticks = (class_labels
             if class_labels is not None else range(cm.shape[0]))
    ax.set_xticks(range(cm.shape[0]),)
    ax.set_yticks(range(cm.shape[0]),)
    ax.set_xticklabels(ticks, rotation=90,)
    ax.set_yticklabels(ticks,)
    ax.set_xlabel("Predicted cluster",)
    ax.set_ylabel("True cluster",)
    ax.set_title(title,)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,)
    return ax


# ---------------------------------------------------------------------------
# 6. SHAP
# ---------------------------------------------------------------------------
def _stack_shap_values(shap_values,):
    """
    Normalize SHAP output to a 3D (n_samples, n_features, n_classes) array.

    Handles both the classic list-of-arrays (one per class) return and the newer
    3D-array return of shap.TreeExplainer for multi-class models.

    Args:
        shap_values: output of TreeExplainer.shap_values (list or ndarray).

    Returns:
        ndarray of shape (n_samples, n_features, n_classes).
    """
    if isinstance(shap_values, list,):
        return np.stack(shap_values, axis=-1,)
    arr = np.asarray(shap_values,)
    if arr.ndim == 2:                                 # single-output fallback
        return arr[:, :, None]
    return arr


def shap_explain(model,
                 X,
                 max_samples=2000,
                 random_state=0,):
    """
    Compute SHAP values for a fitted tree model on a (sampled) feature matrix.

    Args:
        model: fitted XGBoost classifier.
        X: feature DataFrame the model was trained on.
        max_samples: cap on rows explained (sampled without replacement) to keep
            SHAP fast; None uses all rows.
        random_state: RNG seed for the sampling.

    Returns:
        (shap_values, X_sample) where shap_values is a 3D array
        (n_samples, n_features, n_classes) and X_sample is the explained subset.
    """
    if max_samples is not None and len(X) > max_samples:
        X_sample = X.sample(n=max_samples,
                            random_state=random_state,)
    else:
        X_sample = X
    explainer = shap.TreeExplainer(model,)
    shap_values = explainer.shap_values(X_sample,)
    return _stack_shap_values(shap_values,), X_sample


def plot_shap_global(shap_values,
                     X_sample,
                     max_display=20,
                     ax=None,
                     title="Global feature importance (mean |SHAP|, all classes)",):
    """
    Plot global feature importance as mean(|SHAP|) averaged over samples and classes.

    Args:
        shap_values: 3D array (n_samples, n_features, n_classes) from shap_explain.
        X_sample: the explained feature DataFrame (for column names).
        max_display: number of top features to show.
        ax: optional matplotlib Axes.
        title: plot title.

    Returns:
        The matplotlib Axes with the horizontal bar chart.
    """
    importance = np.abs(shap_values).mean(axis=(0, 2),)     # over samples & classes
    order = np.argsort(importance,)[::-1][:max_display]
    names = np.array(X_sample.columns,)[order]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(order))),)
    ax.barh(range(len(order)),
            importance[order][::-1],
            color="#4C72B0",)
    ax.set_yticks(range(len(order)),)
    ax.set_yticklabels(names[::-1],)
    ax.set_xlabel("mean |SHAP value|",)
    ax.set_title(title,)
    return ax


def plot_shap_per_class(shap_values,
                        X_sample,
                        class_index,
                        class_label=None,
                        max_display=15,):
    """
    Draw a SHAP beeswarm summary for one target class (which features drive it).

    Args:
        shap_values: 3D array (n_samples, n_features, n_classes) from shap_explain.
        X_sample: the explained feature DataFrame.
        class_index: encoded class index (column of the 3D array) to plot.
        class_label: original cluster id for the title.
        max_display: number of top features to show.

    Returns:
        None (renders the current matplotlib figure via shap.summary_plot).
    """
    label = class_label if class_label is not None else class_index
    shap.summary_plot(shap_values[:, :, class_index],
                      X_sample,
                      max_display=max_display,
                      show=False,)
    plt.title(f"SHAP — cluster {label}",)
