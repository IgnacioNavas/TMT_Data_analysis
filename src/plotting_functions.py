import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import plotly.express as px
import plotly.graph_objects as go
from IPython.core.pylabtools import figsize
from matplotlib_venn import venn2
from datetime import date


from itertools import cycle

from src.column_spec import *
from src.transformations import parse_columns

#---------------------
# Plotting helpers functions adjusted to the column naming system (CellLine)_(DataType)_(Treatment)_(TimePoint)
#---------------------


def build_legend(legend_plot,
                 color_palette,):
    """Build explicit legend handles and display labels for a figure legend.

    Matplotlib silently ignores any legend label that starts with an
    underscore (its reserved "hidden artist" convention). Because this
    project's palette/condition keys are written as `_EGF_`, `_INS_`,
    `_EGFnINS_`, passing them straight to ``fig.legend(labels=...)`` makes
    those entries vanish, so only part of the legend is drawn. This helper
    creates one proxy line handle per entry (coloured from ``color_palette``)
    and strips the surrounding underscores from the labels, so the full
    legend always renders.

    Args:
      legend_plot: List of label strings to show, in the same order as the
        plotted series (conditions or cell lines).
      color_palette: Either a dict mapping condition/cell-line keys to colours
        or a list/tuple of colours. Colours are matched to ``legend_plot``
        positionally (dict values are used in insertion order).

    Returns:
      Tuple ``(handles, labels)`` ready to pass to ``fig.legend`` as
      ``fig.legend(handles=handles, labels=labels, ...)``. Labels have any
      leading/trailing underscores removed (falling back to the original
      string if stripping leaves it empty).
    """
    if isinstance(color_palette, dict):
        colors = list(color_palette.values())
    else:
        colors = list(color_palette)

    handles = []
    labels = []
    for i, entry in enumerate(legend_plot):
        clean = str(entry).strip("_")
        labels.append(clean if clean else str(entry))
        color = colors[i] if i < len(colors) else "black"
        handles.append(plt.Line2D([0], [0],
                                   color=color,
                                   marker='o',
                                   linestyle='-',))
    return handles, labels


def plot_data(ax,
              row_df,
              data_type="",
              colors={},
              cell_lines=[],
              conditions=[]):
    """

    """
    column_names = row_df.index.tolist()
    sub_dtp = data_type.split(":")  # e.g. ["log2", "FC"]

    means_dir = parse_columns(row_df, cell_lines, data_type, conditions)

    sd_data_type = str(sub_dtp[0])+ ":sd"
    sd_dir = parse_columns(row_df, cell_lines, sd_data_type, conditions)

    x_axis_previous = [element for element in column_names if f"{cell_lines[0]}_{data_type}{conditions[0]}" in element]
    x_axis = [s.split("_")[3] for s in x_axis_previous]

    _site_parts = row_df["site"].split("~")
    _site_id = _site_parts[0]
    # site_index is "nan" (string) when no STY modification was localised in LFQ data
    if _site_id.lower() == "nan" or _site_id == "":
        site = _site_parts[1] if len(_site_parts) > 1 else row_df["site"]
    else:
        site = _site_id
    prot_name = row_df["protein_name"]
    protein_ID = row_df["protein_Id"]
    n_rep = row_df["n:reps"] if "n:reps" in row_df.index else ""

    color_by_cell = any(k in cell_lines for k in colors)

    for condition in conditions:
        for cell in cell_lines:
            color_key = cell if color_by_cell else condition
            ax.errorbar(x=x_axis,
                        y= row_df[means_dir[cell][condition]].values.astype(float),
                        yerr= row_df[sd_dir[cell][condition]].values.astype(float),
                        marker='o',
                        color=colors[color_key],
                        label=color_key,
                        capsize=4,
                        elinewidth=1.3,
                        alpha=1
                        )
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(f"{data_type}")
    ax.set_title(f"{site}_n{n_rep}" if n_rep != "" else site)

def plot_single_phosphosite(row,
                            data_type,
                            cell_lines,
                            conditions,
                            color_palette,
                            legend_plot,
                            len_x,
                            column_selection,
                            use_fixed_ylims,
                            y_lim_min,
                            y_lim_max,
                            saving_folder,
                            saving_path,
                            saving_info,
                            title_info,
                            save_pdf,
                            save_png,
                            plot_close):
    """Create one standalone figure for a single phosphosite row."""
    site_label = row["site"].split("~")[0]

    fig, ax = plt.subplots(figsize=(6, 4))

    plot_data(ax=ax,
              row_df=row,
              data_type=data_type,
              colors=color_palette,
              cell_lines=cell_lines,
              conditions=conditions)

    if use_fixed_ylims is None:
        row_vals = row[column_selection].dropna() if column_selection else pd.Series()
        if not row_vals.empty:
            rmin, rmax = row_vals.min(), row_vals.max()
            pad_min = rmin + rmin * 0.1 if rmin < 0 else rmin * 0.9
            ax.set_ylim(pad_min, rmax * 1.1)
    elif use_fixed_ylims and y_lim_min is not None:
        ax.set_ylim(y_lim_min, y_lim_max)
    elif not use_fixed_ylims and y_lim_min is not None:
        ax.set_ylim(y_lim_min, y_lim_max)

    ax.set_xlim(-1, len_x)

    _handles, _labels = build_legend(legend_plot, color_palette)
    fig.legend(handles=_handles, labels=_labels, loc="upper right", ncol=max(len(_labels), 1))
    fig.suptitle(f"{saving_folder} {site_label} {title_info} ({date.today()})", weight='bold')
    fig.tight_layout()

    safe_site = re.sub(r"[^\w\-]", "_", site_label)
    if save_pdf:
        out = os.path.join(saving_path, saving_folder,
                           f"{saving_folder}_{safe_site}_{data_type}_{saving_info}.pdf")
        plt.savefig(out)
        print(f"Saved PDF: {out}")
    if save_png:
        out = os.path.join(saving_path, saving_folder,
                           f"{saving_folder}_{safe_site}_{data_type}_{saving_info}.png")
        plt.savefig(out)
        print(f"Saved PNG: {out}")
    if not save_pdf and not save_png:
        print(f"{saving_folder}_{safe_site}_{data_type}_{saving_info} — plot not saved")

    if plot_close:
        plt.close(fig)

#-------------------------
# Plotting functions
#-------------------------

def plot_protein_phosphosites(df,
                              data_type="",
                              proteins=None,
                              cell_lines=[],
                              conditions=[],
                              legend_plot=None,
                              color_palette={"_EGF_": "red",
                                             "_INS_": "blue",
                                             "_EGFnINS_": "fuchsia"},
                              saving_path="",
                              saving_info="",
                              title_info="",
                              fit_y_lims=False,
                              one_figure_per_site=False,
                              plot_close=False,
                              save_pdf=False,
                              save_png=False,):
    """Plot phosphosites for a list of proteins.

    When one_figure_per_site=False (default), all sites for a protein are
    arranged as subplots in one figure. When True, each site gets its own
    standalone figure.
    """

    if proteins is None:
        proteins = []
    if legend_plot is None:
        legend_plot = []

    if isinstance(df, pd.DataFrame):
        pass
    elif isinstance(df, str):
        if df.endswith(".xlsx"):
            df = pd.read_excel(df)
        elif df.endswith(".tsv"):
            df = pd.read_csv(df, sep="\t")
        else:
            raise ValueError("Unsupported file format. Use .xlsx or .tsv")

    sub_dtp = data_type.split(":")

    x_axis = ColumnSpec.timepoints_from(df = df, cell_line=cell_lines, data_type= data_type, condition=conditions)
    len_x = len(x_axis)

    for protein in proteins:
        if protein in df['protein_name'].to_list():
            sub_df = df.loc[df['protein_name'] == protein].copy()
        elif protein in df['protein_Id'].to_list():
            sub_df = df.loc[df['protein_Id'] == protein].copy()
        else:
            print(f"The protein {protein} is not present in the dataset")
            continue
        print(f"Ploting sites of protein {protein}")

        saving_folder = f"{sub_df['protein_name'].iloc[0]}_{sub_df['protein_Id'].iloc[0]}"

        if (save_pdf or save_png) and saving_path:
            new_path = os.path.join(saving_path, saving_folder)
            os.makedirs(new_path, exist_ok=True)

        sub_df.sort_values(by=['site'], inplace=True)

        number_phos = len(sub_df)
        sqrt_n_p = int(np.ceil(np.sqrt(number_phos)))
        sqrt_n_p_X = sqrt_n_p

        if sqrt_n_p > 2:
            empty_plots = (sqrt_n_p * sqrt_n_p) - number_phos
            if empty_plots >= sqrt_n_p:
                sqrt_n_p_X = sqrt_n_p - 1

        column_selection = ColumnSpec.select(df = df, cell_lines=cell_lines, data_type=data_type, conditions=conditions,)

        y_lim_min, y_lim_max = None, None

        if isinstance(fit_y_lims, list):
            y_lim_min, y_lim_max = fit_y_lims[0], fit_y_lims[1]
            use_fixed_ylims = True
        elif fit_y_lims is False:
            sub_values_df = sub_df[column_selection] if column_selection else pd.DataFrame()
            if not sub_values_df.empty:
                y_lim_max = sub_values_df.max().max() * 1.1
                y_lim_min = sub_values_df.min().min()
                y_lim_min = y_lim_min + y_lim_min * 0.1 if y_lim_min < 0 else y_lim_min * 0.97
            use_fixed_ylims = False
        else:
            use_fixed_ylims = None

        shared_kwargs = dict(
            data_type=data_type,
            cell_lines=cell_lines,
            conditions=conditions,
            color_palette=color_palette,
            legend_plot=legend_plot,
            len_x=len_x,
            column_selection=column_selection,
            use_fixed_ylims=use_fixed_ylims,
            y_lim_min=y_lim_min,
            y_lim_max=y_lim_max,
            saving_folder=saving_folder,
            saving_path=saving_path,
            saving_info=saving_info,
            title_info=title_info,
            save_pdf=save_pdf,
            save_png=save_png,
            plot_close=plot_close,
        )

        if one_figure_per_site:
            for k in range(number_phos):
                plot_single_phosphosite(row=sub_df.iloc[k], **shared_kwargs)
        else:
            fig, axes = plt.subplots(sqrt_n_p, sqrt_n_p_X, figsize=(18, 13))
            fig.tight_layout(w_pad=1.75, h_pad=3)
            plt.subplots_adjust(top=0.94)

            if number_phos == 1:
                axes = np.array([[axes]])
            else:
                axes = np.atleast_2d(axes)

            k = 0
            for i in range(sqrt_n_p):
                for j in range(sqrt_n_p_X):
                    if k >= number_phos:
                        fig.delaxes(axes[i, j])
                        continue

                    row = sub_df.iloc[k]

                    plot_data(ax=axes[i, j],
                              row_df=row,
                              data_type=data_type,
                              colors=color_palette,
                              cell_lines=cell_lines,
                              conditions=conditions)

                    if use_fixed_ylims is None:
                        row_vals = row[column_selection].dropna() if column_selection else pd.Series()
                        if not row_vals.empty:
                            rmin, rmax = row_vals.min(), row_vals.max()
                            pad_min = rmin + rmin * 0.1 if rmin < 0 else rmin * 0.9
                            axes[i, j].set_ylim(pad_min, rmax * 1.1)
                    elif use_fixed_ylims and y_lim_min is not None:
                        axes[i, j].set_ylim(y_lim_min, y_lim_max)
                    elif not use_fixed_ylims and y_lim_min is not None:
                        axes[i, j].set_ylim(y_lim_min, y_lim_max)

                    axes[i, j].set_xlim(-1, len_x)
                    k += 1

            _handles, _labels = build_legend(legend_plot, color_palette)
            fig.legend(handles=_handles, labels=_labels, loc="upper right", ncol=max(len(_labels), 1))
            fig.suptitle(f"{saving_folder} {cell_lines} {conditions} {title_info} ({date.today()})", weight='bold')
            fig.tight_layout()

            if save_pdf:
                out = os.path.join(saving_path, saving_folder,
                                   f"{saving_folder}_{data_type}_{saving_info}.pdf")
                plt.savefig(out)
                print(f"Saved PDF: {out}")
            if save_png:
                out = os.path.join(saving_path, saving_folder,
                                   f"{saving_folder}_{data_type}_{saving_info}.png")
                plt.savefig(out)
                print(f"Saved PNG: {out}")
            if not save_pdf and not save_png:
                print(f"{saving_folder}_{data_type}_{saving_info} — plot not saved")

            if plot_close:
                plt.close(fig)


def plot_dataset_phosphosites(df,
                              cluster_column="",
                              cluster_number=None,
                              data_type="",
                              cell_lines=[],
                              conditions=[],
                              legend_plot=None,
                              color_palette={"_EGF_": "red",
                                             "_INS_": "blue",
                                             "_EGFnINS_": "fuchsia"},
                              saving_path="",
                              dataset_name="",
                              saving_info="",
                              title_info="",
                              fit_y_lims=False,
                              one_figure_per_site=False,
                              plot_close=False,
                              save_pdf=False,
                              save_png=False):
    """Plot all phosphorylation sites in a dataset, optionally filtered by cluster.

    When one_figure_per_site=False (default), all sites are arranged as subplots
    in one figure. When True, each site gets its own standalone figure via
    plot_single_phosphosite().
    """

    if legend_plot is None:
        legend_plot = []

    if isinstance(df, pd.DataFrame):
        pass
    elif isinstance(df, str):
        if df.endswith(".xlsx"):
            df = pd.read_excel(df)
        elif df.endswith(".tsv"):
            df = pd.read_csv(df, sep="\t")
        else:
            raise ValueError("Unsupported file format. Use .xlsx or .tsv")

    if cluster_column and cluster_number is not None:
        df = df.loc[df[cluster_column] == int(cluster_number)].copy()

    df = df.sort_values(by=['site'])

    x_axis = ColumnSpec.timepoints_from(df=df, cell_line=cell_lines, data_type=data_type, condition=conditions)
    len_x = len(x_axis)

    column_selection = ColumnSpec.select(df=df, cell_lines=cell_lines, data_type=data_type, conditions=conditions)

    number_phos = len(df)
    sqrt_n_p = int(np.ceil(np.sqrt(number_phos)))
    sqrt_n_p_X = sqrt_n_p

    if sqrt_n_p > 2:
        empty_plots = (sqrt_n_p * sqrt_n_p) - number_phos
        if empty_plots >= sqrt_n_p:
            sqrt_n_p_X = sqrt_n_p - 1

    y_lim_min, y_lim_max = None, None

    if isinstance(fit_y_lims, list):
        y_lim_min, y_lim_max = fit_y_lims[0], fit_y_lims[1]
        use_fixed_ylims = True
    elif fit_y_lims is False:
        sub_values_df = df[column_selection] if column_selection else pd.DataFrame()
        if not sub_values_df.empty:
            y_lim_max = sub_values_df.max().max() * 1.1
            y_lim_min = sub_values_df.min().min()
            y_lim_min = y_lim_min + y_lim_min * 0.1 if y_lim_min < 0 else y_lim_min * 0.97
        use_fixed_ylims = False
    else:
        use_fixed_ylims = None

    if (save_pdf or save_png) and saving_path:
        os.makedirs(saving_path, exist_ok=True)

    shared_kwargs = dict(
        data_type=data_type,
        cell_lines=cell_lines,
        conditions=conditions,
        color_palette=color_palette,
        legend_plot=legend_plot,
        len_x=len_x,
        column_selection=column_selection,
        use_fixed_ylims=use_fixed_ylims,
        y_lim_min=y_lim_min,
        y_lim_max=y_lim_max,
        saving_folder=dataset_name,
        saving_path=saving_path,
        saving_info=saving_info,
        title_info=title_info,
        save_pdf=save_pdf,
        save_png=save_png,
        plot_close=plot_close,
    )

    if one_figure_per_site:
        for k in range(number_phos):
            plot_single_phosphosite(row=df.iloc[k], **shared_kwargs)
    else:
        fig, axes = plt.subplots(sqrt_n_p, sqrt_n_p_X, figsize=(18, 13))
        fig.tight_layout(w_pad=1.75, h_pad=3)
        plt.subplots_adjust(top=0.94)

        if number_phos == 1:
            axes = np.array([[axes]])
        else:
            axes = np.atleast_2d(axes)

        k = 0
        for i in range(sqrt_n_p):
            for j in range(sqrt_n_p_X):
                if k >= number_phos:
                    fig.delaxes(axes[i, j])
                    continue

                row = df.iloc[k]

                plot_data(ax=axes[i, j],
                          row_df=row,
                          data_type=data_type,
                          colors=color_palette,
                          cell_lines=cell_lines,
                          conditions=conditions)

                if use_fixed_ylims is None:
                    row_vals = row[column_selection].dropna() if column_selection else pd.Series()
                    if not row_vals.empty:
                        rmin, rmax = row_vals.min(), row_vals.max()
                        pad_min = rmin + rmin * 0.1 if rmin < 0 else rmin * 0.9
                        axes[i, j].set_ylim(pad_min, rmax * 1.1)
                elif use_fixed_ylims and y_lim_min is not None:
                    axes[i, j].set_ylim(y_lim_min, y_lim_max)
                elif not use_fixed_ylims and y_lim_min is not None:
                    axes[i, j].set_ylim(y_lim_min, y_lim_max)

                axes[i, j].set_xlim(-1, len_x)
                k += 1

        _handles, _labels = build_legend(legend_plot, color_palette)
        fig.legend(handles=_handles, labels=_labels, loc="upper right", ncol=max(len(_labels), 1))
        fig.suptitle(
            f"{dataset_name} {cluster_column} {cluster_number} {title_info} ({date.today()})",
            weight='bold'
        )
        fig.tight_layout()

        if save_pdf:
            out = os.path.join(saving_path,
                               f"{dataset_name}_{cluster_column}_{cluster_number}_{data_type}_{saving_info}.pdf")
            plt.savefig(out)
            print(f"Saved PDF: {out}")
        if save_png:
            out = os.path.join(saving_path,
                               f"{dataset_name}_{cluster_column}_{cluster_number}_{data_type}_{saving_info}.png")
            plt.savefig(out)
            print(f"Saved PNG: {out}")
        if not save_pdf and not save_png:
            print(f"{dataset_name}_{cluster_column}_{cluster_number}_{data_type}_{saving_info} — plot not saved")

        if plot_close:
            plt.close(fig)


def plot_protein_profile(df,
                         proteins,
                         data_type="",
                         cell_lines=[],
                         conditions=["_EGF_", "_INS_", "_EGFnINS_"],
                         panel_by="condition",
                         saving_path="",
                         saving_info="",
                         legend=False,
                         save_pdf=False,
                         save_png=False):
    """Plot per-protein profiles as overlaid thin lines.

    panel_by="condition" (default): len(proteins) rows × len(conditions) cols.
        Each panel overlays all sites for that condition across all cell_lines.
    panel_by="cell_line": len(proteins) rows × len(cell_lines) cols.
        Each panel overlays all sites for conditions[0] in that cell line.
        Use when comparing mutant cell lines side-by-side.
    Y-axis is shared per protein row across all panels.
    """

    if not isinstance(df, pd.DataFrame):
        if df.endswith(".xlsx"):
            df = pd.read_excel(df)
        elif df.endswith(".tsv"):
            df = pd.read_csv(df, sep="\t")
        else:
            raise ValueError("Unsupported file format. Use .xlsx or .tsv")

    if (save_pdf or save_png) and saving_path:
        os.makedirs(saving_path, exist_ok=True)

    if panel_by == "condition":
        n_cols = len(conditions)
        x_axis = ColumnSpec.timepoints_from(df=df, cell_line=cell_lines, data_type=data_type, condition=conditions)
        panel_cols = {
            cond: ColumnSpec.select(df=df, cell_lines=cell_lines, data_type=data_type, conditions=[cond])
            for cond in conditions
        }
        all_cols = ColumnSpec.select(df=df, cell_lines=cell_lines, data_type=data_type, conditions=conditions)
        panel_keys = conditions
        panel_label = lambda k: k.strip("_")
    else:  # panel_by == "cell_line"
        n_cols = len(cell_lines)
        ref_cond = conditions[0]
        x_axis = ColumnSpec.timepoints_from(df=df, cell_line=cell_lines, data_type=data_type, condition=[ref_cond])
        panel_cols = {
            cell: ColumnSpec.select(df=df, cell_lines=[cell], data_type=data_type, conditions=[ref_cond])
            for cell in cell_lines
        }
        all_cols = ColumnSpec.select(df=df, cell_lines=cell_lines, data_type=data_type, conditions=[ref_cond])
        panel_keys = cell_lines
        panel_label = lambda k: k

    fig, ax = plt.subplots(len(proteins), n_cols,
                           figsize=(5 * n_cols, 3 * len(proteins)),
                           squeeze=False)

    for c, protein in enumerate(proteins):
        if protein in df['protein_name'].values:
            sub_df = df[df['protein_name'] == protein].copy()
        elif protein in df['protein_Id'].values:
            sub_df = df[df['protein_Id'] == protein].copy()
        else:
            print(f"The protein {protein} is not present in the dataset.")
            continue

        protein_id = str(sub_df['protein_Id'].values[0])
        prot_name = str(sub_df['protein_name'].values[0])
        saving_folder = f"{prot_name}_{protein_id}"
        sub_df.sort_values(by=['site'], inplace=True)

        sub_values_df = sub_df[all_cols] if all_cols else pd.DataFrame()
        if not sub_values_df.empty:
            y_max = sub_values_df.max().max() * 1.05 + 0.1
            y_min_val = sub_values_df.min().min()
            y_min = y_min_val * 0.95 - 0.1 if y_min_val >= 0 else -abs(y_min_val) * 1.05 - 0.1
        else:
            y_min, y_max = None, None

        for d_idx, key in enumerate(panel_keys):
            cols = panel_cols[key]
            for _, row in sub_df.iterrows():
                ax[c][d_idx].plot(x_axis, row[cols])
            ax[c][d_idx].set_title(panel_label(key))
            ax[c][d_idx].axhline(0, color='black', linestyle='--', linewidth=0.5)
            if y_min is not None:
                ax[c][d_idx].set_ylim(y_min, y_max)
        ax[c][0].set_ylabel(f"{saving_folder}\n{data_type}", weight='bold')

    if legend:
        fig.legend(labels=df["site"].unique())

    fig.tight_layout()

    if save_pdf:
        out = os.path.join(saving_path, f"{saving_info}.pdf")
        plt.savefig(out)
        print(f"Saved PDF: {out}")
    if save_png:
        out = os.path.join(saving_path, f"{saving_info}.png")
        plt.savefig(out)
        print(f"Saved PNG: {out}")
    if not save_pdf and not save_png:
        print(f"{saving_info} — plot not saved")

    plt.show()

def plot_volcano(df,
                 fc_col,
                 pval_col,
                 fc_thresh=1.0,
                 pval_thresh=0.05,
                 precomputed=False,
                 title=None,
                 ax=None,
                 highlight_proteins=None,
                 match_cols=("protein_Id", "protein_name"),
                 case_insensitive=True,
                 fit_x_limit=False):
    """Volcano plot with optional multi-protein highlighting.

    Args:
        df: DataFrame with phosphosite data.
        fc_col: column name for log2 fold change.
        pval_col: column name for p-values. When precomputed=False (default)
            the column contains raw p-values and the function computes
            -log10 internally. When precomputed=True the column already
            contains -log10(p-value) and is used directly.
        fc_thresh: |log2FC| cutoff for the vertical threshold lines.
        pval_thresh: p-value cutoff (always expressed as a raw p-value, e.g.
            0.05). The function converts it to -log10 for the threshold line
            and significance check regardless of precomputed mode.
        precomputed: if True, pval_col is treated as already -log10-transformed.
        title: plot title; defaults to "Volcano: {fc_col}".
        ax: existing Axes to draw into; creates a new figure if None.
        highlight_proteins: single protein or list of proteins to highlight by
            name or UniProt ID. Each protein gets a distinct colour.
        match_cols: tuple of columns to search when matching highlight_proteins.
        case_insensitive: ignore case when matching protein names/IDs.
        fit_x_limit: False = auto; True = [-6, 6]; list [min, max] = custom range.
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        created_fig = True

    log2fc = df[fc_col]
    threshold_y = -np.log10(pval_thresh)

    if precomputed:
        neg_log10_pval = df[pval_col]
    else:
        raw_pvals = np.where(np.asarray(df[pval_col], dtype=float) <= 0, np.nan, df[pval_col])
        neg_log10_pval = -np.log10(raw_pvals)

    sig = (np.abs(log2fc) >= fc_thresh) & (neg_log10_pval >= threshold_y)

    ax.scatter(log2fc[~sig], neg_log10_pval[~sig], color="grey", alpha=0.6, s=20, label="not significant")
    ax.scatter(log2fc[sig], neg_log10_pval[sig], color="red", alpha=0.8, s=30, label="significant")

    ax.axhline(threshold_y, color="blue", linestyle="--", linewidth=1)
    ax.axvline(-fc_thresh, color="blue", linestyle="--", linewidth=1)
    ax.axvline(fc_thresh, color="blue", linestyle="--", linewidth=1)

    if highlight_proteins is not None:
        highlight_list = [highlight_proteins] if isinstance(highlight_proteins, str) else list(highlight_proteins)

        comp_cols = [c for c in match_cols if c in df.columns]
        norm_cols = {
            c: df[c].astype(str).str.lower() if case_insensitive else df[c].astype(str)
            for c in comp_cols
        }
        color_cycle = cycle(
            plt.cm.tab10.colors if hasattr(plt.cm, "tab10") else
            ["gold", "cyan", "magenta", "yellow", "green", "blue"]
        )

        for item in highlight_list:
            if item is None:
                continue
            item_key = item.lower() if case_insensitive else str(item)
            mask = np.zeros(len(df), dtype=bool)
            for c in comp_cols:
                mask |= (norm_cols[c] == item_key)
            if mask.any():
                ax.scatter(
                    log2fc[mask], neg_log10_pval[mask],
                    s=90, marker="o", facecolor=next(color_cycle), edgecolor="black",
                    linewidth=0.8, alpha=0.95, zorder=3,
                    label=f"Highlight: {item}"
                )

    ax.set_xlabel(f"{fc_col}") # "log2 Fold Change")
    ax.set_ylabel(f"{pval_col}\n(-log10)") # "-log10(p-value)")
    ax.set_title(title or f"Volcano: {fc_col}")
    ax.legend(fontsize=8, frameon=False)

    if fit_x_limit is not False:
        if fit_x_limit is True:
            ax.set_xlim(-6, 6)
        else:
            ax.set_xlim(fit_x_limit[0], fit_x_limit[1])

    if created_fig:
        plt.tight_layout()
        plt.show()


def plot_volcano_interactive(df,
                             fc_col,
                             pval_col,
                             site_col="site",
                             fc_thresh=1.0,
                             pval_thresh=0.05,
                             precomputed=False,
                             title=None,
                             highlight_proteins=None,
                             match_cols=("protein_Id", "protein_name"),
                             case_insensitive=True,
                             show_highlight_labels=False):
    """Interactive volcano plot (Plotly) with optional multi-protein highlighting.

    Args:
        df: DataFrame with phosphosite data.
        fc_col: column name for log2 fold change.
        pval_col: column name for p-values. When precomputed=False (default)
            the column contains raw p-values and the function computes
            -log10 internally. When precomputed=True the column already
            contains -log10(p-value) and is used directly.
        site_col: column name used for hover labels.
        fc_thresh: |log2FC| cutoff for the vertical threshold lines.
        pval_thresh: p-value cutoff (always expressed as a raw p-value, e.g.
            0.05). Converted to -log10 for threshold line and significance
            check regardless of precomputed mode.
        precomputed: if True, pval_col is treated as already -log10-transformed.
        title: plot title; defaults to "Volcano: {fc_col}".
        highlight_proteins: single protein or list of proteins to highlight by
            name or UniProt ID. Each protein gets a distinct colour.
        match_cols: tuple of columns to search when matching highlight_proteins.
        case_insensitive: ignore case when matching protein names/IDs.
        show_highlight_labels: if True, annotate highlighted points with the
            protein name or ID.

    Returns:
        plotly.graph_objects.Figure
    """
    d = df[[fc_col, pval_col]].copy()
    if site_col in df.columns:
        d[site_col] = df[site_col]
    for c in match_cols:
        if c in df.columns:
            d[c] = df[c]

    threshold_y = -np.log10(pval_thresh)

    if precomputed:
        d["neglog10p"] = pd.to_numeric(d[pval_col], errors="coerce")
    else:
        d[pval_col] = pd.to_numeric(d[pval_col], errors="coerce")
        d.loc[d[pval_col] <= 0, pval_col] = np.nan
        d["neglog10p"] = -np.log10(d[pval_col])

    d["signif"] = np.where(
        (d[fc_col].abs() >= fc_thresh) & (d["neglog10p"] >= threshold_y),
        "significant",
        "not significant",
    )

    hover_data = {
        fc_col: ":.3f",
        "neglog10p": ":.3f",
        **({"protein_Id": True} if "protein_Id" in d.columns else {}),
        **({"protein_name": True} if "protein_name" in d.columns else {}),
    }
    if not precomputed:
        hover_data[pval_col] = ":.3g"

    fig = px.scatter(
        d,
        x=fc_col,
        y="neglog10p",
        color="signif",
        opacity=0.3,
        hover_name=site_col if site_col in d.columns else None,
        hover_data=hover_data,
        title=title or f"Volcano: {fc_col}",
        template="plotly_white",
    )

    fig.add_hline(y=threshold_y, line_dash="dash", line_color="gray")
    fig.add_vline(x=fc_thresh, line_dash="dash", line_color="gray")
    fig.add_vline(x=-fc_thresh, line_dash="dash", line_color="gray")

    fig.update_layout(
        xaxis_title= f"{fc_col}", #"log2 Fold Change",
        yaxis_title= f"{pval_col}\n(-log10)", #"-log10(p-value)",
        legend_title="",
    )

    if highlight_proteins is not None:
        highlight_list = [highlight_proteins] if isinstance(highlight_proteins, str) else list(highlight_proteins)

        comp_cols = [c for c in match_cols if c in d.columns]
        norm_df = {
            c: d[c].astype(str).str.lower() if case_insensitive else d[c].astype(str)
            for c in comp_cols
        }

        palette = px.colors.qualitative.D3
        color_idx = 0

        for item in highlight_list:
            if item is None:
                continue
            key = str(item).lower() if case_insensitive else str(item)

            mask = np.zeros(len(d), dtype=bool)
            for c in comp_cols:
                mask |= (norm_df[c] == key)

            if not mask.any():
                continue

            color = palette[color_idx % len(palette)]
            color_idx += 1

            if show_highlight_labels:
                if "protein_name" in d.columns:
                    text_vals = d.loc[mask, "protein_name"].astype(str)
                elif "protein_Id" in d.columns:
                    text_vals = d.loc[mask, "protein_Id"].astype(str)
                else:
                    text_vals = (d.loc[mask, site_col].astype(str)
                                 if site_col in d.columns else pd.Series([""] * mask.sum()))
            else:
                text_vals = None

            fig.add_trace(go.Scatter(
                x=d.loc[mask, fc_col],
                y=d.loc[mask, "neglog10p"],
                mode="markers+text" if show_highlight_labels else "markers",
                text=text_vals if show_highlight_labels else None,
                textposition="top center",
                marker=dict(
                    size=11,
                    color=color,
                    line=dict(width=1.2, color="black"),
                    opacity=1,
                    symbol="circle",
                ),
                name=f"Highlight: {item}",
                hovertemplate=(
                    f"<b>{item}</b><br>"
                    f"log2FC: %{{x:.3f}}<br>"
                    f"-log10(p): %{{y:.3f}}<br>"
                    + (f"{site_col}: %{{customdata[0]}}<br>" if site_col in d.columns else "")
                ),
                customdata=d.loc[mask, [site_col]].values if site_col in d.columns else None,
            ))

    return fig

##########################################################
# Latest additions
##########################################################
def clusters_plot_linear_mutants(
    df,
    legend=None,
    saving_path="",
    cluster_column="",
    cluster_name="",
    data_type="",
    cell_lines=None,
    conditions=None,
    colors=None,
    panel_by="cell_line",
    plot_different_data=False,
    saving_info="",
    save_pdf=False,
    save_png=False,
    plot_close=False,
    fit_y_lims=False,
    grey_alpha=0.08,
    grey_lw=0.8,
    mean_lw=2.6):
    """
    Plot cluster time-series profiles in a grid layout.

    panel_by="cell_line" (default): grid is n_clusters × n_cell_lines.
        Each column shows one cell line; mean line color keyed by cell_lines entry in `colors`.
        Use case: multiple mutant cell lines, one (or few) conditions.

    panel_by="condition": grid is n_clusters × n_conditions.
        Each column shows one condition; mean line color keyed by condition string in `colors`.
        Use case: single cell line, multiple stimulation conditions (EGF / INS / EGFnINS).

    Args:
        df: DataFrame or path to .xlsx file.
        legend: unused placeholder kept for API compatibility.
        saving_path: directory for output files.
        cluster_column: name of the column holding cluster labels.
        cluster_name: string appended to the figure title / file name.
        data_type: data type string, e.g. "log2:FC".
        cell_lines: list of cell-line prefixes, e.g. ["WT", "BRAFS151A"].
        conditions: list of condition substrings, e.g. ["_EGF_", "_INS_"].
        colors: dict mapping cell-line name (panel_by="cell_line") or condition
                string (panel_by="condition") to a matplotlib colour.
        panel_by: "cell_line" or "condition" — controls which dimension forms columns.
        plot_different_data: set True to suppress the data_type/cluster_column mismatch warning.
        saving_info: extra string appended to saved file names.
        save_pdf / save_png: whether to save output files.
        plot_close: if True, close the figure after saving.
        fit_y_lims: False → shared y-limits from all cluster values;
                    list [min, max] → fixed limits;
                    True → per-row dynamic limits (not yet wired; treated as per-cluster shared).
        grey_alpha / grey_lw: appearance of individual site lines.
        mean_lw: line width for the mean curve.
    """
    if cell_lines is None:
        cell_lines = []
    if conditions is None:
        conditions = []
    if colors is None:
        colors = {}

    if not isinstance(df, pd.DataFrame):
        df = pd.read_excel(df)

    if save_pdf or save_png:
        os.makedirs(saving_path, exist_ok=True)

    clusters = list(set(df[cluster_column]))
    if 999 in clusters:
        clusters.remove(999)

    if data_type not in cluster_column and not plot_different_data:
        print("Remember to plot the same data_type used for clustering, or set plot_different_data=True")
        return

    sorted_clusters = sorted(clusters) if isinstance(clusters[0], int) else \
        sorted(clusters, key=lambda x: int(x.split()[1]))
    n_cluster = len(sorted_clusters)

    # --- Derive x-axis time points via ColumnSpec ---
    time_points = ColumnSpec.timepoints_from(df, cell_line=cell_lines, data_type=data_type, condition=conditions)

    # --- Build per-panel column lookup using ColumnSpec ---
    if panel_by == "cell_line":
        panels = cell_lines
        def _panel_cols(panel):
            return ColumnSpec.select(df, cell_lines=[panel], data_type=data_type, conditions=conditions)
        def _panel_title(cluster, panel, n):
            return f"Cluster {cluster} | {panel} (n={n} sites)"
    elif panel_by == "condition":
        panels = conditions
        def _panel_cols(panel):
            return ColumnSpec.select(df, cell_lines=cell_lines, data_type=data_type, conditions=[panel])
        def _panel_title(cluster, panel, n):
            return f"Cluster {cluster} | {panel.strip('_')} (n={n} sites)"
    else:
        raise ValueError(f"panel_by must be 'cell_line' or 'condition', got {panel_by!r}")

    # All columns used (for shared y-limit calculation)
    all_cols = ColumnSpec.select(df, cell_lines=cell_lines, data_type=data_type, conditions=conditions)

    fig, axes = plt.subplots(
        nrows=n_cluster,
        ncols=len(panels),
        figsize=(6 * len(panels), max(3.2 * n_cluster, 6)),
        squeeze=False
    )

    for r, cluster in enumerate(sorted_clusters):
        sub_df = df.loc[df[cluster_column] == cluster].copy()
        if sub_df.shape[0] == 0:
            continue

        # Resolve shared y-limits for this cluster row
        if isinstance(fit_y_lims, list):
            y_lim_min, y_lim_max = fit_y_lims[0], fit_y_lims[1]
            y_limt_info = f"_y_axis_fixed_{y_lim_min}_{y_lim_max}"
        elif fit_y_lims is False:
            sub_vals = sub_df[all_cols].to_numpy(dtype=float) if all_cols else np.array([])
            if sub_vals.size:
                y_lim_max = np.nanmax(sub_vals) * 1.02
                raw_min = np.nanmin(sub_vals)
                y_lim_min = raw_min + raw_min * 0.1 if raw_min < 0 else raw_min * 0.97
            else:
                y_lim_min, y_lim_max = None, None
            y_limt_info = ""
        else:
            y_lim_min, y_lim_max = None, None
            y_limt_info = "y_axis_perrow"

        for c, panel in enumerate(panels):
            ax = axes[r, c]
            color = colors.get(panel, "steelblue")
            panel_cols = _panel_cols(panel)

            if not panel_cols:
                ax.set_visible(False)
                continue

            mat = sub_df[panel_cols].to_numpy(dtype=float)
            for i in range(mat.shape[0]):
                ax.plot(time_points, mat[i, :], color="grey", alpha=grey_alpha, linewidth=grey_lw)

            mean_curve = np.nanmean(mat, axis=0)
            ax.plot(time_points, mean_curve, color=color, linewidth=mean_lw)

            ax.axhline(0, color="grey", linestyle="--", linewidth=1)
            ax.grid(True, alpha=0.3)

            if fit_y_lims is True:  # per-panel dynamic limits
                flat = mat.flatten()
                flat = flat[~np.isnan(flat)]
                if flat.size:
                    rmin, rmax = flat.min(), flat.max()
                    ax.set_ylim(rmin + rmin * 0.1 if rmin < 0 else rmin * 0.97, rmax * 1.02)
            elif y_lim_min is not None:
                ax.set_ylim(y_lim_min, y_lim_max)

            ax.set_xlim(-1, len(time_points))
            ax.set_xticks(range(len(time_points)))
            ax.set_xticklabels([str(t) for t in time_points])

            if r == n_cluster - 1:
                ax.set_xlabel("Time (min)")
            ax.set_ylabel(data_type if c == 0 else "")
            ax.set_title(_panel_title(cluster, panel, sub_df.shape[0]))

    # Legend keyed by panels
    handles = [
        plt.Line2D([0], [0], color=colors.get(p, "steelblue"), lw=mean_lw, label=str(p).strip("_"))
        for p in panels
    ] + [plt.Line2D([0], [0], color="grey", lw=grey_lw, alpha=0.4, label="Individual sites")]
    fig.legend(handles=handles, loc="upper right", ncol=1)

    fig.suptitle(f"{cluster_column} {cluster_name} {date.today()}", weight="bold")
    fig.tight_layout(rect=[0, 0, 0.88, 0.97])

    if save_pdf:
        plt.savefig(f"{saving_path}/{cluster_name}{saving_info}.pdf", bbox_inches="tight")
        print(f"{cluster_name}{saving_info} Plot saved as PDF")
    if save_png:
        plt.savefig(f"{saving_path}/{cluster_name}{saving_info}.png", bbox_inches="tight", dpi=300)
        print(f"{cluster_name}{saving_info} Plot saved as PNG")
    if not save_pdf and not save_png:
        print(f"{cluster_name}{saving_info} Plot not saved")

    if plot_close:
        plt.close(fig)


def plot_cluster_scores(
    scores,
    cond_order=("EGF", "INS", "EGFnINS"),
    figsize=(14, 5),
    title="Cluster quality per condition",
    ylabel="Dispersion score (lower = tighter)",
    ax=None,
):
    """
    Grouped bar chart of per-cluster, per-condition dispersion scores.

    Typically used to visualise the output of cluster_similarity_per_condition()
    from src/clustering.py — lower bars mean tighter (more homogeneous) clusters.

    Args:
        scores: dict mapping cluster label → {condition_label: scalar}, e.g.
                {0: {"EGF": 0.4, "INS": 0.6, "EGFnINS": 0.5}, 1: ...}.
        cond_order: tuple of condition labels to plot (default all three stimulations).
        figsize: (width, height); ignored when ax is provided.
        title: axes title.
        ylabel: y-axis label.
        ax: optional Axes; a new figure is created when None (default).

    Returns:
        fig, ax
    """
    clusters = sorted(scores.keys())
    x = np.arange(len(clusters))
    width = 0.8 / len(cond_order)

    create_fig = ax is None
    if create_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for i, cond in enumerate(cond_order):
        y = [scores[c].get(cond, float("nan")) for c in clusters]
        offset = (i - (len(cond_order) - 1) / 2) * width
        ax.bar(x + offset, y, width=width, label=cond)

    ax.set_xticks(x)
    ax.set_xticklabels(clusters, rotation=90)
    ax.set_xlabel("Cluster")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    if create_fig:
        fig.tight_layout()
    return fig, ax


def clusters_shared_sites(cluster_df,
                          clustering_1,
                          clustering_2,
                          site=None,
                          clusters=None,):
    """
    Plot a Venn diagram of phosphosites shared between two cluster assignments.

    Provide either `site` (to look up which clusters it belongs to in each assignment)
    or explicit `clusters=[cluster1_id, cluster2_id]`.

    Args:
        cluster_df: DataFrame with a 'site' column and at least two cluster label columns.
        clustering_1: name of the first cluster label column.
        clustering_2: name of the second cluster label column.
        site: site identifier string; if given, cluster IDs are inferred automatically.
        clusters: list [cluster1_id, cluster2_id]; used when `site` is not provided.

    Returns:
        fig, ax
    """
    if clusters is None:
        clusters = [None, None]

    if site is not None:
        row = cluster_df.loc[cluster_df["site"] == site, [clustering_1, clustering_2]]
        if row.empty:
            raise ValueError(f"Site '{site}' not found in cluster_df.")
        cluster1_id, cluster2_id = row.iloc[0][clustering_1], row.iloc[0][clustering_2]
    else:
        if len(clusters) != 2 or None in clusters:
            raise ValueError("Provide `site` or both cluster IDs via clusters=[id1, id2].")
        cluster1_id, cluster2_id = clusters

    set_1 = set(cluster_df.loc[cluster_df[clustering_1] == cluster1_id, "site"])
    set_2 = set(cluster_df.loc[cluster_df[clustering_2] == cluster2_id, "site"])

    fig, ax = plt.subplots(figsize=(6, 4))
    venn2(
        [set_1, set_2],
        set_labels=(f"{clustering_1}\nCluster {cluster1_id}", f"{clustering_2}\nCluster {cluster2_id}"),
        ax=ax,
    )
    ax.set_title("Shared phosphosites between cluster assignments")
    return fig, ax


def plot_cluster_assignment_qc(barycenters, labels, figsize=(14, 10), umap_random_state=42):
    """
    Four-panel figure assessing cluster assignment confidence after fit_transform().

    Requires barycenters — the (n_sites × n_clusters) distance-to-centroid matrix
    returned by tslearn_clustering_KMeans(..., testing=True, barycenter_calculations=True).

    Panels:
      - Top-left:  Heatmap of distances to all centroids, rows sorted by assigned label.
                   A well-separated clustering shows a clear minimum along the diagonal.
      - Top-right: UMAP of the distance-to-centroid matrix, coloured by cluster label.
                   Sites that cluster well form tight, distinct islands.
      - Bottom-left:  Histogram of assignment margin (2nd-best − best distance).
                      Low margin (<0.3) → weak separation; >0.5 → confident assignment.
      - Bottom-right: Boxplot of assignment margin per cluster.
                      Clusters with low median margin are worth inspecting or merging.

    Args:
        barycenters: np.ndarray of shape (n_sites, n_clusters) — distance-to-centroid
                     matrix from fit_transform().
        labels: array-like of cluster labels, length n_sites.
        figsize: (width, height) of the combined figure (default (14, 10)).
        umap_random_state: random seed for UMAP (default 42).

    Returns:
        fig, axes — the Figure and a (2, 2) array of Axes.
    """
    import umap as umap_lib

    labels = np.asarray(labels)
    n_clusters = barycenters.shape[1]

    margin = np.sort(barycenters, axis=1)[:, 1] - np.sort(barycenters, axis=1)[:, 0]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # ── Top-left: distance heatmap ───────────────────────────────────────────
    ax = axes[0, 0]
    order = np.argsort(labels)
    im = ax.imshow(barycenters[order], aspect="auto", cmap="viridis_r")
    fig.colorbar(im, ax=ax, label="DTW distance to centroid")
    ax.set_xlabel("Centroid index")
    ax.set_ylabel("Sites (sorted by assigned cluster)")
    ax.set_title("Distance-to-centroid heatmap")

    # ── Top-right: UMAP ──────────────────────────────────────────────────────
    ax = axes[0, 1]
    Z = umap_lib.UMAP(n_components=2, random_state=umap_random_state).fit_transform(barycenters)
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=labels, alpha=0.5, s=10, cmap="tab20")
    fig.colorbar(sc, ax=ax, label="Cluster")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP of distance-to-centroid representation")

    # ── Bottom-left: margin histogram ────────────────────────────────────────
    ax = axes[1, 0]
    ax.hist(margin, bins=40, color="#4e79a7", edgecolor="white", linewidth=0.4)
    ax.axvline(0.3, color="orange", linestyle="--", linewidth=1.2, label="0.3 (weak)")
    ax.axvline(0.5, color="green",  linestyle="--", linewidth=1.2, label="0.5 (confident)")
    ax.set_xlabel("Margin = 2nd-best − best distance")
    ax.set_ylabel("Number of sites")
    ax.set_title("Cluster assignment margin distribution")
    ax.legend(fontsize=8)

    # ── Bottom-right: margin per cluster (boxplot) ───────────────────────────
    ax = axes[1, 1]
    data_per_cluster = [margin[labels == k] for k in range(n_clusters)]
    ax.boxplot(data_per_cluster, patch_artist=True,
               boxprops=dict(facecolor="#a0cbe8", alpha=0.7),
               medianprops=dict(color="navy", linewidth=1.5),
               flierprops=dict(marker=".", markersize=2, alpha=0.3))
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Margin")
    ax.set_title("Assignment margin per cluster")
    ax.set_xticks(range(1, n_clusters + 1))
    ax.set_xticklabels(range(n_clusters), fontsize=max(5, 8 - n_clusters // 10))

    fig.tight_layout()
    return fig, axes


def plot_kscan_summary(ks, inertias, silhouettes, stabilities, figsize=(14, 4)):
    """
    Three-panel summary plot for a k-scan (inertia, silhouette, stability vs k).

    Intended to be called after running the k-scan loop in the Clustering notebook.

    Args:
        ks: sequence of k values that were scanned (x-axis for all panels).
        inertias: list of inertia values (one per k); may contain None for skipped ks.
        silhouettes: list of silhouette scores (one per k); may contain None.
        stabilities: list of mean ARI stability scores (one per k); may contain None.
        figsize: (width, height) of the full figure (default (14, 4)).

    Returns:
        fig, axes  — the Figure and a (3,) array of Axes.
    """
    ks = list(ks)

    def _plot_metric(ax, ys, title, ylabel, color):
        valid = [(k, y) for k, y in zip(ks, ys) if y is not None]
        if valid:
            kv, yv = zip(*valid)
            ax.plot(kv, yv, marker="o", color=color, linewidth=1.8, markersize=5)
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticks(ks)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    _plot_metric(axes[0], inertias,    "Inertia (elbow method)",
                 "Within-cluster DTW distance", "#e15759")
    _plot_metric(axes[1], silhouettes, "Silhouette score",
                 "Silhouette (higher = better separation)", "#4e79a7")
    _plot_metric(axes[2], stabilities, "Stability (ARI across seeds)",
                 "Mean pairwise ARI (higher = more stable)", "#59a14f")

    fig.tight_layout()
    return fig, axes


def plot_sites_umap(
        df,
        cell_lines,
        conditions,
        data_type="log2:FC",
        exclude_full=True,
        color_col=None,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        figsize=(8, 7),
        s=8,
        alpha=0.5,
        title=None,
        ax=None,
):
    """
    UMAP embedding of phosphosites using their temporal profile as features.

    Each site is a point in a space defined by its log2:FC (or other data_type) values
    across the selected conditions and timepoints.  Sites with similar temporal dynamics
    cluster together in the 2-D projection.

    Colouring:
      - color_col=None        → all points in a single grey colour.
      - color_col=<string>    → column name in df used to colour points.
          * If the column is numeric → continuous viridis colourbar.
          * If the column is categorical / object → one colour per category with a legend
            (tab20 for up to 20 categories, otherwise a larger qualitative palette).

    Args:
        df: DataFrame following the project naming convention; rows are phosphosites.
        cell_lines: list of cell-line prefixes, e.g. ["WT"].
        conditions: list of condition substrings, e.g. ["_EGF_", "_INS_", "_EGFnINS_"].
        data_type: data-type string used for feature selection (default "log2:FC").
        exclude_full: if True, exclude the 'full' timepoint from the feature matrix
                      (default True).
        color_col: column name in df to colour points by, or None for a uniform colour.
        n_neighbors: UMAP n_neighbors parameter (default 15).
        min_dist: UMAP min_dist parameter (default 0.1).
        random_state: random seed for reproducibility (default 42).
        figsize: (width, height) used only when ax=None (default (8, 7)).
        s: marker size (default 8).
        alpha: marker transparency (default 0.5).
        title: plot title; auto-generated from parameters when None.
        ax: existing Axes to draw on; a new Figure/Axes is created when None.

    Returns:
        fig, ax — the Figure and Axes (fig is None when an external ax was passed).
    """
    import umap as umap_lib
    from src.column_spec import ColumnSpec

    cols = ColumnSpec.select(
        df,
        cell_lines=cell_lines,
        data_type=data_type,
        conditions=conditions,
        exclude_full=exclude_full,
        exclude_replicate_cols=True,
    )
    if not cols:
        raise ValueError(
            f"No columns found for cell_lines={cell_lines}, conditions={conditions}, "
            f"data_type={data_type!r}.  Check naming convention."
        )

    X = df[cols].fillna(0).values

    Z = umap_lib.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    ).fit_transform(X)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    if color_col is None:
        ax.scatter(Z[:, 0], Z[:, 1], c="#aaaaaa", s=s, alpha=alpha, linewidths=0)

    elif pd.api.types.is_numeric_dtype(df[color_col]):
        c_vals = df[color_col].fillna(0).values
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=c_vals, cmap="viridis",
                        s=s, alpha=alpha, linewidths=0)
        plt.colorbar(sc, ax=ax, label=color_col)

    else:
        categories = df[color_col].astype(str)
        unique_cats = sorted(categories.unique())
        n_cats = len(unique_cats)
        cmap = plt.get_cmap("tab20" if n_cats <= 20 else "gist_ncar")
        cat_to_color = {cat: cmap(i / n_cats) for i, cat in enumerate(unique_cats)}

        for cat in unique_cats:
            mask = categories == cat
            ax.scatter(Z[mask, 0], Z[mask, 1],
                       color=cat_to_color[cat], label=str(cat),
                       s=s, alpha=alpha, linewidths=0)

        if n_cats <= 30:
            ax.legend(title=color_col, markerscale=2, fontsize=7,
                      loc="best", framealpha=0.6)

    cond_label = "+".join(c.strip("_") for c in conditions)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title or f"UMAP of phosphosites — {data_type} | {cond_label}")

    if own_fig:
        fig.tight_layout()

    return fig, ax


def plot_cluster_hierarchy(centers,
                           Z,
                           condition_names=None,
                           color_threshold=0.7,
                           figsize_dendro=(15, 4),
                           figsize_grid=None,):
    """
    Visualise the hierarchical structure of KMeans cluster centers.

    Produces two figures:
      1. **Dendrogram** — clusters ordered by DTW similarity of their centroids.
         Colour threshold groups clusters into super-clusters.
      2. **Centroid grid** — one row per cluster (in dendrogram leaf order), one
         column per condition.  Clusters belonging to the same super-cluster share
         the same line colour, so related profiles appear adjacent and colour-coded.
         Y-limits are shared per row so the three conditions of one cluster are
         directly comparable.

    Args:
        centers: np.ndarray of shape (n_clusters, n_timepoints, n_conditions) —
                 the cluster_centers_ attribute of a fitted TimeSeriesKMeans model.
        Z: linkage matrix returned by compute_centroid_linkage().
        condition_names: list of condition label strings, length == n_conditions
                         (default: ["EGF", "INS", "EGFnINS"]).
        color_threshold: fraction of the maximum linkage distance used as the
                         dendrogram colour cutoff (default 0.7).
        figsize_dendro: (width, height) of the dendrogram figure (default (15, 4)).
        figsize_grid: (width, height) of the centroid grid figure.
                      Defaults to (2 * n_conditions, 1 * n_clusters).

    Returns:
        fig_dendro, fig_grid — the two Matplotlib Figure objects.
    """
    from scipy.cluster.hierarchy import dendrogram

    n_clusters, n_timepoints, n_conditions = centers.shape

    if condition_names is None:
        condition_names = [f"Cond {i}" for i in range(n_conditions)]

    threshold = color_threshold * max(Z[:, 2])

    # ── Figure 1: dendrogram ─────────────────────────────────────────────────
    fig_dendro, ax_dendro = plt.subplots(figsize=figsize_dendro)
    ddata = dendrogram(
        Z,
        labels=[f"Cluster {i}" for i in range(n_clusters)],
        ax=ax_dendro,
        color_threshold=threshold,
    )
    ax_dendro.set_title("Dendrogram of cluster centers (multivariate DTW)")
    ax_dendro.set_xlabel("Cluster")
    ax_dendro.set_ylabel("DTW distance")
    fig_dendro.tight_layout()

    leaf_order  = ddata["leaves"]
    leaf_colors = ddata["leaves_color_list"]
    cluster_color = {cluster_idx: color
                     for cluster_idx, color in zip(leaf_order, leaf_colors)}

    # ── Figure 2: centroid grid in dendrogram order ──────────────────────────
    if figsize_grid is None:
        figsize_grid = (2 * n_conditions, 1 * n_clusters)

    fig_grid, axes = plt.subplots(
        nrows=n_clusters,
        ncols=n_conditions,
        figsize=figsize_grid,
        sharex=True,
        sharey="row",
    )

    for row_idx, cluster_idx in enumerate(leaf_order):
        color = cluster_color[cluster_idx]

        for feat_idx in range(n_conditions):
            ax = axes[row_idx, feat_idx]
            ax.plot(
                centers[cluster_idx, :, feat_idx],
                color=color,
                linewidth=1.8,
            )
            ax.set_xlim(0, n_timepoints - 1)
            ax.tick_params(labelsize=8)
            ax.set_facecolor((*plt.matplotlib.colors.to_rgb(color), 0.06))

            if feat_idx == 0:
                ax.set_ylabel(f"Cluster {cluster_idx}", fontsize=9,
                              fontweight="bold", color=color)
            if row_idx == 0:
                ax.set_title(condition_names[feat_idx], fontsize=10, fontweight="bold")
            if row_idx == n_clusters - 1:
                ax.set_xlabel("Time step", fontsize=8)

        row_min = centers[cluster_idx].min()
        row_max = centers[cluster_idx].max()
        margin  = 0.05 * (row_max - row_min) if row_max != row_min else 0.1
        axes[row_idx, 0].set_ylim(row_min - margin, row_max + margin)

    fig_grid.suptitle("Cluster centers — ordered by dendrogram", fontsize=12, y=1.01)
    fig_grid.tight_layout()

    return fig_dendro, fig_grid


def plot_site_centroid_distances(df_clustered,
                                 barycenters,
                                 site,
                                 cluster_col,
                                 site_col="site",
                                 ax=None,
                                 figsize=(10, 4),):
    """
    Bar chart of DTW distances from a single phosphosite to every cluster centroid.

    The bar belonging to the assigned cluster is highlighted in a darker colour so
    it is immediately visible whether the assignment is clear-cut (assigned bar is
    much shorter than all others) or ambiguous (several bars of similar height).

    The assignment margin (2nd-best − best distance) is printed in the title as a
    quick confidence indicator.

    Args:
        df_clustered: DataFrame with a site identifier column and a cluster label column,
                      as returned by tslearn_clustering_KMeans(..., testing=True).
        barycenters: np.ndarray of shape (n_sites, n_clusters) — distance-to-centroid
                     matrix from fit_transform(), as returned when barycenter_calculations=True.
        site: site identifier string to look up (e.g. 'EGFR_HUMAN-Y1068y').
        cluster_col: name of the cluster label column in df_clustered.
        site_col: name of the site identifier column (default 'site').
        ax: existing Axes to draw on; a new Figure/Axes is created when None.
        figsize: (width, height) used only when ax=None (default (10, 4)).

    Returns:
        fig, ax — the Figure and Axes (fig is None when an external ax was passed).
    """
    from src.clustering import get_site_centroid_distances

    distances, assigned_cluster, _ = get_site_centroid_distances(
        df_clustered = df_clustered,
        barycenters  = barycenters,
        site         = site,
        cluster_col  = cluster_col,
        site_col     = site_col,
    )

    n_clusters = len(distances)
    sorted_d   = np.sort(distances)
    margin     = sorted_d[1] - sorted_d[0]

    colors = ["#4e79a7" if i != assigned_cluster else "#e15759"
              for i in range(n_clusters)]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    ax.bar(np.arange(n_clusters), distances, color=colors, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("DTW distance to centroid")
    ax.set_title(
        f"{site}  —  assigned to cluster {assigned_cluster}  |  margin = {margin:.3f}",
        fontsize=10,
    )
    ax.set_xticks(np.arange(n_clusters))

    if own_fig:
        fig.tight_layout()

    return fig, ax


def plot_sites_umap_interactive(
        df,
        cell_lines,
        conditions,
        data_type="log2:FC",
        exclude_full=True,
        color_col=None,
        hover_cols=None,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        figsize=(1000, 800),
        title=None,
):
    """
    Interactive Plotly UMAP embedding of phosphosites using their temporal profile as features.

    Each site is a point in a space defined by its log2:FC (or other data_type) values
    across the selected conditions and timepoints.  Hover labels show site, protein name,
    and the colour column so individual sites can be identified directly in the plot.

    Colouring:
      - color_col=None        → all points in a single colour.
      - color_col=<string>    → column name in df used to colour points.
          * If the column is numeric → continuous Viridis colour scale.
          * If the column is categorical / object → discrete colour sequence.

    Args:
        df: DataFrame following the project naming convention; rows are phosphosites.
        cell_lines: list of cell-line prefixes, e.g. ["WT"].
        conditions: list of condition substrings, e.g. ["_EGF_", "_INS_", "_EGFnINS_"].
        data_type: data-type string used for feature selection (default "log2:FC").
        exclude_full: if True, exclude the 'full' timepoint from the feature matrix
                      (default True).
        color_col: column name in df to colour points by, or None for a uniform colour.
        hover_cols: additional column names to show in the hover tooltip beyond the
                    defaults (site, protein_name, protein_Id, color_col).
                    Columns absent from df are silently ignored.
        n_neighbors: UMAP n_neighbors parameter (default 15).
        min_dist: UMAP min_dist parameter (default 0.1).
        random_state: random seed for reproducibility (default 42).
        figsize: (width, height) in pixels for the Plotly figure (default (1000, 800)).
        title: plot title; auto-generated from parameters when None.

    Returns:
        fig: Plotly Figure — interactive UMAP scatter.
        umap_df: DataFrame with UMAP coordinates and site metadata.
    """
    import umap as umap_lib
    from src.column_spec import ColumnSpec

    cols = ColumnSpec.select(df,
                             cell_lines=cell_lines,
                             data_type=data_type,
                             conditions=conditions,
                             exclude_full=exclude_full,
                             exclude_replicate_cols=True,)
    if not cols:
        raise ValueError(
            f"No columns found for cell_lines={cell_lines}, conditions={conditions}, "
            f"data_type={data_type!r}.  Check naming convention."
        )

    X = df[cols].fillna(0).values

    Z = umap_lib.UMAP(n_components=2,
                     n_neighbors=n_neighbors,
                     min_dist=min_dist,
                     random_state=random_state,).fit_transform(X)

    # --- Build metadata DataFrame ---
    umap_df = pd.DataFrame({"UMAP1": Z[:, 0], "UMAP2": Z[:, 1]}, index=df.index)

    default_meta = ["site", "protein_name", "protein_Id"]
    extra_cols   = list(hover_cols) if hover_cols else []
    for col in default_meta + extra_cols + ([color_col] if color_col else []):
        if col and col in df.columns and col not in umap_df.columns:
            umap_df[col] = df[col].values

    # --- Build hover_data dict ---
    hover_data = {c: True for c in default_meta + extra_cols if c in umap_df.columns}
    hover_data.update({"UMAP1": ":.3f", "UMAP2": ":.3f"})
    if color_col and color_col in umap_df.columns:
        hover_data[color_col] = True

    cond_label = "+".join(c.strip("_") for c in conditions)
    auto_title = title or f"UMAP of phosphosites — {data_type} | {cond_label}"

    # --- Categorical vs continuous colour ---
    if color_col and color_col in umap_df.columns and pd.api.types.is_numeric_dtype(umap_df[color_col]):
        fig = px.scatter(umap_df,
                         x="UMAP1", y="UMAP2",
                         color=color_col,
                         hover_name="site" if "site" in umap_df.columns else None,
                         hover_data=hover_data,
                         color_continuous_scale="Viridis",
                         title=auto_title,
                         width=figsize[0], height=figsize[1],)
    else:
        fig = px.scatter(umap_df,
                         x="UMAP1", y="UMAP2",
                         color=color_col,
                         hover_name="site" if "site" in umap_df.columns else None,
                         hover_data=hover_data,
                         title=auto_title,
                         width=figsize[0], height=figsize[1],)

    fig.update_traces(marker=dict(size=4, opacity=0.6))
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")

    return fig, umap_df
