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
from src.transformations import parse_columns, _sort_timepoints_numeric
# The anchored logistic used by notebooks/06_sigmoids. Imported at module level rather than
# lazily because src/curve_fitting.py imports nothing from src, so there is no import cycle.
from src.curve_fitting import anchored_sigmoid, soft_anchored_sigmoid
# CurveCurator's log-logistic, for the equivalent per-protein plot on its output. The chain
# curvecurator_io -> {column_spec, response_shapes} never reaches back here, so this is safe too.
from src.curvecurator_io import logistic_response, pec50_to_t50

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
                           figsize_grid=None,
                           metric_label="Euclidean",):
    """
    Visualise the hierarchical structure of KMeans cluster centroids.

    Produces two figures:
      1. **Dendrogram** — clusters ordered by the similarity of their centroids.
         Colour threshold groups clusters into super-clusters.
      2. **Centroid grid** — one row per cluster (in dendrogram leaf order), one
         column per condition.  Clusters belonging to the same super-cluster share
         the same line colour, so related profiles appear adjacent and colour-coded.
         Y-limits are shared per row so the three conditions of one cluster are
         directly comparable.

    Args:
        centers: np.ndarray of shape (n_clusters, n_timepoints, n_conditions) —
                 the cluster_centers_ attribute of a fitted TimeSeriesKMeans model.
                 A 2-D (n_clusters, n_timepoints) array is accepted as the
                 single-condition case.
        Z: linkage matrix returned by compute_centroid_linkage().
        condition_names: list of condition label strings, length == n_conditions
                         (default: generic "Cond i" labels).
        color_threshold: fraction of the maximum linkage distance used as the
                         dendrogram colour cutoff (default 0.7).
        figsize_dendro: (width, height) of the dendrogram figure (default (15, 4)).
        figsize_grid: (width, height) of the centroid grid figure.
                      Defaults to (2 * n_conditions, 1 * n_clusters).
        metric_label: name of the distance used to build Z, for the axis labels
                      (default "Euclidean" — must match the `metric` passed to
                      compute_centroid_linkage).

    Returns:
        fig_dendro, fig_grid — the two Matplotlib Figure objects.
    """
    from scipy.cluster.hierarchy import dendrogram

    centers = np.asarray(centers)
    # A single-condition run gives cluster_centers_ shape (n_clusters, n_timepoints);
    # add the trailing condition axis so the grid code below is dimension-agnostic.
    if centers.ndim == 2:
        centers = centers[:, :, None]

    n_clusters, n_timepoints, n_conditions = centers.shape

    if condition_names is None:
        condition_names = [f"Cond {i}" for i in range(n_conditions)]
    if len(condition_names) != n_conditions:
        raise ValueError(f"plot_cluster_hierarchy: got {len(condition_names)} condition_names "
                         f"for {n_conditions} conditions in `centers`.")

    threshold = color_threshold * max(Z[:, 2])

    # ── Figure 1: dendrogram ─────────────────────────────────────────────────
    fig_dendro, ax_dendro = plt.subplots(figsize=figsize_dendro)
    ddata = dendrogram(
        Z,
        labels=[f"Cluster {i}" for i in range(n_clusters)],
        ax=ax_dendro,
        color_threshold=threshold,
    )
    ax_dendro.set_title(f"Dendrogram of cluster centroids ({metric_label})")
    ax_dendro.set_xlabel("Cluster")
    ax_dendro.set_ylabel(f"{metric_label} distance")
    fig_dendro.tight_layout()

    leaf_order  = ddata["leaves"]
    leaf_colors = ddata["leaves_color_list"]
    cluster_color = {cluster_idx: color
                     for cluster_idx, color in zip(leaf_order, leaf_colors)}

    # ── Figure 2: centroid grid in dendrogram order ──────────────────────────
    if figsize_grid is None:
        figsize_grid = (2 * n_conditions, 1 * n_clusters)

    # squeeze=False keeps `axes` 2-D even with one row or one column, so axes[r, c]
    # is always valid — without it a single-condition run returns a 1-D array and
    # every axes[row, col] lookup raises IndexError.
    fig_grid, axes = plt.subplots(
        nrows=n_clusters,
        ncols=n_conditions,
        figsize=figsize_grid,
        sharex=True,
        sharey="row",
        squeeze=False,
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


# ---------------------
# Sigmoid fits per protein (notebooks/06_sigmoids/Sigmoid_fitting.ipynb)
# ---------------------

def _sigmoid_panel_row(fit_row,
                       columns,):
    """
    Pull the anchored-sigmoid parameters out of one row of a fit table.

    Kept separate so the panel and the overlay read the parameters the same way, and so a
    reduced fit table (one missing a diagnostic column) degrades to NaN instead of raising.

    Args:
        fit_row: a single row (Series) of the fit table.
        columns: the set of column names available in the table.

    Returns:
        Dict with keys y0, A, k, x50, plateau, t_half_min, se_t_half, rmse — NaN where the
        column is absent or not finite, except `y0`, which falls back to **0.0**: a table
        written before the soft anchor existed, or produced with `free_baseline=False`, has a
        baseline of exactly zero, and NaN there would blank out the whole curve.

    """
    out = {}
    for key in ["A", "k", "x50", "plateau", "t_half_min", "se_t_half", "rmse",]:
        value = fit_row[key] if key in columns else np.nan
        try:
            out[key] = float(value)
        except (TypeError, ValueError,):
            out[key] = np.nan

    y0 = fit_row["y0"] if "y0" in columns else 0.0
    try:
        out["y0"] = float(y0) if np.isfinite(float(y0)) else 0.0
    except (TypeError, ValueError,):
        out["y0"] = 0.0
    return out


def plot_protein_sigmoid_fits(fit_df,
                              values,
                              times,
                              proteins,
                              sem=None,
                              site_col="site",
                              protein_name_col="protein_name",
                              protein_Id_col="protein_Id",
                              gate_col="fit_ok",
                              shape_col="shape_class",
                              log_time=True,
                              include_unfitted=True,
                              only_passing=False,
                              max_sites=48,
                              overlay=True,
                              n_cols=4,
                              figsize_per_panel=(3.4, 2.9),
                              share_y=True,
                              label_chars=30,
                              value_label="log2 fold change vs starve",
                              title_info="",
                              saving_path="",
                              saving_info="",
                              save_pdf=False,
                              save_png=False,
                              plot_close=False,):
    """
    Plot every phosphosite of a protein together with the sigmoid fitted to it.

    One figure per protein, one panel per site: the observed log2 fold-change profile with its
    SEM error bars, the fitted anchored logistic on top, the plateau it approaches and the T50
    it implies. Sites that were never fitted (the shape gate rejected them as transient or
    biphasic) are still drawn, without a curve, so that a protein is never silently shown as
    "three sites" when it has eight — which of a protein's sites a sigmoid *cannot* describe is
    itself the result. With `overlay=True` an extra single-axes figure per protein puts all of
    that protein's fitted curves on the same axes, which is the plot that actually answers
    "do these sites respond at the same time".

    `values` must be row-for-row aligned with `fit_df` — both come from the same source
    DataFrame in Sigmoid_fitting.ipynb (`build_profile_matrix` preserves row order and
    `fit_sigmoid_dataset` returns one row per input row), so passing the notebook's
    `values_mean` and the concatenated `output` table works directly.

    ⚠️ Pass the matrix the fits were **made on**, not the one the shapes were classified on. Soft
    anchored fits (`free_baseline=True`) live on `log2:mean`, so `values` must be the mean-scale
    matrix and `value_label` should say so; hard-anchored fits live on `log2:FC`. The two are
    row-aligned and share a minute axis, so passing the wrong one raises nothing — it just draws
    curves at 0 against data at 18.

    Curves are drawn with the same parameterisation the fit used: `soft_anchored_sigmoid`
    evaluated on log10(t+1) when `log_time` is True, with `y0` read from the table and falling
    back to 0.0 when the column is absent, which reproduces the hard-anchored curve exactly.
    Passing a `log_time` that disagrees with the fit silently draws the wrong curve, so it is
    validated against the table's `log_time` column when that column is present.

    Args:
        fit_df: DataFrame with one row per site, carrying the protein/site identifier columns
            and the fit parameters produced by `fit_sigmoid_dataset` (y0, A, k, x50, plateau,
            t_half_min, se_t_half, rmse) — typically the `output` table of the sigmoid notebook.
        values: (n_sites, T) observed profile matrix from `build_profile_matrix`, positionally
            aligned with `fit_df`.
        times: (T,) minute axis matching the columns of `values`.
        proteins: protein name or UniProt accession, or a list of them. Matched against
            `protein_name_col` first, then `protein_Id_col`.
        sem: optional (n_sites, T) standard errors from `per_site_sem`, drawn as error bars.
        site_col: column holding the site identifier used for panel titles.
        protein_name_col: column holding the protein name.
        protein_Id_col: column holding the UniProt accession (note the lowercase 'd', as on disk).
        gate_col: boolean column marking fits that passed every quality gate. Fits that failed
            are drawn dashed and orange rather than hidden. Ignored if absent.
        shape_col: column holding the response-shape class, shown in the panel title so an
            unfitted site says why it was not fitted. Ignored if absent.
        log_time: evaluate the curve on log10(t+1); must match the axis used for fitting.
        include_unfitted: draw sites with no fitted parameters (observed data only). Set False
            to show only the sites a sigmoid was actually fitted to.
        only_passing: draw only sites whose `gate_col` is True. Overrides `include_unfitted`.
        max_sites: cap on the number of panels per protein. Proteins such as SRRM2 carry
            hundreds of sites; above the cap the selection keeps passing fits first, then other
            fits, then unfitted sites, and prints what it dropped. None plots everything.
        overlay: additionally produce a one-axes figure per protein with all fitted curves
            superimposed, annotated with each site's T50.
        n_cols: panels per row in the per-site figure.
        figsize_per_panel: (width, height) in inches of one panel.
        share_y: give every panel of a protein the same y limits, so amplitudes are comparable.
            A single large-amplitude site then flattens the rest — which is honest, but set
            False when the point is to read the shape of each site rather than compare them.
        label_chars: truncate site labels in panel titles to this many characters.
        value_label: y-axis label. The default describes a `log2:FC` profile; pass
            "log2 mean intensity" (or similar) when `values` is a `log2:mean` matrix and the
            fits were soft-anchored, since the axis is then an absolute level, not a ratio.
        title_info: extra text appended to the figure title.
        saving_path: directory to save into; a `{protein_name}_{protein_Id}` subfolder is
            created inside it, matching `plot_protein_phosphosites`.
        saving_info: suffix added to the saved file name.
        save_pdf: save the figures as PDF.
        save_png: save the figures as PNG.
        plot_close: close the figures after drawing/saving (use when looping over many proteins).

    Returns:
        Dict keyed by protein, each value a dict with keys 'fig' and 'axes' for the per-site
        figure and, when `overlay` is True and at least one site was fitted, 'overlay_fig' and
        'overlay_ax'. Proteins absent from the table are skipped with a message and do not
        appear in the dict.

    """
    if isinstance(fit_df, str,):
        if fit_df.endswith(".tsv",):
            fit_df = pd.read_csv(fit_df, sep="\t", low_memory=False,)
        elif fit_df.endswith(".xlsx",):
            fit_df = pd.read_excel(fit_df,)
        else:
            raise ValueError("plot_protein_sigmoid_fits: unsupported file format, use .tsv or .xlsx")

    values = np.asarray(values, dtype=float,)
    times = np.asarray(times, dtype=float,)

    if values.shape[0] != len(fit_df):
        raise ValueError(f"plot_protein_sigmoid_fits: values has {values.shape[0]} rows but "
                         f"fit_df has {len(fit_df)}. They must be row-for-row aligned — pass the "
                         f"profile matrix and the fit table built from the same DataFrame.")
    if values.shape[1] != times.size:
        raise ValueError(f"plot_protein_sigmoid_fits: values has {values.shape[1]} timepoints but "
                         f"times has {times.size}.")
    if sem is not None:
        sem = np.asarray(sem, dtype=float,)
        if sem.shape != values.shape:
            raise ValueError("plot_protein_sigmoid_fits: sem must have the same shape as values.")

    # A log_time mismatch draws a curve that is not the one that was fitted, silently.
    if "log_time" in fit_df.columns:
        stored = fit_df["log_time"].dropna().unique()
        if stored.size == 1 and bool(stored[0]) != bool(log_time):
            raise ValueError(f"plot_protein_sigmoid_fits: fits were computed with "
                             f"log_time={bool(stored[0])} but log_time={log_time} was requested. "
                             f"The curve would not match the fit.")

    if isinstance(proteins, str,):
        proteins = [proteins]

    columns = set(fit_df.columns)
    if protein_name_col not in columns and protein_Id_col not in columns:
        raise ValueError(f"plot_protein_sigmoid_fits: neither '{protein_name_col}' nor "
                         f"'{protein_Id_col}' is present in fit_df.")

    # Curves are drawn on a dense grid; the observed points sit on the transformed time axis.
    grid_t = np.linspace(0.0, float(times.max()), 400,)
    grid_x = np.log10(grid_t + 1.0,) if log_time else grid_t
    obs_x = np.log10(times + 1.0,) if log_time else times

    results = {}

    for protein in proteins:
        selector = pd.Series(False, index=fit_df.index,)
        if protein_name_col in columns:
            selector = selector | (fit_df[protein_name_col] == protein)
        if protein_Id_col in columns:
            selector = selector | (fit_df[protein_Id_col] == protein)

        if not selector.any():
            print(f"The protein {protein} is not present in the fit table")
            continue

        positions = np.where(selector.to_numpy())[0]

        # Order the panels by site label so the same protein always comes out the same way.
        if site_col in columns:
            labels = fit_df[site_col].to_numpy()
            positions = positions[np.argsort([str(labels[i]) for i in positions],
                                             kind="stable",)]
        else:
            labels = np.array([f"row {i}" for i in range(len(fit_df))],)

        fitted_flag = {i: np.isfinite(_sigmoid_panel_row(fit_df.iloc[i], columns,)["A"])
                       for i in positions}
        # A gate value of NaN (site never entered the fit) must read as "did not pass", which
        # `== True` gives; a site with no curve can never count as passing.
        if gate_col in columns:
            passed_flag = {i: bool(fitted_flag[i] and fit_df[gate_col].iloc[i] == True)
                           for i in positions}
        else:
            passed_flag = {i: fitted_flag[i] for i in positions}

        if only_passing:
            positions = np.array([i for i in positions if passed_flag[i]],)
        elif not include_unfitted:
            positions = np.array([i for i in positions if fitted_flag[i]],)

        # Heavily phosphorylated proteins are not a corner case here: SRRM2 carries 677 sites in
        # hme1_2, of which 91 are fitted and 25 pass. One panel each is unreadable and slow, so
        # the selection is truncated, keeping the informative sites first and saying so out loud.
        n_selected = positions.size
        if max_sites is not None and n_selected > max_sites:
            rank = {i: (0 if passed_flag[i] else 1 if fitted_flag[i] else 2) for i in positions}
            positions = np.array(sorted(positions,
                                        key=lambda i: (rank[i], str(labels[i]),),)[:max_sites],)
            positions = positions[np.argsort([str(labels[i]) for i in positions],
                                             kind="stable",)]
            print(f"{protein}: {n_selected} sites selected, showing {max_sites} "
                  f"(passing fits first, then other fits, then unfitted). "
                  f"Raise max_sites, or narrow with only_passing=True / include_unfitted=False.")

        if positions.size == 0:
            print(f"The protein {protein} has no site left to plot with the current filters "
                  f"(only_passing={only_passing}, include_unfitted={include_unfitted})")
            continue

        name = (fit_df[protein_name_col].iloc[positions[0]]
                if protein_name_col in columns else str(protein))
        accession = (fit_df[protein_Id_col].iloc[positions[0]]
                     if protein_Id_col in columns else "")
        saving_folder = f"{name}_{accession}".strip("_",)

        if (save_pdf or save_png) and saving_path:
            os.makedirs(os.path.join(saving_path, saving_folder,), exist_ok=True,)

        n_sites = positions.size
        n_rows = int(np.ceil(n_sites / n_cols,))
        fig, axes = plt.subplots(n_rows,
                                 min(n_cols, n_sites,),
                                 figsize=(figsize_per_panel[0] * min(n_cols, n_sites,),
                                          figsize_per_panel[1] * n_rows,),
                                 squeeze=False,)
        n_cols_used = axes.shape[1]

        # Shared limits are computed over the observed data *and* the plateaus, so a fit that
        # runs above the last measured point is not cropped out of its own panel.
        y_lim = None
        if share_y:
            pool = [values[positions].ravel()]
            if sem is not None:
                pool.append((values[positions] + sem[positions]).ravel(),)
                pool.append((values[positions] - sem[positions]).ravel(),)
            # The plateau is measured relative to the fitted baseline, so the level actually
            # drawn is y0 + plateau — which is what the axis has to accommodate.
            plateaus = [(lambda p: p["y0"] + p["plateau"])(_sigmoid_panel_row(fit_df.iloc[i], columns,))
                        for i in positions]
            pool.append(np.array(plateaus, dtype=float,),)
            pooled = np.concatenate(pool,)
            pooled = pooled[np.isfinite(pooled)]
            if pooled.size:
                pad = 0.1 * max(np.ptp(pooled,), 0.1,)
                y_lim = (pooled.min() - pad, pooled.max() + pad,)

        for n, i in enumerate(positions):
            ax = axes[n // n_cols_used, n % n_cols_used]
            params = _sigmoid_panel_row(fit_df.iloc[i], columns,)

            ax.errorbar(obs_x,
                        values[i],
                        yerr=sem[i] if sem is not None else None,
                        fmt="o",
                        ms=4,
                        color="black",
                        capsize=2,
                        elinewidth=1.0,
                        zorder=3,
                        label="observed",)

            if fitted_flag[i]:
                colour = "crimson" if passed_flag[i] else "darkorange"
                ax.plot(grid_x,
                        soft_anchored_sigmoid(grid_x,
                                              params["y0"],
                                              params["A"],
                                              params["k"],
                                              params["x50"],),
                        color=colour,
                        ls="-" if passed_flag[i] else "--",
                        lw=1.8,
                        zorder=2,
                        label="fit" if passed_flag[i] else "fit (failed gate)",)
                if np.isfinite(params["plateau"]):
                    ax.axhline(params["y0"] + params["plateau"],
                               color="tab:blue",
                               ls=":",
                               lw=1.0,)
                t_half = params["t_half_min"]
                if np.isfinite(t_half) and t_half >= 0:
                    ax.axvline(np.log10(t_half + 1.0,) if log_time else t_half,
                               color="tab:green",
                               ls="--",
                               lw=1.0,)

            # The reference line is the baseline the response is measured from: exactly 0 on the
            # fold-change scale, the fitted y0 on the mean scale.
            ax.axhline(params["y0"], color="grey", lw=0.7,)
            ax.set_xticks(obs_x,)
            ax.set_xticklabels([f"{t:g}" for t in times], fontsize=7,)
            if y_lim is not None:
                ax.set_ylim(*y_lim,)

            shape = str(fit_df[shape_col].iloc[i]) if shape_col in columns else ""
            if fitted_flag[i] and np.isfinite(params["t_half_min"]):
                se = params["se_t_half"]
                se_txt = f" ± {se:.1f}" if np.isfinite(se) else ""
                second = f"T50 = {params['t_half_min']:.1f}{se_txt} min"
                if np.isfinite(params["rmse"]):
                    second += f"   RMSE = {params['rmse']:.2f}"
                if not passed_flag[i]:
                    second += "   (failed gate)"
            else:
                second = f"not fitted{f' — {shape}' if shape else ''}"
            # TMT site keys run to ~45 characters ('P00533_693_695_1_1_T693~ELVEPLTPSGEAPNQALLR'),
            # which is wider than a panel and would overlap its neighbour's title. Truncate the
            # key itself, not the shape tag appended after it.
            head = str(labels[i])
            if len(head) > label_chars:
                head = head[:label_chars - 1] + "…"
            if shape and fitted_flag[i]:
                head = f"{head}  [{shape}]"
            ax.set_title(f"{head}\n{second}", fontsize=7,)

        for n in range(n_sites, n_rows * n_cols_used,):
            axes[n // n_cols_used, n % n_cols_used].axis("off",)

        n_fitted = int(sum(fitted_flag[i] for i in positions))
        n_passed = int(sum(passed_flag[i] for i in positions))
        shown = (f"{n_sites} sites plotted" if n_sites == n_selected
                 else f"{n_sites} of {n_selected} sites plotted")
        fig.suptitle(f"{saving_folder} — sigmoid fits  "
                     f"({n_passed}/{n_fitted} fits passing of {shown}) "
                     f"{title_info} ({date.today()})",
                     weight="bold",
                     fontsize=11,)
        fig.supxlabel("time (min)")
        fig.supylabel(value_label,)
        # Reserve a fixed strip for the suptitle instead of a fraction, so it does not eat half
        # the figure when there is only one row of panels.
        fig_height = fig.get_size_inches()[1]
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0 - 0.42 / fig_height,),)

        entry = {"fig": fig, "axes": axes,}

        # --- Overlay: every fitted curve of this protein on one axes ---
        overlay_positions = [i for i in positions if fitted_flag[i]]
        if overlay and overlay_positions:
            n_curves = len(overlay_positions)
            # Site labels are long, so an in-axes legend covers the curves it describes from
            # about seven entries on; past that it moves out to the right and the figure widens
            # to pay for it. A single legend column taller than the axes makes tight_layout give
            # up, hence the column count.
            crowded = n_curves > 6
            legend_ncol = int(np.ceil(n_curves / 16,)) if crowded else 1
            o_fig, o_ax = plt.subplots(figsize=(6.5 + 2.6 * legend_ncol if crowded else 6.5,
                                                max(4.4, 0.16 * min(n_curves, 16,) + 2.2,),),)
            # The default cycle holds 10 colours and would start repeating beyond that.
            if n_curves > 10:
                palette = cycle([plt.get_cmap("tab20",)(j / max(n_curves - 1, 1,))
                                 for j in range(n_curves)],)
            else:
                palette = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"],)
            for i in overlay_positions:
                params = _sigmoid_panel_row(fit_df.iloc[i], columns,)
                colour = next(palette,)
                t_half = params["t_half_min"]
                t_txt = f"T50 {t_half:.1f} min" if np.isfinite(t_half) else "T50 n/a"
                o_ax.plot(grid_x,
                          soft_anchored_sigmoid(grid_x,
                                                params["y0"],
                                                params["A"],
                                                params["k"],
                                                params["x50"],),
                          color=colour,
                          ls="-" if passed_flag[i] else "--",
                          lw=1.8,
                          label=f"{labels[i]} ({t_txt})"
                                + ("" if passed_flag[i] else " [failed gate]"),)
                o_ax.plot(obs_x,
                          values[i],
                          "o",
                          ms=3.5,
                          color=colour,
                          alpha=0.55,)
                # One rule per T50 reads well for a handful of curves and becomes a picket fence
                # for twenty; above that the legend already carries every T50.
                if np.isfinite(t_half) and t_half >= 0 and not crowded:
                    o_ax.axvline(np.log10(t_half + 1.0,) if log_time else t_half,
                                 color=colour,
                                 ls=":",
                                 lw=0.9,
                                 alpha=0.7,)

            # A single zero line is the shared baseline only on the fold-change scale. With
            # soft-anchored fits every site has its own y0, so one rule at 0 would sit far off
            # the axis and imply a reference that does not exist.
            _y0s = np.array([_sigmoid_panel_row(fit_df.iloc[i], columns,)["y0"]
                             for i in overlay_positions],)
            if np.allclose(_y0s, 0.0,):
                o_ax.axhline(0.0, color="grey", lw=0.7,)
            o_ax.set_xticks(obs_x,)
            o_ax.set_xticklabels([f"{t:g}" for t in times],)
            o_ax.set_xlabel("time (min)" + (" — log10(t+1) spacing" if log_time else ""),)
            o_ax.set_ylabel(value_label,)
            o_ax.set_title(f"{saving_folder} — all fitted sigmoids {title_info}",
                           weight="bold",
                           fontsize=10,)
            if crowded:
                o_ax.legend(frameon=False,
                            fontsize=6.5,
                            ncol=legend_ncol,
                            loc="center left",
                            bbox_to_anchor=(1.01, 0.5,),)
                # tight_layout only measures artists inside the axes, so a legend parked to the
                # right of it makes the solver fail. Reserve the width it was sized for instead.
                o_fig.subplots_adjust(left=0.09,
                                      right=6.2 / (6.5 + 2.6 * legend_ncol),
                                      top=0.90,
                                      bottom=0.13,)
            else:
                o_ax.legend(frameon=False,
                            fontsize=7,
                            loc="best",)
                o_fig.tight_layout()
            entry["overlay_fig"] = o_fig
            entry["overlay_ax"] = o_ax

        if (save_pdf or save_png) and saving_path:
            base = os.path.join(saving_path,
                                saving_folder,
                                f"{saving_folder}_sigmoid_fits_{saving_info}".rstrip("_",),)
            for ext, flag in [("pdf", save_pdf,), ("png", save_png,)]:
                if not flag:
                    continue
                fig.savefig(f"{base}.{ext}", dpi=200, bbox_inches="tight",)
                print(f"Saved {ext.upper()}: {base}.{ext}")
                if "overlay_fig" in entry:
                    entry["overlay_fig"].savefig(f"{base}_overlay.{ext}",
                                                 dpi=200,
                                                 bbox_inches="tight",)
                    print(f"Saved {ext.upper()}: {base}_overlay.{ext}")
        elif save_pdf or save_png:
            print(f"{saving_folder} — saving_path is empty, plot not saved")

        if plot_close:
            plt.close(fig,)
            if "overlay_fig" in entry:
                plt.close(entry["overlay_fig"],)

        results[protein] = entry

    return results


# ---------------------
# CurveCurator fits per protein (notebooks/06_sigmoids/Sigmoid_fitting_CurveCurator.ipynb)
# ---------------------

def curvecurator_observed_matrix(curves_df,
                                 design,
                                 prefer="Ratio",):
    """
    Pull the response values CurveCurator actually fitted out of its curves file.

    CurveCurator writes three column families per experiment: `Raw {e}` (as supplied),
    `Normalized {e}` (after median centring) and `Ratio {e}` (the normalised value divided by
    the mean of the controls). **The fit is against `Ratio`.** Recomputing ratios from `Raw`
    instead differs by up to ~0.5 on this dataset whenever `normalization = true`, which would
    draw observed points that do not sit on their own fitted curve.

    Args:
        curves_df: the loaded curves table.
        design: the design table from `build_curvecurator_input`, giving the experiment order.
        prefer: which family to read — 'Ratio' (what was fitted), 'Normalized' or 'Raw'. The
            latter two are rescaled to the control mean so they remain plottable against the
            curve, and are only useful for diagnosing the normalisation itself.

    Returns:
        Tuple (values, is_control):
            values: (n_sites, n_experiments) array of responses in ratio space.
            is_control: (n_experiments,) boolean array marking the control channels.

    """
    if prefer not in ("Ratio", "Normalized", "Raw",):
        raise ValueError(f"curvecurator_observed_matrix: prefer must be 'Ratio', 'Normalized' "
                         f"or 'Raw', got {prefer!r}.")

    cols = [f"{prefer} {e}" for e in design["experiment"]]
    missing = [c for c in cols if c not in curves_df.columns]
    if missing:
        raise ValueError(f"curvecurator_observed_matrix: {len(missing)} '{prefer} N' columns are "
                         f"absent from the curves table (first missing: {missing[0]}). "
                         f"CurveCurator writes them alongside the fit parameters — check that "
                         f"this is the curves file and not a reduced export.")

    values = curves_df[cols].to_numpy(dtype=float,)
    is_control = design["is_control"].to_numpy(dtype=bool,)

    if prefer != "Ratio":
        # Put Raw/Normalized on the same scale as the curve, which lives in ratio space.
        with np.errstate(invalid="ignore", divide="ignore",):
            values = values / np.nanmean(values[:, is_control], axis=1, keepdims=True,)
    return values, is_control


def _curvecurator_params(row,
                         columns,):
    """
    Pull the log-logistic parameters out of one row of a CurveCurator curves table.

    Args:
        row: a single row (Series) of the curves table.
        columns: the set of column names available.

    Returns:
        Dict with keys pEC50, slope, front, back, t50_min, rmse, regulation — NaN (or None for
        the regulation label) where the column is absent or unparseable.

    """
    out = {}
    for key, col in [("pEC50", "pEC50",),
                     ("slope", "Curve Slope",),
                     ("front", "Curve Front",),
                     ("back", "Curve Back",),
                     ("t50_min", "t50_min",),
                     ("rmse", "Curve RMSE",),]:
        value = row[col] if col in columns else np.nan
        try:
            out[key] = float(value)
        except (TypeError, ValueError,):
            out[key] = np.nan
    reg = row["Curve Regulation"] if "Curve Regulation" in columns else None
    out["regulation"] = None if (reg is None or pd.isna(reg,)) else str(reg)
    return out


def plot_protein_curvecurator_fits(curves_df,
                                   design,
                                   proteins,
                                   dose_scale=1.0,
                                   prefer="Ratio",
                                   show_control=True,
                                   site_col="site",
                                   protein_name_col="protein_name",
                                   protein_Id_col="protein_Id",
                                   regulated_labels=("up", "down",),
                                   include_unregulated=True,
                                   only_regulated=False,
                                   max_sites=48,
                                   overlay=True,
                                   n_cols=4,
                                   figsize_per_panel=(3.4, 2.9),
                                   share_y=True,
                                   label_chars=30,
                                   title_info="",
                                   saving_path="",
                                   saving_info="",
                                   save_pdf=False,
                                   save_png=False,
                                   plot_close=False,):
    """
    Plot every phosphosite of a protein with the sigmoid CurveCurator fitted to it.

    The CurveCurator counterpart of `plot_protein_sigmoid_fits`, and deliberately the same
    shape: one figure per protein with a panel per site, plus an overlay putting all of that
    protein's curves on one axes so their T50s can be read against each other.

    Three things differ from the anchored-fit version, all of them consequences of the tool:

    - **The y axis is a ratio to the starve control, not a log2 fold change**, so the reference
      line sits at 1 rather than 0.
    - **Every replicate is drawn**, not a mean with error bars — CurveCurator fitted the
      individual replicate ratios, and the scatter around the curve is the actual residual.
    - **The control sits far to the left.** `build_drug_log_concentrations` places dose 0 three
      decades below the smallest dose, so on the fitted x axis the starve points are at
      log10(t_min/1000) with nothing between them and the first real timepoint. That gap is a
      property of the dose-response parameterisation, not of the experiment, and seeing it is
      the point of drawing the control where the fitter put it.

    Sites CurveCurator did not classify as regulated are still drawn, with their curve dashed
    and orange and the reason in the panel title — usually a T50 outside the `pEC50_filter`
    window, i.e. a transition the experiment never observed.

    Args:
        curves_df: curves table from `load_curvecurator_curves`, carrying the fit parameters,
            the `Ratio N` columns and (merged in) the protein annotations.
        design: the design table from `build_curvecurator_input`, mapping experiments to
            timepoints. Must be the design that produced this run.
        proteins: protein name or UniProt accession, or a list of them.
        dose_scale: the numeric `dose_scale` used in the TOML (1.0 when doses are minutes).
        prefer: which response family to draw — see `curvecurator_observed_matrix`.
        show_control: draw the starve points where CurveCurator puts them, three decades to the
            left of the first timepoint. True is faithful to the fit and makes the empty gap
            visible; False drops them and rescales to the sampled window, which is what to use
            when the point is to read the response rather than audit the parameterisation.
        site_col: column holding the site identifier.
        protein_name_col: column holding the protein name.
        protein_Id_col: column holding the UniProt accession.
        regulated_labels: values of `Curve Regulation` that count as regulated.
        include_unregulated: draw sites that were not classified as regulated.
        only_regulated: draw only regulated sites. Overrides `include_unregulated`.
        max_sites: cap on panels per protein; the regulated sites are kept first.
        overlay: also produce the all-curves-on-one-axes figure per protein.
        n_cols: panels per row.
        figsize_per_panel: (width, height) in inches of one panel.
        share_y: give every panel of a protein the same y limits.
        label_chars: truncate site labels in panel titles to this many characters.
        title_info: extra text appended to the figure title.
        saving_path: directory to save into; a `{protein_name}_{protein_Id}` subfolder is made.
        saving_info: suffix added to the saved file name.
        save_pdf: save the figures as PDF.
        save_png: save the figures as PNG.
        plot_close: close the figures after drawing/saving.

    Returns:
        Dict keyed by protein, each value a dict with 'fig' and 'axes' and, when `overlay` is
        True and at least one site was fitted, 'overlay_fig' and 'overlay_ax'. Proteins absent
        from the table are skipped with a message.

    """
    if isinstance(proteins, str,):
        proteins = [proteins]

    columns = set(curves_df.columns)
    if protein_name_col not in columns and protein_Id_col not in columns:
        raise ValueError(f"plot_protein_curvecurator_fits: neither '{protein_name_col}' nor "
                         f"'{protein_Id_col}' is in the curves table. Pass `annotate=` to "
                         f"load_curvecurator_curves so the protein columns are merged in.")

    observed, is_control = curvecurator_observed_matrix(curves_df, design, prefer=prefer,)
    minutes = design["minutes"].to_numpy(dtype=float,)
    treated = minutes > 0

    # Reproduce CurveCurator's own x axis, including where it parks the zero dose.
    control_dose = minutes[treated].min() / 1000.0
    x_axis = np.log10(np.where(treated, minutes, control_dose,) * dose_scale,)
    tick_times = sorted(set(minutes[treated]),)
    tick_x = [np.log10(t * dose_scale,) for t in tick_times]

    # The curve is always evaluated over the range the fit used; only the drawn window changes,
    # so hiding the control never alters the curve, just how much empty axis is shown.
    grid = np.linspace(x_axis.min(), x_axis.max(), 400,)
    x_window = (min(tick_x) - 0.15, max(tick_x) + 0.15,) if not show_control else None
    results = {}

    for protein in proteins:
        selector = pd.Series(False, index=curves_df.index,)
        if protein_name_col in columns:
            selector = selector | (curves_df[protein_name_col] == protein)
        if protein_Id_col in columns:
            selector = selector | (curves_df[protein_Id_col] == protein)

        if not selector.any():
            print(f"The protein {protein} is not present in the curves table")
            continue

        positions = np.where(selector.to_numpy())[0]
        labels = (curves_df[site_col].to_numpy() if site_col in columns
                  else np.array([f"row {i}" for i in range(len(curves_df))],))
        if site_col in columns:
            positions = positions[np.argsort([str(labels[i]) for i in positions],
                                             kind="stable",)]

        params = {i: _curvecurator_params(curves_df.iloc[i], columns,) for i in positions}
        fitted_flag = {i: np.isfinite(params[i]["pEC50"]) for i in positions}
        regulated_flag = {i: bool(fitted_flag[i]
                                  and params[i]["regulation"] in regulated_labels)
                          for i in positions}

        if only_regulated:
            positions = np.array([i for i in positions if regulated_flag[i]],)
        elif not include_unregulated:
            positions = np.array([i for i in positions if fitted_flag[i]],)

        n_selected = positions.size
        if max_sites is not None and n_selected > max_sites:
            rank = {i: (0 if regulated_flag[i] else 1 if fitted_flag[i] else 2)
                    for i in positions}
            positions = np.array(sorted(positions,
                                        key=lambda i: (rank[i], str(labels[i]),),)[:max_sites],)
            positions = positions[np.argsort([str(labels[i]) for i in positions],
                                             kind="stable",)]
            print(f"{protein}: {n_selected} sites selected, showing {max_sites} "
                  f"(regulated first). Raise max_sites, or narrow with only_regulated=True.")

        if positions.size == 0:
            print(f"The protein {protein} has no site left to plot with the current filters "
                  f"(only_regulated={only_regulated}, "
                  f"include_unregulated={include_unregulated})")
            continue

        name = (curves_df[protein_name_col].iloc[positions[0]]
                if protein_name_col in columns else str(protein))
        accession = (curves_df[protein_Id_col].iloc[positions[0]]
                     if protein_Id_col in columns else "")
        saving_folder = f"{name}_{accession}".strip("_",)

        if (save_pdf or save_png) and saving_path:
            os.makedirs(os.path.join(saving_path, saving_folder,), exist_ok=True,)

        n_sites = positions.size
        n_rows = int(np.ceil(n_sites / n_cols,))
        fig, axes = plt.subplots(n_rows,
                                 min(n_cols, n_sites,),
                                 figsize=(figsize_per_panel[0] * min(n_cols, n_sites,),
                                          figsize_per_panel[1] * n_rows,),
                                 squeeze=False,)
        n_cols_used = axes.shape[1]

        y_lim = None
        if share_y:
            visible = observed[positions] if show_control else observed[positions][:, treated]
            pooled = visible.ravel()
            plateaus = np.array([params[i]["back"] for i in positions], dtype=float,)
            pooled = np.concatenate([pooled, plateaus, [1.0]],)
            pooled = pooled[np.isfinite(pooled)]
            if pooled.size:
                pad = 0.1 * max(np.ptp(pooled,), 0.1,)
                y_lim = (pooled.min() - pad, pooled.max() + pad,)

        for n, i in enumerate(positions):
            ax = axes[n // n_cols_used, n % n_cols_used]
            p = params[i]

            if show_control:
                ax.scatter(x_axis[~treated],
                           observed[i][~treated],
                           s=18,
                           color="grey",
                           zorder=3,
                           label="starve",)
            ax.scatter(x_axis[treated],
                       observed[i][treated],
                       s=18,
                       color="black",
                       zorder=3,
                       label="stimulated",)

            if fitted_flag[i]:
                colour = "crimson" if regulated_flag[i] else "darkorange"
                ax.plot(grid,
                        logistic_response(grid,
                                          p["pEC50"],
                                          p["slope"],
                                          p["front"],
                                          p["back"],),
                        color=colour,
                        ls="-" if regulated_flag[i] else "--",
                        lw=1.8,
                        zorder=2,)
                if np.isfinite(p["back"]):
                    ax.axhline(p["back"], color="tab:blue", ls=":", lw=1.0,)
                t50 = p["t50_min"]
                if np.isfinite(t50) and t50 > 0:
                    ax.axvline(np.log10(t50 * dose_scale,),
                               color="tab:green",
                               ls="--",
                               lw=1.0,)

            ax.axhline(1.0, color="grey", lw=0.7,)
            ax.set_xticks(tick_x,)
            ax.set_xticklabels([f"{t:g}" for t in tick_times], fontsize=7,)
            if x_window is not None:
                ax.set_xlim(*x_window,)
            if y_lim is not None:
                ax.set_ylim(*y_lim,)

            head = str(labels[i])
            if len(head) > label_chars:
                head = head[:label_chars - 1] + "…"
            if fitted_flag[i] and np.isfinite(p["t50_min"]):
                second = f"T50 = {p['t50_min']:.1f} min   slope = {p['slope']:.2f}"
                if not regulated_flag[i]:
                    second += f"   ({p['regulation'] or 'unclassified'})"
            else:
                second = "not fitted"
            ax.set_title(f"{head}\n{second}", fontsize=7,)

        for n in range(n_sites, n_rows * n_cols_used,):
            axes[n // n_cols_used, n % n_cols_used].axis("off",)

        n_fitted = int(sum(fitted_flag[i] for i in positions))
        n_reg = int(sum(regulated_flag[i] for i in positions))
        shown = (f"{n_sites} sites plotted" if n_sites == n_selected
                 else f"{n_sites} of {n_selected} sites plotted")
        fig.suptitle(f"{saving_folder} — CurveCurator fits  "
                     f"({n_reg}/{n_fitted} regulated of {shown}) {title_info} ({date.today()})",
                     weight="bold",
                     fontsize=11,)
        fig.supxlabel("time (min) — log10 spacing"
                      + (", control at CurveCurator's zero-dose offset" if show_control
                         else " (starve hidden; it sits 3 decades left)"))
        fig.supylabel("response ratio to starve")
        fig_height = fig.get_size_inches()[1]
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0 - 0.42 / fig_height,),)

        entry = {"fig": fig, "axes": axes,}

        overlay_positions = [i for i in positions if fitted_flag[i]]
        if overlay and overlay_positions:
            n_curves = len(overlay_positions)
            crowded = n_curves > 6
            legend_ncol = int(np.ceil(n_curves / 16,)) if crowded else 1
            o_fig, o_ax = plt.subplots(figsize=(6.5 + 2.6 * legend_ncol if crowded else 6.5,
                                                max(4.4, 0.16 * min(n_curves, 16,) + 2.2,),),)
            if n_curves > 10:
                palette = cycle([plt.get_cmap("tab20",)(j / max(n_curves - 1, 1,))
                                 for j in range(n_curves)],)
            else:
                palette = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"],)

            for i in overlay_positions:
                p = params[i]
                colour = next(palette,)
                t50 = p["t50_min"]
                t_txt = f"T50 {t50:.1f} min" if np.isfinite(t50) else "T50 n/a"
                o_ax.plot(grid,
                          logistic_response(grid, p["pEC50"], p["slope"], p["front"], p["back"],),
                          color=colour,
                          ls="-" if regulated_flag[i] else "--",
                          lw=1.8,
                          label=f"{labels[i]} ({t_txt})"
                                + ("" if regulated_flag[i]
                                   else f" [{p['regulation'] or 'unclassified'}]"),)
                _sel = np.ones_like(treated,) if show_control else treated
                o_ax.plot(x_axis[_sel],
                          observed[i][_sel],
                          "o",
                          ms=3.0,
                          color=colour,
                          alpha=0.4,)
                if np.isfinite(t50) and t50 > 0 and not crowded:
                    o_ax.axvline(np.log10(t50 * dose_scale,),
                                 color=colour,
                                 ls=":",
                                 lw=0.9,
                                 alpha=0.7,)

            o_ax.axhline(1.0, color="grey", lw=0.7,)
            if x_window is not None:
                o_ax.set_xlim(*x_window,)
            o_ax.set_xticks(tick_x,)
            o_ax.set_xticklabels([f"{t:g}" for t in tick_times],)
            o_ax.set_xlabel("time (min) — log10 spacing")
            o_ax.set_ylabel("response ratio to starve")
            o_ax.set_title(f"{saving_folder} — all CurveCurator sigmoids {title_info}",
                           weight="bold",
                           fontsize=10,)
            if crowded:
                o_ax.legend(frameon=False,
                            fontsize=6.5,
                            ncol=legend_ncol,
                            loc="center left",
                            bbox_to_anchor=(1.01, 0.5,),)
                o_fig.subplots_adjust(left=0.09,
                                      right=6.2 / (6.5 + 2.6 * legend_ncol),
                                      top=0.90,
                                      bottom=0.13,)
            else:
                o_ax.legend(frameon=False, fontsize=7, loc="best",)
                o_fig.tight_layout()
            entry["overlay_fig"] = o_fig
            entry["overlay_ax"] = o_ax

        if (save_pdf or save_png) and saving_path:
            base = os.path.join(saving_path,
                                saving_folder,
                                f"{saving_folder}_curvecurator_fits_{saving_info}".rstrip("_",),)
            for ext, flag in [("pdf", save_pdf,), ("png", save_png,)]:
                if not flag:
                    continue
                fig.savefig(f"{base}.{ext}", dpi=200, bbox_inches="tight",)
                print(f"Saved {ext.upper()}: {base}.{ext}")
                if "overlay_fig" in entry:
                    entry["overlay_fig"].savefig(f"{base}_overlay.{ext}",
                                                 dpi=200,
                                                 bbox_inches="tight",)
                    print(f"Saved {ext.upper()}: {base}_overlay.{ext}")
        elif save_pdf or save_png:
            print(f"{saving_folder} — saving_path is empty, plot not saved")

        if plot_close:
            plt.close(fig,)
            if "overlay_fig" in entry:
                plt.close(entry["overlay_fig"],)

        results[protein] = entry

    return results


# ---------------------
# Heatmaps — sites x timepoints
# ---------------------
# Two public functions, plot_fc_heatmap() and plot_step_heatmap(), differing only in the
# data type they colour by:
#
#   log2:FC    value at t vs the starve baseline  -> "how far from baseline is this site now?"
#   log2:step  value at t minus value at t-1      -> "what changed during this interval?"
#
# Both share _plot_value_heatmap() below so the two views of the same sites are drawn
# identically (same row order, same symmetric colour scale, same panel layout) and can be
# read side by side.


def _timepoint_columns(df,
                       cell_line,
                       data_type,
                       condition,
                       timepoints=None,):
    """
    Resolve the data columns of one condition for a chosen, ordered set of timepoints.

    Columns are first selected with ColumnSpec.select() (so the naming convention is applied
    in one place), then matched on the EXACT timepoint field of the column name rather than
    by substring: 'in' matching would let the requested timepoint '1' pull in '15', and '2'
    pull in '2' from another condition.

    Args:
        df: DataFrame following the project naming convention.
        cell_line: cell line prefix, e.g. 'WT'.
        data_type: data type field, e.g. 'log2:FC' or 'log2:step'.
        condition: single condition substring, e.g. '_EGF_'.
        timepoints: ordered list of timepoint labels to keep, e.g. ['2', '5', '90'].
            None (default) keeps every timepoint present, in experimental order.

    Returns:
        Tuple (columns, labels, missing): the matched column names in the requested order,
        their timepoint labels, and the requested labels that have no column in df.
    """
    available = ColumnSpec.select(df,
                                  cell_lines=[cell_line],
                                  data_type=data_type,
                                  conditions=[condition],
                                  exclude_full=False,
                                  exclude_replicate_cols=True,)
    by_timepoint = {}
    for col in available:
        fields = col.split("_")
        if len(fields) >= 4:
            by_timepoint[fields[3]] = col

    if timepoints is None:
        timepoints = _sort_timepoints_numeric(by_timepoint.keys())

    columns = [by_timepoint[tp] for tp in timepoints if tp in by_timepoint]
    labels = [tp for tp in timepoints if tp in by_timepoint]
    missing = [tp for tp in timepoints if tp not in by_timepoint]
    return columns, labels, missing


def _select_heatmap_rows(df,
                         sites=None,
                         proteins=None,
                         site_col="site",):
    """
    Subset the rows (phosphosites) that go on the y-axis of a heatmap.

    Args:
        df: DataFrame with one row per phosphosite.
        sites: list of site keys to keep, matched against `site_col`. The order given is
            preserved, so an explicit list doubles as a manual row order.
        proteins: list of protein names or UniProt accessions; every site of those proteins
            is kept. Combined with `sites` as a union.
        site_col: name of the column holding the site key (default 'site').

    Returns:
        Copy of df restricted to the selected rows, with a fresh unique index — the row
        order is carried by position from here on, so duplicate site keys or a repeated
        `label_col` cannot make a label-based reindex silently multiply rows. If neither
        `sites` nor `proteins` is given, all rows are returned.
    """
    if sites is None and proteins is None:
        return df.copy().reset_index(drop=True,)

    selected = []
    if proteins is not None:
        in_protein = (df.get("protein_name", pd.Series(index=df.index, dtype=object)).isin(proteins)
                      | df.get("protein_Id", pd.Series(index=df.index, dtype=object)).isin(proteins))
        missing_proteins = [p for p in proteins
                            if p not in set(df.get("protein_name", pd.Series(dtype=object)))
                            and p not in set(df.get("protein_Id", pd.Series(dtype=object)))]
        if missing_proteins:
            print(f"  proteins not found in the dataset: {missing_proteins}")
        selected.append(df.loc[in_protein])

    if sites is not None:
        found = df.set_index(site_col).reindex(sites)
        missing_sites = [s for s in sites if s not in set(df[site_col])]
        if missing_sites:
            print(f"  sites not found in the dataset: {missing_sites}")
        selected.append(found.dropna(how="all").reset_index())

    rows = pd.concat(selected) if len(selected) > 1 else selected[0]
    return rows.loc[~rows[site_col].duplicated()].copy().reset_index(drop=True,)


def _order_heatmap_rows(matrix,
                        sort_by="hierarchical",
                        metric="euclidean",
                        method="average",):
    """
    Compute the y-axis order of a heatmap from the matrix that will be plotted.

    The order is computed ONCE on the full matrix (all conditions concatenated), so every
    condition panel shows the same site on the same row and the panels stay comparable.

    Args:
        matrix: DataFrame of plotted values, index = site, columns = the data columns.
        sort_by: 'hierarchical' groups sites by profile similarity (scipy average linkage on
            the row vectors); 'peak' orders by the timepoint of largest absolute value, then
            by that value; 'amplitude' orders by largest absolute value; None keeps the
            incoming order. Any other string is treated as a column of `matrix` to sort on.
        metric: distance metric passed to scipy.linkage when sort_by='hierarchical'.
        method: linkage method passed to scipy.linkage when sort_by='hierarchical'.

    Returns:
        Index in plotting order. Rows carrying NaN are never fed to the linkage (the
        distance would be undefined); they are appended at the bottom in their original
        order and the count is printed.
    """
    if sort_by is None:
        return matrix.index

    if sort_by == "hierarchical":
        complete = matrix.dropna()
        incomplete = matrix.index.difference(complete.index, sort=False)
        if len(incomplete):
            print(f"  {len(incomplete)} site(s) carry NaN and are placed at the bottom "
                  f"(excluded from the linkage)")
        if len(complete) < 3:
            return matrix.index
        from scipy.cluster.hierarchy import linkage, leaves_list
        order = leaves_list(linkage(complete.values, method=method, metric=metric,))
        return complete.index[order].append(incomplete)

    if sort_by in ("peak", "amplitude",):
        absolute = matrix.abs()
        peak_value = absolute.max(axis=1)
        if sort_by == "amplitude":
            return peak_value.sort_values(ascending=False).index
        peak_position = absolute.values.argmax(axis=1)
        ranking = pd.DataFrame({"position": peak_position, "value": -peak_value.values},
                               index=matrix.index,)
        return ranking.sort_values(by=["position", "value"]).index

    if sort_by in matrix.columns:
        return matrix[sort_by].sort_values(ascending=False).index

    raise ValueError(f"sort_by={sort_by!r} is not 'hierarchical', 'peak', 'amplitude', None, "
                     f"or one of the plotted columns.")


def _check_heatmap_row_count(rows,
                             max_sites=300,):
    """
    Raise if a heatmap row selection is empty or too large to be readable.

    Args:
        rows: DataFrame of the selected sites.
        max_sites: maximum number of rows allowed; None disables the guard.

    Returns:
        None. Raises ValueError when the selection is empty or exceeds `max_sites`.
    """
    if rows.empty:
        raise ValueError("No sites left to plot after the sites / proteins selection.")
    if max_sites is not None and len(rows) > max_sites:
        raise ValueError(f"{len(rows)} sites selected but max_sites={max_sites}. A heatmap with "
                         f"that many rows is unreadable — filter the sites first (src/filters.py), "
                         f"pass sites= / proteins=, or set max_sites=None to plot them anyway.")


def _heatmap_matrix(rows,
                    data_type,
                    cell_line="WT",
                    conditions=["_EGF_", "_INS_", "_EGFnINS_"],
                    timepoints=None,):
    """
    Build the value matrix of a heatmap: the selected sites x the selected timepoints.

    The matrix keeps `rows`' own (unique) index rather than the site key, so that ordering
    and plotting are positional and a duplicated site key or `label_col` value cannot
    silently multiply rows through a label-based reindex.

    Args:
        rows: DataFrame of the already-selected sites.
        data_type: data type to read, e.g. 'log2:FC' or 'log2:step'.
        cell_line: cell line prefix, e.g. 'WT'.
        conditions: list of condition substrings, one panel each.
        timepoints: ordered list of timepoint labels; None keeps all, in experimental order.

    Returns:
        Tuple (matrix, panel_columns): the values of every panel concatenated side by side,
        and an ordered dict {condition: (columns, timepoint labels)} describing the panels.
        Conditions with no column for this data type are reported and skipped.
    """
    panel_columns = {}
    for condition in conditions:
        columns, labels, missing = _timepoint_columns(rows,
                                                      cell_line=cell_line,
                                                      data_type=data_type,
                                                      condition=condition,
                                                      timepoints=timepoints,)
        if missing:
            print(f"  {condition.strip('_')}: no {data_type} column for timepoint(s) {missing}")
        if not columns:
            print(f"  {condition.strip('_')}: no {data_type} columns found — panel skipped")
            continue
        panel_columns[condition] = (columns, labels,)

    if not panel_columns:
        raise ValueError(f"No {data_type} columns found for cell_line={cell_line!r}, "
                         f"conditions={conditions}. Check that the data type has been computed.")

    matrix = pd.concat([rows[columns] for columns, _ in panel_columns.values()], axis=1,)
    return matrix, panel_columns


def _heatmap_vmax(matrix,
                  vmax=None,
                  robust=True,):
    """
    Resolve the symmetric colour limit of a heatmap.

    Args:
        matrix: DataFrame of the values to be plotted.
        vmax: explicit limit; returned unchanged when given, so a caller can force a shared
            scale across figures.
        robust: if True, derive the limit from the 99th percentile of |value| instead of the
            maximum, so a few extreme sites do not flatten everything else.

    Returns:
        Positive float used as (-vmax, +vmax). Falls back to 1.0 for an all-zero matrix.
    """
    if vmax is not None:
        return vmax
    values = np.abs(matrix.values.astype(float))
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("The selected columns contain no finite values to plot.")
    limit = np.nanpercentile(values, 99,) if robust else values.max()
    return float(limit) if limit > 0 else 1.0


def _heatmap_figsize(panel_columns,
                     n_rows,
                     row_labels,
                     show_site_labels,):
    """
    Derive a figure size from the shape of one block of panels.

    Args:
        panel_columns: {condition: (columns, labels)} describing one block of panels.
        n_rows: number of sites on the y-axis.
        row_labels: the labels that will be printed on the y-axis (their length drives the
            left margin — site keys are long).
        show_site_labels: whether row labels will be drawn; labelled rows need far more
            height per row than unlabelled ones.

    Returns:
        Tuple (width, height) in inches. Two blocks drawn side by side simply add their
        widths and keep the taller height.
    """
    panel_width = max(1.6, 0.45 * max(len(labels) for _, labels in panel_columns.values()),)
    row_height = 0.16 if show_site_labels else 0.035
    label_width = min(0.07 * max(len(str(label)) for label in row_labels), 6.0,) if show_site_labels else 0.0
    return (2.0 + panel_width * len(panel_columns) + label_width,
            float(np.clip(row_height * n_rows + 2.0, 3.5, 20.0,)),)


def _draw_heatmap_panels(axes,
                         matrix,
                         panel_columns,
                         row_labels,
                         vmax,
                         cmap,
                         show_site_labels,
                         colorbar_label,
                         figure,
                         label_first_panel=True,):
    """
    Draw one block of condition panels into existing axes and attach its colorbar.

    Split out of _plot_value_heatmap() so that the single-data-type figures and the combined
    log2:FC + log2:step figure draw their panels through exactly the same code — the two
    halves of the comparison must be rendered identically for it to be a fair comparison.

    Args:
        axes: sequence of axes, one per condition in `panel_columns`.
        matrix: value matrix in final row order.
        panel_columns: {condition: (columns, labels)} describing the panels.
        row_labels: y-axis labels, in the same order as the matrix rows.
        vmax: symmetric colour limit; the scale is (-vmax, +vmax).
        cmap: diverging matplotlib colormap name.
        show_site_labels: whether to print the row labels.
        colorbar_label: text next to the colorbar.
        figure: the Figure (or SubFigure) that owns the colorbar.
        label_first_panel: whether the leftmost panel of this block carries the y labels;
            False for the right-hand block of the combined figure, which shares the rows
            of the left one.

    Returns:
        The last matplotlib AxesImage drawn (the one the colorbar is built from).
    """
    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad(color="0.85")   # NaN cells stay visibly grey, never imputed to 0
    n_rows = len(matrix)
    image = None

    for panel, (condition, (columns, labels,),) in enumerate(panel_columns.items()):
        ax = axes[panel]
        image = ax.imshow(np.ma.masked_invalid(matrix[columns].values.astype(float)),
                          aspect="auto",
                          cmap=colormap,
                          vmin=-vmax,
                          vmax=vmax,
                          interpolation="nearest",)
        ax.set_xticks(range(len(labels)),)
        ax.set_xticklabels(labels, rotation=45, ha="right",)
        ax.set_title(condition.strip("_"), weight="bold",)
        ax.set_xlabel("timepoint (min)")
        if panel == 0 and label_first_panel:
            if show_site_labels:
                ax.set_yticks(range(n_rows),)
                ax.set_yticklabels(row_labels, fontsize=max(4, min(9, 600 / max(n_rows, 1),)),)
            else:
                ax.set_yticks([],)
            ax.set_ylabel(f"phosphosites (n = {n_rows})", weight="bold",)
        elif panel == 0:
            ax.set_yticks([],)

    colorbar = figure.colorbar(image, ax=list(axes), fraction=0.03, pad=0.02, shrink=0.75,)
    colorbar.set_label(colorbar_label)
    return image


def _save_heatmap(fig,
                  saving_path,
                  file_name,
                  save_pdf=False,
                  save_png=False,):
    """
    Save a heatmap figure following the project's saving convention.

    Args:
        fig: the matplotlib figure to save.
        saving_path: directory for the output (created if needed); empty means do not save.
        file_name: file name without extension.
        save_pdf: save a PDF.
        save_png: save a PNG.

    Returns:
        None. Prints the path of every file written.
    """
    if (save_pdf or save_png) and saving_path:
        os.makedirs(saving_path, exist_ok=True,)
        base = os.path.join(saving_path, file_name,)
        for extension, flag in [("pdf", save_pdf,), ("png", save_png,)]:
            if flag:
                fig.savefig(f"{base}.{extension}", dpi=200, bbox_inches="tight",)
                print(f"Saved {extension.upper()}: {base}.{extension}")
    elif save_pdf or save_png:
        print("saving_path is empty — plot not saved")


def _plot_value_heatmap(df,
                        data_type,
                        cell_line="WT",
                        conditions=["_EGF_", "_INS_", "_EGFnINS_"],
                        timepoints=None,
                        sites=None,
                        proteins=None,
                        sort_by="hierarchical",
                        max_sites=300,
                        show_site_labels="auto",
                        site_col="site",
                        label_col=None,
                        cmap="RdBu_r",
                        vmax=None,
                        robust=True,
                        colorbar_label=None,
                        title=None,
                        figsize=None,
                        saving_path="",
                        saving_info="",
                        save_pdf=False,
                        save_png=False,):
    """
    Draw a sites x timepoints heatmap, one panel per condition, coloured by `data_type`.

    Shared implementation of plot_fc_heatmap() and plot_step_heatmap() — see those for the
    interpretation of the two data types. The colour scale is symmetric around 0 so that the
    diverging colormap reads correctly: white is 'no change', red up, blue down, and the two
    directions are always on the same scale.

    Args:
        df: DataFrame following the project naming convention.
        data_type: data type to colour by, e.g. 'log2:FC' or 'log2:step'.
        cell_line: cell line prefix, e.g. 'WT'.
        conditions: list of condition substrings; one panel is drawn per condition, sharing
            the row order and the colour scale.
        timepoints: ordered list of timepoint labels for the x-axis, e.g. ['2', '5', '90'].
            None keeps every timepoint present, in experimental order.
        sites: list of site keys to plot (also sets the row order when sort_by is None).
        proteins: list of protein names / accessions whose sites are plotted.
        sort_by: row order — 'hierarchical', 'peak', 'amplitude', None, or a plotted column.
        max_sites: guard against plotting an unreadable figure; a selection larger than this
            raises. Pass None to plot everything regardless.
        show_site_labels: True / False, or 'auto' (default) to label rows only when there
            are at most 60 of them.
        site_col: column holding the site key used as the row index (default 'site').
        label_col: optional column used for the row labels instead of `site_col`, e.g.
            'protein_name'.
        cmap: diverging matplotlib colormap name.
        vmax: colour scale limit; the scale is (-vmax, +vmax). None derives it from the data.
        robust: if True (default) and vmax is None, use the 99th percentile of |value|
            rather than the maximum, so a handful of extreme sites do not flatten the plot.
        colorbar_label: text next to the colorbar; defaults to `data_type`.
        title: figure title; a default naming the cell line and data type is built if None.
        figsize: matplotlib figure size; derived from the matrix shape if None.
        saving_path: directory for the saved figure (created if needed).
        saving_info: suffix used in the saved file name.
        save_pdf: save the figure as PDF.
        save_png: save the figure as PNG.

    Returns:
        Tuple (fig, matrix) — the matplotlib figure, and the plotted values as a DataFrame
        indexed by site in plotting order, with the selected data columns of every condition.
    """
    rows = _select_heatmap_rows(df, sites=sites, proteins=proteins, site_col=site_col,)
    _check_heatmap_row_count(rows, max_sites=max_sites,)
    matrix, panel_columns = _heatmap_matrix(rows,
                                            data_type=data_type,
                                            cell_line=cell_line,
                                            conditions=conditions,
                                            timepoints=timepoints,)
    order = _order_heatmap_rows(matrix, sort_by=sort_by,)
    matrix = matrix.loc[order]
    row_labels = (rows[label_col] if label_col is not None else rows[site_col]).loc[order]
    vmax = _heatmap_vmax(matrix, vmax=vmax, robust=robust,)

    n_panels = len(panel_columns)
    n_rows = len(matrix)
    if show_site_labels == "auto":
        show_site_labels = n_rows <= 60
    if figsize is None:
        figsize = _heatmap_figsize(panel_columns,
                                   n_rows=n_rows,
                                   row_labels=row_labels,
                                   show_site_labels=show_site_labels,)

    fig, axes = plt.subplots(1, n_panels,
                             figsize=figsize,
                             squeeze=False,
                             sharey=True,
                             layout="constrained",)
    _draw_heatmap_panels(axes[0],
                         matrix=matrix,
                         panel_columns=panel_columns,
                         row_labels=row_labels,
                         vmax=vmax,
                         cmap=cmap,
                         show_site_labels=show_site_labels,
                         colorbar_label=colorbar_label if colorbar_label is not None else data_type,
                         figure=fig,)

    if title is None:
        title = f"{cell_line} — {data_type}"
    fig.suptitle(title, weight="bold",)

    _save_heatmap(fig,
                  saving_path=saving_path,
                  file_name=f"{cell_line}_{data_type.replace(':', '_')}_heatmap_{saving_info}".rstrip("_",),
                  save_pdf=save_pdf,
                  save_png=save_png,)

    plt.show()
    matrix.index = row_labels.values
    return fig, matrix


def plot_fc_heatmap(df,
                    cell_line="WT",
                    conditions=["_EGF_", "_INS_", "_EGFnINS_"],
                    timepoints=None,
                    sites=None,
                    proteins=None,
                    sort_by="hierarchical",
                    max_sites=300,
                    show_site_labels="auto",
                    site_col="site",
                    label_col=None,
                    cmap="RdBu_r",
                    vmax=None,
                    robust=True,
                    title=None,
                    figsize=None,
                    saving_path="",
                    saving_info="",
                    save_pdf=False,
                    save_png=False,):
    """
    Heatmap of log2:FC — sites on the y-axis, timepoints on the x-axis, one panel per condition.

    Each cell is the fold change of that site at that timepoint against ITS OWN starve control,
    so the whole row is read against one fixed baseline: a site that rises early and stays up
    keeps a strong colour at every later timepoint. Use this to see the state of the system at
    each timepoint. For the increment gained or lost during each interval use
    plot_step_heatmap() instead — the two are complementary views of the same sites.

    The 'starve' timepoint is identically 0 in this data type (FC is defined against it), so it
    is only worth including as a visual zero reference.

    Args:
        df: DataFrame carrying log2:FC columns and a site column.
        cell_line: cell line prefix, e.g. 'WT'.
        conditions: list of condition substrings; one panel each, sharing rows and scale.
        timepoints: ordered list of timepoint labels for the x-axis, e.g. ['2', '5', '90'].
            None (default) plots every timepoint present, in experimental order.
        sites: list of site keys to plot (also the row order when sort_by is None).
        proteins: list of protein names / accessions whose sites are plotted.
        sort_by: row order — 'hierarchical' (default), 'peak', 'amplitude', None, or a column.
        max_sites: refuse to draw more rows than this; None disables the guard.
        show_site_labels: True / False / 'auto' (label rows only when there are <= 60).
        site_col: column holding the site key (default 'site').
        label_col: optional alternative column for the row labels, e.g. 'protein_name'.
        cmap: diverging matplotlib colormap name.
        vmax: colour limit, scale is (-vmax, +vmax); None derives it from the data.
        robust: with vmax=None, use the 99th percentile of |FC| instead of the maximum.
        title: figure title; built from cell_line and data type if None.
        figsize: matplotlib figure size; derived from the matrix shape if None.
        saving_path: directory for the saved figure (created if needed).
        saving_info: suffix used in the saved file name.
        save_pdf: save the figure as PDF.
        save_png: save the figure as PNG.

    Returns:
        Tuple (fig, matrix) — the figure and the plotted log2:FC values as a DataFrame
        indexed by site in plotting order.
    """
    return _plot_value_heatmap(df,
                               data_type="log2:FC",
                               cell_line=cell_line,
                               conditions=conditions,
                               timepoints=timepoints,
                               sites=sites,
                               proteins=proteins,
                               sort_by=sort_by,
                               max_sites=max_sites,
                               show_site_labels=show_site_labels,
                               site_col=site_col,
                               label_col=label_col,
                               cmap=cmap,
                               vmax=vmax,
                               robust=robust,
                               colorbar_label="log2 FC vs starve",
                               title=title,
                               figsize=figsize,
                               saving_path=saving_path,
                               saving_info=saving_info,
                               save_pdf=save_pdf,
                               save_png=save_png,)


def plot_step_heatmap(df,
                      cell_line="WT",
                      conditions=["_EGF_", "_INS_", "_EGFnINS_"],
                      timepoints=None,
                      sites=None,
                      proteins=None,
                      sort_by="hierarchical",
                      max_sites=300,
                      show_site_labels="auto",
                      site_col="site",
                      label_col=None,
                      cmap="PRGn",
                      vmax=None,
                      robust=True,
                      title=None,
                      figsize=None,
                      saving_path="",
                      saving_info="",
                      save_pdf=False,
                      save_png=False,):
    """
    Heatmap of log2:step — sites on the y-axis, timepoints on the x-axis, one panel per condition.

    Each cell is the CHANGE during the interval ending at that timepoint, log2:FC(t) - log2:FC(t-1)
    (see log2_step_size() in src/transformations.py), so a colour marks where movement happened
    rather than where the site currently stands: a site that rises at 2 min and then holds shows
    one strong cell at 2 min and near-white cells afterwards, whereas plot_fc_heatmap() would
    show it strong throughout. White therefore means 'nothing changed during this interval',
    not 'back to baseline'.

    Only timepoints after starve have a log2:step column, so 'full' and 'starve' never appear.
    A different default colormap (PRGn) is used from plot_fc_heatmap() so the two figures are
    not mistaken for each other.

    Note that the intervals are unequal (2, 5, 10, 15, 90 min), so the last column covers 75
    minutes while the first covers 2 — a strong late cell is not necessarily a fast change.

    Args:
        df: DataFrame carrying log2:step columns and a site column.
        cell_line: cell line prefix, e.g. 'WT'.
        conditions: list of condition substrings; one panel each, sharing rows and scale.
        timepoints: ordered list of timepoint labels for the x-axis, e.g. ['2', '5', '90'].
            None (default) plots every timepoint present, in experimental order.
        sites: list of site keys to plot (also the row order when sort_by is None).
        proteins: list of protein names / accessions whose sites are plotted.
        sort_by: row order — 'hierarchical' (default), 'peak', 'amplitude', None, or a column.
        max_sites: refuse to draw more rows than this; None disables the guard.
        show_site_labels: True / False / 'auto' (label rows only when there are <= 60).
        site_col: column holding the site key (default 'site').
        label_col: optional alternative column for the row labels, e.g. 'protein_name'.
        cmap: diverging matplotlib colormap name.
        vmax: colour limit, scale is (-vmax, +vmax); None derives it from the data.
        robust: with vmax=None, use the 99th percentile of |step| instead of the maximum.
        title: figure title; built from cell_line and data type if None.
        figsize: matplotlib figure size; derived from the matrix shape if None.
        saving_path: directory for the saved figure (created if needed).
        saving_info: suffix used in the saved file name.
        save_pdf: save the figure as PDF.
        save_png: save the figure as PNG.

    Returns:
        Tuple (fig, matrix) — the figure and the plotted log2:step values as a DataFrame
        indexed by site in plotting order.
    """
    return _plot_value_heatmap(df,
                               data_type="log2:step",
                               cell_line=cell_line,
                               conditions=conditions,
                               timepoints=timepoints,
                               sites=sites,
                               proteins=proteins,
                               sort_by=sort_by,
                               max_sites=max_sites,
                               show_site_labels=show_site_labels,
                               site_col=site_col,
                               label_col=label_col,
                               cmap=cmap,
                               vmax=vmax,
                               robust=robust,
                               colorbar_label="log2 step vs previous timepoint",
                               title=title,
                               figsize=figsize,
                               saving_path=saving_path,
                               saving_info=saving_info,
                               save_pdf=save_pdf,
                               save_png=save_png,)


def plot_fc_step_heatmap(df,
                         cell_line="WT",
                         conditions=["_EGF_", "_INS_", "_EGFnINS_"],
                         timepoints=None,
                         sites=None,
                         proteins=None,
                         sort_by="hierarchical",
                         sort_on="log2:FC",
                         max_sites=300,
                         show_site_labels="auto",
                         site_col="site",
                         label_col=None,
                         fc_cmap="RdBu_r",
                         step_cmap="PRGn",
                         fc_vmax=None,
                         step_vmax=None,
                         robust=True,
                         title=None,
                         figsize=None,
                         saving_path="",
                         saving_info="",
                         save_pdf=False,
                         save_png=False,):
    """
    Draw the log2:FC and log2:step heatmaps of the SAME sites side by side, as two subfigures.

    This is the comparison figure: plot_fc_heatmap() and plot_step_heatmap() each answer half
    the question, and the answers are only comparable if the two panels put the same site on
    the same row. Here the rows are selected once and ordered once (see `sort_on`), then both
    halves are drawn from that single order — reading straight across a row gives the state of
    a site on the left and the movement that produced it on the right.

    Left block (log2:FC, red/blue): value against the starve control, so a sustained response
    stays coloured at every later timepoint. Right block (log2:step, green/purple): value minus
    the previous timepoint, so only the intervals where something actually changed are
    coloured. A row that is uniformly red on the left with a single dark green cell on the
    right is a step change that then held; a row that stays coloured on both sides is still
    moving. The two blocks get SEPARATE colour scales, because a fold change against baseline
    and an increment between two timepoints are different quantities — the scales are printed
    on their own colorbars and should not be read against each other.

    Args:
        df: DataFrame carrying both log2:FC and log2:step columns.
        cell_line: cell line prefix, e.g. 'WT'.
        conditions: list of condition substrings; one panel per condition inside each block.
        timepoints: ordered list of timepoint labels, e.g. ['2', '5', '90']. None keeps every
            timepoint present. 'full' and 'starve' exist only for log2:FC, so asking for them
            gives an FC column and a reported gap on the step side — which is correct, not a
            failure: no step is defined into the first timepoint of the series.
        sites: list of site keys to plot (also the row order when sort_by is None).
        proteins: list of protein names / accessions whose sites are plotted.
        sort_by: row order — 'hierarchical' (default), 'peak', 'amplitude', None, or a column.
        sort_on: which matrix the row order is computed from — 'log2:FC' (default), 'log2:step',
            or 'both' to order on the two concatenated. Ordering on log2:FC is usually what you
            want: it groups sites by response shape and lets the step block show where each
            shape's movement sits. Ordering on log2:step instead groups sites by WHEN they move.
        max_sites: refuse to draw more rows than this; None disables the guard.
        show_site_labels: True / False / 'auto' (label rows only when there are <= 60).
        site_col: column holding the site key (default 'site').
        label_col: optional alternative column for the row labels, e.g. 'protein_name'.
        fc_cmap: diverging colormap of the log2:FC block.
        step_cmap: diverging colormap of the log2:step block.
        fc_vmax: colour limit of the FC block, scale (-fc_vmax, +fc_vmax); None derives it.
        step_vmax: colour limit of the step block; None derives it.
        robust: with a vmax left as None, use the 99th percentile of |value| instead of the max.
        title: figure title; built from the cell line if None.
        figsize: matplotlib figure size; derived from the matrix shape if None.
        saving_path: directory for the saved figure (created if needed).
        saving_info: suffix used in the saved file name.
        save_pdf: save the figure as PDF.
        save_png: save the figure as PNG.

    Returns:
        Tuple (fig, matrices) where matrices is {'log2:FC': DataFrame, 'log2:step': DataFrame},
        both indexed by site in the shared plotting order so they can be compared row by row.
    """
    rows = _select_heatmap_rows(df, sites=sites, proteins=proteins, site_col=site_col,)
    _check_heatmap_row_count(rows, max_sites=max_sites,)

    matrices = {}
    panels = {}
    for data_type in ("log2:FC", "log2:step",):
        matrices[data_type], panels[data_type] = _heatmap_matrix(rows,
                                                                 data_type=data_type,
                                                                 cell_line=cell_line,
                                                                 conditions=conditions,
                                                                 timepoints=timepoints,)

    if sort_on == "both":
        basis = pd.concat([matrices["log2:FC"], matrices["log2:step"]], axis=1,)
    elif sort_on in matrices:
        basis = matrices[sort_on]
    else:
        raise ValueError(f"sort_on={sort_on!r} must be 'log2:FC', 'log2:step' or 'both'.")

    order = _order_heatmap_rows(basis, sort_by=sort_by,)
    matrices = {data_type: matrix.loc[order] for data_type, matrix in matrices.items()}
    row_labels = (rows[label_col] if label_col is not None else rows[site_col]).loc[order]

    n_rows = len(order)
    if show_site_labels == "auto":
        show_site_labels = n_rows <= 60
    # Both block widths are always needed: they set the subfigure split as well as the default
    # figure size. Splitting on panel count alone would ignore the y-label margin, which lives
    # inside the left block and would leave a gap between the two halves.
    fc_size = _heatmap_figsize(panels["log2:FC"],
                               n_rows=n_rows,
                               row_labels=row_labels,
                               show_site_labels=show_site_labels,)
    step_size = _heatmap_figsize(panels["log2:step"],
                                 n_rows=n_rows,
                                 row_labels=row_labels,
                                 show_site_labels=False,)
    if figsize is None:
        figsize = (fc_size[0] + step_size[0], fc_size[1],)

    fig = plt.figure(figsize=figsize, layout="constrained",)
    blocks = fig.subfigures(1, 2,
                            width_ratios=[fc_size[0], step_size[0]],
                            wspace=0.02,)

    settings = {"log2:FC": (blocks[0], fc_cmap, fc_vmax, "log2 FC vs starve", True,),
                "log2:step": (blocks[1], step_cmap, step_vmax, "log2 step vs previous timepoint", False,),}
    for data_type, (block, cmap, vmax, colorbar_label, label_rows,) in settings.items():
        axes = block.subplots(1, len(panels[data_type]), squeeze=False, sharey=True,)
        _draw_heatmap_panels(axes[0],
                             matrix=matrices[data_type],
                             panel_columns=panels[data_type],
                             row_labels=row_labels,
                             vmax=_heatmap_vmax(matrices[data_type], vmax=vmax, robust=robust,),
                             cmap=cmap,
                             show_site_labels=show_site_labels,
                             colorbar_label=colorbar_label,
                             figure=block,
                             label_first_panel=label_rows,)
        block.suptitle(f"{data_type}", weight="bold", fontsize=12,)

    fig.suptitle(title if title is not None else f"{cell_line} — log2:FC vs log2:step",
                 weight="bold",
                 fontsize=13,)

    _save_heatmap(fig,
                  saving_path=saving_path,
                  file_name=f"{cell_line}_FC_vs_step_heatmap_{saving_info}".rstrip("_",),
                  save_pdf=save_pdf,
                  save_png=save_png,)

    plt.show()
    for matrix in matrices.values():
        matrix.index = row_labels.values
    return fig, matrices
