import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from datetime import date
from IPython.display import display, HTML
from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, KShape

import networkx as nx
from itertools import cycle
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from matplotlib_venn import venn2
import tslearn as tsl
import math
from matplotlib.lines import Line2D

from tslearn.metrics import cdist_dtw
from sklearn.metrics import silhouette_score

import warnings

#%%
def plot_data_refined(ax,
                      row_df,
                      data_type=str,
                      colors={},
                      cell_lines=[],
                      conditions=[],
                      # exclude_rep = list,
                      # plot_individually = False,
                      # legend = [],
                      # replicates = False,
                      ):
    column_names = row_df.index.tolist()
    sub_dtp = data_type.split(":")  # e.g. ["log2", "FC"]

    # Build a lookup: columns_dir[cell][condition] -> {"plotting": [...], "sd": [...]}
    columns_dir = {cell: {condition: {} for condition in conditions} for cell in cell_lines}

    for cell in cell_lines:
        for condition in conditions:
            columns_dir[cell][condition]["plotting"] = [col for col in column_names
                                                        if col.startswith(cell)
                                                        and condition in col
                                                        and data_type == col.split("_")[1]
                                                        and sub_dtp[0] in col
                                                        and sub_dtp[1] in col
                                                        ]
            columns_dir[cell][condition]["sd"] = [col for col in column_names
                                                  if col.startswith(cell)
                                                  and condition in col
                                                  and sub_dtp[0] in col
                                                  and "sd" in col
                                                  ]
    # Derive x-axis time points from the first valid cell/condition combination
    x_axis_previous = [element for element in column_names if
                       f"{cell_lines[0]}_{data_type}{conditions[0]}" in element]  # this could be any set of columns that have the time points
    x_axis = [s.split("_")[3] for s in x_axis_previous]

    site = row_df["site"].split("~")[0]
    seq = row_df["site"].split("~")[1]
    prot_name = row_df["protein_name"]
    protein_ID = row_df["protein_Id"]
    n_rep = row_df["n_rep"]

    for condition in conditions:
        for cell in cell_lines:
            ax.errorbar(
                x=x_axis,
                y=row_df[columns_dir[cell][condition]["plotting"]].values.astype(float),
                yerr=row_df[columns_dir[cell][condition]["sd"]].values.astype(float),
                marker='o',
                color=colors[condition],
                label=condition, # cell
                capsize=4,
                elinewidth=1.3,
                alpha=1
            )
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(f"{data_type}")
    # ax.grid()
    ax.set_title(f"{site}_n{n_rep}")

#%%
def plot_protein_phosphosites_refined(df,
                                      data_type=str,
                                      proteins=list,
                                      cell_lines=[],
                                      conditions=[],
                                      legend_plot = list,
                                      color_palette={"_EGF_": "red",
                                                     "_INS_": "blue",
                                                     "_EGFnINS_": "fuchsia"},
                                      saving_path=str,
                                      saving_info="",
                                      title_info="",
                                      fit_y_lims=False,
                                      plot_close=False,
                                      save_pdf=False,
                                      save_png=False,
                                      # replicates = False,
                                      # exclude_rep = list,
                                      # plot_individually=False,
                                      ):
    '''Plot to PDF ALL phosphosites of a list of proteins. You can decide to plot the phosphosties of the protein
    together in one plot or to plot them separatly.'''

    # Load dataframe if a path was passed instead
    if isinstance(df, pd.DataFrame):
        pass
    elif isinstance(df, str):
        if df.endswith(".xlsx"):
            df = pd.read_excel(df)
        elif df.endswith(".tsv"):
            df = pd.read_csv(df, sep="\t")
        else:
            raise ValueError("Unsupported file format. Use .xlsx or .tsv")

    sub_dtp = data_type.split(":")  # e.g. ["log2", "FC"]

    # Check x_axis
    x_axis_previous = [element for element in df.columns if
                       f"{cell_lines[0]}_{data_type}{conditions[0]}" in element]  # this could be any set of columns that have the time points
    x_axis = [s.split("_")[3] for s in x_axis_previous]
    len_x = len(x_axis)

    for protein in proteins:
        # Create sub-dataframe with only the protein we are interested in. If the protein doesn't exist in the dataframe skip code
        if protein in df['protein_name'].to_list():
            sub_df = df.loc[df['protein_name'] == protein].copy()
        elif protein in df['protein_Id'].to_list():
            sub_df = df.loc[df['protein_Id'] == protein].copy()
        else:
            print(f"The protein {protein} is not present in the dataset")
            continue
        print(f"Ploting sites of protein {protein}")

        # Extract the protein name and protein uniprot code for the folder
        saving_folder = f"{sub_df['protein_name'].iloc[0]}_{sub_df['protein_Id'].iloc[0]}"

        # Create output folder if saving
        if (save_pdf or save_png) and saving_path:
            new_path = os.path.join(saving_path, saving_folder)
            os.makedirs(new_path, exist_ok=True)  # cleaner than manual check

        # Sort the pepetides of the dataframe for better interpretation of the figure generated
        sub_df.sort_values(by=['site'], inplace=True)

        # Determine the dimentional space for the subplot
        number_phos = len(sub_df)
        sqrt_n_p = int(np.ceil(np.sqrt(number_phos)))
        sqrt_n_p_X = sqrt_n_p

        if sqrt_n_p > 2:
            empty_plots = (sqrt_n_p * sqrt_n_p) - number_phos
            if empty_plots >= sqrt_n_p:
                sqrt_n_p_X = sqrt_n_p - 1

        # Identify value columns for y-limit calculation
        column_names = df.columns.tolist()

        # Build a lookup: columns_dir[cell][condition] -> {"plotting": [...], "sd": [...]}
        # columns_dir = {cell: {condition: {} for condition in conditions} for cell in cell_lines}
        column_selection = []

        for cell in cell_lines:
            for condition in conditions:
                # columns_dir[cell][condition]["y_lim"] = [col for col in column_names
                new_cols = [col for col in column_names
                            if col.startswith(cell)
                            and condition in col
                            and data_type == col.split("_")[1]
                            ]
                column_selection = column_selection + new_cols

        # Resolve y-axis limits up front
        if isinstance(fit_y_lims, list):
            y_lim_min, y_lim_max = fit_y_lims[0], fit_y_lims[1]
            y_limt_info = f"_y_axis_fixed_{y_lim_min}_{y_lim_max}"
            use_fixed_ylims = True
        elif fit_y_lims is False:
            sub_values_df = sub_df[column_selection] if column_selection else pd.DataFrame()
            if not sub_values_df.empty:
                y_lim_max = sub_values_df.max().max() * 1.1
                y_lim_min = sub_values_df.min().min()
                y_lim_min = y_lim_min + y_lim_min * 0.1 if y_lim_min < 0 else y_lim_min * 0.97
            else:
                y_lim_min, y_lim_max = None, None
            use_fixed_ylims = False
            y_limt_info = ""
        else:  # fit_y_lims is True -> per-row dynamic limits
            use_fixed_ylims = None  # signals "compute per row"
            y_limt_info = "y_axis_perrow"

        # Build figure
        fig, axes = plt.subplots(sqrt_n_p, sqrt_n_p_X, figsize=(18, 13))
        fig.tight_layout(w_pad=1.75, h_pad=3)
        plt.subplots_adjust(top=0.94)

        # Normalise axes to always be a 2D array
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

                plot_data_refined(ax=axes[i, j],
                                  row_df=row,
                                  data_type=data_type,
                                  colors=color_palette,
                                  # legend=legend_plot,
                                  cell_lines=cell_lines,
                                  conditions=conditions
                                  )

                # Apply y-limits
                if use_fixed_ylims is None:  # per-row dynamic
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
        # # print(color_palette.keys())
        # legend_handles = [
        #     Line2D([0], [0], color=color, linewidth=2, marker='o', label=condition)
        #     for condition, color in color_palette.items()
        # ]
        # fig.legend(handles=legend_handles, loc="upper right", ncol=len(color_palette))
        fig.legend(labels=legend_plot, loc="upper right", ncol=len(legend_plot))
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
#%%
def filter_dynamics_extremes_refined(
        df,
        data_type="log2:FC",
        threshold=0.5,
        exclude_full=False,
        conditions=["_EGF_"],
        cell_lines=["WT", "BRAFS151A", "GAB1Y259A"],
):
    """

    """
    # Select relevant columns
    column_names = df.columns.tolist()
    clustering_dic = {cell: {condition: {} for condition in conditions} for cell in cell_lines}
    # print(column_names)
    for cell in cell_lines:
        for condition in conditions:
            clustering_dic[cell][condition] = [col for col in column_names
                                               if col.startswith(cell)
                                               and condition in col
                                               and data_type == col.split("_")[1]]
            if exclude_full == True:
                clustering_dic[cell][condition] = [col for col in list(clustering_dic[cell][condition]) if
                                                   "full" not in col]

    # print(clustering_dic)
    column_selection = [value for key1 in clustering_dic
                        for key2 in clustering_dic[key1]
                        for value in clustering_dic[key1][key2]]
    # print(column_selection)

    mask = df[column_selection].abs().max(axis=1) >= threshold
    return df.loc[mask].copy()