import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from datetime import date
from IPython.display import display, HTML
from itertools import cycle
import math
import warnings

import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from matplotlib_venn import venn2
from matplotlib.lines import Line2D

from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, KShape
from tslearn.metrics import cdist_dtw
import tslearn as tsl

from sklearn.metrics import silhouette_score


# =============================================================================
# Utility
# =============================================================================

def uniprot_links_for(df, protein_list=[]):
    for protein in protein_list:
        if (protein not in df['protein_name'].values) and (protein not in df['protein_Id'].values):
            uniprot_url = f"https://www.uniprot.org/uniprotkb/{protein}"
            html_link = f'Protein {protein} is not in the database: <a href="{uniprot_url}" target="_blank">{protein}</a>'
            display(HTML(html_link))
        elif protein in df['protein_name'].values:
            protein_ID = df.loc[df['protein_name'] == protein, 'protein_Id'].values[0]
            protein_description = df.loc[df['protein_name'] == protein, 'description'].values[0]
            protein_description = protein_description.split("OS=")[0]
            uniprot_url = f"https://www.uniprot.org/uniprotkb/{protein_ID}"
            html_link = f'Link to protein {protein} <a href="{uniprot_url}" target="_blank">{protein_ID}</a>. {protein_description}'
            display(HTML(html_link))
        else:
            protein_name = df.loc[df['protein_Id'] == protein, 'protein_name'].values[0]
            protein_description = df.loc[df['protein_name'] == protein, 'description'].values[0]
            protein_description = protein_description.split("OS=")[0]
            uniprot_url = f"https://www.uniprot.org/uniprotkb/{protein}"
            html_link = f'Link to protein {protein_name} <a href="{uniprot_url}" target="_blank">{protein}</a>. {protein_description}'
            display(HTML(html_link))


def reshape_df(df,
               time_series,
               dimensions,
               len_time_serie,
               verbose,
               labels="site",
               transpose=False):
    '''Reshape dataframe so it is multivariate format. Return the dataframe in numpy format so can be used, and list with the names of myseries'''
    sub_df = df[time_series].copy()
    mySeries = sub_df.to_numpy()
    namesofMySeries = df[labels]

    multivariate_shape = (len(df), dimensions, len_time_serie)
    if verbose == True and transpose == False:
        print(f"Reshaping dataframe to shape {multivariate_shape}")

    multivariate_df = np.reshape(mySeries, multivariate_shape)
    if transpose == True:
        multivariate_df = multivariate_df.transpose(0, 2, 1)
        if verbose == True:
            print(f"Reshaping dataframe to shape {multivariate_df.shape}")

    return multivariate_df, namesofMySeries


# =============================================================================
# Filtering
# =============================================================================

def filter_dynamics(df, data_type="log2_FC", threshold=0.5, mode="extremes", exclude_full=True):
    """
    Filter rows by the maximum absolute fold-change across the time series.

    Args:
        df: DataFrame with time-series columns.
        data_type: column prefix to select (e.g. "log2_FC" or "WT_log2:FC").
        threshold: absolute fold-change cutoff.
        mode: 'extremes' keeps rows where max|value| >= threshold (dynamic peptides);
              'within'   keeps rows where max|value| <= threshold (stable peptides).
        exclude_full: if True, drop columns containing 'full' from the selection.
    """
    if exclude_full:
        column_selection = [c for c in df.columns if c.startswith(data_type) and "full" not in c]
    else:
        column_selection = [c for c in df.columns if data_type in c and "statistics" not in c]

    if mode == "extremes":
        mask = df[column_selection].abs().max(axis=1) >= threshold
    elif mode == "within":
        mask = df[column_selection].abs().max(axis=1) <= threshold
    else:
        raise ValueError(f"mode must be 'extremes' or 'within', got {mode!r}")
    return df.loc[mask].copy()


def filter_dynamics_extremes(df, data_type="log2_FC", threshold=0.5, exclude_full=True):
    """Backward-compatible alias for filter_dynamics(..., mode='extremes')."""
    return filter_dynamics(df, data_type=data_type, threshold=threshold, mode="extremes", exclude_full=exclude_full)


def filter_dynamics_within(df, data_type="log2_FC", threshold=0.5, exclude_full=True):
    """Backward-compatible alias for filter_dynamics(..., mode='within')."""
    return filter_dynamics(df, data_type=data_type, threshold=threshold, mode="within", exclude_full=exclude_full)


def dynamics_values(df, data_type="log2_FC", exclude_full=True):
    """Return only the time-series value columns for the given data_type."""
    if exclude_full:
        column_selection = [c for c in df.columns if c.startswith(data_type) and "full" not in c]
    else:
        column_selection = [c for c in df.columns if data_type in c and "statistics" not in c]
    return df[column_selection].copy()


def filter_replicates(df, n_reps):
    """Return rows with at least n_reps replicates detected."""
    return df.loc[df["n_rep"] >= n_reps]


def filter_site_localizations(df, loc_sites=False):
    if loc_sites == False:
        return df.loc[df["localized_sites"] == 0]
    else:
        return df.loc[df["localized_sites"] > 0]


def filter_ERK_motif(df, ERK_motif=False):
    if ERK_motif == False:
        return df.loc[df["ERK_motif"] == 0]
    else:
        return df.loc[df["ERK_motif"] == 1]


def filter_functional_score(df, f_score):
    """Return rows whose functional_score is >= f_score."""
    return df.loc[df["functional_score"] >= f_score]


def filter_dynamics_extremes_refined(
        df,
        data_type="log2:FC",
        threshold=0.5,
        exclude_full=False,
        conditions=["_EGF_"],
        cell_lines=["WT", "BRAFS151A", "GAB1Y259A"],
):
    """
    Refined version of filter_dynamics_extremes for the multi-cell-line naming convention:
    (CellLine)_(DataType)_(Treatment)_(TimePoint)_(Replicate).
    """
    column_names = df.columns.tolist()
    clustering_dic = {cell: {condition: {} for condition in conditions} for cell in cell_lines}

    for cell in cell_lines:
        for condition in conditions:
            clustering_dic[cell][condition] = [col for col in column_names
                                               if col.startswith(cell)
                                               and condition in col
                                               and data_type == col.split("_")[1]]
            if exclude_full == True:
                clustering_dic[cell][condition] = [col for col in list(clustering_dic[cell][condition]) if
                                                   "full" not in col]

    column_selection = [value for key1 in clustering_dic
                        for key2 in clustering_dic[key1]
                        for value in clustering_dic[key1][key2]]

    mask = df[column_selection].abs().max(axis=1) >= threshold
    return df.loc[mask].copy()


# =============================================================================
# Plotting — original naming convention (single cell line datasets)
# =============================================================================

def plot_data(ax,
              row_df,
              replicates=False,
              data_type="",
              colors=[],
              legend=[],
              exclude_rep=None,
              plot_individually=False):

    if exclude_rep is None:
        exclude_rep = []

    column_names = row_df.index.tolist()

    EGF_mean = []
    INS_mean = []
    EGFnINS_mean = []
    EGF_sd = []
    INS_sd = []
    EGFnINS_sd = []

    if data_type == "raw" or data_type == "log2":
        EGF_mean = [element for element in column_names if f"{data_type}_mean_EGF_" in element]
        INS_mean = [element for element in column_names if f"{data_type}_mean_INS_" in element]
        EGFnINS_mean = [element for element in column_names if f"{data_type}_mean_EGFnINS_" in element]

        EGF_sd = [element for element in column_names if f"{data_type}_sd_EGF_" in element]
        INS_sd = [element for element in column_names if f"{data_type}_sd_INS_" in element]
        EGFnINS_sd = [element for element in column_names if f"{data_type}_sd_EGFnINS_" in element]

    elif data_type == "log2_FC":
        EGF_mean = [element for element in column_names if f"{data_type}_EGF_" in element]
        INS_mean = [element for element in column_names if f"{data_type}_INS_" in element]
        EGFnINS_mean = [element for element in column_names if f"{data_type}_EGFnINS_" in element]

        data_type = "log2"
        EGF_sd = [element for element in column_names if f"{data_type}_sd_EGF_" in element]
        INS_sd = [element for element in column_names if f"{data_type}_sd_INS_" in element]
        EGFnINS_sd = [element for element in column_names if f"{data_type}_sd_EGFnINS_" in element]
        data_type = "log2_FC"

    elif data_type == "FC_scaled":
        EGF_mean = [element for element in column_names if f"{data_type}_EGF_" in element]
        INS_mean = [element for element in column_names if f"{data_type}_INS_" in element]
        EGFnINS_mean = [element for element in column_names if f"{data_type}_EGFnINS_" in element]

    if data_type == "raw":
        data_type2 = "raw_abs"
    else:
        data_type2 = "log2_abs"

    EGF_r1 = [element for element in column_names if f"{data_type2}_EGF_" in element and "r1" in element]
    EGF_r2 = [element for element in column_names if f"{data_type2}_EGF_" in element and "r2" in element]
    EGF_r3 = [element for element in column_names if f"{data_type2}_EGF_" in element and "r3" in element]
    EGF_r4 = [element for element in column_names if f"{data_type2}_EGF_" in element and "r4" in element]

    INS_r1 = [element for element in column_names if f"{data_type2}_INS_" in element and "r1" in element]
    INS_r2 = [element for element in column_names if f"{data_type2}_INS_" in element and "r2" in element]
    INS_r3 = [element for element in column_names if f"{data_type2}_INS_" in element and "r3" in element]
    INS_r4 = [element for element in column_names if f"{data_type2}_INS_" in element and "r4" in element]

    EGFnINS_r1 = [element for element in column_names if f"{data_type2}_EGFnINS_" in element and "r1" in element]
    EGFnINS_r2 = [element for element in column_names if f"{data_type2}_EGFnINS_" in element and "r2" in element]
    EGFnINS_r3 = [element for element in column_names if f"{data_type2}_EGFnINS_" in element and "r3" in element]
    EGFnINS_r4 = [element for element in column_names if f"{data_type2}_EGFnINS_" in element and "r4" in element]

    rep_list = {"rep1": [EGF_r1, INS_r1, EGFnINS_r1],
                "rep2": [EGF_r2, INS_r2, EGFnINS_r2],
                "rep3": [EGF_r3, INS_r3, EGFnINS_r3],
                "rep4": [EGF_r4, INS_r4, EGFnINS_r4]}

    x_axis_previous = [element for element in column_names if f"log2_FC_EGF_" in element]
    x_axis = [s.split("_")[-1] for s in x_axis_previous]

    if data_type in ["raw", "log2", "log2_FC", "FC_scaled"]:
        mean_all = [EGF_mean, INS_mean, EGFnINS_mean]
        sd_all = [EGF_sd, INS_sd, EGFnINS_sd]

        n_rep = row_df["n_rep"]
        site = row_df["site"]

        for c in range(3):
            al = 0.3 if n_rep == 1 else 1
            y_error = row_df[sd_all[c]]
            if data_type == "FC_scaled":
                y_error = [0, 0, 0, 0, 0, 0, 0]
            ax.errorbar(x=x_axis, y=row_df[mean_all[c]], yerr=y_error, marker='o',
                        color=colors[c], label=legend[c], capsize=4, elinewidth=1.3, alpha=al)

        if n_rep == 1:
            replicates = False
        if replicates == True:
            for element in rep_list:
                if element in exclude_rep:
                    continue
                else:
                    c = 0
                    for condition in rep_list[element]:
                        if len(set(row_df[condition].values)) == 1:
                            continue
                        else:
                            ax.scatter(x=x_axis, y=row_df[condition], marker='x', color=colors[c], alpha=0.7, s=20)
                            c += 1

        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(f"{data_type}")
        ax.grid()

        if plot_individually == True:
            ax.legend()
            ax.set_title(f"{site}_n{n_rep}")
        else:
            splited = site.split("~")
            ax.set_title(f"{splited[0]}_n{n_rep}")
            ax.title.set_size(10)

        return ax

    else:
        print(f"Your data type {data_type} is not supported. Try one of these: 'raw', 'log2', 'log2_FC', 'FC_scaled' ")


def plot_dataset_phosphosites(df,
                              cluster_column="",
                              cluster_number=None,
                              data_type="",
                              legend=None,
                              color_palette=['r', 'b', 'fuchsia'],
                              saving_path="",
                              dataset_name="",
                              saving_info="",
                              plot_individually=False,
                              fit_y_lims=False,
                              plot_close=False,
                              save_pdf=False,
                              save_png=False):
    """Plot all phosphorylation sites of a dataset. Set plot_individually=True for separate axes."""

    if legend is None:
        legend = []

    if not isinstance(df, pd.DataFrame):
        df = pd.read_excel(df)

    if cluster_column != "":
        if cluster_number:
            df = df.loc[df[cluster_column] == int(cluster_number)]

    column_names = df.columns.tolist()
    column_selection = [element for element in column_names if data_type in element]

    time_points_previous = [element for element in column_names if f"log2_FC_EGF_" in element]
    time_points = [s.split("_")[-1] for s in time_points_previous]

    EGF_mean_cols = [col for col in df.columns if any(f"{data_type}_EGF_{t}" in col for t in time_points)]
    INS_mean_cols = [col for col in df.columns if any(f"{data_type}_INS_{t}" in col for t in time_points)]
    EGFnINS_mean_cols = [col for col in df.columns if any(f"{data_type}_EGFnINS_{t}" in col for t in time_points)]

    if "raw" in data_type:
        EGF_sd_cols = [col for col in df.columns if any(f"raw_sd_EGF_{t}" in col for t in time_points)]
        INS_sd_cols = [col for col in df.columns if any(f"raw_sd_INS_{t}" in col for t in time_points)]
        EGFnINS_sd_cols = [col for col in df.columns if any(f"raw_sd_EGFnINS_{t}" in col for t in time_points)]
    else:
        EGF_sd_cols = [col for col in df.columns if any(f"log2_sd_EGF_{t}" in col for t in time_points)]
        INS_sd_cols = [col for col in df.columns if any(f"log2_sd_INS_{t}" in col for t in time_points)]
        EGFnINS_sd_cols = [col for col in df.columns if any(f"log2_sd_EGFnINS_{t}" in col for t in time_points)]

    if save_pdf == False and save_png == False:
        pass
    else:
        if not os.path.exists(saving_path):
            print("Creating saving folder")
            os.makedirs(saving_path)

    df = df.sort_values(by=['site'])
    number_phos = len(df)

    if number_phos == 1 or plot_individually == True:
        for index, row in df.iterrows():
            site = row["site"]
            name = row["protein_name"]
            id = row["protein_Id"]
            all_times = row[column_selection].tolist()
            EGF = row[EGF_mean_cols].tolist()
            INS = row[INS_mean_cols].tolist()
            EGFnINS = row[EGFnINS_mean_cols].tolist()
            groups = [EGF, INS, EGFnINS]
            EGF_sd = row[EGF_sd_cols].tolist()
            INS_sd = row[INS_sd_cols].tolist()
            EGFnINS_sd = row[EGFnINS_sd_cols].tolist()
            groups_sd = [EGF_sd, INS_sd, EGFnINS_sd]
            n_rep = row["n_rep"]

            uniprot_url = f"https://www.uniprot.org/uniprotkb/{id}"
            html_link = f'Link to protein {name} <a href="{uniprot_url}" target="_blank">{id}</a>'
            display(HTML(html_link))

            fig, ax = plt.subplots(figsize=(7, 4))
            for c in range(3):
                al = 0.3 if n_rep == 1 else 1
                ax.errorbar(x=time_points, y=groups[c], yerr=groups_sd[c], marker='o',
                            color=color_palette[c], label=legend[c], capsize=4, elinewidth=1.3, alpha=al)

            ax.set_title(f"{str(re.findall(r'^.*~', site))[2:-3]}{cluster_column}{cluster_number}")
            ax.set_xlabel("Time (min)")
            ax.set_ylabel(f"{data_type}")
            ax.set_xlim(-1, 7)
            if fit_y_lims == True:
                ax.set_ylim(min(all_times) * 1.1 - 0.1, max(all_times) * 1.1 + 0.1)
                y_lim = ""
            elif type(fit_y_lims) == list:
                ax.set_ylim(fit_y_lims[0], fit_y_lims[1])
                y_lim = f"_y_axis_fixed_{fit_y_lims[0]}_{fit_y_lims[1]}"
            else:
                y_lim = ""

            ax.legend()
            ax.grid()

            if save_pdf == True:
                plt.savefig(f"{saving_path}/{dataset_name}{name}_{site}{y_lim}{saving_info}.pdf")
                print(f"{name}_{site}{y_lim}{saving_info}.pdf Plot saved as PDF")
            if save_png == True:
                plt.savefig(f"{saving_path}/{name}_{site}{y_lim}{saving_info}.png")
                print(f"{name}_{site}{y_lim}{saving_info}.png Plot saved as PNG")
            if save_pdf == False and save_png == False:
                print(f"{name}_{site}{y_lim}{saving_info} Plot not saved")
            if plot_close == True:
                plt.close()

    else:
        sqrt_n_p = int(np.ceil(np.sqrt(number_phos)))
        if sqrt_n_p <= 2:
            empty_plots = 0
        else:
            empty_plots = (sqrt_n_p * sqrt_n_p) - number_phos

        if empty_plots >= sqrt_n_p:
            sqrt_n_p_X = sqrt_n_p - 1
        else:
            sqrt_n_p_X = sqrt_n_p

        if fit_y_lims == True:
            y_fixed = "_y_axis_fixed"
        elif type(fit_y_lims) == list:
            y_lim_min = fit_y_lims[0]
            y_lim_max = fit_y_lims[1]
            y_fixed = f"_y_axis_fixed_{y_lim_min}_{y_lim_max}"
        else:
            sub_values_df = df[column_selection]
            y_lim_max = sub_values_df.max().max() * 1.1
            y_lim_min = sub_values_df.min().min() * 1.1
            y_fixed = "_y_axis_general"

        k = 0
        fig, ax = plt.subplots(sqrt_n_p, sqrt_n_p_X, figsize=(18, 13))
        fig.tight_layout(w_pad=1.75, h_pad=3)
        plt.subplots_adjust(top=0.94)

        for i in range(sqrt_n_p):
            for j in range(sqrt_n_p_X):
                if k == number_phos:
                    continue
                else:
                    row = df.iloc[k, :]
                    site = row["site"]
                    name = row["protein_name"]
                    id = row["protein_Id"]
                    all_times = row[column_selection].tolist()
                    EGF = row[EGF_mean_cols].tolist()
                    INS = row[INS_mean_cols].tolist()
                    EGFnINS = row[EGFnINS_mean_cols].tolist()
                    groups = [EGF, INS, EGFnINS]
                    EGF_sd = row[EGF_sd_cols].tolist()
                    INS_sd = row[INS_sd_cols].tolist()
                    EGFnINS_sd = row[EGFnINS_sd_cols].tolist()
                    groups_sd = [EGF_sd, INS_sd, EGFnINS_sd]
                    n_rep = row["n_rep"]

                    for c in range(3):
                        al = 0.3 if n_rep == 1 else 1
                        ax[i, j].errorbar(x=time_points, y=groups[c], yerr=groups_sd[c],
                                          marker='o', color=color_palette[c], alpha=al, capsize=4, elinewidth=1.3)

                    ax[i, j].set_title(f"{str(re.findall(r'^.*~', site))[2:-3]}_n{n_rep}")
                    ax[i, j].set_xlabel("Time (min)")
                    ax[i, j].set_ylabel(f"{data_type}")
                    ax[i, j].grid()

                    if fit_y_lims == True:
                        ax[i, j].set_ylim(min(all_times) * 1.1 - 0.1, max(all_times) * 1.1 + 0.1)
                    else:
                        ax[i, j].set_ylim(y_lim_min, y_lim_max)
                    ax[i, j].set_xlim(-1, 7)
                    k = k + 1

        fig.legend(labels=legend, loc="upper right", ncol=len(groups))
        fig.suptitle(f"{dataset_name} {cluster_column} {cluster_number} {saving_info} {date.today()}", weight='bold')

        if save_pdf == True:
            plt.savefig(f"{dataset_name}{cluster_column}_group{cluster_number}_{saving_info}.pdf")
            print(f"{dataset_name}{cluster_column}_group{cluster_number}_{saving_info}.pdf Plot saved as PDF")
        if save_png == True:
            plt.savefig(f"{saving_path}/{dataset_name}{cluster_column}_group{cluster_number}_{saving_info}.png")
            print(f"{dataset_name}{cluster_column}_group{cluster_number}_{saving_info}.png Plot saved as PNG")
        if save_pdf == False and save_png == False:
            print(f"{dataset_name}{cluster_column}_group{cluster_number}_{saving_info} Plot not saved")
        if plot_close == True:
            plt.close()


def plot_protein_phosphosites(df,
                              data_type="",
                              proteins=None,
                              replicates=False,
                              exclude_rep=None,
                              legend_plot=None,
                              color_palette=['r', 'b', 'fuchsia'],
                              saving_path="",
                              saving_info="",
                              title_info="",
                              plot_individually=False,
                              fit_y_lims=False,
                              plot_close=False,
                              save_pdf=False,
                              save_png=False):
    """Plot all phosphosites for a list of proteins, one protein per figure or all combined."""

    if proteins is None:
        proteins = []
    if exclude_rep is None:
        exclude_rep = []
    if legend_plot is None:
        legend_plot = []

    if not isinstance(df, pd.DataFrame):
        df = pd.read_excel(df)

    if save_pdf == False and save_png == False:
        pass
    else:
        if not os.path.exists(saving_path):
            print("Creating saving folder")
            os.makedirs(saving_path)

    for protein in proteins:
        if protein in df['protein_name'].to_list():
            sub_df = df.loc[df['protein_name'] == protein].copy()
            print(f"Ploting sites of protein {protein}")
        elif protein in df['protein_Id'].to_list():
            sub_df = df.loc[df['protein_Id'] == protein].copy()
            print(f"Ploting sites of protein {protein}")
        else:
            print(f"The protein {protein} is not present in the dataset")
            continue

        saving_folder = f"{list(sub_df.protein_name)[0]}_{list(sub_df.protein_Id)[0]}"

        if save_pdf == False and save_png == False:
            pass
        else:
            if saving_folder in os.listdir(saving_path):
                pass
            else:
                new_path = f"{saving_path}/{saving_folder}"
                print(f"Createating saving folder for {saving_folder}")
                os.makedirs(new_path)

        sub_df.sort_values(by=['site'], inplace=True)

        number_phos = len(sub_df)
        sqrt_n_p = int(np.ceil(np.sqrt(number_phos)))
        if sqrt_n_p <= 2:
            empty_plots = 0
        else:
            empty_plots = (sqrt_n_p * sqrt_n_p) - number_phos
        if empty_plots >= sqrt_n_p:
            sqrt_n_p_X = sqrt_n_p - 1
        else:
            sqrt_n_p_X = sqrt_n_p

        column_names = df.columns.tolist()
        if data_type == "raw":
            data_type = "raw_mean"
            column_selection = [element for element in column_names if data_type in element]
            data_type = "raw"
        elif data_type == "log2":
            data_type = "log2_mean"
            column_selection = [element for element in column_names if data_type in element]
            data_type = "log2"
        else:
            column_selection = [element for element in column_names if data_type in element and "clusters" not in element]

        if plot_individually == True:
            for index, row in sub_df.iterrows():
                protein_name = row["protein_name"]
                site = row["site"]
                if fit_y_lims == True:
                    y_fixed = "y_axis_fixed"
                elif type(fit_y_lims) == list:
                    y_lim_min = fit_y_lims[0]
                    y_lim_max = fit_y_lims[1]
                    y_fixed = f"_y_axis_fixed_{y_lim_min}_{y_lim_max}"
                elif fit_y_lims == False:
                    sub_values_df = sub_df.loc[:, column_selection]
                    y_lim_max = (sub_values_df.max().max()) * 1.05
                    y_lim_min = (sub_values_df.min().min()) * 0.95
                    if y_lim_min < 0:
                        y_lim_min = (sub_values_df.min().min()) + (sub_values_df.min().min()) * 0.1

                fig, axes = plt.subplots()
                plot_data(ax=axes, row_df=row, replicates=replicates, data_type=data_type, colors=color_palette,
                          legend=legend_plot, exclude_rep=exclude_rep, plot_individually=plot_individually)
                if fit_y_lims == True:
                    if data_type == "raw" or data_type == "log2_FC":
                        if min(row[column_selection]) < 0:
                            y_lim_min = min(row[column_selection]) + min(row[column_selection]) * 0.1
                            axes.set_ylim(y_lim_min, max(row[column_selection]) * 1.1)
                        else:
                            axes.set_ylim(min(row[column_selection]) * 0.9, max(row[column_selection]) * 1.1)
                    else:
                        axes.set_ylim(min(row[column_selection]) * 0.97, max(row[column_selection]) * 1.02)
                else:
                    axes.set_ylim(y_lim_min, y_lim_max)
                axes.set_xlim(-1, 7)

                if save_pdf == True:
                    plt.savefig(f"{saving_path}/{saving_folder}/{protein_name}_{site}_{saving_info}.pdf")
                    print(f"{protein_name}_{site}_{saving_info}.pdf Plot saved as PDF")
                if save_png == True:
                    plt.savefig(f"{saving_path}/{saving_folder}/{protein_name}_{site}_{saving_info}.png")
                    print(f"{protein_name}_{site}_{saving_info}.png Plot saved as PNG")
                if save_pdf == False and save_png == False:
                    print(f"{protein_name}_{site}_{saving_info} Plot not saved")

        else:
            k = 0
            fig, axes = plt.subplots(sqrt_n_p, sqrt_n_p_X, figsize=(18, 13))
            fig.tight_layout(w_pad=1.75, h_pad=3)
            plt.subplots_adjust(top=0.94)

            if len(sub_df) == 1:
                axes = np.array([[axes]])
            elif sqrt_n_p == 1 or sqrt_n_p_X == 1:
                axes = np.atleast_2d(axes)

            if fit_y_lims == True:
                y_fixed = "y_axis_fixed"
            elif type(fit_y_lims) == list:
                y_lim_min = fit_y_lims[0]
                y_lim_max = fit_y_lims[1]
                y_fixed = f"_y_axis_fixed_{y_lim_min}_{y_lim_max}"
            elif fit_y_lims == False:
                sub_values_df = sub_df.loc[:, column_selection]
                y_lim_max = (sub_values_df.max().max()) * 1.02
                y_lim_min = (sub_values_df.min().min()) * 0.97
                if y_lim_min < 0:
                    y_lim_min = (sub_values_df.min().min()) + (sub_values_df.min().min()) * 0.1

            for i in range(sqrt_n_p):
                for j in range(sqrt_n_p_X):
                    if k >= number_phos:
                        fig.delaxes(axes[i, j])
                    else:
                        row = sub_df.iloc[k, :]
                        plot_data(ax=axes[i, j], row_df=row, replicates=replicates, data_type=data_type,
                                  colors=color_palette, legend=legend_plot, exclude_rep=exclude_rep,
                                  plot_individually=plot_individually)

                        if fit_y_lims == True:
                            if data_type == "raw" or data_type == "log2_FC":
                                if min(row[column_selection]) < 0:
                                    y_lim_min = min(row[column_selection]) + min(row[column_selection]) * 0.1
                                    axes[i, j].set_ylim(y_lim_min, max(row[column_selection]) * 1.1)
                                else:
                                    axes[i, j].set_ylim(min(row[column_selection]) * 0.9, max(row[column_selection]) * 1.1)
                            else:
                                axes[i, j].set_ylim(min(row[column_selection]) * 0.97, max(row[column_selection]) * 1.02)
                        else:
                            axes[i, j].set_ylim(y_lim_min, y_lim_max)

                        axes[i, j].set_xlim(-1, 7)
                        k = k + 1

            fig.legend(labels=legend_plot, loc="upper right", ncol=3)
            fig.suptitle(f"{saving_folder} {title_info} ({date.today()})", weight='bold')
            fig.tight_layout()

            if save_pdf == True:
                plt.savefig(f"{saving_path}/{saving_folder}/{saving_folder}_{data_type}_{saving_info}.pdf")
                print(f"{saving_folder}_{data_type}_{saving_info}.pdf Plot saved as PDF")
            if save_png == True:
                plt.savefig(f"{saving_path}/{saving_folder}/{saving_folder}_{data_type}_{saving_info}.png")
                print(f"{saving_folder}_{data_type}_{saving_info}.png Plot saved as PNG")
            if save_pdf == False and save_png == False:
                print(f"{saving_folder}_{data_type}_{saving_info} Plot not saved")
    if plot_close == True:
        plt.close(fig)


def plot_protein_profile(df,
                         proteins,
                         data_type="",
                         saving_path="",
                         saving_info="",
                         legend=False,
                         save_pdf=False,
                         save_png=False):

    if not isinstance(df, pd.DataFrame):
        df = pd.read_excel(df)

    if (save_pdf or save_png) and not os.path.exists(saving_path):
        print("Creating saving folder")
        os.makedirs(saving_path)

    column_names = df.columns.tolist()
    all_conditions = [element for element in column_names if f"{data_type}_" in element and "cluster" not in element]
    EGF = [element for element in column_names if f"{data_type}_EGF_" in element]
    INS = [element for element in column_names if f"{data_type}_INS_" in element]
    EGFnINS = [element for element in column_names if f"{data_type}_EGFnINS_" in element]

    x_axis_previous = [element for element in column_names if f"log2_FC_EGF_" in element]
    time_points = [s.split("_")[-1] for s in x_axis_previous]

    fig, ax = plt.subplots(len(proteins), 3, figsize=(10, 2 * len(proteins)))
    if len(proteins) == 1:
        ax = [ax]

    for c, protein in enumerate(proteins):
        if protein in df['protein_name'].values:
            sub_df = df[df['protein_name'] == protein].copy()
        elif protein in df['protein_Id'].values:
            sub_df = df[df['protein_Id'] == protein].copy()
        else:
            print(f"The protein {protein} is not present in the dataset.")
            continue

        protein_for_url = str(sub_df['protein_Id'].values[0])
        prot_name = str(sub_df['protein_name'].values[0])
        uniprot_url = f"https://www.uniprot.org/uniprotkb/{protein_for_url}"
        html_link = f'Plotting sites of protein <a href="{uniprot_url}" target="_blank">{protein_for_url}</a> {prot_name}'
        display(HTML(html_link))

        saving_folder = f"{sub_df['protein_name'].values[0]}_{sub_df['protein_Id'].values[0]}"
        sub_df.sort_values(by=['site'], inplace=True)

        for _, row in sub_df.iterrows():
            ax[c][0].plot(time_points, row[EGF])
            ax[c][0].set_title("EGF")
            ax[c][0].axhline(0, color='black', linestyle='--', linewidth=0.5)

            ax[c][1].plot(time_points, row[EGFnINS])
            ax[c][1].set_title("EGFnINS")
            ax[c][1].axhline(0, color='black', linestyle='--', linewidth=0.5)

            ax[c][2].plot(time_points, row[INS])
            ax[c][2].set_title("INS")
            ax[c][2].axhline(0, color='black', linestyle='--', linewidth=0.5)

        sub_values_df = sub_df[all_conditions]
        y_max = sub_values_df.max().max() * 1.05 + 0.1
        y_min_val = sub_values_df.min().min()
        y_min = y_min_val * 0.95 - 0.1 if y_min_val >= 0 else -abs(y_min_val) * 1.05 - 0.1

        for i in range(3):
            ax[c][i].set_ylim(y_min, y_max)
        ax[c][0].set_ylabel(f"{saving_folder}\n{data_type}")

    if legend == True:
        fig.legend(labels=df["site"].unique())

    fig.tight_layout()

    if save_pdf:
        plt.savefig(os.path.join(saving_path, f"{saving_info}.pdf"))
    if save_png:
        plt.savefig(os.path.join(saving_path, f"{saving_info}.png"))
    if not save_pdf and not save_png:
        print(f"{saving_info} Plot not saved")

    plt.show()


def plot_protein_profiles_fine_line(df,
                                    proteins=None,
                                    data_type="",
                                    saving_path="",
                                    legend=False,
                                    saving_info="",
                                    save_pdf=False,
                                    save_png=False):

    if proteins is None:
        proteins = []

    if not isinstance(df, pd.DataFrame):
        df = pd.read_excel(df)

    if save_pdf == False and save_png == False:
        pass
    else:
        if not os.path.exists(saving_path):
            print("Creating saving folder")
            os.makedirs(saving_path)

    column_names = df.columns.tolist()
    all_conditions = [element for element in column_names if f"{data_type}_" in element and "cluster" not in element]
    EGF = [element for element in column_names if f"{data_type}_EGF_" in element]
    INS = [element for element in column_names if f"{data_type}_INS_" in element]
    EGFnINS = [element for element in column_names if f"{data_type}_EGFnINS_" in element]

    x_axis_previous = [element for element in column_names if f"log2_FC_EGF_" in element]
    time_points = [s.split("_")[-1] for s in x_axis_previous]

    for protein in proteins:
        if protein in df['protein_name'].to_list():
            sub_df = df.loc[df['protein_name'] == protein].copy()
        elif protein in df['protein_Id'].to_list():
            sub_df = df.loc[df['protein_Id'] == protein].copy()
        else:
            print(f"The protein {protein} is not present in the dataset")
            continue

        protein_for_url = str(sub_df['protein_Id'].values[0])
        prot_name = str(sub_df['protein_name'].values[0])
        uniprot_url = f"https://www.uniprot.org/uniprotkb/{protein_for_url}"
        html_link = f'Plotting sites of protein <a href="{uniprot_url}" target="_blank">{protein_for_url}</a> {prot_name}'
        display(HTML(html_link))

        saving_folder = f"{list(sub_df.protein_name)[0]}_{list(sub_df.protein_Id)[0]}"

        if save_pdf == False and save_png == False:
            pass
        else:
            if saving_folder in os.listdir(saving_path):
                pass
            else:
                new_path = f"{saving_path}/{saving_folder}"
                print(f"Createating saving folder for {saving_folder}")
                os.makedirs(new_path)

        sub_df.sort_values(by=['site'], inplace=True)

        fig, ax = plt.subplots(1, 3, figsize=(20, 5))
        for index, row in sub_df.iterrows():
            ax[0].errorbar(x=time_points, y=row[EGF])
            ax[0].title.set_text("EGF")

            ax[1].errorbar(x=time_points, y=row[EGFnINS])
            ax[1].title.set_text("EGFnINS")

            ax[2].errorbar(x=time_points, y=row[INS])
            ax[2].title.set_text("INS")

            sub_values_df = sub_df.loc[:, all_conditions]
            y_lim_max = (sub_values_df.max().max()) * 1.05
            if sub_values_df.min().min() >= 0:
                y_lim_min = (sub_values_df.min().min()) * 0.95
            else:
                y_lim_min = (abs(sub_values_df.min().min()) * 1.05) * -1
            ax[0].set_ylim(y_lim_min, y_lim_max)
            ax[0].set_ylabel(f"{data_type}")
            ax[1].set_ylim(y_lim_min, y_lim_max)
            ax[2].set_ylim(y_lim_min, y_lim_max)

        fig.suptitle(f"{saving_folder}", weight='bold')
        fig.tight_layout()
        if legend == True:
            fig.legend(labels=list(sub_df["site"]), loc="upper right", ncol=1)

        if save_pdf == True:
            plt.savefig(f"{saving_path}/{saving_folder}_{saving_info}.pdf")
            print(f"{saving_folder}_{data_type}_{saving_info}.pdf Plot saved as PDF")
        if save_png == True:
            plt.savefig(f"{saving_path}/{saving_folder}_{saving_info}.png")
            print(f"{saving_folder}_{data_type}_{saving_info}.png Plot saved as PNG")
        if save_pdf == False and save_png == False:
            print(f"{saving_folder} Plot not saved")


def clusters_plot(df,
                  legend=None,
                  saving_path="",
                  cluster_column="",
                  cluster_name="",
                  data_type="",
                  plot_different_data=False,
                  saving_info="",
                  save_pdf=False,
                  save_png=False,
                  plot_close=False,
                  y_lims_list=False):
    """Take a dataset with a cluster column, compute per-cluster means, and plot the average curves."""

    if legend is None:
        legend = []

    if not isinstance(df, pd.DataFrame):
        df = pd.read_excel(df)

    if save_pdf == False and save_png == False:
        pass
    else:
        if not os.path.exists(saving_path):
            print("Creating saving folder")
            os.makedirs(saving_path)

    clusters = list(set(df[cluster_column]))
    if 999 in clusters:
        clusters.remove(999)
    if data_type not in cluster_column and plot_different_data == False:
        print("Remember to plot the same data_type used to make the clustering or put: plot_different_data = TRUE")
    else:
        if type(clusters[0]) == int:
            sorted_clusters = sorted(clusters)
        else:
            sorted_clusters = sorted(clusters, key=lambda x: int(x.split()[1]))

        n_cluster = len(sorted_clusters)
        sqrt_n_c = int(np.ceil(np.sqrt(n_cluster)))
        empty_plots = (sqrt_n_c * sqrt_n_c) - n_cluster

        if empty_plots >= sqrt_n_c:
            sqrt_n_c_X = sqrt_n_c - 1
        else:
            sqrt_n_c_X = sqrt_n_c

        i_list = list(range(sqrt_n_c))
        i_c = 0
        j_list = list(range(sqrt_n_c_X))
        j_c = 0

        fig, ax = plt.subplots(sqrt_n_c, sqrt_n_c_X, figsize=(18, 13))
        fig.tight_layout(w_pad=1.75, h_pad=3)
        plt.subplots_adjust(top=0.94)

        column_names = df.columns.tolist()
        time_points_previous = [element for element in column_names if f"log2_FC_EGF_" in element]
        time_points = [s.split("_")[-1] for s in time_points_previous]

        EGF_matching_cols = [col for col in df.columns if any(f"{data_type}_EGF_{t}" in col for t in time_points)]
        INS_matching_cols = [col for col in df.columns if any(f"{data_type}_INS_{t}" in col for t in time_points)]
        EGFnINS_matching_cols = [col for col in df.columns if any(f"{data_type}_EGFnINS_{t}" in col for t in time_points)]

        for cluster in sorted_clusters:
            sub_df = df.loc[df[cluster_column] == cluster].copy()

            EGF_means = [sub_df[col].mean() for col in EGF_matching_cols]
            INS_means = [sub_df[col].mean() for col in INS_matching_cols]
            EGFnINS_means = [sub_df[col].mean() for col in EGFnINS_matching_cols]

            EGF_err = [sub_df[col].std() for col in EGF_matching_cols]
            INS_err = [sub_df[col].std() for col in INS_matching_cols]
            EGFnINS_err = [sub_df[col].std() for col in EGFnINS_matching_cols]

            groups = [EGF_means, INS_means, EGFnINS_means]
            groups_sd = [EGF_err, INS_err, EGFnINS_err]
            colors = ['r', 'b', 'fuchsia']

            for c in range(3):
                ax[i_list[i_c], j_list[j_c]].errorbar(x=time_points, y=groups[c],
                                                       yerr=groups_sd[c], marker='o', color=colors[c],
                                                       capsize=4, elinewidth=1.3)

            ax[i_list[i_c], j_list[j_c]].set_xlabel("Time (min)")
            ax[i_list[i_c], j_list[j_c]].set_ylabel(f"{data_type}")
            ax[i_list[i_c], j_list[j_c]].grid()
            ax[i_list[i_c], j_list[j_c]].set_title(f"Cluster {cluster} ({len(sub_df)} sites)")
            ax[i_list[i_c], j_list[j_c]].set_ylim(min(min(groups)) - 0.3 * 1.3, max(max(groups)) + 0.5 * 1.5)
            if type(y_lims_list) == list:
                ax[i_list[i_c], j_list[j_c]].set_ylim(y_lims_list[0], y_lims_list[1])

            if j_c == len(j_list) - 1:
                j_c = 0
                i_c = i_c + 1
            else:
                j_c = j_c + 1

        fig.legend(labels=legend, loc="upper right", ncol=len(groups))
        fig.suptitle(f"{cluster_column} {cluster_name} {date.today()}", weight='bold')

        if save_pdf == True:
            plt.savefig(f"{saving_path}/{cluster_name}{saving_info}.pdf")
            print(f"{cluster_name}{saving_info} Plot saved as PDF")
        if save_png == True:
            plt.savefig(f"{saving_path}/{cluster_name}{saving_info}.png")
            print(f"{cluster_name}{saving_info} Plot saved as PNG")
        if save_pdf == False and save_png == False:
            print(f"{cluster_name}{saving_info} Plot not saved")
        if plot_close == True:
            plt.close()


# =============================================================================
# Plotting — refined naming convention (CellLine)_(DataType)_(Treatment)_(TimePoint)
# =============================================================================

def plot_data_refined(ax,
                      row_df,
                      data_type="",
                      colors={},
                      cell_lines=[],
                      conditions=[]):
    column_names = row_df.index.tolist()
    sub_dtp = data_type.split(":")  # e.g. ["log2", "FC"]

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

    x_axis_previous = [element for element in column_names if
                       f"{cell_lines[0]}_{data_type}{conditions[0]}" in element]
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
                label=condition,
                capsize=4,
                elinewidth=1.3,
                alpha=1
            )
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(f"{data_type}")
    ax.set_title(f"{site}_n{n_rep}")


def plot_protein_phosphosites_refined(df,
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
                                      plot_close=False,
                                      save_pdf=False,
                                      save_png=False):
    """Plot all phosphosites for a list of proteins using the multi-cell-line naming convention."""

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

    x_axis_previous = [element for element in df.columns if
                       f"{cell_lines[0]}_{data_type}{conditions[0]}" in element]
    x_axis = [s.split("_")[3] for s in x_axis_previous]
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

        column_names = df.columns.tolist()
        column_selection = []

        for cell in cell_lines:
            for condition in conditions:
                new_cols = [col for col in column_names
                            if col.startswith(cell)
                            and condition in col
                            and data_type == col.split("_")[1]
                            ]
                column_selection = column_selection + new_cols

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
        else:
            use_fixed_ylims = None
            y_limt_info = "y_axis_perrow"

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

                plot_data_refined(ax=axes[i, j],
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


# =============================================================================
# Volcano plots
# =============================================================================

def plot_volcano(
    df,
    fc_col,
    pval_col,
    fc_thresh=1.0,
    pval_thresh=0.05,
    title=None,
    ax=None,
    highlight_proteins=None,
    match_cols=("protein_Id", "protein_name"),
    case_insensitive=True,
    fit_x_limit=False
):
    """
    Volcano plot with optional multi-protein highlighting across protein_Id / protein_name.

    Args:
        df (pd.DataFrame): data
        fc_col (str): log2FC column
        pval_col (str): p-value column (raw p; function converts to -log10)
        fc_thresh (float): threshold for |log2FC|
        pval_thresh (float): p-value threshold for significance
        title (str): title
        ax (matplotlib.axes.Axes): draw into this axes if provided
        highlight_proteins (str | list[str]): protein(s) to highlight
        match_cols (tuple[str]): columns to match against
        case_insensitive (bool): if True, case-insensitive matching
        fit_x_limit (bool | list): if True fixes x to [-6, 6]; if list uses [min, max]
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
        created_fig = True

    log2fc = df[fc_col]
    pvals = df[pval_col]
    pvals = np.where(np.asarray(pvals, dtype=float) <= 0, np.nan, pvals)
    neg_log10_pval = -np.log10(pvals)

    sig = (np.abs(log2fc) >= fc_thresh) & (df[pval_col] <= pval_thresh)

    ax.scatter(log2fc[~sig], neg_log10_pval[~sig], color="grey", alpha=0.6, s=20, label="not significant")
    ax.scatter(log2fc[sig], neg_log10_pval[sig], color="red", alpha=0.8, s=30, label="significant")

    ax.axhline(-np.log10(pval_thresh), color="blue", linestyle="--", linewidth=1)
    ax.axvline(-fc_thresh, color="blue", linestyle="--", linewidth=1)
    ax.axvline(fc_thresh, color="blue", linestyle="--", linewidth=1)

    if highlight_proteins is not None:
        if isinstance(highlight_proteins, str):
            highlight_list = [highlight_proteins]
        else:
            highlight_list = list(highlight_proteins)

        comp_cols = [c for c in match_cols if c in df.columns]
        norm_cols = {}
        for c in comp_cols:
            if case_insensitive:
                norm_cols[c] = df[c].astype(str).str.lower()
            else:
                norm_cols[c] = df[c].astype(str)

        color_cycle = cycle(plt.cm.tab10.colors if hasattr(plt.cm, "tab10") else ["gold", "cyan", "magenta", "yellow", "green", "blue"])

        for item in highlight_list:
            if item is None:
                continue
            item_key = item.lower() if case_insensitive else str(item)

            mask = np.zeros(len(df), dtype=bool)
            for c in comp_cols:
                mask |= (norm_cols[c] == item_key)

            if mask.any():
                color = next(color_cycle)
                ax.scatter(
                    log2fc[mask], neg_log10_pval[mask],
                    s=90, marker="o", facecolor=color, edgecolor="black",
                    linewidth=0.8, alpha=0.95, zorder=3,
                    label=f"Highlight: {item}"
                )

    ax.set_xlabel("log2 Fold Change")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title(title or f"Volcano: {fc_col}")
    ax.legend(fontsize=8, frameon=False)
    if fit_x_limit is not False:
        if fit_x_limit == True:
            ax.set_xlim(-6, 6)
        else:
            ax.set_xlim(fit_x_limit[0], fit_x_limit[1])

    if created_fig:
        plt.tight_layout()
        plt.show()


def plot_volcano_interactive_plotly(
    df: pd.DataFrame,
    fc_col: str,
    pval_col: str,
    site_col: str = "site",
    fc_thresh: float = 1.0,
    pval_thresh: float = 0.05,
    title: str = "",
):
    """
    Interactive volcano plot (Plotly).
    - x: log2FC column (fc_col)
    - y: -log10(p-value) from pval_col
    - hover: shows site
    """
    d = df[[fc_col, pval_col, site_col]].copy()

    d[pval_col] = pd.to_numeric(d[pval_col], errors="coerce")
    d.loc[d[pval_col] <= 0, pval_col] = np.nan
    d["neglog10p"] = -np.log10(d[pval_col])

    d["signif"] = np.where(
        (d[fc_col].abs() >= fc_thresh) & (d[pval_col] <= pval_thresh),
        "significant",
        "not significant"
    )

    fig = px.scatter(
        d,
        x=fc_col,
        y="neglog10p",
        color="signif",
        hover_name=site_col,
        hover_data={fc_col: ':.3f', "neglog10p": ':.3f', pval_col: ':.3g'},
        title=title or f"Volcano: {fc_col}",
        template="plotly_white",
    )

    fig.add_hline(y=-np.log10(pval_thresh), line_dash="dash")
    fig.add_vline(x=fc_thresh, line_dash="dash")
    fig.add_vline(x=-fc_thresh, line_dash="dash")

    fig.update_layout(
        xaxis_title="log2 Fold Change",
        yaxis_title="-log10(p-value)",
        legend_title="",
    )
    return fig


def plot_volcano_interactive_plotly_highlighting(
    df: pd.DataFrame,
    fc_col: str,
    pval_col: str,
    site_col: str = "site",
    *,
    fc_thresh: float = 1.0,
    pval_thresh: float = 0.05,
    title: str = None,
    highlight_proteins=None,
    match_cols=("protein_Id", "protein_name"),
    case_insensitive: bool = True,
    show_highlight_labels: bool = False,
):
    """
    Interactive volcano plot (Plotly) with multi-protein highlighting.

    - x: log2FC column (fc_col)
    - y: -log10(p-value) from pval_col
    - Hover shows site + details
    - `highlight_proteins` can be a string or list of strings; matches any of `match_cols`
    """
    d = df[[fc_col, pval_col]].copy()
    if site_col in df.columns:
        d[site_col] = df[site_col]
    for c in match_cols:
        if c in df.columns:
            d[c] = df[c]

    d[pval_col] = pd.to_numeric(d[pval_col], errors="coerce")
    d.loc[d[pval_col] <= 0, pval_col] = np.nan
    d["neglog10p"] = -np.log10(d[pval_col])

    d["signif"] = np.where(
        (d[fc_col].abs() >= fc_thresh) & (d[pval_col] <= pval_thresh),
        "significant",
        "not significant",
    )

    fig = px.scatter(
        d,
        x=fc_col,
        y="neglog10p",
        color="signif",
        opacity=0.3,
        hover_name=site_col if site_col in d.columns else None,
        hover_data={
            fc_col: ':.3f',
            "neglog10p": ':.3f',
            pval_col: ':.3g',
            **({"protein_Id": True} if "protein_Id" in d.columns else {}),
            **({"protein_name": True} if "protein_name" in d.columns else {}),
        },
        title=title or f"Volcano: {fc_col}",
        template="plotly_white",
    )

    fig.add_hline(y=-np.log10(pval_thresh), line_dash="dash", line_color="gray")
    fig.add_vline(x=fc_thresh, line_dash="dash", line_color="gray")
    fig.add_vline(x=-fc_thresh, line_dash="dash", line_color="gray")

    fig.update_layout(
        xaxis_title="log2 Fold Change",
        yaxis_title="-log10(p-value)",
        legend_title="",
    )

    if highlight_proteins is not None:
        if isinstance(highlight_proteins, str):
            highlight_list = [highlight_proteins]
        else:
            highlight_list = list(highlight_proteins)

        comp_cols = [c for c in match_cols if c in d.columns]
        norm_df = {}
        for c in comp_cols:
            norm_df[c] = d[c].astype(str).str.lower() if case_insensitive else d[c].astype(str)

        palette = px.colors.qualitative.D3
        color_idx = 0

        for item in highlight_list:
            key = str(item).lower() if case_insensitive else str(item)
            if not comp_cols:
                continue

            mask = np.zeros(len(d), dtype=bool)
            for c in comp_cols:
                mask |= (norm_df[c] == key)

            if not mask.any():
                continue

            color = palette[color_idx % len(palette)]
            color_idx += 1

            if show_highlight_labels:
                if "protein_name" in d.columns:
                    text_vals = df.loc[mask, "protein_name"].astype(str)
                elif "protein_Id" in d.columns:
                    text_vals = df.loc[mask, "protein_Id"].astype(str)
                else:
                    text_vals = (df.loc[mask, site_col].astype(str)
                                 if site_col in df.columns else pd.Series([""] * mask.sum()))
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
                    symbol="circle"
                ),
                name=f"Highlight: {item}",
                hovertemplate=(
                    f"<b>{item}</b><br>"
                    f"log2FC: %{'x'}:.3f<br>"
                    f"-log10(p): %{'y'}:.3f<br>"
                    + (f"{site_col}: %{{customdata[0]}}<br>" if site_col in d.columns else "")
                    + ("%{text}" if show_highlight_labels else "")
                ),
                customdata=d.loc[mask, [site_col]].values if site_col in d.columns else None,
            ))

    return fig


# =============================================================================
# Clustering
# =============================================================================

def tslearn_clustering_KMeans(df_to_cluster,
                              data_type,
                              condition_for_clustering=None,
                              exclude_full=False,
                              cluster_column_name="",
                              number_of_clusters=10,
                              max_iterations=1000,
                              n_init=5,
                              metric='euclidean',
                              df_dimensions=None,
                              random_state=0,
                              time_series_length=None,
                              transpose=False,
                              verbose=True,
                              testing=False,
                              barycenter_calculations=False):

    if condition_for_clustering is None:
        condition_for_clustering = []
    if df_dimensions is None:
        raise ValueError("df_dimensions is required")
    if time_series_length is None:
        raise ValueError("time_series_length is required")

    column_names = df_to_cluster.columns.tolist()
    if len(condition_for_clustering) == 0:
        column_selection = [element for element in column_names if element.startswith(f"{data_type}")]
        if exclude_full == True:
            column_selection = [element for element in column_names if element.startswith(f"{data_type}") and "full" not in element]
    else:
        column_selection = [element for element in column_names if element.startswith(f"{data_type}") and any(cond in element for cond in condition_for_clustering)]
        if exclude_full == True:
            column_selection = [element for element in column_names if element.startswith(f"{data_type}") and any(cond in element for cond in condition_for_clustering) and "full" not in element]
    if verbose == True:
        print(f"Column selection: {column_selection}\n")

    multivariate_df, names_of_myseries = reshape_df(df=df_to_cluster, time_series=column_selection,
                                                     dimensions=df_dimensions, len_time_serie=time_series_length,
                                                     transpose=transpose, labels="site", verbose=verbose)

    if verbose == True:
        print(f"\nThe size of the dataset is {multivariate_df.shape}")
        print(f"Example:\n{multivariate_df[0]}")

    clustering = TimeSeriesKMeans(n_clusters=number_of_clusters, max_iter=max_iterations, n_init=n_init,
                                  metric=metric, max_iter_barycenter=1000, verbose=verbose,
                                  random_state=random_state).fit(multivariate_df)
    df_to_cluster[f"{cluster_column_name}"] = clustering.labels_

    if testing == True:
        if barycenter_calculations == True:
            barycenters_distances = TimeSeriesKMeans(n_clusters=number_of_clusters, max_iter=max_iterations,
                                                     n_init=n_init, metric=metric, max_iter_barycenter=1000,
                                                     verbose=verbose, random_state=random_state).fit_transform(multivariate_df)
            return df_to_cluster, clustering, multivariate_df, barycenters_distances
        else:
            return df_to_cluster, clustering, multivariate_df
    else:
        return df_to_cluster


def tslearn_clustering_KShape(df_to_cluster,
                              data_type,
                              condition_for_clustering=None,
                              exclude_full=False,
                              number_of_clusters=10,
                              cluster_column_name="",
                              n_init=5,
                              max_iterations=1000,
                              df_dimensions=None,
                              time_series_length=None,
                              random_state=0,
                              transpose=False,
                              verbose=True,
                              testing=False):

    if condition_for_clustering is None:
        condition_for_clustering = []
    if df_dimensions is None:
        raise ValueError("df_dimensions is required")
    if time_series_length is None:
        raise ValueError("time_series_length is required")

    column_names = df_to_cluster.columns.tolist()
    if len(condition_for_clustering) == 0:
        column_selection = [element for element in column_names if element.startswith(f"{data_type}")]
        if exclude_full == True:
            column_selection = [element for element in column_names if element.startswith(f"{data_type}") and "full" not in element]
    else:
        column_selection = [element for element in column_names if element.startswith(f"{data_type}") and any(cond in element for cond in condition_for_clustering)]
        if exclude_full == True:
            column_selection = [element for element in column_names if element.startswith(f"{data_type}") and any(cond in element for cond in condition_for_clustering) and "full" not in element]
    if verbose == True:
        print(f"Column selection: {column_selection}\n")

    multivariate_df, names_of_myseries = reshape_df(df=df_to_cluster, time_series=column_selection,
                                                     dimensions=df_dimensions, len_time_serie=time_series_length,
                                                     transpose=transpose, labels="site", verbose=verbose)

    if verbose == True:
        print(f"\nThe size of the dataset is {multivariate_df.shape}")

    clustering = KShape(n_clusters=number_of_clusters, max_iter=max_iterations, n_init=n_init,
                        verbose=verbose, random_state=random_state).fit(multivariate_df)

    df_to_cluster[f"{cluster_column_name}"] = clustering.labels_

    if testing == True:
        return df_to_cluster, clustering, multivariate_df
    else:
        return df_to_cluster


def kernnel_clustering(df_to_cluster,
                       transpose=True,
                       data_type="log2_FC",
                       exclude_full=True,
                       condition_for_clustering=None,
                       df_dimensions=None,
                       time_series_length=None,
                       seed=0,
                       n_clusters=25,
                       n_init=20,
                       verbose=True,
                       kernel="gak",
                       kernel_params={"sigma": "auto"},
                       cluster_column_name="",
                       testing=False):

    if condition_for_clustering is None:
        condition_for_clustering = []
    if df_dimensions is None:
        raise ValueError("df_dimensions is required")
    if time_series_length is None:
        raise ValueError("time_series_length is required")

    column_names = df_to_cluster.columns.tolist()
    if len(condition_for_clustering) == 0:
        column_selection = [element for element in column_names if element.startswith(f"{data_type}")]
        if exclude_full == True:
            column_selection = [element for element in column_names if element.startswith(f"{data_type}") and "full" not in element]
    else:
        column_selection = [element for element in column_names if element.startswith(f"{data_type}") and any(cond in element for cond in condition_for_clustering)]
        if exclude_full == True:
            column_selection = [element for element in column_names if element.startswith(f"{data_type}") and any(cond in element for cond in condition_for_clustering) and "full" not in element]
    if verbose == True:
        print(f"Column selection: {column_selection}\n")

    multivariate_df, names_of_myseries = reshape_df(df=df_to_cluster, time_series=column_selection, labels="site",
                                                     dimensions=df_dimensions, len_time_serie=time_series_length,
                                                     transpose=transpose, verbose=verbose)

    if verbose == True:
        print(f"\nThe size of the dataset is {multivariate_df.shape}")
        print(f"Example:\n{multivariate_df[0]}")

    clustering_gak_km = KernelKMeans(n_clusters=n_clusters, kernel=kernel, kernel_params=kernel_params,
                                     n_init=n_init, verbose=verbose, random_state=seed).fit(multivariate_df)

    df_to_cluster[f"{cluster_column_name}"] = clustering_gak_km.labels_

    if testing == True:
        return df_to_cluster, clustering_gak_km, multivariate_df
    else:
        return df_to_cluster


# =============================================================================
# Cluster quality / similarity metrics
# =============================================================================

# def cluster_similarity_cdist_dtw(
#         df,
#         transpose=True,
#         data_type="log2_FC",
#         cluster_column_name="",
#         mean=True,
#         median=False,
#         verbose=False
# ):
#     '''
#     DTW distance between peptides' full multivariate time series within each cluster.
#     Returns a dict {cluster: mean/median DTW distance}.
#     '''
#     column_selection = [element for element in df.columns.tolist() if f"{data_type}" in element and "cluster" not in element]
#
#     cluster_metric = {}
#
#     for cluster in sorted(df[cluster_column_name].unique()):
#         if cluster == 999:
#             continue
#         df1 = df.loc[df[cluster_column_name] == cluster]
#         X1_tp, y_qc = reshape_df(df=df1,
#                                  time_series=column_selection,
#                                  labels="site", dimensions=3,
#                                  len_time_serie=7,
#                                   transpose=transpose,
#                                  verbose=verbose)
#
#         dist_tp = tsl.metrics.cdist_dtw(dataset1=X1_tp)
#         if mean == True and median == False:
#             mean_dtw_tp = np.mean(dist_tp[np.triu_indices_from(dist_tp, k=1)])
#             cluster_metric[cluster] = mean_dtw_tp
#         elif mean == False and median == True:
#             median_dtw_tp = np.median(dist_tp[np.triu_indices_from(dist_tp, k=1)])
#             cluster_metric[cluster] = median_dtw_tp
#         else:
#             print("Select mean or median")
#
#     return cluster_metric


# def mean_dtw_within_cluster_per_condition(X_time_cond):
#     """
#     X_time_cond: (n_peptides, n_timepoints, 3) where dim 0..2 are conditions.
#     Returns dict with mean DTW per condition.
#     """
#     out = {}
#     for cond_i, cond_name in enumerate(["EGF", "INS", "EGFnINS"]):
#         Xc = X_time_cond[:, :, cond_i][:, :, None]
#         D = cdist_dtw(Xc)
#         out[cond_name] = float(np.mean(D[np.triu_indices_from(D, k=1)]))
#     return out


# def cluster_similarity_per_condition(
#         df,
#         transpose=True,
#         data_type="log2_FC",
#         cluster_column_name="",
#         mean=True,
#         median=False,
#         verbose=False
# ):
#     '''Computes mean DTW per condition (one scalar per condition per cluster).'''
#     column_selection = [element for element in df.columns.tolist() if f"{data_type}" in element and "cluster" not in element]
#
#     cluster_metric = {}
#
#     for cluster in sorted(df[cluster_column_name].unique()):
#         if cluster == 999:
#             continue
#         df1 = df.loc[df[cluster_column_name] == cluster]
#         X1_tp, y_qc = reshape_df(df=df1, time_series=column_selection, labels="site", dimensions=3, len_time_serie=7,
#                                   transpose=transpose, verbose=verbose)
#
#         dist_per_condition = mean_dtw_within_cluster_per_condition(X1_tp)
#         cluster_metric[cluster] = dist_per_condition
#
#     return cluster_metric


# def timepoint_pairwise_distances_within_condition(X_time_cond, summary="mean"):
#     """
#     X_time_cond: (n_peptides, n_timepoints, 3) where last axis is conditions.
#     Computes pairwise distances between peptides at each timepoint, per condition.
#
#     Returns:
#       dict cond -> (T,) mean/median upper-triangle distance per timepoint
#     """
#     X = np.asarray(X_time_cond)
#     n, T, C = X.shape
#     cond_names = ["EGF", "INS", "EGFnINS"]
#
#     results = {}
#
#     for c in range(C):
#         M = X[:, :, c]
#         summary_t = np.full(T, np.nan, dtype=float)
#         if n < 2:
#             results[cond_names[c]] = summary_t
#             continue
#
#         iu = np.triu_indices(n, k=1)
#
#         for t in range(T):
#             v = M[:, t][:, None]
#             Dt = np.abs(v - v.T)
#             tri = Dt[iu]
#             if summary == "mean":
#                 summary_t[t] = float(np.mean(tri))
#             elif summary == "median":
#                 summary_t[t] = float(np.median(tri))
#             else:
#                 raise ValueError("summary must be 'mean' or 'median'")
#
#         results[cond_names[c]] = summary_t
#
#     return results


# def cluster_similarity_per_condition_per_timepoint(
#         df,
#         transpose=True,
#         data_type="log2_FC",
#         cluster_column_name="",
#         mean=True,
#         median=False,
#         verbose=False
# ):
#     '''Computes mean pairwise distance per condition and time point (one array per condition per cluster).'''
#     column_selection = [element for element in df.columns.tolist() if f"{data_type}" in element and "cluster" not in element]
#
#     cluster_metric = {}
#
#     for cluster in sorted(df[cluster_column_name].unique()):
#         if cluster == 999:
#             continue
#         df1 = df.loc[df[cluster_column_name] == cluster]
#         X1_tp, y_qc = reshape_df(df=df1, time_series=column_selection, labels="site", dimensions=3, len_time_serie=7,
#                                   transpose=transpose, verbose=verbose)
#
#         dist_per_condition = timepoint_pairwise_distances_within_condition(X1_tp, summary="mean")
#         cluster_metric[cluster] = dist_per_condition
#
#     return cluster_metric


# def combine_conditions(scores_per_cluster, how="mean", cond_order=("EGF", "INS", "EGFnINS")):
#     """Collapse per-condition scores into a single scalar per cluster."""
#     combined = {}
#     for k, d in scores_per_cluster.items():
#         vals = np.array([d[c] for c in cond_order], dtype=float)
#         if how == "mean":
#             combined[k] = float(np.nanmean(vals))
#         elif how == "max":
#             combined[k] = float(np.nanmax(vals))
#         elif how == "median":
#             combined[k] = float(np.nanmedian(vals))
#         else:
#             raise ValueError("how must be 'mean', 'median', or 'max'")
#     return combined
#
#
# def clusters_shared_peptides(
#     cluster_df,
#     clustering_1: str,
#     clustering_2: str,
#     site: str = None,
#     clusters=None
# ):
#     """
#     Plot a Venn diagram of sites shared between two cluster assignments.
#     Provide either `site` (to infer cluster IDs automatically) or explicit `clusters=[id1, id2]`.
#     """
#     if clusters is None:
#         clusters = [None, None]
#
#     if site:
#         row = cluster_df.loc[cluster_df["site"] == site, [clustering_1, clustering_2]]
#         if row.empty:
#             raise ValueError(f"Site '{site}' not found in cluster_df['site'].")
#         cluster1_id = row.iloc[0][clustering_1]
#         cluster2_id = row.iloc[0][clustering_2]
#     else:
#         if len(clusters) != 2:
#             raise ValueError("`clusters` must be a list/tuple of length 2: [cluster1_id, cluster2_id].")
#         cluster1_id, cluster2_id = clusters
#         if cluster1_id is None or cluster2_id is None:
#             raise ValueError("Provide `site` or both cluster IDs in `clusters=[cluster1_id, cluster2_id]`.")
#
#     set_1 = set(cluster_df.loc[cluster_df[clustering_1] == cluster1_id, "site"].tolist())
#     set_2 = set(cluster_df.loc[cluster_df[clustering_2] == cluster2_id, "site"].tolist())
#
#     plt.figure(figsize=(6, 4))
#     venn2(
#         [set_1, set_2],
#         set_labels=(f"{clustering_1}\nCluster {cluster1_id}", f"{clustering_2}\nCluster {cluster2_id}")
#     )
#     plt.title("Venn Diagram")
#     plt.show()


# =============================================================================
# Network graphs
# =============================================================================

def build_graph_from_edges(df, source_col="node1", target_col="node2", directed=True):
    """Build a directed or undirected graph from an edge list DataFrame."""
    G = nx.DiGraph() if directed else nx.Graph()
    edges = df[[source_col, target_col]].dropna().values.tolist()
    G.add_edges_from(edges)
    return G


def plot_graph(G, title="Directed Network"):
    """Plot a networkx graph with arrows if directed."""
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=42)

    if G.is_directed():
        nx.draw_networkx_nodes(G, pos, node_size=600, node_color="lightblue", edgecolors="black")
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
        nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=20, edge_color="gray")
    else:
        nx.draw(G, pos, with_labels=True, node_size=600, node_color="lightblue", edgecolors="black")

    plt.title(title)
    plt.axis("off")
    plt.show()


# =============================================================================
# Visualisation helpers
# =============================================================================

# Moved to src/plotting_functions.py as plot_cluster_scores() — improved version
# def plot_grouped_bars(scores, cond_order=("EGF", "INS", "EGFnINS"), figsize=(14, 5)):
#     '''Plot per-cluster, per-condition dispersion scores as a grouped bar chart.'''
#     clusters = sorted(scores.keys())
#     x = np.arange(len(clusters))
#     width = 0.25
#
#     plt.figure(figsize=figsize)
#
#     for i, cond in enumerate(cond_order):
#         y = [scores[c][cond] for c in clusters]
#         plt.bar(x + (i - (len(cond_order) - 1) / 2) * width, y, width=width, label=cond)
#
#     plt.xticks(x, clusters, rotation=90)
#     plt.xlabel("Cluster")
#     plt.ylabel("Dispersion score (lower = tighter)")
#     plt.title("Cluster quality per condition")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()