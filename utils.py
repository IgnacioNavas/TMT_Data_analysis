import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from datetime import date
from IPython.display import display, HTML
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import KShape

import networkx as nx
from itertools import cycle
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from matplotlib_venn import venn2
from tslearn.clustering import KernelKMeans
import tslearn as tsl
from tslearn.metrics import cdist_dtw
import math

from tslearn.metrics import cdist_dtw
from sklearn.metrics import silhouette_score


import warnings

# from tslearn_cluster_individual_stimulations import cluster_column


# from Recalculating_FC import cv_list


#%% Optimized
def uniprot_links_for(df,
                      protein_list=[]):
    for protein in protein_list:
        # Check if protein is neither in name nor ID
        if (protein not in df['protein_name'].values) and (protein not in df['protein_Id'].values):
            uniprot_url = f"https://www.uniprot.org/uniprotkb/{protein}"
            html_link = f'Protein {protein} is not in the database: <a href="{uniprot_url}" target="_blank">{protein}</a>'
            display(HTML(html_link))

        # If it's a name, look up ID
        elif protein in df['protein_name'].values:
            protein_ID = df.loc[df['protein_name'] == protein, 'protein_Id'].values[0]
            protein_description = df.loc[df['protein_name'] == protein, 'description'].values[0]
            protein_description = protein_description.split("OS=")[0]
            uniprot_url = f"https://www.uniprot.org/uniprotkb/{protein_ID}"
            html_link = f'Link to protein {protein} <a href="{uniprot_url}" target="_blank">{protein_ID}</a>. {protein_description}'
            display(HTML(html_link))

        # If it's an ID, look up name
        else:
            protein_name = df.loc[df['protein_Id'] == protein, 'protein_name'].values[0]
            protein_description = df.loc[df['protein_name'] == protein, 'description'].values[0]
            protein_description = protein_description.split("OS=")[0]
            uniprot_url = f"https://www.uniprot.org/uniprotkb/{protein}"
            html_link = f'Link to protein {protein_name} <a href="{uniprot_url}" target="_blank">{protein}</a>. {protein_description}'
            display(HTML(html_link))

#%% Optimized for all experiment
def plot_dataset_phosphosites(df,
                              cluster_column = "",
                              cluster_number = int,
                              data_type = str,
                              legend = list,
                              color_palette = ['r', 'b', 'fuchsia'],
                              saving_path=str,
                              dataset_name=str,
                              saving_info="",
                              plot_individually=False,
                              fit_y_lims=False,
                              plot_close=False,
                              save_pdf=False,
                              save_png=False):
    '''Plot all the phosphorilation sites of a dataset. You can decide to plot them separately (although I think it
    does not make sense.'''

    # Check if the df is a pandas dataframe already or the path to it
    if type(df) == pd.DataFrame:
        pass
    else:
        df = pd.read_excel(df)

    if cluster_column != "": # If there would be not entry for the cluster_column the function will plot the whole dataset
        if cluster_number: # If there would be no cluster number to select, I am assuming that the selection has already been done
            df = df.loc[df[cluster_column] == int(cluster_number)]

    # Columns to plot the data
    column_names = df.columns.tolist()
    column_selection = [element for element in column_names if data_type in element]

    # X_axis time points setting
    time_points_previous = [element for element in column_names if f"log2_FC_EGF_" in element] # this could be any set of columns that have the time points
    time_points = [s.split("_")[-1] for s in time_points_previous]

    # Mean columns
    EGF_mean_cols = [col for col in df.columns if any(f"{data_type}_EGF_{t}" in col for t in time_points)]
    INS_mean_cols = [col for col in df.columns if any(f"{data_type}_INS_{t}" in col for t in time_points)]
    EGFnINS_mean_cols = [col for col in df.columns if any(f"{data_type}_EGFnINS_{t}" in col for t in time_points)]

    # sd columns
    if "raw" in data_type:
        EGF_sd_cols = [col for col in df.columns if any(f"raw_sd_EGF_{t}" in col for t in time_points)]
        INS_sd_cols = [col for col in df.columns if any(f"raw_sd_INS_{t}" in col for t in time_points)]
        EGFnINS_sd_cols = [col for col in df.columns if any(f"raw_sd_EGFnINS_{t}" in col for t in time_points)]
    else:
        EGF_sd_cols = [col for col in df.columns if any(f"log2_sd_EGF_{t}" in col for t in time_points)]
        INS_sd_cols = [col for col in df.columns if any(f"log2_sd_INS_{t}" in col for t in time_points)]
        EGFnINS_sd_cols = [col for col in df.columns if any(f"log2_sd_EGFnINS_{t}" in col for t in time_points)]

    # If the file generated is not going to be saved don't create the saving folder
    if save_pdf == False and save_png == False:
        pass
    else:  # If the files are going to be saved, check if the path exist, if not, create it
        if not os.path.exists(saving_path):
            print("Creating saving folder")
            os.makedirs(saving_path)
            ### INCLUDE DATASET NAME????

    # sorting dataframe based on the site, so if two sites belong to a same protein they appear together
    df.sort_values(by=['site'], inplace=True)

    # Geting some basic information and parameters for the plots
    number_phos = len(df)

    if number_phos == 1 or plot_individually == True:  # Plotting individually
        for index, row in df.iterrows():
            # Collect identification data of the phosphorylatio site
            site = row["site"]
            name = row["protein_name"]
            id = row["protein_Id"]
            # Collect data of the time points of the phosphosites
            all_times = row[column_selection].tolist()
            EGF = row[EGF_mean_cols].tolist()
            INS = row[INS_mean_cols].tolist()
            EGFnINS = row[EGFnINS_mean_cols].tolist()
            groups = [EGF, INS, EGFnINS]
            # Collect data of the standard deviation of each timepoint
            EGF_sd = row[EGF_sd_cols].tolist()
            INS_sd = row[INS_sd_cols].tolist()
            EGFnINS_sd = row[EGFnINS_sd_cols].tolist()
            groups_sd = [EGF_sd, INS_sd, EGFnINS_sd]
            # Collect data about number of replicates in which the phosphosite was detected
            n_rep = row["n_rep"]

            uniprot_url = f"https://www.uniprot.org/uniprotkb/{id}"
            html_link = f'Link to protein {name} <a href="{uniprot_url}" target="_blank">{id}</a>'
            display(HTML(html_link))

            fig, ax = plt.subplots(figsize=(7, 4))
            for c in range(3):
                if n_rep == 1:
                    al = 0.3
                else:
                    al = 1
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

    else:  # Plotting all sites of a protein together
        sqrt_n_p = int(np.ceil(np.sqrt(number_phos)))  # Plotting time points in a matrix
        if sqrt_n_p <= 2:  # 1 plot doesn't work, 2 plots leave empty row below
            empty_plots = 0
        else:
            empty_plots = (sqrt_n_p * sqrt_n_p) - number_phos

        # Avoid getting rows with empty plots
        if empty_plots >= sqrt_n_p:
            sqrt_n_p_X = sqrt_n_p - 1
        else:
            sqrt_n_p_X = sqrt_n_p

        # Decide wether to fit the y axes or not
        if fit_y_lims == True:  # Fit "y" limits for each phosphosite
            y_fixed = "_y_axis_fixed"
        elif type(fit_y_lims) == list:
            y_lim_min = fit_y_lims[0]
            y_lim_max = fit_y_lims[1]
            y_fixed = f"_y_axis_fixed_{y_lim_min}_{y_lim_max}"
        else:  # Fit the same "y" limit for all phosphosites
            sub_values_df = df[column_selection]
            y_lim_max = sub_values_df.max().max() * 1.1
            y_lim_min = sub_values_df.min().min() * 1.1
            y_fixed = "_y_axis_general"

        k = 0  # Counter to stop ploting when there is no more phosphosites

        # Seting subplots aprameters
        fig, ax = plt.subplots(sqrt_n_p, sqrt_n_p_X, figsize=(18, 13))  # figsize=(7, 4) figsize=(18, 13) figsize=(29.7, 21)
        fig.tight_layout(w_pad=1.75, h_pad=3)
        plt.subplots_adjust(top=0.94)  # percentage of the figure that the plots are using

        # Go through the sub_df to plot all the phosphosites
        for i in range(sqrt_n_p):  # y
            for j in range(sqrt_n_p_X):  # X
                if k == number_phos:  # Stop plotting, all phosphorylation sites have been plotted
                    continue
                else:
                    # IN THIS CASE I AM ACCESSING THE ROW BY INDEXING,
                    row = df.iloc[k,:]  # Go through the rows of the subdataset with the phsophosites for the protein
                    # Collect identification data of the phosphorylatio site
                    site = row["site"]
                    name = row["protein_name"]
                    id = row["protein_Id"]
                    # Collect data of the time points of the phosphosites
                    all_times = row[column_selection].tolist()
                    EGF = row[EGF_mean_cols].tolist()
                    INS = row[INS_mean_cols].tolist()
                    EGFnINS = row[EGFnINS_mean_cols].tolist()
                    groups = [EGF, INS, EGFnINS]
                    # Collect data of the standard deviation of each timepoint
                    EGF_sd = row[EGF_sd_cols].tolist()
                    INS_sd = row[INS_sd_cols].tolist()
                    EGFnINS_sd = row[EGFnINS_sd_cols].tolist()
                    groups_sd = [EGF_sd, INS_sd, EGFnINS_sd]

                    # Collect data about number of replicates in which the phosphosite was detected
                    n_rep = row["n_rep"]

                    # Start plotting
                    for c in range(3):
                        if n_rep == 1:
                            al = 0.3
                        else:
                            al = 1
                        ax[i, j].errorbar(x=time_points, y=groups[c], yerr=groups_sd[c],
                                          marker='o', color=color_palette[c], alpha=al, capsize=4, elinewidth=1.3)

                        # Subplot parameters
                    ax[i, j].set_title(f"{str(re.findall(r'^.*~', site))[2:-3]}_n{n_rep}")  # , weight='bold'
                    ax[i, j].set_xlabel("Time (min)")
                    ax[i, j].set_ylabel(f"{data_type}")
                    ax[i, j].grid()

                    # Using specific limits
                    if fit_y_lims == True:
                        ax[i, j].set_ylim(min(all_times) * 1.1 - 0.1, max(all_times) * 1.1 + 0.1)
                    else:  # Using general limits
                        ax[i, j].set_ylim(y_lim_min, y_lim_max)
                    ax[i, j].set_xlim(-1, 7)

                    # count the phosphorylation site as plotted
                    k = k + 1

        # General parameters of the plot
        fig.legend(labels=legend, loc="upper right", ncol=len(groups))
        fig.suptitle(f"{dataset_name} {cluster_column} {cluster_number} {saving_info} {date.today()}", weight='bold')

        # Saving the plot
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

#%% Optimized for all experiment
def clusters_plot(df,
                  legend=list,
                  saving_path=str,
                  cluster_column = str,
                  cluster_name="",
                  data_type = str,
                  plot_different_data = False,
                  saving_info="",
                  save_pdf=False,
                  save_png=False,
                  plot_close=False,
                  y_lims_list=False):
    '''Take a dataset that include an extra column for the "cluster" the site belongs to. Calculates the mean for each
     time point and plot it. This way you can see the average curves of each cluster.'''
    if type(df) == pd.DataFrame:
        pass
    else:
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

        # Geting some basic information and parameters for the plots
        n_cluster = len(sorted_clusters)
        sqrt_n_c = int(np.ceil(np.sqrt(n_cluster)))
        empty_plots = (sqrt_n_c * sqrt_n_c) - n_cluster

        # Avoid getting rows with empty plots
        if empty_plots >= sqrt_n_c:
            sqrt_n_c_X = sqrt_n_c - 1
        else:
            sqrt_n_c_X = sqrt_n_c

        i_list = list(range(sqrt_n_c))
        i_c = 0
        j_list = list(range(sqrt_n_c_X))
        j_c = 0

        # Seting subplots aprameters
        fig, ax = plt.subplots(sqrt_n_c, sqrt_n_c_X, figsize=(18, 13))  # figsize=(7, 4) figsize=(18, 13) figsize=(29.7, 21)
        fig.tight_layout(w_pad=1.75, h_pad=3)
        plt.subplots_adjust(top=0.94)  # percentage of the figure that the plots are using

        # time points of the dataset
        column_names = df.columns.tolist()
        time_points_previous = [element for element in column_names if f"log2_FC_EGF_" in element] # this could be any set of columns that have the time points
        time_points = [s.split("_")[-1] for s in time_points_previous]

        EGF_matching_cols = [col for col in df.columns if any(f"{data_type}_EGF_{t}" in col for t in time_points)]
        INS_matching_cols = [col for col in df.columns if any(f"{data_type}_INS_{t}" in col for t in time_points)]
        EGFnINS_matching_cols = [col for col in df.columns if any(f"{data_type}_EGFnINS_{t}" in col for t in time_points)]

        for cluster in sorted_clusters:
            # Create sub-dataframe with only the protein we are interested in. If the protein doesn't exist in the dataframe skip code
            sub_df = df.loc[df[cluster_column] == cluster].copy()

            # Calculate averages for each time points of the phosphosites in the cluster
            EGF_means = [sub_df[col].mean() for col in EGF_matching_cols]
            INS_means = [sub_df[col].mean() for col in INS_matching_cols]
            EGFnINS_means = [sub_df[col].mean() for col in EGFnINS_matching_cols]

            # Calculate standard deviations for each time points of the phosphosites in the cluster
            EGF_err = [sub_df[col].std() for col in EGF_matching_cols]
            INS_err = [sub_df[col].std() for col in INS_matching_cols]
            EGFnINS_err = [sub_df[col].std() for col in EGFnINS_matching_cols]

            groups = [EGF_means, INS_means, EGFnINS_means]
            groups_sd = [EGF_err, INS_err, EGFnINS_err]
            colors = ['r', 'b', 'fuchsia']

            for c in range(3):
                ax[i_list[i_c], j_list[j_c]].errorbar(x=time_points, y=groups[c],
                                                      yerr=groups_sd[c], marker='o', color=colors[c], capsize=4,
                                                      elinewidth=1.3)

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

        # Saving the plot
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

#%%
def reshape_df(df,
               time_series,
               dimensions,
               len_time_serie,
               verbose,
               labels = str,
               transpose = False):
    '''Reshape dataframe so it is multivariate format. Return the dataframe in numpy format so can be used, and list with the names of myseries'''

    sub_df = df[time_series].copy()
    mySeries = sub_df.to_numpy()
    namesofMySeries = df[labels]

    multivariate_shape = (len(df), dimensions, len_time_serie)
    ###        #add an "if" variable so if shape[1] of df 1 !== 21, the selection of the columns is wrong
    if verbose == True and transpose == False:
        print(f"Reshaping dataframe to shape {multivariate_shape}")

    multivariate_df = np.reshape(mySeries, multivariate_shape)
    if transpose == True:
        multivariate_df = multivariate_df.transpose(0, 2, 1)
        if verbose == True:
            print(f"Reshaping dataframe to shape {multivariate_df.shape}")


    return multivariate_df, namesofMySeries

#%% Optimized for all experiment
def plot_protein_phosphosites(df,
                              data_type = str,
                              proteins=list,
                              replicates = False,
                              exclude_rep = list,
                              legend_plot = list,
                              color_palette = ['r', 'b', 'fuchsia'],
                              saving_path=str,
                              saving_info="",
                              title_info = "",
                              plot_individually=False,
                              fit_y_lims=False,
                              plot_close=False,
                              save_pdf=False,
                              save_png=False):
    '''Plot to PDF ALL phosphosites of a list of proteins. You can decide to plot the phosphosties of the protein
    together in one plot or to plot them separatly.'''

    # Check if the df is a pandas dataframe already or the path to it
    if type(df) == pd.DataFrame:
        pass
    else:
        df = pd.read_excel(df)

    # If the file generated is not going to be saved don't create the saving folder
    if save_pdf == False and save_png == False:
        pass
    else:  # If the files are going to be saved, check if the path exist, if not, create it
        if not os.path.exists(saving_path):
            print("Creating saving folder")
            os.makedirs(saving_path)

    for protein in proteins:
        # Create sub-dataframe with only the protein we are interested in. If the protein doesn't exist in the dataframe skip code
        if protein in df['protein_name'].to_list():
            sub_df = df.loc[df['protein_name'] == protein].copy()
            print(f"Ploting sites of protein {protein}")
        elif protein in df['protein_Id'].to_list():
            sub_df = df.loc[df['protein_Id'] == protein].copy()
            print(f"Ploting sites of protein {protein}")
        else:
            print(f"The protein {protein} is not present in the dataset")
            continue

        # Extract the protein name and protein uniprot code for the folder
        saving_folder = f"{list(sub_df.protein_name)[0]}_{list(sub_df.protein_Id)[0]}"

        # Check if a folder for the desired protein exists. If no, create one
        if save_pdf == False and save_png == False:
            pass
        else:
            if saving_folder in os.listdir(saving_path):
                pass
            else:
                new_path = f"{saving_path}/{saving_folder}"
                print(f"Createating saving folder for {saving_folder}")
                os.makedirs(new_path)

        # Sort the pepetides of the dataframe for better interpretation of the figure generated
        sub_df.sort_values(by=['site'], inplace=True)

        # Determine the dimentional space for the subplot
        number_phos = len(sub_df)
        sqrt_n_p = int(np.ceil(np.sqrt(number_phos)))  # Plotting time points in a matrix
        if sqrt_n_p <= 2:  # 1 plot doesn't work, 2 plots leave empty row below
            empty_plots = 0
        else:
            empty_plots = (sqrt_n_p * sqrt_n_p) - number_phos
        # Avoid getting rows with empty plots
        if empty_plots >= sqrt_n_p:
            sqrt_n_p_X = sqrt_n_p - 1
        else:
            sqrt_n_p_X = sqrt_n_p

        # Columns to determine the "y" axis limit values
        column_names = df.columns.tolist()
        # column_selection = [element for element in column_names if data_type in element]
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
                # Y axis limit
                if fit_y_lims == True:
                    y_fixed = "y_axis_fixed"
                elif type(fit_y_lims) == list:
                    y_lim_min = fit_y_lims[0]
                    y_lim_max = fit_y_lims[1]
                    y_fixed = f"_y_axis_fixed_{y_lim_min}_{y_lim_max}"
                elif fit_y_lims == False:
                    sub_values_df = sub_df.loc[:,column_selection]
                    y_lim_max = (sub_values_df.max().max()) * 1.05
                    y_lim_min = (sub_values_df.min().min()) * 0.95
                    if y_lim_min < 0:
                        y_lim_min = (sub_values_df.min().min()) + (sub_values_df.min().min())* 0.1

                fig, axes = plt.subplots()
                plot_data(ax = axes, row_df = row, replicates = replicates, data_type= data_type, colors = color_palette, legend = legend_plot, exclude_rep=exclude_rep, plot_individually=plot_individually)
                # Apply the y limit axis
                if fit_y_lims == True:
                    if data_type == "raw" or data_type == "log2_FC":
                        if min(row[column_selection]) < 0 :
                            y_lim_min = min(row[column_selection]) + min(row[column_selection])*0.1
                            axes.set_ylim(y_lim_min, max(row[column_selection])*1.1)
                        else:
                            axes.set_ylim(min(row[column_selection])*0.9, max(row[column_selection])*1.1)
                    else:
                        axes.set_ylim(min(row[column_selection])*0.97, max(row[column_selection])*1.02)
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
            fig, axes = plt.subplots(sqrt_n_p, sqrt_n_p_X , figsize=(18, 13))  #
            fig.tight_layout(w_pad=1.75, h_pad=3)
            plt.subplots_adjust(top=0.94)

            # Force axes into a 2D array
            if len(sub_df) == 1:
                axes = np.array([[axes]])  # Wrap single Axes in a 2D array
            elif sqrt_n_p == 1 or sqrt_n_p_X == 1:
                axes = np.atleast_2d(axes)

            # Y axis limit
            if fit_y_lims == True:
                y_fixed = "y_axis_fixed"
            elif type(fit_y_lims) == list:
                y_lim_min = fit_y_lims[0]
                y_lim_max = fit_y_lims[1]
                y_fixed = f"_y_axis_fixed_{y_lim_min}_{y_lim_max}"
            elif fit_y_lims == False:
                sub_values_df = sub_df.loc[:,column_selection]
                y_lim_max = (sub_values_df.max().max()) * 1.02
                y_lim_min = (sub_values_df.min().min()) * 0.97
                if y_lim_min < 0:
                    y_lim_min = (sub_values_df.min().min()) + (sub_values_df.min().min())* 0.1

            for i in range(sqrt_n_p):  # y
                for j in range(sqrt_n_p_X):  # X
                    if k >= number_phos:  # Stop plotting, all phosphorylation sites have been plotted
                        fig.delaxes(axes[i, j])

                    else:
                        row = sub_df.iloc[k,:]
                        # print(data_type)

                        plot_data(ax = axes[i,j], row_df = row, replicates = replicates,  data_type= data_type, colors = color_palette, legend = legend_plot, exclude_rep=exclude_rep, plot_individually=plot_individually) # data_type= data_type,

                        if fit_y_lims == True:
                            if data_type == "raw" or data_type == "log2_FC":
                                if min(row[column_selection]) < 0 :
                                    y_lim_min = min(row[column_selection]) + min(row[column_selection])*0.1
                                    axes[i,j].set_ylim(y_lim_min, max(row[column_selection])*1.1)
                                else:
                                    axes[i,j].set_ylim(min(row[column_selection])*0.9, max(row[column_selection])*1.1)
                            else:
                                axes[i,j].set_ylim(min(row[column_selection])*0.97, max(row[column_selection])*1.02)
                        else:
                            axes[i, j].set_ylim(y_lim_min, y_lim_max)

                        axes[i, j].set_xlim(-1, 7)
                        # fig.tight_layout()
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


#%% Optimized for all experiments
def plot_data(ax,
              row_df,
              replicates = False,
              data_type = str,
              colors = [],
              legend = [],
              exclude_rep = list,
              plot_individually = False,):

    column_names = row_df.index.tolist()

    # data_type = "raw"
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
        # print(EGF_mean)

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

    # print(data_type)
    if data_type == "raw":
        data_type2 = "raw_abs"
    else: #data_type == "log2":
        data_type2 = "log2_abs"
    # else:
    #     data_type2 = "log2"
    # print(data_type)

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

    x_axis_previous = [element for element in column_names if f"log2_FC_EGF_" in element] # this could be any set of columns that have the time points
    x_axis = [s.split("_")[-1] for s in x_axis_previous]

    # print(data_type, data_type2)
    if data_type in ["raw", "log2", "log2_FC", "FC_scaled"]:
        mean_all = [EGF_mean, INS_mean, EGFnINS_mean]
        sd_all = [EGF_sd, INS_sd, EGFnINS_sd]

        n_rep = row_df["n_rep"]
        site = row_df["site"]
        prot_name = row_df["protein_name"]
        protein_ID = row_df["protein_Id"]

        # fig, ax = plt.subplots() #figsize=(7, 4)
        for c in range(3):
            if n_rep == 1:
                al = 0.3
            else:
                al = 1

            y_error = row_df[sd_all[c]]
            if data_type == "FC_scaled":
                y_error = [0,0,0,0,0,0,0]
            ax.errorbar(x=x_axis, y=row_df[mean_all[c]], yerr= y_error, marker='o',color=colors[c], label=legend[c], capsize=4, elinewidth=1.3, alpha=al)

        if n_rep == 1:
            replicates = False
        if replicates == True:
            for element in rep_list:
                if element in exclude_rep:
                    continue
                else:
                    c=0
                    for condition in rep_list[element]:
                        if len(set(row_df[condition].values)) == 1: #There are some peptides that are not present fot specific time points in replictes like P00533_1190_1197_2_2_T1191Y1197~GStAENAEyLR
                            continue
                        else:
                            ax.scatter(x = x_axis, y=row_df[condition], marker='x', color=colors[c],  alpha=0.7, s=20)
                            c+=1
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(f"{data_type}")
        ax.grid()
        # ax.set_title(f"{site}_n{n_rep}")
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

#%% Optimized for all experiments
def plot_protein_profile(df,
                         proteins,
                         data_type = str,
                         saving_path="",
                         saving_info="",
                         legend=False,
                         save_pdf=False,
                         save_png=False):

    # Load DataFrame from file if needed
    if not isinstance(df, pd.DataFrame):
        df = pd.read_excel(df)

    # Create saving path if necessary
    if (save_pdf or save_png) and not os.path.exists(saving_path):
        print("Creating saving folder")
        os.makedirs(saving_path)

    column_names = df.columns.tolist()
    all_conditions = [element for element in column_names if f"{data_type}_" in element]
    EGF = [element for element in column_names if f"{data_type}_EGF_" in element]
    INS = [element for element in column_names if f"{data_type}_INS_" in element]
    EGFnINS = [element for element in column_names if f"{data_type}_EGFnINS_" in element]

    x_axis_previous = [element for element in column_names if f"log2_FC_EGF_" in element] # this could be any set of columns that have the time points
    time_points = [s.split("_")[-1] for s in x_axis_previous]

    fig, ax = plt.subplots(len(proteins), 3, figsize=(10, 2 * len(proteins)))
    if len(proteins) == 1:
        ax = [ax]  # Handle case of single protein correctly

    for c, protein in enumerate(proteins):
        # Subset the DataFrame
        if protein in df['protein_name'].values:
            sub_df = df[df['protein_name'] == protein].copy()
        elif protein in df['protein_Id'].values:
            sub_df = df[df['protein_Id'] == protein].copy()
        else:
            print(f"The protein {protein} is not present in the dataset.")
            continue

        # print(f"Plotting sites of protein {protein}")
######
        protein_for_url = str(sub_df['protein_Id'].values[0])
        prot_name = str(sub_df['protein_name'].values[0])
        uniprot_url = f"https://www.uniprot.org/uniprotkb/{protein_for_url}"
        html_link = f'Plotting sites of protein <a href="{uniprot_url}" target="_blank">{protein_for_url}</a> {prot_name}'
        display(HTML(html_link))
######
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

        # Y-axis limits
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

    # Save
    if save_pdf:
        plt.savefig(os.path.join(saving_path, f"{saving_info}.pdf"))
    if save_png:
        plt.savefig(os.path.join(saving_path, f"{saving_info}.png"))
    if not save_pdf and not save_png:
        print(f"{saving_info} Plot not saved")

    plt.show()

#%%  Optimized for all dataset
def plot_protein_profiles_fine_line(df,
                                    proteins=list,
                                    data_type=str,
                                    saving_path=str,
                                    legend=False,
                                    saving_info = str,
                                    save_pdf=False,
                                    save_png=False):
    # Check if the df is a pandas dataframe already or the path to it
    if type(df) == pd.DataFrame:
        pass
    else:
        df = pd.read_excel(df)

    # If the file generated is not going to be saved don't create the saving folder
    if save_pdf == False and save_png == False:
        pass
    else:  # If the files are going to be saved, check if the path exist, if not, create it
        if not os.path.exists(saving_path):
            print("Creating saving folder")
            os.makedirs(saving_path)

    column_names = df.columns.tolist()
    all_conditions = [element for element in column_names if f"{data_type}_" in element]
    EGF = [element for element in column_names if f"{data_type}_EGF_" in element]
    INS = [element for element in column_names if f"{data_type}_INS_" in element]
    EGFnINS = [element for element in column_names if f"{data_type}_EGFnINS_" in element]

    x_axis_previous = [element for element in column_names if
                       f"log2_FC_EGF_" in element]  # this could be any set of columns that have the time points
    time_points = [s.split("_")[-1] for s in x_axis_previous]

    for protein in proteins:
        # Create sub-dataframe with only the protein we are interested in. If the protein doesn't exist in the dataframe skip code
        if protein in df['protein_name'].to_list():
            sub_df = df.loc[df['protein_name'] == protein].copy()
            # print(f"Ploting sites of protein {protein}")
        elif protein in df['protein_Id'].to_list():
            sub_df = df.loc[df['protein_Id'] == protein].copy()
            # print(f"Ploting sites of protein {protein}")
        else:
            print(f"The protein {protein} is not present in the dataset")
            continue
        ########### #
        protein_for_url = str(sub_df['protein_Id'].values[0])
        prot_name = str(sub_df['protein_name'].values[0])
        uniprot_url = f"https://www.uniprot.org/uniprotkb/{protein_for_url}"
        html_link = f'Plotting sites of protein <a href="{uniprot_url}" target="_blank">{protein_for_url}</a> {prot_name}'
        display(HTML(html_link))
        ###########
        # Extract the protein name and protein uniprot code for the folder
        saving_folder = f"{list(sub_df.protein_name)[0]}_{list(sub_df.protein_Id)[0]}"

        # Check if a folder for the desired protein exists. If no, create one
        if save_pdf == False and save_png == False:
            pass
        else:
            if saving_folder in os.listdir(saving_path):
                pass
            else:
                new_path = f"{saving_path}/{saving_folder}"
                print(f"Createating saving folder for {saving_folder}")
                os.makedirs(new_path)

        # Sort the pepetides of the dataframe for better interpretation of the figure generated
        sub_df.sort_values(by=['site'], inplace=True)

        fig, ax = plt.subplots(1, 3, figsize=(20, 5))
        for index, row in sub_df.iterrows():
            ax[0].errorbar(x=time_points,
                           y=row[EGF]
                           # y = row[["FC_EGF_full", "FC_EGF_starve", "FC_EGF1","FC_EGF2", "FC_EGF5", "FC_EGF10",  "FC_EGF90"]]
                           )
            ax[0].title.set_text("EGF")

            ax[1].errorbar(x=time_points,
                           y=row[EGFnINS]
                           # y = row[["FC_EGFnINS_full", "FC_EGFnINS_starve", "FC_EGFnINS1", "FC_EGFnINS2", "FC_EGFnINS5", "FC_EGFnINS10",  "FC_EGFnINS90"]]
                           )
            ax[1].title.set_text("EGFnINS")

            ax[2].errorbar(x=time_points,
                           y=row[INS]
                           # y = row[["FC_INS_full", "FC_INS_starve", "FC_INS1", "FC_INS2", "FC_INS5", "FC_INS10", "FC_INS90"]]
                           )
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

#%% Optimized (could have more variables)
def tslearn_clustering_KMeans(df_to_cluster,
                              data_type,
                              condition_for_clustering = list,
                              exclude_full = False,
                              cluster_column_name = str,
                              number_of_clusters = int,
                              max_iterations = 1000,
                              n_init = 5,
                              metric='euclidean',
                              df_dimensions = int,
                              random_state = 0,
                              time_series_length = int,
                              transpose = False,
                              verbose = True,
                              testing = False,
                              barycenter_calculations = False):

    column_names = df_to_cluster.columns.tolist()
    if len(condition_for_clustering) == 0:
        column_selection = [element for element in column_names if element.startswith(f"{data_type}")] #f"{data_type}" in element]
        if exclude_full == True:
         column_selection = [element for element in column_names if element.startswith(f"{data_type}") and "full" not in element]
    else:
        column_selection = [element for element in column_names if element.startswith(f"{data_type}") and any(cond in element for cond in condition_for_clustering)]
        if exclude_full == True:
            column_selection = [element for element in column_names if element.startswith(f"{data_type}") and any(cond in element for cond in condition_for_clustering) and "full" not in element]
    if verbose == True:
        print(f"Column selection: {column_selection}\n")

    multivariate_df, names_of_myseries = reshape_df(df = df_to_cluster, time_series = column_selection, dimensions = df_dimensions, len_time_serie = time_series_length, transpose=transpose, labels = "site", verbose=verbose)

    if verbose == True:
        print(f"\nThe size of the dataset is {multivariate_df.shape}")
        print(f"Example:\n{multivariate_df[0]}")

    clustering = TimeSeriesKMeans(n_clusters=number_of_clusters, max_iter=max_iterations, n_init=n_init,  metric=metric, max_iter_barycenter=1000, verbose=verbose, random_state=random_state).fit(multivariate_df)
        # I dont fing a difference between using fit() or using fit_predict()
    df_to_cluster[f"{cluster_column_name}"] = clustering.labels_# If I use fit_predict(), I dont need ".labels_"

    if testing == True:
        if barycenter_calculations == True:
            barycenters_distances = TimeSeriesKMeans(n_clusters=number_of_clusters, max_iter=max_iterations, n_init=n_init,
                                          metric=metric, max_iter_barycenter=1000, verbose=verbose,
                                          random_state=random_state).fit_transform(multivariate_df)
            # print("Returning the following elements in order: "
            #       "\n- dataframe with the clusters made"
            #       "\n- clustering model (model.fit()) (here you can apply functions for the model as .labels_"
            #       "\n- multivariate_df (np array to cluster (n, t, d) (the np.array)")
            return df_to_cluster, clustering, multivariate_df, barycenters_distances
        else:
            return df_to_cluster, clustering, multivariate_df
    else:
        return df_to_cluster

#%%
def tslearn_clustering_KShape(df_to_cluster,
                              data_type,
                              condition_for_clustering = list,
                              exclude_full = False,
                              number_of_clusters = int,
                              cluster_column_name = str,
                              max_iterations = 1000,
                              # metric='euclidean',
                              df_dimensions = int,
                              time_series_length = int,
                              transpose = False,
                              verbose = True ):

    column_names = df_to_cluster.columns.tolist()
    if len(condition_for_clustering) == 0:
        column_selection = [element for element in column_names if element.startswith(f"{data_type}")] #f"{data_type}" in element]
        if exclude_full == True:
         column_selection = [element for element in column_names if element.startswith(f"{data_type}") and "full" not in element]
    else:
        column_selection = [element for element in column_names if element.startswith(f"{data_type}") and any(cond in element for cond in condition_for_clustering)]
        if exclude_full == True:
            column_selection = [element for element in column_names if element.startswith(f"{data_type}") and any(cond in element for cond in condition_for_clustering) and "full" not in element]
    if verbose == True:
        print(f"Column selection: {column_selection}\n")

    multivariate_df, names_of_myseries = reshape_df(df = df_to_cluster, time_series = column_selection, dimensions = df_dimensions, len_time_serie = time_series_length, transpose=transpose, labels = "site",verbose=verbose)

    print(f"\nThe size of the dataset is {multivariate_df.shape}")

    clustering = KShape(n_clusters=number_of_clusters, max_iter=max_iterations, n_init = 10, verbose=verbose, random_state=0).fit(multivariate_df)

    df_to_cluster[f"{cluster_column_name}"] = clustering.labels_

    return df_to_cluster

#%%
def build_graph_from_edges(df, source_col="node1", target_col="node2", directed=True):
    """
    Build a directed or undirected graph from edge list.
    """
    G = nx.DiGraph() if directed else nx.Graph()
    edges = df[[source_col, target_col]].dropna().values.tolist()
    G.add_edges_from(edges)
    return G

def plot_graph(G, title="Directed Network"):
    """
    Plot a networkx graph with arrows if directed.
    """
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

#%% Optimized for all data
def plot_volcano(
    df,
    fc_col,
    pval_col,
    fc_thresh=1.0,
    pval_thresh=0.05,
    title=None,
    ax=None,
    highlight_proteins=None,            # str or list[str]
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
        match_cols (tuple[str]): columns to match against (default: protein_Id, protein_name)
        case_insensitive (bool): if True, case-insensitive matching
    """
    # Prepare axes
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
        created_fig = True

    # Extract and transform
    log2fc = df[fc_col]
    pvals = df[pval_col]
    pvals = np.where(np.asarray(pvals, dtype=float) <= 0, np.nan, pvals)
    neg_log10_pval = -np.log10(pvals)

    # Significance mask
    sig = (np.abs(log2fc) >= fc_thresh) & (df[pval_col] <= pval_thresh)

    # Base scatter
    ax.scatter(log2fc[~sig], neg_log10_pval[~sig], color="grey", alpha=0.6, s=20, label="not significant")
    ax.scatter(log2fc[sig],  neg_log10_pval[sig],  color="red",  alpha=0.8, s=30, label="significant")

    # Threshold lines
    ax.axhline(-np.log10(pval_thresh), color="blue", linestyle="--", linewidth=1)
    ax.axvline(-fc_thresh, color="blue", linestyle="--", linewidth=1)
    ax.axvline(fc_thresh,  color="blue", linestyle="--", linewidth=1)

    # Highlight logic
    if highlight_proteins is not None:
        if isinstance(highlight_proteins, str):
            highlight_list = [highlight_proteins]
        else:
            highlight_list = list(highlight_proteins)

        # Prepare comparison columns (with case normalization if requested)
        comp_cols = [c for c in match_cols if c in df.columns]
        norm_cols = {}
        for c in comp_cols:
            if case_insensitive:
                norm_cols[c] = df[c].astype(str).str.lower()
            else:
                norm_cols[c] = df[c].astype(str)

        # Color cycle
        color_cycle = cycle(plt.cm.tab10.colors if hasattr(plt.cm, "tab10") else ["gold", "cyan", "magenta", "yellow", "green", "blue"])

        for item in highlight_list:
            if item is None:
                continue
            item_key = item.lower() if case_insensitive else str(item)

            # build mask across chosen columns
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

    # Labels & legend
    ax.set_xlabel("log2 Fold Change")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title(title or f"Volcano: {fc_col}")
    ax.legend(fontsize=8, frameon=False)
    if fit_x_limit is not False:
        if fit_x_limit == True:
            ax.set_xlim(-6,6)
        else:
            ax.set_xlim(fit_x_limit[0],fit_x_limit[1])

    if created_fig:
        plt.tight_layout()
        plt.show()

#%% Optimized for all data
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

    # Clean p-values: avoid log(0)
    d[pval_col] = pd.to_numeric(d[pval_col], errors="coerce")
    d.loc[d[pval_col] <= 0, pval_col] = np.nan
    d["neglog10p"] = -np.log10(d[pval_col])

    # Significance classification
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

    # Threshold lines
    fig.add_hline(y=-np.log10(pval_thresh), line_dash="dash")
    fig.add_vline(x=fc_thresh, line_dash="dash")
    fig.add_vline(x=-fc_thresh, line_dash="dash")

    fig.update_layout(
        xaxis_title="log2 Fold Change",
        yaxis_title="-log10(p-value)",
        legend_title="",
    )
    return fig


    fig.add_hline(y=-np.log10(pval_thresh), line_dash="dash")
    fig.add_vline(x=fc_thresh, line_dash="dash")
    fig.add_vline(x=-fc_thresh, line_dash="dash")

    fig.update_layout(
        xaxis_title="log2 Fold Change",
        yaxis_title="-log10(p-value)",
        legend_title="",
    )
    return fig

#%%
def plot_volcano_interactive_plotly_highlighting(
    df: pd.DataFrame,
    fc_col: str,
    pval_col: str,
    site_col: str = "site",
    *,
    fc_thresh: float = 1.0,
    pval_thresh: float = 0.05,
    title: str = None,
    highlight_proteins=None,                 # str or list[str]
    match_cols=("protein_Id", "protein_name"),
    case_insensitive: bool = True,
    show_highlight_labels: bool = False,     # show text labels on highlighted points
):
    """
    Interactive volcano plot (Plotly) with multi-protein highlighting.

    - x: log2FC column (fc_col)
    - y: -log10(p-value) from pval_col
    - Hover shows site + details
    - `highlight_proteins` can be a string or list of strings; matches any of `match_cols`
    """

    # --- prepare base data ---
    d = df[[fc_col, pval_col]].copy()
    if site_col in df.columns:
        d[site_col] = df[site_col]
    for c in match_cols:
        if c in df.columns:
            d[c] = df[c]

    # clean p-values (avoid log(0))
    d[pval_col] = pd.to_numeric(d[pval_col], errors="coerce")
    d.loc[d[pval_col] <= 0, pval_col] = np.nan
    d["neglog10p"] = -np.log10(d[pval_col])

    # significance class
    d["signif"] = np.where(
        (d[fc_col].abs() >= fc_thresh) & (d[pval_col] <= pval_thresh),
        "significant",
        "not significant",
    )

    # --- base scatter ---
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
            **({ "protein_Id": True } if "protein_Id" in d.columns else {}),
            **({ "protein_name": True } if "protein_name" in d.columns else {}),
        },
        title=title or f"Volcano: {fc_col}",
        template="plotly_white",
    )

    # threshold lines
    fig.add_hline(y=-np.log10(pval_thresh), line_dash="dash", line_color="gray")
    fig.add_vline(x=fc_thresh, line_dash="dash", line_color="gray")
    fig.add_vline(x=-fc_thresh, line_dash="dash", line_color="gray")

    fig.update_layout(
        xaxis_title="log2 Fold Change",
        yaxis_title="-log10(p-value)",
        legend_title="",
    )

    # --- highlighting ---
    if highlight_proteins is not None:
        if isinstance(highlight_proteins, str):
            highlight_list = [highlight_proteins]
        else:
            highlight_list = list(highlight_proteins)

        # normalized lookup columns
        comp_cols = [c for c in match_cols if c in d.columns]
        norm_df = {}
        for c in comp_cols:
            norm_df[c] = d[c].astype(str).str.lower() if case_insensitive else d[c].astype(str)

        # colors cycle (distinct from base)
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
                continue  # nothing to highlight for this item

            color = palette[color_idx % len(palette)]
            color_idx += 1

            # optional labels (protein_name if available else protein_Id else site)
            if show_highlight_labels:
                if "protein_name" in d.columns:
                    text_vals = df.loc[mask, "protein_name"].astype(str)
                elif "protein_Id" in d.columns:
                    text_vals = df.loc[mask, "protein_Id"].astype(str)
                else:
                    text_vals = (df.loc[mask, site_col].astype(str)
                                 if site_col in df.columns else pd.Series([""]*mask.sum()))
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
                    f"log2FC: %{ 'x' }:.3f<br>"
                    f"-log10(p): %{ 'y' }:.3f<br>"
                    + (f"{site_col}: %{{customdata[0]}}<br>" if site_col in d.columns else "")
                    + ("%{text}" if show_highlight_labels else "")
                ),
                customdata=d.loc[mask, [site_col]].values if site_col in d.columns else None,
            ))

    return fig

#%%
def filter_dynamics_extremes(
    df,
    data_type="log2_FC",
    threshold=0.5,
    exclude_full = True):
    """

    """
    # Select relevant columns
    if exclude_full == True:
        column_selection = [element for element in df.columns if
                            element.startswith(f"{data_type}") and "full" not in element]
    else:
        column_selection = [c for c in df.columns if data_type in c and "statistics" not in c]

    mask = df[column_selection].abs().max(axis=1) >= threshold
    return df.loc[mask].copy()

#%%
def filter_dynamics_within(
    df,
    data_type="log2_FC",
    threshold=0.5,
    exclude_full = True):
    """

    """
    # Select relevant columns
    if exclude_full == True:
        column_selection = [element for element in df.columns if
                            element.startswith(f"{data_type}") and "full" not in element]
    else:
        column_selection = [c for c in df.columns if data_type in c and "statistics" not in c]

    mask = df[column_selection].abs().max(axis=1) <= threshold
    return df.loc[mask].copy()

#%%
def dynamics_values(
        df,
        data_type="log2_FC",
        exclude_full = True,):
    if exclude_full == True:
        column_selection = [element for element in df.columns if
                            element.startswith(f"{data_type}") and "full" not in element]
    else:
        column_selection = [c for c in df.columns if data_type in c and "statistics" not in c]

    return df[column_selection].copy()


#%%
def filter_replicates(df, n_reps = int):
    '''Return df with the same amount of replicates or more'''
    return df.loc[df["n_rep"] >= n_reps]

#%%
def filter_site_localizations(df, loc_sites = False):
    if loc_sites == False:
        return df.loc[df["localized_sites"] == 0]
    else:
        return df.loc[df["localized_sites"] > 0]

#%%
def filter_ERK_motif(df, ERK_motif = False):
    if ERK_motif == False:
        return df.loc[df["ERK_motif"] == 0]
    else:
        return df.loc[df["ERK_motif"] == 1]

#%%
def filter_functional_score(df, f_score = int):
    return df.loc[df["functional_score"] >= f_score]

#%%
def clusters_shared_peptides(
    cluster_df,
    clustering_1: str,
    clustering_2: str,
    site: str = None,
    clusters=None
):
    """
    Plot a Venn diagram of sites shared between:
      - cluster 'clusters[0]' in column clustering_1
      - cluster 'clusters[1]' in column clustering_2

    If `site` is provided, the function infers those two cluster IDs from the row
    corresponding to that site.
    """
    if clusters is None:
        clusters = [None, None]

    # Infer cluster IDs from `site` if provided
    if site:
        row = cluster_df.loc[cluster_df["site"] == site, [clustering_1, clustering_2]]
        if row.empty:
            raise ValueError(f"Site '{site}' not found in cluster_df['site'].")

        cluster1_id = row.iloc[0][clustering_1]
        cluster2_id = row.iloc[0][clustering_2]
    else:
        if len(clusters) != 2:
            raise ValueError("`clusters` must be a list/tuple of length 2: [cluster1_id, cluster2_id].")
        cluster1_id, cluster2_id = clusters

        if cluster1_id is None or cluster2_id is None:
            raise ValueError("Provide `site` or both cluster IDs in `clusters=[cluster1_id, cluster2_id]`.")

    # Build sets on the FULL dataframe (not filtered to one site)
    set_1 = set(cluster_df.loc[cluster_df[clustering_1] == cluster1_id, "site"].tolist())
    set_2 = set(cluster_df.loc[cluster_df[clustering_2] == cluster2_id, "site"].tolist())

    plt.figure(figsize=(6, 4))
    venn2(
        [set_1, set_2],
        set_labels=(f"{clustering_1}\nCluster {cluster1_id}", f"{clustering_2}\nCluster {cluster2_id}")
    )
    plt.title("Venn Diagram")
    plt.show()

#%%
def kernnel_clustering(df,
                       transpose = True,
                       data_type = "log2_FC",
                       exclude_full = True,
                       condition_for_clustering = list,
                       df_dimensions = int,
                       time_series_length = int,
                       seed = 0,
                       n_clusters = 25,
                       n_init = 20,
                       verbose = True,
                       kernel = "gak",
                       kernel_params={"sigma": "auto"},
                       cluster_column_name = ""
                       ):

    column_names = df.columns.tolist()
    if len(condition_for_clustering) == 0:
        column_selection = [element for element in column_names if element.startswith(f"{data_type}")] #f"{data_type}" in element]
        if exclude_full == True:
         column_selection = [element for element in column_names if element.startswith(f"{data_type}") and "full" not in element]
    else:
        column_selection = [element for element in column_names if element.startswith(f"{data_type}") and any(cond in element for cond in condition_for_clustering)]
        if exclude_full == True:
            column_selection = [element for element in column_names if element.startswith(f"{data_type}") and any(cond in element for cond in condition_for_clustering) and "full" not in element]
    if verbose == True:
        print(f"Column selection: {column_selection}\n")

    X, y = reshape_df(df=df, time_series=column_selection, labels="site", dimensions = df_dimensions, len_time_serie = time_series_length,  transpose= transpose,verbose=verbose)

    if verbose == True:
        print(f"\nThe size of the dataset is {X.shape}")
        print(f"Example:\n{X[0]}")

    gak_km = KernelKMeans(n_clusters=n_clusters,
                          kernel=kernel,
                          kernel_params=kernel_params,
                          n_init=n_init,
                          verbose=verbose,
                          random_state=seed)

    clusters_predicted = gak_km.fit_predict(X)
    df[f"{cluster_column_name}"] = clusters_predicted

    return df

#%%
def cluster_similarity_cdist_dtw(
        df,
        transpose = True,
        data_type = "log2_FC",
        cluster_column_name = str,
        mean = True,
        median = False,
        verbose = False
):
    '''
    DTW distance between two peptides’ full multivariate time series (7 timepoints), where each timepoint is a 3D vector.

    In other words, for two peptides i and j, DTW is aligning their trajectories over time and, at each aligned timepoint, it uses a vector distance in 3D (your 3 dims).

    cluster_metric[cluster] is The average pairwise DTW distance between peptides within that cluster, computed on the full multivariate time series

    This is a reasonable shape-based cluster compactness score (if that’s what you want), but it is not “distance at each timepoint” and it is not “within-condition only”.
    '''
    column_selection = [element for element in df.columns.tolist() if f"{data_type}" in element and "cluster" not in element]

    cluster_metric = {}

    for cluster in sorted(df[cluster_column_name].unique()):
        if cluster == 999:
            continue
        # print(cluster)
        df1 = df.loc[df[cluster_column_name] == cluster]
        X1_tp, y_qc = reshape_df(df=df1, time_series=column_selection, labels="site", dimensions=3, len_time_serie=7,
                                 transpose=transpose, verbose=verbose)

        dist_tp = tsl.metrics.cdist_dtw(dataset1=X1_tp)
        if mean == True and median == False:
            mean_dtw_tp = np.mean(dist_tp[np.triu_indices_from(dist_tp, k=1)])
            cluster_metric[cluster] = mean_dtw_tp
        elif mean == False and median == True:
            median_dtw_tp = np.median(dist_tp[np.triu_indices_from(dist_tp, k=1)])
            cluster_metric[cluster] = median_dtw_tp
        else:
            print("Select mean or median")

    return cluster_metric

#%%
def mean_dtw_within_cluster_per_condition(X_time_cond):
    """
    X_time_cond: (n_peptides, n_timepoints, 3) where dim 0..2 are conditions
    Returns dict with mean DTW per condition.
    """
    out = {}
    for cond_i, cond_name in enumerate(["EGF", "INS", "EGFnINS"]):
        Xc = X_time_cond[:, :, cond_i][:, :, None]  # (n, T, 1) univariate for tslearn
        D = cdist_dtw(Xc)
        out[cond_name] = float(np.mean(D[np.triu_indices_from(D, k=1)]))
    return out

def cluster_similarity_per_condition(
        df,
        transpose = True,
        data_type = "log2_FC",
        cluster_column_name = str,
        mean = True,
        median = False,
        verbose = False
):
    '''value per condition, not per time point per condition
    per condition there is one value'''
    column_selection = [element for element in df.columns.tolist() if f"{data_type}" in element and "cluster" not in element]

    cluster_metric = {}

    for cluster in sorted(df[cluster_column_name].unique()):
        if cluster == 999:
            continue
        # print(cluster)
        df1 = df.loc[df[cluster_column_name] == cluster]
        X1_tp, y_qc = reshape_df(df=df1, time_series=column_selection, labels="site", dimensions=3, len_time_serie=7,
                                 transpose=transpose, verbose=verbose)
        print(X1_tp.shape)

        dist_per_condition = mean_dtw_within_cluster_per_condition(X1_tp)
        cluster_metric[cluster] = dist_per_condition


    return cluster_metric

#%%
def plot_grouped_bars(scores,
                      cond_order=("EGF", "INS", "EGFnINS"),
                      figsize=(14, 5)):
    '''plot dictionary of scores as bat plot, of per condition values.
    Per condition there is one value'''
    clusters = sorted(scores.keys())
    x = np.arange(len(clusters))
    width = 0.25

    plt.figure(figsize=figsize)

    for i, cond in enumerate(cond_order):
        y = [scores[c][cond] for c in clusters]
        plt.bar(x + (i - (len(cond_order)-1)/2)*width, y, width=width, label=cond)

    plt.xticks(x, clusters, rotation=90)
    plt.xlabel("Cluster")
    plt.ylabel("Dispersion score (lower = tighter)")
    plt.title("Cluster quality per condition")
    plt.legend()
    plt.tight_layout()
    plt.show()

#%%
def timepoint_pairwise_distances_within_condition(X_time_cond,
                                                  summary="mean"):
    """
    X_time_cond: (n_peptides, n_timepoints, 3) where last axis is conditions.
    Computes, for each condition separately, the pairwise distances between peptides
    at each timepoint t (no cross-condition mixing).

    Returns:
      dict cond -> dict with:
        D_t: (T, n, n) distance matrices per timepoint
        summary_t: (T,) mean/median upper-triangle distance per timepoint
    """
    X = np.asarray(X_time_cond)
    n, T, C = X.shape
    cond_names = ["EGF", "INS", "EGFnINS"]

    results = {}

    for c in range(C):
        # values at each timepoint: (n, T) for this condition
        M = X[:, :, c]  # (n, T)

        D_t = np.zeros((T, n, n), dtype=float)
        summary_t = np.full(T, np.nan, dtype=float)
        if n < 2:
            results[cond_names[c]] = {"D_t": D_t, "summary_t": summary_t}
            continue

        iu = np.triu_indices(n, k=1)

        for t in range(T):
            v = M[:, t][:, None]               # (n, 1)
            Dt = np.abs(v - v.T)               # (n, n) absolute distance for scalars
            # D_t[t] = Dt

            tri = Dt[iu]
            if summary == "mean":
                summary_t[t] = float(np.mean(tri))
            elif summary == "median":
                summary_t[t] = float(np.median(tri))
            else:
                raise ValueError("summary must be 'mean' or 'median'")

        results[cond_names[c]] =  summary_t #{"summary_t": summary_t} #"D_t": D_t,

    return results

def cluster_similarity_per_condition_per_timepoint(
        df,
        transpose = True,
        data_type = "log2_FC",
        cluster_column_name = str,
        mean = True,
        median = False,
        verbose = False
):
    '''value per condition and time point
    per condition there is one a list, one value per time points'''
    column_selection = [element for element in df.columns.tolist() if f"{data_type}" in element and "cluster" not in element]

    cluster_metric = {}

    for cluster in sorted(df[cluster_column_name].unique()):
        if cluster == 999:
            continue
        # print(cluster)
        df1 = df.loc[df[cluster_column_name] == cluster]
        X1_tp, y_qc = reshape_df(df=df1, time_series=column_selection, labels="site", dimensions=3, len_time_serie=7,
                                 transpose=transpose, verbose=verbose)
        # print(X1_tp.shape)

        dist_per_condition = timepoint_pairwise_distances_within_condition(X1_tp, summary="mean")
        cluster_metric[cluster] = dist_per_condition


    return cluster_metric

def combine_conditions(
    scores_per_cluster,
    how="mean",
    cond_order=("EGF", "INS", "EGFnINS")
):
    """
    use for the value per time point per condition
    Returns: dict cluster -> scalar
    """
    combined = {}

    for k, d in scores_per_cluster.items():
        vals = np.array([d[c] for c in cond_order], dtype=float)

        if how == "mean":
            combined[k] = float(np.nanmean(vals))
        elif how == "max":
            combined[k] = float(np.nanmax(vals))
        elif how == "median":
            combined[k] = float(np.nanmedian(vals))
        else:
            raise ValueError("how must be 'mean', 'median', or 'max'")

    return combined
#%%
def plot_cluster_timepoint_dispersion(
    cluster_QC_test,
    timepoints=None,
    cond_order=("EGF", "INS", "EGFnINS"),
    ncols=5,
    figsize_per_subplot=(3.2, 2.6),
    sharey=True,
    suptitle="Within-cluster dispersion per timepoint (per condition)"
):
    """
    Plot timepoint-by-timepoint within-condition dispersion for each cluster.

    Parameters
    ----------
    cluster_QC_test : dict
        cluster_id -> dict(condition -> array length T)
    timepoints : array-like or None
        If None, uses 0..T-1. Otherwise used as x-axis labels/values.
    cond_order : tuple
        Order of conditions to plot per subplot.
    ncols : int
        Number of subplot columns.
    figsize_per_subplot : tuple
        Size per subplot (width, height). Total figure size scales accordingly.
    sharey : bool
        Share y-axis across subplots for easier comparison.
    suptitle : str
        Figure title.
    """
    # Sort clusters for stable layout
    clusters = sorted(cluster_QC_test.keys())
    n_clusters = len(clusters)
    if n_clusters == 0:
        raise ValueError("cluster_QC_test is empty.")

    # Infer T from first cluster/condition
    first_cluster = clusters[0]
    first_cond = cond_order[0]
    T = len(cluster_QC_test[first_cluster][first_cond])

    # X-axis values
    if timepoints is None:
        x = np.arange(T)
        x_label = "Timepoint index"
    else:
        x = np.asarray(timepoints)
        if len(x) != T:
            raise ValueError(f"timepoints length ({len(x)}) must match T ({T}).")
        x_label = "Time"

    # Figure layout
    nrows = math.ceil(n_clusters / ncols)
    fig_w = figsize_per_subplot[0] * ncols
    fig_h = figsize_per_subplot[1] * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h), sharex=True, sharey=sharey)

    # axes can be 2D or 1D depending on nrows/ncols
    axes = np.atleast_2d(axes)

    # Plot each cluster
    for i, cluster_id in enumerate(clusters):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        for cond in cond_order:
            y = np.asarray(cluster_QC_test[cluster_id].get(cond, np.full(T, np.nan)))
            ax.plot(x, y, marker="o", linewidth=1.5, markersize=3, label=cond)

        ax.set_title(f"Cluster {cluster_id}", fontsize=10)
        ax.grid(True, linewidth=0.5, alpha=0.4)

        if r == nrows - 1:
            ax.set_xlabel(x_label)
        if c == 0:
            ax.set_ylabel("Dispersion\n(mean pairwise |Δ|)")

    # Turn off any unused subplots
    for j in range(n_clusters, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r, c].axis("off")

    # Single legend for whole figure (cleaner than repeating)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=True)

    fig.suptitle(suptitle, y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()

#%%
def clusters_plot_linear(
    df,
    legend=list,
    saving_path=str,
    cluster_column=str,
    cluster_name="",
    data_type=str,
    plot_different_data=False,
    saving_info="",
    save_pdf=False,
    save_png=False,
    plot_close=False,
    y_lims_list=False,
    grey_alpha=0.08,
    grey_lw=0.8,
    mean_lw=2.6):
    ''' Aplying mixing chatGPT code and my own'''

    #Load the dataframe provided, variable or path
    if type(df) == pd.DataFrame:
        pass
    else:
        df = pd.read_excel(df)

    # Prepare saving folder
    if save_pdf == False and save_png == False:
        pass
    else:
        if not os.path.exists(saving_path):
            print("Creating saving folder")
            os.makedirs(saving_path)

    # Cluster list
    clusters = list(set(df[cluster_column]))
    if 999 in clusters:
        clusters.remove(999)
    if data_type not in cluster_column and plot_different_data == False:
            print("Remember to plot the same data_type used to make the clustering or put: plot_different_data = TRUE")
    else:
        #Sort clusters
        if type(clusters[0]) == int:
            sorted_clusters = sorted(clusters)
        else:
            sorted_clusters = sorted(clusters, key=lambda x: int(x.split()[1]))

        # Geting some basic information and parameters for the plots
        n_cluster = len(sorted_clusters)

        # time points of the dataset
        column_names = df.columns.tolist()
        time_points_previous = [element for element in column_names if f"log2_FC_EGF_" in element] # this could be any set of columns that have the time points
        time_points = [s.split("_")[-1] for s in time_points_previous]

        #Getting the columns for each condition
        EGF_matching_cols = [col for col in df.columns if any(f"{data_type}_EGF_{t}" in col for t in time_points)]
        INS_matching_cols = [col for col in df.columns if any(f"{data_type}_INS_{t}" in col for t in time_points)]
        EGFnINS_matching_cols = [col for col in df.columns if any(f"{data_type}_EGFnINS_{t}" in col for t in time_points)]

        #Groupping the information of stimulation condition, columns with the data, color used for representation
        conditions = [("EGF", EGF_matching_cols, "red"),
                      ("INS", INS_matching_cols, "blue"),
                      ("EGFnINS", EGFnINS_matching_cols, "fuchsia")]
        # Layout: one row per cluster, 3 columns (EGF / INS / EGFnINS)
        fig, axes = plt.subplots(
            nrows=n_cluster,
            ncols=3,
            figsize=(18, max(3.2 * n_cluster, 6)),
            squeeze=False
        )
        # plt.subplots_adjust(top=0.92, wspace=0.25, hspace=0.6)

        for r, cluster in enumerate(sorted_clusters):
            sub_df = df.loc[df[cluster_column] == cluster].copy()
            if sub_df.shape[0] == 0: #Checking that there is at least one
                continue

            # If you want consistent y-lims across the 3 condition panels within the same cluster,
            # compute global min/max from all three condition matrices.
            all_vals = np.concatenate([
                sub_df[EGF_matching_cols].to_numpy().ravel(),
                sub_df[INS_matching_cols].to_numpy().ravel(),
                sub_df[EGFnINS_matching_cols].to_numpy().ravel()
            ])
            finite_vals = all_vals[np.isfinite(all_vals)]
            if finite_vals.size == 0:
                y_min, y_max = -1, 1
            else:
                y_min, y_max = float(np.min(finite_vals)), float(np.max(finite_vals))
                pad = 0.15 * (y_max - y_min + 1e-9)
                y_min, y_max = y_min - pad, y_max + pad

            if isinstance(y_lims_list, list) and len(y_lims_list) == 2:
                y_min, y_max = y_lims_list[0], y_lims_list[1]

            for c, (cond_name, cond_cols, color) in enumerate(conditions):
                ax = axes[r, c]

                # Plot all individual site trajectories in grey
                mat = sub_df[cond_cols].to_numpy(dtype=float)  # shape (n_sites, n_timepoints)
                for i in range(mat.shape[0]):
                    ax.plot(time_points, mat[i, :], color="grey", alpha=grey_alpha, linewidth=grey_lw)

                # Overlay mean trajectory (ignore NaNs)
                mean_curve = np.nanmean(mat, axis=0)
                ax.plot(time_points, mean_curve, color=color, linewidth=mean_lw)

                ax.axhline(0, color="grey", linestyle="--", linewidth=1)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(y_min, y_max)

                # Labels/titles
                if r == n_cluster - 1:
                    ax.set_xlabel("Time (min)" if time_points != list(range(len(time_points))) else "Time index")
                ax.set_ylabel(f"{data_type}" if c == 0 else "")

                ax.set_title(f"Cluster {cluster} | {cond_name} (n={len(sub_df)} sites)")

                # If time_points are numeric, use them as ticks; otherwise use extracted strings
                ax.set_xticks(time_points)
                # if time_points == list(range(len(time_points_str))):
                #     ax.set_xticklabels(time_points_str)
                # else:
                ax.set_xticklabels([str(t) for t in time_points])

        # Legend (simple, for mean curves)
        # If user passed legend, use it; otherwise use defaults
        if legend == list or legend is None or len(legend) == 0:
            legend = ["EGF mean", "INS mean", "EGF+INS mean"]
        # Add a single legend at figure level
        handles = [
            plt.Line2D([0], [0], color="red", lw=mean_lw),
            plt.Line2D([0], [0], color="blue", lw=mean_lw),
            plt.Line2D([0], [0], color="fuchsia", lw=mean_lw),
            plt.Line2D([0], [0], color="grey", lw=grey_lw, alpha=0.4)
        ]
        labels = [legend[0], legend[1], legend[2], "Individual sites"]
        fig.legend(handles, labels, loc="upper right", ncol=1)

        fig.suptitle(f"{cluster_column} {cluster_name} {date.today()}", weight="bold")

        # Saving
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
        # return fig, axes


#%%
def clusters_plot_overlay_conditions(
        df,
        legend=None,
        saving_path="",
        cluster_column="",
        cluster_name="",
        data_type="log2_FC",
        plot_different_data=False,
        saving_info="",
        save_pdf=False,
        save_png=False,
        plot_close=False,
        y_lims_list=False,
        individual_alpha=0.08,
        individual_lw=0.8,
        mean_lw=2.8
):
    """
    One subplot per cluster.
    In each subplot:
      - Plot all individual site time profiles for EGF/INS/EGFnINS in their color with low alpha
      - Overlay the mean profile per condition in the same color with alpha=1
    """

    # Load the dataframe provided, variable or path
    if type(df) == pd.DataFrame:
        pass
    else:
        df = pd.read_excel(df)

    # Prepare saving folder
    if save_pdf == False and save_png == False:
        pass
    else:
        if not os.path.exists(saving_path):
            print("Creating saving folder")
            os.makedirs(saving_path)

    # Cluster list
    clusters = list(set(df[cluster_column]))
    if 999 in clusters:
        clusters.remove(999)
    if data_type not in cluster_column and plot_different_data == False:
        print("Remember to plot the same data_type used to make the clustering or put: plot_different_data = TRUE")
    else:
        # Sort clusters
        if type(clusters[0]) == int:
            sorted_clusters = sorted(clusters)
        else:
            sorted_clusters = sorted(clusters, key=lambda x: int(x.split()[1]))

        n_cluster = len(sorted_clusters)
        sqrt_n_c = int(np.ceil(np.sqrt(n_cluster)))
        empty_plots = (sqrt_n_c * sqrt_n_c) - n_cluster

        # Avoid getting rows with empty plots
        if empty_plots >= sqrt_n_c:
            sqrt_n_c_X = sqrt_n_c - 1
        else:
            sqrt_n_c_X = sqrt_n_c

        # Subplots
        fig, ax = plt.subplots(sqrt_n_c, sqrt_n_c_X, figsize=(18, 13))
        fig.tight_layout(w_pad=1.75, h_pad=3)
        plt.subplots_adjust(top=0.92)

        # Flatten axes for simpler indexing (works even if 1 row/col)
        ax = np.array(ax).reshape(-1)

        # Infer timepoints from EGF columns
        column_names = df.columns.tolist()
        time_points_previous = [element for element in column_names if
                                f"log2_FC_EGF_" in element]  # this could be any set of columns that have the time points
        time_points = [s.split("_")[-1] for s in time_points_previous]

        # Getting the columns for each condition
        EGF_matching_cols = [col for col in df.columns if any(f"{data_type}_EGF_{t}" in col for t in time_points)]
        INS_matching_cols = [col for col in df.columns if any(f"{data_type}_INS_{t}" in col for t in time_points)]
        EGFnINS_matching_cols = [col for col in df.columns if
                                 any(f"{data_type}_EGFnINS_{t}" in col for t in time_points)]

        # Groupping the information of stimulation condition, columns with the data, color used for representation
        conditions = [("EGF", EGF_matching_cols, "red"),
                      ("INS", INS_matching_cols, "blue"),
                      ("EGFnINS", EGFnINS_matching_cols, "fuchsia")]

        # Default legend
        if legend is None:
            legend = ["EGF", "INS", "EGF+INS"]

        for idx, cluster in enumerate(sorted_clusters):
            sub_df = df.loc[df[cluster_column] == cluster].copy()
            if sub_df.shape[0] == 0:  # Check there is no empty subdataframe
                continue

            a = ax[idx]

            # Compute y-limits consistently per cluster (across all conditions)
            all_vals = np.concatenate(
                [sub_df[c].to_numpy(dtype=float).ravel() for _, cols, _ in conditions for c in cols])
            finite_vals = all_vals[np.isfinite(all_vals)]
            if finite_vals.size == 0:
                y_min, y_max = -1, 1
            else:
                y_min, y_max = float(np.min(finite_vals)), float(np.max(finite_vals))
                pad = 0.15 * (y_max - y_min + 1e-9)
                y_min, y_max = y_min - pad, y_max + pad

            if isinstance(y_lims_list, list) and len(y_lims_list) == 2:
                y_min, y_max = y_lims_list[0], y_lims_list[1]

            # Plot each condition: all lines (low alpha) + mean (alpha 1)
            for cond_name, cond_cols, color in conditions:
                mat = sub_df[cond_cols].to_numpy(dtype=float)  # (n_sites, n_timepoints)

                # Individual profiles
                for i in range(mat.shape[0]):
                    a.plot(time_points, mat[i, :], color=color, alpha=individual_alpha, linewidth=individual_lw)

                # Mean profile
                mean_curve = np.nanmean(mat, axis=0)
                a.plot(time_points, mean_curve, color=color, alpha=1.0, linewidth=mean_lw)

            a.axhline(0, color="grey", linestyle="--", linewidth=1)
            a.grid(True, alpha=0.3)
            a.set_ylim(y_min, y_max)
            a.set_title(f"Cluster {cluster} (n={len(sub_df)} sites)")
            a.set_xlabel("Time (min)" if time_points != list(range(len(time_points))) else "Time index")
            a.set_ylabel(f"{data_type}")

            a.set_xticks(time_points)
            if time_points == list(range(len(time_points))):
                a.set_xticklabels(time_points)
            else:
                a.set_xticklabels([str(t) for t in time_points])

        # Hide unused axes
        for j in range(len(sorted_clusters), len(ax)):
            ax[j].axis("off")

        # Figure legend for mean lines (use thicker handles)
        handles = [
            plt.Line2D([0], [0], color="red", lw=mean_lw),
            plt.Line2D([0], [0], color="blue", lw=mean_lw),
            plt.Line2D([0], [0], color="fuchsia", lw=mean_lw),
        ]
        fig.legend(handles, legend[:3], loc="upper right", ncol=1)

        fig.suptitle(f"{cluster_column} {cluster_name} {date.today()}", weight="bold")

        # Save
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

        # return fig, ax