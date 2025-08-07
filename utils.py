import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from datetime import date
from IPython.display import display, HTML
import warnings


#%% Optimized
def uniprot_links_for(df, protein_list=[]):
    for protein in protein_list:
        # Check if protein is neither in name nor ID
        if (protein not in df['protein_name'].values) and (protein not in df['protein_Id'].values):
            uniprot_url = f"https://www.uniprot.org/uniprotkb/{protein}"
            html_link = f'Protein {protein} is not in the database: <a href="{uniprot_url}" target="_blank">{protein}</a>'
            display(HTML(html_link))

        # If it's a name, look up ID
        elif protein in df['protein_name'].values:
            protein_ID = df.loc[df['protein_name'] == protein, 'protein_Id'].values[0]
            uniprot_url = f"https://www.uniprot.org/uniprotkb/{protein_ID}"
            html_link = f'Link to protein {protein} <a href="{uniprot_url}" target="_blank">{protein_ID}</a>'
            display(HTML(html_link))

        # If it's an ID, look up name
        else:
            protein_name = df.loc[df['protein_Id'] == protein, 'protein_name'].values[0]
            uniprot_url = f"https://www.uniprot.org/uniprotkb/{protein}"
            html_link = f'Link to protein {protein_name} <a href="{uniprot_url}" target="_blank">{protein}</a>'
            display(HTML(html_link))

#%% Optimized for all experiment
def plot_dataset_phosphosites(df, cluster_column = "", cluster_number = int,
                              data_type = str,
                              legend = list,  color_palette = ['r', 'b', 'fuchsia'],
                              saving_path=str, dataset_name=str, saving_info="",
                              plot_individually=False, fit_y_lims=False, plot_close=False,
                              save_pdf=False, save_png=False):
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
def clusters_plot(df, legend=list,
                  saving_path=str, cluster_column = str, cluster_name="", data_type = str, plot_different_data = False,
                  saving_info="",
                  save_pdf=False, save_png=False, plot_close=False, y_lims_list=False):
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
    if data_type not in cluster_column and plot_different_data == False:
            print("Remember to plot the same data_type used to make the clustering")
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
def reshape_df(df, time_series, dimensions, len_time_serie):
    '''Reshape dataframe so it is multivariate format. Return the dataframe in numpy format so can be used, and list with the names of myseries'''

    sub_df = df[time_series].copy()
    mySeries = sub_df.to_numpy()
    namesofMySeries = df["site"]

    multivariate_shape = (len(df), dimensions, len_time_serie)
    multivariate_df = np.reshape(mySeries, multivariate_shape)

    return multivariate_df, namesofMySeries

#%% Optimized for all experiment
def plot_protein_phosphosites(df, data_type = str,
                              proteins=list, replicates = False,  exclude_rep = list,
                              legend_plot = list, color_palette = ['r', 'b', 'fuchsia'],
                              saving_path=str, saving_info="", title_info = "",
                              plot_individually=False, fit_y_lims=False,
                              plot_close=False,
                              save_pdf=False, save_png=False):
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
            column_selection = [element for element in column_names if data_type in element]

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
def plot_data(ax, row_df, replicates = False, data_type = str, colors = [], legend = [], exclude_rep = list, plot_individually = False,):

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
        exit(1)

#%% Optimized for all experiments
def plot_protein_profile(df, proteins, data_type = str,
                         saving_path="", saving_info="",
                         legend=False, save_pdf=False, save_png=False):

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
def plot_protein_profiles_fine_line(df, proteins=list, data_type=str,
                                    saving_path=str, legend=False,
                                    saving_info = str,
                                    save_pdf=False, save_png=False):
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
