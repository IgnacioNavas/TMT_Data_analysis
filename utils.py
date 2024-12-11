import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import warnings
# THIS FUNCTIONS ARE NOT BEING REDEFINED WHEN IMPORTING IN THE MAIN PLOTTING SCREEN , THIS IS WHY I PUT THEM BACK IN THE PLOTTING JUPYTER NOTEBOOK
# Plotting functions

# %%
def plot_protein_phosphosites_together(df, proteins=list, legend=list, path=str, saving_path=str,
                                       saving_info="", save_pdf=False, save_png=False, plot_close=False,
                                       fit_y_lims=False):
    '''Plot to PDF ALL phosphosites of a protein together in one single plot. "y" limits are ajusted to the limits of each phosphosites'''

    for protein in proteins:

        # Create sub-dataframe with only the protein we are interested in. If the protein doesn't exist in the dataframe skip code
        if protein in df['prot_name'].to_list():
            sub_df = df.loc[df['prot_name'] == protein].copy()
            print(f"Ploting sites of protein {protein}")
        elif protein in df['protein_ID'].to_list():
            sub_df = df.loc[df['protein_ID'] == protein].copy()
            print(f"Ploting sites of protein {protein}")
        else:
            print(f"The protein {protein} is not present in the dataset")
            continue
        # Check if a folder for the desired protein exists. If no, create one
        to_save = re.sub("Data.*", saving_path, path)
        if protein in os.listdir(to_save):
            pass
        else:
            new_path = f"{to_save}/{protein}"
            os.makedirs(new_path)

        sub_df.sort_values(by=['site'], inplace=True)

        # Geting some basic information and parameters for the plots
        number_phos = len(sub_df)
        sqrt_n_p = int(np.ceil(np.sqrt(number_phos)))
        empty_plots = (sqrt_n_p * sqrt_n_p) - number_phos

        # If the protein only has one phosphosite the code cannot make a plot with one subplot, so I'll have to make it individually
        single_phsite_proteins = []
        if number_phos == 1:
            single_phsite_proteins.append(protein)
            print(f"Protein {protein} has only one phosphosite. Plot it individually")
            continue

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
            sub_values_df = sub_df.iloc[:, 3:]
            y_lim_max = (sub_values_df.max().max()) * 1.1
            y_lim_min = (sub_values_df.min().min()) * 1.1
            y_fixed = "_y_axis_general"

        k = 0  # Counter to stop ploting when there is no more phosphosites

        # Seting subplots aprameters
        fig, ax = plt.subplots(sqrt_n_p, sqrt_n_p_X,
                               figsize=(18, 13))  # figsize=(7, 4) figsize=(18, 13) figsize=(29.7, 21)
        fig.tight_layout(w_pad=1.75, h_pad=3)
        plt.subplots_adjust(top=0.94)  # percentage of the figure that the plots are using

        # Go through the sub_df to plot all the phosphosites
        for i in range(sqrt_n_p):  # y
            for j in range(sqrt_n_p_X):  # X
                if k == number_phos:  # Stop plotting, all phosphorylation sites have been plotted
                    continue
                else:
                    # IN THIS CASE I AM ACCESSING THE ROW BY INDEXING,
                    # that's why the "id" is alreay at possition 0 and all the indexing is shifted compared to the datafram by -1
                    row = sub_df.iloc[k,
                          :]  # Go through the rows of the subdataset with the phsophosites for the protein
                    # Collect identification data of the phosphorylatio site
                    site = row[2]
                    name = row[1]
                    id = row[0]

                    # Collect data of the time points of the phosphosites
                    all_times = row[3:24]
                    EGF = row[3:10]
                    INS = row[10:17]
                    EGFnINS = row[17:24]

                    groups = [EGF, INS, EGFnINS]
                    colors = ['r', 'b', 'fuchsia']

                    # Collect data of the standard deviation of each timepoint
                    EGF_sd = row[26:33]
                    INS_sd = row[33:40]
                    EGFnINS_sd = row[40:]
                    groups_sd = [EGF_sd, INS_sd, EGFnINS_sd]
                    # Collect data about number of replicates in which the phosphosite was detected
                    n_rep = row[24]

                    # Start plotting
                    for c in range(3):
                        if n_rep == 1:
                            al = 0.3
                        else:
                            al = 1
                        ax[i, j].errorbar(x=['Full', '0', '1', '2', '5', '10', '90'], y=groups[c], yerr=groups_sd[c],
                                          marker='o', color=colors[c], alpha=al, capsize=4, elinewidth=1.3)

                        # Subplot parameters
                    ax[i, j].set_title(
                        f"{str(re.findall(r'^.*~', site))[2:-3]}_n_reps{n_rep}")  # , weight='bold'  INCLUDE THE POSIBILITY OF ADDING THE PHOSPHOSITE SEQUENCE
                    ax[i, j].set_xlabel("Time (min)")
                    ax[i, j].set_ylabel("Log2FC")
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
        fig.suptitle(f"{name}_{id}", weight='bold')

        # Saving the plot
        if save_pdf == True:
            plt.savefig(f"{to_save}/{protein}/{name}_{id}_0_All{y_fixed}{saving_info}.pdf")
            print(f"{protein} Plot saved as PDF")
        else:
            print(f"{protein} Plot not saved")
        if save_png == True:
            plt.savefig(f"{to_save}/{protein}/{name}_{id}_0_All{y_fixed}{saving_info}.png")
            print(f"{protein} Plot saved as PNG")
        if plot_close == True:
            plt.close()

# %%
def plot_protein_phosphosites_individually(df, proteins=list, legend=list, path=str, saving_path=str,
                                           saving_info="", save_pdf=False, save_png=False, plot_close=False,
                                           fit_y_lims=True):
    '''Create a PDF for each phosphorylation sites of a protein'''

    for protein in proteins:

        # Create sub-dataframe with only the protein we are interested in. If the protein doesn't exist in the dataframe skip code
        if protein in df['prot_name'].to_list():
            sub_df = df.loc[df['prot_name'] == protein].copy()
            print(f"Ploting sites of protein {protein}")
        elif protein in df['protein_ID'].to_list():
            sub_df = df.loc[df['protein_ID'] == protein].copy()
            print(f"Ploting sites of protein {protein}")
        else:
            print(f"The protein {protein} is not present in the dataset")
            continue
        # Check if a folder for the desired protein exists. If no, create one
        to_save = re.sub("Data.*", saving_path, path)
        if protein in os.listdir(to_save):
            pass
        else:
            new_path = f"{to_save}/{protein}"
            os.makedirs(new_path)

        for row in sub_df.itertuples():
            # HERE THE ROW IS TAKEN OUT OF THE SUBDATAFRAME.
            # This is why position "0" is not usefull, is just the index in the dataframe
            site = row[3]
            name = row[2]
            id = row[1]

            all_times = row[4:25]
            EGF = row[4:11]
            INS = row[11:18]
            EGFnINS = row[18:25]
            groups = [EGF, INS, EGFnINS]

            # Collect data of the standard deviation of each timepoint
            EGF_sd = row[27:34]
            INS_sd = row[34:41]
            EGFnINS_sd = row[41:]
            groups_sd = [EGF_sd, INS_sd, EGFnINS_sd]

            # Collect data about number of replicates in which the phosphosite was detected
            n_rep = row[25]

            colors = ['r', 'b', 'fuchsia']

            fig, ax = plt.subplots(figsize=(7, 4))

            for c in range(3):
                if n_rep == 1:
                    al = 0.3
                else:
                    al = 1
                ax.errorbar(x=['Full', '0', '1', '2', '5', '10', '90'], y=groups[c], yerr=groups_sd[c], marker='o',
                            color=colors[c], label=legend[c], capsize=4, elinewidth=1.3, alpha=al)

            ax.set_title(f"{name}_{site}_n_rep{n_rep}")
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Log2FC")
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
                plt.savefig(f"{to_save}/{protein}/{name}_{site}{y_lim}{saving_info}.pdf")
                print(f"{site} Plot saved as PDF")
            else:
                print(f"{site} Plot not saved")
            if save_png == True:
                plt.savefig(f"{to_save}/{protein}/{name}_{site}{y_lim}{saving_info}.png")
                print(f"{site} Plot saved as PNG")
            if plot_close == True:
                plt.close()

# %%
def plot_dataset_phosphosites_together(df, proteins=list, legend=list, path=str, saving_path=str, dataset_folder=str,
                                       dataset_name=str,
                                       saving_info="", save_pdf=False, save_png=False, plot_close=False,
                                       fit_y_lims=True):
    '''Plot all the phosphorilation sites of a dataset'''

    # Check if a folder for the desired protein exists. If no, create one
    to_save = re.sub("Data.*", saving_path, path)
    if dataset_name in os.listdir(to_save):
        pass
    else:
        new_path = f"{to_save}/{dataset_name}"
        os.makedirs(new_path)

    # Geting some basic information and parameters for the plots
    number_phos = len(df)
    sqrt_n_p = int(np.ceil(np.sqrt(number_phos)))
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
        y_lim_max = fit_y_lims[1]
        y_lim_min = fit_y_lims[0]
        y_fixed = f"_y_axis_fixed_{y_lim_min}_{y_lim_max}"
    else:  # Fit the same "y" limit for all phosphosites
        sub_values_df = df.iloc[:, 3:47]
        y_lim_max = (sub_values_df.max().max()) * 1.1
        y_lim_min = (sub_values_df.min().min()) * 1.1
        y_fixed = "_Y_fixed_general"

    k = 0  # Counter to stop ploting when there is no more phosphosites
    # Seting subplots parameters
    fig, ax = plt.subplots(sqrt_n_p, sqrt_n_p_X, figsize=(18, 13))  # figsize=(7, 4) figsize=(18, 13) figsize=(29.7, 21)
    fig.tight_layout(w_pad=1.75, h_pad=3)
    plt.subplots_adjust(top=0.94)  # percentage of the figure that the plots are using

    # Go through the sub_df to plot all the phosphosites
    for i in range(sqrt_n_p):  # y
        for j in range(sqrt_n_p_X):  # X
            if k == number_phos:  # Stop plotting, all phosphorylation sites have been plotted
                continue
            else:
                row = df.iloc[k, :]  # Go through the rows of the subdataset with the phsophosites for the protein
                # Collect identification data of the phosphorylatio site
                site = row[2]
                name = row[1]
                id = row[0]
                # Collect data of the time points of the phosphosites
                all_times = row[3:24]
                EGF = row[3:10]
                INS = row[10:17]
                EGFnINS = row[17:24]

                groups = [EGF, INS, EGFnINS]
                colors = ['r', 'b', 'fuchsia']

                # Collect data of the standard deviation of each timepoint

                EGF_sd = row[26:33]
                INS_sd = row[33:40]
                EGFnINS_sd = row[40:47]
                groups_sd = [EGF_sd, INS_sd, EGFnINS_sd]
                # print(groups, groups_sd)
                # Collect data about number of replicates in which the phosphosite was detected
                n_rep = row[24]

                # Start plotting
                for c in range(3):
                    if n_rep == 1:
                        al = 0.3
                    else:
                        al = 1
                    ax[i, j].errorbar(x=['Full', '0', '1', '2', '5', '10', '90'], y=groups[c], yerr=groups_sd[c],
                                      marker='o', color=colors[c], alpha=al, capsize=4, elinewidth=1.3)

                # Subplot parameters
                ax[i, j].set_title(
                    f"{str(re.findall(r'^.*~', site))[2:-3]}_n_reps{n_rep}")  # , weight='bold'  INCLUDE THE POSIBILITY OF ADDING THE PHOSPHOSITE SEQUENCE
                ax[i, j].set_xlabel("Time (min)")
                ax[i, j].set_ylabel("Log2FC")
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
    fig.suptitle(f"{dataset_name}", weight='bold')

    # Saving the plot
    if save_pdf == True:
        plt.savefig(f"{to_save}/{dataset_folder}/{dataset_name}_All{y_fixed}{saving_info}.pdf")
        print(f"{dataset_name} Plot saved as PDF")
    else:
        print(f"{dataset_name} Plot not saved")
    if save_png == True:
        plt.savefig(f"{to_save}/{dataset_folder}/{dataset_name}_All{y_fixed}{saving_info}.png")
        print(f"{dataset_name} Plot saved as PNG")
    if plot_close == True:
        plt.close()


# %%
def test(hola):
    print(hola)
    print(len(hola), hola)