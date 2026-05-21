import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import date
from IPython.display import display, HTML

import networkx as nx
import plotly.express as px
import plotly.graph_objects as go


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


# =============================================================================
# Filtering
# =============================================================================

def dynamics_values(df, data_type="log2_FC", exclude_full=True):
    """Return only the time-series value columns for the given data_type."""
    if exclude_full:
        column_selection = [c for c in df.columns if c.startswith(data_type) and "full" not in c]
    else:
        column_selection = [c for c in df.columns if data_type in c and "statistics" not in c]
    return df[column_selection].copy()

def filter_ERK_motif(df, ERK_motif=False):
    if ERK_motif == False:
        return df.loc[df["ERK_motif"] == 0]
    else:
        return df.loc[df["ERK_motif"] == 1]


def filter_functional_score(df, f_score):
    """Return rows whose functional_score is >= f_score."""
    return df.loc[df["functional_score"] >= f_score]


# =============================================================================
# Plotting — original naming convention (single cell line datasets)
# moved to src/plotting_functions.py: plot_data, plot_dataset_phosphosites,
#   plot_protein_phosphosites, plot_protein_profile
# =============================================================================

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

