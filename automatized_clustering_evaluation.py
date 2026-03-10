import argparse

import matplotlib.pyplot as plt

from utils import *

parser = argparse.ArgumentParser(description="Quality control check for clusters")

parser.add_argument("input_file_path", type=str, help="Path to input TSV file")
parser.add_argument("output_file_path", type=str, help="Path to output png file")
#arser.add_argument("scaling_factor", type=float, help="Scaling factor")
args = parser.parse_args()

# path = "Experiment/2_hTERT_HME1/Data/Processed/Full_dataset_2_hTERT_HME1_functional_names_diff_scaled_phPlus.tsv"

#%% Initial import of the dataframe and filtering
df_2_hME1_scaled = pd.read_csv(args.input_file_path, sep="\t")
df_2_hME1_scaled = df_2_hME1_scaled.fillna(0)
print(f"The Dataframe shape: {df_2_hME1_scaled.shape}")
#Filter the dataframe
nreps_df = filter_replicates(df_2_hME1_scaled, n_reps=2)
localized_sites_df = filter_site_localizations(nreps_df, loc_sites=True)
df_filtered = filter_dynamics_extremes(df = df_2_hME1_scaled,
                                       data_type="log2_FC",
                                       threshold=0.5,
                                       exclude_full=True)
print(f"Original Dataframe shape: {df_2_hME1_scaled.shape}"
      f"\nnreps_df: {nreps_df.shape}"
      f"\nlocalized_sites_df: {localized_sites_df.shape}"
      f"\ndf_filtered: {df_filtered.shape}")

#%% Clustering evaluation
#Evaluate k in a range (e.g., 5 → 40).

#% Inertia
from tslearn.metrics import cdist_dtw
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score


df_clustered = df_filtered.copy()

ks = range(2,100)
seeds = range(10)

inertias =[]
silhouettes =[]
stabilities =[]

for k in ks:

    labels_list= []
    for seed in seeds:
        cluster_name = f"KMeans_{k}_cluster_seed{seed}_on_log2FC_noFull_nrep>1_locSite=True_log2FC>0.5"
        print(f"Clustering: {cluster_name}")

        df_clustered, model, multivariate = tslearn_clustering_KMeans(df_to_cluster= df_clustered,
                                                     data_type= "log2_FC",
                                                     condition_for_clustering=["_EGF_", "_INS_","_EGFnINS_",],
                                                     exclude_full=True,
                                                     cluster_column_name=cluster_name,
                                                     number_of_clusters=k,
                                                     max_iterations=200,
                                                     n_init = 5,
                                                     metric="euclidean",
                                                     df_dimensions=3,
                                                     time_series_length=6,
                                                     random_state = seed,
                                                     transpose = True,
                                                     verbose=True,
                                                     testing=True)

        if seed == 0:
            inertias.append(model.inertia_)

            D = cdist_dtw(multivariate)
            sil = silhouette_score(D, model.labels_, metric="euclidean")
            silhouettes.append(sil)

        labels_list.append(model.labels_)

    # pairwise ARI between runs
    ari_scores = []
    for i in range(len(labels_list)):
        for j in range(i+1, len(labels_list)):
            ari_scores.append(
                adjusted_rand_score(labels_list[i], labels_list[j]) # function from sklearn
            )
    stabilities.append(np.mean(ari_scores))

#inertias
plt.scatter(x=ks, y=inertias)
plt.title("Inertia vs Clusters")
plt.xlabel("Inertia")
plt.ylabel("Clusters")
plt.savefig(f"{args.output_file_path}_inertias.png")
plt.close()

#silhouetes
plt.scatter(x=ks, y=silhouettes)
plt.title("Silhouettes vs Clusters")
plt.xlabel("Silhouettes")
plt.ylabel("Clusters")
plt.savefig(f"{args.output_file_path}_silhouettes.png")
plt.close()

#stabilities
plt.scatter(x=ks, y=stabilities)
plt.title("Stabilities vs Clusters")
plt.xlabel("Stabilities")
plt.ylabel("Clusters")
plt.savefig(f"{args.output_file_path}_stabilities.png")
plt.close()


