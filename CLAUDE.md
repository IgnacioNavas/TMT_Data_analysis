# Project: The MAPK/ERK signaling pathway crosstalk and regulation at phosphoproteome level

## Project overview

This project aims to understand how cells use phosphorylation of Serine, Threonine and Tyrosine
residues to transduce signal information and generate an adequate cellular response. The main focus
is on the regulation of the MAPK/ERK signaling pathway. The secondary aim of the project is to have a better 
understanding on how does crosstalk between EGF and insulin signaling happens at systems level. For this I have 
information about cells being stimulated with EGF, insulin and the combination of both. The cell lines I have been 
working with are hTERT-HME1 and HEK293T cell lines.

Cells grow in full media with growth factors and supplements (full). In this media, signaling reaches
an equilibrium state in which cells can proliferate with basal levels of signaling. To synchronize
cells and reduce basal signaling, full media is replaced by media without growth factors or
supplements for 2 hours (starve). After starvation, cells are stimulated with EGF, insulin (INS),
or a combination of both (EGFnINS), and lysates are collected at multiple time points. Two controls
are also collected: cells in full media, and cells that were starved but not stimulated.

These lysates are processed for phosphoproteomics using two protocols:
- **TMT-LC-MS/MS** — Tandem mass tag liquid chromatography mass spectrometry. Here data dependent aquisition (dda) 
method was used.
- **LFQ-LC-MS/MS** — label free quantification liquid chromatography mass spectrometry. Here for a small test dataset I 
used "dda" as acquisition method. Later, fto generate the complete dataset I used the "dda" test data to optimize a 
diaPASEF method that would define the windowa to improve the phosphosites identifications.

The raw data has been analyzed using FragPipe v24.0 software

In addition to the experiments carried out with wild type cell lines, regulatory S/T/Y residues in the MAPK/ERK pathway 
were mutated to Alanine (which cannot be phosphorylated) to disrupt negative feedback regulation. These mutant
cell lines undergo the same starvation and stimulation protocol, but in this case I only stimulated with EGF. 
For mutant cell lines I also have full and starve controls.

## Datasets available

Own datasets:
- hme1_1: 
  - Path: `Experiment/hme1_1/` (Data/ = FragPipe output, Results/); dev sample: `data/hme1_1_transformed.tsv`
  - Cell lines: hTERT-HME1 wild type
  - Controls: Full and starve 
  - Stimulation conditions: 4.7 nM EGF, 10 nM insulin, co-stimulations with EGF and insulin
  - Time points: 1, 2, 5, 10, 90 minutes
  - Mass spectrometry protocol: TMT-LC-MS/MS
  - Acquisition method: data dependent acquisition
  - Number of replicates: 4
- hme1_2: **main working dataset** — clustering, kinase prediction and all downstream analyses use it
  - Path: `Experiment/hme1_2/`; clustered + kinase-annotated (8723 sites):
    `notebooks/03_clustering/Data_clustered/20260715_hTERT_HEM1_2_processed_phPlus_clustered_kinasepred.tsv`
  - Cell lines: hTERT-HME1 wild type
  - Controls: Full and starve 
  - Stimulation conditions: 1.57 nM EGF, 100 nM insulin, co-stimulations with EGF and insulin
  - Time points: 2, 5, 10, 15, 90 minutes
  - Mass spectrometry protocol: TMT-LC-MS/MS
  - Acquisition method: data dependent acquisition
  - Number of replicates: 4
- hek_1: 
  - Path: `Experiment/hek_1/`; dev sample: `data/hek_1_transformed.tsv`
  - Cell lines: HEK293T wild type
  - Controls: Full and starve 
  - Stimulation conditions: 1.57 nM EGF, 100 nM insulin, co-stimulations with EGF and insulin
  - Time points: 2, 5, 10, 15, 90 minutes
  - Mass spectrometry protocol: TMT-LC-MS/MS
  - Acquisition method: data dependent acquisition
  - Number of replicates: 4
- hme1_mutants_test:
  - Path: `Experiment/hme1_mutants_test/`; dev samples: `data/hTERT_HME1_mutants_test_sample_transformed.tsv`,
    `data/LFQ_HME1_mutants_sample_dataset.tsv`
  - Cell lines: hTERT-HME1 wild type, hTERT-HME1 BRAF-S151A mutant, hTERT-HME1 GAB1-Y259A mutant
  - Controls: starve 
  - Stimulation conditions: 0.157 nM EGF
  - Time points: 2, 10, 25 minutes
  - Mass spectrometry protocol: LFQ-LC-MS/MS
  - Acquisition method: data dependent acquisition
  - Number of replicates: 2
- hme1_lfq: data acquisition on going
  - Path: not on disk yet; preprocessing stub at `notebooks/01_preprocessing/LFQ_diaPASEF.ipynb`
  - Cell lines: 
    - 1 - hTERT-HME1 wild type, 
    - 2 - hTERT-HME1 EGFR-T693A mutant, 
    - 3 - hTERT-HME1 BRAF-S151A mutant (biological replicate 1),
    - 4 - hTERT-HME1 SOS1-S1178A mutant, 
    - 5 - hTERT-HME1 SHOC2-T71A mutant, 
    - 6 - hTERT-HME1 BRAF-S151A mutant (biological replicate 2),
    - 7 - hTERT-HME1 GAB1-Y259A mutant, 
    - 8 - hTERT-HME1 RPS6KA3-S375A mutant,
  - Controls: full and starve 
  - Stimulation conditions: 0.157 nM EGF
  - Time points: 2, 5, 10, 15, 20, 30, 90 minutes
  - Mass spectrometry protocol: LFQ-LC-MS/MS
  - Acquisition method: data independent acquisition diaPASEF, with optimised windows with dda data from hme1_mutants   
  - Number of replicates: 3

The software for identification and quantification is FragPipe

**Where the data lives.** Full FragPipe outputs and processed files stay under `Experiment/{dataset}/`.
Everything in `data/` is a **truncated development sample** (100–1000 rows), not the full dataset — never
draw biological conclusions from it. The full clustered/annotated hme1_2 table is in
`notebooks/03_clustering/Data_clustered/`.

External datasets:
- MCF10A EGF time course, TMT-LC-MS/MS, timepoints: 2, 4, 8, 12 min. from Feng Song et al. is available in `External_Data/` 


## Repository structure

```
src/                        Python modules
  column_spec.py            ColumnSpec class — the standard way to select data columns
  transformations.py        Data transformation pipeline (run_all_transformations)
  filters.py                Row filtering: n:reps, contaminants (CON/REV), dynamics range, incomplete
                            time series, localization, by protein / by site
  QC.py                     Quality control functions (sections 1–4, see notebooks/02_qc/)
  plotting_functions.py     Visualisation: time series, cluster, protein profile plots
  clustering.py             Clustering utilities (tslearn KMeans/KShape/KernelKMeans, HDBSCAN, quality metrics)
  adaptive_clustering.py    Adaptive divisive+agglomerative KMeans (inertia-driven split then centroid merge)
  hierarchical_clustering.py Agglomerative hierarchical clustering (scipy linkage): tree-shaped QC
                            (cophenetic correlation, merge-distance elbow, free k-scan) + constrained
                            merging of clusters that share a parental node
  lfq_pretreatment.py       LFQ-specific preprocessing (mutant datasets)
  kinase_prediction.py      Kinase imputation via kinase_library: UniProt ±7 windows, top-5 kinase + percentile prediction (TMT & LFQ)
  cluster_composition.py    Cross-tabulate cluster labels vs annotations (which clusters contain sites of kinase X / protein Y)
  cluster_enrichment.py     Fisher's-exact per-cluster enrichment of kinases / motifs / metadata (odds ratio, log2 enrichment, BH-FDR)
  kinase_activity.py        KSEA kinase-activity inference (z-scores) across conditions × timepoints
  xgboost_model.py          ON HOLD — XGBoost cluster classifier + SHAP (see "XGBoost classifier" below)
  utils.py                  Shared utilities + older plotting/network helpers (partly superseded by
                            plotting_functions.py; kept for comparison)

notebooks/
  01_preprocessing/         Raw data processing and transformation
    TMT_dataset_preprocessing.ipynb   Transform WT TMT datasets; merge PhosphoSitePlus
    LFQ_dataset_preprocessing.ipynb   Preprocess LFQ mutant datasets; extract site metadata; filter
    LFQ_diaPASEF.ipynb                STUB — hme1_lfq (diaPASEF) preprocessing; currently only the
                                      sample-number → column-name mapping (8 cell lines × 9 timepoints × 3 reps
                                      + mix/mixb/mixc). Waiting on data acquisition.
    REPORT_01_preprocessing.md        Folder report: what preprocessing does and why
    tps_file_creator.ipynb            LEGACY — TPS input formatter (old column naming, do not use)
  02_qc/                    Quality control
    General_QC.ipynb        Markdown reference — QC checklist organized by stage (not executable)
    MSMS_data_QC.ipynb      Executable QC: missing values, CV, PCA, UMAP, dataset overlap
    QC_notes.md             Interpretation notes (e.g. PCA vs UMAP for replicate agreement)
    REPORT_02_qc.md         Folder report: QC results per dataset
    Results/                Exported QC figures, one dir per run: {YYYYMMDD}_QC_{dataset}_files/
  03_clustering/            Unsupervised clustering
    Clustering.ipynb                    WT k-scan and final clustering (TimeSeriesKMeans primary method)
    Adaptive_clustering.ipynb           Divisive+agglomerative adaptive clustering (src/adaptive_clustering.py)
    Adaptive_clustering_sweep.ipynb     2D threshold sweep + substructure diagnostics (margin, bootstrap stability)
    Hierarchical_clustering.ipynb       Hierarchical clustering (src/hierarchical_clustering.py): linkage-method
                                        comparison, k-scan on one tree, then constrained merging of clusters
                                        sharing a parental node. Same filtering/columns as Clustering.ipynb,
                                        so the two are ARI-comparable (last section does this)
    Clustering_mutant_cell_lines.ipynb  Assign mutant-dataset sites using the WT clustering; per-cluster
                                        PCA / plots / protein lookup / kinase imputation
    Autoencoder_clustering.ipynb        EXPLORATORY — 1D CNN autoencoder embedding + KMeans / DBSCAN / GMM
    clustering_overview.md              Reference document for all clustering strategies and parameters
    clustering_method_decision.md       ⭐ KMeans vs hierarchical decision document (2026-08-03). Objective
                                        functions, geometry, stability, mutant transfer; transformations, QC and
                                        statistics; sigmoid/T50 and the Chechik-Koller impulse model. Contains
                                        the four verified preprocessing defects (see "Known issues") and the
                                        measured null cluster-switching rate. READ BEFORE changing clustering
    council_note_1_clustering_mathematics.md        Working notes behind the decision document — the four
    council_note_2_qc_statistics_transformations.md independent analyses it synthesises, kept for the
    council_note_3_temporal_curve_modelling.md      derivations and intermediate tables the synthesis
    council_note_4_adversarial_review.md            compresses. Read clustering_method_decision.md first;
                                                    these are single-source and mostly not re-verified
    Data_clustered/                     Datasets with cluster-label columns appended (see "Cluster-label columns")
    Results/                            Exported figures, one dir per run: {YYYYMMDD}_{run_name}_files/
  04_visualization/         Time series and profile plots
    Plotting_time_series.ipynb   Main plotting notebook (src/plotting_functions.py)
    profiles_difference.ipynb    Euclidean distance between temporal profiles; simplified profile plots
  05_downstream/            Downstream analysis (enrichment, kinase activity, classifier)
    kinase_prediction.ipynb                      Predict top-5 kinases + percentiles per site (uses src/kinase_prediction.py); adds cluster-composition demo (src/cluster_composition.py)
    cluster_enrichment_and_kinase_activity.ipynb Fisher enrichment + KSEA, heavily documented for defense (src/cluster_enrichment.py, src/kinase_activity.py)
    XGBoost_model.ipynb                          ON HOLD — cluster classifier + SHAP (src/xgboost_model.py)
    phosx_implementation.ipynb                   EXPLORATORY — PhosX kinase-activity inference, data pre-processing stage
    omnipath.ipynb                               STUB — OmniPath prior-knowledge network (1 cell)
    Protein_ratio_of_phosphorylation.ipynb       EXPLORATORY — phospho-site / protein-level ratio (4 cells)
    kinase_library_implementation.ipynb          LEGACY prototype — superseded by kinase_prediction.ipynb (kept for reference)
  scratch/                  Exploratory / throwaway notebooks (Testing.ipynb)

Experiment/                 Raw and processed data per experiment — one folder per dataset key,
                            each with Data/ (FragPipe output) and Results/
  hme1_1/  hme1_2/  hek_1/  hme1_mutants_test/
  General_result/           Cross-experiment summary material
  Progress_report/          Progress-report material

External_Data/
  Metadata/                 Reference annotation tables
    uniprotkb_AND_reviewed_true_AND_model_o_2026_05_13_de_compressed.tsv
                            Full UniProt sequences — source of the ±7 kinase windows (src/kinase_prediction.py)
    PhosphoSitePlus.tsv     Curated site annotations (functional_score, ON_FUNCTION, ...)
    KSEA_app.csv/.xlsx      Kinase–substrate sets used for KSEA
    Phosphosite/ Reactome/ String/ TFactors/   Other prior-knowledge resources
    extendend_list_of_possible_ERK_sites.txt, kinases_list_TINA.txt,
    kinmaplabels_CORAL_P145.txt, protein_class_Kinases.json
  Time_series/
    P146_Feng_Song_phospho_proteomics_data/   MCF10A EGF time course (reformatted TSVs available)

data/                       Intermediate / shared processed data files ({dataset}_raw_sample.tsv,
                            {dataset}_transformed.tsv, small sample datasets for development)

Server/                     Long-running clustering jobs run off-laptop
  scripts/                  Parallelised tslearn sweeps (kmeans, kshape, kernelkmeans, FC/scaled)
  Server_results/           Returned results, one dir per run date + Clusters_evaluation/

PhosX/                      PhosX working directory (seqrnk input, phosx_output/)
Claude_promts/              Saved prompts / notes used to drive code generation
Old/                        Pre-refactoring notebooks — legacy, do not run
```

## Environment

Conda env **`TMT_Data_analysis`** (`~/miniconda3/envs/TMT_Data_analysis`), Python 3.10.20.
Notebooks import project code as `from src.xxx import yyy`, so they must be run with the repo root on
`sys.path` (relative data paths in the notebooks are written from `notebooks/<folder>/`).

Installed and in use: pandas 2.2.3, numpy 1.26.4, scipy 1.14.1, scikit-learn 1.6.1, tslearn 0.8.1,
matplotlib 3.8.4, plotly 6.9.0, xgboost 2.0.3, shap 0.49.1, kinase_library 1.5.1, umap-learn,
matplotlib-venn.

**Not installed** (code paths that need them will fail): `hdbscan` (lazily imported by
`hdbscan_clustering()` in `src/clustering.py`), `tensorflow` / `torch` (needed by
`Autoencoder_clustering.ipynb`).

## Naming conventions and data structure

### Column naming scheme

All data columns follow the pattern:

```
{CellLine}_{DataType}_{Concentration}_{Treatment}_{TimePoint}_{Replicate}
```

**CellLine** — e.g. `WT`, `BRAFS151A`, `GAB1Y259A`, `MCF10A`

**DataType** — two terms separated by `:`:
- First term: transformation applied
  - `raw` — detector intensity, no log transformation
  - `log2` — log2-transformed
- Second term: what the value represents
  - `abs` — per-replicate intensity (column name includes replicate suffix)
  - `mean` — mean across replicates for this condition × timepoint
  - `median` — median across replicates
  - `FC` — fold change relative to the starve control (log2 scale only)
  - `scaled` — FC scaled so amplitude is in [−1, 1] and starve = 0
  - `zscore` — per-site z-score of the FC temporal profile, standardised across the time series **per condition per cell line** (mean 0, std 1); isolates response shape from amplitude
  - `sd` — standard deviation across replicates
  - `cv` — coefficient of variation (%)
  - `var` — variance
  - `FDR` — false discovery rate
  - `pvalue` — p-value
  - `adjustedFDR` — −log10(FDR), also called adjusted p-value

**Treatment** — `EGF`, `INS`, `EGFnINS`

**TimePoint** — ordered qualitatively along the x-axis:
- `full` — cells in full media (first point)
- `starve` — 2 h serum-starved, no stimulation (reference / second point)
- `1`, `2`, `5`, ... — minutes post-stimulation (dataset-dependent)

**Replicate** — only present for `abs` columns: `r1`, `r2`, `r3`, `r4`

Examples:
```
WT_raw:abs_EGF_full_r1        per-replicate TMT intensity, WT, EGF arm, full media
WT_log2:FC_EGF_2              log2 FC vs starve, WT, EGF arm, 2 min
BRAFS151A_log2:mean_EGF_5     log2 mean across replicates, BRAFS151A mutant, 5 min
MCF10A_log2:abs_EGF_starve_r1 external MCF10A dataset, starve control replicate 1
```

### Metadata columns

Common to all datasets:

| Column | Description                                                                                                                                                                                                                                                                                     |
|--------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `site` | Composite phosphosite key: `{site_index}~{modified_sequence}` (LFQ) or `ProteinName-Residue` (TMT)                                                                                                                                                                                              |
| `protein_name` | Protein name                                                                                                                                                                                                                                                                                    |
| `protein_Id` | UniProt accession. **Note the spelling** — lowercase `d`, as on disk in every dataset (`protein_ID` exists only as a local variable name in `src/utils.py` / `src/plotting_functions.py`)                                                                                                                                                                                                        |
| `n:reps` | Number of replicates in which the peptide was detected. For the LFQ experiments this is the minimun amount of sapmles replicate the phosphosite was found. This means that if for one site the sindings are [[No, Yes, Yes], [Yes, No, Yes], [No, No, Yes]], the number of replicates will be 1 |
| `AScore` | Phosphosite localization ambiguity score                                                                                                                                                                                                                                                        |

LFQ-specific columns (added by `add_modification_metadata()` in `src/lfq_pretreatment.py`):

| Column | Description |
|--------|-------------|
| `site_index` | Canonical site key: `{protein}_{start}_{end}_{n_mods}_{n_sty}_{positions}` |
| `n_localized` | Total number of modifications on the peptide |
| `STY_localized` | Number of phosphorylations (S/T/Y) specifically |
| `other_localized` | Non-phosphorylation modifications (oxidation, acetylation, etc.) |
| `assigned_modifications_clean` | Cleaned modification string, e.g. `S479S484` |
| `sty_positions` | Phosphosite positions only, e.g. `S471S479` |

PhosphoSitePlus annotation columns (merged in preprocessing notebooks):

| Column | Description |
|--------|-------------|
| `functional_score` | PhosphoSitePlus functional relevance score |
| `ERK_motif` | Whether the site matches an ERK consensus motif |
| `ON_FUNCTION` | Known function of this phosphorylation event |
| `ON_PROCESS` | Biological process associated with this site |
| `ON_PROT_INTERACT` | Protein interactions regulated by this site |

Kinase-prediction columns (added by `predict_top_kinases()` / `add_kinase_windows()` in `src/kinase_prediction.py`; flat per-site annotations — **not** the `{CellLine}_{DataType}_...` data-column scheme, so `ColumnSpec.select()` ignores them):

| Column | Description |
|--------|-------------|
| `kinase_window` | ±7 (15-mer) sequence window centered on the phospho-acceptor, cut from the full UniProt protein sequence |
| `kinase_residue` | The phospho-acceptor residue (S/T/Y); routes the site to the ser_thr vs tyrosine kinome |
| `uniprot_seq_match` | QC flag: whether the detected peptide matches the UniProt sequence at its position (False ≈ isoform/mapping mismatch) |
| `predicted_kinase_1` … `predicted_kinase_5` | Top-5 predicted kinases (rank 1 = most likely), from the kinase_library motif matrices |
| `predicted_kinase_1_prob` … `predicted_kinase_5_prob` | Percentile score (0–100) of each predicted kinase — the "how likely" measure; non-increasing across ranks |

Only computed for **single-localized** sites (TMT `LocalizedNumPhos == 1`; LFQ `n_localized == 1 & other_localized == 0`); other rows hold NaN.

### FragPipe-derived columns carried through the pipeline

These come straight from the FragPipe output and are kept in the processed tables. They are used by
`src/filters.py` and as features in `src/xgboost_model.py`, so do not drop them during preprocessing:

| Column | Description |
|--------|-------------|
| `description` | Full UniProt protein description |
| `protein_length`, `nrPeptides`, `nr_tryptic_peptides` | Protein-level identification quality |
| `peptide_index`, `peptide_seq`, `SequenceWindow` | Peptide identity and its sequence context |
| `Start`, `End` | Peptide start/end position in the protein |
| `startModSite`, `endModSite`, `PhosSites` | Modification positions on the peptide |
| `NumPhos`, `LocalizedNumPhos` | Number of phosphorylations detected / confidently localized |
| `MaxPepProb` | Maximum peptide identification probability |
| `ReferenceIntensity` | FragPipe reference-channel intensity |
| `CON`, `REV` | Contaminant / reverse-decoy flags — filtered by `filter_contaminants()` |
| `isotopeLabel` | TMT channel / label information |

### Cluster-label columns

Clustering writes one label column per (algorithm × transform) combination, named:

```
{Algorithm}_cluster_{CellLine}_{Condition}_{transform}
```

The six columns present in `Data_clustered/20260715_hTERT_HEM1_2_processed_phPlus_clustered*.tsv`:

```
KMeans_adaptive_cluster_WT_EGF_log2_FC       KMeans_11_cluster_WT_EGF_log2_FC
KMeans_adaptive_cluster_WT_EGF_log2_scaled   KMeans_11_cluster_WT_EGF_log2_scaled
KMeans_adaptive_cluster_WT_EGF_log2_zscore   KMeans_11_cluster_WT_EGF_log2_zscore
```

**Important naming caveat:** the `_EGF_` in the `KMeans_adaptive_*` names is misleading — those labels were
computed from **all three conditions jointly** (EGF + INS + EGFnINS). The `KMeans_11_*` labels were computed
from **EGF only**. This distinction determines what counts as information leakage in any supervised model
built on these labels.

These are flat annotation columns, not `{CellLine}_{DataType}_...` data columns, so `ColumnSpec.select()`
ignores them.


## Data analysis overview

The analysis plan is to cluster the temporal phosphorylation dynamics across all stimulation
conditions. Each phosphopeptide is represented as a multidimensional time series (one dimension
per condition: EGF, INS, EGFnINS), and peptides with similar temporal profiles should cluster
together.

**Workflow:**
1. Unsupervised clustering on the WT hTERT-HME1 dataset
2. Use cluster assignments as labels to train a classifier
3. Apply the classifier to the mutant cell line datasets

Interpretation: if a peptide in the mutant dataset is assigned to the same cluster as in WT, its
temporal profile is unchanged and it was not required to buffer the introduced perturbation. If it
moves to a different cluster, its dynamics changed — either directly affected by the mutation or
compensating for it.

Question: i can train a ML model to be used as clasifier, but I could also just do the clustering in one cell line, and 
then use the centroids of those clusters to cluster following the smae criteria the other cell lines. What would make 
more sense? 

### ⚠️ XGBoost classifier — IMPLEMENTED BUT ON HOLD (direction not decided)

> **Status flag:** `src/xgboost_model.py` + `notebooks/05_downstream/XGBoost_model.ipynb` exist and run,
> but **this direction is not committed to**. Do not build on it, extend it, or wire it into other
> notebooks without asking first. It is kept because the code and the negative result are both informative.

**What it does.** Trains multi-class XGBoost classifiers to predict a site's dynamic cluster label from
*metadata* rather than the raw profile, and uses SHAP to attribute the prediction to features. Three
models are compared:

| Feature family | Content | Role |
|----------------|---------|------|
| **A — bio** (`build_bio_features`, 47 features) | predicted kinase (top-K one-hot), residue, ERK motif, functional score, localization counts, peptide quality, curated-annotation flags | independent predictor — "does intrinsic site biology explain the cluster?" |
| **B — temporal** (`build_temporal_features`, 36 features) | per-condition peak / AUC / slopes / amplitude / transient score + cross-condition synergy, from the FC profiles | descriptive of cluster geometry (leaky for the adaptive target) |
| **A+B combined** (83 features) | both | — |

**Results so far** (target `KMeans_adaptive_cluster_WT_EGF_log2_FC`, hme1_2, 5-fold CV; majority-class
baseline accuracy 0.617):

| model | CV accuracy | CV macro-F1 |
|-------|-------------|-------------|
| bio | 0.454 ± 0.013 | 0.208 ± 0.009 |
| temporal | 0.976 ± 0.003 | 0.924 ± 0.024 |
| combined | 0.974 ± 0.003 | 0.919 ± 0.026 |

**How to read this — and why the direction is in question:**
- The temporal model near-perfectly recovers the labels, which is **expected and uninformative**: those
  descriptors reconstruct the clustering input (leakage by design). Its only legitimate use is *describing*
  what shape defines each cluster.
- The bio model is the actually interesting question, and it **does not work** (macro-F1 0.21, accuracy
  *below* the majority baseline). Intrinsic site metadata does not predict which dynamic cluster a site
  lands in on this dataset. That is a real negative result, not a bug.
- Consequence for the workflow above: a metadata-based classifier is a weak route to the mutant datasets.
  The **nearest-centroid / cluster-transfer** alternative in the open question above looks more defensible,
  and is already prototyped in `notebooks/03_clustering/Clustering_mutant_cell_lines.ipynb`.

**Caveats recorded in the notebook:** class imbalance (balanced sample weights → accuracy can drop below
baseline while macro-F1 rises, so judge on macro-F1); ~40% of sites have NaN kinase features
(non-single-localized); single dataset (WT hme1_2 only). Swapping `TARGET_COL` to
`KMeans_11_cluster_WT_EGF_log2_FC` (EGF-only clustering) would make the INS/EGFnINS temporal descriptors
a genuinely non-leaky crosstalk test — **that variant has not been run.**


## Current status

**Done:**
- Preprocessing and transformation pipeline (`src/transformations.py`, `src/lfq_pretreatment.py`)
- Full QC function library (`src/QC.py`): missing values, intensity distributions, CV, PCA
  (static and interactive), PCA distance heatmap, UMAP (sample and site level), Venn diagrams, overlap stats
- Plotting functions restructured and unified (`src/plotting_functions.py`): `clusters_plot_linear`
  supports `panel_by="condition"` and `panel_by="cell_line"`; mutant-specific redundant functions removed
- External MCF10A reference dataset reformatted to project naming convention
- Clustering implemented and working (TimeSeriesKMeans primary; KShape, KernelKMeans, HDBSCAN, autoencoder also explored)
- Preprocessing and QC folder reports written (`REPORT_01_preprocessing.md`, `REPORT_02_qc.md`)
- Clustering strategy documented (`notebooks/03_clustering/clustering_overview.md`)
- Kinase imputation pipeline (`src/kinase_prediction.py`): UniProt-derived ±7 windows + top-5 kinase/percentile prediction, cross-dataset (TMT & LFQ), with sequence validation
- Cluster-composition queries (`src/cluster_composition.py`): which clusters hold sites of given kinases / proteins
- Downstream biology: Fisher's-exact cluster enrichment (`src/cluster_enrichment.py`) and KSEA kinase-activity inference (`src/kinase_activity.py`), both fully documented for defense
- Row-filtering module (`src/filters.py`): n:reps, contaminants, dynamics range, incomplete time series, localization, by protein / by site
- XGBoost cluster classifier + SHAP (`src/xgboost_model.py`, `XGBoost_model.ipynb`) — runs, but **direction on hold**; see the flagged section above. Bio-metadata features do not predict cluster (macro-F1 0.21); temporal features are leaky by design.
- Hierarchical clustering (`src/hierarchical_clustering.py`, `Hierarchical_clustering.ipynb`) — tree built once and cut afterwards, so the k-scan is near-free; tree-level QC (cophenetic correlation, merge-distance elbow, per-cluster silhouette, nearest-centroid agreement) plus **constrained merging** of clusters sharing a parental node. Notebook verified end-to-end on hme1_2.

- Clustering method decision document (`notebooks/03_clustering/clustering_method_decision.md`) — KMeans vs hierarchical, transformations, QC/statistics, sigmoid/T50 and impulse modelling; includes the four verified preprocessing defects and the measured null cluster-switching rate.

**In progress / pending:**
- **Preprocessing fixes — highest priority, blocking everything downstream.** Normalisation, the inert dynamics filter, the z-score basis, and the on-disk `fillna(0)` (see "Known issues"). Every clustering result to date was computed on unnormalised data through an inert filter.
- Clustering optimisation (parameter tuning, optimal number of clusters) — **method recommendation now made**: KMeans (`n_init` ≥ 25) per condition on a corrected z-score, with Ward redeployed to the centroid-level tree. k chosen by stability + interpretability + biological content, reported as a profile across k rather than a single value. See `clustering_method_decision.md` §6.
- Transfer of WT cluster labels to the mutant datasets — **decision now made in favour of nearest-centroid**, but *soft* (distance vector + assignment margin, calibrated against a WT split-half null), on shared timepoints only, in an amplitude-free representation, and **within a single experiment/platform** — never hme1_2 TMT centroids onto LFQ mutant data (10× dose + platform + grid confound). The XGBoost route stays on hold. **Prerequisite: measure the null switching rate first** — it is 38–66% within WT.
- Establish the analysis nulls before any mutant claim: split-half switching rate, mutation positive control (does the pipeline recover its own mutations?), XGBoost label-noise ceiling, gap statistic / dip test / bootstrap Jaccard. See `clustering_method_decision.md` §12 Phase 3.
- Parametric curve fitting (`src/response_shapes.py`, `src/curve_fitting.py` — not yet written): shape classification + model-free descriptors, then anchored 3-parameter log-time sigmoid (monotonic/sustained sites) and exponential-difference model (transient sites); impulse model once hme1_lfq lands.
- Downstream analysis: PhosX integration — `notebooks/05_downstream/phosx_implementation.ipynb` is at the data-pre-processing stage; `PhosX/` holds a seqrnk input and output dir (Fisher enrichment + KSEA now done)
- `omnipath.ipynb` (prior-knowledge network) and `Protein_ratio_of_phosphorylation.ipynb` are stubs — not started
- `notebooks/01_preprocessing/LFQ_diaPASEF.ipynb` is a stub (sample→column-name map only), waiting on hme1_lfq data acquisition
- Kinase prediction on LFQ datasets: cross-dataset path implemented but only run on TMT hme1_2 so far (LFQ path smoke-tested, not run on a full mutant dataset)
- LFQ preprocessing: add INS and EGFnINS conditions (currently EGF only)
- TMT preprocessing: add filtering steps to match LFQ filtering (no-PTM, not-in-starve, missing-replicates)

### Session 2026-07-16 — kinase imputation & cluster-label downstream analysis

**Topics discussed:**
- The `density=True` histogram y-axis in `diagnose_cluster_substructure` (probability density vs counts; why bar heights can exceed 1 — area, not height, sums to 1). Explanation added to `Adaptive_clustering_sweep.ipynb` and `Clustering.ipynb`.
- Design of a clean, cross-dataset kinase-imputation workflow around the `kinase_library` package (v1.5.1, installed); why site percentile (0–100) is the defensible "how likely" metric; S/T vs Y kinome routing; single-localized-only scoring.
- What further exploration of the cluster labels is worth doing before the classifier.

**Implemented this session:**
- `src/kinase_prediction.py` + `notebooks/05_downstream/kinase_prediction.ipynb` — top-5 kinases + percentiles per site. Windows cut from full UniProt sequences (`External_Data/Metadata/uniprotkb_..._de_compressed.tsv`) using absolute phospho positions (uniform for TMT/LFQ); `uniprot_seq_match` validation flag (hme1_2: 91 genuine peptide mismatches). Replaces the legacy `kinase_library_implementation.ipynb` (13-mer window, ser_thr-only, magic column slice).
- `src/cluster_composition.py` — `kinase_cluster_table` / `protein_cluster_table` / `plot_cluster_composition`; demo cells appended to `kinase_prediction.ipynb`.
- `src/cluster_enrichment.py` + `src/kinase_activity.py` + `notebooks/05_downstream/cluster_enrichment_and_kinase_activity.ipynb` — idea **#1** (Fisher enrichment: ERK/RSK/P38 substrates and ERK motifs enrich in specific clusters, BH-FDR) and idea **#2** (KSEA activity trajectories: recovers EGF→ERK/AKT dynamics; compares EGF/INS/co-stimulation). Notebook written with extensive statistical/biological documentation + limitations + references (Fisher, Benjamini-Hochberg, Casado 2013 KSEA, Wiredja 2017, Johnson 2023).

**Ideas discussed but NOT yet implemented** (candidate next steps, roughly in priority order):
- **#3 Label reliability / consensus** — cross-tabulate the 6 cluster columns (adaptive vs KMeans_11 × FC/scaled/zscore) with ARI + contingency to find sites with consistent vs flip-flopping labels; use to confidence-weight or filter classifier training data.
- **#4 Condition-specificity / crosstalk within clusters** — classify sites/clusters as EGF-specific, INS-specific, shared, or synergistic (EGFnINS ≠ EGF+INS additive); the core crosstalk question.
- **#5 Multi-site proteins** — do multiple sites on the same protein co-cluster or scatter across clusters (coordinated vs site-specific regulation)?
- **#6 External MCF10A validation** — map homologous EGF-response sites onto the dynamic clusters.
- **Classifier feature framing** — decide whether to predict cluster from temporal features (transfer/nearest-centroid to mutants → detect cluster-switching) vs from static features (sequence/kinase/motif → sites deviating in mutants are "rewired"). Noted as a design decision, not yet made.
- **`posthoc_enrichment_fisher` stub** in `src/adaptive_clustering.py` is now superseded by `src/cluster_enrichment.py` (left in place, not wired).

### Session 2026-07-30 — CLAUDE.md audit and structure refresh

Went through the repo against this file and brought it back in sync. Added/corrected:
- **Repository structure** fully rewritten — it was missing `src/filters.py`, `src/xgboost_model.py`, 8 notebooks (`LFQ_diaPASEF`, `Adaptive_clustering_sweep`, `Autoencoder_clustering`, `Clustering_mutant_cell_lines`, `Plotting_time_series`, `profiles_difference`, `XGBoost_model`, `phosx_implementation`, `omnipath`, `Protein_ratio_of_phosphorylation`) and 5 top-level dirs (`Server/`, `PhosX/`, `Old/`, `Claude_promts/`, plus the `External_Data/{Metadata,Time_series}/` reorganisation). `Experiment/` subfolders were listed under their old names (`1_HEK293T`, `2_hTERT_HME1`, ...) and are now dataset keys (`hek_1`, `hme1_2`, ...).
- **Dataset paths** filled in, and an explicit note that everything in `data/` is a 100–1000-row development sample, not the full dataset.
- **`protein_ID` → `protein_Id`** — the documented spelling did not match any dataset on disk.
- **New column documentation**: FragPipe-derived columns carried through the pipeline, and the cluster-label column convention including the caveat that `KMeans_adaptive_*_EGF_*` labels were actually computed from all three conditions.
- **Environment section** (conda env, versions, and the not-installed packages that break `hdbscan_clustering` and the autoencoder notebook).
- **XGBoost section added and flagged as on hold** at the user's request — direction not yet decided.

### Session 2026-08-03 — hierarchical clustering with constrained merging

New module `src/hierarchical_clustering.py` + `notebooks/03_clustering/Hierarchical_clustering.ipynb`,
built as a parallel to `Clustering.ipynb` (same filtering, same `ColumnSpec` selection, same
`reshape_df`, so the two partitions are directly ARI-comparable). Notebook executed end-to-end on
hme1_2 (WT / EGF / `log2:zscore`, 8723 sites): 33 code cells, 0 errors.

**Why hierarchical rather than another KMeans variant.** The tree is built once and *cut afterwards*,
which changes three things: (1) the k-scan is one `fcluster()` call per k instead of a refit, (2) the
method is deterministic so seed-stability is meaningless and QC has to judge the tree instead, and
(3) every cluster **is** a tree node, so clusters under a common parent can be merged exactly — the
merged cluster is that parent. This is the "inspect the clusters, then constrain the result" loop
that motivated the session.

**Merging API** — `cluster_level_linkage()` extracts the sub-hierarchy above the cut (real merge
heights, unlike `clustering.compute_centroid_linkage()`, which re-clusters centroids and invents a
hierarchy the model never had); `mergeable_cluster_groups()` lists every legal group with height,
merged size and centroid distance; `suggest_merges()` (max-height / min-size rules, union-find
deduplicated); `merge_clusters()`; `merge_clusters_by_height()`; `plot_cluster_tree()`.
Non-sibling merges are **refused by default** — the result would not be a subtree and no further tree
operation would be defined on it (`allow_non_sibling=True` forces it, with a warning).

**Tree-level QC** — `cophenetic_correlation()` / `compare_linkage_methods()`, `plot_merge_distances()`
(hierarchical elbow), `hierarchical_kscan()` + `plot_hierarchical_scan()`, `silhouette_per_cluster()`,
`assignment_agreement()`. Reuses the existing `plot_cluster_assignment_qc`, `plot_cluster_scores`,
`cluster_similarity_per_condition`, `clusters_plot_linear_mutants` and
`diagnose_cluster_substructure` — `HierarchicalResult.distances_to_centroids` is deliberately shaped
like the KMeans `barycenters` matrix.

**Findings on hme1_2 (WT, EGF, log2:zscore, ward, k=12), recorded so they are not re-derived:**
- Linkage comparison: `average` has the best cophenetic correlation (0.76 vs ward 0.71, weighted 0.64,
  complete 0.52) but puts **49%** of sites in one cluster against ward's 19%. Cophenetic correlation
  must be read together with `largest_cluster_frac`; ward remains the default.
- The merge-distance elbow suggests **k=3** — the largest absolute gap sits at the top of the tree.
  Documented in the notebook as a caveat: read local gaps within the k range of interest instead.
- Nearest-centroid agreement is **78.7%** — i.e. ~21% of sites are not closest to their own centroid.
  Directly relevant to the pending mutant-transfer decision: that is the error a nearest-centroid
  transfer would inherit before any biology enters.
- ARI vs the existing KMeans columns: highest against `KMeans_11_cluster_WT_EGF_log2_zscore` (**0.43**),
  lowest against `KMeans_adaptive_cluster_WT_EGF_log2_FC` (0.10 — expected, since the adaptive labels
  were computed from all three conditions jointly).
- **Worked negative example kept in the notebook:** the min-size rule proposes merging clusters 6 and
  11, which *lowers* the mean silhouette (0.143 → 0.137) and pushes the merged cluster negative. That
  pair has the largest centroid distance of any sibling pair (~2.55 vs ~0.92 for the cheapest). A small
  cluster is a reason to look, never on its own a reason to merge. The notebook default
  (`MERGE_MAX_HEIGHT = 25.0`) instead merges 2 and 5 and improves every metric
  (silhouette 0.143 → 0.154, worst cluster −0.033 → −0.010, agreement 78.7% → 79.4%).

**Not done / open:** no hierarchical run yet on multiple conditions jointly or on the mutant datasets;
bootstrap stability is written into the notebook but left commented out; `plot_cluster_hierarchy()`
from `plotting_functions.py` indexes `axes[row, col]` and therefore needs ≥2 conditions — it is not
used in this notebook, which uses `plot_cluster_tree()` instead.

### Session 2026-08-03 (part 2) — clustering method decision document

Wrote **`notebooks/03_clustering/clustering_method_decision.md`** (~1350 lines), a decision document
covering KMeans vs hierarchical for this data, transformations, QC/statistics, and parametric curve
fitting (sigmoid/T50 and the Chechik–Koller impulse model). Produced by four independent analyses
(clustering mathematics; QC/statistics/transformations; temporal-curve modelling; adversarial
thesis-defence review), each computing directly on the hme1_2 clustered file. Numbers in the document
are labelled **[verified]** (re-run and reproduced), **[measured]** (computed on the real data by one
analysis, not re-checked), or **[theory]**.

**The method verdict — KMeans, not hierarchical, and the usual arguments run backwards:**
- Ward and Lloyd's KMeans minimise the *same* functional $W$ (within-cluster sum of squares); Ward does
  it greedily under a nestedness constraint and never revisits a merge, costing **+15.5% excess $W$ at
  k=12** and leaving 21% of sites not nearest their own centroid.
- **Determinism ≠ stability.** Ward is deterministic but ill-conditioned: 80% subsample ARI **0.408**,
  +1% noise ARI 0.411, only **5.1%** of sites with consensus > 0.8. KMeans (`n_init≥10`) gives **0.875**,
  0.914, and **83.8%**. Ward's saturation (0.411 at 1% noise, 0.371 at 20%) is diagnostic of
  ill-conditioning, not noise sensitivity.
- **The Ward-vs-KMeans ARI of 0.434 is uninformative** — Ward disagrees with *itself* by 0.408.
- **Decisive constraint: transfer.** A WT→WT split-half nearest-centroid transfer (zero biology, same
  distribution) gives ARI **0.812** for KMeans and **0.417** for Ward. Ward-based mutant switching calls
  would be measuring the method.
- Ward is redeployed, not discarded: near-free k-scan, coarse→fine narrative, constrained merging — but
  **build the tree on 12–25 KMeans centroids**, not on 8723 sites.

**The structural finding (constrains what may be claimed):** the response space is a **continuum**, not
discrete groups. Effective dimensionality **2.85** (PC1 = early-vs-late 53%, PC2 = transient-vs-sustained
21%; in FC space PC1 alone is 82%). A covariance-matched structureless null reproduces **~75% of the
silhouette at every k**, and the excess is flat — **the data votes for no k beyond 2–4**. Two scalars
(signed peak amplitude, peak time) recover **65%** of the 11-cluster labels. Clustering must be described
as *quantisation of a continuum* / an organising device, never as discovery of classes.

**⚠️ The measured null switching rate — the most important number for the mutant work.** Same cells, same
experiment, same peptides, only the replicates differ (split-half on the 2897 sites with 4/4 reps):
**37.7% (log2:FC) to 66.0% (log2:zscore) of sites change cluster at k≈12**. In the real mutant comparison
the null is higher (different cultures, plexes, platform, 10× dose, different timepoint grid). **Until this
is calibrated — ideally against the BRAF-S151A biological duplicate in hme1_lfq, samples 3 and 6 — no
mutant switching rate is interpretable.** Do not spend that duplicate as extra n.

**Other quantitative results recorded so they are not re-derived:**
- Cophenetic correlation is **anti-correlated** with partition quality here (average 0.76 / 49% giant
  cluster; ward 0.71 / 19%). It measures *tree* fidelity, not partition quality — `compare_linkage_methods`
  currently sorts by it, putting the worst partitioner on top.
- **The representation choice changes the answer more than the algorithm choice**: ARI(ward-zscore,
  KMeans-FC) = 0.216 vs 0.434 on zscore.
- Uniform timepoint weighting ≈ log-time $L^2$ (**r = 0.987**), because the grid {0,2,5,10,15,90} is already
  log-spaced. Real-time weighting would put **86% of the distance on the 15 and 90 min points**. State this
  as a design choice. But a 1.3% metric change reshuffles the partition to ARI 0.372 — more ill-conditioning.
- **Co-stimulation is strongly sub-additive**: observed Var(EGFnINS) 0.382 vs additive prediction 0.747,
  median |synergy| 0.199 against median |signal| 0.412. A headline crosstalk result, currently invisible to
  the clustering. Conditions are highly redundant (median per-site ρ: EGF~INS 0.78, EGF~EGFnINS 0.84), so
  joint clustering averages the crosstalk away — **cluster per condition and cross-tabulate**.
- DTW is wrong here (warping *is* the signal; not a metric, so Ward is undefined on it). `compute_centroid_linkage()`
  uses `cdist_dtw` — switch to Euclidean. `transpose` in `reshape_df` cannot change a Euclidean result.

**Parametric modelling (the direction three of the four analyses independently recommended):**
- **Shape census, EGF: 69.3% of sites are transient or biphasic**; only **30.3%** are sigmoid-legitimate.
  73% peak at 5 or 10 min. A monotone sigmoid structurally cannot represent the majority — classify before
  fitting and never report T50 for a site that failed the gate.
- Fit the **anchored 3-parameter** sigmoid in log-time, not a 4PL in linear time: `log2:FC_*_starve` is
  identically 0, so the baseline is a *constraint*, not a parameter. In linear time the 90-min point has
  **leverage 0.979**; log-time is not a better *fit* (RMSE 0.1048 vs 0.1051) but a better *parameterisation*
  (k varies 30-fold with T50 in linear time, 1.5-fold in log-time).
- **T50 is a set-level statistic**: per-site SE ≈ 2.5 min against a biological IQR of ~3.1 min, left-censored
  below 2 min. Defensible claim: *"substrates of kinase K are delayed by 1.8 ± 0.4 min"*, not *"site X is delayed."*
- **Chechik & Koller (2009), J Comput Biol 16(2):279–290 — citation and 6-parameter functional form both
  verified** against the ImpulseDE2 reference implementation. Two traps: the published $1/h_1$ product form is
  singular on signed log2 FC data (use Calico's offset parametrisation), and $h_1$ is *not* the peak height.
- **Why it cannot be fitted on hme1_2 — exactly: zero lack-of-fit degrees of freedom** (d−1−p = 6−1−5 = 0).
  It interpolates any site including pure noise; 43% of test fits land inside the noise, 76.5% drive β into
  step-function territory, max parameter correlation 0.995, and SE(t₂) reaches **1164 min for a true 45 min**.
  **Replicates do not help** — they buy pure-error df, not lack-of-fit df.
- hme1_lfq (7 post-stimulation points) makes the 5-parameter impulse identifiable with **2 lack-of-fit df** —
  minimum viable. It removes the +42 min bias in t₂ for slow sites but leaves SE(t₂) ≈ 16.8 min.
- ⭐ **If the hme1_lfq schedule is still open, add a 45 min timepoint**: slow-site SE(t₂) 16.8 → **4.2 min**
  (Monte-Carlo SD 25.6 → 2.7). **A 3 min point improves essentially nothing.** Drop `full` or 20 min if the
  count is fixed; keep 90.

**Council working notes** — the four underlying analyses, kept for their derivations and intermediate
tables: `notebooks/03_clustering/council_note_{1_clustering_mathematics, 2_qc_statistics_transformations,
3_temporal_curve_modelling, 4_adversarial_review}.md`. Read `clustering_method_decision.md` first; the
notes are single-source and mostly not re-verified. Note 4 was reconstructed from the session transcript
(it was returned inline rather than persisted), which is flagged in its header.

### Known issues

**⚠️ Four preprocessing defects, all independently verified on `Data_clustered/20260715_*.tsv` (2026-08-03).
Fix these before drawing conclusions from any clustering — each changes the partition more than the choice
of algorithm. Full detail and the fixes: `notebooks/03_clustering/clustering_method_decision.md` §1.**
- **No sample-loading normalisation anywhere in the pipeline.** `run_all_transformations` goes
  `raw:abs → log2:abs → log2:mean → log2:FC` with no between-sample scaling. The 84 `raw:abs` column medians
  span **18.30–19.39 (1.09 log2, 2.1-fold)**, which propagates straight into every fold change: median
  `log2:FC` across *all 8723 sites* is **+0.566 at EGF 5 min**, +0.508 at 10 min, +0.312 at 90 min, while the
  IQR is only 0.33–0.61 — the entire distribution is shifted, not a tail. `normalization_boxplots()` in
  `src/QC.py` **cannot detect this**: it compares `raw:abs` against its own logarithm, and is giving false
  reassurance. Fix: median-centre or quantile-normalise each sample column in log2 space before
  `compute_log2_stats`; consider Internal Reference Scaling (Plubell 2017) if the bridge channel is available.
- **Consequence: `filter_dynamics(threshold=0.5, mode="extremes")` passes 100.0% of sites** on the EGF arm —
  it is currently filtering nothing. After median normalisation, 46.6% pass. Better still, replace the
  amplitude cutoff with a limma moderated-F responsiveness test + BH-FDR (expected yield ~2650 sites), which
  also brings the dataset under `consensus_stability`'s `max_sites_for_coassociation=6000` guard.
- **`compute_zscore_fc()` standardises over 7 timepoints (`exclude_full=False` default) but the notebooks
  cluster 6** (`EXCLUDE_FULL=True`). The vectors fed to the clusterer have neither mean 0 (max |row mean| =
  0.402) nor SD 1 (median 1.043), and the retained `WT_log2:zscore_EGF_starve` column — identically 0 in FC
  space, since FC is defined relative to starve — becomes a free-floating feature carrying **25.9% of total
  clustering variance** and correlating **−0.72 with `log2:FC_EGF_full`**. About a quarter of the clustering
  geometry is "how different is this site in full media", not "how does it respond to EGF". Fixing it changes
  the partition (ARI **0.283** vs the current one) more than switching algorithms does, raises silhouette
  (0.138 → 0.165 at k=12) and stops it collapsing at high k (k=20: 0.087 → 0.147). Fix: standardise over the
  stimulation timepoints only, excluding both `full` and `starve`.
- **A `.fillna(0)` reached disk.** In `Data_clustered/20260715_*.tsv`, `log2:abs` is exactly 0.0 wherever
  `raw:abs` is 0.0 — in **100% of such cells**, covering 19–36% of cells per replicate column. The derived
  statistics in the file are still correct (stored `log2:mean` matches the non-zero mean), so the clustering
  input is not corrupted, but anyone recomputing from `log2:abs` gets garbage (a naive replicate SD returns
  ~8.8 instead of ~0.21). Violates the project's own no-fillna rule at the level of data on disk. Fix: write
  NaN and add an assertion to every save step.
- **Biological consequence worth checking before the mutant work:** after median normalisation, every strong
  marker survives and several get cleaner (SOS1 Y1196 becomes an unambiguous 2-min spike; JUN S73 an
  unambiguous late response), but **BRAF S151 — one of the mutated sites — loses its EGF response entirely**
  (0.58/0.66 at 5/10 min → −0.04/0.10). MEK1 T292 halves. Verify before interpreting the BRAF-S151A line.

- ~~`MSMS_data_QC.ipynb` loads data with `.fillna(0)` — this invalidates the missing value analysis and distorts CV and intensity distributions.~~ **FIXED (2026-07-09):** the four load lines now use `low_memory=False` instead of `.fillna(0)`, so missing intensities stay NaN (this also cleared the `DtypeWarning` on the sparse annotation columns). `peptide_count_per_sample()` in `src/QC.py` was the only QC function that counted NaN as detected (`df != missing_value`); it now uses `notna() & (df != missing_value)`, matching `replicate_detection_map`. All QC functions treat NaN as missing, so the `missing_value=0.0` arguments in the notebook cells are safe to keep.
- ~~PhosphoSitePlus merge in both preprocessing notebooks produces almost entirely NaN for annotation columns~~ **FIXED (2026-07-09):** `get_column_infos()` in `src/transformations.py` (used by `merge_phosphoplus_info`) iterated over an empty list instead of the parsed sites, so it always returned NaN. This affected every `merge_phosphoplus_info` column (`ERK_motif`, `ON_FUNCTION`, `ON_PROCESS`, etc.) in both TMT and LFQ notebooks; `functional_score` (via `merge_functional_score`/`get_average_score`) was already correct. Key alignment (protein ID + residue) was fine. Lower hit rates for regulatory-site columns are expected (small curated table), not a bug.
- `tps_file_creator.ipynb` uses the old pre-refactoring column naming and legacy `.xlsx` paths — it cannot be run on current data without rewriting.
- `Autoencoder_clustering.ipynb` is also pre-refactoring code: `from utils import *`, old function names (`filter_replicates`, `filter_site_localizations`, `filter_dynamics_extremes`), a legacy `Experiment/hme1_2/Data/Processed/...` path, and `.fillna(0)` on load (violates the no-fillna rule). It additionally needs tensorflow/torch, which are not installed. Treat as legacy — rewrite before reusing.
- `hdbscan_clustering()` in `src/clustering.py` lazily imports `hdbscan`, which is not installed in the env, so the function fails at call time. Its own docstring starts "NOT A GOOD CLUSTERING METHOD" — do not use without a reason.
- `src/utils.py` still holds older plotting/filtering helpers that overlap with `src/plotting_functions.py` and `src/filters.py` (kept for comparison per project rule). Prefer the newer modules; treat `utils.py` duplicates as deprecated.
- `src/__pycache__/*.pyc` files are tracked in git (visible in `git status`) — they should be gitignored rather than committed.
- Biological QC (known MAPK/ERK marker behavior) is not yet implemented in `MSMS_data_QC.ipynb`.
- Sample-level UMAP (`umap_plot_interactive`) separates the duplicated `full`/`starve` samples that overlap in PCA — this is **expected UMAP behavior, not a bug** (UMAP is a stochastic force-directed layout that does not co-locate identical points). Use PCA for replicate-agreement/overlap QC; use UMAP only for non-linear structure, and prefer the site-level UMAP. Documented in `notebooks/02_qc/QC_notes.md` and the `umap_plot_interactive` docstring.
- ~~`pca_plot_interactive`/`umap_plot_interactive` marker shape was hard-wired to stimulation type, so mutant datasets (EGF-only) rendered every point as one marker and timepoints were indistinguishable.~~ **FIXED (2026-07-09):** both functions now take a `symbol_by` argument (`"stimulation"` default / `"timepoint"` / `"condition"`); default behavior is unchanged. For mutants use `color_by="cell_line", symbol_by="timepoint"`. The mutant UMAP cell in `MSMS_data_QC.ipynb` was updated to use `hme1_mutants` with these settings.


## Important rules

- Always add docstrings to new functions explaining arguments and return values
- Add documentation to old code when editing it
- Never overwrite `.csv` or `.xlsx` files — always append a descriptive suffix or version tag to the output filename (e.g. `_reformatted`, `_filtered`)
- Use `ColumnSpec.select()` from `src/column_spec.py` for all data column selection — do not build column lists by hand with string matching
- Processed data files should be saved as `.tsv`
- If I ask you to move functions from one file to the other don't delete them from the original file, comment them with "#" so I can still compare the old functions with the new ones in the new file
- Never load data with `.fillna(0)` before running QC — missing values must remain as NaN so that QC functions can detect and report them correctly
- `tps_file_creator.ipynb` is legacy code — do not edit or extend it without explicit instruction; if TPS analysis is needed, it must be rewritten using the current column naming convention and `ColumnSpec.select()`
- The XGBoost direction (`src/xgboost_model.py`, `notebooks/05_downstream/XGBoost_model.ipynb`) is **on hold** — do not extend it or wire it into other notebooks without asking first
- `Old/` and `notebooks/scratch/` are legacy / throwaway — never import from them and never treat their code as the current API
- Never use `data/` files for biological conclusions — they are truncated development samples; use the full tables under `Experiment/{dataset}/` or `notebooks/03_clustering/Data_clustered/`


## Code format

- Whenever a function is defined or called the format used to write it must be:
  function_name(var_1,
                var_2,
                var_n,)
- All functions when defined need to have the following description 
  """
  Description of what the function does

  Args:
    Arg1: description
    Arg2: description

  Returns:
    What it returns

  """