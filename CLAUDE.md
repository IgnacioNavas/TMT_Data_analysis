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
- hme1_lfq — **on disk since 2026-08, dataset key `hme1_diaPASEF`**
  - Path: `Experiment/hme1_diaPASEF/`
    - `Data/dia-quant-output/abundance_multi-site_MS2quant_None.tsv` — FragPipe DIA output (`None` =
      no FragPipe normalisation)
    - `Data/Processed/20260818_peptide_MS2quant_None_transformed_nolimma.tsv` — after
      `run_diapasef_transformations()`
    - `Data/Processed/20260818_peptide_MS2quant_None_limma_pvalues.tsv` — R limma output
    - `Data/Processed/20260818_peptide_MS2quant_None_transformed_limma_phPlus.tsv` — + limma + PhosphoSitePlus
    - dev sample: `data/20260818_peptide_MS2quant_None_transformed_nolimma_sample.tsv` (200 rows)
  - Notebooks: `01_preprocessing/LFQ_diaPASEF.ipynb`, `01_preprocessing/limma_for_pvalues_diapasef.rmd`,
    `02_qc/diaPASEF_QC.ipynb`
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
  transformations.py        Data transformation pipeline. Two parallel chains: `run_all_transformations`
                            (TMT, one cell line per call) and `run_diapasef_transformations` (diaPASEF /
                            LFQ, all cell lines in one pass — see "diaPASEF transformations" below).
                            Also `merge_limma_results`, `merge_functional_score`, `merge_phosphoplus_info`,
                            and `correct_channel_offsets` (TMT: removes the plex-to-plex scatter of each
                            channel's offset vs starve, keeps the systematic shift; applied to raw:abs
                            BEFORE `run_all_transformations` in the "Channel-offset correction" section
                            of `TMT_dataset_preprocessing.ipynb` — not inside `run_all_transformations`)
  filters.py                Row filtering: n:reps, contaminants (CON/REV), dynamics range, incomplete
                            time series, localization, by protein / by site, limma responsiveness
                            (`filter_by_ffdr` — omnibus F-test FDR, the reproducibility filter)
  QC.py                     Quality control functions (sections 1–5, see notebooks/02_qc/); section 5
                            reads the merged limma statistics (peak timing, responsive-site counts).
                            Replicate-coverage family (added for diaPASEF, works on any dataset):
                            `replicate_coverage_matrix` -> `replicate_coverage_depth` ->
                            `filter_by_coverage`, wrapped for the single-group case by
                            `subset_by_coverage`; plus `coverage_summary`, `coverage_per_timepoint`,
                            `plot_replicate_coverage`, `plot_coverage_per_timepoint`, `add_nreps_columns`.
                            Section 6 (2026-09-16) — temporal-profile correlation between cell lines /
                            datasets: `site_match_key` (cross-dataset key: accession + UniMod-free
                            phosphopeptide; the raw `site` string does not match across datasets),
                            `fully_localized_mask`, `deduplicate_by_key`, `profile_matrix`
                            (`center="median"` option), `common_timepoints`, `align_profiles`,
                            `global_profile_correlation`, `per_site_profile_correlation` (uncentered by
                            default; `shuffle=True` = shuffled-site null), `per_timepoint_correlation`,
                            and their plots
  plotting_functions.py     Visualisation: time series, cluster, protein profile plots; sites x timepoints
                            heatmaps (`plot_fc_heatmap`, `plot_step_heatmap`, `plot_fc_step_heatmap`);
                            per-protein fit panels (`plot_protein_sigmoid_fits`, `plot_protein_curvecurator_fits`)
  clustering.py             Clustering utilities (tslearn KMeans/KShape/KernelKMeans, HDBSCAN, quality metrics)
  adaptive_clustering.py    Adaptive divisive+agglomerative KMeans (inertia-driven split then centroid merge)
  hierarchical_clustering.py Agglomerative hierarchical clustering (scipy linkage): tree-shaped QC
                            (cophenetic correlation, merge-distance elbow, free k-scan) + constrained
                            merging of clusters that share a parental node
  lfq_pretreatment.py       LFQ-specific preprocessing (mutant datasets, DDA)
  lfq_diaPASED_pretreatment.py  diaPASEF-specific preprocessing (note the typo in the filename, it is
                            the name on disk): `add_site_identificator` builds the `site` key from the
                            FragPipe precursor column, `add_zscore_normalization`, STY-position helpers
  kinase_prediction.py      Kinase imputation via kinase_library: UniProt ±7 windows, top-5 kinase + percentile prediction (TMT & LFQ)
  cluster_composition.py    Cross-tabulate cluster labels vs annotations (which clusters contain sites of kinase X / protein Y)
  cluster_enrichment.py     Fisher's-exact per-cluster enrichment of kinases / motifs / metadata (odds ratio, log2 enrichment, BH-FDR)
  kinase_activity.py        KSEA kinase-activity inference (z-scores) across conditions × timepoints
  response_shapes.py        Model-free shape classification + descriptors + per-timepoint SEM
                            (`per_site_sem(target=...)` — see "The two scales" below)
  curve_fitting.py          Anchored logistic fit and T50; hard anchor (log2:FC) and soft anchor
                            (log2:mean, `free_baseline=True` — the recommended mode)
  curvecurator_io.py        CurveCurator input/output adapters (time-for-dose adaptation)
  r_utils.R                 **R** — shared infrastructure for every .rmd in 01_preprocessing:
                            `find_project_root`, `checkpoint`, `stage`, `report_memory`,
                            `muffle_warnings`, `say`. Verbosity via `options(tmt.verbose = ...)`
  limma_common.R            **R** — helpers identical in both limma notebooks: `sort_timepoints`,
                            `normalize_matrix`, `build_contrasts`, `order_by_datatype`,
                            `add_test_columns`
  limma_tmt.R               **R** — TMT limma (`limma_for_pvalues.rmd`), keyed on the **plex**:
                            `parse_abs_columns`, `deduplicate_controls`, `read_dataset`,
                            `run_limma_dataset`, `verify_within_plex`, `limma_output_path`
  limma_diapasef.R          **R** — diaPASEF limma (`limma_for_pvalues_diapasef.rmd`), keyed on the
                            **replicate**: the same four names as `limma_tmt.R` plus `site_coverage`,
                            `replicate_effect_report`, `run_limma_cell_line`,
                            `verify_contrast_definition`. ⚠️ never source both limma modules in one
                            session — each warns if the other is already loaded
  batch_correction_diapasef.R  **R** — `removeBatchEffect` helpers, sourced by BOTH drivers
                            (`batch_correction_diapasef.rmd` and its sibling `.R` script):
                            `output_path`, `read_tsv`, `canonicalise_sheet`, `split_matrix_columns`,
                            `to_log2_if_needed`, `build_bio_group`, `check_batch_orthogonality`,
                            `count_nonestimable`, `reference_spread`, `plot_reference_qc`
  trendy_patterns.R         **R**, not Python — helpers for the Trendy segmented-regression notebook
                            (`notebooks/01_preprocessing/trendy_temporal_patterns.rmd`): project-root /
                            input-file resolution, `stage()` progress reporting, `parse_condition_columns`,
                            `normalize_matrix` / `center_plex`, `build_time_vector` / `axis_to_minutes`,
                            `trendy_result_table`, `pattern_census`. No Trendy dependency, so it can be
                            sourced and tested on its own
  dynamic_rpcst.py          Dynamic EGF kinase-phosphosite network optimisation (dynamic RPCST) —
                            helpers for `notebooks/07_dynamics_RPCST/`: peak-FC site table,
                            kinase-substrate loading, TF terminals, candidate-graph construction,
                            the cvxpy ILP (`dynamic_rpcst_selection`), validation, output writing
                            and Graphviz rendering. Needs cvxpy + a MILP solver (GUROBI preferred)
  xgboost_model.py          ON HOLD — XGBoost cluster classifier + SHAP (see "XGBoost classifier" below)
  utils.py                  Shared utilities + older plotting/network helpers (partly superseded by
                            plotting_functions.py; kept for comparison)

notebooks/
  01_preprocessing/         Raw data processing and transformation
    TMT_dataset_preprocessing.ipynb   Transform WT TMT datasets; merge PhosphoSitePlus. Last section
                                      (2026-09-15): channel-offset correction of hme1_2, two variants
                                      written as 20260915_hTERT_HEM1_2_chcorr_{random,total}_* (random =
                                      default, total = sensitivity run), then limma merge + comparison
    LFQ_dataset_preprocessing.ipynb   Preprocess LFQ mutant datasets; extract site metadata; filter
    LFQ_diaPASEF.ipynb                hme1_diaPASEF preprocessing, end to end: sample-number → column-name
                                      mapping (8 cell lines × 9 timepoints × 3 reps + mix/mixb/mixc),
                                      `add_site_identificator`, `run_diapasef_transformations`, merge of the
                                      R limma table, merge of PhosphoSitePlus. Writes the 200-row dev sample
                                      used by the R notebook
    limma_for_pvalues.rmd             R/limma statistics for the TMT datasets (blocked on plex) — see
                                      "Limma statistics columns". Helpers in `src/r_utils.R`,
                                      `src/limma_common.R`, `src/limma_tmt.R`. Time is a factor (one
                                      mean per timepoint), so no shape/linearity is assumed — splines
                                      were considered and rejected (documented in the notebook).
                                      `RUN_DATASETS` picks which runs to fit (default: the two
                                      hme1_2 chcorr tables); `DATASET_FOLDERS` maps extra run labels
                                      to their Experiment/ folder
    limma_for_pvalues_diapasef.rmd    R/limma statistics for hme1_diaPASEF: per cell line, EGF only,
                                      unblocked by default — see "limma on diaPASEF" below. Helpers in
                                      `src/r_utils.R`, `src/limma_common.R`, `src/limma_diapasef.R`
    batch_correction_diapasef.rmd     R `removeBatchEffect` for hme1_diaPASEF — the "subtract for the
                                      eyes" track (clustering/PCA/plots), never for inference. Sibling
                                      script `batch_correction_diapasef.R` for Rscript/cron; both source
                                      `src/batch_correction_diapasef.R`, but their CONFIGURATION still
                                      differs (see the table at the top of the .rmd)
    trendy_temporal_patterns.rmd      R Trendy segmented-regression / breakpoint fitting (exploratory).
                                      Helpers live in `src/trendy_patterns.R`, sourced by the setup chunk
    REPORT_01_preprocessing.md        Folder report: what preprocessing does and why
    tps_file_creator.ipynb            LEGACY — TPS input formatter (old column naming, do not use)
  02_qc/                    Quality control
    General_QC.ipynb        Markdown reference — QC checklist organized by stage (not executable)
    MSMS_data_QC.ipynb      Executable QC (TMT): missing values, CV, PCA, UMAP, limma statistics
                            (peak timing + responsive sites), dataset overlap
    diaPASEF_QC.ipynb       Executable QC (hme1_diaPASEF): missingness, replicate coverage per cell line /
                            per timepoint, coverage-based filtering, localization completeness, intensity
                            distributions, CV, PCA (coverage-filtered subsets), temporal profiles
    TMT_channel_offsets.ipynb  Per-channel loading offsets vs starve (random vs systematic part),
                            replicate SD per timepoint before/after `correct_channel_offsets`, intensity
                            and timing-jitter checks, non-phospho loading control from FragPipe psm.tsv
    diaPASEF_profile_correlation.ipynb  hme1_diaPASEF: log2:FC profile correlation of every cell line vs WT
                            and between cell lines (sites with >= 1 rep at every timepoint in all 8
                            lines; BRAF duplicate = calibration). Profiles from the batch-corrected table,
                            `site` + FFDR from the uncorrected one (the corrected table's `site` has
                            uppercase sequences, so join on `peptide_index`)
    WT_datasets_profile_correlation.ipynb  WT hme1_1 vs hme1_2 vs hme1_diaPASEF on shared timepoints
                            (2, 5, 10, 90) and shared localized sites. ⚠ Per-timepoint median centring is
                            the default: uncentred, the per-site r mostly measures each dataset's shared shift
    QC_notes.md             Interpretation notes (e.g. PCA vs UMAP for replicate agreement)
    REPORT_02_qc.md         Folder report: QC results per dataset
    Results/                Exported QC figures, one dir per run: {YYYYMMDD}_QC_{dataset}_files/
  03_clustering/            Unsupervised clustering
    Clustering.ipynb                    WT k-scan and final clustering (TimeSeriesKMeans primary method)
    Adaptive_clustering.ipynb           Divisive+agglomerative adaptive clustering (src/adaptive_clustering.py)
    Adaptive_clustering_sweep.ipynb     2D threshold sweep + substructure diagnostics (margin, bootstrap stability)
                                        All three share one filtering chain: n:reps → |log2FC| →
                                        limma FFDR (see "Filtering chain" below)
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
    Heatmaps.ipynb               Sites × timepoints heatmaps: log2:FC, the derived log2:step
                                 (timepoint-to-timepoint change) and the two side by side
    profiles_difference.ipynb    Euclidean distance between temporal profiles; simplified profile plots
  05_downstream/            Downstream analysis (enrichment, kinase activity, classifier)
    kinase_prediction.ipynb                      Predict top-5 kinases + percentiles per site (uses src/kinase_prediction.py); adds cluster-composition demo (src/cluster_composition.py)
    cluster_enrichment_and_kinase_activity.ipynb Fisher enrichment + KSEA, heavily documented for defense (src/cluster_enrichment.py, src/kinase_activity.py)
    XGBoost_model.ipynb                          ON HOLD — cluster classifier + SHAP (src/xgboost_model.py)
    phosx_implementation.ipynb                   EXPLORATORY — PhosX kinase-activity inference, data pre-processing stage
    omnipath.ipynb                               STUB — OmniPath prior-knowledge network (1 cell)
    Protein_ratio_of_phosphorylation.ipynb       EXPLORATORY — phospho-site / protein-level ratio (4 cells)
    kinase_library_implementation.ipynb          LEGACY prototype — superseded by kinase_prediction.ipynb (kept for reference)
  06_sigmoids/              Parametric response-shape modelling and T50
    Sigmoid_fitting.ipynb              Shape census + anchored-sigmoid fit + T50 (src/response_shapes.py,
                                       src/curve_fitting.py). Classifies on log2:FC, fits on log2:mean
                                       with a free baseline — see "The two scales" below
    Sigmoid_fitting_CurveCurator.ipynb CurveCurator route (src/curvecurator_io.py); the package is not
                                       installed, so no real fit has run
    Data_fitted/                       Saved fit tables, {RUN_NAME}_fits.tsv
  07_dynamics_RPCST/        Dynamic network optimisation (upstream of the ODE/mechanistic model)
    dynamic_network_optimization_tutorial.ipynb
                            End-to-end tutorial: EGF peak-FC site table -> kinase-phosphosite
                            candidate graph (ROOT->EGFR ... TF site->SINK) -> temporally
                            constrained rooted prize-collecting Steiner ILP -> validation ->
                            Graphviz. All functions live in `src/dynamic_rpcst.py`; the parameter
                            cell carries editable copies of every module default. **Section 12
                            "Observations" records a 2026-09-15 code/method review** — read it
                            before trusting a selected network
    collaborator_dynamic_rpcst_cost_1p2.*
                            Saved run at NODE_COST=1.2 (graph pkl, node/edge CSVs, summary JSON,
                            TF list, DOT/SVG/PNG)
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
{CellLine}_{DataType}_{Treatment}_{TimePoint}_{Replicate}
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
  - `wtFC` — (diaPASEF only) fold change relative to the **WT** starve: `log2:mean(X, t) − log2:mean(WT, starve)`,
    written for every timepoint incl. `full` and `starve`. Equals `FC` for WT; for a mutant its `starve` column
    is the basal offset vs WT. `dia_compute_wt_fold_change()`
  - `scaled` — FC scaled so amplitude is in [−1, 1] and starve = 0
  - `zscore` — per-site z-score of the FC temporal profile, standardised across the time series **per condition per cell line** (mean 0, std 1); isolates response shape from amplitude
  - `step` — change relative to the **previous** timepoint, `log2:FC(t_i) − log2:FC(t_i−1)`, chained from starve and named after the timepoint it arrives at (`log2:step_EGF_5` = the 2 → 5 min change). No column for `full` or `starve`. `log2_step_size()` (TMT) / `dia_log2_step_size()` (diaPASEF)
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

### Peak-timing columns

`add_peak_timepoints()` (`src/transformations.py`, shared by both preprocessing notebooks — it matches
the cell line on its exact field, so it is safe on the 8-cell-line diaPASEF table) labels each site
with **where** its profile peaks:

| Column | Meaning |
|---|---|
| `{cell_line}_peak:FC_{condition}` | timepoint at which \|log2:FC\| is largest — when the site is furthest from its starve baseline |
| `{cell_line}_peak:step_{condition}` | timepoint at which \|log2:step\| is largest — the interval in which the site changed most |

- Ranked on the **absolute** value, so a downregulated site peaks at its deepest point and the largest
  negative step counts. The direction is not recoverable from these columns — read the underlying
  `log2:FC` / `log2:step` for the sign.
- The value is the timepoint **label as a string** (`'2'`, `'90'`), NaN when the profile is entirely
  missing; ties go to the earliest timepoint. `full` and `starve` are excluded by default.
- These have no timepoint field, so they do **not** follow the
  `{CellLine}_{DataType}_{Treatment}_{TimePoint}` scheme and `ColumnSpec.select()` ignores them — flat
  per-site annotations, like the kinase-prediction columns.
- Related but not the same as QC section 5's `limma_peak_timepoints()`, which ranks `|limmaFC|` and
  counts only limma-responsive sites. `peak:FC` is computed for **every** site from the mean-based
  `log2:FC`, so on a non-responder it is the argmax of noise. On sparse diaPASEF sites it is the peak
  of *what was measured*, not of the profile.

### diaPASEF transformations — `run_diapasef_transformations()`

Parallel to `run_all_transformations` (which is untouched), for `hme1_diaPASEF`. It exists because
three things differ: `_sort_timepoints()` sorts against a fixed `_TP_ORDER` with no 20 or 30 min (the
diaPASEF grid would come out `… 15, 90, 20, 30`, so `_sort_timepoints_numeric()` sorts numerically
instead); all 8 cell lines are processed in one pass with each column assigned by its **exact** first
field (`BRAFS151A1` vs `BRAFS151A2` cannot pull each other in, and `MIX_*` is never selected); and the
~1500 new columns are attached with one `pd.concat` per block rather than column by column.

Chain: `raw:mean/median/sd/cv` → `log2:abs` (zeros → NaN) → `log2:mean/median/sd` → `log2:FC` (vs
starve; **no starve, no FC — the site stays NaN**, decided per site *per cell line*) → `log2:scaled` →
`log2:zscore`.

**The normalisation basis is the one deliberate difference from the TMT functions.** `compute_scaled_fc`
/ `compute_zscore_fc` standardise over every timepoint present, `full` included — the defect recorded
in `clustering_method_decision.md` §1. The diaPASEF versions take the basis as an argument and exclude
`full` by default (and `starve` too for the z-score, where it is a structural zero):

| argument | default | why |
|---|---|---|
| `exclude_from_scale` | `("full",)` | full media is not a response; its \|FC\| is routinely the row maximum, so as denominator it squashes the actual response |
| `exclude_from_zscore_basis` | `("full", "starve")` | `full` carried ~26% of the clustering variance on hme1_2; `starve` is identically 0 in FC space |
| `min_zscore_timepoints` | `3` | a sd over 1–2 points gives ±1 by construction |

**`log2:wtFC`** (added 2026-09-17, `dia_compute_wt_fold_change()`, own section in `LFQ_diaPASEF.ipynb` after
`log2:step`; not part of `run_diapasef_transformations`) puts every cell line on the WT starve baseline.
`log2:scaled` / `log2:zscore` / `log2:step` are still built from the own-starve `log2:FC`. Measured on the
batch-corrected table: 20,837 of 71,087 sites have no WT starve (NaN in every cell line), and ~5,700–8,000 per
mutant have their own starve but no WT starve. The median mutant starve offset is −0.47 to −0.01 log2
(no between-cell-line normalisation, so this includes loading).

Columns are still written for **every** timepoint — only the basis changes. So `log2:scaled` at `full`
may exceed 1, and `log2:zscore` at `starve` reads as how far the baseline sits below the mean response,
in SDs. Passing `exclude_from_scale=()`, `exclude_from_zscore_basis=()`, `min_zscore_timepoints=1`
reproduces the TMT functions exactly (verified numerically).

### Limma statistics columns

`notebooks/01_preprocessing/limma_for_pvalues.rmd` (R) fits `log2:abs ~ group + plex` per site and its
output is merged onto the processed tables by `merge_limma_results`. Two families:

| Column | Test |
|---|---|
| `WT_log2:limmaFC_EGF_5` / `pvalue` / `FDR` / `adjustedFDR` | moderated *t*, one timepoint vs starve; **BH is applied within each contrast column, across sites** |
| `WT_log2:Fpvalue_EGF_omnibus` / `FFDR` / `adjustedFFDR` | omnibus moderated *F*: that condition's timepoint contrasts tested **jointly** against starve |
| `WT_log2:FFDR_ALL_omnibus` | the same *F* over all 15 stimulation contrasts of all three conditions |

Two facts verified against the R source, both easy to get wrong:
- The omnibus *F* is built over `setdiff(timepoints, c("full", "starve"))`, so **`full` is already
  excluded from it**. An `exclude_full` argument cannot change an omnibus result.
- OR-ing the per-timepoint FDR columns ("responsive at any timepoint") has **no joint error control**
  — each column is BH-corrected on its own, so the union's false-positive rate grows with the number
  of timepoints, and the count is not comparable between datasets with different timepoint grids.
  The omnibus is one test either way. Full write-up in the `MSMS_data_QC.ipynb` limma section.

⚠ **Never `.fillna(0)` a whole DataFrame that carries these columns.** A site limma could not test
carries NaN in `FFDR`; filling it with 0.0 makes it *perfectly significant* and every untested site
survives the responsiveness filter. Fill the data columns only (both adaptive-clustering notebooks
now do this explicitly in their load cell).

### limma on diaPASEF — `limma_for_pvalues_diapasef.rmd`

Same column names and the same three families of statistics, so everything downstream
(`merge_limma_results`, `filter_by_ffdr`, QC section 5) works unchanged. Four things differ, all
driven by the experiment rather than by preference:

| | TMT (`limma_for_pvalues.rmd`) | diaPASEF (`limma_for_pvalues_diapasef.rmd`) |
|---|---|---|
| unit | 3 datasets, 1 cell line each | 1 dataset, **8 cell lines**, one fit each, merged on `site` (outer join) |
| conditions | EGF / INS / EGFnINS | EGF only — `..._ALL_omnibus` is **identical** to `..._EGF_omnibus`, kept for naming compatibility |
| design | `~ 0 + group + plex` — the plex is a **physical** block | `~ 0 + group` by default (`BLOCK_ON_REPLICATE`), because the runs are independent injections in randomised order |
| site filter | `n:reps >= MIN_PLEX` | observation counts — the diaPASEF table has **no `n:reps` column** |

Four consequences worth remembering:

- **`MIN_OBS` is derived, not chosen.** `df.residual = n_obs − rank(design)` and rank ≤ `ncol(design)`,
  so requiring `n_obs ≥ ncol(design) + MIN_RESIDUAL_DF` guarantees the residual df; the notebook
  asserts it after the fit. `MIN_CONTROL_OBS` separately requires the starve control to be observed.
- **The omnibus F is NA whenever any timepoint was never observed** for that site — limma builds it
  from the vector of contrast *t*-statistics and one NA propagates to the whole F. Per-timepoint *t*
  contrasts of the same site stay valid. NA here means *not tested*, never *not significant*.
- **Under the default unblocked design `limmaFC` is numerically the same estimator as the pipeline's
  `log2:FC`** (verified: r = 1.0000, max diff 1e-14, every site). limma adds the variance model, not a
  different fold change. Under `BLOCK_ON_REPLICATE = TRUE` they agree only on fully observed sites.
- **The replicate term is an empirical question here, and it is not empty.**
  `replicate_effect_report()` runs in both modes and tests per site whether the replicate offsets are
  zero. On the 200-row dev sample: significant at FDR < 0.05 in 0% (BRAFS151A1) to 73% (RPS6KA3S375A)
  of sites, median largest offset **0.47–0.85 log2** — the order of the biological fold changes — and
  `NORMALIZE = "median"`/`"quantile"` barely moves it (WT 0.585 → 0.537 → 0.481), so it is batch
  structure rather than loading. Unblocked puts that variation in the residual (conservative); blocked
  removes it but assumes additive batch offsets and costs 2 df. **Re-measure on the full file and run
  both ways before settling.**

### Filtering chain — the three clustering notebooks

`Clustering.ipynb`, `Adaptive_clustering.ipynb` and `Adaptive_clustering_sweep.ipynb` run the **same
three filters in the same order**, so their partitions stay ARI-comparable:

1. `filter_by_nreps(min_reps=MIN_REPS)`
2. `filter_dynamics(threshold=MIN_FC, mode="extremes")` — amplitude only, says nothing about
   reproducibility (and see "Known issues": on unnormalised data it currently passes ~100% of sites)
3. `filter_by_ffdr(max_ffdr=MAX_FFDR, combine="any")` — limma omnibus *F*; the reproducibility
   criterion step 2 cannot give. `MAX_FFDR = None` skips it, so no `if` is needed around the call.

`filter_by_ffdr` builds `{cell_line}_log2:FFDR_{condition}_omnibus` from the `CELL_LINES` ×
`CONDITIONS` product. `combine="any"` is the union over the clustered conditions; `combine="all"`
requires all of them; `conditions=["_ALL_"]` resolves to the single `..._ALL_omnibus` column, which
is the stricter choice when several conditions are clustered jointly (one test instead of a union of
three). Sites limma never tested carry NaN and are always dropped.

For the sweep notebook this is not cosmetic: it *chooses* `INERTIA_THRESHOLD` and
`MERGE_DISTANCE_THRESHOLD`, which are absolute distances in the clustered space, so tuning them on a
different (noisier) site set than the one they are applied to transfers numbers that no longer mean
the same thing.


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
- Row-filtering module (`src/filters.py`): n:reps, contaminants, dynamics range, incomplete time series, localization, by protein / by site, **limma responsiveness (`filter_by_ffdr`)**
- limma results are read in two places now: QC section 5 in `src/QC.py` (`limma_peak_timepoints`, `limma_responsive_sites`) describes them, `filter_by_ffdr` filters on them. Both share the omnibus-vs-per-timepoint distinction documented in `MSMS_data_QC.ipynb`.
- XGBoost cluster classifier + SHAP (`src/xgboost_model.py`, `XGBoost_model.ipynb`) — runs, but **direction on hold**; see the flagged section above. Bio-metadata features do not predict cluster (macro-F1 0.21); temporal features are leaky by design.
- Hierarchical clustering (`src/hierarchical_clustering.py`, `Hierarchical_clustering.ipynb`) — tree built once and cut afterwards, so the k-scan is near-free; tree-level QC (cophenetic correlation, merge-distance elbow, per-cluster silhouette, nearest-centroid agreement) plus **constrained merging** of clusters sharing a parental node. Notebook verified end-to-end on hme1_2.

- Clustering method decision document (`notebooks/03_clustering/clustering_method_decision.md`) — KMeans vs hierarchical, transformations, QC/statistics, sigmoid/T50 and impulse modelling; includes the four verified preprocessing defects and the measured null cluster-switching rate.

- diaPASEF pipeline (`hme1_diaPASEF`): `run_diapasef_transformations()` (raw/log2 statistics → FC →
  scaled → zscore, all 8 cell lines in one pass), `src/lfq_diaPASED_pretreatment.py`, the replicate-coverage
  QC family in `src/QC.py`, and `limma_for_pvalues_diapasef.rmd`. See "diaPASEF transformations" and
  "limma on diaPASEF"

**In progress / pending:**
- **Preprocessing fixes — highest priority, blocking everything downstream.** Normalisation, the inert dynamics filter, the z-score basis, and the on-disk `fillna(0)` (see "Known issues"). Every clustering result to date was computed on unnormalised data through an inert filter.
- Clustering optimisation (parameter tuning, optimal number of clusters) — **method recommendation now made**: KMeans (`n_init` ≥ 25) per condition on a corrected z-score, with Ward redeployed to the centroid-level tree. k chosen by stability + interpretability + biological content, reported as a profile across k rather than a single value. See `clustering_method_decision.md` §6.
- Transfer of WT cluster labels to the mutant datasets — **decision now made in favour of nearest-centroid**, but *soft* (distance vector + assignment margin, calibrated against a WT split-half null), on shared timepoints only, in an amplitude-free representation, and **within a single experiment/platform** — never hme1_2 TMT centroids onto LFQ mutant data (10× dose + platform + grid confound). The XGBoost route stays on hold. **Prerequisite: measure the null switching rate first** — it is 38–66% within WT.
- Establish the analysis nulls before any mutant claim: split-half switching rate, mutation positive control (does the pipeline recover its own mutations?), XGBoost label-noise ceiling, gap statistic / dip test / bootstrap Jaccard. See `clustering_method_decision.md` §12 Phase 3.
- Parametric curve fitting — **built and running** (`src/response_shapes.py`, `src/curve_fitting.py`, `notebooks/06_sigmoids/Sigmoid_fitting.ipynb`): shape classification + model-free descriptors, then the anchored log-time sigmoid on monotonic/sustained sites, soft-anchored on `log2:mean` (see below). Still pending: the exponential-difference model for transient sites, and the impulse model — **hme1_lfq has now landed** (7 post-stimulation points, 2 lack-of-fit df), so the model is fittable there; it stays untestable on hme1_2 (6 points, 5 parameters = 1 lack-of-fit df, and it is **not** rescued by the mean scale).
- Downstream analysis: PhosX integration — `notebooks/05_downstream/phosx_implementation.ipynb` is at the data-pre-processing stage; `PhosX/` holds a seqrnk input and output dir (Fisher enrichment + KSEA now done)
- `omnipath.ipynb` (prior-knowledge network) and `Protein_ratio_of_phosphorylation.ipynb` are stubs — not started
- **hme1_diaPASEF (hme1_lfq) is in.** Preprocessing, transformation, R/limma statistics and QC all run end
  to end (`LFQ_diaPASEF.ipynb`, `limma_for_pvalues_diapasef.rmd`, `diaPASEF_QC.ipynb`). Open decisions on it:
  (a) `NORMALIZE` — the limma notebook currently fits the data as stored, matching the Python pipeline;
  (b) `BLOCK_ON_REPLICATE` — unblocked by default, but the measured replicate offsets are not negligible
  (see "limma on diaPASEF"); (c) no clustering / downstream analysis has been run on it yet
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

### Session 2026-08-11 — soft-anchoring the sigmoid fit (the two scales)

Implemented `Claude_promts/CURVE_FITTING_INSTRUCTIONS.md`. A delta against working code, not a
rewrite: `anchored_sigmoid`, the `log10(t+1)` axis, `DEFAULT_FIXED_K = 10`, the shape gate, the
chi-square-difference rationale and `fit_quality_gates` are all untouched. Nothing was replaced,
so nothing needed commenting out — every change is additive with a backwards-compatible default.

**The two scales — the thing to remember.** The notebook now **classifies on `log2:FC` and fits
on `log2:mean`.**

| consumer | profile | SEM |
|---|---|---|
| `shape_descriptors` | `log2:FC` | — |
| `classify_response_shape` | `log2:FC` | `per_site_sem(target="FC")` |
| `fit_sigmoid_*` | **`log2:mean`** | **`per_site_sem(target="mean")`** |
| `center_timepoint_medians` | `log2:FC` only | — |

The model gained a free baseline, `y(x) = y0 + A[sigma(k(x-x50)) - sigma(-k*x50)]`
(`soft_anchored_sigmoid`, `fit_sigmoid_site(free_baseline=True)`). **The df ledger is neutral** —
fitting the mean adds one observation (starve stops being a dropped structural zero) *and* one
parameter. The argument is **uncorrelated errors**: every `log2:FC` timepoint carries the same
`-mean(starve)` term, so the diagonal `1/SEM` weighting was assuming an independence the FC scale
does not have. On the mean scale it is correct rather than approximate, and the starve replicates
start doing work. This answers the standing objection in `response_shapes.py` about losing the
baseline. **It is statistical honesty, not accuracy** — report it that way.

**Measured on hme1_2 (WT, EGF, 1500 sites fitted both ways, k = 10)** — the validation is in the
notebook and is a *negative* result by design:
- median |ΔT50| **0.153 min = 4.7% of the median per-site SE (3.25 min)**; 74% agree within 0.1·SE
- Spearman(T50) **0.981** on sites above 3 min, 0.917 on [2,60]; **0.800 overall — censoring, not
  disagreement** (40% of sites pile against the 2 min left-censoring bound, where ranks are noise)
- Spearman: A 0.947, plateau 0.972, RMSE 0.979, SE(T50) 0.957
- |ΔT50| correlates with SEM(starve), ρ = 0.215, p = 4e-17; noisy half 0.185 min vs 0.129 min for
  the well-measured half — exactly the predicted pattern, and the one plot that argues for the change
- ⚠️ **expect the lack-of-fit test to get stricter**: not-rejected falls 45.3% → 20.5%, because the
  FC SEM double-counts the starve variance (median 0.113 vs 0.082) and inflated errors flatter any
  model. Median chi-square 8.28 → 14.43 on residuals whose median RMSE *improves*, 0.139 → 0.131.

**`compare_to_flat` — the highest-risk change.** On the mean scale the null is a **fitted constant**
`y = y0` (one free parameter, weighted RSS about the weighted mean over *all* timepoints), not the
exact curve `y = 0`. Test df becomes `n_free_sigmoid - 1`; `bic_flat` is charged its parameter.
Verified: 400 pure-noise sites flat at 18.4 give **0% significant** under the correct null and
**100%** under the zero null. `compare_to_flat` takes `free_baseline` explicitly for this reason —
it must match the flag the fits were made with.

**Defaults re-chosen from measurement, not inherited** (both change existing output slightly):
- `per_site_sem(prior_df=)` **4.0 → 2.0**. The prior's weight is `prior_df/(prior_df+d)`, `d = n-1`:
  at prior_df 4 that is 57% at n=4 and **80% at n=3**, and a third of hme1_2 counts are n=3. Swept
  {0,1,2,4,8}: % sigmoid-legitimate 44.0 / 46.5 / 47.1 / 47.1 / 46.4 — rises to 2 then **flat**, so
  2.0 sits on the plateau at 40%/50% prior weight. `prior_df=0` is asserted identical to
  `moderate=False`.
- `per_site_sem(min_n=)` **2 → 3**. An n=2 SD has 1 df — fine in a limma test, poor as a fit weight.
  Looks expensive (32.6% → 54.1% of SEM cells blanked across all 50002 sites) and is not: among
  responsive sites every per-timepoint count is 3 or 4 (one cell at 2), and all 7816 shape-gated
  sites keep all six timepoints either way. The discarded cells never reach the fit.
- `Sigmoid_fitting_CurveCurator.ipynb` now pins both explicitly, so its census does not drift silently.

**Also added:** `starve_level_summary()` (mean-scale replacement for the structural-zero warning,
which is now gated on `data_type`); `replicate_count_table()` (per-timepoint replicate counts +
an `uneven` flag — the per-site `n:reps` filter cannot see *where* the replicates are; on hme1_2
only 0.02% of sites are uneven); `PARAM_NAMES_SOFT`; `y0`/`se_y0`/`free_baseline`/`y0_null`/
`df_vs_flat` columns on the fit table. `plot_protein_sigmoid_fits` and `plot_fit_examples` read
`y0` (defaulting to 0.0, so old tables still draw correctly) and take a `value_label`.

**Deliberately NOT implemented** (both refusals are correct and stand): replicate-level fitting of
`log2:abs`, and the 5-parameter impulse model.

### Session 2026-08-13 — limma QC section, and the FFDR filter across the clustering notebooks

**`localization_completeness()` rewritten** (`src/QC.py`). The printed report is now the plain split
over all phospho-peptides — all phosphosites localized vs not, with the incomplete half broken into
"none localized" and "some not localized". Left panel plots exactly that; right panel plots the
fully-localized count per `n:reps` group (1/2/3/4) with the group total behind it as a grey bar, so
each bar reads as both a count and a fraction. The old `NumPhos`-split panel and the `max_numphos`
argument are gone. Measured on hme1_2 (8723 sites): 67.7% fully localized, 31.2% nothing localized,
1.1% partial; fully-localized rate rises with replication (58.4% / 66.8% / 78.9% at n:reps 2/3/4).

**New QC section 5 — limma statistics** (`src/QC.py`, wired into `MSMS_data_QC.ipynb`):
- `limma_peak_timepoints()` — bar plot of how many sites peak at each timepoint. Peak = largest
  **|limmaFC|**, so a minimum counts as a peak; `conditions=` takes one token or several (grouped
  bars); `exclude_full`, `min_fc`, and `split_direction` (stack up- vs down-regulated peaks). Counts
  only sites limma calls responsive — a non-responder still has a largest-|FC| timepoint, driven by
  noise, and including those buries the timing distribution under a flat background.
- `limma_responsive_sites()` — responsive sites per condition + an "any condition" bar, and a second
  panel counting how many conditions each site responds in (0…3), so the `0` bar is the unresponsive
  remainder. Prints percentages against both denominators (all sites / sites limma tested).
- Shared internals: `_responsive_mask()` (omnibus vs any_timepoint, optional `min_fc`) and
  `_order_timepoints()` (full, starve, then numeric — so 90 sorts after 15).
- The notebook carries a long markdown section on **`"omnibus"` vs `"any_timepoint"`**, checked
  against the R source. See "Limma statistics columns" above for the two facts worth remembering.

**`filter_by_ffdr()` added to `src/filters.py`** and applied in all three clustering notebooks
(`Clustering.ipynb`, `Adaptive_clustering.ipynb`, `Adaptive_clustering_sweep.ipynb`), replacing the
inline block that previously existed only in `Clustering.ipynb`. See "Filtering chain" above for the
API and the ordering. Both adaptive notebooks additionally needed:
- their load path moved from the `20260714` file to **`20260807`**, the one limma was merged into —
  the older file has no omnibus columns at all, so the filter could only ever raise. Old load lines
  kept commented. **Their stored outputs are therefore stale** (`Adaptive_clustering.ipynb` still
  shows 33719 → 12249 sites and a 25→44→32 cluster run from the old input); re-run before reading.
- their whole-frame `.fillna(0)` restricted to the non-statistics columns. Measured on a 100-row dev
  merge: whole-frame fill keeps **77** sites through the filter, statistics-as-NaN keeps **47** — the
  30 difference is exactly the never-tested set being waved through as FFDR = 0.0.
- `MAX_FFDR` added to the sweep's saved `params_record`, since it changes the input site set.

### Session 2026-08-18 — diaPASEF: scaled/zscore transforms, and limma for the LFQ design

**`run_diapasef_transformations()` now ends at `log2:zscore`, not `log2:FC`.** Added
`dia_compute_scaled_fc()` and `dia_compute_zscore_fc()` (+ the `_dia_fc_columns()` helper) in the
diaPASEF section of `src/transformations.py`, wired into the runner with three new arguments
(`exclude_from_scale`, `exclude_from_zscore_basis`, `min_zscore_timepoints`). Full rationale and the
defaults table: "diaPASEF transformations" above. Verified on a synthetic 3-cell-line frame with 15%
missing values: basis mean 0 / sd 1, `max|scaled| = 1` over the non-`full` timepoints, no-starve sites
NaN throughout, and — with the exclusions emptied — **max abs difference 0.0 against
`compute_scaled_fc` / `compute_zscore_fc`**. The notebook markdown in `LFQ_diaPASEF.ipynb` was updated
with the two new chain steps and the basis argument.

**`notebooks/01_preprocessing/limma_for_pvalues_diapasef.rmd`** — the diaPASEF sibling of the TMT limma
notebook, verified end to end on the 200-row dev sample (8 cell lines, ~1 s, writes
`data/…_sample_limma_pvalues.tsv`; a full run writes into `Experiment/hme1_diaPASEF/Data/Processed/`).
Statistics, column names and the checkpoint/warning machinery are carried over unchanged; see "limma on
diaPASEF" above for the four differences and their consequences.

The design question was settled during the session and is worth recording, because it inverts the TMT
reasoning: **the TMT plex is a physical block** (a channel can only be compared to the starve of its own
plex), whereas the diaPASEF runs are independent injections loaded sequentially in randomised order, so
nothing pairs a timepoint to a particular replicate's control. Default is therefore `~ 0 + group`, with
`BLOCK_ON_REPLICATE = TRUE` available and `replicate_effect_report()` measuring what the term would
absorb in either mode. Verification adapts to the design: unblocked checks `limmaFC == mean(observed t)
− mean(observed starve)` on **every** tested site; blocked checks the within-replicate mean on fully
observed sites only (where the two definitions coincide). Both pass at ~1e-15.

Also measured and recorded in the notebook: the per-run `log2:abs` median span (0.92–3.63 log2 within a
cell line on the sample — LFQ has no reporter-ion multiplexing, so `NORMALIZE = "none"` is a more
exposed choice here than in TMT), and the `Partial NA coefficients` warning is the only one the sample
produces. One silent approximation is documented rather than warned: with NAs, limma returns
`cov.coefficients` computed from the **complete** design, so the omnibus F's between-contrast
correlation is the one it would have had with complete data — read the F as a screening statistic.

**Deliberately not done:** a joint model over all 216 runs (`group = cell line × timepoint`). It would
buy residual df and is the right design for *mutant vs WT* contrasts, but it assumes one residual
variance per site across all eight cell lines and is not needed for "does this site respond to EGF,
within this cell line?". Per-cell-line fitting is not df-starved: 27 runs against 9 coefficients leave
18 residual df.

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
  **Update 2026-09-15 (full table, `02_qc/TMT_channel_offsets.ipynb`):** the offset of each channel vs the
  starve of its own plex has two parts. The *random* part (plex-to-plex scatter, up to ±0.23 log2, largest
  for `full` and EGF 2 min) is technical and is what made the replicate SD differ between timepoints;
  `correct_channel_offsets()` removes it and the SD becomes nearly flat (EGF: 0.11–0.14 at every
  timepoint). The *systematic* part is half the size the 8723-site subset suggested (EGF 5 min +0.29, not
  +0.57) and is partly carried by responding sites, so a plain median normalisation would subtract
  biology. Every timepoint sat in the same TMT channel in every plex, so loading vs a genuine global
  phosphorylation increase can only be settled with a loading control (non-phospho peptides, section 6).
  Section 7 (median vs mode) shows the *whole* distribution slides (mode ≈ median in every channel), so
  the phospho table alone cannot decide it. The hme1_2 raw table was produced by the FGCZ (no local
  FragPipe output, no `psm.tsv`; obtainable on request) and no proteome aliquot was measured.
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
  stimulation timepoints only, excluding both `full` and `starve`. **Fixed on the diaPASEF side only**
  (2026-08-18): `dia_compute_zscore_fc` / `dia_compute_scaled_fc` take the basis as an argument and exclude
  `full` (and `starve`) by default — see "diaPASEF transformations". The TMT `compute_zscore_fc` still
  defaults to `exclude_full=False`, so every existing TMT clustering result carries this defect.
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
- Never `.fillna(0)` a whole DataFrame that carries limma statistics columns (`pvalue`, `FDR`, `FFDR`, `limmaFC`, ...) — a site limma could not test carries NaN, and 0.0 reads as *perfectly significant*, so every untested site passes any FDR filter. Clustering may fill the **data** columns; the statistics columns must stay NaN
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