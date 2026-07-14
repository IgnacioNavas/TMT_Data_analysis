# Project: The MAPK/ERK signaling pathway crosstalk and regulation at phosphoproteome level

## Project overview

This project aims to understand how cells use phosphorylation of Serine, Threonine and Tyrosine
residues to transduce signal information and generate an adequate cellular response. The main focus
is on the regulation of the MAPK/ERK signaling pathway. For this I have information about cells being stimulated with
EGF, insulin and the combination of both. The cell lines I have been working with are hTERT-HME1 and HEK293T cell lines.

Cells grow in full media with growth factors and supplements (full). In this media, signaling reaches
an equilibrium state in which cells can proliferate with basal levels of signaling. To synchronize
cells and reduce basal signaling, full media is replaced by media without growth factors or
supplements for 2 hours (starve). After starvation, cells are stimulated with EGF, insulin (INS),
or a combination of both (EGFnINS), and lysates are collected at multiple time points. Two controls
are also collected: cells in full media, and cells that were starved but not stimulated.

These lysates are processed for phosphoproteomics using two protocols:
- **TMT-LC-MS/MS** — Tandem mass tag liquid chromatography mass spectrometry. Here data dependent aquisition (dda) method was used
- **LFQ-LC-MS/MS** — label free quantification liquid chromatography mass spectrometry. Here a for a small dataset I used "dda" as aquisition method. 
Later, for the big dataset I used the "dda" data to optimize a diaPASEF method that would improve the phosphosites identifications.

In addition to experients carried out with wild type cell lines, regulatory S/T/Y residues in the MAPK/ERK pathway were mutated
to Alanine (which cannot be phosphorylated) to disrupt negative feedback regulation. These mutant
cell lines undergo the same starvation and stimulation protocol, but in this case I only stimulated with EGF. For mutant cell lines I also have full and starve controls.

## Datasets available

Own datasets:
- hme1_1: 
  - Path: 
  - Cell lines: hTERT-HME1 wild type
  - Controls: Full and starve 
  - Stimulation conditions: 4.7 nM EGF, 10 nM insulin, co-stimulations with EGF and insulin
  - Time points: 1, 2, 5, 10, 90 minutes
  - Mass spectrometry protocol: TMT-LC-MS/MS
  - Acquisition method: data dependent acquisition
  - Number of replicates: 4
- hme1_2: 
  - Path: 
  - Cell lines: hTERT-HME1 wild type
  - Controls: Full and starve 
  - Stimulation conditions: 1.57 nM EGF, 100 nM insulin, co-stimulations with EGF and insulin
  - Time points: 2, 5, 10, 15, 90 minutes
  - Mass spectrometry protocol: TMT-LC-MS/MS
  - Acquisition method: data dependent acquisition
  - Number of replicates: 4
- hek_1: 
  - Path: 
  - Cell lines: HEK293T wild type
  - Controls: Full and starve 
  - Stimulation conditions: 1.57 nM EGF, 100 nM insulin, co-stimulations with EGF and insulin
  - Time points: 2, 5, 10, 15, 90 minutes
  - Mass spectrometry protocol: TMT-LC-MS/MS
  - Acquisition method: data dependent acquisition
  - Number of replicates: 4
- hme1_mutants_test:
  - Path: 
  - Cell lines: hTERT-HME1 wild type, hTERT-HME1 BRAF-S151A mutant, hTERT-HME1 GAB1-Y259A mutant
  - Controls: starve 
  - Stimulation conditions: 0.157 nM EGF
  - Time points: 2, 10, 25 minutes
  - Mass spectrometry protocol: LFQ-LC-MS/MS
  - Acquisition method: data dependent acquisition
  - Number of replicates: 2
- hme1_lfq: data acquisition on going
  - Path: 
  - Cell lines: 
    - 1 - hTERT-HME1 wild type, 
    - 2 - hTERT-HME1 EGFR-T693A mutant, 
    - 3 - hTERT-HME1 BRAF-S151A mutant (biological replicate 1),
    - 4 - hTERT-HME1 SOS1-S1178A mutant, 
    - 5 - hTERT-HME1 SHOC2-T71A mutant, 
    - 6 - hTERT-HME1 BRAF-S151A mutant (biological replicate 2),
    - 7 - hTERT-HME1 GAB1-Y259A mutant, 
    - 8 - hTERT-HME1 RPS6KA3-S375A mutant,
  - Controls: starve, full 
  - Stimulation conditions: 0.157 nM EGF
  - Time points: 2, 5, 10, 15, 20, 30, 90 minutes
  - Mass spectrometry protocol: LFQ-LC-MS/MS
  - Acquisition method: data independent acquisition diaPASEF, with optimiced windows with dda data from hme1_mutants   
  - Number of replicates: 3

The software for identification and quantification is FragPipe

External datasets:
- MCF10A EGF time course, TMT-LC-MS/MS, timepoints: 2, 4, 8, 12 min. from Feng Song et al. is available in `External_Data/` 


## Repository structure

```
src/                        Python modules
  column_spec.py            ColumnSpec class — the standard way to select data columns
  transformations.py        Data transformation pipeline (run_all_transformations)
  QC.py                     Quality control functions (sections 1–4, see notebooks/02_qc/)
  plotting_functions.py     Visualisation: time series, cluster, protein profile plots
  clustering.py             Clustering utilities (tslearn KMeans/KShape/KernelKMeans, HDBSCAN, quality metrics)
  adaptive_clustering.py    Adaptive divisive+agglomerative KMeans (inertia-driven split then centroid merge)
  lfq_pretreatment.py       LFQ-specific preprocessing (mutant datasets)
  utils.py                  Shared utilities

notebooks/
  01_preprocessing/         Raw data processing and transformation
    TMT_dataset_preprocessing.ipynb   Transform WT TMT datasets; merge PhosphoSitePlus
    LFQ_dataset_preprocessing.ipynb   Preprocess LFQ mutant datasets; extract site metadata; filter
    tps_file_creator.ipynb            LEGACY — TPS input formatter (old column naming, do not use)
  02_qc/                    Quality control
    General_QC.ipynb        Markdown reference — QC checklist organized by stage (not executable)
    MSMS_data_QC.ipynb      Executable QC: missing values, CV, PCA, UMAP, dataset overlap
  03_clustering/            Unsupervised clustering
    Clustering.ipynb            WT k-scan and final clustering (TimeSeriesKMeans primary method)
    Adaptive_clustering.ipynb   Divisive+agglomerative adaptive clustering (src/adaptive_clustering.py)
    clustering_overview.md      Reference document for all clustering strategies and parameters
  04_visualization/         Time series and profile plots
  05_downstream/            Downstream analysis (pathway enrichment, classifier)
  scratch/                  Exploratory / throwaway notebooks

Experiment/                 Raw and processed data per experiment
  1_HEK293T/
  1_hTERT_HME1/
  2_hTERT_HME1/
  3_hTERT_HME1_mutants_comparison/

External_Data/              External reference datasets
  P146_Feng_Song_phospho_proteomics_data/   MCF10A EGF time course (reformatted TSVs available)

data/                       Intermediate / shared processed data files
```


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

**Concentration ?????** - `4.75nM`, `1.56nM`, `0.156nM`, `10nM`, `100nM`

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
| `protein_ID` | UniProt accession                                                                                                                                                                                                                                                                               |
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

**In progress / pending:**
- Clustering optimisation (parameter tuning, optimal number of clusters)
- Classifier training on WT cluster labels and transfer to mutant datasets
- Downstream analysis (pathway enrichment, PhosX integration)
- LFQ preprocessing: add INS and EGFnINS conditions (currently EGF only)
- TMT preprocessing: add filtering steps to match LFQ filtering (no-PTM, not-in-starve, missing-replicates)

**Known issues:**
- ~~`MSMS_data_QC.ipynb` loads data with `.fillna(0)` — this invalidates the missing value analysis and distorts CV and intensity distributions.~~ **FIXED (2026-07-09):** the four load lines now use `low_memory=False` instead of `.fillna(0)`, so missing intensities stay NaN (this also cleared the `DtypeWarning` on the sparse annotation columns). `peptide_count_per_sample()` in `src/QC.py` was the only QC function that counted NaN as detected (`df != missing_value`); it now uses `notna() & (df != missing_value)`, matching `replicate_detection_map`. All QC functions treat NaN as missing, so the `missing_value=0.0` arguments in the notebook cells are safe to keep.
- ~~PhosphoSitePlus merge in both preprocessing notebooks produces almost entirely NaN for annotation columns~~ **FIXED (2026-07-09):** `get_column_infos()` in `src/transformations.py` (used by `merge_phosphoplus_info`) iterated over an empty list instead of the parsed sites, so it always returned NaN. This affected every `merge_phosphoplus_info` column (`ERK_motif`, `ON_FUNCTION`, `ON_PROCESS`, etc.) in both TMT and LFQ notebooks; `functional_score` (via `merge_functional_score`/`get_average_score`) was already correct. Key alignment (protein ID + residue) was fine. Lower hit rates for regulatory-site columns are expected (small curated table), not a bug.
- `tps_file_creator.ipynb` uses the old pre-refactoring column naming and legacy `.xlsx` paths — it cannot be run on current data without rewriting.
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