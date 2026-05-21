# Project: The MAPK/ERK signaling pathway crosstalk and regulation at phosphoproteome level

## Project overview

This project aims to understand how cells use phosphorylation of Serine, Threonine and Tyrosine
residues to transduce signal information and generate an adequate cellular response. The main focus
is on the regulation of the MAPK/ERK signaling pathway and the crosstalk between EGF and insulin
stimulation, using hTERT-HME1 and HEK293T cell lines.

Cells grow in full media with growth factors and supplements (full). In this media, signaling reaches
an equilibrium state in which cells can proliferate with basal levels of signaling. To synchronize
cells and reduce basal signaling, full media is replaced by media without growth factors or
supplements for 2 hours (starve). After starvation, cells are stimulated with EGF, insulin (INS),
or a combination of both (EGFnINS), and lysates are collected at multiple time points. Two controls
are also collected: cells in full media, and cells that were starved but not stimulated.

These lysates are processed for phosphoproteomics using two protocols:
- **TMT-LC-MS/MS** — used for wild-type hTERT-HME1 and HEK293T
- **LFQ-LC-MS/MS** — used for the mutant cell lines

In addition to wild-type experiments, regulatory S/T/Y residues in the MAPK/ERK pathway are mutated
to Alanine (which cannot be phosphorylated) to disrupt negative feedback regulation. These mutant
cell lines undergo the same stimulation protocol (EGF, INS, EGFnINS) with full and starve controls.

An external reference dataset (MCF10A EGF time course, TMT-LC-MS/MS, timepoints: 2, 4, 8, 12 min)
from Feng Song et al. is available in `External_Data/` for comparison with the hTERT-HME1 results.


## Repository structure

```
src/                        Python modules
  column_spec.py            ColumnSpec class — the standard way to select data columns
  transformations.py        Data transformation pipeline (run_all_transformations)
  QC.py                     Quality control functions (sections 1–4, see notebooks/02_qc/)
  plotting_functions.py     Visualisation: time series, cluster, protein profile plots
  clustering.py             Clustering utilities
  lfq_pretreatment.py       LFQ-specific preprocessing (mutant datasets)
  utils.py                  Shared utilities

notebooks/
  01_preprocessing/         Raw data processing and transformation
  02_qc/                    Quality control (MSMS_data_QC.ipynb, PCA.ipynb, General_QC.ipynb)
  03_clustering/            Unsupervised clustering
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
  - `scaled` — FC scaled so amplitude is in [−1, 1] and starve = 0
  - `sd` — standard deviation across replicates
  - `cv` — coefficient of variation (%)
  - `var` — variance
  - `FDR` — false discovery rate
  - `pvalue` — p-value

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

| Column | Description |
|--------|-------------|
| `site` | Phosphosite identifier (`ProteinName-Residue`, e.g. `EGFR_HUMAN-Y1068y`) |
| `sequence` | Peptide sequence |
| `protein_name` | Protein name |
| `protein_ID` | UniProt accession |
| `residue` | Modified residue(s), e.g. `S151s` |
| `n:reps` | Number of replicates in which the peptide was detected |
| `AScore` | Phosphosite localization ambiguity score |


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
  (static and interactive), PCA distance heatmap, normalization checks, Venn diagrams, overlap stats
- Plotting functions restructured and unified (`src/plotting_functions.py`): `clusters_plot_linear`
  supports `panel_by="condition"` and `panel_by="cell_line"`; mutant-specific redundant functions removed
- External MCF10A reference dataset reformatted to project naming convention
- Clustering is implemented and working

**In progress / pending:**
- Clustering optimisation (parameter tuning, optimal number of clusters)
- Classifier training on WT cluster labels and transfer to mutant datasets
- Downstream analysis (pathway enrichment, PhosX integration)


## Important rules

- Always add docstrings to new functions explaining arguments and return values
- Add documentation to old code when editing it
- Never overwrite `.csv` or `.xlsx` files — always append a descriptive suffix or version tag to the output filename (e.g. `_reformatted`, `_filtered`)
- Use `ColumnSpec.select()` from `src/column_spec.py` for all data column selection — do not build column lists by hand with string matching
- Processed data files should be saved as `.tsv`
- If I ask you to move functions from one file to the other don't delete them from the original file, comment them with "#" so I can still compare the old functions with the new ones in the new file
- 


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