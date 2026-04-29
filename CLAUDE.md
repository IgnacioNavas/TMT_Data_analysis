# Project: The MAPK/ERK signaling pathway crosstalk and regulation at phosphoproteome level

## Project overview
This project aims to understand how cells use phosphorylation of Serine, Threonine and Tyrosine residues to transduce 
signal information in order to generate an adecuate response. For this hTERT-HME1 and HEK293T cell lines are being used.
I am putting my main effort tin the regulation of the MAPK/ERK sinaling pathway, and the crosstalk between EGF and 
insulin stimulation.

Cells grow in full media with growth factors and supplements (full). In this media the level of signaling reaches and 
equilibrium state in wich cell can proligerate having basal levels of signaling. In order to try to syncronize the cells
and reduce the levels of basal signaling to a minimum, the full media is replaced by media without any growth factors
(starve) or supplements for 2 hours. After this 2 hours starvation cell are being stimulated with EGF, insulin (INS), 
and a combination of both (EGFnINS). After this treatment, cell lysate samples are collected at different time points.
In addition to the stimulation treatments, 2 controlls are being collected: cells growing in the full media, and cells
which where starved for 2 hours but did not go under any stimulation treatment. These lysates are treated for 
phosphoproteomics experiments, following two protocols: tandem mass tag - liquid chromatography - mass spectrometry 
(TMT-LC-MS/MS), and label free quantification - liquid chromatography -mass spectrometry (LFQ-LC-MS/MS). 

In addition to stimulation with EGF, INS and EGFnINS, I am introducing perturbations in known regulatory sites of the
MAPK/ERK signaling pathway trying to interup negative regualation of it. I am changing regulatory S/T/Y residues by 
Alanine which cannot be phosphorylated. With the new cell lines I am generating I am doing the same stimulation protocol
with EGF, INS and EGFnINS, and collecting samples at different time points together with the "full" and "starve" 
controls.

The phosphoproteomics data collected from the wild type cell lines (hTERT-HEM1 and HEK293T) where done using TMT-LC-MS/MS
and the data from the mutants was collected using LFQ-LC-MS/MS. 

## Repository structure

The structure of the repository is still not define, it is chaotic. However, the main code for running is in this project folder.

## Naming conventions and Data structure

Not all the datasets have the following labeling sistem but this will be the code use for it. These will be the labeling 
methodology for the data (columns)
 - samples label: (CellLine)_ (DataType)_ (Treatment)_ (TimePoint)_ (Replicate)
   - Cell lines examples: WT or BRAFS151A
   - Data type: it has several caracteristics separated by ":" like "raw:abs" or "log2:mean". 
     - The first term will be if the numbers represent the raw absorbance detected by the spectrometer (raw) or if this data has been transformed (usually log2 transformations) (log2)
     - The second term is what is the data representing:
       - "abs" absorbance per sample (the name of this sample will have the replicate its coming from at the end)
       - "mean" mean for the replicates for this time point and condition
       - "media" median for the replicates for this time point and condition
       - "FC" fold change respect the starve time point (the starve time point is our control time point), only for the log2 transformed data
       - "scaled" scaled fold change to have the max amplitude between -1 an 1 and centered around "0" in the starve time point
       - "sd" standard deviation for the replicates for this time point and condition
       - "var"
       - "cv" coeficient of variance
       - "FDR" false discovery rate
       - "pvalue"
   - Treatment: can be EGF, INS, EGFnINS
   - Time point: the time point for representation will be used as qualitative "x" axis
     - "full" is the first time point
     - "starve" is the second time point
     - "1", "2", "5", etc., follow in numerical order (different datasets migth have different time points)
   - Replicate: the columns for the original data will contain the replicate, but hte column sof the rest not necessary. If the column represent log2:mean, there is no replicate to indicate
     - "r" followed by the number of replicate it is, for example: "r1" or "r2"
 
In addition, of these columns that contain absorbance data from the spectrometer and its normalization and/or 
transformations, there is data corresponding to other parameters like:
 - sequence: peptide sequence identified
 - n_rep: number of replicates in which the peptide has been detected
 - protein_name: protein to which the peptide belongs to
 - protein_ID: uniprot ID of the protein teh peptide belongs
 - etc.,

## Data analysis overview

To analyze this data I am planing to cluster the temporal dynamics the phosphorylated peptides can adopt. Phosphorylated
peptides with similar temporal dynamics should cluster together. For the clustering I want to consider each stimulation
condition as an extra dimension of a multidimensional time series profile for each phosphorylated peptide.

My idea is to do the unsupervised clustering on the data for the WT cell line. The use these clusters assignation as a 
label to train a classifier and use this new classifier model on the data from the mutant cell lines. My idea is that if 
a phosphorylated peptide appears in the same cluster as in the WT cell line, the temporal profile of the peptide has not
changed and its function was not necessary to buffer the perturbation introduced. If the peptide goes to a different 
cluster thant the same peptide used to train the model, then the time profile has changed and either the peptide was 
needed to buffer the preturbation introduced or is being affected by the perturbation.


## Current status

I think initial quality controls of the data are missing in some points

There are many functions working but they are un-structured and need to be ajusted so it works with all datasets.

Clustering functions are working.However, clustering need to be further optimized.

Clustering labes have not been used yet to train any classification model.

## Important rules
 - Always add documentation to new code to better understand what it does
 - If needed add documentation to old code
 - Never overwrite .csv or .xlsx files, always add tracking number at the end or a word that explains the new modification
 - 

