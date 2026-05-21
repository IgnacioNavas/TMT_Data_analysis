# Claude code prompt 

## Task 

You have acess to data/hTERT_HME1_mutants_test_sample.tsv and to notebooks/01_preprocessing/Mutant_experiment_pre_analysis.ipynb. 

In the notebook file there is code to do some redefining functions and create some more data columns, together with functions 
to filter the datafrane, and calculate log2 transformations, etc. About the log2 transformations dont worry since the functions 
in src.transformations will work with the data format of the sample file.

about the code in the re define section I want you to optimize it and create a file (src.lfq_pretreatment.py) with the optimized funtions
to add all the comlumns with extra data defined in the jupyter notebook. 

The filtering code in the jupyter file can be also optimized and added to the src.lfq_pretreatment.py file

Do not eddit the transformation.py file, but take it as a reference for the coding stile.

## Task2

Go to section "Create column for math dataset with same naming method" in notebooks/03_clustering/Clustering_mutant_cell_lines.ipynb and implement data part of code to
add the new_site column to the code in src.lfq_pretreatment.py

## Task3

I want you to create the code necessary to replicate column "site" form data/heme1_2_raw_saple.tsv The structure ot the data in this dataset is:
(protein_Id)_(peptide_start)_(peptide_end)_(total amount of STY modifications detected in the phosphorylated peptide)_(amount of STY modifications which could be localized)(if there is any STY modification localized there is one extra feature: _(localized modification with its position in the protein))~(aminoacid sequence of the modified peptide)

I want the new column to be called "site_matching"
