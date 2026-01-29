# TMT_Data_analysis

Wellcome this repository makes the data processing and data analysis from the "The functional crosstalk between EGF and 
insulin signalling" project in Prof. Dr. Tina Perica lab. 

In this repository there is also data from other publications that has been used to compare with our data generated. 

This repository is coded by Ignacio Navas


## Code overview
- Data_filtration_and_processing
  - Extract information from the "All" dataset, reorganizing and refactoring
  - Extract normalized data of abundances
  - Add standard deviation (SD), the number of replicates
  - filter by log2FC values, number of replicates and ir number of phosphorylations identified in the peptide
    - some functions for this pourpose, but they do not analyze based on name of the column but the number, so they are dependent on how the data is organized
  - Expressing the curve behaviour as qualitative change
  - Check sites which pass filters
  - External data analysis
- dtwsom_clustering
  - use the pipeline of a webpage to do clustering using distances and Self organising maps
  - plots the data (curves per site)
  - histograms of distributions of sites in the clusters
- feature_extraction_tsfresh
  - uses the tsfresh package to auto-extract data from the time series data
  - in theory all these feature can be used to understand the time series better, not sure exactly how
- mega_dataset_compiler
  - use the pandas function **merge** to merge dataframes. It is way faster
- mega_dataset_compiler
  - have the functions necessary to plot the site with raw, normalized or FC
  - can plot or without replicates
- PCA
  - import PCA function from sklearn.decomposition
- Ploting data
  - first ploting funtions i made. 
  - all of these plotting functions can be done by the megaplotting function
- profiles_difference
  - con plot a whole protein profile (together or individually of the EGF, INS or EGFnINS)
  - can calculate differences between stimulation conditions
  - can plot the histograms of distributions of how different stimulations are one to eachother
  - can calculate the euclidean distances
- recalculating_FC
  - calculates LOG2
  - calculate average log2 trasformation
  - calculate the FC respect the starvation time point
  - calculate the fold change using using this new log2
  - calculate the fold change using the log2 calculated by fgcz
- Testing
  - chunk of code to test some plotting functinos and PCA
- tps_file_creator
  - tries adjust the data to be able to use the tps
- tslearn_clustering
  - uses tslearn to do clustering
  - plot the clusters