CPCE (Capacity-Profile Clustering Ensemble)
This repository contains the official Python implementation for the paper: "CPCE: A Dual-Core Consensus Framework for Escaping the Majority Trap in Rare Cell Discovery".

Install dependencies
This anonymous repository contains the official Python implementation for the paper: 
"CPCE: A Dual-Core Consensus Framework for Escaping the Majority Trap in Rare Cell Discovery".

Installation
Create a virtual environment and install the required dependencies:
pip install -r requirements.txt

Datasets Preparation
Due to file size limits, the raw .h5ad datasets are not included in this repository. 
Instructions and pre-processing scripts for the 7 benchmark datasets (sourced from CZ CELLXGENE) will be fully updated upon publication.

To test the code immediately, please download Dataset 1 from the following link:
Dataset 1 Download: https://datasets.cellxgene.cziscience.com/2e1f6ddf-f3d3-44e6-a8ba-32fa9072a877.h5ad

Important: After downloading, place the .h5ad file into a directory named data/ in the project root and rename it to dataset_1.h5ad.

Your directory structure should look like this before running:

Plaintext
├── data/
│   └── dataset_1.h5ad
├── scripts/
├── src/
└── requirements.txt

Quick Start (Reproduction)
Please execute the scripts from the root directory of the repository.

1. Run the Main Benchmark
Executes CPCE and baseline algorithms on the prepared dataset:
python scripts/02_run_main_benchmark.py

2. Run Ablation Studies
Validates the individual contributions of the Micro-Radar and Macro-Valve modules:
python scripts/03_run_ablations.py

3. Generate Figures and Case Study
Reproduces the UMAP projections, quadrant scatter plots, and extracts the biological marker genes for rare niches:
python scripts/04_generate_figures2.py
python scripts/06_generate_extensions_figures.py
