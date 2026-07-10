CPCE (Capacity-Profile Clustering Ensemble)  
This anonymous repository contains the official Python implementation for the paper: "CPCE: A Dual-Core Consensus Framework for Escaping the Majority Trap in Rare Cell Discovery".   
  
Installation  
Create a virtual environment and install the required dependencies:  
pip install -r requirements.txt  
  
Datasets Preparation  
Due to file size limits, the raw `.h5ad` datasets are not included directly. To ensure full reproducibility of the core findings while respecting size constraints, we provide the two most critical datasets used in the paper.  
Instructions and pre-processing scripts for the 7 benchmark datasets (sourced from CZ CELLXGENE) will be fully updated upon publication.  
  
Please download them from the following links:  
Dataset 5 (Healthy Small Intestine):https://datasets.cellxgene.cziscience.com/674a4e43-a99a-4a76-8f23-a63e3e85933a.h5ad  
Dataset 7 (Living Donor Kidney):https://datasets.cellxgene.cziscience.com/a305d1c6-2e04-42af-8643-ff300cfafacf.h5ad  
  
**Important:** After downloading, place both `.h5ad` files into the `data/processed/` directory in the project root and explicitly rename them to `dataset_5.h5ad` and `dataset_7.h5ad`.  
  
Important: After downloading, place the .h5ad file into a directory named data/ in the project root and rename it to dataset_1.h5ad.  
  
Your directory structure should look like this before running:  
  
Plaintext  
├── data/  
│     └── processed/  
│           ├── dataset_5.h5ad  
│           └── dataset_7.h5ad  
├── scripts/  
├── src/  
└── requirements.txt  

Quick Start (Reproduction)  
Please execute the scripts from the root directory of the repository.  
  
1. Run the Main Benchmark (Reproduces Table II)  
Executes CPCE and baseline algorithms on the prepared datasets:  
python scripts/02_run_main_benchmark.py  
  
2. Run Core Ablation Studies (Reproduces Table III)  
Validates the individual contributions of the Micro-Radar and Macro-Valve modules:  
python scripts/03_run_ablations.py  
  
3. Run Extended Experiments (Reproduces Table IV)  
Executes parameter sensitivity analysis and noise robustness subsampling tests:  
python scripts/05_run_extensions.py  
  
4. Generate Figures and Tables (Reproduces Fig 2, Fig 3 & Supplementary Data)  
Generates the quadrant scatter plot (Fig 2) exposing the ARI trap, the UMAP projections (Fig 3) for biological validation, and parses the CSV logs into final metric tables:  
python scripts/04_generate_figures1.py  
python scripts/04_generate_figures2.py  
python scripts/06_generate_extensions_figures.py  
