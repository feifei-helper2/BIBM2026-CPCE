#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: Extended Experiments Orchestrator
Purpose: Parameter sensitivity analysis and noise robustness (subsampling) evaluation.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.cpce_engine import run_synthesized_cpce
from src.metrics import calc_rare_class_f1_strict

DIR_DATA = os.path.join(PROJECT_ROOT, "data", "processed")
DIR_TABLES = os.path.join(PROJECT_ROOT, "results", "tables")
os.makedirs(DIR_TABLES, exist_ok=True)

def append_csv(record, path):
    df = pd.DataFrame([record])
    df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)

def main():
    print("Starting Extended Experiments (Sensitivity & Robustness)...")
    
    sens_csv = os.path.join(DIR_TABLES, "sensitivity_results.csv")
    rob_csv = os.path.join(DIR_TABLES, "robustness_results.csv")
    
    # Load dataset_7 for sensitivity and robustness tests
    print("\nLoading dataset_7 for sensitivity and robustness tests...")
    data_path = os.path.join(DIR_DATA, "dataset_7.h5ad")
    if not os.path.exists(data_path):
        print(f"Error: Dataset missing: {data_path}")
        return
        
    adata = sc.read_h5ad(data_path)
    X_pca, y_true = adata.obsm['X_pca'], adata.obs['ground_truth'].values
    K = len(np.unique(y_true))

    # ==================================================
    # Module 1: Parameter Sensitivity Analysis
    # ==================================================
    print("\n--- Module 1: Parameter Sensitivity Analysis (T & alpha_max) ---")
    
    # Evaluate T (Ensemble Size)
    for T_val in [3, 5, 10, 15, 20]:
        for run in range(5): 
            print(f"Testing T={T_val} | Run {run}")
            labels_pred = run_synthesized_cpce(X_pca, K, T=T_val, alpha_max=2.0, seed=42+run)
            m = calc_rare_class_f1_strict(y_true, labels_pred)
            append_csv({"Parameter": "T", "Value": T_val, "Run": run, "ARI": adjusted_rand_score(y_true, labels_pred), "Geo_F1": m["Soft_Geo_Rare_F1"]}, sens_csv)
            
    # Evaluate alpha_max (Zipf Exponent)
    for alpha_val in [1.5, 2.0, 2.5, 3.0]:
        for run in range(5):
            print(f"Testing alpha_max={alpha_val} | Run {run}")
            labels_pred = run_synthesized_cpce(X_pca, K, T=10, alpha_max=alpha_val, seed=42+run)
            m = calc_rare_class_f1_strict(y_true, labels_pred)
            append_csv({"Parameter": "alpha_max", "Value": alpha_val, "Run": run, "ARI": adjusted_rand_score(y_true, labels_pred), "Geo_F1": m["Soft_Geo_Rare_F1"]}, sens_csv)

    # ==================================================
    # Module 2: Noise Robustness (Cellular Subsampling)
    # ==================================================
    print("\n--- Module 2: Cellular Subsampling Robustness ---")
    N_total = len(y_true)
    ratios = [0.9, 0.7, 0.5]
    
    for ratio in ratios:
        for run in range(5):
            np.random.seed(42 + run)
            n_keep = int(N_total * ratio)
            idx = np.random.choice(N_total, n_keep, replace=False)
            
            X_sub, y_sub = X_pca[idx], y_true[idx]
            K_sub = len(np.unique(y_sub)) 
            
            print(f"Testing subsampling {int(ratio*100)}% | Run {run} | Retained cells: {n_keep} | K={K_sub}")
            
            # Execute CPCE
            labels_cpce = run_synthesized_cpce(X_sub, K_sub, seed=42+run)
            m_cpce = calc_rare_class_f1_strict(y_sub, labels_cpce)
            append_csv({"Ratio": ratio, "Algorithm": "CPCE", "Run": run, "ARI": adjusted_rand_score(y_sub, labels_cpce), "Geo_F1": m_cpce["Soft_Geo_Rare_F1"]}, rob_csv)
            
            # Execute KMeans as baseline comparison
            labels_km = KMeans(n_clusters=K_sub, random_state=42+run, n_init=10).fit_predict(X_sub)
            m_km = calc_rare_class_f1_strict(y_sub, labels_km)
            append_csv({"Ratio": ratio, "Algorithm": "KMeans", "Run": run, "ARI": adjusted_rand_score(y_sub, labels_km), "Geo_F1": m_km["Soft_Geo_Rare_F1"]}, rob_csv)

    print("\nExtended experiments completed successfully!")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
