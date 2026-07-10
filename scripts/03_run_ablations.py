#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: Ablation Study Orchestrator
Purpose: Validate the effectiveness of CPCE core mechanisms across extreme datasets.
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Force single threading to prevent deadlocks
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.cpce_engine import run_synthesized_cpce
from src.metrics import calc_rare_class_f1_strict

# ==========================================
# 1. Configuration and Path Setup
# ==========================================
DIR_DATA = os.path.join(PROJECT_ROOT, "data", "processed")
DIR_TABLES = os.path.join(PROJECT_ROOT, "results", "tables")
DIR_LABELS = os.path.join(PROJECT_ROOT, "results", "saved_labels_ablation")

os.makedirs(DIR_LABELS, exist_ok=True)
os.makedirs(DIR_TABLES, exist_ok=True)
OUTPUT_CSV = os.path.join(DIR_TABLES, "ablation_results.csv")

RARE_THRESHOLD = 0.05
RUNS = 10 
BASE_SEED = 42

# Core Ablation Variants (NoGS integrated, avg_link removed)
VARIANT_MAPPING = {
    'full': 'CPCE_Full',
    'no_dpc': 'CPCE_NoDPC',
    'no_zipf': 'CPCE_NoZipf',
    'no_gs': 'CPCE_NoGS'
}

def append_result(record, csv_path):
    df = pd.DataFrame([record])
    file_exists = os.path.isfile(csv_path)
    df.to_csv(csv_path, mode='a', header=not file_exists, index=False)

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logging.info("Starting CPCE Core Mechanism Ablation Study...")

    completed = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            df = pd.read_csv(OUTPUT_CSV)
            completed = set(zip(df['Dataset'], df['Variant'], df['Run']))
            logging.info(f"Detected {len(completed)} historical records. Activating incremental resume mode.")
        except Exception:
            pass

    # Dynamically scan for all .h5ad files in the directory
    dataset_files = [f for f in os.listdir(DIR_DATA) if f.endswith('.h5ad')]
    
    if not dataset_files:
        logging.error(f"Error: No datasets found in {DIR_DATA}")
        return

    for file_name in dataset_files:
        # Remove extension to use as the exact dataset name (e.g., "dataset_5")
        paper_name = file_name.replace(".h5ad", "")
        file_path = os.path.join(DIR_DATA, file_name)
            
        logging.info(f"\nLoading dataset: {paper_name}")
        adata = sc.read_h5ad(file_path)
        X_pca = adata.obsm['X_pca']
        y_true = adata.obs['ground_truth'].values
        K = int(adata.uns['n_clusters_gt'])
        N = X_pca.shape[0]

        for internal_v, paper_v in VARIANT_MAPPING.items():
            for run_idx in range(RUNS):
                seed = BASE_SEED + run_idx
                task_key = (paper_name, paper_v, run_idx)
                
                if task_key in completed:
                    continue
                    
                logging.info(f"   Processing: {paper_v} (Run {run_idx+1}/{RUNS}) ...")
                
                label_path = os.path.join(DIR_LABELS, f"{raw_name}_{internal_v}_run{run_idx}.npy")
                time_used = np.nan
                start_time = time.time()
                
                try:
                    if os.path.exists(label_path):
                        labels_pred = np.load(label_path)
                    else:
                        labels_pred = run_synthesized_cpce(X_pca, K, T=10, alpha_max=2.0, seed=seed, variant=internal_v)
                        time_used = time.time() - start_time
                        np.save(label_path, labels_pred)
                        
                    ari = adjusted_rand_score(y_true, labels_pred)
                    nmi = normalized_mutual_info_score(y_true, labels_pred)
                    metrics = calc_rare_class_f1_strict(y_true, labels_pred, RARE_THRESHOLD)
                    
                    status = "Success"
                    logging.info(f"      Success | ARI: {ari:.3f} | Geo_F1: {metrics['Soft_Geo_Rare_F1']:.3f} | Det: {metrics['Detection_Rate']:.2f}")
                    
                except Exception as e:
                    logging.error(f"      Failed: {e}")
                    ari, nmi = np.nan, np.nan
                    metrics = {"Mean_Rare_F1": np.nan, "Soft_Geo_Rare_F1": np.nan, "Detection_Rate": np.nan}
                    status = "Failed"

                record = {
                    "Dataset": paper_name,
                    "Variant": paper_v,
                    "Run": run_idx,
                    "Seed": seed,
                    "Time_s": round(time_used, 2) if not np.isnan(time_used) else np.nan,
                    "ARI": round(ari, 4) if pd.notna(ari) else np.nan,
                    "NMI": round(nmi, 4) if pd.notna(nmi) else np.nan,
                    "Mean_F1": round(metrics.get("Mean_Rare_F1", np.nan), 4),
                    "Geo_F1": round(metrics.get("Soft_Geo_Rare_F1", np.nan), 4),
                    "Det_Rate": round(metrics.get("Detection_Rate", np.nan), 4),
                    "Status": status
                }
                
                append_result(record, OUTPUT_CSV)

    logging.info(f"\nAblation batch completed! Results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
