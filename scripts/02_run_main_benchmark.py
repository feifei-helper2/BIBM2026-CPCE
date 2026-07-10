#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: Main Benchmark Orchestrator
Purpose: Automated batch processing for clustering algorithms across datasets.
Features:
1. Breakpoint Resume: Dual resume mechanism based on CSV and .npy to avoid redundant calculations.
2. Real-time Append: Secure incremental append mode for CSV writing.
3. Exception Isolation: try-except blocks wrapping I/O and computation to protect pipeline execution.
"""
import os
import sys
import glob
import time
import logging
import traceback
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# ==========================================
# 0. Environment Path Alignment and Modules
# ==========================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.baselines import BaselineModels
from src.cpce_engine import run_synthesized_cpce
from src.metrics import calc_rare_class_f1_strict

# Directory initialization
DIR_DATA = os.path.join(PROJECT_ROOT, "data", "processed")
DIR_TABLES = os.path.join(PROJECT_ROOT, "results", "tables")
DIR_LABELS = os.path.join(PROJECT_ROOT, "results", "saved_labels")
DIR_LOGS = os.path.join(PROJECT_ROOT, "logs")

for d in [DIR_TABLES, DIR_LABELS, DIR_LOGS]:
    os.makedirs(d, exist_ok=True)

# Dual-stream logging setup
log_file = os.path.join(DIR_LOGS, "experiment_runtime.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# ==========================================
# 1. Hyperparameters and Algorithm Registry
# ==========================================
GLOBAL_SEED = 42
RARE_THRESHOLD = 0.05
OUTPUT_CSV = os.path.join(DIR_TABLES, "main_benchmark_results.csv")

def run_cpce_wrapper(X_pca, K, seed):
    """Inject dynamic random seeds into the CPCE engine for stability testing."""
    # 1. Global defense: Control numpy's base random state
    np.random.seed(seed) 
    
    # 2. Parameter routing: Control sklearn/scanpy internal random states
    return run_synthesized_cpce(X_pca, K, seed=seed)

# Algorithm Registry (Unified interface: lambda X_pca, K, seed, adata)
ALGORITHMS_REGISTRY = {
    # --- Classical and standard baselines ---
    "KMeans": {"func": lambda X, K, s, adata: BaselineModels.run_kmeans(X, K, s), "runs": 10},
    "GMM": {"func": lambda X, K, s, adata: BaselineModels.run_gmm(X, K, s), "runs": 10},
    "Spectral": {"func": lambda X, K, s, adata: BaselineModels.run_spectral(X, K, s), "runs": 10},
    "Leiden": {"func": lambda X, K, s, adata: BaselineModels.run_leiden_with_bisection(X, K, s), "runs": 10},
    
    # --- Ablation baselines ---
    "Balanced_KMeans": {"func": lambda X, K, s, adata: BaselineModels.run_balanced_kmeans(X, K, s), "runs": 10},
    "Random_Ensemble": {"func": lambda X, K, s, adata: BaselineModels.run_random_ensemble(X, K, s), "runs": 10},
    
    # --- Advanced SOTA models ---
    "scCAD": {"func": lambda X, K, s, adata: BaselineModels.run_sccad(X, adata, K, s), "runs": 10},
    "GiniClust3": {"func": lambda X, K, s, adata: BaselineModels.run_giniclust3(X, adata, K, s), "runs": 10},
    "scVI": {"func": lambda X, K, s, adata: BaselineModels.run_scvi_kmeans(X, adata, K, s), "runs": 10},
    
    # --- Core Engine ---
    "CPCE": {"func": lambda X, K, s, adata: run_cpce_wrapper(X, K, s), "runs": 10}
}

# ==========================================
# 2. Atomic Writing Components
# ==========================================
def append_result_to_csv(record_dict, csv_path):
    """Secure incremental append to avoid file corruption during full overwrites."""
    df_row = pd.DataFrame([record_dict])
    file_exists = os.path.isfile(csv_path)
    df_row.to_csv(csv_path, mode='a', header=not file_exists, index=False)

def get_completed_tasks(csv_path):
    """Read historical logs to provide row-level breakpoint resume execution."""
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            return set(zip(df['Dataset'], df['Algorithm'], df['Run_ID']))
        except Exception as e:
            logging.warning(f"Warning: Failed to read historical CSV ({e}). Starting fresh.")
    return set()

# ==========================================
# 3. Main Orchestration Bus
# ==========================================
def main():
    logging.info("============ CPCE Main Benchmark Pipeline Started ============")
    
    dataset_files = sorted(glob.glob(os.path.join(DIR_DATA, "*.h5ad")))
    if not dataset_files:
        logging.error(f"Critical Error: No standard datasets found in {DIR_DATA}!")
        return
        
    logging.info(f"Discovered {len(dataset_files)} benchmark datasets.")
    
    completed_tasks = get_completed_tasks(OUTPUT_CSV)
    if completed_tasks:
        logging.info(f"Detected {len(completed_tasks)} historical completion records. Activating incremental resume mode.")

    # [Layer 1] Iterate through datasets
    for file_path in dataset_files:
        dataset_name = os.path.basename(file_path).replace(".h5ad", "")
        logging.info(f"\n{'-'*60}\nLoading Dataset: {dataset_name}\n{'-'*60}")
        
        try:
            adata = sc.read_h5ad(file_path)
            X_pca = adata.obsm['X_pca']
            y_true = adata.obs['ground_truth'].values
            K_gt = int(adata.uns['n_clusters_gt'])
            N_cells = X_pca.shape[0]
            logging.info(f"   [Parameters] N={N_cells}, K={K_gt}")
        except Exception as e:
            logging.error(f"Dataset parsing failed, skipping. Error: {e}")
            continue

        # [Layer 2] Iterate through competitor algorithms
        for algo_name, config in ALGORITHMS_REGISTRY.items():
            algo_func = config["func"]
            n_runs = config["runs"]
            
            # [Layer 3] Stability batches
            for run_idx in range(n_runs):
                current_seed = GLOBAL_SEED + run_idx
                task_fingerprint = (dataset_name, algo_name, run_idx)
                
                # Defense 1: Row-level resume
                if task_fingerprint in completed_tasks:
                    continue
                    
                logging.info(f"   Executing: {algo_name} (Run {run_idx+1}/{n_runs}) ...")
                
                label_save_path = os.path.join(DIR_LABELS, f"{dataset_name}_{algo_name}_run{run_idx}.npy")
                labels_pred = None
                exec_time = np.nan  
                
                start_time = time.time()
                try:
                    # Defense 2: Asset-level quick load
                    if os.path.exists(label_save_path):
                        logging.info(f"      [Asset Load] Target .npy file found locally. Skipping computation.")
                        labels_pred = np.load(label_save_path)
                    else:
                        # True execution
                        labels_pred = algo_func(X_pca, K_gt, current_seed, adata)
                        exec_time = time.time() - start_time
                        np.save(label_save_path, labels_pred)
                        
                    # Encapsulate metric calculation within try-except
                    ari = adjusted_rand_score(y_true, labels_pred)
                    nmi = normalized_mutual_info_score(y_true, labels_pred)
                    rare_metrics = calc_rare_class_f1_strict(y_true, labels_pred, RARE_THRESHOLD)
                    mean_f1 = rare_metrics.get("Mean_Rare_F1", np.nan)
                    soft_geo_f1 = rare_metrics.get("Soft_Geo_Rare_F1", np.nan)
                    detection_rate = rare_metrics.get("Detection_Rate", np.nan)
                    median_f1 = rare_metrics.get("Median_Rare_F1", np.nan)
                    q1_f1 = rare_metrics.get("Q1_Rare_F1", np.nan)

                    status = "Success"
                    time_log = f"{exec_time:.1f}s" if pd.notna(exec_time) else "Cached"
                    logging.info(f"      [Success] Time: {time_log} | ARI: {ari:.3f} | Geo_F1: {soft_geo_f1:.3f} | Det_Rate: {detection_rate:.2f}")
                    
                except Exception as e:
                    if np.isnan(exec_time):
                        exec_time = time.time() - start_time
                    error_msg = str(e).split('\n')[0]
                    logging.error(f"      [Aborted] {error_msg}")
                    
                    ari, nmi, rare_f1 = np.nan, np.nan, np.nan
                    status = f"Failed: {error_msg}"
                    mean_f1 = soft_geo_f1 = detection_rate = median_f1 = q1_f1 = np.nan

                # Atomic record dump
                record = {
                    "Dataset": dataset_name,
                    "Algorithm": algo_name,
                    "Run_ID": run_idx,
                    "Seed": current_seed,
                    "N_Cells": N_cells,
                    "K": K_gt,
                    "Time_s": round(exec_time, 2) if pd.notna(exec_time) else np.nan,
                    "ARI": round(ari, 4),
                    "NMI": round(nmi, 4),
                    "Rare_F1": round(mean_f1, 4) if pd.notna(mean_f1) else "NaN",
                    "Soft_Geo_F1": round(soft_geo_f1, 4) if pd.notna(soft_geo_f1) else "NaN",
                    "Detection_Rate": round(detection_rate, 4) if pd.notna(detection_rate) else "NaN",
                    "Median_F1": round(median_f1, 4) if pd.notna(median_f1) else "NaN",
                    "Q1_F1": round(q1_f1, 4) if pd.notna(q1_f1) else "NaN",
                    "Status": status
                }
                
                append_result_to_csv(record, OUTPUT_CSV)

    logging.info(f"\n============ Batch Execution Completed ============")
    logging.info(f"Aggregated results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
