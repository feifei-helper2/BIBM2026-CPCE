#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: Generate Figure 3 (UMAP Projections) and perform DEG validation for Mesangial cells.
Reference: Section IV.D Biological Case Study: Visualizing the Majority Trap
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Standard academic plotting configurations
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['pdf.fonttype'] = 42

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DIR_DATA = os.path.join(PROJECT_ROOT, "data", "processed")
DIR_LABELS = os.path.join(PROJECT_ROOT, "results", "saved_labels")
DIR_FIGURES = os.path.join(PROJECT_ROOT, "results", "figures")
os.makedirs(DIR_FIGURES, exist_ok=True)

def main():
    print("Starting Biological Case Study: Dataset 7 (Living Donor Kidney) - Mesangial Cells")
    
    # 1. Load core data and labels
    data_path = os.path.join(DIR_DATA, "output18.h5ad")
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}. Please ensure Dataset 7 is downloaded.")
        return
        
    adata = sc.read_h5ad(data_path)
    
    cpce_path = os.path.join(DIR_LABELS, "output18_CPCE_run0.npy")
    leiden_path = os.path.join(DIR_LABELS, "output18_Leiden_run0.npy")
    
    if not (os.path.exists(cpce_path) and os.path.exists(leiden_path)):
        print("Error: Required clustering result files (.npy) not found. Please run 02_run_main_benchmark.py first.")
        return

    # Force type to string Categorical for downstream differential expression analysis
    adata.obs['CPCE'] = pd.Categorical(np.load(cpce_path).astype(str))
    adata.obs['Leiden'] = pd.Categorical(np.load(leiden_path).astype(str))
    
    target_cell = 'Mesangial'

    # ==========================================
    # Module A: Generate Figure 3 (UMAP Projections)
    # ==========================================
    print("\n[1/2] Generating Figure 3: UMAP Projections...")
    methods = ['ground_truth', 'Leiden', 'CPCE']
    titles = ['(A) Ground Truth', '(B) Leiden (Over-merged Artifact)', '(C) CPCE (Accurate Isolation)']

    umap = adata.obsm['X_umap']
    x, y = umap[:, 0], umap[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    # Dynamically capture the target cluster ID for CPCE
    cpce_mesangial_cluster = None

    for i, (method, title) in enumerate(zip(methods, titles)):
        ax = axes[i]
        
        # Plot light gray background for global manifold
        ax.scatter(x, y, c='#E0E0E0', s=2, alpha=0.3, edgecolors='none')
        
        if method == 'ground_truth':
            highlight_idx = (adata.obs['ground_truth'] == target_cell).values
        else:
            true_target_idx = (adata.obs['ground_truth'] == target_cell)
            labels_of_target = adata.obs.loc[true_target_idx, method]
            best_cluster = labels_of_target.value_counts().index[0]
            highlight_idx = (adata.obs[method] == best_cluster).values
            
            # Record CPCE's target cluster ID for subsequent gene analysis
            if method == 'CPCE':
                cpce_mesangial_cluster = best_cluster

        # Highlight target cells in red
        ax.scatter(x[highlight_idx], y[highlight_idx],
                   c='#E63946', s=30, alpha=1.0, edgecolors='none', zorder=5)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(DIR_FIGURES, "Fig3_UMAP_CaseStudy.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"Successfully generated Figure 3: {save_path}")

    # ==========================================
    # Module B: Biological Specificity Validation (DEG Analysis)
    # ==========================================
    print(f"\n[2/2] Validating gene purity for CPCE isolated cluster {cpce_mesangial_cluster}...")
    
    # Execute Wilcoxon rank-sum test for differentially expressed genes
    sc.tl.rank_genes_groups(
        adata, 
        'CPCE', 
        method='wilcoxon', 
        groups=[cpce_mesangial_cluster], 
        reference='rest'
    )

    result = adata.uns['rank_genes_groups']
    genes = pd.DataFrame(result['names'])[cpce_mesangial_cluster].head(10).values
    pvals = pd.DataFrame(result['pvals'])[cpce_mesangial_cluster].head(10).values

    print(f"\n--- Mesangial Cell Top Marker Genes (Cluster {cpce_mesangial_cluster}) ---")
    
    # Target markers highlighted in the paper
    target_markers = ['CALD1', 'TAGLN', 'ACTA2']
    found_markers = []
    
    for gene, pval in zip(genes, pvals):
        if gene in target_markers:
            print(f"[*] Verified Core Marker -> {gene}: p-value = {pval:.2e}")
            found_markers.append(gene)
        else:
            print(f"    - {gene}: p-value = {pval:.2e}")
            
    if set(target_markers).issubset(set(found_markers)):
        print("\nSuccess: Markers CALD1, TAGLN, and ACTA2 mentioned in the paper were fully verified.")
    else:
        print("\nNotice: Some core markers mentioned in the paper did not appear in the Top 10. Consider expanding the output range.")

if __name__ == "__main__":
    main()
