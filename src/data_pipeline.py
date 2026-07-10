# -*- coding: utf-8 -*-
"""
File: src/data_pipeline.py
Purpose: Data preprocessing and quality control pipeline for highly imbalanced datasets.
"""
import scanpy as sc
import numpy as np
import scipy.sparse as sp
import pandas as pd
import warnings
import os

warnings.filterwarnings("ignore")

class DataPipeline:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.adata = None
        self.K = None
        
    def _print_report(self, title, adata, label_col=None):
        """Generates physical diagnostics and validation reports for the dataset."""
        print(f"\n{'='*20} {title} {'='*20}")
        print(f"[Data Overview]")
        print(f"  - Cells (Observations): {adata.n_obs}")
        print(f"  - Genes (Variables): {adata.n_vars}")
        
        X_data = adata.X.data if sp.issparse(adata.X) else adata.X
        is_sparse = sp.issparse(adata.X)
        print(f"  - Matrix Type: {'Sparse' if is_sparse else 'Dense'}")
        if is_sparse:
             print(f"  - Sparsity Ratio: {1.0 - adata.X.nnz / (adata.n_obs * adata.n_vars):.4f}")
        
        min_val = np.min(X_data) if X_data.size > 0 else 0
        max_val = np.max(X_data) if X_data.size > 0 else 0
        is_logged = max_val < 100
        is_scaled = min_val < 0
        print(f"  - Value Distribution: {'Contains negative values (Scaled)' if is_scaled else ('Log-transformed' if is_logged else 'Raw Counts')}")
        
        if label_col and label_col in adata.obs:
             counts = adata.obs[label_col].value_counts()
             K = len(counts)
             imbalance_ratio = counts.max() / counts.min() if counts.min() > 0 else float('inf')
             rare_classes = counts[counts / adata.n_obs <= 0.05]
             
             print(f"\n[Label Statistics based on '{label_col}']")
             print(f"  - Cluster Count (K): {K}")
             print(f"  - Max/Min Class Imbalance Ratio: {imbalance_ratio:.2f}x")
             print(f"  - Extremely Rare Classes (<=5%): {len(rare_classes)}")
             if len(rare_classes) > 0:
                 print(f"  - Rare Class Breakdown: \n{rare_classes.to_string()}")
             
        print("="*60)

    def run(self):
        if not os.path.exists(self.input_path):
            print(f"Error: Input file not found {self.input_path}")
            return False

        print(f"Loading data from: {self.input_path}")
        self.adata = sc.read_h5ad(self.input_path)
        
        # 1. Identify Ground Truth column
        obs_keys = self.adata.obs.keys()
        label_col = next((col for col in ['author_cell_type', 'cell_type_original', 'cell_type'] if col in obs_keys), None)
        
        if not label_col:
            print("Error: Valid cell label column not found in observation metadata.")
            return False
            
        self._print_report("Original Dataset Status", self.adata, label_col)
        print("\nExecuting preprocessing and long-tail validation...")
        
        # 2. Dense Matrix Protection
        if not sp.issparse(self.adata.X):
             print("  -> Warning: Dense matrix detected. Forcing conversion to CSR sparse format...")
             self.adata.X = sp.csr_matrix(self.adata.X)
             
        # 3. Label cleaning and isolation removal
        valid_cells = self.adata.obs[label_col].notna() & ~self.adata.obs[label_col].astype(str).str.contains("unknown|unassigned", case=False)
        self.adata = self.adata[valid_cells].copy() 
        if self.adata.obs[label_col].dtype.name == 'category':
            self.adata.obs[label_col] = self.adata.obs[label_col].cat.remove_unused_categories()
            
        counts = self.adata.obs[label_col].value_counts()
        self.K = len(counts)
        imbalance_ratio = counts.max() / counts.min()
        rare_classes = counts[counts / self.adata.n_obs <= 0.05]

        # 4. Long-tail feature assertion
        if self.K < 3 or self.K > 20:
            print(f"Validation Failed: Cluster count ({self.K}) must be between 3 and 20.")
            return False
        if imbalance_ratio < 3.0:
            print(f"Validation Failed: Imbalance ratio ({imbalance_ratio:.1f}x) is too low for rare cell evaluation.")
            return False
        if len(rare_classes) < 1:
            print("Validation Failed: No rare classes detected (<=5%). Unsuitable for strict F1 evaluation.")
            return False
            
        self.adata.obs['ground_truth'] = self.adata.obs[label_col].astype('category')

        # 5. Raw component extraction: Restore raw attributes if present in adata.raw
        if self.adata.raw is not None:
             print("  -> Detected adata.raw backup. Restoring full original gene space...")
             
             saved_obs = self.adata.obs.copy() 
             saved_obsm = self.adata.obsm.copy()
             saved_uns = self.adata.uns.copy()
             saved_obsp = self.adata.obsp.copy() if hasattr(self.adata, 'obsp') else None
             
             raw_adata = self.adata.raw.to_adata().copy()
             raw_adata = raw_adata[saved_obs.index].copy()
             
             raw_adata.obs = saved_obs
             raw_adata.obsm = saved_obsm
             raw_adata.uns = saved_uns
             if saved_obsp is not None:
                 raw_adata.obsp = saved_obsp
                 
             self.adata = raw_adata
             
             if not sp.issparse(self.adata.X):
                 self.adata.X = sp.csr_matrix(self.adata.X)

        # 6. Data status detection and layer backup
        X_data = self.adata.X.data if sp.issparse(self.adata.X) else self.adata.X
        is_scaled = np.min(X_data) < 0
        is_integer = np.allclose(X_data, X_data.astype(int)) if len(X_data) > 0 else False
        is_logged = (np.max(X_data) < 100) and (not is_integer)
        
        if 'raw_counts' not in self.adata.layers:
            if not is_scaled and not is_logged:
                self.adata.layers["raw_counts"] = self.adata.X.copy()
            elif 'counts' in self.adata.layers:
                self.adata.layers["raw_counts"] = self.adata.layers['counts'].copy()
            else:
                self.adata.layers["raw_counts"] = self.adata.X.copy()

        # 7 & 8. HVG distribution alignment processing
        has_raw_counts = False
        if 'raw_counts' in self.adata.layers:
            layer_data = self.adata.layers['raw_counts'].data if sp.issparse(self.adata.layers['raw_counts']) else self.adata.layers['raw_counts']
            if len(layer_data) > 0 and np.allclose(layer_data[:10], layer_data[:10].astype(int)):
                has_raw_counts = True

        if self.adata.n_vars > 4500:
            if has_raw_counts:
                print("  -> Running Poisson modeling (seurat_v3) to extract highly variable features...")
                self.adata.X = self.adata.layers["raw_counts"].copy()
                sc.pp.filter_genes(self.adata, min_cells=1)
                # seurat_v3 must be executed prior to log transformation
                sc.pp.highly_variable_genes(self.adata, n_top_genes=4000, flavor='seurat_v3', subset=False)
                
                print("  -> HVG extraction complete. Executing global normalization and log transformation...")
                self.adata.X = self.adata.X.astype(np.float32) 
                sc.pp.normalize_total(self.adata, target_sum=1e4)
                sc.pp.log1p(self.adata)
            else:
                if not is_logged:
                    print("  -> Raw integer matrix absent. Forcing normalization and log transformation...")
                    self.adata.X = self.adata.X.astype(np.float32) 
                    sc.pp.normalize_total(self.adata, target_sum=1e4)
                    sc.pp.log1p(self.adata)
                    is_logged = True  
                
                print("  -> Expanding feature pool to 4500 to protect long-tail distribution (using standard seurat)...")
                sc.pp.highly_variable_genes(self.adata, n_top_genes=4500, flavor='seurat', subset=False)
        else:
            if not is_logged:
                print("  -> Bypassing HVG selection. Executing global normalization and log transformation...")
                self.adata.X = self.adata.X.astype(np.float32) 
                sc.pp.normalize_total(self.adata, target_sum=1e4)
                sc.pp.log1p(self.adata)
             
        # 9. PCA reduction based on HVG
        print("  -> Computing core Principal Component Analysis (X_pca)...")
        has_hvg = 'highly_variable' in self.adata.var.columns
        sc.tl.pca(self.adata, n_comps=50, use_highly_variable=has_hvg, random_state=42)
        X_cpce_input = self.adata.obsm['X_pca'][:, :50]
        
        # 10. Construct standardized processed output
        clean_adata = sc.AnnData(
            X=self.adata.X, 
            obs=self.adata.obs[['ground_truth']].copy(),
            var=self.adata.var.copy(),  
            layers={"raw_counts": self.adata.layers.get("raw_counts", self.adata.X)}
        )
        clean_adata.obsm['X_pca'] = X_cpce_input 
        clean_adata.uns['n_clusters_gt'] = int(self.K) 
        
        # 11. Visual Manifold Assets Management
        has_vis = False
        for key in ['X_umap', 'X_tsne', 'X_draw_graph_fa']:
             if key in self.adata.obsm:
                 clean_adata.obsm[key] = self.adata.obsm[key]
                 has_vis = True
                 break
                 
        if not has_vis:
            print("  -> Valid visual coordinates not found. Generating UMAP locally...")
            sc.pp.neighbors(clean_adata, use_rep='X_pca', random_state=42)
            sc.tl.umap(clean_adata, random_state=42)
            
        self._print_report("Processed Dataset Status", clean_adata, label_col='ground_truth')
        
        clean_adata.write_h5ad(self.output_path)
        print(f"Success! Standardized dataset saved to: {self.output_path}")
        return True

if __name__ == "__main__":
    # Updated paths targeting Dataset 7
    INPUT_FILE = "data/raw/dataset_7_raw.h5ad" 
    OUTPUT_FILE = "data/processed/dataset_7.h5ad"
    
    pipeline = DataPipeline(INPUT_FILE, OUTPUT_FILE)
    pipeline.run()
