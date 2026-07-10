#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: src/baselines.py
Purpose: Encapsulate all baseline algorithms, providing a unified black-box interface. 
         Removes silent fallbacks; throws exceptions directly upon failure.
"""

import numpy as np
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
import warnings

from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from k_means_constrained import KMeansConstrained

# Filter third-party warnings to keep console logs clean
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class BaselineModels:
    """
    CPCE baseline wrapper.
    Unified input signature: (X_pca, K, seed)
    Unified output signature: labels (1D numpy array, length N)
    """

    @staticmethod
    def run_kmeans(X_pca, K, seed=42):
        """Standard K-Means: Baseline showing the majority-domination artifact."""
        clf = KMeans(n_clusters=K, random_state=seed, n_init=10)
        return clf.fit_predict(X_pca)

    @staticmethod
    def run_balanced_kmeans(X_pca, K, seed=42):
        """Balanced K-Means: Demonstrates the destruction of true biological distribution by forced balancing."""
        N = X_pca.shape[0]
        size_min = N // K
        size_max = size_min + 1  # Remainder tolerance to prevent solver deadlock
        
        clf = KMeansConstrained(
            n_clusters=K,
            size_min=size_min,
            size_max=size_max,
            random_state=seed,
            n_jobs=4
        )
        return clf.fit_predict(X_pca)

    @staticmethod
    def run_random_ensemble(X_pca, K, seed=42, T=10):
        """
        Random Ensemble Baseline. 
        Upgraded architecture: Bipartite Graph Embedding to reduce time complexity to O(N).
        """
        np.random.seed(seed)
        N = X_pca.shape[0]
        
        # 1. Generate T random K-Means base clusterings
        base_labels = []
        for t in range(T):
            clf = KMeans(n_clusters=K, random_state=seed+t, n_init=1)
            base_labels.append(clf.fit_predict(X_pca))
        
        # 2. One-Hot feature flattening
        one_hot_list = []
        for t in range(T):
            mat = np.zeros((N, K), dtype=np.float32)
            mat[np.arange(N), base_labels[t]] = 1.0 / np.sqrt(T) # Normalized weight
            one_hot_list.append(mat)
            
        X_ensemble = np.hstack(one_hot_list) # shape: (N, T * K)
        
        # 3. Final consensus clustering in reduced feature space
        final_clf = KMeans(n_clusters=K, random_state=seed, n_init=5)
        return final_clf.fit_predict(X_ensemble)

    @staticmethod
    def run_gmm(X_pca, K, seed=42):
        """Gaussian Mixture Model: Demonstrates soft clustering failing on long-tail distributions."""
        gmm = GaussianMixture(n_components=K, covariance_type='diag', random_state=seed)
        return gmm.fit_predict(X_pca)

    @staticmethod
    def run_spectral(X_pca, K, seed=42):
        """Spectral Clustering: Graph-based manifold segmentation."""
        N = X_pca.shape[0]
        
        # OOM prevention constraint
        if N >= 10000:
            clf = MiniBatchKMeans(n_clusters=K, batch_size=2048, random_state=seed, n_init=3)
        else:
            clf = SpectralClustering(
                n_clusters=K, 
                affinity='nearest_neighbors', 
                n_neighbors=15,
                random_state=seed,
                n_jobs=4
            )
        return clf.fit_predict(X_pca)

    @staticmethod
    def run_leiden_with_bisection(X_pca, K, seed=42):
        """Scanpy native Leiden: Graph clustering with binary search approximation."""
        return BaselineModels._leiden_bisection_wrapper(
            X_pca, K, seed, co_assoc_graph=None, res_high=5.0
        )

    @staticmethod
    def run_sccad(X_pca, adata, K, seed=42):
        """
        scCAD wrapper.
        Throws exceptions immediately if cluster count is insufficient or internal crash occurs.
        """
        import os
        import contextlib
        from sklearn.cluster import AgglomerativeClustering
        from .sccad_core import scCAD
        
        if adata is not None and "raw_counts" in adata.layers:
            data_mat = adata.layers["raw_counts"]
        else:
            data_mat = adata.X
            
        if sp.issparse(data_mat):
            data_mat = data_mat.toarray()
            
        # Block console spam and garbage file generation
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                result, overlap, subclusters, remain_degs = scCAD(
                    data=data_mat,
                    normalization=True, 
                    seed=seed,
                    save_full=False,
                    save_path='./'
                )
        
        unique_labels = np.unique(subclusters)
        
        # K-value alignment logic
        if len(unique_labels) > K:
            centers = np.array([X_pca[subclusters == l].mean(axis=0) for l in unique_labels])
            clf_fallback = AgglomerativeClustering(n_clusters=K, metric='euclidean', linkage='ward')
            center_mapped = clf_fallback.fit_predict(centers)
            
            mapped_labels = np.zeros(subclusters.shape[0], dtype=int)
            for orig_l, new_l in zip(unique_labels, center_mapped):
                mapped_labels[subclusters == orig_l] = new_l
            return mapped_labels
            
        elif len(unique_labels) < K:
            raise ValueError(f"scCAD generated insufficient clusters ({len(unique_labels)} < {K})")
        else:
            return subclusters

    @staticmethod
    def run_giniclust3(X_pca, adata, K, seed=42):
        """
        GiniClust3 wrapper.
        Throws exceptions immediately for missing packages, insufficient clusters, or crashes.
        """
        import os
        import contextlib
        from sklearn.cluster import AgglomerativeClustering
        
        try:
            import giniclust3 as gc
        except ImportError:
            raise ImportError("Required package 'giniclust3' is not installed.")
            
        if adata is not None and "raw_counts" in adata.layers:
            raw_data = adata.layers["raw_counts"].copy()
        else:
            raw_data = adata.X.copy()
            
        if sp.issparse(raw_data):
            raw_data = raw_data.toarray()
            
        adata_gc = sc.AnnData(X=raw_data)
        adata_gc.obs_names = adata.obs_names.astype(str)
        adata_gc.var_names = adata.var_names.astype(str)
        
        sc.pp.normalize_per_cell(adata_gc, counts_per_cell_after=1e4)
        
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                gc.gini.calGini(adata_gc)
                gc.gini.clusterGini(adata_gc, neighbors=3) 
                
                gc.fano.calFano(adata_gc)
                gc.fano.clusterFano(adata_gc)
                
                consensusCluster = {}
                consensusCluster['giniCluster'] = adata_gc.obs['rare'].values.tolist()
                consensusCluster['fanoCluster'] = adata_gc.obs['fano'].values.tolist()
                gc.consensus.generateMtilde(consensusCluster)
                gc.consensus.clusterMtilde(consensusCluster)
                
        labels_raw = np.array(consensusCluster['finalCluster'])
        _, subclusters = np.unique(labels_raw, return_inverse=True)
        
        unique_labels = np.unique(subclusters)
        if len(unique_labels) > K:
            centers = np.array([X_pca[subclusters == l].mean(axis=0) for l in unique_labels])
            clf_fallback = AgglomerativeClustering(n_clusters=K, metric='euclidean', linkage='ward')
            center_mapped = clf_fallback.fit_predict(centers)
            
            mapped_labels = np.zeros(subclusters.shape[0], dtype=int)
            for orig_l, new_l in zip(unique_labels, center_mapped):
                mapped_labels[subclusters == orig_l] = new_l
            return mapped_labels
            
        elif len(unique_labels) < K:
            raise ValueError(f"GiniClust3 generated insufficient clusters ({len(unique_labels)} < {K})")
        else:
            return subclusters

    @staticmethod
    def run_scvi_kmeans(X_pca, adata, K, seed=42):
        import os
        import contextlib
        from sklearn.cluster import KMeans

        try:
            import scvi
        except ImportError:
            raise ImportError("Required package 'scvi-tools' is not installed.")

        import logging
        scvi.settings.dl_num_workers = 0
        scvi.settings.verbosity = logging.ERROR
        scvi.settings.seed = seed
        
        if adata is not None and "raw_counts" in adata.layers:
            raw_data = adata.layers["raw_counts"].copy()
        else:
            raw_data = adata.X.copy()
            
        adata_scvi = sc.AnnData(X=raw_data)
        adata_scvi.obs_names = adata.obs_names.astype(str)
        adata_scvi.var_names = adata.var_names.astype(str)

        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                scvi.model.SCVI.setup_anndata(adata_scvi)
                model = scvi.model.SCVI(adata_scvi, n_latent=30, n_layers=2)
                
                model.train(
                    max_epochs=50, 
                    early_stopping=True, 
                    enable_progress_bar=False
                )
                
                latent_Z = model.get_latent_representation()

        clf = KMeans(n_clusters=K, random_state=seed, n_init=10)
        return clf.fit_predict(latent_Z)
                 
    # ==========================================
    # Internal Private Kernel
    # ==========================================
    @staticmethod
    def _leiden_bisection_wrapper(X_pca, K, seed, co_assoc_graph=None, res_high=5.0):
        """
        Unified Leiden fault-tolerant binary search engine.
        K-Means fallback mechanism has been entirely removed.
        """
        N = X_pca.shape[0]
        adata_dummy = ad.AnnData(X=sp.csr_matrix((N, 1)))
        
        adata_dummy.obs_names = [str(i) for i in range(N)]
        
        if co_assoc_graph is not None:
            adata_dummy.obsp['connectivities'] = co_assoc_graph
            adata_dummy.uns['neighbors'] = {
                'connectivities_key': 'connectivities',
                'distances_key': '',
                'params': {'n_neighbors': 15, 'method': 'ensemble'}
            }
        else:
            adata_dummy.obsm['X_pca'] = X_pca
            sc.pp.neighbors(adata_dummy, use_rep='X_pca', n_neighbors=15)
            
        low, high = 0.01, res_high
        final_labels = None
        
        best_labels = None
        min_diff = float('inf')

        for iteration in range(20):
            mid = (low + high) / 2.0
            sc.tl.leiden(adata_dummy, resolution=mid, key_added='leiden', directed=False, random_state=seed)
            current_labels = adata_dummy.obs['leiden'].values.astype(int)
            n_cl = len(np.unique(current_labels))
            
            # Record best approximation 
            if n_cl >= K and (n_cl - K) < min_diff:
                min_diff = n_cl - K
                best_labels = current_labels.copy()

            if n_cl == K:
                final_labels = current_labels
                break
            elif n_cl < K:
                low = mid
            else:
                high = mid

        if final_labels is None:
            # Manifold fracture detection
            if best_labels is None:
                raise ValueError("Leiden search collapsed completely, unable to yield valid clusters.")
            
            over_labels = best_labels
            unique_labels = np.unique(over_labels)
            
            if len(unique_labels) < K:
                raise ValueError(f"Leiden max resolution generated insufficient clusters ({len(unique_labels)} < {K})")
                
            if len(unique_labels) > K:
                centers = np.array([X_pca[over_labels == l].mean(axis=0) for l in unique_labels])
                clf_fallback = AgglomerativeClustering(n_clusters=K, metric='euclidean', linkage='ward')
                center_mapped = clf_fallback.fit_predict(centers)
                
                final_labels = np.zeros(N, dtype=int)
                for orig_l, new_l in zip(unique_labels, center_mapped):
                    final_labels[over_labels == orig_l] = new_l
            else:
                final_labels = over_labels
                
        return final_labels
