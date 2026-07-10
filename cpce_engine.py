# -*- coding: utf-8 -*-
"""
File: src/cpce_engine.py
Purpose: Core implementation of the Capacity-Profile Clustering Ensemble (CPCE).
"""

import os
# Constrain low-level threading to prevent C-library deadlocks
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import heapq
import numpy as np
import scipy.sparse as sp
import scanpy as sc
import anndata as ad
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors

def _compute_density_peaks(X):
    """
    Pre-computation phase: Calculate global density peaks (DPC Anchors).
    """
    N = X.shape[0]
    # Revert to a stable number of nearest neighbors to avoid noise
    nbrs = NearestNeighbors(n_neighbors=15, algorithm='auto').fit(X)
    distances, _ = nbrs.kneighbors(X)
    rho = 1.0 / (distances.mean(axis=1) + 1e-8) 

    delta = np.zeros(N)
    sorted_rho_idx = np.argsort(rho)[::-1]
    
    delta[sorted_rho_idx[0]] = np.max(euclidean_distances(X[sorted_rho_idx[0]].reshape(1, -1), X))
    for i in range(1, N):
        idx = sorted_rho_idx[i]
        higher_density_points = X[sorted_rho_idx[:i]]
        min_dist = np.min(np.linalg.norm(higher_density_points - X[idx], axis=1))
        delta[idx] = min_dist

    gamma = rho * delta
    return gamma, rho

def _gs_zipf_assignment(X, K, size_max, centers):
    """
    Allocation phase: Dynamic capacity alignment + Local asymmetric exemption 
    + Gale-Shapley stable matching.
    """
    N = X.shape[0]

    dist_to_centers = euclidean_distances(X, centers)
    nearest_center_idx = np.argmin(dist_to_centers, axis=1)
    
    natural_sizes = np.bincount(nearest_center_idx, minlength=K)
    sorted_k_indices = np.argsort(natural_sizes)[::-1]
    sorted_size_max = np.sort(size_max)[::-1]
    
    matched_size_max = np.zeros(K, dtype=int)
    for rank, k_idx in enumerate(sorted_k_indices):
        zipf_quota = sorted_size_max[rank]
        natural_size = natural_sizes[k_idx]
        
        # Local asymmetric exemption to prevent dominant class overflow
        if rank < max(2, K // 5) and natural_size > zipf_quota:
            matched_size_max[k_idx] = max(zipf_quota, int(natural_size * 1.05))
        else:
            matched_size_max[k_idx] = zipf_quota

    preferences = np.argsort(dist_to_centers, axis=1) 
    pointer = np.zeros(N, dtype=int)
    
    free_cells = list(range(N))
    cluster_heaps = {k: [] for k in range(K)}

    while free_cells:
        c = free_cells.pop()
        pref_idx = pointer[c]

        # Forced assignment if preference list is exhausted
        if pref_idx >= K:
            current_sizes = [len(cluster_heaps[k]) for k in range(K)]
            best_k = np.argmax(matched_size_max - np.array(current_sizes))
            heapq.heappush(cluster_heaps[best_k], (-dist_to_centers[c, best_k], c))
            continue

        k = preferences[c, pref_idx]
        d = dist_to_centers[c, k]
        heapq.heappush(cluster_heaps[k], (-d, c))

        # Evict the furthest cell if cluster capacity is exceeded
        if len(cluster_heaps[k]) > matched_size_max[k]:
            kicked_neg_d, kicked_c = heapq.heappop(cluster_heaps[k])
            pointer[kicked_c] += 1
            free_cells.append(kicked_c)

    labels = np.full(N, -1, dtype=int)
    for k in range(K):
        for neg_d, c in cluster_heaps[k]:
            labels[c] = k

    return labels

def run_synthesized_cpce(X_pca, K, T=10, alpha_max=2.0, seed=42, template_type='zipf', variant='full'):
    """
    variant parameters:
    - 'full': Full model.
    - 'no_dpc': Ablate Density Peak anchoring.
    - 'no_zipf': Ablate Zipfian capacity constraints.
    - 'no_gs': Ablate Gale-Shapley matching (use greedy nearest neighbor).
    - 'avg_link': Replace Complete-linkage with Average-linkage.
    """
    N = X_pca.shape[0]
    alphas = np.linspace(1.0, alpha_max + 1.0, T)
    
    # ==========================================
    # [Variant 1: No-DPC] Remove Density Peak Protection
    # ==========================================
    if variant == 'no_dpc':
        np.random.seed(seed)
        random_idx = np.random.choice(N, K, replace=False)
        base_centers = X_pca[random_idx]
    else:
        # Full model: Extract high-potential density anchors
        gamma, rho = _compute_density_peaks(X_pca)
        M = min(N, max(K * 5, 100))
        candidate_indices = np.argsort(gamma)[-M:]
        candidate_peaks = X_pca[candidate_indices]
        
        # Eliminate KMeans stochasticity; lock isolated anchors
        agg_anchor = AgglomerativeClustering(n_clusters=K, metric='euclidean', linkage='average')
        peak_labels = agg_anchor.fit_predict(candidate_peaks)
        base_centers = np.array([candidate_peaks[peak_labels == l].mean(axis=0) for l in range(K)])
    
    labels_list = []
    
    for t in range(T):
        if template_type == 'zipf':
            shift = max(0.0, (K - 12) * 0.15)
            k_ranks = np.arange(1, K + 1) + shift
            q_unnorm = k_ranks ** (-alphas[t])
            q_t = q_unnorm / np.sum(q_unnorm)
        else:
            q_t = np.ones(K) / K
        
        # ==========================================
        # [Variant 2: No-Zipf] Remove Physical Constraints
        # ==========================================
        if variant == 'no_zipf':
            # Release capacity to infinity, allowing majority absorption
            size_max = np.full(K, N + 1).tolist()
        else:
            # Full model: Strict Capacity Bounding
            adaptive_buffer = max(10, min(60, int(N * 0.0015)))
            if N > 20000:
                floor_size = max(15, min(40, int(N * 0.001)))
            else:
                floor_size = max(30, min(150, int(N * 0.003)))
                
            size_max = np.maximum(np.ceil(q_t * N * 1.05) + adaptive_buffer, floor_size).astype(int).tolist()
        
        # Simulated physical annealing jitter
        np.random.seed(seed + t)
        noise_scale = 0.1 / K
        noise = np.random.normal(0, noise_scale, base_centers.shape) 
        centers_t = base_centers + noise
        
        # ==========================================
        # [Variant 4: No-GS] Remove Gale-Shapley Stable Matching
        # ==========================================
        if variant == 'no_gs':
            # Greedy Allocation (First-come, first-served)
            labels_t = np.full(N, -1, dtype=int)
            cluster_counts = np.zeros(K, dtype=int)
            dist_to_centers = euclidean_distances(X_pca, centers_t)
            
            # Shuffle cells to simulate disordered access
            np.random.seed(seed + t)
            cell_order = np.random.permutation(N)
            
            for c in cell_order:
                prefs = np.argsort(dist_to_centers[c])
                for p in prefs:
                    if cluster_counts[p] < size_max[p]:
                        labels_t[c] = p
                        cluster_counts[p] += 1
                        break
                # Fallback if all preferred capacities are technically full
                if labels_t[c] == -1:
                    labels_t[c] = prefs[0]
        else:
            # Full model: Gale-Shapley Deferred Acceptance 
            labels_t = _gs_zipf_assignment(X_pca, K, size_max, centers_t)
            
        labels_list.append(labels_t)
        
    # 3. Bipartite Graph Embedding
    one_hot_list = []
    for t in range(T):
        mat = np.zeros((N, K), dtype=np.float32)
        mat[np.arange(N), labels_list[t]] = 1.0 / np.sqrt(T) 
        one_hot_list.append(mat)
        
    X_ensemble = np.hstack(one_hot_list) 
    
    # Fixed n_neighbors=15 to prevent connectivity oscillations
    adata_dummy = ad.AnnData(X=X_ensemble)
    sc.pp.neighbors(adata_dummy, n_neighbors=15, use_rep='X', metric='cosine', random_state=seed)
    
    low, high = 0.01, 5.0 
    final_labels = None
    best_labels = labels_list[0] 
    min_k_diff = float('inf')
    
    # 4. Adaptive Leiden Consensus Partitioning
    for iteration in range(15):
        mid = (low + high) / 2.0
        sc.tl.leiden(adata_dummy, resolution=mid, key_added='tmp_leiden', directed=False, random_state=seed)
        current_labels = adata_dummy.obs['tmp_leiden'].values.astype(int)
        n_cl = len(np.unique(current_labels))
        
        if n_cl >= K and (n_cl - K) < min_k_diff:
            min_k_diff = n_cl - K
            best_labels = current_labels.copy()
            
        if n_cl == K:
            final_labels = current_labels
            break
        elif n_cl < K:
            low = mid  
        else:
            high = mid 

    # 5. Topologically safe manifold consolidation (Fallback)
    if final_labels is None:
        unique_labels = np.unique(best_labels)
        
        if len(unique_labels) > K:
            centers = np.array([X_pca[best_labels == l].mean(axis=0) for l in unique_labels])
            
            # ==========================================
            # [Variant 3: Average-Link] Force ablation of Complete-linkage
            # ==========================================
            if variant == 'avg_link':
                chosen_linkage = 'average'
            else:
                # Full model: Complete-linkage to protect long-tail clusters
                chosen_linkage = 'complete'
            
            clf_fallback = AgglomerativeClustering(n_clusters=K, metric='euclidean', linkage=chosen_linkage)
            center_mapped_labels = clf_fallback.fit_predict(centers)
            
            final_labels = np.zeros(N, dtype=int)
            for orig_l, new_l in zip(unique_labels, center_mapped_labels):
                final_labels[best_labels == orig_l] = new_l
        else:
            final_labels = best_labels
            
    return final_labels
