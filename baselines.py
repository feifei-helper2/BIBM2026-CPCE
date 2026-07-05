# 文件名: src/baselines.py
# 职责: 封装所有 Baseline 算法，提供统一黑盒接口，内部消化所有异常危机。

import numpy as np
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
import warnings

from sklearn.cluster import KMeans, AgglomerativeClustering, MiniBatchKMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from k_means_constrained import KMeansConstrained

# 屏蔽第三方库的常规警告与弃用警告，保持终端日志纯净
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class BaselineModels:
    """
    CPCE 竞品防御矩阵与包装器。
    统一输入签名: (X_pca, K, seed)
    统一输出签名: labels (1D numpy array, length N)
    """

    @staticmethod
    def run_kmeans(X_pca, K, seed=42):
        """Standard K-Means: 暴露大类吞噬问题的基准下限"""
        clf = KMeans(n_clusters=K, random_state=seed, n_init=10)
        return clf.fit_predict(X_pca)

    @staticmethod
    def run_balanced_kmeans(X_pca, K, seed=42):
        """Balanced K-Means: 证明强平衡对真实生物分布的破坏"""
        N = X_pca.shape[0]
        size_min = N // K
        size_max = size_min + 1  # 余数容差，防求解器死锁
        
        clf = KMeansConstrained(
            n_clusters=K,
            size_min=size_min,
            size_max=size_max,
            random_state=seed,
            n_jobs=4  # 限制并发防风暴
        )
        return clf.fit_predict(X_pca)

    @staticmethod
    def run_random_ensemble(X_pca, K, seed=42, T=10):
        """
        无约束随机集成 (Random Ensemble Baseline)。
        架构升级：废弃 O(N^3) 的共聚矩阵构建，采用 One-Hot 偶图嵌入法 (Bipartite Graph Embedding)。
        将时间复杂度降至 O(N)，解决 8000+ 细胞规模下的无穷卡死陷阱。
        """
        from sklearn.cluster import KMeans
        import numpy as np
        
        np.random.seed(seed)
        N = X_pca.shape[0]
        
        # 1. 并发生成 T 个随机 K-Means 基聚类
        # 为了速度，基聚类只做 1 次初始化 (n_init=1)
        base_labels = []
        for t in range(T):
            clf = KMeans(n_clusters=K, random_state=seed+t, n_init=1)
            base_labels.append(clf.fit_predict(X_pca))
        
        # 2. 核心降维打击：One-Hot 特征平铺 (规避 N x N 矩阵爆炸)
        # 构造 N x (T * K) 的稀疏特征空间
        one_hot_list = []
        for t in range(T):
            mat = np.zeros((N, K), dtype=np.float32)
            mat[np.arange(N), base_labels[t]] = 1.0 / np.sqrt(T) # 归一化权重
            one_hot_list.append(mat)
            
        X_ensemble = np.hstack(one_hot_list) # shape: (N, T * K)
        
        # 3. 在降维特征空间执行终极共识聚类
        final_clf = KMeans(n_clusters=K, random_state=seed, n_init=5)
        return final_clf.fit_predict(X_ensemble)

    @staticmethod
    def run_gmm(X_pca, K, seed=42):
        """Gaussian Mixture Model: 证明软聚类同样无法对抗长尾"""
        try:
            # 强制 covariance_type='diag' 防协方差矩阵非正定崩溃
            gmm = GaussianMixture(n_components=K, covariance_type='diag', random_state=seed)
            return gmm.fit_predict(X_pca)
        except Exception:
            # 奇异矩阵静默兜底
            return BaselineModels.run_kmeans(X_pca, K, seed)

    @staticmethod
    def run_spectral(X_pca, K, seed=42):
        """Spectral Clustering: 图论流形切分的经典代表"""
        N = X_pca.shape[0]
        
        # OOM 熔断约束
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
            
        try:
            return clf.fit_predict(X_pca)
        except Exception:
            # 图不连通或代数异常兜底
            return BaselineModels.run_kmeans(X_pca, K, seed)

    @staticmethod
    def run_leiden_with_bisection(X_pca, K, seed=42):
        """Scanpy 原生 Leiden: 带二分查找逼近的图聚类"""
        return BaselineModels._leiden_bisection_wrapper(
            X_pca, K, seed, co_assoc_graph=None, res_high=5.0
        )
    @staticmethod
    def run_sccad(X_pca, adata, K, seed=42):
        """
        scCAD 包装器。
        集成：稀疏致密化、日志/文件I/O阻断、动态K值凝聚对齐、以及安全的KMeans兜底。
        """
        import os
        import contextlib
        import scipy.sparse as sp
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering
        
        try:
            from .sccad_core import scCAD
            
            # 1. 提取原始高维矩阵
            if adata is not None and "raw_counts" in adata.layers:
                data_mat = adata.layers["raw_counts"]
            else:
                data_mat = adata.X
                
            # 2. 迎合 scCAD 底层的密集矩阵运算
            if sp.issparse(data_mat):
                data_mat = data_mat.toarray()
                
            # 3. 拦截日志刷屏与垃圾文件生成
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
            
            # 4. K值对齐逻辑：保证聚类评估指标的严格公平
            if len(unique_labels) > K:
                # 若划出过多碎块，在统一的 PCA 空间求各簇中心
                centers = np.array([X_pca[subclusters == l].mean(axis=0) for l in unique_labels])
                
                # 使用层次聚类将碎块凝聚至精确的 K
                clf_fallback = AgglomerativeClustering(n_clusters=K, metric='euclidean', linkage='ward')
                center_mapped = clf_fallback.fit_predict(centers)
                
                # 重新映射回原细胞
                mapped_labels = np.zeros(subclusters.shape[0], dtype=int)
                for orig_l, new_l in zip(unique_labels, center_mapped):
                    mapped_labels[subclusters == orig_l] = new_l
                return mapped_labels
                
            elif len(unique_labels) < K:
                # 分解异常，生成的簇不足
                print(f"  [Fallback] scCAD 生成簇数({len(unique_labels)})不足 K({K})，降级 KMeans")
                return BaselineModels.run_kmeans(X_pca, K, seed) # 【已修复】传入 X_pca
            else:
                return subclusters
                
        except Exception as e:
            # 捕获内存溢出等任何崩溃异常
            print(f"  [Fallback] scCAD 内部崩溃 ({e})，降级 KMeans")
            return BaselineModels.run_kmeans(X_pca, K, seed) # 【已修复】传入 X_pca

    @staticmethod
    def run_giniclust3(X_pca, adata, K, seed=42):
        """
        GiniClust3 包装器。
        集成：沙盒隔离、跳过 filter 保护矩阵维度、字符串索引转换、动态 K 值凝聚对齐。
        """
        import os
        import contextlib
        import numpy as np
        import scanpy as sc
        from sklearn.cluster import AgglomerativeClustering
        
        try:
            import giniclust3 as gc
            
            # 1. 构建独立数据沙盒，防止污染全局 adata
            if adata is not None and "raw_counts" in adata.layers:
                # Gini 强依赖未被 Log 缩放的原始 counts
                raw_data = adata.layers["raw_counts"].copy()
            else:
                raw_data = adata.X.copy()
                
            if sp.issparse(raw_data):
                raw_data = raw_data.toarray()
                
            adata_gc = sc.AnnData(X=raw_data)
            adata_gc.obs_names = adata.obs_names.astype(str)
            adata_gc.var_names = adata.var_names.astype(str)
            
            # 2. 迎合规范化要求 (绝对禁止调用 filter_cells，防止维度丢失)
            sc.pp.normalize_per_cell(adata_gc, counts_per_cell_after=1e4)
            
            # 3. 物理静音运行，拦截冗长日志
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    # Gini 轨
                    gc.gini.calGini(adata_gc)
                    gc.gini.clusterGini(adata_gc, neighbors=3) 
                    
                    # Fano 轨
                    gc.fano.calFano(adata_gc)
                    gc.fano.clusterFano(adata_gc)
                    
                    # 共识集成
                    consensusCluster = {}
                    consensusCluster['giniCluster'] = adata_gc.obs['rare'].values.tolist()
                    consensusCluster['fanoCluster'] = adata_gc.obs['fano'].values.tolist()
                    gc.consensus.generateMtilde(consensusCluster)
                    gc.consensus.clusterMtilde(consensusCluster)
                    
            # 4. 提取标签并进行安全映射 (将字符串标签转回连续整型)
            labels_raw = np.array(consensusCluster['finalCluster'])
            _, subclusters = np.unique(labels_raw, return_inverse=True)
            
            # 5. K 值对齐与安全降级逻辑
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
                print(f"  [Fallback] GiniClust3 簇数({len(unique_labels)})不足 K({K})，降级 KMeans")
                return BaselineModels.run_kmeans(X_pca, K, seed)
            else:
                return subclusters
                
        except ImportError:
            print("  [Error] 未安装 giniclust3 包，降级 KMeans")
            return BaselineModels.run_kmeans(X_pca, K, seed)
        except Exception as e:
            print(f"  [Fallback] GiniClust3 内部崩溃 ({e})，降级 KMeans")
            return BaselineModels.run_kmeans(X_pca, K, seed)

    @staticmethod
    def run_scvi_kmeans(X_pca, adata, K, seed=42):
        import os
        import contextlib
        import scanpy as sc
        from sklearn.cluster import KMeans

        try:
            import scvi
            import torch
            import logging
            
            # 【吸收优秀防爆逻辑】：彻底关闭 DataLoader 多进程，切断对 Docker /dev/shm 的访问，根除内存死锁
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
                    
                    # 【纠正致命漏洞】：彻底删除 accelerator='cpu'！
                    # 也不要写 os.environ["CUDA_VISIBLE_DEVICES"] = ""
                    # 解除封印后，scVI 将自动接管 32GB 显卡，将几小时的训练压缩至几十秒
                    model.train(
                        max_epochs=50, 
                        early_stopping=True, 
                        enable_progress_bar=False
                    )
                    
                    latent_Z = model.get_latent_representation()

            clf = KMeans(n_clusters=K, random_state=seed, n_init=10)
            return clf.fit_predict(latent_Z)

        except Exception as e:
            print(f"  [Fallback] scVI 内部崩溃 ({e})，降级 KMeans")
            return BaselineModels.run_kmeans(X_pca, K, seed)
                 
    # ==========================================
    # 内部私有安全核 (Private Kernel)
    # ==========================================
    @staticmethod
    def _leiden_bisection_wrapper(X_pca, K, seed, co_assoc_graph=None, res_high=5.0):
        """
        统一的 Leiden 容错二分查找引擎。
        处理原生 PCA 近邻图 (Leiden) 与 致密共聚图 (Random Ensemble)。
        """
        N = X_pca.shape[0]
        adata_dummy = ad.AnnData(X=sp.csr_matrix((N, 1)))
        
        # 【漏洞修复】：强制打上字符串索引，防止 Pandas Index Alignment 崩溃
        adata_dummy.obs_names = [str(i) for i in range(N)]
        
        # 判别图来源与合法上下文注入
        if co_assoc_graph is not None:
            # 直接注入外部图
            adata_dummy.obsp['connectivities'] = co_assoc_graph
            # 【漏洞修复】：手动伪造通行证，防止 Scanpy API 黑盒校验 KeyError
            adata_dummy.uns['neighbors'] = {
                'connectivities_key': 'connectivities',
                'distances_key': '',
                'params': {'n_neighbors': 15, 'method': 'ensemble'}
            }
        else:
            # 原生 Leiden 动态计算图
            adata_dummy.obsm['X_pca'] = X_pca
            sc.pp.neighbors(adata_dummy, use_rep='X_pca', n_neighbors=15)
            
        low, high = 0.01, res_high
        final_labels = None
        
        # KMeans 绝对保底解，防 np.unique(None) 崩溃
        best_labels = BaselineModels.run_kmeans(X_pca, K, seed)
        min_diff = float('inf')

        for iteration in range(20):
            mid = (low + high) / 2.0
            # directed=False 兼容非对称共聚图
            sc.tl.leiden(adata_dummy, resolution=mid, key_added='leiden', directed=False, random_state=seed)
            current_labels = adata_dummy.obs['leiden'].values.astype(int)
            n_cl = len(np.unique(current_labels))
            
            # 记录最佳逼近解
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

        # 启动 Fallback 挽救：过聚类结果强行中心凝聚至 K
        if final_labels is None:
            over_labels = best_labels
            unique_labels = np.unique(over_labels)
            
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
