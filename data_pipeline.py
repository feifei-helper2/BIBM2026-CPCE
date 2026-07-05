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
        """物理信息探针与体检报告生成器"""
        print(f"\n{'='*20} {title} {'='*20}")
        print(f"📌 数据概览:")
        print(f"  - 细胞数 (Observations): {adata.n_obs}")
        print(f"  - 基因数 (Variables): {adata.n_vars}")
        
        X_data = adata.X.data if sp.issparse(adata.X) else adata.X
        is_sparse = sp.issparse(adata.X)
        print(f"  - 矩阵类型: {'Sparse (稀疏)' if is_sparse else 'Dense (密集)'}")
        if is_sparse:
             print(f"  - 稀疏度: {1.0 - adata.X.nnz / (adata.n_obs * adata.n_vars):.4f}")
        
        min_val = np.min(X_data) if X_data.size > 0 else 0
        max_val = np.max(X_data) if X_data.size > 0 else 0
        is_logged = max_val < 100
        is_scaled = min_val < 0
        print(f"  - 矩阵状态嗅探: {'包含负数 (Scaled)' if is_scaled else ('已对数化 (Logged)' if is_logged else '原始计数 (Raw Counts)')}")
        
        if label_col and label_col in adata.obs:
             counts = adata.obs[label_col].value_counts()
             K = len(counts)
             imbalance_ratio = counts.max() / counts.min() if counts.min() > 0 else float('inf')
             rare_classes = counts[counts / adata.n_obs <= 0.05]
             
             print(f"\n📊 标签统计 (基于 '{label_col}'):")
             print(f"  - 类别数 K: {K}")
             print(f"  - 最大/最小类失衡比: {imbalance_ratio:.2f} 倍")
             print(f"  - 极稀有类数量 (<=5%): {len(rare_classes)}")
             if len(rare_classes) > 0:
                 print(f"  - 极稀有类明细: \n{rare_classes.to_string()}")
             
        print("="*60)

    def run(self):
        if not os.path.exists(self.input_path):
            print(f"❌ 致命错误：未找到输入文件 {self.input_path}")
            return False

        print(f"🚀 开始加载数据: {self.input_path}")
        self.adata = sc.read_h5ad(self.input_path)
        
        # 1. 真实标签列锁定
        obs_keys = self.adata.obs.keys()
        label_col = next((col for col in ['author_cell_type', 'cell_type_original', 'cell_type'] if col in obs_keys), None)
        
        if not label_col:
            print("❌ 致命错误：未找到有效的细胞标签列。")
            return False
            
        self._print_report("原版数据状态报告", self.adata, label_col)
        print("\n⚙️ 开始执行清洗与长尾免疫处理...")
        
        # 2. 密集矩阵强制防御
        if not sp.issparse(self.adata.X):
             print("  -> 警告：检测到密集矩阵，正强制转换为 CSR 稀疏矩阵...")
             self.adata.X = sp.csr_matrix(self.adata.X)
             
        # 3. 标签清洗与孤立类清除
        valid_cells = self.adata.obs[label_col].notna() & ~self.adata.obs[label_col].astype(str).str.contains("unknown|unassigned", case=False)
        self.adata = self.adata[valid_cells].copy() 
        if self.adata.obs[label_col].dtype.name == 'category':
            self.adata.obs[label_col] = self.adata.obs[label_col].cat.remove_unused_categories()
            
        counts = self.adata.obs[label_col].value_counts()
        self.K = len(counts)
        imbalance_ratio = counts.max() / counts.min()
        rare_classes = counts[counts / self.adata.n_obs <= 0.05]

        # 4. 长尾物理特征断言
        if self.K < 3 or self.K > 20:
            print(f"❌ 淘汰：类别数 {self.K} 不合规 (实验设计需 3~20)。")
            return False
        if imbalance_ratio < 3.0:
            print(f"❌ 淘汰：失衡比 {imbalance_ratio:.1f} 倍过低，长尾特征不明显。")
            return False
        if len(rare_classes) < 1:
            print("❌ 淘汰：毫无极端稀有类别，无法使用 F1 严苛评测体系。")
            return False
            
        self.adata.obs['ground_truth'] = self.adata.obs[label_col].astype('category')

        # 5. 底层解包：防止维度撕裂、死者苏醒与可视化资产丢失
        if self.adata.raw is not None:
             print("  -> 探测到 adata.raw 备份，正解包恢复全量原始基因空间...")
             
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

        # 6. 数据状态探测与备份
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

        # 7 & 8. 融合重构：严格遵守 HVG 算法的数据分布前置要求
        has_raw_counts = False
        if 'raw_counts' in self.adata.layers:
            layer_data = self.adata.layers['raw_counts'].data if sp.issparse(self.adata.layers['raw_counts']) else self.adata.layers['raw_counts']
            if len(layer_data) > 0 and np.allclose(layer_data[:10], layer_data[:10].astype(int)):
                has_raw_counts = True

        if self.adata.n_vars > 4500:
            if has_raw_counts:
                print("  -> 启动泊松建模 (seurat_v3) 提取稀有特征 (基于原始整数矩阵)...")
                self.adata.X = self.adata.layers["raw_counts"].copy()
                sc.pp.filter_genes(self.adata, min_cells=1)
                # seurat_v3 必须在 Log 之前执行
                sc.pp.highly_variable_genes(self.adata, n_top_genes=4000, flavor='seurat_v3', subset=False)
                
                print("  -> 打标完成，执行全基因组级归一化与对数化...")
                self.adata.X = self.adata.X.astype(np.float32) 
                sc.pp.normalize_total(self.adata, target_sum=1e4)
                sc.pp.log1p(self.adata)
            else:
                if not is_logged:
                    print("  -> 缺失整数矩阵，强制先行执行归一化与对数化...")
                    self.adata.X = self.adata.X.astype(np.float32) 
                    sc.pp.normalize_total(self.adata, target_sum=1e4)
                    sc.pp.log1p(self.adata)
                    is_logged = True  
                
                print("  -> 强制扩容常规特征池至 4500 保护长尾 (基于 Log 化矩阵)...")
                # seurat 必须在 Log 之后执行
                sc.pp.highly_variable_genes(self.adata, n_top_genes=4500, flavor='seurat', subset=False)
        else:
            if not is_logged:
                print("  -> 无需 HVG 打标，直接执行全基因组级归一化与对数化...")
                self.adata.X = self.adata.X.astype(np.float32) 
                sc.pp.normalize_total(self.adata, target_sum=1e4)
                sc.pp.log1p(self.adata)
             
        # 9. 基于 HVG 动态切片的 PCA
        print("  -> 计算核心降维矩阵 (X_pca)...")
        has_hvg = 'highly_variable' in self.adata.var.columns
        sc.tl.pca(self.adata, n_comps=50, use_highly_variable=has_hvg, random_state=42)
        X_cpce_input = self.adata.obsm['X_pca'][:, :50]
        
        # 10. 标准件物理组装
        clean_adata = sc.AnnData(
            X=self.adata.X, 
            obs=self.adata.obs[['ground_truth']].copy(),
            var=self.adata.var.copy(),  
            layers={"raw_counts": self.adata.layers.get("raw_counts", self.adata.X)}
        )
        clean_adata.obsm['X_pca'] = X_cpce_input 
        clean_adata.uns['n_clusters_gt'] = int(self.K) 
        
        # 11. 可视化流形资产管理
        has_vis = False
        for key in ['X_umap', 'X_tsne', 'X_draw_graph_fa']:
             if key in self.adata.obsm:
                 clean_adata.obsm[key] = self.adata.obsm[key]
                 has_vis = True
                 break
                 
        if not has_vis:
            print("  -> 未发现有效可视化坐标，正在即时计算 UMAP...")
            sc.pp.neighbors(clean_adata, use_rep='X_pca', random_state=42)
            sc.tl.umap(clean_adata, random_state=42)
            
        self._print_report("清洗后标准件状态报告", clean_adata, label_col='ground_truth')
        
        clean_adata.write_h5ad(self.output_path)
        print(f"🎉 成功！已生成标准化就绪文件: {self.output_path}")
        return True

if __name__ == "__main__":
    # 配置数据路径
    INPUT_FILE = "input1.h5ad" 
    OUTPUT_FILE = "output1.h5ad"
    
    pipeline = DataPipeline(INPUT_FILE, OUTPUT_FILE)
    pipeline.run()
