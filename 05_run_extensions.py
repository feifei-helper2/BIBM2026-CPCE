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

# 物理映射字典
DATASET_MAPPING = {"output3": "dataset_2", "output7": "dataset_4", "output18": "dataset_7"}

def append_csv(record, path):
    df = pd.DataFrame([record])
    df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)

def main():
    print("🚀 启动 BIBM 冲刺扩展实验 (NoGS消融 + 敏感性 + 鲁棒性)...")
    
    ablation_csv = os.path.join(DIR_TABLES, "ablation_results.csv")
    sens_csv = os.path.join(DIR_TABLES, "sensitivity_results.csv")
    rob_csv = os.path.join(DIR_TABLES, "robustness_results.csv")
    
    # ==================================================
    # 模块 1：补齐消融实验 (NoGS)
    # ==================================================
    print("\n--- 模块 1: 补充 NoGaleShapley 消融实验 ---")
    for raw_name, paper_name in DATASET_MAPPING.items():
        adata = sc.read_h5ad(os.path.join(DIR_DATA, f"{raw_name}.h5ad"))
        X_pca, y_true = adata.obsm['X_pca'], adata.obs['ground_truth'].values
        K = len(np.unique(y_true))
        
        for run in range(10):
            seed = 42 + run
            print(f"⚙️ 运行 NoGS 消融 | {paper_name} | Run {run}")
            labels_pred = run_synthesized_cpce(X_pca, K, seed=seed, variant='no_gs')
            metrics = calc_rare_class_f1_strict(y_true, labels_pred)
            record = {
                "Dataset": paper_name, "Variant": "CPCE_NoGS", "Run": run, "Seed": seed,
                "Time_s": np.nan, "ARI": adjusted_rand_score(y_true, labels_pred),
                "NMI": np.nan, "Mean_F1": metrics["Mean_Rare_F1"],
                "Geo_F1": metrics["Soft_Geo_Rare_F1"], "Det_Rate": metrics["Detection_Rate"], "Status": "Success"
            }
            append_csv(record, ablation_csv)

    # 取 dataset_4 (output7) 进行敏感性和鲁棒性测试
    print("\n--- 载入 dataset_4 (output7) 准备敏感性与鲁棒性测试 ---")
    adata = sc.read_h5ad(os.path.join(DIR_DATA, "output7.h5ad"))
    X_pca, y_true = adata.obsm['X_pca'], adata.obs['ground_truth'].values
    K = len(np.unique(y_true))

    # ==================================================
    # 模块 2：参数敏感性分析
    # ==================================================
    print("\n--- 模块 2: 参数敏感性分析 (T & alpha_max) ---")
    
    # 测 T
    for T_val in [3, 5, 10, 15, 20]:
        for run in range(5): # 测 5 次足够了
            print(f"⚙️ 测试 T={T_val} | Run {run}")
            labels_pred = run_synthesized_cpce(X_pca, K, T=T_val, alpha_max=2.0, seed=42+run)
            m = calc_rare_class_f1_strict(y_true, labels_pred)
            append_csv({"Parameter": "T", "Value": T_val, "Run": run, "ARI": adjusted_rand_score(y_true, labels_pred), "Geo_F1": m["Soft_Geo_Rare_F1"]}, sens_csv)
            
    # 测 alpha_max
    for alpha_val in [1.5, 2.0, 2.5, 3.0]:
        for run in range(5):
            print(f"⚙️ 测试 alpha_max={alpha_val} | Run {run}")
            labels_pred = run_synthesized_cpce(X_pca, K, T=10, alpha_max=alpha_val, seed=42+run)
            m = calc_rare_class_f1_strict(y_true, labels_pred)
            append_csv({"Parameter": "alpha_max", "Value": alpha_val, "Run": run, "ARI": adjusted_rand_score(y_true, labels_pred), "Geo_F1": m["Soft_Geo_Rare_F1"]}, sens_csv)

    # ==================================================
    # 模块 3：抗噪鲁棒性分析 (下采样 90%, 70%, 50%)
    # ==================================================
    print("\n--- 模块 3: 细胞级下采样鲁棒性 (Downsampling) ---")
    N_total = len(y_true)
    ratios = [0.9, 0.7, 0.5]
    
    for ratio in ratios:
        for run in range(5):
            np.random.seed(42 + run)
            n_keep = int(N_total * ratio)
            idx = np.random.choice(N_total, n_keep, replace=False)
            
            X_sub, y_sub = X_pca[idx], y_true[idx]
            K_sub = len(np.unique(y_sub)) # 下采样可能导致极小类丢失，动态重置 K
            
            print(f"⚙️ 测试下采样 {int(ratio*100)}% | Run {run} | 剩余细胞 {n_keep} | K={K_sub}")
            
            # 跑 CPCE
            labels_cpce = run_synthesized_cpce(X_sub, K_sub, seed=42+run)
            m_cpce = calc_rare_class_f1_strict(y_sub, labels_cpce)
            append_csv({"Ratio": ratio, "Algorithm": "CPCE", "Run": run, "ARI": adjusted_rand_score(y_sub, labels_cpce), "Geo_F1": m_cpce["Soft_Geo_Rare_F1"]}, rob_csv)
            
            # 跑 KMeans (作为对照组，体现你的降维打击)
            labels_km = KMeans(n_clusters=K_sub, random_state=42+run, n_init=10).fit_predict(X_sub)
            m_km = calc_rare_class_f1_strict(y_sub, labels_km)
            append_csv({"Ratio": ratio, "Algorithm": "KMeans", "Run": run, "ARI": adjusted_rand_score(y_sub, labels_km), "Geo_F1": m_km["Soft_Geo_Rare_F1"]}, rob_csv)

    print("\n🎉 扩展实验跑批全部完成！请准备作图。")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
