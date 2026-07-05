#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本职责: 主实验全自动化跑批总线 (The Grand Orchestrator)
核心纪律:
1. 断点续传: 基于 CSV 和 .npy 的双重断点续传，拒绝重复计算。
2. 实时刷盘: 采用安全的增量追加 (Append) 模式写入 CSV，防死机清零。
3. 异常隔离: try-except 彻底覆盖 I/O 与计算报错，保全整体流水线。
4. 资源控制: O(N^3) 图算法限制单次执行，保障时效性。
"""
import os
#os.environ["MKL_NUM_THREADS"] = "1"
#os.environ["NUMEXPR_NUM_THREADS"] = "1"
#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["LOKY_MAX_CPU_COUNT"] = "1"
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
# 0. 环境路径对齐与模块挂载
# ==========================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.baselines import BaselineModels
from src.cpce_engine import run_synthesized_cpce
from src.metrics import calc_rare_class_f1_strict

# 目录基建查缺补漏
DIR_DATA = os.path.join(PROJECT_ROOT, "data", "processed")
DIR_TABLES = os.path.join(PROJECT_ROOT, "results", "tables")
DIR_LABELS = os.path.join(PROJECT_ROOT, "results", "saved_labels")
DIR_LOGS = os.path.join(PROJECT_ROOT, "logs")

for d in [DIR_TABLES, DIR_LABELS, DIR_LOGS]:
    os.makedirs(d, exist_ok=True)

# 日志双流监控挂载
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
# 1. 超参数与算法注册表
# ==========================================
GLOBAL_SEED = 42
RARE_THRESHOLD = 0.05
OUTPUT_CSV = os.path.join(DIR_TABLES, "main_benchmark_results.csv")

def run_cpce_wrapper(X_pca, K, seed):
    """为 CPCE 引擎注入动态随机种子，确保稳定性测试真实有效"""
    # 1. 【全局保底防线】：控制 numpy 的基础随机状态
    np.random.seed(seed) 
    
    # 2. 【精确参数控流】：控制 sklearn / scanpy 的内部随机状态
    return run_synthesized_cpce(X_pca, K, seed=seed)

# 算法黑盒注册表 (统一接口规范: lambda X_pca, K, seed, adata)
ALGORITHMS_REGISTRY = {
    # --- 古典与常规基线 (吞掉 adata) ---
    "KMeans": {"func": lambda X, K, s, adata: BaselineModels.run_kmeans(X, K, s), "runs": 10},
    "GMM": {"func": lambda X, K, s, adata: BaselineModels.run_gmm(X, K, s), "runs": 10},
    "Spectral": {"func": lambda X, K, s, adata: BaselineModels.run_spectral(X, K, s), "runs": 10},
    "Leiden": {"func": lambda X, K, s, adata: BaselineModels.run_leiden_with_bisection(X, K, s), "runs": 10},
    
    # --- 消融实验组 (同批次运行，后续画图再分离) ---
    "Balanced_KMeans": {"func": lambda X, K, s, adata: BaselineModels.run_balanced_kmeans(X, K, s), "runs": 10},
    "Random_Ensemble": {"func": lambda X, K, s, adata: BaselineModels.run_random_ensemble(X, K, s), "runs": 10},
    
    # --- 前沿与深度 SOTA ---
    "scCAD": {"func": lambda X, K, s, adata: BaselineModels.run_sccad(X, adata, K, s), "runs": 10},
    "GiniClust3": {"func": lambda X, K, s, adata: BaselineModels.run_giniclust3(X, adata, K, s), "runs": 10},
    "scVI": {"func": lambda X, K, s, adata: BaselineModels.run_scvi_kmeans(X, adata, K, s), "runs": 10},
    
    # --- 核心引擎 ---
    "CPCE": {"func": lambda X, K, s, adata: run_cpce_wrapper(X, K, s), "runs": 10}
}

# ==========================================
# 2. 安全原子化落盘组件
# ==========================================
def append_result_to_csv(record_dict, csv_path):
    """安全增量追加，规避 pd.to_csv 全量覆写时的文件损坏风险"""
    df_row = pd.DataFrame([record_dict])
    file_exists = os.path.isfile(csv_path)
    df_row.to_csv(csv_path, mode='a', header=not file_exists, index=False)

def get_completed_tasks(csv_path):
    """读取历史战报，为行级断点续传提供依据"""
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            return set(zip(df['Dataset'], df['Algorithm'], df['Run_ID']))
        except Exception as e:
            logging.warning(f"⚠️ 读取历史跑分表失败 ({e})，将重头开始计算。")
    return set()

# ==========================================
# 3. 暴君式主引擎总线
# ==========================================
def main():
    logging.info("🚀 ============ CPCE 主实验跑批总线启动 ============")
    
    dataset_files = sorted(glob.glob(os.path.join(DIR_DATA, "*.h5ad")))
    if not dataset_files:
        logging.error(f"❌ 致命错误：未在 {DIR_DATA} 找到标准件！")
        return
        
    logging.info(f"📂 发现 {len(dataset_files)} 个标准数据集。")
    
    completed_tasks = get_completed_tasks(OUTPUT_CSV)
    if completed_tasks:
        logging.info(f"🔄 检测到 {len(completed_tasks)} 条历史完成记录，激活增量断点续传模式。")

    # [一层] 遍历数据集
    for file_path in dataset_files:
        dataset_name = os.path.basename(file_path).replace(".h5ad", "")
        logging.info(f"\n{'-'*60}\n🔬 载入数据集: {dataset_name}\n{'-'*60}")
        
        try:
            adata = sc.read_h5ad(file_path)
            X_pca = adata.obsm['X_pca']
            y_true = adata.obs['ground_truth'].values
            K_gt = int(adata.uns['n_clusters_gt'])
            N_cells = X_pca.shape[0]
            logging.info(f"   ✔️ 物理参数: N={N_cells}, K={K_gt}")
        except Exception as e:
            logging.error(f"❌ 数据集解析失败，跳过。报错: {e}")
            continue

        # [二层] 遍历竞争算法
        for algo_name, config in ALGORITHMS_REGISTRY.items():
            algo_func = config["func"]
            n_runs = config["runs"]
            
            # [三层] 稳定性跑批
            for run_idx in range(n_runs):
                current_seed = GLOBAL_SEED + run_idx
                task_fingerprint = (dataset_name, algo_name, run_idx)
                
                # 防重机制 1: 行级断点续传
                if task_fingerprint in completed_tasks:
                    continue
                    
                logging.info(f"   ⚔️ 运算中: {algo_name} (Run {run_idx+1}/{n_runs}) ...")
                
                label_save_path = os.path.join(DIR_LABELS, f"{dataset_name}_{algo_name}_run{run_idx}.npy")
                labels_pred = None
                exec_time = np.nan  # 初始化为 NaN，防止污染时间均值
                
                start_time = time.time()
                try:
                    # 防重机制 2: 资产级断点极速加载
                    if os.path.exists(label_save_path):
                        logging.info(f"      ⏭️ [资产续传] 已命中本地标签文件，跳过聚类计算。")
                        labels_pred = np.load(label_save_path)
                    else:
                        # 真实计算执行
                        labels_pred = algo_func(X_pca, K_gt, current_seed, adata)
                        exec_time = time.time() - start_time
                        np.save(label_save_path, labels_pred)
                        
                    # 指标核算一并纳入 try-except，防止坏账 .npy 导致报错崩溃
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
                    logging.info(f"      ✅ 完成 | Time: {time_log} | ARI: {ari:.3f} | Geo_F1: {soft_geo_f1:.3f} | Det_Rate: {detection_rate:.2f}")
                    
                except Exception as e:
                    if np.isnan(exec_time):
                        exec_time = time.time() - start_time
                    error_msg = str(e).split('\n')[0]
                    logging.error(f"      💥 [熔断] {error_msg}")
                    
                    ari, nmi, rare_f1 = np.nan, np.nan, np.nan
                    status = f"Failed: {error_msg}"
                    mean_f1 = soft_geo_f1 = detection_rate = median_f1 = q1_f1 = np.nan
                    status = f"Failed: {error_msg}"

                # 原子落库
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

    logging.info(f"\n🎉 ============ 跑批任务圆满收官 ============")
    logging.info(f"📄 汇总大表已落地: {OUTPUT_CSV}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
