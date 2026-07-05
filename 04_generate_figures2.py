import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 0. 全局科研绘图风格与核心配置
# ==========================================
sns.set_theme(style="ticks", context="paper", font_scale=1.4)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['pdf.fonttype'] = 42  
plt.rcParams['axes.linewidth'] = 1.5

# 蓝红对决配色字典 (补全了所有算法)
PALETTE = {
    'CPCE': '#D62728',              # 战神红
    'KMeans': '#9ECAE1',            
    'GMM': '#6BAED6',               
    'scCAD': '#4292C6',             
    'GiniClust3': '#2171B5',        
    'Spectral': '#084594',          
    'Leiden': '#C6DBEF',            
    'Balanced_KMeans': '#4A98C9',   
    'Random_Ensemble': '#BDBDBD',   
    'scVI': '#737373'               
}

DIR_FIGURES = "results/figures"
DIR_TABLES = "results/tables"
os.makedirs(DIR_FIGURES, exist_ok=True)

# 物理文件到论文展示名的安全映射字典
GLOBAL_MAPPING = {
    "output1": "dataset_1", "output3": "dataset_2", "output4": "dataset_3",
    "output7": "dataset_4", "output8": "dataset_5", "output11": "dataset_6",
    "output18": "dataset_7"
}

# ==========================================
# 🔬 Figure 2: 双核引擎崩塌验证 (原 Figure 3)
# ==========================================
def plot_fig2_ablation(df_ablation):
    print("🎨 正在生成 Figure 2: 消融实验双核崩塌验证图 (FacetGrid)...")
    
    # 严格过滤，只对比三大核心机制，抛弃 Average-Link
    df_ab = df_ablation[df_ablation['Variant'].isin(['CPCE_Full', 'CPCE_NoDPC', 'CPCE_NoZipf'])]
    
    # 使用 melt 将 ARI 和 Geo_F1 融合成一列，方便分组画柱状图
    df_melt = df_ab.melt(id_vars=['Dataset', 'Variant', 'Run'], 
                         value_vars=['ARI', 'Geo_F1'], 
                         var_name='Metric', value_name='Score')
    
    # 使用纯净的高级蓝配色区分两种指标
    g = sns.catplot(data=df_melt, x='Variant', y='Score', hue='Metric', col='Dataset',
                    kind='bar', palette={'ARI': '#9ECAE1', 'Geo_F1': '#08519C'}, 
                    capsize=.1, errwidth=1.5, height=5, aspect=0.8)
    
    # 【工程防御】：为长文本索要底部物理空间，防止 X 轴标签被切断
    g.fig.subplots_adjust(top=0.85, bottom=0.25)
    g.fig.suptitle('Ablation Study: Collapse of Dual-Core Defense Mechanism', fontweight='bold', fontsize=16)
    g.set_titles("{col_name}", fontweight='bold')
    g.set_axis_labels("", "Score")
    
    for ax in g.axes.flat:
        ax.tick_params(axis='x', rotation=45)
        
    save_path = os.path.join(DIR_FIGURES, "Fig2_Ablation_Facet.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ Figure 2 生成成功: {save_path}")

# ==========================================
# 🎯 Figure 3: 评价体系四象限散点图 (原 Figure 4)
# ==========================================
def plot_fig3_quadrant_scatter(df_main):
    print("🎨 正在生成 Figure 3: 四象限评价体系散点云 (Scatter Plot)...")
    
    # 仅针对 dataset_5 绘制
    df_ds5 = df_main[df_main['Dataset'] == 'dataset_5']
    if df_ds5.empty: 
        print("⚠️ 找不到 dataset_5，跳过图 3。")
        return
    
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    
    # 1. 绘制半透明散点云 (展示波动方差)
    sns.scatterplot(data=df_ds5, x='ARI', y='Geo_F1', hue='Algorithm', 
                    palette=PALETTE, alpha=0.6, s=80, legend=False, edgecolor='black', linewidth=0.3, ax=ax)
    
    # 2. 算均值，绘制大号十字星
    df_mean = df_ds5.groupby('Algorithm', as_index=False)[['ARI', 'Geo_F1']].mean()
    sns.scatterplot(data=df_mean, x='ARI', y='Geo_F1', hue='Algorithm', 
                    palette=PALETTE, s=400, marker='X', edgecolor='black', linewidth=1.5, ax=ax)
    
    # 3. 画中位数十字分割线
    ari_mid = df_mean['ARI'].median()
    geo_mid = df_mean['Geo_F1'].median()
    ax.axvline(x=ari_mid, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(y=geo_mid, color='gray', linestyle='--', alpha=0.7)
    
    # 🎯 视觉绝杀 1：右上角 (CPCE 统治区)
    ax.text(0.85, 0.85, 'The Absolute SOTA Domain\n(Uncharted Territory)', 
             color='#D62728', fontweight='bold', fontsize=12, ha='center', va='center', alpha=0.9,
             transform=ax.transAxes)
             
    # 🎯 视觉绝杀 2：右下角/中下部 (基线陷阱区)
    ax.text(1.02, 0.15, 'High Global-Accuracy\nLow Rare-Protection\n(The Conventional Trap)', 
         color='#08519C', fontweight='bold', fontsize=10, ha='right', va='center', alpha=0.8,
         transform=ax.transAxes)

    plt.title("Case Study on dataset_5: Escaping the ARI Trap", fontweight='bold', pad=15)
    plt.xlabel("Adjusted Rand Index (ARI)")
    plt.ylabel("Geometric Mean Rare F1 (Geo_F1)")
    
    # 图例移到外侧
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, title='Algorithm', fontsize=10)
    sns.despine()
    
    save_path = os.path.join(DIR_FIGURES, "Fig3_Quadrant_Scatter.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Figure 3 生成成功: {save_path}")

# ==========================================
# 📄 Supplementary Table 1: 完整均值与标准差表格
# ==========================================
def generate_supp_table(df_main):
    print("📄 正在生成 Supplementary Table 1 (CSV 格式)...")
    # 只取数值列进行聚合，防止抛出 FutureWarning
    agg_funcs = {
        'ARI': ['mean', 'std'],
        'Geo_F1': ['mean', 'std'],
        'Det_Rate': ['mean', 'std']
    }
    df_agg = df_main.groupby(['Dataset', 'Algorithm']).agg(agg_funcs).reset_index()
    
    df_final = pd.DataFrame()
    df_final['Dataset'] = df_agg['Dataset']
    df_final['Algorithm'] = df_agg['Algorithm']
    
    for metric in ['ARI', 'Geo_F1', 'Det_Rate']:
        mean_col = df_agg[metric]['mean']
        std_col = df_agg[metric]['std']
        df_final[metric] = [f"{m:.3f} ± {s:.3f}" if pd.notna(m) else "DNC" for m, s in zip(mean_col, std_col)]
        
    save_path = os.path.join(DIR_TABLES, "Supplementary_Table_1.csv")
    df_final.to_csv(save_path, index=False)
    print(f"✅ 表格生成成功: {save_path}")

# ==========================================
# 主程序入口
# ==========================================
def main():
    print("🚀 启动 CPCE 顶会图表渲染流水线 (Part 2)...")
    
    main_csv = os.path.join(DIR_TABLES, "main_benchmark_results.csv")
    ablation_csv = os.path.join(DIR_TABLES, "ablation_results.csv")
    
    # -------------- 处理主表 (图 3 与 附表) --------------
    if os.path.exists(main_csv):
        df_main = pd.read_csv(main_csv)
        
        # 列名自适应修复
        rename_dict = {}
        if 'Soft_Geo_Rare_F1' in df_main.columns:
            rename_dict['Soft_Geo_Rare_F1'] = 'Geo_F1'
        elif 'Soft_Geo_F1' in df_main.columns:
            rename_dict['Soft_Geo_F1'] = 'Geo_F1'
        if 'Detection_Rate' in df_main.columns:
            rename_dict['Detection_Rate'] = 'Det_Rate'
        if rename_dict:
            df_main = df_main.rename(columns=rename_dict)
            
        # 内存换皮映射
        df_main['Dataset'] = df_main['Dataset'].replace(GLOBAL_MAPPING)
        
        plot_fig3_quadrant_scatter(df_main)
        generate_supp_table(df_main)
    else:
        print(f"❌ 找不到主实验数据表: {main_csv}")
        
    # -------------- 处理消融表 (图 2) --------------
    if os.path.exists(ablation_csv):
        df_ab = pd.read_csv(ablation_csv)
        plot_fig2_ablation(df_ab)
    else:
        print(f"⚠️ 找不到消融实验数据表: {ablation_csv}")

    print(f"\n🎉 渲染完结！请移步 {DIR_FIGURES}/ 查看最终绝杀图表。")

if __name__ == "__main__":
    main()
