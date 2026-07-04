import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 顶会级风格与配色字典
# ==========================================
sns.set_theme(style="ticks", context="paper", font_scale=1.4)
# 强制使用 DejaVu Sans 避免 Linux 字体报警
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['pdf.fonttype'] = 42  
plt.rcParams['axes.linewidth'] = 1.5

# 蓝红对决配色字典
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

# 物理文件到展示名的映射
GLOBAL_MAPPING = {
    "output1": "dataset_1", "output3": "dataset_2", "output4": "dataset_3",
    "output7": "dataset_4", "output8": "dataset_5", "output11": "dataset_6",
    "output18": "dataset_7"
}

# 严格锁死 X 轴展示顺序（剔除探讨集 dataset_5）
KEEP_DATASETS = ['dataset_1', 'dataset_2', 'dataset_3', 'dataset_4', 'dataset_6', 'dataset_7']

# ==========================================
# 2. 核心绘图函数
# ==========================================
def plot_fig1_global_landscape(df_main):
    print("🎨 正在渲染 Figure 1: 综合性能全景图 (带红色视觉锚点)...")
    
    # 过滤数据集并精简要展示的算法（避免图表过挤）
    df_sub = df_main[df_main['Dataset'].isin(KEEP_DATASETS)]
    algos = ['KMeans', 'Spectral', 'scCAD', 'GiniClust3', 'Leiden', 'CPCE']
    df_sub = df_sub[df_sub['Algorithm'].isin(algos)]
    
    # 算均值用于热力图
    df_mean = df_sub.groupby(['Dataset', 'Algorithm']).mean(numeric_only=True).reset_index()
    
    # 构建透视表：确保行是数据集（严格按顺序），列是算法
    # 将 Dataset 转为 Categorical 以强制行排序
    df_mean['Dataset'] = pd.Categorical(df_mean['Dataset'], categories=KEEP_DATASETS, ordered=True)
    pivot_ari = df_mean.pivot(index='Dataset', columns='Algorithm', values='ARI')[algos]
    pivot_geo = df_mean.pivot(index='Dataset', columns='Algorithm', values='Geo_F1')[algos]
    
    # 开辟 1:1:1.2 的画幅比例
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), gridspec_kw={'width_ratios': [1, 1, 1.2]})
    
    # ------------------ 子图 A: ARI 热力图 ------------------
    sns.heatmap(pivot_ari, annot=True, fmt=".2f", cmap="Blues", cbar=False, 
                linewidths=1, linecolor='white', ax=axes[0])
    axes[0].set_title('A. Adjusted Rand Index (ARI)', fontweight='bold', pad=15)
    axes[0].set_ylabel('')
    axes[0].set_xlabel('')
    
    # 【视觉锚点大招】：为 CPCE 列绘制醒目红框
    cpce_idx = algos.index('CPCE')
    # Rectangle((x, y), width, height)
    axes[0].add_patch(Rectangle((cpce_idx, 0), 1, len(KEEP_DATASETS), 
                                fill=False, edgecolor='#D62728', lw=4, clip_on=False, zorder=10))
    
    # ------------------ 子图 B: Geo_F1 热力图 ------------------
    sns.heatmap(pivot_geo, annot=True, fmt=".2f", cmap="Blues", cbar=False, 
                linewidths=1, linecolor='white', ax=axes[1])
    axes[1].set_title('B. Geometric Mean Rare F1', fontweight='bold', pad=15)
    axes[1].set_ylabel('')
    axes[1].set_xlabel('')
    axes[1].set_yticks([]) # 隐藏重复 Y 轴
    
    # 【视觉锚点大招】：同理，画红框
    axes[1].add_patch(Rectangle((cpce_idx, 0), 1, len(KEEP_DATASETS), 
                                fill=False, edgecolor='#D62728', lw=4, clip_on=False, zorder=10))
    
    # 将底部 X 轴的 "CPCE" 字体单独加粗标红
    for ax in [axes[0], axes[1]]:
        for tick_label in ax.get_xticklabels():
            if tick_label.get_text() == 'CPCE':
                tick_label.set_color('#D62728')
                tick_label.set_fontweight('bold')
    
    # ------------------ 子图 C: Det_Rate 柱状图 ------------------
    # 强制传入 order=KEEP_DATASETS 彻底解决乱序问题
    sns.barplot(data=df_sub, x='Dataset', y='Det_Rate', hue='Algorithm', 
                palette=PALETTE, order=KEEP_DATASETS, hue_order=algos,
                capsize=.1, errwidth=1.2, ax=axes[2])
    
    axes[2].set_title('C. Rare Cell Detection Rate', fontweight='bold', pad=15)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel('Detection Rate')
    axes[2].set_xlabel('')
    # 把图例移到右侧空白处
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    
    sns.despine(ax=axes[2])
    plt.tight_layout()
    save_path = os.path.join(DIR_FIGURES, "Fig1_Global_Landscape_Fixed.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Figure 1 生成成功: {save_path}")

# ==========================================
# 3. 驱动逻辑
# ==========================================
if __name__ == "__main__":
    main_csv = os.path.join(DIR_TABLES, "main_benchmark_results.csv")
    if os.path.exists(main_csv):
        df_main = pd.read_csv(main_csv)
        
        # 【列名自适应修复】：兼容旧版本 metrics 产生的列名
        rename_dict = {}
        if 'Soft_Geo_Rare_F1' in df_main.columns:
            rename_dict['Soft_Geo_Rare_F1'] = 'Geo_F1'
        elif 'Soft_Geo_F1' in df_main.columns:
            rename_dict['Soft_Geo_F1'] = 'Geo_F1'
            
        if 'Detection_Rate' in df_main.columns:
            rename_dict['Detection_Rate'] = 'Det_Rate'
            
        if rename_dict:
            df_main = df_main.rename(columns=rename_dict)
        
        # 执行数据集内存映射 (outputX -> dataset_X)
        df_main['Dataset'] = df_main['Dataset'].replace(GLOBAL_MAPPING)
        
        plot_fig1_global_landscape(df_main)
    else:
        print(f"❌ 找不到数据表: {main_csv}，请检查路径！")
