import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

sns.set_theme(style="ticks", context="paper", font_scale=1.4)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.linewidth'] = 1.5

DIR_FIGURES = "results/figures"
DIR_TABLES = "results/tables"

# 统一变体配色
VARIANT_PALETTE = {
    'CPCE_Full': '#D62728',     # 战神红
    'CPCE_NoDPC': '#BDBDBD',    # 浅灰
    'CPCE_NoZipf': '#737373',   # 中灰
    'CPCE_NoGS': '#1F77B4'      # 深蓝 (突显贪心算法的崩溃)
}

# ==========================================
# 更新版 Figure 2: 加入 NoGS 的四大变体消融
# ==========================================
def plot_updated_ablation():
    df = pd.read_csv(os.path.join(DIR_TABLES, "ablation_results.csv"))
    df['Dataset'] = df['Dataset'].str.capitalize()
    # 取出 4 个变体
    df = df[df['Variant'].isin(['CPCE_Full', 'CPCE_NoDPC', 'CPCE_NoZipf', 'CPCE_NoGS'])]
    
    # 设定画图的强制顺序
    variant_order = ['CPCE_Full', 'CPCE_NoDPC', 'CPCE_NoZipf', 'CPCE_NoGS']
    df['Variant'] = pd.Categorical(df['Variant'], categories=variant_order, ordered=True)
    
    df_melt = df.melt(id_vars=['Dataset', 'Variant', 'Run'], 
                      value_vars=['ARI', 'Geo_F1'], var_name='Metric', value_name='Score')
    
    g = sns.catplot(data=df_melt, x='Variant', y='Score', hue='Metric', col='Dataset',
                    kind='bar', palette={'ARI': '#9ECAE1', 'Geo_F1': '#08519C'}, 
                    capsize=.1, errwidth=1.5, height=5, aspect=0.9)
    
    g.fig.subplots_adjust(top=0.85, bottom=0.3)
    g.fig.suptitle('Ablation Study: The Collapse of Triple-Core Defenses', fontweight='bold', fontsize=16)
    g.set_titles("{col_name}", fontweight='bold')
    g.set_axis_labels("", "Score")
    
    for ax in g.axes.flat:
        ax.tick_params(axis='x', rotation=45)
        
    plt.savefig(os.path.join(DIR_FIGURES, "Fig2_Ablation_Facet_Updated.pdf"), dpi=300)
    plt.close()
    print("✅ 更新版 Figure 2 (Ablation) 已保存。")

# ==========================================
# Figure 4: 参数敏感性折线图
# ==========================================
def plot_sensitivity():
    df = pd.read_csv(os.path.join(DIR_TABLES, "sensitivity_results.csv"))
    
    df_T = df[df['Parameter'] == 'T']
    df_alpha = df[df['Parameter'] == 'alpha_max']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 子图 1: T
    sns.lineplot(data=df_T, x='Value', y='Geo_F1', ax=axes[0], marker='o', markersize=8, color='#D62728', label='Geo_F1', linewidth=2.5)
    sns.lineplot(data=df_T, x='Value', y='ARI', ax=axes[0], marker='s', markersize=8, color='#08519C', label='ARI', linewidth=2.5)
    axes[0].set_title('Sensitivity to Ensemble Size ($T$)', fontweight='bold')
    axes[0].set_xlabel('Ensemble Size ($T$)')
    axes[0].set_ylabel('Score')
    axes[0].set_ylim(0, 1)
    
    # 子图 2: alpha_max
    sns.lineplot(data=df_alpha, x='Value', y='Geo_F1', ax=axes[1], marker='o', markersize=8, color='#D62728', label='Geo_F1', linewidth=2.5)
    sns.lineplot(data=df_alpha, x='Value', y='ARI', ax=axes[1], marker='s', markersize=8, color='#08519C', label='ARI', linewidth=2.5)
    axes[1].set_title('Sensitivity to Zipf Exponent ($\\alpha_{max}$)', fontweight='bold')
    axes[1].set_xlabel('Zipf Maximum Exponent ($\\alpha_{max}$)')
    axes[1].set_ylabel('')
    axes[1].set_ylim(0, 1)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIGURES, "Fig4_Parameter_Sensitivity.pdf"), dpi=300)
    plt.close()
    print("✅ Figure 4 (Sensitivity) 已保存。")

# ==========================================
# Figure 5: 细胞下采样抗噪鲁棒性
# ==========================================
def plot_robustness():
    df = pd.read_csv(os.path.join(DIR_TABLES, "robustness_results.csv"))
    
    # 确保 100% 比例作为基准线存在 (我们可以从主表中借用，或者就只展示 0.5, 0.7, 0.9)
    # 为图表直观，我们这里直接画下采样折线
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Geo_F1 对比
    sns.lineplot(data=df, x='Ratio', y='Geo_F1', hue='Algorithm', palette={'CPCE': '#D62728', 'KMeans': '#9ECAE1'},
                 marker='D', markersize=10, linewidth=3, ax=axes[0])
    axes[0].set_title('Robustness of Rare Cell Protection', fontweight='bold')
    axes[0].set_xlabel('Subsampling Ratio (Retained Cells)')
    axes[0].set_ylabel('Geometric Mean Rare F1')
    axes[0].set_xlim(0.45, 0.95)
    axes[0].set_xticks([0.5, 0.7, 0.9])
    
    # ARI 对比
    sns.lineplot(data=df, x='Ratio', y='ARI', hue='Algorithm', palette={'CPCE': '#D62728', 'KMeans': '#9ECAE1'},
                 marker='D', markersize=10, linewidth=3, ax=axes[1])
    axes[1].set_title('Robustness of Global Topology', fontweight='bold')
    axes[1].set_xlabel('Subsampling Ratio (Retained Cells)')
    axes[1].set_ylabel('Adjusted Rand Index (ARI)')
    axes[1].set_xlim(0.45, 0.95)
    axes[1].set_xticks([0.5, 0.7, 0.9])
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIGURES, "Fig5_Noise_Robustness.pdf"), dpi=300)
    plt.close()
    print("✅ Figure 5 (Robustness) 已保存。")

if __name__ == "__main__":
    plot_updated_ablation()
    plot_sensitivity()
    plot_robustness()
