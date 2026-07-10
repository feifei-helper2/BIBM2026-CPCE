#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: Generate supplementary figures for ablation, sensitivity, and robustness experiments.
Reference: Table III (Ablation) and Table IV (Sensitivity & Robustness) in the paper.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Global plot settings
sns.set_theme(style="ticks", context="paper", font_scale=1.4)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['pdf.fonttype'] = 42

DIR_FIGURES = "results/figures"
DIR_TABLES = "results/tables"
os.makedirs(DIR_FIGURES, exist_ok=True)

# Consistent variant palette
VARIANT_PALETTE = {
    'CPCE_Full': '#D62728',     
    'CPCE_NoDPC': '#BDBDBD',    
    'CPCE_NoZipf': '#737373',   
    'CPCE_NoGS': '#1F77B4'      
}

# ==========================================
# Supplementary Plot: Updated Ablation (including NoGS)
# ==========================================
def plot_updated_ablation():
    print("Generating Supplementary Plot: Updated Ablation (with NoGS)...")
    csv_path = os.path.join(DIR_TABLES, "ablation_results.csv")
    if not os.path.exists(csv_path):
        print(f"Notice: File not found {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    df['Dataset'] = df['Dataset'].str.capitalize()
    
    df = df[df['Variant'].isin(['CPCE_Full', 'CPCE_NoDPC', 'CPCE_NoZipf', 'CPCE_NoGS'])]
    
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
        
    save_path = os.path.join(DIR_FIGURES, "Supplementary_Ablation_Plot_Updated.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Successfully generated: {save_path}")

# ==========================================
# Extended Results: Parameter Sensitivity
# ==========================================
def plot_sensitivity():
    print("Generating Extended Results Plot: Parameter Sensitivity...")
    csv_path = os.path.join(DIR_TABLES, "sensitivity_results.csv")
    if not os.path.exists(csv_path):
        print(f"Notice: File not found {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    df_T = df[df['Parameter'] == 'T']
    df_alpha = df[df['Parameter'] == 'alpha_max']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.lineplot(data=df_T, x='Value', y='Geo_F1', ax=axes[0], marker='o', markersize=8, color='#D62728', label='Geo_F1', linewidth=2.5)
    sns.lineplot(data=df_T, x='Value', y='ARI', ax=axes[0], marker='s', markersize=8, color='#08519C', label='ARI', linewidth=2.5)
    axes[0].set_title('Sensitivity to Ensemble Size ($T$)', fontweight='bold')
    axes[0].set_xlabel('Ensemble Size ($T$)')
    axes[0].set_ylabel('Score')
    axes[0].set_ylim(0, 1)
    
    sns.lineplot(data=df_alpha, x='Value', y='Geo_F1', ax=axes[1], marker='o', markersize=8, color='#D62728', label='Geo_F1', linewidth=2.5)
    sns.lineplot(data=df_alpha, x='Value', y='ARI', ax=axes[1], marker='s', markersize=8, color='#08519C', label='ARI', linewidth=2.5)
    axes[1].set_title('Sensitivity to Zipf Exponent ($\\alpha_{max}$)', fontweight='bold')
    axes[1].set_xlabel('Zipf Maximum Exponent ($\\alpha_{max}$)')
    axes[1].set_ylabel('')
    axes[1].set_ylim(0, 1)
    
    sns.despine()
    plt.tight_layout()
    save_path = os.path.join(DIR_FIGURES, "Extended_Results_for_Table_IV_Sensitivity.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Successfully generated: {save_path}")

# ==========================================
# Extended Results: Noise Robustness (Subsampling)
# ==========================================
def plot_robustness():
    print("Generating Extended Results Plot: Noise Robustness...")
    csv_path = os.path.join(DIR_TABLES, "robustness_results.csv")
    if not os.path.exists(csv_path):
        print(f"Notice: File not found {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.lineplot(data=df, x='Ratio', y='Geo_F1', hue='Algorithm', palette={'CPCE': '#D62728', 'KMeans': '#9ECAE1'},
                 marker='D', markersize=10, linewidth=3, ax=axes[0])
    axes[0].set_title('Robustness of Rare Cell Protection', fontweight='bold')
    axes[0].set_xlabel('Subsampling Ratio (Retained Cells)')
    axes[0].set_ylabel('Geometric Mean Rare F1')
    axes[0].set_xlim(0.45, 0.95)
    axes[0].set_xticks([0.5, 0.7, 0.9])
    
    sns.lineplot(data=df, x='Ratio', y='ARI', hue='Algorithm', palette={'CPCE': '#D62728', 'KMeans': '#9ECAE1'},
                 marker='D', markersize=10, linewidth=3, ax=axes[1])
    axes[1].set_title('Robustness of Global Topology', fontweight='bold')
    axes[1].set_xlabel('Subsampling Ratio (Retained Cells)')
    axes[1].set_ylabel('Adjusted Rand Index (ARI)')
    axes[1].set_xlim(0.45, 0.95)
    axes[1].set_xticks([0.5, 0.7, 0.9])
    
    sns.despine()
    plt.tight_layout()
    save_path = os.path.join(DIR_FIGURES, "Extended_Results_for_Table_IV_Robustness.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Successfully generated: {save_path}")

if __name__ == "__main__":
    print("Starting Extension Figures Generation Pipeline...")
    plot_updated_ablation()
    plot_sensitivity()
    plot_robustness()
    print("Pipeline execution completed.")
