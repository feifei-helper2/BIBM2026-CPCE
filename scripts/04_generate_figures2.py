#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: Generate Figure 2 (Quadrant Scatter Plot) and Supplementary results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 0. Global Plotting Configurations
# ==========================================
sns.set_theme(style="ticks", context="paper", font_scale=1.4)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['pdf.fonttype'] = 42  
plt.rcParams['axes.linewidth'] = 1.5

# Color palette dictionary
PALETTE = {
    'CPCE': '#D62728',              
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

# ==========================================
# Supplementary Plot: Ablation Study (Corresponding to Table III)
# ==========================================
def plot_supplementary_ablation(df_ablation):
    print("Generating Supplementary Plot: Ablation Study (FacetGrid)...")
    
    # Filter specific core mechanisms
    df_ab = df_ablation[df_ablation['Variant'].isin(['CPCE_Full', 'CPCE_NoDPC', 'CPCE_NoZipf'])]
    
    df_melt = df_ab.melt(id_vars=['Dataset', 'Variant', 'Run'], 
                         value_vars=['ARI', 'Geo_F1'], 
                         var_name='Metric', value_name='Score')
    
    g = sns.catplot(data=df_melt, x='Variant', y='Score', hue='Metric', col='Dataset',
                    kind='bar', palette={'ARI': '#9ECAE1', 'Geo_F1': '#08519C'}, 
                    capsize=.1, errwidth=1.5, height=5, aspect=0.8)
    
    g.fig.subplots_adjust(top=0.85, bottom=0.25)
    g.fig.suptitle('Ablation Study: Collapse of Dual-Core Defense Mechanism', fontweight='bold', fontsize=16)
    g.set_titles("{col_name}", fontweight='bold')
    g.set_axis_labels("", "Score")
    
    for ax in g.axes.flat:
        ax.tick_params(axis='x', rotation=45)
        
    save_path = os.path.join(DIR_FIGURES, "Supplementary_Ablation_Plot.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Successfully generated: {save_path}")

# ==========================================
# Figure 2: Quadrant Scatter Plot (Escaping the ARI Trap)
# ==========================================
def plot_fig2_quadrant_scatter(df_main):
    print("Generating Figure 2: Quadrant Scatter Plot...")
    
    df_ds5 = df_main[df_main['Dataset'] == 'dataset_5']
    if df_ds5.empty: 
        print("Warning: dataset_5 not found, skipping Figure 2 generation.")
        return
    
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    
    sns.scatterplot(data=df_ds5, x='ARI', y='Geo_F1', hue='Algorithm', 
                    palette=PALETTE, alpha=0.6, s=80, legend=False, edgecolor='black', linewidth=0.3, ax=ax)
    
    df_mean = df_ds5.groupby('Algorithm', as_index=False)[['ARI', 'Geo_F1']].mean()
    sns.scatterplot(data=df_mean, x='ARI', y='Geo_F1', hue='Algorithm', 
                    palette=PALETTE, s=400, marker='X', edgecolor='black', linewidth=1.5, ax=ax)
    
    ari_mid = df_mean['ARI'].median()
    geo_mid = df_mean['Geo_F1'].median()
    ax.axvline(x=ari_mid, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(y=geo_mid, color='gray', linestyle='--', alpha=0.7)
    
    # Text annotations matching the paper
    ax.text(0.85, 0.85, 'The Absolute SOTA Domain\n(Uncharted Territory)', 
             color='#D62728', fontweight='bold', fontsize=12, ha='center', va='center', alpha=0.9,
             transform=ax.transAxes)
             
    ax.text(1.02, 0.15, 'High Global-Accuracy\nLow Rare-Protection\n(The Conventional Trap)', 
         color='#08519C', fontweight='bold', fontsize=10, ha='right', va='center', alpha=0.8,
         transform=ax.transAxes)

    plt.title("Case Study on dataset_5: Escaping the ARI Trap", fontweight='bold', pad=15)
    plt.xlabel("Adjusted Rand Index (ARI)")
    plt.ylabel("Geometric Mean Rare F1 (Geo_F1)")
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, title='Algorithm', fontsize=10)
    sns.despine()
    
    save_path = os.path.join(DIR_FIGURES, "Fig2_Quadrant_Scatter.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated Figure 2: {save_path}")

# ==========================================
# Supplementary Table 1: Complete Means and Standard Deviations
# ==========================================
def generate_supp_table(df_main):
    print("Generating Supplementary Table 1 (CSV format)...")
    
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
    print(f"Successfully generated: {save_path}")

# ==========================================
# Main Execution
# ==========================================
def main():
    print("Starting Chart Rendering Pipeline (Part 2)...")
    
    main_csv = os.path.join(DIR_TABLES, "main_benchmark_results.csv")
    ablation_csv = os.path.join(DIR_TABLES, "ablation_results.csv")
    
    if os.path.exists(main_csv):
        df_main = pd.read_csv(main_csv)
        
        rename_dict = {}
        if 'Soft_Geo_Rare_F1' in df_main.columns:
            rename_dict['Soft_Geo_Rare_F1'] = 'Geo_F1'
        elif 'Soft_Geo_F1' in df_main.columns:
            rename_dict['Soft_Geo_F1'] = 'Geo_F1'
        if 'Detection_Rate' in df_main.columns:
            rename_dict['Detection_Rate'] = 'Det_Rate'
        if rename_dict:
            df_main = df_main.rename(columns=rename_dict)
        
        plot_fig2_quadrant_scatter(df_main)
        generate_supp_table(df_main)
    else:
        print(f"Error: Main benchmark results not found at {main_csv}")
        
    if os.path.exists(ablation_csv):
        df_ab = pd.read_csv(ablation_csv)
        plot_supplementary_ablation(df_ab)
    else:
        print(f"Notice: Ablation results not found at {ablation_csv}")

    print(f"\nRendering complete. Check {DIR_FIGURES}/ for final outputs.")

if __name__ == "__main__":
    main()
