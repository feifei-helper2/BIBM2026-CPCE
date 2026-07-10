# -*- coding: utf-8 -*-
"""
File: src/plotter.py
Purpose: Static academic visual utility library.
Guidelines:
1. Strictly decoupled: only accepts axis handles and raw data.
2. Enforces Nature Publishing Group (NPG) academic aesthetics and vectorization compatibility.
"""

import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Circle

# ==========================================
# 1. Global Academic Aesthetics and Palette Management
# ==========================================
def setup_academic_style():
    """Override Matplotlib parameters to enforce academic paper standards."""
    sns.set_theme(style="ticks", context="paper")
    
    # [Fix] Mute findfont warnings on Linux servers missing Arial to prevent log pollution
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    
    mpl.rcParams.update({
        # Font specifications (Priority: Arial)
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": False,
        
        # Despine and tick adjustments
        "axes.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        
        # Vector graphics output and AI compatibility
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.format": "pdf",
        "savefig.bbox": "tight"
    })

# Pre-defined top-tier conference color palette (NPG style)
PALETTE_DICT = {
    "CPCE": "#DC0000",             # Scientific Crimson (visual focus)
    "KMeans": "#4DBBD5",           
    "GMM": "#7E6148",              
    "Spectral": "#8491B4",         
    "Leiden": "#F39B7F",           
    "scCAD": "#3C5488",            
    "GiniClust3": "#00A087",       
    "scVI": "#B09C85",             
    "Balanced_KMeans": "#91D1C2",  
    "Random_Ensemble": "#E64B35",  
    "Uniform": "#A9A9A9",          
    "Zipf": "#DC0000"              
}

def get_color(algo_name):
    """Failsafe color extractor: returns cool gray for unknown algorithms."""
    return PALETTE_DICT.get(algo_name, "#CCCCCC")

# Initialize and mount theme
setup_academic_style()

# ==========================================
# 2. Atomic Plotting Functions (Axes Injection Mode)
# ==========================================

def plot_benchmark_bar(ax, df, x_col, y_col, hue_col, title="", ylabel="", show_legend=True):
    """
    Plot main performance comparison bar chart with error bars using Seaborn bootstrapping.
    Requires raw data input.
    """
    safe_palette = {alg: get_color(alg) for alg in df[hue_col].unique()}
    
    sns.barplot(
        data=df, x=x_col, y=y_col, hue=hue_col, 
        ax=ax, palette=safe_palette,
        capsize=0.1, errorbar="sd", err_kws={'linewidth': 1.5}, 
        edgecolor="black", linewidth=1.0
    )
    
    if title:
        ax.set_title(title, pad=15)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # [Architecture Upgrade] Allow external scripts to toggle subplot legends
    if ax.get_legend() is not None:
        if show_legend:
            ax.legend(frameon=False, loc='best')
        else:
            ax.get_legend().remove()

def plot_ablation_line(ax, df, x_col, y_col, hue_col, title="", xlabel="", ylabel="", show_legend=True):
    """
    Plot smooth trend line with semi-transparent error bands.
    """
    safe_palette = {alg: get_color(alg) for alg in df[hue_col].unique()}
    
    sns.lineplot(
        data=df, x=x_col, y=y_col, hue=hue_col, 
        ax=ax, palette=safe_palette,
        marker="o", markersize=8, markeredgecolor="white", markeredgewidth=1.2,
        linewidth=2.5, err_style="band", errorbar="sd"
    )
    
    if title:
        ax.set_title(title, pad=15)
    if xlabel:
        ax.set_xlabel(xlabel, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontweight='bold')
        
    ax.grid(True, linestyle='--', alpha=0.3)
    
    if ax.get_legend() is not None:
        if show_legend:
            ax.legend(frameon=False, loc='best')
        else:
            ax.get_legend().remove()

def plot_umap_contrast(ax, embedding, labels, title="", target_indices=None):
    """
    Plot UMAP scatter with physical geometric targeting (dashed red circle) for specified clusters.
    Uses z-order layering to prevent global highlighting artifacts.
    """
    unique_labels = np.unique(labels)
    cmap = sns.color_palette("husl", len(unique_labels))
    
    # Build global row indices for high-speed bitwise masking
    global_indices = np.arange(len(labels))
    
    # 1. Plot base scatter (foreground/background layered rendering)
    for i, cluster_id in enumerate(unique_labels):
        mask = (labels == cluster_id)
        
        # [Fix] Single-cell physical layering, rejecting full-cluster collateral highlighting
        if target_indices is not None and len(target_indices) > 0:
            target_mask = mask & np.isin(global_indices, target_indices)
            bg_mask = mask & (~np.isin(global_indices, target_indices))
        else:
            bg_mask = mask
            target_mask = np.zeros_like(mask, dtype=bool)
            
        # 1.1 Plot background cells of the current cluster (semi-transparent, zorder=1)
        if np.any(bg_mask):
            ax.scatter(
                embedding[bg_mask, 0], embedding[bg_mask, 1], 
                s=12, alpha=0.5, color=cmap[i], edgecolors='none', zorder=1
            )
            
        # 1.2 Plot target cells within the cluster (opaque, zorder=5)
        if np.any(target_mask):
            ax.scatter(
                embedding[target_mask, 0], embedding[target_mask, 1], 
                s=15, alpha=1.0, color=cmap[i], edgecolors='black', linewidths=0.5, zorder=5
            )
        
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    
    # 2. Geometric canvas: Fixed absolute physical crosshair
    if target_indices is not None and len(target_indices) > 0:
        global_x_range = np.max(embedding[:, 0]) - np.min(embedding[:, 0])
        min_radius = global_x_range * 0.05 
        
        # Extract absolute physical coordinates of target cells to locate the center
        target_points = embedding[target_indices]
        center_x = np.mean(target_points[:, 0])
        center_y = np.mean(target_points[:, 1])
        
        if len(target_points) == 1:
            radius = min_radius
        else:
            # Calculate radius using 95th percentile to prevent isolated noise from inflating the circle
            distances = np.linalg.norm(target_points - [center_x, center_y], axis=1)
            calc_radius = np.percentile(distances, 95) * 1.5
            radius = max(calc_radius, min_radius)
        
        circle = Circle(
            (center_x, center_y), radius, 
            fill=False, color=PALETTE_DICT.get("CPCE", "red"), 
            linewidth=2, linestyle='--', zorder=10
        )
        ax.add_patch(circle)

def plot_heatmap(ax, matrix, title="", cmap="rocket_r"):
    """
    Plot high-precision Co-association Matrix heatmap.
    """
    sns.heatmap(
        matrix, ax=ax, cmap=cmap, cbar=True, 
        xticklabels=False, yticklabels=False, 
        square=True, 
        rasterized=True
    )
    if title:
        ax.set_title(title, pad=15)
