# -*- coding: utf-8 -*-
"""
文件：src/plotter.py
职责：纯静态学术视觉工具库。
纪律：
1. 绝对解耦：只接收 ax 句柄与绘图数据，不调用 plt.figure() 或 plt.savefig()。
2. 美学强控：统一 NPG 顶会配色、适配最新 Seaborn API、几何高亮抗畸变与矢量化兼容。
3. 数据契约：要求传入未经聚合的原始明细数据 (Raw Data)，由 Seaborn 底层完成均值与误差带渲染。
"""

import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Circle

# ==========================================
# 1. 全局学术美学与调色盘接管
# ==========================================
def setup_academic_style():
    """暴力接管 Matplotlib 参数，强制注入学术论文规范"""
    sns.set_theme(style="ticks", context="paper")
    
    # 【修复：防日志污染】静音 Linux 服务器上缺失 Arial 时的 findfont 警告
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    
    mpl.rcParams.update({
        # 字体规范 (优先 Arial)
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
        
        # 边框与刻度物理规整 (Despine)
        "axes.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        
        # 矢量图输出与 AI 兼容性强制要求
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.format": "pdf",
        "savefig.bbox": "tight"
    })

# 预定义顶会级配色字典 (Nature Publishing Group 风格)
PALETTE_DICT = {
    "CPCE": "#DC0000",             # 科研猩红 (绝对视觉中心)
    "KMeans": "#4DBBD5",           # 亮蓝
    "GMM": "#7E6148",              # 深棕
    "Spectral": "#8491B4",         # 冷灰蓝
    "Leiden": "#F39B7F",           # 浅橙
    "scCAD": "#3C5488",            # 深邃蓝 (SOTA)
    "GiniClust3": "#00A087",       # 翡翠绿 (SOTA)
    "scVI": "#B09C85",             # 暖灰 (SOTA)
    "Balanced_KMeans": "#91D1C2",  # 浅青 (消融)
    "Random_Ensemble": "#E64B35",  # 稍弱的红色 (消融)
    "Uniform": "#A9A9A9",          # 暗灰 (模板消融)
    "Zipf": "#DC0000"              # 与 CPCE 同色
}

def get_color(algo_name):
    """防错调色提取器：若遇到未知算法，返回兜底的冷灰色"""
    return PALETTE_DICT.get(algo_name, "#CCCCCC")

# 初始化即挂载主题
setup_academic_style()

# ==========================================
# 2. 原子级绘图函数 (Axes 注入模式)
# ==========================================

def plot_benchmark_bar(ax, df, x_col, y_col, hue_col, title="", ylabel="", show_legend=True):
    """
    绘制带误差棒的主性能对比柱状图。
    要求传入 raw data，利用 Seaborn 的自举法自动计算标准差。
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
    
    # 【架构升级：图例开关】允许外部脚本关闭子图图例，以绘制全局统一图例
    if ax.get_legend() is not None:
        if show_legend:
            ax.legend(frameon=False, loc='best')
        else:
            ax.get_legend().remove()

def plot_ablation_line(ax, df, x_col, y_col, hue_col, title="", xlabel="", ylabel="", show_legend=True):
    """
    绘制带有半透明误差带 (Error Band) 的平滑折线趋势图。
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
    绘制 UMAP 散点图，支持几何画板红色虚线圈高亮靶标细胞簇。
    已升级为“绝对物理准星”模式，并彻底修复大类连带高亮漏洞（物理图层分离渲染）。
    """
    unique_labels = np.unique(labels)
    cmap = sns.color_palette("husl", len(unique_labels))
    
    # 建立全局行号索引，用于毫秒级高速掩码位运算
    global_indices = np.arange(len(labels))
    
    # 1. 绘制底层散点 (采用前景与背景分层高精渲染机制)
    for i, cluster_id in enumerate(unique_labels):
        mask = (labels == cluster_id)
        
        # 核心修复：单细胞级物理分层切片，拒绝全簇连带高亮
        if target_indices is not None and len(target_indices) > 0:
            # 靶标细胞：既属于当前簇，又属于绝对靶标行号 (位运算交集)
            target_mask = mask & np.isin(global_indices, target_indices)
            # 背景细胞：属于当前簇，但物理上不是靶标细胞 (位运算取反)
            bg_mask = mask & (~np.isin(global_indices, target_indices))
        else:
            bg_mask = mask
            target_mask = np.zeros_like(mask, dtype=bool)
            
        # 1.1 先绘制当前簇的背景细胞 (赋予中度透明度，沉在底层 zorder=1)
        if np.any(bg_mask):
            ax.scatter(
                embedding[bg_mask, 0], embedding[bg_mask, 1], 
                s=12, alpha=0.5, color=cmap[i], edgecolors='none', zorder=1
            )
            
        # 1.2 再强行绘制当前簇中包含的靶标细胞 (全不透明，加黑色细边，霸榜最顶层 zorder=5)
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
    
    # 2. 几何画板：上帝视角的固定绝对物理准星
    if target_indices is not None and len(target_indices) > 0:
        global_x_range = np.max(embedding[:, 0]) - np.min(embedding[:, 0])
        min_radius = global_x_range * 0.05 
        
        # 提取靶标细胞的绝对物理坐标进行圆心定位
        target_points = embedding[target_indices]
        center_x = np.mean(target_points[:, 0])
        center_y = np.mean(target_points[:, 1])
        
        if len(target_points) == 1:
            radius = min_radius
        else:
            # 使用 95% 分位数计算半径，防止孤立噪点将圆圈无限拉大
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
    绘制高精度的共聚矩阵 (Co-association Matrix) 热力图。
    """
    sns.heatmap(
        matrix, ax=ax, cmap=cmap, cbar=True, 
        xticklabels=False, yticklabels=False, 
        square=True, 
        rasterized=True
    )
    if title:
        ax.set_title(title, pad=15)
