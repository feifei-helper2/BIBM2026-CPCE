# -*- coding: utf-8 -*-
"""
File: src/metrics.py
Purpose: Strict evaluation metrics for rare cell discovery, avoiding the ARI trap.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

def calc_rare_class_f1_strict(y_true, y_pred, rare_threshold_ratio=0.05):
    """
    Strict F1 evaluation with global exclusive matching + multi-dimensional robustness metrics.
    Forces 1-to-1 mapping between true and predicted clusters to eliminate the majority-domination artifact.
    Returns a dictionary containing Mean, Geometric Mean, Detection Rate, etc.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    unique_true, counts = np.unique(y_true, return_counts=True)
    total_cells = len(y_true)
    
    rare_classes = [cls for cls, count in zip(unique_true, counts) 
                    if (count / total_cells <= rare_threshold_ratio)]
            
    if not rare_classes:
        return {"Mean_Rare_F1": np.nan, "Soft_Geo_Rare_F1": np.nan, 
                "Detection_Rate": np.nan, "Median_Rare_F1": np.nan, "Q1_Rare_F1": np.nan}
        
    # 1. Use crosstab to avoid mixed-type broadcasting issues
    cross_tab = pd.crosstab(y_true, y_pred)
    cm = cross_tab.values
    row_labels = cross_tab.index.values 
    
    # 2. Construct cost matrix: use -F1 as weights for global optimization
    cost_matrix = np.zeros(cm.shape)
    for i in range(cm.shape[0]):
        actual_positives = np.sum(cm[i, :])
        for j in range(cm.shape[1]):
            true_positives = cm[i, j]
            if true_positives == 0:
                cost_matrix[i, j] = 0
                continue
            predicted_positives = np.sum(cm[:, j])
            
            precision = true_positives / predicted_positives if predicted_positives > 0 else 0
            recall = true_positives / actual_positives if actual_positives > 0 else 0
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
                
            cost_matrix[i, j] = -f1  
            
    # 3. Academic gold standard: Hungarian algorithm for global maximum weight matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Initialize all rare class scores to 0 to prevent unassigned rare classes from being missed
    rare_f1_dict = {cls: 0.0 for cls in rare_classes}
    
    for row, col in zip(row_ind, col_ind):
        true_label = row_labels[row] 
        if true_label in rare_classes:
            f1_score = -cost_matrix[row, col]
            rare_f1_dict[true_label] = f1_score
            
    # Extract 1D score array
    rare_f1_scores = np.array(list(rare_f1_dict.values()))

    # ==========================================
    # 4. Execute multi-dimensional rare class evaluation engine
    # ==========================================
    
    # Standard arithmetic mean (for baseline comparison)
    mean_rare_f1 = np.mean(rare_f1_scores)
    
    # Soft Geometric Mean (smoothing factor 0.05) - penalizes baselines that completely eradicate any rare class
    soft_epsilon = 0.05
    geometric_rare_f1 = np.exp(np.mean(np.log(rare_f1_scores + soft_epsilon))) - soft_epsilon
    geometric_rare_f1 = max(0.0, geometric_rare_f1)
    
    # Rare class detection rate - considered captured if score > 0.05
    detection_rate = np.sum(rare_f1_scores > 0.05) / len(rare_f1_scores)
    
    # Robustness quantile defense
    median_rare_f1 = np.median(rare_f1_scores)
    q1_rare_f1 = np.percentile(rare_f1_scores, 25)
    
    return {
        "Mean_Rare_F1": mean_rare_f1,
        "Soft_Geo_Rare_F1": geometric_rare_f1,
        "Detection_Rate": detection_rate,
        "Median_Rare_F1": median_rare_f1,
        "Q1_Rare_F1": q1_rare_f1
    }
