import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

def calc_rare_class_f1_strict(y_true, y_pred, rare_threshold_ratio=0.05):
    """
    引入全局排他匹配的严苛 F1 评测 + 多维度抗辩指标。
    强制真实簇与预测簇 1对1 映射，彻底刺穿大类吞噬假象。
    返回包含 Mean, Geo, Detection Rate 等多维指标的字典。
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
        
    # 1. 使用 crosstab 规避混型扩容灾难
    cross_tab = pd.crosstab(y_true, y_pred)
    cm = cross_tab.values
    row_labels = cross_tab.index.values 
    
    # 2. 构建代价矩阵：以 -F1 为权重进行全局寻优
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
            
    # 3. 严守学术金标准：匈牙利全局最大权匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 初始化所有稀有类分数为0，防范由于预测簇数量不足导致未被分配的稀有类被漏算
    rare_f1_dict = {cls: 0.0 for cls in rare_classes}
    
    for row, col in zip(row_ind, col_ind):
        true_label = row_labels[row] 
        if true_label in rare_classes:
            f1_score = -cost_matrix[row, col]
            rare_f1_dict[true_label] = f1_score
            
    # 获取一维分数数组
    rare_f1_scores = np.array(list(rare_f1_dict.values()))

    # ==========================================
    # 4. 🌟 挂载多维度罕见类评估引擎 (抗辩指标)
    # ==========================================
    
    # 传统算术平均 (用于兜底对比)
    mean_rare_f1 = np.mean(rare_f1_scores)
    
    # Soft Geometric Mean (平滑因子退火至 0.05) - 惩罚基线算法的全盲现象
    soft_epsilon = 0.05
    geometric_rare_f1 = np.exp(np.mean(np.log(rare_f1_scores + soft_epsilon))) - soft_epsilon
    geometric_rare_f1 = max(0.0, geometric_rare_f1)
    
    # 罕见类检出率 (Detection Rate) - 只要大于 0.05 即认为捕获
    detection_rate = np.sum(rare_f1_scores > 0.05) / len(rare_f1_scores)
    
    # 稳健性分位数防御
    median_rare_f1 = np.median(rare_f1_scores)
    q1_rare_f1 = np.percentile(rare_f1_scores, 25)
    
    # 将多维度指标打包返回
    return {
        "Mean_Rare_F1": mean_rare_f1,
        "Soft_Geo_Rare_F1": geometric_rare_f1,
        "Detection_Rate": detection_rate,
        "Median_Rare_F1": median_rare_f1,
        "Q1_Rare_F1": q1_rare_f1
    }
