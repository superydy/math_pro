#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题2 - 模型B：物理影响规律分析
=================================

目的：探索各风箱负压、温度、大烟道参数对CO浓度的影响规律

方法：
1. 相关性分析（Pearson + Spearman）- 线性/单调关系
2. Ridge回归 - 线性影响方向和强度
3. 随机森林特征重要性 - 非线性重要性排序
4. 偏依赖图(PDP) - 单变量对CO的非线性影响曲线
5. 分区分析 - 前/中/后段风箱的分区影响
6. 交互效应 - 关键变量之间的交互作用

输出：
- 物理影响规律总结
- 可视化图表

运行：python Q2_physical_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import partial_dependence
from sklearn.metrics import r2_score
import json
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def print_flush(msg):
    print(msg, flush=True)


def compute_xcorr(x, y, max_lag=60):
    """FFT互相关"""
    xn = (x - np.mean(x)) / (np.std(x) * len(x) + 1e-8)
    yn = (y - np.mean(y)) / (np.std(y) + 1e-8)
    corr = fftconvolve(xn[::-1], yn, mode='full')
    center = len(xn) - 1
    lags = np.arange(-max_lag, max_lag + 1)
    cc = corr[center - max_lag: center + max_lag + 1]
    return lags, cc


# ============================================================
# Step 1: 数据准备
# ============================================================
def step1_data_prep():
    """加载、清洗、时滞对齐"""
    print_flush("="*70)
    print_flush("Step 1: 数据准备")
    print_flush("="*70)
    
    df = pd.read_csv('data/processed_data.csv')
    
    # 去除异常样本
    abnormal = list(range(1045, 1062))
    df = df[~df.index.isin(abnormal)].reset_index(drop=True)
    print_flush(f"去除异常后: {len(df)} 行")
    
    # 时滞对齐
    co = df['CO浓度'].values
    var_list = ['机速']
    for i in range(1, 19):
        var_list.extend([f'负压_{i}', f'温度_{i}'])
    var_list.extend(['大烟道负压_1', '大烟道负压_2', '大烟道温度_1', '大烟道温度_2'])
    
    lag_results = {}
    for var in var_list:
        lags, cc = compute_xcorr(df[var].values, co, 60)
        best_idx = np.argmax(np.abs(cc))
        lag_results[var] = int(lags[best_idx])
    
    for var, lag in lag_results.items():
        if lag != 0:
            df[f'{var}_al'] = df[var].shift(-lag)
        else:
            df[f'{var}_al'] = df[var]
    
    al_cols = [f'{var}_al' for var in var_list]
    df = df.dropna(subset=al_cols + ['CO浓度']).reset_index(drop=True)
    print_flush(f"时滞对齐后: {len(df)} 行")
    
    return df, var_list, lag_results


# ============================================================
# Step 2: 相关性分析
# ============================================================
def step2_correlation_analysis(df, var_list):
    """Pearson和Spearman相关性分析"""
    print_flush("\n" + "="*70)
    print_flush("Step 2: 相关性分析（Pearson + Spearman）")
    print_flush("="*70)
    
    co = df['CO浓度'].values
    
    results = []
    
    print_flush(f"\n{'变量':15s} {'Pearson':>10s} {'p值':>10s} {'Spearman':>10s} {'p值':>10s} {'方向':>8s} {'强度':>8s}")
    print_flush("-" * 75)
    
    for var in var_list:
        col = f'{var}_al'
        x = df[col].values
        
        p_corr, p_pval = pearsonr(x, co)
        s_corr, s_pval = spearmanr(x, co)
        
        # 影响方向
        direction = "正相关↑" if p_corr > 0 else "负相关↓"
        
        # 影响强度
        abs_corr = abs(p_corr)
        if abs_corr > 0.3:
            strength = "强"
        elif abs_corr > 0.15:
            strength = "中"
        elif abs_corr > 0.05:
            strength = "弱"
        else:
            strength = "极弱"
        
        sig = "*" if p_pval < 0.05 else ""
        
        print_flush(f"{var:15s} {p_corr:10.4f} {p_pval:10.2e} {s_corr:10.4f} {s_pval:10.2e} {direction:>8s} {strength:>6s}{sig}")
        
        results.append({
            'variable': var,
            'pearson': round(p_corr, 4),
            'pearson_pval': float(p_pval),
            'spearman': round(s_corr, 4),
            'spearman_pval': float(s_pval),
            'direction': direction,
            'strength': strength,
            'significant': p_pval < 0.05
        })
    
    # 按Pearson绝对值排序
    results_sorted = sorted(results, key=lambda x: abs(x['pearson']), reverse=True)
    
    print_flush(f"\n相关性排序（按|Pearson|降序）:")
    print_flush(f"{'排名':>4s} {'变量':15s} {'Pearson':>10s} {'方向':>8s} {'强度':>6s}")
    print_flush("-" * 50)
    for rank, r in enumerate(results_sorted, 1):
        print_flush(f"  {rank:2d}  {r['variable']:15s} {r['pearson']:10.4f} {r['direction']:>8s} {r['strength']:>6s}")
    
    return results, results_sorted


# ============================================================
# Step 3: Ridge回归分析
# ============================================================
def step3_ridge_regression(df, var_list):
    """Ridge回归：线性影响方向和强度"""
    print_flush("\n" + "="*70)
    print_flush("Step 3: Ridge回归分析（线性影响方向和强度）")
    print_flush("="*70)
    
    # 只用物理特征（不含CO自回归）
    phys_features = [f'{v}_al' for v in var_list]
    
    X = df[phys_features].values
    y = df['CO浓度'].values
    
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    # Ridge回归（alpha=1.0防止过拟合）
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_s, y)
    
    y_pred = ridge.predict(X_s)
    r2 = r2_score(y, y_pred)
    print_flush(f"\nRidge回归 R² = {r2:.4f}")
    
    # 回归系数分析
    coeffs = ridge.coef_
    
    print_flush(f"\n{'排名':>4s} {'变量':15s} {'回归系数':>12s} {'影响方向':>12s} {'相对强度':>10s}")
    print_flush("-" * 58)
    
    sorted_idx = np.argsort(np.abs(coeffs))[::-1]
    
    ridge_results = []
    for rank, idx in enumerate(sorted_idx, 1):
        var_name = var_list[idx]
        coeff = coeffs[idx]
        direction = "正向(↑CO)" if coeff > 0 else "负向(↓CO)"
        
        # 相对强度（归一化到百分比）
        rel_strength = abs(coeff) / np.sum(np.abs(coeffs)) * 100
        
        print_flush(f"  {rank:2d}  {var_name:15s} {coeff:12.2f} {direction:>12s} {rel_strength:9.1f}%")
        
        ridge_results.append({
            'rank': rank,
            'variable': var_name,
            'coefficient': round(float(coeff), 2),
            'direction': direction,
            'relative_strength': round(rel_strength, 1)
        })
    
    return ridge_results, ridge, scaler


# ============================================================
# Step 4: 随机森林特征重要性
# ============================================================
def step4_random_forest(df, var_list):
    """随机森林：非线性特征重要性"""
    print_flush("\n" + "="*70)
    print_flush("Step 4: 随机森林特征重要性（非线性）")
    print_flush("="*70)
    
    phys_features = [f'{v}_al' for v in var_list]
    
    X = df[phys_features].values
    y = df['CO浓度'].values
    
    # 随机森林
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=12, 
        min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    rf.fit(X, y)
    
    y_pred = rf.predict(X)
    r2 = r2_score(y, y_pred)
    print_flush(f"\n随机森林 R² = {r2:.4f}")
    
    importance = rf.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]
    
    print_flush(f"\n{'排名':>4s} {'变量':15s} {'重要性':>10s} {'相对占比':>10s}")
    print_flush("-" * 44)
    
    rf_results = []
    for rank, idx in enumerate(sorted_idx, 1):
        var_name = var_list[idx]
        imp = importance[idx]
        rel = imp / np.sum(importance) * 100
        
        print_flush(f"  {rank:2d}  {var_name:15s} {imp:10.4f} {rel:9.1f}%")
        
        rf_results.append({
            'rank': rank,
            'variable': var_name,
            'importance': round(float(imp), 4),
            'relative_pct': round(rel, 1)
        })
    
    # 分类汇总
    pressure_imp = sum(importance[i] for i in range(len(var_list)) if '负压' in var_list[i])
    temp_imp = sum(importance[i] for i in range(len(var_list)) if '温度' in var_list[i] and '大烟道' not in var_list[i])
    speed_imp = importance[0]  # 机速
    flue_imp = sum(importance[i] for i in range(len(var_list)) if '大烟道' in var_list[i])
    
    print_flush(f"\n各类别总重要性:")
    print_flush(f"  机速:      {speed_imp:.4f} ({speed_imp*100:.1f}%)")
    print_flush(f"  负压类:    {pressure_imp:.4f} ({pressure_imp*100:.1f}%)")
    print_flush(f"  温度类:    {temp_imp:.4f} ({temp_imp*100:.1f}%)")
    print_flush(f"  大烟道类:  {flue_imp:.4f} ({flue_imp*100:.1f}%)")
    
    return rf_results, rf, {
        'speed': round(float(speed_imp), 4),
        'pressure': round(float(pressure_imp), 4),
        'temperature': round(float(temp_imp), 4),
        'flue': round(float(flue_imp), 4)
    }


# ============================================================
# Step 5: 偏依赖图（PDP）
# ============================================================
def step5_partial_dependence_plots(rf, df, var_list):
    """偏依赖图：展示单变量对CO的非线性影响"""
    print_flush("\n" + "="*70)
    print_flush("Step 5: 偏依赖图（PDP）")
    print_flush("="*70)
    
    phys_features = [f'{v}_al' for v in var_list]
    X = df[phys_features].values
    
    # 选Top 9最重要的特征画PDP
    importance = rf.feature_importances_
    top9_idx = np.argsort(importance)[::-1][:9]
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()
    
    pdp_results = []
    
    for plot_idx, feat_idx in enumerate(top9_idx):
        ax = axes[plot_idx]
        var_name = var_list[feat_idx]
        
        # 计算PDP
        display = partial_dependence(rf, X, [feat_idx], 
                                     kind='average', grid_resolution=50)
        
        pdp_vals = display['average'][0]
        feat_vals = display['grid_values'][0]
        
        ax.plot(feat_vals, pdp_vals, 'b-', lw=2.5)
        ax.fill_between(feat_vals, pdp_vals.min(), pdp_vals.max(), alpha=0.1, color='blue')
        
        # 判断影响方向和形状
        trend = pdp_vals[-1] - pdp_vals[0]
        if abs(trend) < 20:
            shape = "无明显影响"
            direction = "无"
        elif trend > 0:
            direction = "正向(↑CO)"
            # 判断是否线性
            mid_val = (pdp_vals[0] + pdp_vals[-1]) / 2
            if abs(pdp_vals[len(pdp_vals)//2] - mid_val) < 30:
                shape = "近似线性"
            else:
                shape = "非线性"
        else:
            direction = "负向(↓CO)"
            mid_val = (pdp_vals[0] + pdp_vals[-1]) / 2
            if abs(pdp_vals[len(pdp_vals)//2] - mid_val) < 30:
                shape = "近似线性"
            else:
                shape = "非线性"
        
        ax.set_title(f'{var_name}\n方向:{direction}, 形状:{shape}', fontsize=11)
        ax.set_xlabel(f'{var_name}', fontsize=10)
        ax.set_ylabel('预测CO均值 (mg/m³)', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 标注变化幅度
        delta = pdp_vals[-1] - pdp_vals[0]
        ax.text(0.05, 0.95, f'ΔCO={delta:+.0f} mg/m³',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        pdp_results.append({
            'variable': var_name,
            'direction': direction,
            'shape': shape,
            'co_change': round(float(delta), 1)
        })
    
    plt.suptitle('偏依赖图（PDP）：各物理参数对CO浓度的影响', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/Q2B_pdp_top9.png', dpi=200, bbox_inches='tight')
    plt.close()
    print_flush("  ✓ figures/Q2B_pdp_top9.png")
    
    return pdp_results


# ============================================================
# Step 6: 分区分析
# ============================================================
def step6_zone_analysis(df):
    """前/中/后段风箱的分区影响分析"""
    print_flush("\n" + "="*70)
    print_flush("Step 6: 分区影响分析（前/中/后段）")
    print_flush("="*70)
    
    co = df['CO浓度'].values
    
    # 定义分区
    zones = {
        '前段(1-6#)': list(range(1, 7)),
        '中段(7-12#)': list(range(7, 13)),
        '后段(13-18#)': list(range(13, 19))
    }
    
    zone_results = {}
    
    print_flush(f"\n{'分区':15s} {'负压相关':>10s} {'温度相关':>10s} {'负压强度':>10s} {'温度强度':>10s}")
    print_flush("-" * 58)
    
    for zone_name, bellows in zones.items():
        # 负压均值
        p_cols = [f'负压_{i}_al' for i in bellows]
        p_mean = df[p_cols].mean(axis=1).values
        p_corr = pearsonr(p_mean, co)[0]
        
        # 温度均值
        t_cols = [f'温度_{i}_al' for i in bellows]
        t_mean = df[t_cols].mean(axis=1).values
        t_corr = pearsonr(t_mean, co)[0]
        
        p_strength = "强" if abs(p_corr) > 0.3 else ("中" if abs(p_corr) > 0.15 else "弱")
        t_strength = "强" if abs(t_corr) > 0.3 else ("中" if abs(t_corr) > 0.15 else "弱")
        
        p_dir = "↑" if p_corr > 0 else "↓"
        t_dir = "↑" if t_corr > 0 else "↓"
        
        print_flush(f"{zone_name:15s} {p_corr:9.4f}{p_dir} {t_corr:9.4f}{t_dir} {p_strength:>10s} {t_strength:>10s}")
        
        zone_results[zone_name] = {
            'pressure_corr': round(float(p_corr), 4),
            'temp_corr': round(float(t_corr), 4),
            'pressure_strength': p_strength,
            'temp_strength': t_strength,
            'pressure_direction': p_dir,
            'temp_direction': t_dir
        }
    
    # 画分区图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    zone_names = list(zones.keys())
    p_corrs = [zone_results[z]['pressure_corr'] for z in zone_names]
    t_corrs = [zone_results[z]['temp_corr'] for z in zone_names]
    
    colors_p = ['#2196F3', '#FF9800', '#F44336']
    colors_t = ['#2196F3', '#FF9800', '#F44336']
    
    axes[0].bar(zone_names, p_corrs, color=colors_p, alpha=0.8)
    axes[0].axhline(y=0, color='black', lw=0.5)
    axes[0].set_ylabel('与CO的Pearson相关系数')
    axes[0].set_title('各分区负压对CO的影响')
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(p_corrs):
        axes[0].text(i, v + (0.005 if v >= 0 else -0.015), f'{v:.4f}', 
                    ha='center', va='bottom' if v >= 0 else 'top', fontsize=11, fontweight='bold')
    
    axes[1].bar(zone_names, t_corrs, color=colors_t, alpha=0.8)
    axes[1].axhline(y=0, color='black', lw=0.5)
    axes[1].set_ylabel('与CO的Pearson相关系数')
    axes[1].set_title('各分区温度对CO的影响')
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(t_corrs):
        axes[1].text(i, v + (0.005 if v >= 0 else -0.015), f'{v:.4f}', 
                    ha='center', va='bottom' if v >= 0 else 'top', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figures/Q2B_zone_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    print_flush("  ✓ figures/Q2B_zone_analysis.png")
    
    return zone_results


# ============================================================
# Step 7: 综合可视化
# ============================================================
def step7_visualization(corr_sorted, ridge_results, rf_results, rf_category):
    """综合可视化"""
    print_flush("\n" + "="*70)
    print_flush("Step 7: 综合可视化")
    print_flush("="*70)
    
    os.makedirs('figures', exist_ok=True)
    
    # ---- 图1: 相关性排序图 ----
    fig, ax = plt.subplots(figsize=(10, 12))
    
    top20 = corr_sorted[:20]
    names = [r['variable'] for r in top20][::-1]
    corrs = [r['pearson'] for r in top20][::-1]
    
    colors = ['#F44336' if c > 0 else '#2196F3' for c in corrs]
    
    ax.barh(range(len(names)), corrs, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Pearson相关系数', fontsize=12)
    ax.set_title('Top 20: 物理参数与CO浓度的相关性', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', lw=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    # 图例
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor='#F44336', alpha=0.8, label='正相关(↑CO)'),
        Patch(facecolor='#2196F3', alpha=0.8, label='负相关(↓CO)'),
    ]
    ax.legend(handles=legend, fontsize=10, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('figures/Q2B_correlation_ranking.png', dpi=200, bbox_inches='tight')
    plt.close()
    print_flush("  ✓ figures/Q2B_correlation_ranking.png")
    
    # ---- 图2: Ridge回归系数图 ----
    fig, ax = plt.subplots(figsize=(10, 12))
    
    top20_ridge = ridge_results[:20]
    names_r = [r['variable'] for r in top20_ridge][::-1]
    coeffs_r = [r['coefficient'] for r in top20_ridge][::-1]
    
    colors_r = ['#F44336' if c > 0 else '#2196F3' for c in coeffs_r]
    
    ax.barh(range(len(names_r)), coeffs_r, color=colors_r, alpha=0.8)
    ax.set_yticks(range(len(names_r)))
    ax.set_yticklabels(names_r, fontsize=10)
    ax.set_xlabel('Ridge回归系数（标准化后）', fontsize=12)
    ax.set_title('Top 20: Ridge回归系数（线性影响方向和强度）', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', lw=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(handles=legend, fontsize=10, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('figures/Q2B_ridge_coefficients.png', dpi=200, bbox_inches='tight')
    plt.close()
    print_flush("  ✓ figures/Q2B_ridge_coefficients.png")
    
    # ---- 图3: 随机森林重要性图 ----
    fig, ax = plt.subplots(figsize=(10, 12))
    
    top20_rf = rf_results[:20]
    names_rf = [r['variable'] for r in top20_rf][::-1]
    imps_rf = [r['importance'] for r in top20_rf][::-1]
    
    # 按类别着色
    colors_rf = []
    for name in names_rf:
        if '负压' in name:
            colors_rf.append('#2196F3')
        elif '温度' in name and '大烟道' not in name:
            colors_rf.append('#F44336')
        elif '机速' in name:
            colors_rf.append('#4CAF50')
        elif '大烟道' in name:
            colors_rf.append('#FF9800')
    
    ax.barh(range(len(names_rf)), imps_rf, color=colors_rf, alpha=0.8)
    ax.set_yticks(range(len(names_rf)))
    ax.set_yticklabels(names_rf, fontsize=10)
    ax.set_xlabel('特征重要性 (Gain)', fontsize=12)
    ax.set_title('Top 20: 随机森林特征重要性（非线性）', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    legend_rf = [
        Patch(facecolor='#2196F3', alpha=0.8, label='负压'),
        Patch(facecolor='#F44336', alpha=0.8, label='温度'),
        Patch(facecolor='#4CAF50', alpha=0.8, label='机速'),
        Patch(facecolor='#FF9800', alpha=0.8, label='大烟道'),
    ]
    ax.legend(handles=legend_rf, fontsize=10, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('figures/Q2B_rf_importance.png', dpi=200, bbox_inches='tight')
    plt.close()
    print_flush("  ✓ figures/Q2B_rf_importance.png")
    
    # ---- 图4: 四合一总结图 ----
    fig = plt.figure(figsize=(16, 12))
    
    # 子图1: 类别重要性饼图
    ax1 = fig.add_subplot(2, 2, 1)
    labels = ['机速', '负压', '温度', '大烟道']
    sizes = [rf_category['speed'], rf_category['pressure'], 
             rf_category['temperature'], rf_category['flue']]
    colors_pie = ['#4CAF50', '#2196F3', '#F44336', '#FF9800']
    ax1.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', 
            startangle=90, textprops={'fontsize': 12})
    ax1.set_title('各类别对CO的影响占比', fontsize=14, fontweight='bold')
    
    # 子图2: Top 10相关性
    ax2 = fig.add_subplot(2, 2, 2)
    top10 = corr_sorted[:10]
    names_t = [r['variable'] for r in top10][::-1]
    corrs_t = [r['pearson'] for r in top10][::-1]
    colors_t = ['#F44336' if c > 0 else '#2196F3' for c in corrs_t]
    ax2.barh(range(len(names_t)), corrs_t, color=colors_t, alpha=0.8)
    ax2.set_yticks(range(len(names_t)))
    ax2.set_yticklabels(names_t, fontsize=10)
    ax2.set_xlabel('Pearson相关系数')
    ax2.set_title('Top 10 相关性', fontsize=13, fontweight='bold')
    ax2.axvline(x=0, color='black', lw=0.5)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 子图3: Top 10 Ridge系数
    ax3 = fig.add_subplot(2, 2, 3)
    top10_r = ridge_results[:10]
    names_tr = [r['variable'] for r in top10_r][::-1]
    coeffs_tr = [r['coefficient'] for r in top10_r][::-1]
    colors_tr = ['#F44336' if c > 0 else '#2196F3' for c in coeffs_tr]
    ax3.barh(range(len(names_tr)), coeffs_tr, color=colors_tr, alpha=0.8)
    ax3.set_yticks(range(len(names_tr)))
    ax3.set_yticklabels(names_tr, fontsize=10)
    ax3.set_xlabel('Ridge回归系数')
    ax3.set_title('Top 10 Ridge系数', fontsize=13, fontweight='bold')
    ax3.axvline(x=0, color='black', lw=0.5)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 子图4: Top 10 RF重要性
    ax4 = fig.add_subplot(2, 2, 4)
    top10_rf = rf_results[:10]
    names_trf = [r['variable'] for r in top10_rf][::-1]
    imps_trf = [r['importance'] for r in top10_rf][::-1]
    ax4.barh(range(len(names_trf)), imps_trf, color='#333333', alpha=0.8)
    ax4.set_yticks(range(len(names_trf)))
    ax4.set_yticklabels(names_trf, fontsize=10)
    ax4.set_xlabel('随机森林重要性')
    ax4.set_title('Top 10 随机森林重要性', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('物理影响规律综合分析', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figures/Q2B_summary_4in1.png', dpi=200, bbox_inches='tight')
    plt.close()
    print_flush("  ✓ figures/Q2B_summary_4in1.png")


# ============================================================
# Step 8: 保存结果
# ============================================================
def step8_save_results(corr_results, corr_sorted, ridge_results, rf_results, 
                       rf_category, pdp_results, zone_results, lag_results):
    """保存所有分析结果"""
    print_flush("\n" + "="*70)
    print_flush("Step 8: 保存结果")
    print_flush("="*70)
    
    os.makedirs('results', exist_ok=True)
    
    final = {
        'correlation_analysis': {
            'all_results': corr_results,
            'sorted_by_abs_pearson': corr_sorted
        },
        'ridge_regression': ridge_results,
        'random_forest': {
            'feature_importance': rf_results,
            'category_importance': rf_category
        },
        'pdp_analysis': pdp_results,
        'zone_analysis': zone_results,
        'time_lag_results': {k: int(v) for k, v in lag_results.items()}
    }
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    with open('results/Q2B_physical_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    print_flush("  ✓ results/Q2B_physical_analysis.json")


# ============================================================
# 主程序
# ============================================================
def main():
    print_flush("="*70)
    print_flush("问题2 - 模型B：物理影响规律分析")
    print_flush("="*70)
    
    os.makedirs('figures', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Step 1: 数据准备
    df, var_list, lag_results = step1_data_prep()
    
    # Step 2: 相关性分析
    corr_results, corr_sorted = step2_correlation_analysis(df, var_list)
    
    # Step 3: Ridge回归
    ridge_results, ridge_model, scaler = step3_ridge_regression(df, var_list)
    
    # Step 4: 随机森林
    rf_results, rf_model, rf_category = step4_random_forest(df, var_list)
    
    # Step 5: 偏依赖图
    pdp_results = step5_partial_dependence_plots(rf_model, df, var_list)
    
    # Step 6: 分区分析
    zone_results = step6_zone_analysis(df)
    
    # Step 7: 综合可视化
    step7_visualization(corr_sorted, ridge_results, rf_results, rf_category)
    
    # Step 8: 保存结果
    step8_save_results(corr_results, corr_sorted, ridge_results, rf_results,
                       rf_category, pdp_results, zone_results, lag_results)
    
    # ============================================================
    # 物理影响规律总结
    # ============================================================
    print_flush("\n" + "="*70)
    print_flush("物理影响规律总结")
    print_flush("="*70)
    
    print_flush("\n1. 最重要的影响因素（随机森林排序）:")
    for r in rf_results[:10]:
        print_flush(f"   {r['rank']:2d}. {r['variable']:15s} ({r['relative_pct']:.1f}%)")
    
    print_flush("\n2. 影响方向（Ridge回归）:")
    positive = [r for r in ridge_results if '正向' in r['direction']][:5]
    negative = [r for r in ridge_results if '负向' in r['direction']][:5]
    print_flush("   使CO升高的因素(正向):")
    for r in positive:
        print_flush(f"     {r['variable']:15s}: 系数={r['coefficient']:+.1f}")
    print_flush("   使CO降低的因素(负向):")
    for r in negative:
        print_flush(f"     {r['variable']:15s}: 系数={r['coefficient']:+.1f}")
    
    print_flush("\n3. 分区影响规律:")
    for zone, result in zone_results.items():
        p_dir = "增大负压→CO↑" if result['pressure_direction'] == '↑' else "增大负压→CO↓"
        t_dir = "升高温度→CO↑" if result['temp_direction'] == '↑' else "升高温度→CO↓"
        print_flush(f"   {zone}: {p_dir}({result['pressure_strength']}), {t_dir}({result['temp_strength']})")
    
    print_flush("\n4. 各类别影响占比:")
    print_flush(f"   机速:     {rf_category['speed']*100:.1f}%")
    print_flush(f"   负压类:   {rf_category['pressure']*100:.1f}%")
    print_flush(f"   温度类:   {rf_category['temperature']*100:.1f}%")
    print_flush(f"   大烟道:   {rf_category['flue']*100:.1f}%")
    
    print_flush(f"\n📁 输出文件:")
    print_flush(f"  results/Q2B_physical_analysis.json")
    print_flush(f"  figures/Q2B_correlation_ranking.png")
    print_flush(f"  figures/Q2B_ridge_coefficients.png")
    print_flush(f"  figures/Q2B_rf_importance.png")
    print_flush(f"  figures/Q2B_pdp_top9.png")
    print_flush(f"  figures/Q2B_zone_analysis.png")
    print_flush(f"  figures/Q2B_summary_4in1.png")


if __name__ == '__main__':
    main()
