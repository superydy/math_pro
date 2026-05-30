#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题3：基于CO浓度预测模型的风箱负压优化调控
==============================================

完整流程：
1. 复用Q2数据预处理与特征工程（保持一致性）
2. 加载Q2最优超参数，重新训练全量XGBoost模型
3. 确定各风箱负压调节范围（历史5%-95%分位数）
4. 构建稳态优化模型（不动点迭代处理CO自回归特征）
5. PSO粒子群优化：最小化稳态CO浓度
6. 敏感性分析：量化各风箱负压对CO的边际影响
7. 可视化与结果保存
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from scipy.signal import fftconvolve
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 字体设置（中文显示）
# ============================================================
_font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
fm.fontManager.addfont(_font_path)
_fp = fm.FontProperties(fname=_font_path)
_FONT = _fp.get_name()
plt.rcParams['font.family'] = _FONT
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


def _fp_label(ax):
    """将坐标轴所有文字设为中文字体"""
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontproperties(_fp)


def print_flush(msg):
    print(msg, flush=True)


# ============================================================
# 复用Q2数据处理（保持与Q2完全一致）
# ============================================================

def compute_xcorr(x, y, max_lag=60):
    xn = (x - np.mean(x)) / (np.std(x) * len(x) + 1e-8)
    yn = (y - np.mean(y)) / (np.std(y) + 1e-8)
    corr = fftconvolve(xn[::-1], yn, mode='full')
    center = len(xn) - 1
    lags = np.arange(-max_lag, max_lag + 1)
    cc = corr[center - max_lag: center + max_lag + 1]
    return lags, cc


def load_and_preprocess():
    """
    Steps 1-3: 数据加载、时滞对齐、特征工程（与Q2完全一致）
    直接加载Q2结果中的最优时滞，避免重复计算互相关
    """
    # Step 1: 加载数据
    df = pd.read_csv('data/processed_data.csv')
    abnormal = list(range(1045, 1062))
    df = df[~df.index.isin(abnormal)].reset_index(drop=True)

    # Step 2: 时滞对齐（直接复用Q2已知的最优时滞）
    with open('results/Q2_complete_results.json', 'r', encoding='utf-8') as f:
        q2_results = json.load(f)
    lag_results = q2_results['lag_results']

    var_list = ['机速']
    for i in range(1, 19):
        var_list.extend([f'负压_{i}', f'温度_{i}'])
    var_list.extend(['大烟道负压_1', '大烟道负压_2', '大烟道温度_1', '大烟道温度_2'])

    df_aligned = df.copy()
    for var in var_list:
        lag = lag_results.get(var, 0)
        df_aligned[f'{var}_al'] = df_aligned[var].shift(-lag)

    al_cols = [f'{var}_al' for var in var_list]
    df_aligned = df_aligned.dropna(subset=al_cols + ['CO浓度']).reset_index(drop=True)

    # Step 3: 特征工程（84个特征，与Q2完全一致）
    physical_features = ['机速_al']
    for i in range(1, 19):
        physical_features.extend([f'负压_{i}_al', f'温度_{i}_al'])
    physical_features.extend(['大烟道负压_1_al', '大烟道负压_2_al',
                               '大烟道温度_1_al', '大烟道温度_2_al'])

    gradient_features = []
    for i in range(1, 18):
        df_aligned[f'pgrad_{i}'] = df_aligned[f'负压_{i+1}_al'] - df_aligned[f'负压_{i}_al']
        df_aligned[f'tgrad_{i}'] = df_aligned[f'温度_{i+1}_al'] - df_aligned[f'温度_{i}_al']
        gradient_features.extend([f'pgrad_{i}', f'tgrad_{i}'])

    pcols = [f'负压_{i}_al' for i in range(1, 19)]
    df_aligned['p_mean'] = df_aligned[pcols].mean(axis=1)
    df_aligned['p_mid']  = df_aligned[[f'负压_{i}_al' for i in range(6, 13)]].mean(axis=1)
    df_aligned['p_back'] = df_aligned[[f'负压_{i}_al' for i in range(13, 19)]].mean(axis=1)
    df_aligned['t_back'] = df_aligned[[f'温度_{i}_al' for i in range(13, 19)]].mean(axis=1)
    stat_features = ['p_mean', 'p_mid', 'p_back', 't_back']

    df_aligned['co_lag1']  = df_aligned['CO浓度'].shift(1)
    df_aligned['co_lag2']  = df_aligned['CO浓度'].shift(2)
    df_aligned['co_lag5']  = df_aligned['CO浓度'].shift(5)
    df_aligned['co_ma5']   = df_aligned['CO浓度'].rolling(5).mean()
    df_aligned['co_diff1'] = df_aligned['CO浓度'].diff(1)
    co_features = ['co_lag1', 'co_lag2', 'co_lag5', 'co_ma5', 'co_diff1']

    all_features = physical_features + gradient_features + stat_features + co_features
    df_final = df_aligned.dropna(subset=all_features + ['CO浓度']).reset_index(drop=True)

    X = df_final[all_features].values
    y = df_final['CO浓度'].values

    return X, y, all_features, df_final


# ============================================================
# Step 4: 训练全量预测模型
# ============================================================

def train_full_model(X, y, best_params):
    """加载Q2最优参数，在全量数据上训练XGBoost"""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    model.fit(X_s, y, verbose=False)
    return model, scaler


# ============================================================
# Step 5: 确定风箱负压调节范围
# ============================================================

def compute_pressure_bounds(df_final, pct_low=5, pct_high=95):
    """
    从历史数据计算各风箱负压的可行调节范围
    采用5%-95%分位数，排除传感器异常和极端工况
    """
    bounds = {}
    for i in range(1, 19):
        col = f'负压_{i}_al'
        vals = df_final[col].dropna().values
        bounds[f'负压_{i}'] = {
            'min':    float(np.percentile(vals, pct_low)),
            'median': float(np.median(vals)),
            'max':    float(np.percentile(vals, pct_high)),
            'mean':   float(np.mean(vals)),
            'std':    float(np.std(vals)),
        }
    return bounds


# ============================================================
# 优化目标：稳态CO不动点迭代
# ============================================================

def build_feature_vector(pressures, fixed_vals, co_star, all_features):
    """
    构建单个样本的84维特征向量

    参数
    ----
    pressures  : array (18,)  — 18个风箱负压决策变量
    fixed_vals : dict         — 非压力变量（温度/机速/大烟道）的固定中位值
    co_star    : float        — 当前稳态CO估计值
    all_features: list        — 84个特征名（与Q2训练时完全一致）
    """
    feat = {}

    # 物理基础特征
    feat['机速_al'] = fixed_vals['机速_al']
    for i in range(1, 19):
        feat[f'负压_{i}_al'] = pressures[i - 1]
        feat[f'温度_{i}_al'] = fixed_vals[f'温度_{i}_al']
    feat['大烟道负压_1_al'] = fixed_vals['大烟道负压_1_al']
    feat['大烟道负压_2_al'] = fixed_vals['大烟道负压_2_al']
    feat['大烟道温度_1_al'] = fixed_vals['大烟道温度_1_al']
    feat['大烟道温度_2_al'] = fixed_vals['大烟道温度_2_al']

    # 梯度特征（压力梯度由决策变量推导，温度梯度固定）
    for i in range(1, 18):
        feat[f'pgrad_{i}'] = pressures[i] - pressures[i - 1]
        feat[f'tgrad_{i}'] = fixed_vals[f'tgrad_{i}']

    # 统计特征（压力统计量由决策变量推导）
    feat['p_mean'] = float(np.mean(pressures))
    feat['p_mid']  = float(np.mean(pressures[5:12]))   # 6#-12#
    feat['p_back'] = float(np.mean(pressures[12:18]))  # 13#-18#
    feat['t_back'] = fixed_vals['t_back']

    # CO自回归特征（稳态假设：CO不再变化）
    feat['co_lag1']  = co_star
    feat['co_lag2']  = co_star
    feat['co_lag5']  = co_star
    feat['co_ma5']   = co_star
    feat['co_diff1'] = 0.0   # 稳态：ΔCO = 0

    return np.array([feat[name] for name in all_features])


def steady_state_co(pressures, fixed_vals, model, scaler, all_features,
                    init_co=None, max_iter=80, tol=0.5):
    """
    不动点迭代求解稳态CO：CO* = f(pressures, CO*)

    物理含义：在当前负压设定下，CO浓度最终收敛的均衡值。
    迭代使用阻尼更新防止振荡：CO_new = α·f(CO) + (1-α)·CO

    返回
    ----
    co_star    : 收敛的稳态CO浓度
    converged  : 是否在max_iter步内收敛
    """
    co_star = init_co if init_co is not None else fixed_vals['co_median']
    alpha = 0.4   # 阻尼系数

    for _ in range(max_iter):
        x = build_feature_vector(pressures, fixed_vals, co_star, all_features)
        x_s = scaler.transform(x.reshape(1, -1))
        co_new = float(model.predict(x_s)[0])
        co_new = max(0.0, co_new)   # CO不能为负

        if abs(co_new - co_star) < tol:
            return co_new, True

        co_star = alpha * co_new + (1.0 - alpha) * co_star

    return co_star, False


# ============================================================
# Step 6: PSO粒子群优化
# ============================================================

def pso_optimize(pressure_bounds, fixed_vals, model, scaler, all_features,
                 n_particles=30, n_iter=50):
    """
    PSO粒子群优化：求使稳态CO最小的18个风箱负压设定

    决策变量 : p ∈ R^18，各分量对应 负压_1 ~ 负压_18
    目标函数 : min CO*(p)  （稳态CO浓度）
    约束     : p_i^{min} ≤ p_i ≤ p_i^{max}（历史5%-95%范围）

    PSO参数  : 惯性权重线性衰减 w: 0.9→0.4，c1=c2=2.0，速度限幅
    """
    print_flush(f"\n  粒子数: {n_particles}，迭代数: {n_iter}")
    print_flush(f"  总共评估约 {n_particles * n_iter} 次稳态CO")

    n_dim = 18
    lb = np.array([pressure_bounds[f'负压_{i}']['min'] for i in range(1, 19)])
    ub = np.array([pressure_bounds[f'负压_{i}']['max'] for i in range(1, 19)])
    v_max = (ub - lb) * 0.2

    np.random.seed(42)

    # 初始化粒子（包含历史中位值作为一个粒子）
    positions = lb + np.random.rand(n_particles, n_dim) * (ub - lb)
    positions[0] = np.array([pressure_bounds[f'负压_{i}']['median'] for i in range(1, 19)])
    velocities = np.zeros((n_particles, n_dim))

    pbest_pos = positions.copy()
    pbest_val = np.full(n_particles, np.inf)
    gbest_pos = positions[0].copy()
    gbest_val = np.inf

    w_max, w_min, c1, c2 = 0.9, 0.4, 2.0, 2.0
    history = []
    t0 = time.time()

    for it in range(n_iter):
        w = w_max - (w_max - w_min) * it / max(n_iter - 1, 1)

        for p in range(n_particles):
            co, _ = steady_state_co(positions[p], fixed_vals, model, scaler,
                                    all_features, init_co=fixed_vals['co_median'])
            if co < pbest_val[p]:
                pbest_val[p] = co
                pbest_pos[p] = positions[p].copy()
            if co < gbest_val:
                gbest_val = co
                gbest_pos = positions[p].copy()

        history.append(float(gbest_val))

        if (it + 1) % 10 == 0 or it == 0:
            print_flush(f"  迭代 {it+1:3d}/{n_iter}: "
                        f"最优稳态CO = {gbest_val:.1f} mg/m³  "
                        f"(耗时 {time.time()-t0:.0f}s)")

        r1 = np.random.rand(n_particles, n_dim)
        r2 = np.random.rand(n_particles, n_dim)
        velocities = (w * velocities
                      + c1 * r1 * (pbest_pos - positions)
                      + c2 * r2 * (gbest_pos - positions))
        velocities = np.clip(velocities, -v_max, v_max)
        positions = np.clip(positions + velocities, lb, ub)

    print_flush(f"\n  PSO完成，总耗时 {time.time()-t0:.0f}s")
    return gbest_pos, gbest_val, history


# ============================================================
# Step 7: 敏感性分析
# ============================================================

def sensitivity_analysis(optimal_pressures, fixed_vals, model, scaler,
                          all_features, pressure_bounds, n_points=20):
    """
    单变量扫描敏感性分析：固定其他17个风箱在最优值，
    单独扫描第i个风箱，量化其对稳态CO的边际影响。
    """
    sensitivities = []

    for i in range(18):
        key = f'负压_{i+1}'
        lo = pressure_bounds[key]['min']
        hi = pressure_bounds[key]['max']
        scan_vals = np.linspace(lo, hi, n_points)
        co_vals = []

        for val in scan_vals:
            p_test = optimal_pressures.copy()
            p_test[i] = val
            co, _ = steady_state_co(p_test, fixed_vals, model, scaler,
                                    all_features, init_co=fixed_vals['co_median'])
            co_vals.append(co)

        co_arr = np.array(co_vals)
        co_range = float(co_arr.max() - co_arr.min())

        # 估计最优点处的斜率（中心差分）
        mid = n_points // 2
        slope = float((co_arr[mid+1] - co_arr[mid-1]) /
                      (scan_vals[mid+1] - scan_vals[mid-1] + 1e-10))

        sensitivities.append({
            'variable': key,
            'index': i + 1,
            'range': co_range,
            'slope': slope,
            'optimal': float(optimal_pressures[i]),
            'median': pressure_bounds[key]['median'],
            'scan_vals': scan_vals.tolist(),
            'co_vals': co_arr.tolist(),
        })

    sensitivities.sort(key=lambda x: x['range'], reverse=True)
    return sensitivities


# ============================================================
# Step 8: 可视化
# ============================================================

def visualize(optimal_pressures, baseline_co, optimal_co, history,
              sensitivities, pressure_bounds):
    os.makedirs('figures', exist_ok=True)

    # ---- 图1: PSO收敛曲线 ----
    fig, ax = plt.subplots(figsize=(9, 5))
    iters = range(1, len(history) + 1)
    ax.plot(iters, history, color='#2980B9', lw=2)
    ax.axhline(y=baseline_co, color='#C0392B', ls='--', lw=1.5,
               label=f'基准CO（历史中位负压）: {baseline_co:.0f} mg/m³')
    ax.axhline(y=optimal_co, color='#27AE60', ls='--', lw=1.5,
               label=f'PSO最优CO: {optimal_co:.0f} mg/m³')
    ax.fill_between(iters, history, baseline_co, where=[h < baseline_co for h in history],
                    alpha=0.15, color='#27AE60')
    ax.set_xlabel('PSO迭代次数', fontproperties=_fp)
    ax.set_ylabel('最优稳态CO浓度 (mg/m³)', fontproperties=_fp)
    ax.set_title('PSO优化收敛过程', fontproperties=_fp)
    ax.legend(prop=_fp, fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    _fp_label(ax)
    plt.tight_layout()
    plt.savefig('figures/Q3_pso_convergence.png', dpi=150)
    plt.close()
    print_flush("  ✓ figures/Q3_pso_convergence.png")

    # ---- 图2: 各风箱负压调节范围与最优决策 ----
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(18)
    lb_arr = [pressure_bounds[f'负压_{i}']['min'] for i in range(1, 19)]
    ub_arr = [pressure_bounds[f'负压_{i}']['max'] for i in range(1, 19)]
    med_arr = [pressure_bounds[f'负压_{i}']['median'] for i in range(1, 19)]

    bar_heights = [hi - lo for hi, lo in zip(ub_arr, lb_arr)]
    ax.bar(x, bar_heights, bottom=lb_arr, color='#AED6F1', alpha=0.55, label='调节范围 (5%~95%)')
    ax.plot(x, med_arr, 'o-', color='#2980B9', lw=1.5, ms=5, label='历史中位值')
    ax.plot(x, optimal_pressures, 's--', color='#C0392B', lw=1.5, ms=5, label='PSO最优值')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}#' for i in range(1, 19)], fontproperties=_fp)
    ax.set_xlabel('风箱编号', fontproperties=_fp)
    ax.set_ylabel('负压值 (Pa)', fontproperties=_fp)
    ax.set_title('各风箱负压调节范围与最优调控决策', fontproperties=_fp)
    ax.legend(prop=_fp)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    _fp_label(ax)
    plt.tight_layout()
    plt.savefig('figures/Q3_pressure_range.png', dpi=150)
    plt.close()
    print_flush("  ✓ figures/Q3_pressure_range.png")

    # ---- 图3: 最优负压 vs 历史中位（差值图）----
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(18)
    deltas = optimal_pressures - np.array(med_arr)
    colors = ['#C0392B' if d > 0 else '#2980B9' for d in deltas]
    ax.bar(x, deltas, color=colors, alpha=0.8)
    ax.axhline(y=0, color='black', lw=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}#' for i in range(1, 19)], fontproperties=_fp)
    ax.set_xlabel('风箱编号', fontproperties=_fp)
    ax.set_ylabel('负压调整量 (Pa)\n（最优值 - 历史中位值）', fontproperties=_fp)
    ax.set_title('各风箱负压调整方向与幅度', fontproperties=_fp)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#C0392B', alpha=0.8, label='增大负压'),
        Patch(facecolor='#2980B9', alpha=0.8, label='减小负压'),
    ]
    ax.legend(handles=legend_elements, prop=_fp)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    _fp_label(ax)
    plt.tight_layout()
    plt.savefig('figures/Q3_pressure_adjustment.png', dpi=150)
    plt.close()
    print_flush("  ✓ figures/Q3_pressure_adjustment.png")

    # ---- 图4: 敏感性分析 Top 10 ----
    top10 = sensitivities[:10]
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [s['variable'] for s in top10][::-1]
    ranges = [s['range'] for s in top10][::-1]
    slopes = [s['slope'] for s in top10][::-1]
    bar_colors = ['#C0392B' if sl > 0 else '#2980B9' for sl in slopes]

    ax.barh(range(10), ranges, color=bar_colors, alpha=0.8)
    ax.set_yticks(range(10))
    ax.set_yticklabels(labels, fontproperties=_fp)
    ax.set_xlabel('CO浓度变化范围 (mg/m³)', fontproperties=_fp)
    ax.set_title('各风箱负压敏感性分析 (Top 10)', fontproperties=_fp)
    ax.grid(axis='x', linestyle='--', alpha=0.4)

    legend_elements = [
        Patch(facecolor='#C0392B', alpha=0.8, label='增大负压 → CO升高'),
        Patch(facecolor='#2980B9', alpha=0.8, label='增大负压 → CO降低'),
    ]
    ax.legend(handles=legend_elements, prop=_fp)
    _fp_label(ax)
    plt.tight_layout()
    plt.savefig('figures/Q3_sensitivity.png', dpi=150)
    plt.close()
    print_flush("  ✓ figures/Q3_sensitivity.png")

    # ---- 图5: 典型风箱的CO-压力响应曲线（Top 4）----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for idx, s in enumerate(sensitivities[:4]):
        ax = axes[idx]
        ax.plot(s['scan_vals'], s['co_vals'], color='#2980B9', lw=2)
        ax.axvline(x=s['optimal'], color='#C0392B', ls='--', lw=1.5,
                   label=f"最优值: {s['optimal']:.1f}")
        ax.axvline(x=s['median'], color='gray', ls=':', lw=1.2,
                   label=f"历史中位: {s['median']:.1f}")
        ax.set_title(s['variable'], fontproperties=_fp)
        ax.set_xlabel('负压值 (Pa)', fontproperties=_fp)
        ax.set_ylabel('稳态CO (mg/m³)', fontproperties=_fp)
        ax.legend(prop=_fp, fontsize=8)
        ax.grid(linestyle='--', alpha=0.4)
        _fp_label(ax)
    plt.suptitle('敏感性最高的4个风箱 — CO响应曲线', fontproperties=_fp, y=1.01)
    plt.tight_layout()
    plt.savefig('figures/Q3_response_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print_flush("  ✓ figures/Q3_response_curves.png")


# ============================================================
# 主程序
# ============================================================

def main():
    print_flush("=" * 70)
    print_flush("问题3：CO排放最小化 — 风箱负压优化调控")
    print_flush("=" * 70)
    t_total = time.time()

    # ---- Steps 1-3: 数据与特征（复用Q2）----
    print_flush("\n[Steps 1-3] 数据加载与特征工程（复用Q2流程）...")
    X, y, all_features, df_final = load_and_preprocess()
    print_flush(f"  样本数: {len(y)}，特征数: {len(all_features)}")

    # ---- Step 4: 重训练全量模型 ----
    print_flush("\n[Step 4] 加载Q2最优参数，训练全量XGBoost模型...")
    with open('results/Q2_complete_results.json', 'r', encoding='utf-8') as f:
        q2_results = json.load(f)

    best_params = q2_results['model_info']['best_params']
    best_params['max_depth']    = int(best_params['max_depth'])
    best_params['n_estimators'] = int(best_params['n_estimators'])

    model, scaler = train_full_model(X, y, best_params)
    r2_full = r2_score(y, model.predict(scaler.transform(X)))
    print_flush(f"  全量模型 R² = {r2_full:.4f}（验证与Q2一致）")

    # ---- Step 5: 压力调节范围 ----
    print_flush("\n[Step 5] 确定各风箱负压调节范围（5%~95%分位数）...")
    pressure_bounds = compute_pressure_bounds(df_final)

    print_flush(f"\n  {'风箱':8s} {'下限(5%)':>10s} {'中位值':>10s} {'上限(95%)':>10s} {'调节范围':>10s}")
    print_flush("  " + "-" * 52)
    for i in range(1, 19):
        b = pressure_bounds[f'负压_{i}']
        print_flush(f"  负压_{i:2d}   {b['min']:10.1f} {b['median']:10.1f} "
                    f"{b['max']:10.1f} {b['max']-b['min']:10.1f}")

    # ---- 固定非压力特征（温度/机速/大烟道）为历史中位值 ----
    fixed_vars = ['机速_al']
    for i in range(1, 19):
        fixed_vars.append(f'温度_{i}_al')
    fixed_vars += ['大烟道负压_1_al', '大烟道负压_2_al', '大烟道温度_1_al', '大烟道温度_2_al']

    fixed_vals = {v: float(df_final[v].median()) for v in fixed_vars}
    for i in range(1, 18):
        fixed_vals[f'tgrad_{i}'] = (fixed_vals[f'温度_{i+1}_al'] - fixed_vals[f'温度_{i}_al'])
    fixed_vals['t_back']    = float(df_final['t_back'].median())
    fixed_vals['co_median'] = float(np.median(y))

    print_flush(f"\n  固定变量（温度/机速/大烟道）设为历史中位值")
    print_flush(f"  历史CO中位浓度: {fixed_vals['co_median']:.1f} mg/m³")

    # 基准CO：历史中位负压下的稳态预测
    median_pressures = np.array([pressure_bounds[f'负压_{i}']['median'] for i in range(1, 19)])
    baseline_co, conv = steady_state_co(median_pressures, fixed_vals, model, scaler, all_features)
    print_flush(f"  历史中位负压稳态CO: {baseline_co:.1f} mg/m³ (收敛={conv})")

    # ---- Step 6: PSO优化 ----
    print_flush("\n[Step 6] PSO粒子群优化...")
    optimal_pressures, optimal_co, history = pso_optimize(
        pressure_bounds, fixed_vals, model, scaler, all_features,
        n_particles=30, n_iter=50
    )

    reduction = baseline_co - optimal_co
    pct = reduction / baseline_co * 100

    print_flush("\n" + "=" * 70)
    print_flush("优化结果汇总")
    print_flush("=" * 70)
    print_flush(f"  基准稳态CO浓度 : {baseline_co:.1f} mg/m³")
    print_flush(f"  最优稳态CO浓度 : {optimal_co:.1f} mg/m³")
    print_flush(f"  CO降低量       : {reduction:.1f} mg/m³  ({pct:.1f}%)")

    print_flush(f"\n  {'风箱':8s} {'下限':>8s} {'历史中位':>10s} {'最优值':>10s} {'上限':>8s} {'调整量':>10s}")
    print_flush("  " + "-" * 58)
    for i in range(1, 19):
        b = pressure_bounds[f'负压_{i}']
        opt = optimal_pressures[i - 1]
        delta = opt - b['median']
        print_flush(f"  负压_{i:2d}   {b['min']:8.1f} {b['median']:10.1f} "
                    f"{opt:10.1f} {b['max']:8.1f} {delta:+10.1f}")

    # ---- Step 7: 敏感性分析 ----
    print_flush("\n[Step 7] 敏感性分析...")
    sensitivities = sensitivity_analysis(
        optimal_pressures, fixed_vals, model, scaler,
        all_features, pressure_bounds, n_points=20
    )

    print_flush(f"\n  {'排名':4s} {'变量':10s} {'CO变化范围(mg/m³)':>20s} {'影响方向':>14s}")
    print_flush("  " + "-" * 52)
    for rank, s in enumerate(sensitivities[:10], 1):
        direction = "增压→CO升高" if s['slope'] > 0 else "增压→CO降低"
        print_flush(f"  {rank:3d}  {s['variable']:10s} {s['range']:20.1f} {direction:>14s}")

    # ---- Step 8: 可视化 ----
    print_flush("\n[Step 8] 生成可视化图表...")
    visualize(optimal_pressures, baseline_co, optimal_co, history,
              sensitivities, pressure_bounds)

    # ---- 保存结果 ----
    os.makedirs('results', exist_ok=True)
    results_out = {
        'optimization_summary': {
            'baseline_co_mg_m3': round(baseline_co, 2),
            'optimal_co_mg_m3':  round(optimal_co, 2),
            'reduction_mg_m3':   round(reduction, 2),
            'reduction_pct':     round(pct, 2),
        },
        'pso_config': {
            'n_particles': 30,
            'n_iterations': 50,
            'convergence_history': history,
        },
        'pressure_ranges_and_decision': {
            f'负压_{i}': {
                'lower_bound': round(pressure_bounds[f'负压_{i}']['min'], 2),
                'median':      round(pressure_bounds[f'负压_{i}']['median'], 2),
                'upper_bound': round(pressure_bounds[f'负压_{i}']['max'], 2),
                'optimal':     round(float(optimal_pressures[i - 1]), 2),
                'adjustment':  round(float(optimal_pressures[i - 1]) -
                                     pressure_bounds[f'负压_{i}']['median'], 2),
            }
            for i in range(1, 19)
        },
        'sensitivity_ranking': [
            {
                'rank': rank,
                'variable': s['variable'],
                'co_range_mg_m3': round(s['range'], 2),
                'direction': '正向（增压→CO升）' if s['slope'] > 0 else '负向（增压→CO降）',
                'slope': round(s['slope'], 4),
                'optimal_value': round(s['optimal'], 2),
            }
            for rank, s in enumerate(sensitivities, 1)
        ],
    }

    with open('results/Q3_optimization_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_out, f, ensure_ascii=False, indent=2)
    print_flush("  ✓ results/Q3_optimization_results.json")

    print_flush(f"\n⏱️  总耗时: {time.time()-t_total:.0f}s")
    print_flush("\n📁 输出文件:")
    print_flush("  results/Q3_optimization_results.json")
    print_flush("  figures/Q3_pso_convergence.png")
    print_flush("  figures/Q3_pressure_range.png")
    print_flush("  figures/Q3_pressure_adjustment.png")
    print_flush("  figures/Q3_sensitivity.png")
    print_flush("  figures/Q3_response_curves.png")


if __name__ == '__main__':
    main()
