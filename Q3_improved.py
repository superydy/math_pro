#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题3（改进版）：可靠性惩罚 + 自适应PSO
=========================================

改进点：
1. 可靠性惩罚：优化器不仅最小化CO，还要远离边界
2. 自适应PSO：PSO参数随迭代动态调整
3. 收紧范围：10%-90%分位数（原来5%-95%）
4. 稳态固定点法：co_lag1=co_lag2=...=X, 求解X=f(...,X,...)

目标函数 = CO预测值 + α(iteration) × 惩罚项
惩罚项 = Σ exp(-10 × min_dist_to_boundary)

自适应PSO：
  初期(0-30%): w=0.9, c1=2.0, c2=1.5 (探索)
  中期(30-70%): w=0.7, c1=1.5, c2=1.5 (平衡)
  后期(70-100%): w=0.4, c1=1.0, c2=2.0 (开发)

动态α：从50逐渐增加到500
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import json, os, time, warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 全局参数
PSO_PARTICLES = 40
PSO_ITERATIONS = 150
BOUND_LOW = 0.10    # 10%分位数（原来5%）
BOUND_HIGH = 0.90   # 90%分位数（原来95%）

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

def pf(msg):
    print(msg, flush=True)


def compute_xcorr(x, y, max_lag=60):
    xn = (x - np.mean(x)) / (np.std(x) * len(x) + 1e-8)
    yn = (y - np.mean(y)) / (np.std(y) + 1e-8)
    corr = fftconvolve(xn[::-1], yn, mode='full')
    center = len(xn) - 1
    lags = np.arange(-max_lag, max_lag + 1)
    cc = corr[center - max_lag: center + max_lag + 1]
    return lags, cc


# ============================================================
# Step 1: 训练问题2模型
# ============================================================
def step1_train_model():
    pf("=" * 70)
    pf("Step 1: 训练问题2预测模型")
    pf("=" * 70)

    df = pd.read_csv('data/processed_data.csv')
    abnormal = list(range(1045, 1062))
    df = df[~df.index.isin(abnormal)].reset_index(drop=True)
    pf(f"去除异常后: {len(df)} 行")

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

    df_al = df.copy()
    for var, lag in lag_results.items():
        if lag != 0:
            df_al[f'{var}_al'] = df_al[var].shift(-lag)
        else:
            df_al[f'{var}_al'] = df_al[var]

    al_cols = [f'{var}_al' for var in var_list]
    df_al = df_al.dropna(subset=al_cols + ['CO浓度']).reset_index(drop=True)

    # 特征工程
    physical_features = ['机速_al']
    for i in range(1, 19):
        physical_features.extend([f'负压_{i}_al', f'温度_{i}_al'])
    physical_features.extend(['大烟道负压_1_al', '大烟道负压_2_al',
                             '大烟道温度_1_al', '大烟道温度_2_al'])

    gradient_features = []
    for i in range(1, 18):
        df_al[f'pgrad_{i}'] = df_al[f'负压_{i+1}_al'] - df_al[f'负压_{i}_al']
        df_al[f'tgrad_{i}'] = df_al[f'温度_{i+1}_al'] - df_al[f'温度_{i}_al']
        gradient_features.extend([f'pgrad_{i}', f'tgrad_{i}'])

    stat_features = []
    pcols = [f'负压_{i}_al' for i in range(1, 19)]
    df_al['p_mean'] = df_al[pcols].mean(axis=1); stat_features.append('p_mean')
    df_al['p_mid'] = df_al[[f'负压_{i}_al' for i in range(6, 13)]].mean(axis=1); stat_features.append('p_mid')
    df_al['p_back'] = df_al[[f'负压_{i}_al' for i in range(13, 19)]].mean(axis=1); stat_features.append('p_back')
    df_al['t_back'] = df_al[[f'温度_{i}_al' for i in range(13, 19)]].mean(axis=1); stat_features.append('t_back')

    co_features = ['co_lag1', 'co_lag2', 'co_lag5', 'co_ma5', 'co_diff1']
    df_al['co_lag1'] = df_al['CO浓度'].shift(1)
    df_al['co_lag2'] = df_al['CO浓度'].shift(2)
    df_al['co_lag5'] = df_al['CO浓度'].shift(5)
    df_al['co_ma5'] = df_al['CO浓度'].rolling(5).mean()
    df_al['co_diff1'] = df_al['CO浓度'].diff(1)

    all_features = physical_features + gradient_features + stat_features + co_features
    df_final = df_al.dropna(subset=all_features + ['CO浓度']).reset_index(drop=True)

    X = df_final[all_features].values
    y = df_final['CO浓度'].values

    best_params = {
        'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 262,
        'subsample': 0.51, 'colsample_bytree': 0.99,
        'reg_alpha': 0.88, 'reg_lambda': 1.76,
        'random_state': 42, 'n_jobs': -1
    }

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = XGBRegressor(**best_params)
    model.fit(X_s, y, verbose=False)
    pf(f"模型训练完成，全量R² = {r2_score(y, model.predict(X_s)):.4f}")

    fixed_values = {}
    fixed_values['机速_al'] = float(df_final['机速_al'].median())
    for i in range(1, 19):
        fixed_values[f'温度_{i}_al'] = float(df_final[f'温度_{i}_al'].median())
    for name in ['大烟道负压_1_al', '大烟道负压_2_al', '大烟道温度_1_al', '大烟道温度_2_al']:
        fixed_values[name] = float(df_final[name].median())
    for i in range(1, 18):
        fixed_values[f'tgrad_{i}'] = float(df_final[f'tgrad_{i}'].median())
    fixed_values['t_back'] = float(df_final['t_back'].median())

    co_init = float(df_final['CO浓度'].median())
    pf(f"CO初始中位数: {co_init:.1f} mg/m³")

    feat_idx = {name: i for i, name in enumerate(all_features)}

    return model, scaler, feat_idx, all_features, fixed_values, co_init, df_final


# ============================================================
# Step 2: 构建稳态固定点 + 可靠性惩罚目标函数
# ============================================================
def step2_build_objective(model, scaler, feat_idx, all_features, fixed_values, co_init):
    pf("\n" + "=" * 70)
    pf("Step 2: 构建目标函数（稳态固定点 + 可靠性惩罚）")
    pf("=" * 70)

    n_features = len(all_features)

    def predict_steady_co(pressures):
        """稳态固定点法求解CO"""
        fv = np.zeros(n_features)

        fv[feat_idx['机速_al']] = fixed_values['机速_al']
        for i in range(1, 19):
            fv[feat_idx[f'负压_{i}_al']] = pressures[i - 1]
        for i in range(1, 19):
            fv[feat_idx[f'温度_{i}_al']] = fixed_values[f'温度_{i}_al']
        for name in ['大烟道负压_1_al', '大烟道负压_2_al', '大烟道温度_1_al', '大烟道温度_2_al']:
            fv[feat_idx[name]] = fixed_values[name]
        for i in range(1, 18):
            fv[feat_idx[f'pgrad_{i}']] = pressures[i] - pressures[i - 1]
        for i in range(1, 18):
            fv[feat_idx[f'tgrad_{i}']] = fixed_values[f'tgrad_{i}']
        fv[feat_idx['p_mean']] = np.mean(pressures)
        fv[feat_idx['p_mid']] = np.mean(pressures[5:12])
        fv[feat_idx['p_back']] = np.mean(pressures[12:18])
        fv[feat_idx['t_back']] = fixed_values['t_back']

        X_guess = co_init
        for iteration in range(50):
            fv[feat_idx['co_lag1']] = X_guess
            fv[feat_idx['co_lag2']] = X_guess
            fv[feat_idx['co_lag5']] = X_guess
            fv[feat_idx['co_ma5']] = X_guess
            fv[feat_idx['co_diff1']] = 0.0

            fv_s = scaler.transform(fv.reshape(1, -1))
            X_new = float(model.predict(fv_s)[0])

            if abs(X_new - X_guess) < 1.0:
                return X_new

            X_guess = 0.5 * X_guess + 0.5 * X_new

        return X_guess

    def compute_penalty(pressures, bounds):
        """计算可靠性惩罚"""
        penalty = 0.0
        for i in range(len(pressures)):
            lo, hi = bounds[i]
            range_size = hi - lo
            if range_size < 0.01:
                continue
            
            dist_lower = (pressures[i] - lo) / range_size
            dist_upper = (hi - pressures[i]) / range_size
            min_dist = min(dist_lower, dist_upper)
            
            # 距离越小，惩罚越大（指数衰减）
            penalty += np.exp(-10 * min_dist)
        
        return penalty

    def objective_with_penalty(pressures, bounds, alpha):
        """带惩罚的目标函数"""
        co = predict_steady_co(pressures)
        penalty = compute_penalty(pressures, bounds)
        return co + alpha * penalty, co, penalty

    # 测试
    test_p = np.array([-11.43, -11.69, -11.23, -11.32, -11.32,
                       -14.25, -14.25, -14.28, -14.28, -14.20,
                       -14.20, -14.17, -14.17, -13.99, -13.99,
                       -13.61, -13.40, -11.14])
    test_co = predict_steady_co(test_p)
    pf(f"当前配置稳态CO: {test_co:.1f} mg/m³")

    return predict_steady_co, compute_penalty, objective_with_penalty, test_co


# ============================================================
# Step 3: 确定调节范围
# ============================================================
def step3_define_ranges(df_final):
    pf("\n" + "=" * 70)
    pf(f"Step 3: 确定调节范围（{BOUND_LOW*100:.0f}%-{BOUND_HIGH*100:.0f}%分位数）")
    pf("=" * 70)

    bounds = []
    pressure_medians = []

    pf(f"\n{'风箱':>6s} {'下限':>10s} {'上限':>10s} {'中位数':>10s} {'范围宽度':>10s}")
    pf("-" * 50)

    for i in range(1, 19):
        col = f'负压_{i}_al'
        lo = float(df_final[col].quantile(BOUND_LOW))
        hi = float(df_final[col].quantile(BOUND_HIGH))
        median = float(df_final[col].median())

        bounds.append((lo, hi))
        pressure_medians.append(median)

        pf(f"  {i:2d}#  {lo:10.2f} {hi:10.2f} {median:10.2f} {hi-lo:10.2f}")

    return bounds, pressure_medians


# ============================================================
# Step 4: 改进PSO（可靠性惩罚 + 自适应参数）
# ============================================================
def step4_improved_pso(objective_with_penalty, bounds, current_co):
    pf("\n" + "=" * 70)
    pf(f"Step 4: 改进PSO优化 ({PSO_PARTICLES}粒子 × {PSO_ITERATIONS}迭代)")
    pf("=" * 70)
    pf(f"  可靠性惩罚: 指数衰减 exp(-10 × 归一化边界距离)")
    pf(f"  自适应参数: 初期探索 → 中期平衡 → 后期开发")
    pf(f"  动态α: 50 → 500")

    n_dims = 18
    np.random.seed(42)

    # 初始化
    positions = np.zeros((PSO_PARTICLES, n_dims))
    velocities = np.zeros((PSO_PARTICLES, n_dims))

    for i in range(n_dims):
        lo, hi = bounds[i]
        # 初始化在中间区域
        mid = (lo + hi) / 2
        spread = (hi - lo) * 0.3
        positions[:, i] = np.random.uniform(mid - spread, mid + spread, PSO_PARTICLES)
        velocities[:, i] = np.random.uniform(-0.2, 0.2, PSO_PARTICLES)

    # 评估
    fitness = np.full(PSO_PARTICLES, np.inf)
    co_values = np.full(PSO_PARTICLES, np.inf)

    for i in range(PSO_PARTICLES):
        alpha_init = 50
        obj, co, pen = objective_with_penalty(positions[i], bounds, alpha_init)
        fitness[i] = obj
        co_values[i] = co

    pbest_pos = positions.copy()
    pbest_val = fitness.copy()

    gbest_idx = np.argmin(fitness)
    gbest_pos = positions[gbest_idx].copy()
    gbest_val = fitness[gbest_idx]
    gbest_co = co_values[gbest_idx]

    history_co = [gbest_co]
    history_penalty = []
    history_alpha = []

    start_time = time.time()

    for iteration in range(PSO_ITERATIONS):
        # 自适应参数
        ratio = iteration / PSO_ITERATIONS
        if ratio < 0.3:
            w, c1, c2 = 0.9, 2.0, 1.5  # 探索
        elif ratio < 0.7:
            w, c1, c2 = 0.7, 1.5, 1.5  # 平衡
        else:
            w, c1, c2 = 0.4, 1.0, 2.0  # 开发

        # 动态α
        alpha = 50 + 450 * ratio  # 50 → 500

        for i in range(PSO_PARTICLES):
            obj, co, pen = objective_with_penalty(positions[i], bounds, alpha)
            fitness[i] = obj
            co_values[i] = co

            if obj < pbest_val[i]:
                pbest_val[i] = obj
                pbest_pos[i] = positions[i].copy()

            if obj < gbest_val:
                gbest_val = obj
                gbest_pos = positions[i].copy()
                gbest_co = co

        history_co.append(gbest_co)
        history_alpha.append(alpha)

        if (iteration + 1) % 10 == 0 or iteration == 0:
            elapsed = time.time() - start_time
            reduction = (current_co - gbest_co) / current_co * 100
            pen_val = compute_penalty_val(gbest_pos, bounds)
            pf(f"  迭代 {iteration+1:3d}: CO={gbest_co:.1f}, 减排={reduction:.1f}%, "
               f"α={alpha:.0f}, 惩罚={pen_val:.2f}, w={w:.2f}, 耗时={elapsed:.0f}s")

        # PSO更新
        r1 = np.random.random((PSO_PARTICLES, n_dims))
        r2 = np.random.random((PSO_PARTICLES, n_dims))
        velocities = w * velocities + c1 * r1 * (pbest_pos - positions) + c2 * r2 * (gbest_pos - positions)
        positions += velocities

        for i in range(n_dims):
            lo, hi = bounds[i]
            positions[:, i] = np.clip(positions[:, i], lo, hi)

    elapsed = time.time() - start_time
    reduction = (current_co - gbest_co) / current_co * 100
    final_penalty = compute_penalty_val(gbest_pos, bounds)

    pf(f"\n✓ 改进PSO完成: {elapsed:.0f}s")
    pf(f"  最优CO: {gbest_co:.1f} mg/m³")
    pf(f"  减排率: {reduction:.1f}%")
    pf(f"  最终惩罚: {final_penalty:.2f}")

    # 检查贴边界情况
    boundary_count = 0
    for i in range(n_dims):
        lo, hi = bounds[i]
        rng = hi - lo
        if abs(gbest_pos[i] - lo) < 0.05 * rng or abs(gbest_pos[i] - hi) < 0.05 * rng:
            boundary_count += 1

    pf(f"  贴近边界(5%内)的风箱数: {boundary_count}/18")

    return gbest_pos, gbest_co, history_co


def compute_penalty_val(pressures, bounds):
    """计算惩罚值（用于显示）"""
    penalty = 0.0
    for i in range(len(pressures)):
        lo, hi = bounds[i]
        range_size = hi - lo
        if range_size < 0.01:
            continue
        dist_lower = (pressures[i] - lo) / range_size
        dist_upper = (hi - pressures[i]) / range_size
        min_dist = min(dist_lower, dist_upper)
        penalty += np.exp(-10 * min_dist)
    return penalty


# ============================================================
# Step 5: 基础PSO对比
# ============================================================
def step5_baseline_pso(predict_steady_co, bounds, current_co):
    pf("\n" + "=" * 70)
    pf(f"Step 5: 基础PSO对比（无惩罚，无自适应）")
    pf("=" * 70)

    n_dims = 18
    np.random.seed(42)

    positions = np.zeros((PSO_PARTICLES, n_dims))
    velocities = np.zeros((PSO_PARTICLES, n_dims))

    for i in range(n_dims):
        lo, hi = bounds[i]
        positions[:, i] = np.random.uniform(lo, hi, PSO_PARTICLES)
        velocities[:, i] = np.random.uniform(-0.3, 0.3, PSO_PARTICLES)

    fitness = np.array([predict_steady_co(positions[i]) for i in range(PSO_PARTICLES)])
    pbest_pos = positions.copy()
    pbest_val = fitness.copy()
    gbest_idx = np.argmin(fitness)
    gbest_pos = positions[gbest_idx].copy()
    gbest_val = fitness[gbest_idx]

    w, c1, c2 = 0.7, 1.5, 1.5

    start_time = time.time()

    for iteration in range(PSO_ITERATIONS):
        for i in range(PSO_PARTICLES):
            co = predict_steady_co(positions[i])
            fitness[i] = co
            if co < pbest_val[i]:
                pbest_val[i] = co
                pbest_pos[i] = positions[i].copy()
            if co < gbest_val:
                gbest_val = co
                gbest_pos = positions[i].copy()

        if (iteration + 1) % 30 == 0:
            elapsed = time.time() - start_time
            reduction = (current_co - gbest_val) / current_co * 100
            pf(f"  迭代 {iteration+1:3d}: CO={gbest_val:.1f}, 减排={reduction:.1f}%, 耗时={elapsed:.0f}s")

        r1 = np.random.random((PSO_PARTICLES, n_dims))
        r2 = np.random.random((PSO_PARTICLES, n_dims))
        velocities = w * velocities + c1 * r1 * (pbest_pos - positions) + c2 * r2 * (gbest_pos - positions)
        positions += velocities
        for i in range(n_dims):
            lo, hi = bounds[i]
            positions[:, i] = np.clip(positions[:, i], lo, hi)

    elapsed = time.time() - start_time
    reduction = (current_co - gbest_val) / current_co * 100

    # 检查贴边界
    boundary_count = 0
    for i in range(n_dims):
        lo, hi = bounds[i]
        rng = hi - lo
        if abs(gbest_pos[i] - lo) < 0.05 * rng or abs(gbest_pos[i] - hi) < 0.05 * rng:
            boundary_count += 1

    pf(f"\n✓ 基础PSO完成: {elapsed:.0f}s")
    pf(f"  最优CO: {gbest_val:.1f} mg/m³")
    pf(f"  减排率: {reduction:.1f}%")
    pf(f"  贴近边界(5%内)的风箱数: {boundary_count}/18")

    return gbest_pos, gbest_val


# ============================================================
# Step 6: 对比分析
# ============================================================
def step6_analyze(current_co, improved_pos, improved_co, baseline_pos, baseline_co,
                  bounds, pressure_medians, predict_steady_co):
    pf("\n" + "=" * 70)
    pf("Step 6: 对比分析")
    pf("=" * 70)

    pf(f"\n{'方案':20s} {'CO(mg/m³)':>12s} {'减排率':>10s} {'达标':>6s}")
    pf("-" * 55)
    pf(f"{'当前工况':20s} {current_co:12.1f} {'-':>10s} {'-':>6s}")
    
    base_red = (current_co - baseline_co) / current_co * 100
    pf(f"{'基础PSO':20s} {baseline_co:12.1f} {base_red:9.1f}% {'✅' if baseline_co<=2800 else '❌':>6s}")
    
    imp_red = (current_co - improved_co) / current_co * 100
    pf(f"{'改进PSO(推荐)':20s} {improved_co:12.1f} {imp_red:9.1f}% {'✅' if improved_co<=2800 else '❌':>6s}")

    # 边界检查
    pf(f"\n{'风箱':>6s} {'当前':>10s} {'基础PSO':>10s} {'改进PSO':>10s} {'改进-当前':>10s}")
    pf("-" * 50)
    for i in range(18):
        curr = pressure_medians[i]
        base = baseline_pos[i]
        imp = improved_pos[i]
        delta = imp - curr
        pf(f"  {i+1:2d}#  {curr:10.2f} {base:10.2f} {imp:10.2f} {delta:+10.2f}")

    return improved_pos, improved_co


# ============================================================
# Step 7: 可视化
# ============================================================
def step7_visualize(current_co, improved_co, baseline_co, improved_pos, baseline_pos,
                    pressure_medians, bounds, history_co):
    pf("\n" + "=" * 70)
    pf("Step 7: 生成可视化图表")
    pf("=" * 70)

    # 图1: 三方案CO对比
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = ['Current', 'Baseline PSO', 'Improved PSO']
    values = [current_co, baseline_co, improved_co]
    colors = ['#95a5a6', '#e74c3c', '#3498db']
    bars = ax.bar(methods, values, color=colors, alpha=0.8, width=0.5)
    ax.axhline(y=2800, color='red', ls='--', lw=2, label='Standard: 2800')
    ax.set_ylabel('CO Concentration (mg/m³)', fontsize=12)
    ax.set_title(f'Q3: CO Optimization Comparison', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f'{v:.0f}', ha='center', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/Q3_co_comparison.png', dpi=200)
    plt.close()
    pf("  ✓ Q3_co_comparison.png")

    # 图2: 负压配置三方案对比
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(1, 19)
    width = 0.25
    ax.bar(x - width, pressure_medians, width, label='Current', color='#95a5a6', alpha=0.8)
    ax.bar(x, baseline_pos, width, label='Baseline PSO', color='#e74c3c', alpha=0.8)
    ax.bar(x + width, improved_pos, width, label='Improved PSO', color='#3498db', alpha=0.8)
    ax.set_xlabel('Bellows Number', fontsize=12)
    ax.set_ylabel('Negative Pressure (KPa)', fontsize=12)
    ax.set_title('Q3: Pressure Configuration Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}#' for i in x])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('figures/Q3_pressure_config.png', dpi=200)
    plt.close()
    pf("  ✓ Q3_pressure_config.png")

    # 图3: 改进PSO调整量
    fig, ax = plt.subplots(figsize=(14, 6))
    changes = [improved_pos[i] - pressure_medians[i] for i in range(18)]
    colors_c = ['#e74c3c' if c < -0.01 else '#2ecc71' if c > 0.01 else '#95a5a6' for c in changes]
    bars = ax.bar(range(1, 19), changes, color=colors_c, alpha=0.8)
    ax.axhline(y=0, color='black', lw=1)
    ax.set_xlabel('Bellows Number', fontsize=12)
    ax.set_ylabel('Pressure Change (KPa)', fontsize=12)
    ax.set_title('Q3: Improved PSO Pressure Adjustments', fontsize=14)
    ax.set_xticks(range(1, 19))
    ax.set_xticklabels([f'{i}#' for i in range(1, 19)])
    ax.grid(True, alpha=0.3, axis='y')
    for bar, ch in zip(bars, changes):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.02 if ch >= 0 else -0.05),
                f'{ch:+.2f}', ha='center', va='bottom' if ch >= 0 else 'top', fontsize=9)
    plt.tight_layout()
    plt.savefig('figures/Q3_adjustments.png', dpi=200)
    plt.close()
    pf("  ✓ Q3_adjustments.png")

    # 图4: 改进PSO收敛曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(history_co)), history_co, 'b-o', lw=2, markersize=3, label='Improved PSO')
    ax.axhline(y=current_co, color='gray', ls='--', lw=1, label=f'Current: {current_co:.0f}')
    ax.axhline(y=2800, color='red', ls='--', lw=2, label='Standard: 2800')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Steady-state CO (mg/m³)', fontsize=12)
    ax.set_title('Q3: Improved PSO Convergence', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/Q3_convergence.png', dpi=200)
    plt.close()
    pf("  ✓ Q3_convergence.png")

    # 图5: 边界距离对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, pos, name in [(axes[0], baseline_pos, 'Baseline PSO'), 
                           (axes[1], improved_pos, 'Improved PSO')]:
        dists = []
        for i in range(18):
            lo, hi = bounds[i]
            rng = hi - lo
            d = min(pos[i] - lo, hi - pos[i]) / rng
            dists.append(d)
        
        colors_d = ['#F44336' if d < 0.1 else '#FF9800' if d < 0.2 else '#4CAF50' for d in dists]
        ax.bar(range(1, 19), dists, color=colors_d, alpha=0.8)
        ax.axhline(y=0.1, color='red', ls='--', lw=1, label='Warning: 10%')
        ax.axhline(y=0.2, color='orange', ls='--', lw=1, label='Caution: 20%')
        ax.set_xlabel('Bellows Number', fontsize=11)
        ax.set_ylabel('Normalized Distance to Boundary', fontsize=11)
        ax.set_title(f'{name}: Distance to Boundary', fontsize=12)
        ax.set_xticks(range(1, 19))
        ax.set_xticklabels([f'{i}#' for i in range(1, 19)], fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 0.6)
    
    plt.tight_layout()
    plt.savefig('figures/Q3_boundary_distance.png', dpi=200)
    plt.close()
    pf("  ✓ Q3_boundary_distance.png")


# ============================================================
# Step 8: 保存结果
# ============================================================
def step8_save(current_co, improved_co, baseline_co, improved_pos, baseline_pos,
               pressure_medians, bounds):
    pf("\n" + "=" * 70)
    pf("Step 8: 保存结果")
    pf("=" * 70)

    imp_red = (current_co - improved_co) / current_co * 100
    base_red = (current_co - baseline_co) / current_co * 100

    results = {
        'current_co': float(current_co),
        'baseline_pso': {
            'co': float(baseline_co),
            'reduction_pct': float(base_red),
            'meets_standard': bool(baseline_co <= 2800)
        },
        'improved_pso': {
            'co': float(improved_co),
            'reduction_pct': float(imp_red),
            'meets_standard': bool(improved_co <= 2800),
            'method': 'reliability_penalty + adaptive_pso',
            'penalty_type': 'exponential_decay',
            'alpha_range': [50, 500],
            'bound_percentile': f'{BOUND_LOW*100:.0f}%-{BOUND_HIGH*100:.0f}%'
        },
        'optimal_pressures_improved': {f'bellows_{i+1}': float(improved_pos[i]) for i in range(18)},
        'optimal_pressures_baseline': {f'bellows_{i+1}': float(baseline_pos[i]) for i in range(18)},
        'current_pressures': {f'bellows_{i+1}': float(pressure_medians[i]) for i in range(18)},
        'pressure_ranges': {f'bellows_{i+1}': [float(bounds[i][0]), float(bounds[i][1])] for i in range(18)}
    }

    with open('results/Q3_optimization_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    pf("  ✓ results/Q3_optimization_results.json")

    # CSV
    rows = []
    for i in range(18):
        lo, hi = bounds[i]
        rng = hi - lo
        imp = improved_pos[i]
        d = min(imp - lo, hi - imp) / rng
        rows.append({
            '风箱': f'{i+1}#',
            '当前负压(KPa)': round(pressure_medians[i], 3),
            '基础PSO(KPa)': round(float(baseline_pos[i]), 3),
            '改进PSO(KPa)': round(float(imp), 3),
            '改进调整量(KPa)': round(float(imp) - pressure_medians[i], 3),
            '下限(KPa)': round(lo, 3),
            '上限(KPa)': round(hi, 3),
            '边界距离(归一化)': round(d, 3),
        })
    pd.DataFrame(rows).to_csv('results/Q3_pressure_config.csv', index=False, encoding='utf-8-sig')
    pf("  ✓ results/Q3_pressure_config.csv")


# ============================================================
# 主程序
# ============================================================
def main():
    total_start = time.time()

    pf("=" * 70)
    pf("问题3（改进版）：可靠性惩罚 + 自适应PSO")
    pf("=" * 70)

    # Step 1
    model, scaler, feat_idx, all_features, fixed_values, co_init, df_final = step1_train_model()

    # Step 2
    predict_steady_co, compute_penalty, objective_with_penalty, current_co = \
        step2_build_objective(model, scaler, feat_idx, all_features, fixed_values, co_init)

    # Step 3
    bounds, pressure_medians = step3_define_ranges(df_final)

    # Step 4: 改进PSO
    improved_pos, improved_co, history_co = \
        step4_improved_pso(objective_with_penalty, bounds, current_co)

    # Step 5: 基础PSO对比
    baseline_pos, baseline_co = step5_baseline_pso(predict_steady_co, bounds, current_co)

    # Step 6: 分析
    step6_analyze(current_co, improved_pos, improved_co, baseline_pos, baseline_co,
                  bounds, pressure_medians, predict_steady_co)

    # Step 7: 可视化
    step7_visualize(current_co, improved_co, baseline_co, improved_pos, baseline_pos,
                    pressure_medians, bounds, history_co)

    # Step 8: 保存
    step8_save(current_co, improved_co, baseline_co, improved_pos, baseline_pos,
               pressure_medians, bounds)

    elapsed = time.time() - total_start
    pf(f"\n{'='*70}")
    pf(f"问题3（改进版）完成！总耗时: {elapsed:.0f}s ({elapsed/60:.1f}分钟)")
    pf(f"{'='*70}")


if __name__ == '__main__':
    main()
