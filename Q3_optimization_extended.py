#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题3（扩展版）：CO排放最小化 — 负压 + 温度联合优化 + 扰动鲁棒性分析
=====================================================================

改进点（相比Q3 v1）：
1. 决策变量扩展至 36 个：18个风箱负压 + 18个风箱温度
2. 增加扰动鲁棒性分析（Monte Carlo，500次/噪声级别）
3. 与v1（仅优化负压）结果对比
4. 给出置信区间和达标概率 P(CO < 2800 mg/m³)
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
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 字体
# ============================================================
_font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
fm.fontManager.addfont(_font_path)
_fp  = fm.FontProperties(fname=_font_path)
_FONT = _fp.get_name()
plt.rcParams['font.family'] = _FONT
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False

EMISSION_STD = 2800   # 排放标准 mg/m³


def _fp_label(ax):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontproperties(_fp)

def p(msg):
    print(msg, flush=True)


# ============================================================
# 数据预处理（复用Q2流程）
# ============================================================

def load_and_preprocess():
    """与Q2/Q3v1完全一致的数据处理流程"""
    df = pd.read_csv('data/processed_data.csv')
    df = df[~df.index.isin(range(1045, 1062))].reset_index(drop=True)

    with open('results/Q2_complete_results.json', 'r', encoding='utf-8') as f:
        q2 = json.load(f)
    lag_results = q2['lag_results']

    var_list = ['机速']
    for i in range(1, 19):
        var_list.extend([f'负压_{i}', f'温度_{i}'])
    var_list += ['大烟道负压_1', '大烟道负压_2', '大烟道温度_1', '大烟道温度_2']

    df_al = df.copy()
    for var in var_list:
        lag = lag_results.get(var, 0)
        df_al[f'{var}_al'] = df_al[var].shift(-lag)

    al_cols = [f'{v}_al' for v in var_list]
    df_al = df_al.dropna(subset=al_cols + ['CO浓度']).reset_index(drop=True)

    phys = ['机速_al']
    for i in range(1, 19):
        phys += [f'负压_{i}_al', f'温度_{i}_al']
    phys += ['大烟道负压_1_al', '大烟道负压_2_al', '大烟道温度_1_al', '大烟道温度_2_al']

    grad = []
    for i in range(1, 18):
        df_al[f'pgrad_{i}'] = df_al[f'负压_{i+1}_al'] - df_al[f'负压_{i}_al']
        df_al[f'tgrad_{i}'] = df_al[f'温度_{i+1}_al'] - df_al[f'温度_{i}_al']
        grad += [f'pgrad_{i}', f'tgrad_{i}']

    pcols = [f'负压_{i}_al' for i in range(1, 19)]
    df_al['p_mean'] = df_al[pcols].mean(axis=1)
    df_al['p_mid']  = df_al[[f'负压_{i}_al' for i in range(6, 13)]].mean(axis=1)
    df_al['p_back'] = df_al[[f'负压_{i}_al' for i in range(13, 19)]].mean(axis=1)
    df_al['t_back'] = df_al[[f'温度_{i}_al' for i in range(13, 19)]].mean(axis=1)
    stat = ['p_mean', 'p_mid', 'p_back', 't_back']

    df_al['co_lag1']  = df_al['CO浓度'].shift(1)
    df_al['co_lag2']  = df_al['CO浓度'].shift(2)
    df_al['co_lag5']  = df_al['CO浓度'].shift(5)
    df_al['co_ma5']   = df_al['CO浓度'].rolling(5).mean()
    df_al['co_diff1'] = df_al['CO浓度'].diff(1)
    co_f = ['co_lag1', 'co_lag2', 'co_lag5', 'co_ma5', 'co_diff1']

    all_features = phys + grad + stat + co_f
    df_final = df_al.dropna(subset=all_features + ['CO浓度']).reset_index(drop=True)
    X = df_final[all_features].values
    y = df_final['CO浓度'].values
    return X, y, all_features, df_final


def train_full_model(X, y, best_params):
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    model.fit(X_s, y, verbose=False)
    return model, scaler


# ============================================================
# 可行域：压力 + 温度的历史分位数范围
# ============================================================

def compute_bounds(df_final, pct_low=5, pct_high=95):
    """返回 18个负压 + 18个温度 的 [lb, ub, median] 边界"""
    p_bounds, t_bounds = {}, {}
    for i in range(1, 19):
        pvals = df_final[f'负压_{i}_al'].values
        tvals = df_final[f'温度_{i}_al'].values
        p_bounds[f'负压_{i}'] = {
            'min': float(np.percentile(pvals, pct_low)),
            'median': float(np.median(pvals)),
            'max': float(np.percentile(pvals, pct_high)),
            'std': float(np.std(pvals)),
        }
        t_bounds[f'温度_{i}'] = {
            'min': float(np.percentile(tvals, pct_low)),
            'median': float(np.median(tvals)),
            'max': float(np.percentile(tvals, pct_high)),
            'std': float(np.std(tvals)),
        }
    return p_bounds, t_bounds


# ============================================================
# 特征向量构建（负压 + 温度均为决策变量）
# ============================================================

def build_fv(pressures, temperatures, fixed_vals, co_star, all_features):
    """
    构建84维特征向量。
    决策变量：18个负压 + 18个温度
    固定值  ：机速、大烟道4个参数、co_median（用于初始化）
    """
    feat = {}
    feat['机速_al']        = fixed_vals['机速_al']
    feat['大烟道负压_1_al'] = fixed_vals['大烟道负压_1_al']
    feat['大烟道负压_2_al'] = fixed_vals['大烟道负压_2_al']
    feat['大烟道温度_1_al'] = fixed_vals['大烟道温度_1_al']
    feat['大烟道温度_2_al'] = fixed_vals['大烟道温度_2_al']

    for i in range(1, 19):
        feat[f'负压_{i}_al'] = pressures[i - 1]
        feat[f'温度_{i}_al'] = temperatures[i - 1]

    for i in range(1, 18):
        feat[f'pgrad_{i}'] = pressures[i] - pressures[i - 1]
        feat[f'tgrad_{i}'] = temperatures[i] - temperatures[i - 1]

    feat['p_mean'] = float(np.mean(pressures))
    feat['p_mid']  = float(np.mean(pressures[5:12]))
    feat['p_back'] = float(np.mean(pressures[12:18]))
    feat['t_back'] = float(np.mean(temperatures[12:18]))

    feat['co_lag1']  = co_star
    feat['co_lag2']  = co_star
    feat['co_lag5']  = co_star
    feat['co_ma5']   = co_star
    feat['co_diff1'] = 0.0

    return np.array([feat[name] for name in all_features])


def steady_co(pressures, temperatures, fixed_vals, model, scaler, all_features,
              init_co=None, max_iter=80, tol=0.5):
    """不动点迭代求稳态CO：CO* = f(p, T, CO*)"""
    co = init_co if init_co is not None else fixed_vals['co_median']
    alpha = 0.4
    for _ in range(max_iter):
        x = build_fv(pressures, temperatures, fixed_vals, co, all_features)
        co_new = max(0.0, float(model.predict(scaler.transform(x.reshape(1, -1)))[0]))
        if abs(co_new - co) < tol:
            return co_new, True
        co = alpha * co_new + (1 - alpha) * co
    return co, False


# ============================================================
# PSO — 36维联合优化
# ============================================================

def pso_joint(p_bounds, t_bounds, fixed_vals, model, scaler, all_features,
              n_particles=40, n_iter=80):
    """
    PSO最小化稳态CO。
    决策向量 x ∈ R^36：前18维为负压，后18维为温度。
    约束：各维在历史5%~95%范围内。
    """
    n_dim = 36
    lb = np.array(
        [p_bounds[f'负压_{i}']['min'] for i in range(1, 19)] +
        [t_bounds[f'温度_{i}']['min'] for i in range(1, 19)]
    )
    ub = np.array(
        [p_bounds[f'负压_{i}']['max'] for i in range(1, 19)] +
        [t_bounds[f'温度_{i}']['max'] for i in range(1, 19)]
    )
    v_max = (ub - lb) * 0.2

    np.random.seed(42)
    pos = lb + np.random.rand(n_particles, n_dim) * (ub - lb)

    # 第0号粒子：历史中位值（热启动）
    pos[0] = np.array(
        [p_bounds[f'负压_{i}']['median'] for i in range(1, 19)] +
        [t_bounds[f'温度_{i}']['median'] for i in range(1, 19)]
    )
    # 第1号粒子：Q3 v1最优负压 + 中位温度（继承v1结果）
    try:
        with open('results/Q3_optimization_results.json', 'r', encoding='utf-8') as f:
            q3v1 = json.load(f)
        v1_p = [q3v1['pressure_ranges_and_decision'][f'负压_{i}']['optimal']
                for i in range(1, 19)]
        pos[1, :18] = v1_p
        pos[1, 18:] = [t_bounds[f'温度_{i}']['median'] for i in range(1, 19)]
    except Exception:
        pass

    vel = np.zeros((n_particles, n_dim))
    pbest_pos = pos.copy()
    pbest_val = np.full(n_particles, np.inf)
    gbest_pos = pos[0].copy()
    gbest_val = np.inf

    w_max, w_min, c1, c2 = 0.9, 0.4, 2.0, 2.0
    history = []
    t0 = time.time()

    p(f"  粒子数: {n_particles}，迭代数: {n_iter}，决策维度: {n_dim}")
    p(f"  总共约 {n_particles * n_iter} 次稳态评估\n")

    for it in range(n_iter):
        w = w_max - (w_max - w_min) * it / max(n_iter - 1, 1)

        for i in range(n_particles):
            pr = pos[i, :18]
            tp = pos[i, 18:]
            co, _ = steady_co(pr, tp, fixed_vals, model, scaler, all_features,
                               init_co=fixed_vals['co_median'])
            if co < pbest_val[i]:
                pbest_val[i] = co
                pbest_pos[i] = pos[i].copy()
            if co < gbest_val:
                gbest_val = co
                gbest_pos = pos[i].copy()

        history.append(float(gbest_val))

        if (it + 1) % 20 == 0 or it == 0:
            p(f"  迭代 {it+1:3d}/{n_iter}: 最优稳态CO = {gbest_val:.1f} mg/m³  "
              f"(耗时 {time.time()-t0:.0f}s)")

        r1 = np.random.rand(n_particles, n_dim)
        r2 = np.random.rand(n_particles, n_dim)
        vel = (w * vel
               + c1 * r1 * (pbest_pos - pos)
               + c2 * r2 * (gbest_pos - pos))
        vel = np.clip(vel, -v_max, v_max)
        pos = np.clip(pos + vel, lb, ub)

    p(f"\n  PSO完成，耗时 {time.time()-t0:.0f}s")
    opt_p = gbest_pos[:18]
    opt_t = gbest_pos[18:]
    return opt_p, opt_t, gbest_val, history


# ============================================================
# 扰动鲁棒性：Monte Carlo 分析
# ============================================================

def monte_carlo(opt_p, opt_t, fixed_vals, model, scaler, all_features,
                p_bounds, t_bounds, noise_levels=(0.10, 0.20, 0.30),
                n_samples=500):
    """
    对最优方案施加不同幅度的随机扰动，评估CO浓度分布。

    噪声模型：均匀随机扰动，幅度 = noise_level × 各变量调节范围
    clipping：扰动后仍保持在历史可行域内

    返回
    ----
    mc_results : dict，每个噪声级别对应CO分布统计量
    """
    lb_p = np.array([p_bounds[f'负压_{i}']['min'] for i in range(1, 19)])
    ub_p = np.array([p_bounds[f'负压_{i}']['max'] for i in range(1, 19)])
    lb_t = np.array([t_bounds[f'温度_{i}']['min'] for i in range(1, 19)])
    ub_t = np.array([t_bounds[f'温度_{i}']['max'] for i in range(1, 19)])

    range_p = ub_p - lb_p
    range_t = ub_t - lb_t

    mc_results = {}
    for nl in noise_levels:
        co_samples = np.zeros(n_samples)
        for s in range(n_samples):
            # 均匀扰动
            dp = np.random.uniform(-nl * range_p / 2, nl * range_p / 2)
            dt = np.random.uniform(-nl * range_t / 2, nl * range_t / 2)
            p_noisy = np.clip(opt_p + dp, lb_p, ub_p)
            t_noisy = np.clip(opt_t + dt, lb_t, ub_t)
            co, _ = steady_co(p_noisy, t_noisy, fixed_vals, model, scaler,
                               all_features, init_co=fixed_vals['co_median'])
            co_samples[s] = co

        mc_results[nl] = {
            'samples': co_samples,
            'mean':    float(np.mean(co_samples)),
            'std':     float(np.std(co_samples)),
            'p5':      float(np.percentile(co_samples, 5)),
            'p25':     float(np.percentile(co_samples, 25)),
            'p50':     float(np.percentile(co_samples, 50)),
            'p75':     float(np.percentile(co_samples, 75)),
            'p95':     float(np.percentile(co_samples, 95)),
            'prob_below_std': float(np.mean(co_samples < EMISSION_STD)),
        }
        p(f"  噪声 {int(nl*100):2d}%: CO = {mc_results[nl]['mean']:.0f} ± "
          f"{mc_results[nl]['std']:.0f} mg/m³, "
          f"P(CO<{EMISSION_STD}) = {mc_results[nl]['prob_below_std']*100:.1f}%")

    return mc_results


# ============================================================
# 可视化
# ============================================================

def visualize(opt_p, opt_t, v1_co, opt_co, history,
              mc_results, p_bounds, t_bounds,
              baseline_co, hist_p, hist_t):
    os.makedirs('figures', exist_ok=True)

    # ---- 图1: PSO收敛（对比v1）----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(history)+1), history, color='#2980B9', lw=2, label='联合优化（负压+温度）')
    ax.axhline(baseline_co, color='gray', ls=':', lw=1.2, label=f'基准CO: {baseline_co:.0f}')
    ax.axhline(v1_co,       color='#E67E22', ls='--', lw=1.5, label=f'v1仅优化负压: {v1_co:.0f}')
    ax.axhline(opt_co,      color='#27AE60', ls='--', lw=1.5, label=f'联合最优CO: {opt_co:.0f}')
    ax.axhline(EMISSION_STD, color='#C0392B', ls='-', lw=1.5, alpha=0.7, label=f'排放标准: {EMISSION_STD}')
    ax.set_xlabel('PSO迭代次数', fontproperties=_fp)
    ax.set_ylabel('最优稳态CO (mg/m³)', fontproperties=_fp)
    ax.set_title('PSO收敛过程（联合优化 vs 仅优化负压）', fontproperties=_fp)
    ax.legend(prop=_fp, fontsize=8)
    ax.grid(axis='y', ls='--', alpha=0.35)
    _fp_label(ax)
    plt.tight_layout()
    plt.savefig('figures/Q3v2_convergence.png', dpi=150)
    plt.close()
    p("  ✓ figures/Q3v2_convergence.png")

    # ---- 图2: 最优负压（v2 vs 历史中位）----
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(18)
    ax.plot(x, hist_p, 'o-', color='#2980B9', lw=1.5, ms=5, label='历史中位值')
    ax.plot(x, opt_p,  's--', color='#C0392B', lw=1.5, ms=5, label='PSO最优值（联合）')
    ax.fill_between(x,
                    [p_bounds[f'负压_{i}']['min'] for i in range(1, 19)],
                    [p_bounds[f'负压_{i}']['max'] for i in range(1, 19)],
                    alpha=0.15, color='#2980B9', label='调节范围 (5%~95%)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}#' for i in range(1, 19)], fontproperties=_fp)
    ax.set_xlabel('风箱编号', fontproperties=_fp)
    ax.set_ylabel('负压值 (Pa)', fontproperties=_fp)
    ax.set_title('联合优化：最优风箱负压决策', fontproperties=_fp)
    ax.legend(prop=_fp)
    ax.grid(axis='y', ls='--', alpha=0.35)
    _fp_label(ax)
    plt.tight_layout()
    plt.savefig('figures/Q3v2_optimal_pressure.png', dpi=150)
    plt.close()
    p("  ✓ figures/Q3v2_optimal_pressure.png")

    # ---- 图3: 最优温度（v2 vs 历史中位）----
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(18)
    ax.plot(x, hist_t, 'o-', color='#2980B9', lw=1.5, ms=5, label='历史中位值')
    ax.plot(x, opt_t,  's--', color='#C0392B', lw=1.5, ms=5, label='PSO最优值（联合）')
    ax.fill_between(x,
                    [t_bounds[f'温度_{i}']['min'] for i in range(1, 19)],
                    [t_bounds[f'温度_{i}']['max'] for i in range(1, 19)],
                    alpha=0.15, color='#E67E22', label='调节范围 (5%~95%)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}#' for i in range(1, 19)], fontproperties=_fp)
    ax.set_xlabel('风箱编号', fontproperties=_fp)
    ax.set_ylabel('温度 (°C)', fontproperties=_fp)
    ax.set_title('联合优化：最优风箱温度决策', fontproperties=_fp)
    ax.legend(prop=_fp)
    ax.grid(axis='y', ls='--', alpha=0.35)
    _fp_label(ax)
    plt.tight_layout()
    plt.savefig('figures/Q3v2_optimal_temperature.png', dpi=150)
    plt.close()
    p("  ✓ figures/Q3v2_optimal_temperature.png")

    # ---- 图4: Monte Carlo CO分布（箱线图）----
    noise_labels = [f'±{int(nl*100/2)}%范围\n(σ={int(nl*100/2)}%×range)'
                    for nl in mc_results.keys()]
    samples_list = [v['samples'] for v in mc_results.values()]
    probs = [v['prob_below_std'] for v in mc_results.values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # 左：箱线图
    bp = ax1.boxplot(samples_list, labels=noise_labels, patch_artist=True,
                     medianprops=dict(color='black', lw=2))
    colors_box = ['#AED6F1', '#F9E79F', '#F1948A']
    for patch, col in zip(bp['boxes'], colors_box):
        patch.set_facecolor(col)
        patch.set_alpha(0.8)
    ax1.axhline(EMISSION_STD, color='#C0392B', ls='--', lw=1.5,
                label=f'排放标准 {EMISSION_STD} mg/m³')
    ax1.axhline(opt_co, color='#27AE60', ls=':', lw=1.2, label=f'名义最优 {opt_co:.0f}')
    ax1.set_ylabel('稳态CO浓度 (mg/m³)', fontproperties=_fp)
    ax1.set_title('扰动鲁棒性 — CO分布箱线图', fontproperties=_fp)
    ax1.legend(prop=_fp, fontsize=8)
    ax1.grid(axis='y', ls='--', alpha=0.35)
    _fp_label(ax1)

    # 右：达标概率柱状图
    nl_pct = [int(nl * 100) for nl in mc_results.keys()]
    bar_colors = ['#27AE60' if pr >= 0.8 else '#E67E22' if pr >= 0.5 else '#C0392B'
                  for pr in probs]
    bars = ax2.bar(range(len(probs)), [p_ * 100 for p_ in probs],
                   color=bar_colors, alpha=0.85, width=0.5)
    ax2.axhline(80, color='#27AE60', ls='--', lw=1.2, alpha=0.7, label='80%达标线')
    ax2.axhline(50, color='#E67E22', ls='--', lw=1.2, alpha=0.7, label='50%达标线')
    ax2.set_xticks(range(len(probs)))
    ax2.set_xticklabels([f'噪声{nl}%' for nl in nl_pct], fontproperties=_fp)
    ax2.set_ylabel(f'P(CO < {EMISSION_STD}) (%)', fontproperties=_fp)
    ax2.set_title('不同扰动水平下的排放达标概率', fontproperties=_fp)
    ax2.set_ylim(0, 105)
    ax2.legend(prop=_fp, fontsize=8)
    ax2.grid(axis='y', ls='--', alpha=0.35)
    for bar, val in zip(bars, probs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val*100:.1f}%', ha='center', fontproperties=_fp, fontsize=9)
    _fp_label(ax2)

    plt.tight_layout()
    plt.savefig('figures/Q3v2_robustness.png', dpi=150)
    plt.close()
    p("  ✓ figures/Q3v2_robustness.png")

    # ---- 图5: 优化效果汇总对比 ----
    fig, ax = plt.subplots(figsize=(10, 5))
    categories = ['历史中位\n（基准）', '仅优化\n负压(v1)', '联合优化\n负压+温度(v2)']
    values = [baseline_co, v1_co, opt_co]
    bar_cols = ['#BDC3C7', '#E67E22', '#27AE60']
    bars = ax.bar(categories, values, color=bar_cols, alpha=0.85, width=0.5)
    ax.axhline(EMISSION_STD, color='#C0392B', ls='--', lw=2,
               label=f'排放标准 {EMISSION_STD} mg/m³')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f'{val:.0f}', ha='center', va='bottom', fontproperties=_fp, fontsize=11)
    ax.set_ylabel('稳态CO浓度 (mg/m³)', fontproperties=_fp)
    ax.set_title('各优化方案CO浓度对比', fontproperties=_fp)
    ax.legend(prop=_fp)
    ax.grid(axis='y', ls='--', alpha=0.35)
    _fp_label(ax)
    plt.tight_layout()
    plt.savefig('figures/Q3v2_comparison.png', dpi=150)
    plt.close()
    p("  ✓ figures/Q3v2_comparison.png")


# ============================================================
# 主程序
# ============================================================

def main():
    p("=" * 70)
    p("问题3（扩展版）：负压 + 温度联合优化 + 扰动鲁棒性分析")
    p("=" * 70)
    t0 = time.time()

    # ---- Steps 1-3: 数据 ----
    p("\n[Step 1-3] 数据加载与特征工程...")
    X, y, all_features, df_final = load_and_preprocess()
    p(f"  样本数: {len(y)}，特征数: {len(all_features)}")

    # ---- Step 4: 模型 ----
    p("\n[Step 4] 加载Q2最优参数训练全量模型...")
    with open('results/Q2_complete_results.json', 'r', encoding='utf-8') as f:
        q2 = json.load(f)
    best_params = q2['model_info']['best_params']
    best_params['max_depth']    = int(best_params['max_depth'])
    best_params['n_estimators'] = int(best_params['n_estimators'])
    model, scaler = train_full_model(X, y, best_params)
    p(f"  全量模型 R² = {r2_score(y, model.predict(scaler.transform(X))):.4f}")

    # ---- Step 5: 可行域 ----
    p("\n[Step 5] 计算36个决策变量的调节范围...")
    p_bounds, t_bounds = compute_bounds(df_final)

    p(f"\n  {'变量':12s} {'下限(5%)':>10s} {'中位值':>10s} {'上限(95%)':>10s} {'范围':>8s} {'std':>8s}")
    p("  " + "-" * 58)
    for i in range(1, 19):
        b = p_bounds[f'负压_{i}']
        p(f"  负压_{i:2d}     {b['min']:10.1f} {b['median']:10.1f} {b['max']:10.1f} "
          f"{b['max']-b['min']:8.1f} {b['std']:8.2f}")
    p("")
    for i in range(1, 19):
        b = t_bounds[f'温度_{i}']
        p(f"  温度_{i:2d}     {b['min']:10.1f} {b['median']:10.1f} {b['max']:10.1f} "
          f"{b['max']-b['min']:8.1f} {b['std']:8.1f}")

    # 固定特征（只有机速 + 大烟道4个参数）
    fixed_vars = ['机速_al', '大烟道负压_1_al', '大烟道负压_2_al',
                  '大烟道温度_1_al', '大烟道温度_2_al']
    fixed_vals = {v: float(df_final[v].median()) for v in fixed_vars}
    fixed_vals['co_median'] = float(np.median(y))

    # 历史中位值向量（用于对比）
    hist_p = np.array([p_bounds[f'负压_{i}']['median'] for i in range(1, 19)])
    hist_t = np.array([t_bounds[f'温度_{i}']['median'] for i in range(1, 19)])

    # 基准CO（历史中位压力+温度下的稳态）
    baseline_co, _ = steady_co(hist_p, hist_t, fixed_vals, model, scaler, all_features)
    p(f"\n  历史中位设定的稳态CO: {baseline_co:.1f} mg/m³")

    # v1 结果（仅优化负压）
    try:
        with open('results/Q3_optimization_results.json', 'r', encoding='utf-8') as f:
            q3v1 = json.load(f)
        v1_co = q3v1['optimization_summary']['optimal_co_mg_m3']
    except Exception:
        v1_co = 2909.0
    p(f"  Q3 v1（仅优化负压）最优CO: {v1_co:.1f} mg/m³")

    # ---- Step 6: PSO联合优化 ----
    p("\n[Step 6] PSO联合优化（36维）...")
    opt_p, opt_t, opt_co, history = pso_joint(
        p_bounds, t_bounds, fixed_vals, model, scaler, all_features,
        n_particles=40, n_iter=80
    )

    reduction_vs_baseline = baseline_co - opt_co
    reduction_vs_v1       = v1_co - opt_co
    pct_vs_baseline       = reduction_vs_baseline / baseline_co * 100
    reach_std             = opt_co < EMISSION_STD

    p("\n" + "=" * 70)
    p("优化结果汇总")
    p("=" * 70)
    p(f"  基准CO（历史中位）  : {baseline_co:.1f} mg/m³")
    p(f"  v1 CO（仅优化负压） : {v1_co:.1f} mg/m³")
    p(f"  v2 CO（联合优化）   : {opt_co:.1f} mg/m³")
    p(f"  相比基准降低        : {reduction_vs_baseline:.1f} mg/m³  ({pct_vs_baseline:.1f}%)")
    p(f"  相比v1进一步降低    : {reduction_vs_v1:.1f} mg/m³")
    p(f"  是否低于排放标准({EMISSION_STD}): {'✓ 是' if reach_std else '✗ 否'}")

    p(f"\n  最优决策（完整列表）:")
    p(f"  {'风箱':8s} {'负压历史中位':>12s} {'负压最优':>10s} {'负压调整':>10s} "
      f"{'温度历史中位':>12s} {'温度最优':>10s} {'温度调整':>10s}")
    p("  " + "-" * 76)
    for i in range(1, 19):
        pb = p_bounds[f'负压_{i}']
        tb = t_bounds[f'温度_{i}']
        p_opt = opt_p[i-1]
        t_opt = opt_t[i-1]
        p(f"  {i:2d}#     {pb['median']:12.1f} {p_opt:10.1f} {p_opt-pb['median']:+10.1f} "
          f"{tb['median']:12.1f} {t_opt:10.1f} {t_opt-tb['median']:+10.1f}")

    # ---- Step 7: Monte Carlo鲁棒性 ----
    p("\n[Step 7] Monte Carlo 扰动鲁棒性分析（500次 × 3种噪声级别）...")
    mc_results = monte_carlo(
        opt_p, opt_t, fixed_vals, model, scaler, all_features,
        p_bounds, t_bounds,
        noise_levels=(0.10, 0.20, 0.30),
        n_samples=500
    )

    p(f"\n  鲁棒性汇总（名义最优CO = {opt_co:.1f} mg/m³）:")
    p(f"  {'扰动幅度':10s} {'均值CO':>10s} {'标准差':>8s} {'P5':>8s} {'P95':>8s} "
      f"{'P(CO<{EMISSION_STD})':>14s}")
    p("  " + "-" * 60)
    for nl, res in mc_results.items():
        p(f"  ±{int(nl*100/2):2d}%范围   {res['mean']:10.1f} {res['std']:8.1f} "
          f"{res['p5']:8.1f} {res['p95']:8.1f} {res['prob_below_std']*100:14.1f}%")

    # ---- Step 8: 可视化 ----
    p("\n[Step 8] 生成可视化图表...")
    visualize(opt_p, opt_t, v1_co, opt_co, history,
              mc_results, p_bounds, t_bounds,
              baseline_co, hist_p, hist_t)

    # ---- 保存结果 ----
    os.makedirs('results', exist_ok=True)
    out = {
        'comparison': {
            'baseline_co':  round(baseline_co, 2),
            'v1_co':        round(v1_co, 2),
            'v2_co':        round(opt_co, 2),
            'reduction_vs_baseline': round(reduction_vs_baseline, 2),
            'reduction_pct': round(pct_vs_baseline, 2),
            'reach_emission_std': bool(reach_std),
        },
        'optimal_decision': {
            f'负压_{i}': {
                'optimal': round(float(opt_p[i-1]), 2),
                'median':  round(p_bounds[f'负压_{i}']['median'], 2),
                'adjust':  round(float(opt_p[i-1]) - p_bounds[f'负压_{i}']['median'], 2),
                'lb': round(p_bounds[f'负压_{i}']['min'], 2),
                'ub': round(p_bounds[f'负压_{i}']['max'], 2),
            }
            for i in range(1, 19)
        } | {
            f'温度_{i}': {
                'optimal': round(float(opt_t[i-1]), 2),
                'median':  round(t_bounds[f'温度_{i}']['median'], 2),
                'adjust':  round(float(opt_t[i-1]) - t_bounds[f'温度_{i}']['median'], 2),
                'lb': round(t_bounds[f'温度_{i}']['min'], 2),
                'ub': round(t_bounds[f'温度_{i}']['max'], 2),
            }
            for i in range(1, 19)
        },
        'robustness': {
            f'noise_{int(nl*100)}pct': {
                'mean_co': round(res['mean'], 2),
                'std_co':  round(res['std'], 2),
                'p5':      round(res['p5'], 2),
                'p95':     round(res['p95'], 2),
                'prob_below_emission_std': round(res['prob_below_std'], 4),
            }
            for nl, res in mc_results.items()
        },
        'pso_history': history,
    }

    with open('results/Q3_v2_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    p("  ✓ results/Q3_v2_results.json")

    p(f"\n⏱️  总耗时: {time.time()-t0:.0f}s")
    p("\n📁 输出文件:")
    for fn in ['results/Q3_v2_results.json',
               'figures/Q3v2_convergence.png', 'figures/Q3v2_optimal_pressure.png',
               'figures/Q3v2_optimal_temperature.png', 'figures/Q3v2_robustness.png',
               'figures/Q3v2_comparison.png']:
        p(f"  {fn}")


if __name__ == '__main__':
    main()
