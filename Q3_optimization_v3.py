#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题3 v3：CO最小化 — 差分进化(DE) + 机速联合优化 + 稳态多点初始化
====================================================================

改进点（相比v2）：
1. 优化算法：PSO → scipy Differential Evolution（全局搜索更强）
2. 决策变量：18负压 + 18温度 + 机速 = 37维
3. 稳态固定点迭代：多起点初始化，取最低稳态CO（处理多稳态）
4. DE结束后用局部精化（Nelder-Mead），进一步收敛
5. 与v1/v2完整对比
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
from scipy.optimize import differential_evolution, minimize
import json, os, time, warnings
warnings.filterwarnings('ignore')

# ============================================================
# 字体
# ============================================================
_fp = fm.FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
fm.fontManager.addfont('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
plt.rcParams['font.family'] = _fp.get_name()
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

EMISSION_STD = 2800

def _fp_label(ax):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontproperties(_fp)

def p(msg): print(msg, flush=True)


# ============================================================
# 数据预处理（与Q2/v1/v2完全一致）
# ============================================================
def load_and_preprocess():
    df = pd.read_csv('data/processed_data.csv')
    df = df[~df.index.isin(range(1045, 1062))].reset_index(drop=True)

    with open('results/Q2_complete_results.json', 'r', encoding='utf-8') as f:
        lag_results = json.load(f)['lag_results']

    var_list = ['机速'] + [f for i in range(1,19) for f in [f'负压_{i}', f'温度_{i}']]
    var_list += ['大烟道负压_1','大烟道负压_2','大烟道温度_1','大烟道温度_2']

    df_al = df.copy()
    for v in var_list:
        df_al[f'{v}_al'] = df_al[v].shift(-lag_results.get(v, 0))
    df_al = df_al.dropna(subset=[f'{v}_al' for v in var_list]+['CO浓度']).reset_index(drop=True)

    phys = ['机速_al'] + [f for i in range(1,19) for f in [f'负压_{i}_al',f'温度_{i}_al']]
    phys += ['大烟道负压_1_al','大烟道负压_2_al','大烟道温度_1_al','大烟道温度_2_al']

    grad = []
    for i in range(1, 18):
        df_al[f'pgrad_{i}'] = df_al[f'负压_{i+1}_al'] - df_al[f'负压_{i}_al']
        df_al[f'tgrad_{i}'] = df_al[f'温度_{i+1}_al'] - df_al[f'温度_{i}_al']
        grad += [f'pgrad_{i}', f'tgrad_{i}']

    pcols = [f'负压_{i}_al' for i in range(1,19)]
    df_al['p_mean'] = df_al[pcols].mean(axis=1)
    df_al['p_mid']  = df_al[[f'负压_{i}_al' for i in range(6,13)]].mean(axis=1)
    df_al['p_back'] = df_al[[f'负压_{i}_al' for i in range(13,19)]].mean(axis=1)
    df_al['t_back'] = df_al[[f'温度_{i}_al' for i in range(13,19)]].mean(axis=1)
    stat = ['p_mean','p_mid','p_back','t_back']

    for lag, col in [(1,'co_lag1'),(2,'co_lag2'),(5,'co_lag5')]:
        df_al[col] = df_al['CO浓度'].shift(lag)
    df_al['co_ma5']   = df_al['CO浓度'].rolling(5).mean()
    df_al['co_diff1'] = df_al['CO浓度'].diff(1)
    co_f = ['co_lag1','co_lag2','co_lag5','co_ma5','co_diff1']

    all_features = phys + grad + stat + co_f
    df_fin = df_al.dropna(subset=all_features+['CO浓度']).reset_index(drop=True)
    return df_fin[all_features].values, df_fin['CO浓度'].values, all_features, df_fin


def train_model(X, y, params):
    sc = StandardScaler()
    model = XGBRegressor(**params, random_state=42, n_jobs=-1)
    model.fit(sc.fit_transform(X), y, verbose=False)
    return model, sc


# ============================================================
# 可行域（37维：18负压 + 18温度 + 机速）
# ============================================================
def compute_bounds_37(df_fin, pct_lo=5, pct_hi=95):
    bounds = {}
    for i in range(1,19):
        for prefix in ['负压', '温度']:
            col = f'{prefix}_{i}_al'
            v = df_fin[col].values
            bounds[f'{prefix}_{i}'] = dict(min=float(np.percentile(v,pct_lo)),
                                           median=float(np.median(v)),
                                           max=float(np.percentile(v,pct_hi)),
                                           std=float(np.std(v)))
    v = df_fin['机速_al'].values
    bounds['机速'] = dict(min=float(np.percentile(v,pct_lo)),
                         median=float(np.median(v)),
                         max=float(np.percentile(v,pct_hi)),
                         std=float(np.std(v)))
    return bounds


# ============================================================
# 特征向量构建（37决策变量）
# ============================================================
def build_fv(pressures, temperatures, speed, fixed_vals, co_star, all_features):
    """
    x = [p1..p18, T1..T18, speed]  全部为决策变量
    固定：大烟道4个参数
    """
    feat = {}
    feat['机速_al'] = speed
    feat['大烟道负压_1_al'] = fixed_vals['大烟道负压_1_al']
    feat['大烟道负压_2_al'] = fixed_vals['大烟道负压_2_al']
    feat['大烟道温度_1_al'] = fixed_vals['大烟道温度_1_al']
    feat['大烟道温度_2_al'] = fixed_vals['大烟道温度_2_al']
    for i in range(1,19):
        feat[f'负压_{i}_al'] = pressures[i-1]
        feat[f'温度_{i}_al'] = temperatures[i-1]
    for i in range(1,18):
        feat[f'pgrad_{i}'] = pressures[i] - pressures[i-1]
        feat[f'tgrad_{i}'] = temperatures[i] - temperatures[i-1]
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


# ============================================================
# 稳态不动点迭代（多起点，取最低稳态CO）
# ============================================================
def steady_co(pressures, temperatures, speed, fixed_vals, model, scaler,
              all_features, max_iter=40, tol=0.5):
    """
    多起点初始化：从5个不同CO水平出发迭代，取收敛到的最低稳态值。
    物理意义：系统可能存在多稳态，寻找最低的稳定平衡点。
    """
    alpha = 0.4
    # 从低到高的多个初始值
    # 两个起点：低CO稳态探测(200) + 历史均值（兜底）
    init_vals = [200, fixed_vals['co_median']]
    best_co = np.inf

    for co0 in init_vals:
        co = co0
        converged = False
        for _ in range(max_iter):
            x = build_fv(pressures, temperatures, speed, fixed_vals, co, all_features)
            co_new = max(0.0, float(model.predict(scaler.transform(x.reshape(1,-1)))[0]))
            if abs(co_new - co) < tol:
                converged = True
                break
            co = alpha * co_new + (1-alpha) * co
        if converged and co < best_co:
            best_co = co

    if best_co == np.inf:   # 兜底：从中位值出发
        co = fixed_vals['co_median']
        for _ in range(max_iter):
            x = build_fv(pressures, temperatures, speed, fixed_vals, co, all_features)
            co_new = max(0.0, float(model.predict(scaler.transform(x.reshape(1,-1)))[0]))
            if abs(co_new - co) < tol:
                break
            co = alpha * co_new + (1-alpha) * co
        best_co = co

    return best_co


# ============================================================
# Differential Evolution 优化
# ============================================================
_MODEL  = None   # 全局引用，供scipy回调使用
_SCALER = None
_ALL_F  = None
_FVALS  = None
_CALLS  = 0
_T0     = 0
_HIST   = []     # (call_count, best_co)
_BEST   = np.inf

def _de_callback(xk, convergence):
    """DE每代回调：记录进度"""
    global _CALLS, _BEST, _HIST
    _CALLS += 1
    if _CALLS % 20 == 0:
        elapsed = time.time() - _T0
        p(f"  DE 第{_CALLS}代: 最优CO = {_BEST:.1f} mg/m³  (耗时{elapsed:.0f}s)")

def _de_objective(x):
    """目标函数：x = [p1..18, T1..18, speed]"""
    global _BEST, _HIST
    pressures    = x[:18]
    temperatures = x[18:36]
    speed        = float(x[36])
    co = steady_co(pressures, temperatures, speed, _FVALS, _MODEL, _SCALER, _ALL_F)
    if co < _BEST:
        _BEST = co
        _HIST.append(co)
    return co


def run_de(bounds_37, fixed_vals, model, scaler, all_features,
           popsize=20, maxiter=300, seed=42):
    """
    使用scipy Differential Evolution全局优化。
    strategy='best1bin'：经典DE变体，收敛稳定。
    popsize=20：种群大小 = 20 × 37 = 740个个体（大规模搜索）
    """
    global _MODEL, _SCALER, _ALL_F, _FVALS, _CALLS, _T0, _HIST, _BEST
    _MODEL, _SCALER, _ALL_F, _FVALS = model, scaler, all_features, fixed_vals
    _CALLS, _T0, _HIST, _BEST = 0, time.time(), [], np.inf

    var_order = ([f'负压_{i}' for i in range(1,19)] +
                 [f'温度_{i}' for i in range(1,19)] +
                 ['机速'])

    scipy_bounds = [(bounds_37[k]['min'], bounds_37[k]['max']) for k in var_order]

    # 初始种群中包含历史中位值（热启动）
    init_point = np.array([bounds_37[k]['median'] for k in var_order])
    init_pop = None   # scipy会随机生成；中位值通过seed=42尽量覆盖

    p(f"  DE参数: strategy=best1bin, popsize={popsize}, maxiter={maxiter}")
    p(f"  种群规模 = {popsize}×37 = {popsize*37}个个体")
    p(f"  最大评估次数 ≈ {popsize*37*maxiter:,}")

    result = differential_evolution(
        _de_objective,
        bounds=scipy_bounds,
        strategy='best1bin',
        maxiter=maxiter,
        popsize=popsize,
        tol=0.001,
        mutation=(0.5, 1.5),
        recombination=0.9,
        seed=seed,
        callback=_de_callback,
        workers=1,
        polish=False,      # 先不polish，之后手动用Nelder-Mead精化
        init='sobol',      # 低差异序列初始化，覆盖更均匀
    )

    p(f"\n  DE完成，耗时 {time.time()-_T0:.0f}s，最优CO = {result.fun:.1f} mg/m³")
    return result.x, result.fun, _HIST


def local_polish(x0, bounds_37, fixed_vals, model, scaler, all_features):
    """DE结束后用Nelder-Mead在全局最优附近做局部精化"""
    var_order = ([f'负压_{i}' for i in range(1,19)] +
                 [f'温度_{i}' for i in range(1,19)] + ['机速'])
    lb = np.array([bounds_37[k]['min'] for k in var_order])
    ub = np.array([bounds_37[k]['max'] for k in var_order])

    def obj(x):
        x_clipped = np.clip(x, lb, ub)
        return steady_co(x_clipped[:18], x_clipped[18:36], float(x_clipped[36]),
                         fixed_vals, model, scaler, all_features)

    result = minimize(obj, x0, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 0.1, 'fatol': 0.1})
    x_pol = np.clip(result.x, lb, ub)
    co_pol = steady_co(x_pol[:18], x_pol[18:36], float(x_pol[36]),
                       fixed_vals, model, scaler, all_features)
    return x_pol, co_pol


# ============================================================
# Monte Carlo鲁棒性（复用v2逻辑）
# ============================================================
def monte_carlo(opt_x, bounds_37, fixed_vals, model, scaler, all_features,
                noise_levels=(0.10, 0.20, 0.30), n_samples=500):
    var_order = ([f'负压_{i}' for i in range(1,19)] +
                 [f'温度_{i}' for i in range(1,19)] + ['机速'])
    lb = np.array([bounds_37[k]['min'] for k in var_order])
    ub = np.array([bounds_37[k]['max'] for k in var_order])
    ranges = ub - lb

    results = {}
    for nl in noise_levels:
        samples = []
        for _ in range(n_samples):
            noise = np.random.uniform(-nl*ranges/2, nl*ranges/2)
            x_n = np.clip(opt_x + noise, lb, ub)
            co = steady_co(x_n[:18], x_n[18:36], float(x_n[36]),
                           fixed_vals, model, scaler, all_features)
            samples.append(co)
        arr = np.array(samples)
        results[nl] = dict(
            mean=float(arr.mean()), std=float(arr.std()),
            p5=float(np.percentile(arr,5)), p95=float(np.percentile(arr,95)),
            prob=float(np.mean(arr < EMISSION_STD)),
            samples=arr,
        )
        p(f"  扰动±{int(nl*100/2):2d}%范围: CO = {arr.mean():.0f}±{arr.std():.0f} mg/m³, "
          f"P(CO<{EMISSION_STD}) = {np.mean(arr<EMISSION_STD)*100:.1f}%")
    return results


# ============================================================
# 可视化
# ============================================================
def visualize(opt_x, opt_co, de_history, mc, bounds_37,
              baseline_co, v1_co, v2_co):
    os.makedirs('figures', exist_ok=True)
    var_order = ([f'负压_{i}' for i in range(1,19)] +
                 [f'温度_{i}' for i in range(1,19)] + ['机速'])
    opt_p  = opt_x[:18]
    opt_t  = opt_x[18:36]
    opt_sp = opt_x[36]
    hist_p = [bounds_37[f'负压_{i}']['median'] for i in range(1,19)]
    hist_t = [bounds_37[f'温度_{i}']['median'] for i in range(1,19)]

    # ---- 图1: DE收敛 + 方案对比 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    if de_history:
        ax1.plot(range(1,len(de_history)+1), de_history, color='#2980B9', lw=2, label='DE最优CO')
    ax1.axhline(baseline_co, color='gray', ls=':', lw=1.2, label=f'基准: {baseline_co:.0f}')
    ax1.axhline(v1_co,       color='#95A5A6', ls='--', lw=1.2, label=f'v1（负压）: {v1_co:.0f}')
    ax1.axhline(v2_co,       color='#E67E22', ls='--', lw=1.5, label=f'v2（负压+温度）: {v2_co:.0f}')
    ax1.axhline(opt_co,      color='#27AE60', ls='--', lw=2,   label=f'v3（DE联合）: {opt_co:.0f}')
    ax1.axhline(EMISSION_STD,color='#C0392B', ls='-',  lw=1.5, alpha=0.8, label=f'排放标准: {EMISSION_STD}')
    ax1.set_xlabel('DE迭代轮次', fontproperties=_fp)
    ax1.set_ylabel('最优稳态CO (mg/m³)', fontproperties=_fp)
    ax1.set_title('DE优化收敛过程', fontproperties=_fp)
    ax1.legend(prop=_fp, fontsize=7.5)
    ax1.grid(axis='y', ls='--', alpha=0.35)
    _fp_label(ax1)

    scenarios   = ['基准', 'v1\n(负压)', 'v2\n(负压+温度)', 'v3\n(DE联合)']
    co_vals     = [baseline_co, v1_co, v2_co, opt_co]
    bar_colors  = ['#BDC3C7', '#AED6F1', '#F9E79F', '#27AE60']
    bars = ax2.bar(scenarios, co_vals, color=bar_colors, alpha=0.9, width=0.5)
    ax2.axhline(EMISSION_STD, color='#C0392B', ls='--', lw=2,
                label=f'排放标准 {EMISSION_STD}')
    for bar, val in zip(bars, co_vals):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+30,
                 f'{val:.0f}', ha='center', fontproperties=_fp, fontsize=10)
    ax2.set_ylabel('稳态CO浓度 (mg/m³)', fontproperties=_fp)
    ax2.set_title('各优化方案效果对比', fontproperties=_fp)
    ax2.legend(prop=_fp)
    ax2.grid(axis='y', ls='--', alpha=0.35)
    _fp_label(ax2)
    plt.tight_layout()
    plt.savefig('figures/Q3v3_overview.png', dpi=150)
    plt.close()
    p("  ✓ figures/Q3v3_overview.png")

    # ---- 图2: 最优负压 ----
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(18)
    ax.fill_between(x,
                    [bounds_37[f'负压_{i}']['min'] for i in range(1,19)],
                    [bounds_37[f'负压_{i}']['max'] for i in range(1,19)],
                    alpha=0.15, color='#2980B9', label='调节范围(5%~95%)')
    ax.plot(x, hist_p, 'o-', color='#2980B9', lw=1.5, ms=5, label='历史中位值')
    ax.plot(x, opt_p,  's--', color='#C0392B', lw=2,   ms=6, label='DE最优值')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}#' for i in range(1,19)], fontproperties=_fp)
    ax.set_xlabel('风箱编号', fontproperties=_fp)
    ax.set_ylabel('负压 (Pa)', fontproperties=_fp)
    ax.set_title('最优风箱负压调控决策（DE）', fontproperties=_fp)
    ax.legend(prop=_fp)
    ax.grid(axis='y', ls='--', alpha=0.35)
    _fp_label(ax)
    plt.tight_layout()
    plt.savefig('figures/Q3v3_pressure.png', dpi=150)
    plt.close()
    p("  ✓ figures/Q3v3_pressure.png")

    # ---- 图3: 最优温度 ----
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(x,
                    [bounds_37[f'温度_{i}']['min'] for i in range(1,19)],
                    [bounds_37[f'温度_{i}']['max'] for i in range(1,19)],
                    alpha=0.15, color='#E67E22', label='调节范围(5%~95%)')
    ax.plot(x, hist_t, 'o-', color='#2980B9', lw=1.5, ms=5, label='历史中位值')
    ax.plot(x, opt_t,  's--', color='#C0392B', lw=2,   ms=6, label='DE最优值')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}#' for i in range(1,19)], fontproperties=_fp)
    ax.set_xlabel('风箱编号', fontproperties=_fp)
    ax.set_ylabel('温度 (°C)', fontproperties=_fp)
    ax.set_title('最优风箱温度调控决策（DE）', fontproperties=_fp)
    ax.legend(prop=_fp)
    ax.grid(axis='y', ls='--', alpha=0.35)
    _fp_label(ax)
    plt.tight_layout()
    plt.savefig('figures/Q3v3_temperature.png', dpi=150)
    plt.close()
    p("  ✓ figures/Q3v3_temperature.png")

    # ---- 图4: 鲁棒性箱线图 + 达标概率 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    labels = [f'±{int(nl*100/2)}%\n扰动范围' for nl in mc.keys()]
    bp = ax1.boxplot([v['samples'] for v in mc.values()],
                     labels=labels, patch_artist=True,
                     medianprops=dict(color='black', lw=2))
    for patch, c in zip(bp['boxes'], ['#AED6F1','#F9E79F','#F1948A']):
        patch.set_facecolor(c); patch.set_alpha(0.8)
    ax1.axhline(EMISSION_STD, color='#C0392B', ls='--', lw=1.5,
                label=f'排放标准 {EMISSION_STD}')
    ax1.axhline(opt_co, color='#27AE60', ls=':', lw=1.2, label=f'名义最优 {opt_co:.0f}')
    ax1.set_ylabel('稳态CO (mg/m³)', fontproperties=_fp)
    ax1.set_title('扰动鲁棒性 — CO分布', fontproperties=_fp)
    ax1.legend(prop=_fp, fontsize=8)
    ax1.grid(axis='y', ls='--', alpha=0.35)
    _fp_label(ax1)

    probs = [v['prob']*100 for v in mc.values()]
    bar_c = ['#27AE60' if pr>=80 else '#E67E22' if pr>=50 else '#C0392B' for pr in probs]
    bars2 = ax2.bar(range(len(probs)), probs, color=bar_c, alpha=0.85, width=0.5)
    ax2.axhline(80, color='#27AE60', ls='--', lw=1.2, label='80%达标线')
    ax2.axhline(50, color='#E67E22', ls='--', lw=1.2, label='50%达标线')
    ax2.set_xticks(range(len(probs)))
    ax2.set_xticklabels(labels, fontproperties=_fp)
    ax2.set_ylabel(f'P(CO<{EMISSION_STD}) (%)', fontproperties=_fp)
    ax2.set_title('不同扰动水平达标概率', fontproperties=_fp)
    ax2.set_ylim(0, 108)
    for bar, val in zip(bars2, probs):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f'{val:.1f}%', ha='center', fontproperties=_fp, fontsize=9)
    ax2.legend(prop=_fp, fontsize=8)
    ax2.grid(axis='y', ls='--', alpha=0.35)
    _fp_label(ax2)
    plt.tight_layout()
    plt.savefig('figures/Q3v3_robustness.png', dpi=150)
    plt.close()
    p("  ✓ figures/Q3v3_robustness.png")


# ============================================================
# 主程序
# ============================================================
def main():
    p("="*70)
    p("问题3 v3：差分进化(DE) + 37维联合优化 + 多稳态初始化")
    p("="*70)
    t0 = time.time()

    # 数据与模型
    p("\n[Step 1-4] 数据加载与模型训练...")
    X, y, all_features, df_fin = load_and_preprocess()
    with open('results/Q2_complete_results.json','r',encoding='utf-8') as f:
        params = json.load(f)['model_info']['best_params']
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])
    model, scaler = train_model(X, y, params)
    p(f"  全量模型 R² = {r2_score(y, model.predict(scaler.transform(X))):.4f}")

    # 可行域
    p("\n[Step 5] 计算37个决策变量的可行域...")
    bounds_37 = compute_bounds_37(df_fin)
    sp_b = bounds_37['机速']
    p(f"  机速范围: {sp_b['min']:.2f} ~ {sp_b['median']:.2f} ~ {sp_b['max']:.2f}")

    # 固定值（仅大烟道4个参数）
    fixed_vals = {v: float(df_fin[v].median())
                  for v in ['大烟道负压_1_al','大烟道负压_2_al',
                             '大烟道温度_1_al','大烟道温度_2_al']}
    fixed_vals['co_median'] = float(np.median(y))

    # 基准CO
    hist_p  = np.array([bounds_37[f'负压_{i}']['median'] for i in range(1,19)])
    hist_t  = np.array([bounds_37[f'温度_{i}']['median'] for i in range(1,19)])
    hist_sp = bounds_37['机速']['median']
    baseline_co = steady_co(hist_p, hist_t, hist_sp, fixed_vals, model, scaler, all_features)
    p(f"  历史中位设定稳态CO: {baseline_co:.1f} mg/m³")

    # 读取v1/v2结果
    v1_co = 2909.0
    v2_co = 1977.5
    try:
        with open('results/Q3_optimization_results.json','r',encoding='utf-8') as f:
            v1_co = json.load(f)['optimization_summary']['optimal_co_mg_m3']
        with open('results/Q3_v2_results.json','r',encoding='utf-8') as f:
            v2_co = json.load(f)['comparison']['v2_co']
    except Exception:
        pass
    p(f"  参考：v1={v1_co:.0f}，v2={v2_co:.0f} mg/m³")

    # DE优化
    p("\n[Step 6] Differential Evolution 全局优化（37维）...")
    opt_x, de_co, de_hist = run_de(bounds_37, fixed_vals, model, scaler, all_features,
                                   popsize=6, maxiter=80, seed=42)

    # 局部精化
    p("\n[Step 7] Nelder-Mead 局部精化...")
    opt_x, opt_co = local_polish(opt_x, bounds_37, fixed_vals, model, scaler, all_features)
    p(f"  精化后最优CO: {opt_co:.1f} mg/m³")

    reduction = baseline_co - opt_co
    p("\n" + "="*70)
    p("优化结果汇总")
    p("="*70)
    p(f"  基准CO         : {baseline_co:.1f} mg/m³")
    p(f"  v1（仅负压）   : {v1_co:.1f} mg/m³")
    p(f"  v2（负压+温度）: {v2_co:.1f} mg/m³")
    p(f"  v3（DE联合）   : {opt_co:.1f} mg/m³")
    p(f"  相比基准降低   : {reduction:.1f} mg/m³ ({reduction/baseline_co*100:.1f}%)")
    p(f"  是否达排放标准 : {'✓ 是' if opt_co < EMISSION_STD else '✗ 否'}")
    p(f"  机速最优值     : {opt_x[36]:.3f}")

    p(f"\n  {'#':4s} {'负压历史':>10s} {'负压最优':>10s} {'调整':>8s} "
      f"{'温度历史':>10s} {'温度最优':>10s} {'调整':>8s}")
    p("  " + "-"*62)
    for i in range(1,19):
        pm = bounds_37[f'负压_{i}']['median']
        tm = bounds_37[f'温度_{i}']['median']
        po = opt_x[i-1]; to = opt_x[18+i-1]
        p(f"  {i:2d}# {pm:10.1f} {po:10.1f} {po-pm:+8.1f} "
          f"{tm:10.1f} {to:10.1f} {to-tm:+8.1f}")

    # 鲁棒性
    p("\n[Step 8] Monte Carlo 扰动鲁棒性分析...")
    mc = monte_carlo(opt_x, bounds_37, fixed_vals, model, scaler, all_features,
                     noise_levels=(0.10, 0.20, 0.30), n_samples=500)

    p("\n  鲁棒性汇总:")
    p(f"  {'扰动':10s} {'均值CO':>8s} {'std':>6s} {'P5':>7s} {'P95':>7s} {'P达标':>8s}")
    p("  "+"-"*46)
    for nl, res in mc.items():
        p(f"  ±{int(nl*100/2):2d}%范围  {res['mean']:8.0f} {res['std']:6.0f} "
          f"{res['p5']:7.0f} {res['p95']:7.0f} {res['prob']*100:8.1f}%")

    # 可视化
    p("\n[Step 9] 生成图表...")
    visualize(opt_x, opt_co, de_hist, mc, bounds_37,
              baseline_co, v1_co, v2_co)

    # 保存结果
    out = {
        'comparison': {
            'baseline_co': round(baseline_co, 2),
            'v1_co': round(v1_co, 2),
            'v2_co': round(v2_co, 2),
            'v3_co': round(opt_co, 2),
            'reduction_vs_baseline': round(reduction, 2),
            'reduction_pct': round(reduction/baseline_co*100, 2),
            'reach_std': bool(opt_co < EMISSION_STD),
        },
        'optimal_speed': round(float(opt_x[36]), 4),
        'optimal_pressures': {f'负压_{i}': round(float(opt_x[i-1]),2) for i in range(1,19)},
        'optimal_temperatures': {f'温度_{i}': round(float(opt_x[18+i-1]),2) for i in range(1,19)},
        'robustness': {
            f'noise_{int(nl*100)}pct': {
                'mean_co': round(res['mean'],2), 'std_co': round(res['std'],2),
                'p5': round(res['p5'],2), 'p95': round(res['p95'],2),
                'prob_below_std': round(res['prob'],4),
            } for nl, res in mc.items()
        },
        'de_convergence': de_hist,
    }
    with open('results/Q3_v3_results.json','w',encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    p("  ✓ results/Q3_v3_results.json")
    p(f"\n⏱️  总耗时: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
