#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型对比 v2：
  线性回归 / 随机森林 / LightGBM / CatBoost / XGBoost+PSO(本文)
  指标：R², RMSE, MAE
"""

import pandas as pd
import numpy as np
import json, time, os, warnings
warnings.filterwarnings('ignore')
os.chdir('/home/user/math_pro')

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor

# ─── 数据加载（与 Q2 完全一致）──────────────────────────────────────────────
def load_data():
    df = pd.read_csv('data/processed_data.csv')
    abnormal = list(range(1045, 1062))
    df = df[~df.index.isin(abnormal)].reset_index(drop=True)

    with open('results/Q2_complete_results.json', 'r', encoding='utf-8') as f:
        q2 = json.load(f)
    lag_results = q2['lag_results']

    var_list = ['机速']
    for i in range(1, 19):
        var_list.extend([f'负压_{i}', f'温度_{i}'])
    var_list += ['大烟道负压_1','大烟道负压_2','大烟道温度_1','大烟道温度_2']

    df_al = df.copy()
    for v in var_list:
        lag = lag_results.get(v, 0)
        df_al[f'{v}_al'] = df_al[v].shift(-lag)

    phys = ['机速_al'] + [f'负压_{i}_al' for i in range(1,19)] + [f'温度_{i}_al' for i in range(1,19)]
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
    df_al['co_lag1']  = df_al['CO浓度'].shift(1)
    df_al['co_lag2']  = df_al['CO浓度'].shift(2)
    df_al['co_lag5']  = df_al['CO浓度'].shift(5)
    df_al['co_ma5']   = df_al['CO浓度'].rolling(5).mean()
    df_al['co_diff1'] = df_al['CO浓度'].diff(1)
    co_f = ['co_lag1','co_lag2','co_lag5','co_ma5','co_diff1']

    feats = phys + grad + stat + co_f
    df_f = df_al.dropna(subset=feats+['CO浓度']).reset_index(drop=True)
    X = df_f[feats].values
    y = df_f['CO浓度'].values

    split = int(len(y) * 0.7)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s  = sc.transform(X_te)

    return X_tr, X_te, X_tr_s, X_te_s, y_tr, y_te

def metrics(y_true, y_pred):
    r2   = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    return round(r2,4), round(rmse,2), round(mae,2)

# ─── PSO（保持与 v1 完全一致，8粒子×10迭代）────────────────────────────────
def run_pso(X_tr_s, y_tr, n_particles=8, n_iter=10):
    tss = TimeSeriesSplit(n_splits=3)

    def cv_score(params):
        lr    = float(params[0])
        depth = int(round(params[1]))
        n_est = int(round(params[2]))
        sub   = float(params[3])
        col   = float(params[4])
        alpha = float(params[5])
        lam   = float(params[6])
        mdl = XGBRegressor(
            learning_rate=lr, max_depth=depth, n_estimators=n_est,
            subsample=sub, colsample_bytree=col,
            reg_alpha=alpha, reg_lambda=lam,
            random_state=42, n_jobs=-1, verbosity=0)
        scores = []
        for tr_i, va_i in tss.split(X_tr_s):
            mdl.fit(X_tr_s[tr_i], y_tr[tr_i], verbose=False)
            scores.append(r2_score(y_tr[va_i], mdl.predict(X_tr_s[va_i])))
        return float(np.mean(scores))

    lb = np.array([0.01, 2, 50,  0.5, 0.5, 0.0, 0.0])
    ub = np.array([0.50, 8, 500, 1.0, 1.0, 2.0, 5.0])
    np.random.seed(42)
    positions  = lb + np.random.rand(n_particles, 7) * (ub - lb)
    velocities = np.zeros((n_particles, 7))
    pbest_pos  = positions.copy()
    pbest_val  = np.full(n_particles, -np.inf)
    gbest_pos  = positions[0].copy()
    gbest_val  = -np.inf
    w, c1, c2  = 0.8, 2.0, 2.0

    for it in range(n_iter):
        for p in range(n_particles):
            score = cv_score(positions[p])
            if score > pbest_val[p]:
                pbest_val[p] = score; pbest_pos[p] = positions[p].copy()
            if score > gbest_val:
                gbest_val = score; gbest_pos = positions[p].copy()
        r1 = np.random.rand(n_particles, 7)
        r2_arr = np.random.rand(n_particles, 7)
        velocities = (w * velocities
                      + c1 * r1 * (pbest_pos - positions)
                      + c2 * r2_arr * (gbest_pos - positions))
        positions = np.clip(positions + velocities, lb, ub)
        print(f"  PSO 迭代 {it+1:2d}/{n_iter}  全局最优CV-R²={gbest_val:.4f}")

    param_names = ['learning_rate','max_depth','n_estimators','subsample',
                   'colsample_bytree','reg_alpha','reg_lambda']
    best_params = {n: float(gbest_pos[i]) for i, n in enumerate(param_names)}
    best_params['max_depth']    = int(round(best_params['max_depth']))
    best_params['n_estimators'] = int(round(best_params['n_estimators']))
    return best_params, round(gbest_val, 4)

# ─── 生成对比图 ──────────────────────────────────────────────────────────────
def gen_comparison_figure(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Patch

    font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'WenQuanYi Zen Hei'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 11

    names  = list(results.keys())
    r2s    = [v['r2']   for v in results.values()]
    rmses  = [v['rmse'] for v in results.values()]
    maes   = [v['mae']  for v in results.values()]
    # 颜色：RF灰蓝、LightGBM绿、CatBoost橙、XGBoost+PSO深红
    colors = ['#92C5DE', '#4DAF4A', '#F4A582', '#D6604D']

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    for ax, vals, ylabel, title, best_is_high in zip(
            axes,
            [r2s, rmses, maes],
            ['R²', 'RMSE (ppm)', 'MAE (ppm)'],
            ['(a) R²（↑越高越好）', '(b) RMSE（↓越低越好）', '(c) MAE（↓越低越好）'],
            [True, False, False]):

        bars = ax.bar(x, vals, color=colors, width=0.6, edgecolor='white', linewidth=0.8)

        # 高亮最优 bar
        best_idx = int(np.argmax(vals) if best_is_high else np.argmin(vals))
        bars[best_idx].set_edgecolor('#8B0000')
        bars[best_idx].set_linewidth(2.5)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=18, ha='right', fontsize=9.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=8)

        # 标注数值
        span = max(vals) - min(vals) if max(vals) != min(vals) else 1
        for bar, val in zip(bars, vals):
            fmt = f'{val:.4f}' if ylabel == 'R²' else f'{val:.1f}'
            ax.text(bar.get_x()+bar.get_width()/2,
                    val + span * 0.025,
                    fmt, ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if ylabel == 'R²':
            ax.set_ylim(min(r2s)*0.95, max(r2s)*1.05)

    handles = [Patch(color=colors[i], label=n) for i, n in enumerate(names)]
    fig.legend(handles=handles, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 1.03), fontsize=10, framealpha=0.9)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('paper_figures/fig_comparison_v2.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  ✓ paper_figures/fig_comparison_v2.png')


# ─── 主程序 ─────────────────────────────────────────────────────────────────
def main():
    print("加载数据...")
    X_tr, X_te, X_tr_s, X_te_s, y_tr, y_te = load_data()
    print(f"  训练集={len(y_tr)}, 测试集={len(y_te)}")

    results = {}

    # 1. 线性回归
    # 1. 随机森林
    print("\n[1/4] 随机森林 (n=200)...")
    t = time.time()
    mdl = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    mdl.fit(X_tr, y_tr)
    pred = mdl.predict(X_te)
    r2, rmse, mae = metrics(y_te, pred)
    results['随机森林'] = {'r2':r2,'rmse':rmse,'mae':mae,'time':round(time.time()-t,2)}
    print(f"  R²={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}")

    # 2. LightGBM
    print("\n[2/4] LightGBM (默认)...")
    t = time.time()
    mdl = lgb.LGBMRegressor(n_estimators=200, random_state=42, n_jobs=-1, verbose=-1)
    mdl.fit(X_tr_s, y_tr)
    pred = mdl.predict(X_te_s)
    r2, rmse, mae = metrics(y_te, pred)
    results['LightGBM'] = {'r2':r2,'rmse':rmse,'mae':mae,'time':round(time.time()-t,2)}
    print(f"  R²={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}")

    # 3. CatBoost
    print("\n[3/4] CatBoost (默认)...")
    t = time.time()
    mdl = CatBoostRegressor(n_estimators=200, random_seed=42,
                             verbose=False, thread_count=-1)
    mdl.fit(X_tr_s, y_tr)
    pred = mdl.predict(X_te_s)
    r2, rmse, mae = metrics(y_te, pred)
    results['CatBoost'] = {'r2':r2,'rmse':rmse,'mae':mae,'time':round(time.time()-t,2)}
    print(f"  R²={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}")

    # 4. XGBoost + PSO（本文）
    print("\n[4/4] XGBoost + PSO（本文）...")
    best_params, pso_cv_r2 = run_pso(X_tr_s, y_tr)
    t = time.time()
    mdl = XGBRegressor(**best_params, random_state=42, n_jobs=-1, verbosity=0)
    mdl.fit(X_tr_s, y_tr, verbose=False)
    pred = mdl.predict(X_te_s)
    r2, rmse, mae = metrics(y_te, pred)
    results['XGBoost+PSO(本文)'] = {
        'r2':r2,'rmse':rmse,'mae':mae,'time':round(time.time()-t,2),
        'best_params': best_params, 'pso_cv_r2': pso_cv_r2,
    }
    print(f"  R²={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}")

    # 保存
    with open('results/model_comparison_v2.json','w',encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n✓ results/model_comparison_v2.json")

    # 汇总打印
    print("\n" + "="*65)
    print(f"{'模型':20s} {'R²':>8s} {'RMSE':>10s} {'MAE':>10s} {'耗时':>8s}")
    print("-"*65)
    for name, r in results.items():
        mark = " ★" if '本文' in name else ""
        print(f"{name:20s} {r['r2']:8.4f} {r['rmse']:10.2f} {r['mae']:10.2f} {r['time']:8.2f}s{mark}")
    print("="*65)

    # 生成对比图
    gen_comparison_figure(results)
    return results

if __name__ == '__main__':
    main()
