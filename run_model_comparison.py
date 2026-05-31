#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实训练对比模型 + 重跑PSO记录迭代过程
对比方案：Ridge → RF → GBR(默认) → LightGBM → XGBoost(默认) → XGBoost+PSO(本文)
"""

import pandas as pd
import numpy as np
import json, time, os, warnings
warnings.filterwarnings('ignore')
os.chdir('/home/user/math_pro')

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBRegressor
import lightgbm as lgb

# ─── 数据加载（与Q2完全一致）──────────────────────────────────────────────────
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

    return X_tr, X_te, X_tr_s, X_te_s, y_tr, y_te, sc, feats

# ─── PSO（带详细日志）─────────────────────────────────────────────────────────
def pso_with_log(X_tr_s, y_tr, n_particles=8, n_iter=10):
    """重跑PSO，记录每次迭代的最优粒子参数和全部粒子得分"""
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
            random_state=42, n_jobs=-1, verbosity=0
        )
        scores = []
        for tr_i, va_i in tss.split(X_tr_s):
            mdl.fit(X_tr_s[tr_i], y_tr[tr_i], verbose=False)
            scores.append(r2_score(y_tr[va_i], mdl.predict(X_tr_s[va_i])))
        return float(np.mean(scores))

    # 搜索空间 [lb, ub]
    lb = np.array([0.01, 2, 50,  0.5, 0.5, 0.0, 0.0])
    ub = np.array([0.50, 8, 500, 1.0, 1.0, 2.0, 5.0])

    np.random.seed(42)
    positions  = lb + np.random.rand(n_particles, 7) * (ub - lb)
    velocities = np.zeros((n_particles, 7))
    pbest_pos  = positions.copy()
    pbest_val  = np.full(n_particles, -np.inf)
    gbest_pos  = positions[0].copy()
    gbest_val  = -np.inf

    w, c1, c2 = 0.8, 2.0, 2.0
    iter_log   = []   # 每次迭代记录
    all_scores = []   # 全部粒子分数记录

    param_names = ['learning_rate','max_depth','n_estimators','subsample',
                   'colsample_bytree','reg_alpha','reg_lambda']

    print(f"  PSO: {n_particles}粒子 × {n_iter}迭代")
    t0 = time.time()

    for it in range(n_iter):
        it_scores = []
        for p in range(n_particles):
            score = cv_score(positions[p])
            it_scores.append(score)
            if score > pbest_val[p]:
                pbest_val[p] = score
                pbest_pos[p] = positions[p].copy()
            if score > gbest_val:
                gbest_val = score
                gbest_pos = positions[p].copy()

        # 记录本次迭代
        best_p_idx = np.argmax(it_scores)
        iter_info  = {
            'iter': it + 1,
            'gbest_r2': round(gbest_val, 4),
            'iter_best_r2': round(max(it_scores), 4),
            'iter_mean_r2': round(float(np.mean(it_scores)), 4),
            'gbest_params': {n: round(float(gbest_pos[i]), 4) for i, n in enumerate(param_names)},
            'particle_scores': [round(s, 4) for s in it_scores],
        }
        iter_log.append(iter_info)

        print(f"  迭代 {it+1:2d}/{n_iter}  全局最优CV-R²={gbest_val:.4f}  "
              f"本轮均值={float(np.mean(it_scores)):.4f}  耗时{time.time()-t0:.0f}s")

        # 更新速度和位置
        r1 = np.random.rand(n_particles, 7)
        r2 = np.random.rand(n_particles, 7)
        velocities = (w * velocities
                      + c1 * r1 * (pbest_pos - positions)
                      + c2 * r2 * (gbest_pos - positions))
        positions  = np.clip(positions + velocities, lb, ub)

    print(f"  PSO完成，总耗时{time.time()-t0:.0f}s")
    return gbest_pos, gbest_val, iter_log

# ─── 主程序 ───────────────────────────────────────────────────────────────────
def main():
    print("加载数据...")
    X_tr, X_te, X_tr_s, X_te_s, y_tr, y_te, sc, feats = load_data()
    print(f"  训练集: {len(y_tr)}, 测试集: {len(y_te)}, 特征: {len(feats)}")

    results = {}

    # ── 1. Ridge 回归 ──────────────────────────────────────────────────────────
    print("\n[1/6] Ridge 回归...")
    t = time.time()
    mdl = Ridge(alpha=1.0)
    mdl.fit(X_tr_s, y_tr)
    y_pred = mdl.predict(X_te_s)
    results['Ridge回归'] = {
        'r2':   round(r2_score(y_te, y_pred), 4),
        'mae':  round(mean_absolute_error(y_te, y_pred), 2),
        'rmse': round(float(np.sqrt(np.mean((y_te-y_pred)**2))), 2),
        'time': round(time.time()-t, 2),
    }
    print(f"  R²={results['Ridge回归']['r2']:.4f}")

    # ── 2. 随机森林 ────────────────────────────────────────────────────────────
    print("\n[2/6] 随机森林 (n=100)...")
    t = time.time()
    mdl = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    mdl.fit(X_tr, y_tr)            # RF不需要标准化
    y_pred = mdl.predict(X_te)
    results['随机森林'] = {
        'r2':   round(r2_score(y_te, y_pred), 4),
        'mae':  round(mean_absolute_error(y_te, y_pred), 2),
        'rmse': round(float(np.sqrt(np.mean((y_te-y_pred)**2))), 2),
        'time': round(time.time()-t, 2),
    }
    print(f"  R²={results['随机森林']['r2']:.4f}")

    # ── 3. Gradient Boosting（sklearn默认）────────────────────────────────────
    print("\n[3/6] Gradient Boosting (sklearn默认)...")
    t = time.time()
    mdl = GradientBoostingRegressor(n_estimators=100, random_state=42)
    mdl.fit(X_tr_s, y_tr)
    y_pred = mdl.predict(X_te_s)
    results['梯度提升树(GBR)'] = {
        'r2':   round(r2_score(y_te, y_pred), 4),
        'mae':  round(mean_absolute_error(y_te, y_pred), 2),
        'rmse': round(float(np.sqrt(np.mean((y_te-y_pred)**2))), 2),
        'time': round(time.time()-t, 2),
    }
    print(f"  R²={results['梯度提升树(GBR)']['r2']:.4f}")

    # ── 4. LightGBM ────────────────────────────────────────────────────────────
    print("\n[4/6] LightGBM (默认参数)...")
    t = time.time()
    mdl = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
    mdl.fit(X_tr_s, y_tr)
    y_pred = mdl.predict(X_te_s)
    results['LightGBM'] = {
        'r2':   round(r2_score(y_te, y_pred), 4),
        'mae':  round(mean_absolute_error(y_te, y_pred), 2),
        'rmse': round(float(np.sqrt(np.mean((y_te-y_pred)**2))), 2),
        'time': round(time.time()-t, 2),
    }
    print(f"  R²={results['LightGBM']['r2']:.4f}")

    # ── 5. XGBoost（默认参数，无PSO调优）──────────────────────────────────────
    print("\n[5/6] XGBoost (默认参数，无PSO)...")
    t = time.time()
    mdl = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0)
    mdl.fit(X_tr_s, y_tr, verbose=False)
    y_pred = mdl.predict(X_te_s)
    results['XGBoost(默认)'] = {
        'r2':   round(r2_score(y_te, y_pred), 4),
        'mae':  round(mean_absolute_error(y_te, y_pred), 2),
        'rmse': round(float(np.sqrt(np.mean((y_te-y_pred)**2))), 2),
        'time': round(time.time()-t, 2),
    }
    print(f"  R²={results['XGBoost(默认)']['r2']:.4f}")

    # ── 6. XGBoost + PSO（本文方法）──────────────────────────────────────────
    print("\n[6/6] XGBoost + PSO（本文方法）...")
    gbest_pos, gbest_val, iter_log = pso_with_log(X_tr_s, y_tr, n_particles=8, n_iter=10)

    # 用最优参数训练最终模型
    param_names = ['learning_rate','max_depth','n_estimators','subsample',
                   'colsample_bytree','reg_alpha','reg_lambda']
    best_params = {n: float(gbest_pos[i]) for i, n in enumerate(param_names)}
    best_params['max_depth']    = int(round(best_params['max_depth']))
    best_params['n_estimators'] = int(round(best_params['n_estimators']))

    t = time.time()
    mdl_final = XGBRegressor(**best_params, random_state=42, n_jobs=-1, verbosity=0)
    mdl_final.fit(X_tr_s, y_tr, verbose=False)
    y_pred_final = mdl_final.predict(X_te_s)
    results['XGBoost+PSO(本文)'] = {
        'r2':   round(r2_score(y_te, y_pred_final), 4),
        'mae':  round(mean_absolute_error(y_te, y_pred_final), 2),
        'rmse': round(float(np.sqrt(np.mean((y_te-y_pred_final)**2))), 2),
        'time': round(time.time()-t, 2),
        'best_params': best_params,
        'pso_cv_r2':   round(gbest_val, 4),
    }
    print(f"  R²={results['XGBoost+PSO(本文)']['r2']:.4f}")

    # ── 保存 ──────────────────────────────────────────────────────────────────
    out = {
        'model_comparison': results,
        'pso_iteration_log': iter_log,
    }
    with open('results/model_comparison_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n✓ results/model_comparison_results.json")

    # 打印汇总
    print("\n" + "="*65)
    print(f"{'模型':20s} {'R²':>8s} {'MAE':>10s} {'RMSE':>10s} {'耗时(s)':>8s}")
    print("-"*65)
    for name, r in results.items():
        marker = " ★" if name == 'XGBoost+PSO(本文)' else ""
        print(f"{name:20s} {r['r2']:8.4f} {r['mae']:10.2f} {r['rmse']:10.2f} {r['time']:8.2f}{marker}")
    print("="*65)

if __name__ == '__main__':
    main()
