#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CatBoost + PSO 超参数优化  —  问题二 CO浓度预测
与 XGBoost+PSO 完全相同的数据流程，仅替换核心模型
"""
import os, json, warnings, random
os.chdir('/home/user/math_pro')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error
from catboost import CatBoostRegressor

np.random.seed(42)
random.seed(42)

# ── 1. 加载数据（与 Q2 完全一致）─────────────────────────────────────
df = pd.read_csv('data/processed_data.csv')
abnormal = list(range(1045, 1062))
df = df[~df.index.isin(abnormal)].reset_index(drop=True)
print(f"去除异常后：{len(df)} 行")

# ── 2. 时滞对齐（复用 Q2 最优时滞）──────────────────────────────────
with open('results/Q2_complete_results.json', 'r', encoding='utf-8') as f:
    q2 = json.load(f)
lag_results = q2['lag_results']

var_list = ['机速']
for i in range(1, 19):
    var_list.extend([f'负压_{i}', f'温度_{i}'])
var_list += ['大烟道负压_1', '大烟道负压_2', '大烟道温度_1', '大烟道温度_2']

df_al = df.copy()
for v in var_list:
    lag = lag_results.get(v, 0)
    df_al[f'{v}_al'] = df_al[v].shift(-lag)

# ── 3. 84维特征工程（与 Q2 完全一致）─────────────────────────────────
phys = ['机速_al'] + [f'负压_{i}_al' for i in range(1,19)] + \
       [f'温度_{i}_al' for i in range(1,19)] + \
       ['大烟道负压_1_al','大烟道负压_2_al','大烟道温度_1_al','大烟道温度_2_al']

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
df_f  = df_al.dropna(subset=feats + ['CO浓度']).reset_index(drop=True)
print(f"特征工程后：{len(df_f)} 行，{len(feats)} 个特征")

X = df_f[feats].values
y = df_f['CO浓度'].values

# ── 4. 训练/测试划分（7:3 时序）──────────────────────────────────────
split   = int(len(y) * 0.7)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]

sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr)
X_te_s  = sc.transform(X_te)
print(f"训练集：{len(y_tr)} 条，测试集：{len(y_te)} 条")

# ── 5. PSO 超参数搜索空间 ─────────────────────────────────────────────
# [learning_rate, depth, iterations, l2_leaf_reg, subsample, colsample_bylevel]
LB = np.array([0.01, 2,  50, 0.0, 0.6, 0.6])
UB = np.array([0.30, 8, 500, 10., 1.0, 1.0])
DIM = len(LB)

CV_FOLDS = 3   # PSO内部加速评估
tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

def fitness(params):
    lr, depth, iters, l2, sub, col = params
    depth = int(round(depth))
    iters = int(round(iters))
    model = CatBoostRegressor(
        learning_rate   = lr,
        depth           = depth,
        iterations      = iters,
        l2_leaf_reg     = l2,
        subsample       = sub,
        colsample_bylevel = col,
        random_seed     = 42,
        verbose         = False,
        thread_count    = -1,
    )
    scores = []
    for tr_idx, val_idx in tscv.split(X_tr_s):
        model.fit(X_tr_s[tr_idx], y_tr[tr_idx])
        pred = model.predict(X_tr_s[val_idx])
        scores.append(r2_score(y_tr[val_idx], pred))
    return float(np.mean(scores))

# ── 6. PSO 主循环（8粒子 × 10迭代，与 Q2 相同配置）─────────────────
N_PART   = 8
N_ITER   = 10
W        = 0.8
C1 = C2  = 2.0

pos = LB + np.random.rand(N_PART, DIM) * (UB - LB)
vel = np.zeros((N_PART, DIM))
pbest      = pos.copy()
pbest_fit  = np.array([fitness(p) for p in pos])
gbest      = pbest[np.argmax(pbest_fit)].copy()
gbest_fit  = pbest_fit.max()

print(f"\n{'='*55}")
print(f"{'迭代':>4}  {'全局最优CV-R²':>14}  {'本轮最优CV-R²':>14}  {'本轮均值CV-R²':>14}")
print(f"{'='*55}")

history = []
for t in range(1, N_ITER + 1):
    r1 = np.random.rand(N_PART, DIM)
    r2 = np.random.rand(N_PART, DIM)
    vel = W * vel + C1*r1*(pbest - pos) + C2*r2*(gbest - pos)
    pos = np.clip(pos + vel, LB, UB)

    fits = np.array([fitness(p) for p in pos])
    improved = fits > pbest_fit
    pbest[improved]     = pos[improved].copy()
    pbest_fit[improved] = fits[improved]

    if pbest_fit.max() > gbest_fit:
        gbest     = pbest[np.argmax(pbest_fit)].copy()
        gbest_fit = pbest_fit.max()

    print(f"{t:>4}  {gbest_fit:>14.4f}  {fits.max():>14.4f}  {fits.mean():>14.4f}")
    history.append({
        'iter': t, 'gbest_cv': round(gbest_fit, 4),
        'best_cv': round(float(fits.max()), 4),
        'mean_cv': round(float(fits.mean()), 4)
    })

lr_opt, depth_opt, iters_opt, l2_opt, sub_opt, col_opt = gbest
depth_opt = int(round(depth_opt))
iters_opt = int(round(iters_opt))

print(f"\n{'='*55}")
print(f"最优超参数:")
print(f"  learning_rate    = {lr_opt:.4f}")
print(f"  depth            = {depth_opt}")
print(f"  iterations       = {iters_opt}")
print(f"  l2_leaf_reg      = {l2_opt:.4f}")
print(f"  subsample        = {sub_opt:.4f}")
print(f"  colsample_bylevel= {col_opt:.4f}")

# ── 7. 最终模型训练与评估 ─────────────────────────────────────────────
best_model = CatBoostRegressor(
    learning_rate     = lr_opt,
    depth             = depth_opt,
    iterations        = iters_opt,
    l2_leaf_reg       = l2_opt,
    subsample         = sub_opt,
    colsample_bylevel = col_opt,
    random_seed       = 42,
    verbose           = False,
    thread_count      = -1,
)
best_model.fit(X_tr_s, y_tr)

pred_tr = best_model.predict(X_tr_s)
pred_te = best_model.predict(X_te_s)

r2_tr   = r2_score(y_tr, pred_tr)
r2_te   = r2_score(y_te, pred_te)
mae_te  = mean_absolute_error(y_te, pred_te)
rmse_te = float(np.sqrt(np.mean((y_te - pred_te)**2)))

print(f"\n{'='*55}")
print(f"训练集  R² = {r2_tr:.4f}")
print(f"测试集  R² = {r2_te:.4f}")
print(f"测试集  RMSE = {rmse_te:.2f} mg/m³")
print(f"测试集  MAE  = {mae_te:.2f} mg/m³")

# ── 8. 5折CV 最终稳定性评估 ──────────────────────────────────────────
tscv5 = TimeSeriesSplit(n_splits=5)
cv5_scores = []
for tr_idx, val_idx in tscv5.split(X_tr_s):
    m = CatBoostRegressor(
        learning_rate=lr_opt, depth=depth_opt, iterations=iters_opt,
        l2_leaf_reg=l2_opt, subsample=sub_opt, colsample_bylevel=col_opt,
        random_seed=42, verbose=False, thread_count=-1,
    )
    m.fit(X_tr_s[tr_idx], y_tr[tr_idx])
    cv5_scores.append(r2_score(y_tr[val_idx], m.predict(X_tr_s[val_idx])))

print(f"\n5折CV 均值R² = {np.mean(cv5_scores):.4f}  ±  {np.std(cv5_scores):.4f}")
print(f"各折R²: {[round(s,4) for s in cv5_scores]}")

# ── 9. 与 XGBoost+PSO 对比汇总 ──────────────────────────────────────
print(f"\n{'='*55}")
print(f"{'模型':<20} {'测试R²':>8} {'RMSE':>10} {'MAE':>10}")
print(f"{'='*55}")
print(f"{'XGBoost+PSO(Q2)':<20} {'0.9337':>8} {'115.03':>10} {'51.09':>10}")
print(f"{'CatBoost+PSO(新)':<20} {r2_te:>8.4f} {rmse_te:>10.2f} {mae_te:>10.2f}")

# ── 10. 保存结果 ──────────────────────────────────────────────────────
result = {
    'model': 'CatBoost+PSO',
    'best_params': {
        'learning_rate': round(lr_opt, 4),
        'depth': depth_opt,
        'iterations': iters_opt,
        'l2_leaf_reg': round(l2_opt, 4),
        'subsample': round(sub_opt, 4),
        'colsample_bylevel': round(col_opt, 4),
    },
    'pso_config': {'n_particles': N_PART, 'n_iter': N_ITER, 'w': W, 'c1': C1, 'c2': C2, 'cv_folds': CV_FOLDS},
    'pso_history': history,
    'final_cv_r2': round(gbest_fit, 4),
    'test_r2':   round(r2_te, 4),
    'test_rmse': round(rmse_te, 2),
    'test_mae':  round(mae_te, 2),
    'cv5_mean':  round(float(np.mean(cv5_scores)), 4),
    'cv5_std':   round(float(np.std(cv5_scores)), 4),
    'cv5_folds': [round(s, 4) for s in cv5_scores],
}
with open('results/catboost_pso_results.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"\n✓ 结果已保存至 results/catboost_pso_results.json")
