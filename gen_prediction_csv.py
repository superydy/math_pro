#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 Q2 最优参数重新跑一遍，生成预测值 vs 真实值对比表格
输出：
  results/Q2_prediction_detail.csv   全量（训练+测试）
  results/Q2_test_prediction.csv     仅测试集（附原始特征部分列）
"""
import os, json, warnings
os.chdir('/home/user/math_pro')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from scipy.signal import fftconvolve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

# ── 1. 加载数据（与 Q2 完全一致）─────────────────────────────────────────────
df = pd.read_csv('data/processed_data.csv')
abnormal = list(range(1045, 1062))
df = df[~df.index.isin(abnormal)].reset_index(drop=True)
print(f"去除异常后：{len(df)} 行")

with open('results/Q2_complete_results.json', 'r', encoding='utf-8') as f:
    q2 = json.load(f)
lag_results = q2['lag_results']
best_params  = q2['model_info']['best_params']

# ── 2. 时滞对齐 ───────────────────────────────────────────────────────────────
var_list = ['机速']
for i in range(1, 19):
    var_list.extend([f'负压_{i}', f'温度_{i}'])
var_list += ['大烟道负压_1', '大烟道负压_2', '大烟道温度_1', '大烟道温度_2']

df_al = df.copy()
for v in var_list:
    lag = lag_results.get(v, 0)
    df_al[f'{v}_al'] = df_al[v].shift(-lag)

# ── 3. 特征工程（84 维）─────────────────────────────────────────────────────
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
df_f = df_al.dropna(subset=feats + ['CO浓度']).reset_index(drop=True)
print(f"特征工程后：{len(df_f)} 行，{len(feats)} 个特征")

X = df_f[feats].values
y = df_f['CO浓度'].values

# ── 4. 训练/测试划分 ──────────────────────────────────────────────────────────
split = int(len(y) * 0.7)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]

sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr)
X_te_s  = sc.transform(X_te)

print(f"训练集：{len(y_tr)} 条，测试集：{len(y_te)} 条")

# ── 5. 训练模型 ───────────────────────────────────────────────────────────────
model = XGBRegressor(**best_params, random_state=42, n_jobs=-1, verbosity=0)
model.fit(X_tr_s, y_tr, verbose=False)

pred_tr = model.predict(X_tr_s)
pred_te = model.predict(X_te_s)

r2_tr  = r2_score(y_tr, pred_tr)
r2_te  = r2_score(y_te, pred_te)
mae_te = mean_absolute_error(y_te, pred_te)
rmse_te = np.sqrt(np.mean((y_te - pred_te)**2))
print(f"\n训练集 R²={r2_tr:.4f}")
print(f"测试集  R²={r2_te:.4f}  RMSE={rmse_te:.2f}  MAE={mae_te:.2f}")

# ── 6. 生成完整对比表格（全量）───────────────────────────────────────────────
df_all = df_f.copy()

# 拼接训练+测试预测值
pred_all = np.concatenate([pred_tr, pred_te])
set_label = ['训练集'] * split + ['测试集'] * (len(y) - split)

df_out = pd.DataFrame({
    '序号':         range(1, len(df_all) + 1),
    '数据集':       set_label,
    '机速(m/min)':  df_all['机速_al'].values,
    '负压_均值(Pa)': df_all['p_mean'].values,
    '负压_中段均值': df_all['p_mid'].values,
    '负压_后段均值': df_all['p_back'].values,
    '前1步CO(ppm)': df_all['co_lag1'].values,
    '前2步CO(ppm)': df_all['co_lag2'].values,
    '滑动均值CO5':  df_all['co_ma5'].values,
    'CO真实值(ppm)':  np.round(y,       2),
    'CO预测值(ppm)':  np.round(pred_all, 2),
    '绝对误差(ppm)':  np.round(np.abs(y - pred_all), 2),
    '相对误差(%)':    np.round(np.abs(y - pred_all) / (np.abs(y) + 1e-6) * 100, 2),
})

out_all = 'results/Q2_prediction_detail.csv'
df_out.to_csv(out_all, index=False, encoding='utf-8-sig')
print(f"\n✓ {out_all}  ({len(df_out)} 行)")

# ── 7. 仅测试集，附更多原始特征 ───────────────────────────────────────────────
df_te_raw = df_f.iloc[split:].reset_index(drop=True)

p_cols_show = [f'负压_{i}_al' for i in range(1,19)]
t_cols_show = [f'温度_{i}_al' for i in range(1,19)]

df_te_out = pd.DataFrame({'序号': range(1, len(y_te)+1)})
df_te_out['机速(m/min)'] = df_te_raw['机速_al'].values

# 18路负压（保留2位小数）
for i in range(1, 19):
    df_te_out[f'负压_{i}(Pa)'] = df_te_raw[f'负压_{i}_al'].values.round(2)

# 18路温度
for i in range(1, 19):
    df_te_out[f'温度_{i}(℃)'] = df_te_raw[f'温度_{i}_al'].values.round(2)

# CO 自回归
df_te_out['前1步CO'] = df_te_raw['co_lag1'].values.round(2)
df_te_out['前2步CO'] = df_te_raw['co_lag2'].values.round(2)
df_te_out['滑动均值CO5'] = df_te_raw['co_ma5'].values.round(2)

# 预测结果
df_te_out['CO真实值(ppm)'] = y_te.round(2)
df_te_out['CO预测值(ppm)'] = pred_te.round(2)
df_te_out['绝对误差(ppm)'] = np.abs(y_te - pred_te).round(2)
df_te_out['相对误差(%)']   = (np.abs(y_te - pred_te) / (np.abs(y_te)+1e-6) * 100).round(2)
df_te_out['误差方向']      = np.where(pred_te > y_te, '偏高', '偏低')

out_te = 'results/Q2_test_prediction.csv'
df_te_out.to_csv(out_te, index=False, encoding='utf-8-sig')
print(f"✓ {out_te}  ({len(df_te_out)} 行，{len(df_te_out.columns)} 列)")

# ── 8. 简要统计摘要 ───────────────────────────────────────────────────────────
errs = np.abs(y_te - pred_te)
print(f"""
=== 测试集预测统计 ===
样本数       : {len(y_te)}
R²           : {r2_te:.4f}
RMSE         : {rmse_te:.2f} ppm
MAE          : {mae_te:.2f} ppm
误差 <50ppm  : {(errs<50).sum()} / {len(errs)}  ({(errs<50).mean()*100:.1f}%)
误差 <100ppm : {(errs<100).sum()} / {len(errs)}  ({(errs<100).mean()*100:.1f}%)
误差 <200ppm : {(errs<200).sum()} / {len(errs)}  ({(errs<200).mean()*100:.1f}%)
最大误差     : {errs.max():.2f} ppm
中位误差     : {np.median(errs):.2f} ppm
""")
