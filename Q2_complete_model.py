#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题2：CO浓度预测模型 - 完整版（含PSO调参）
=============================================

完整流程：
1. 数据加载与预处理
2. 时滞对齐（互相关分析）
3. 特征工程（84个特征）
4. PSO粒子群优化XGBoost超参数
5. 用最优参数训练模型
6. 70/30划分评估 + 5折交叉验证
7. 特征重要性分析
8. 可视化

依赖库：
- pandas>=1.3.0
- numpy>=1.20.0
- scikit-learn>=1.0.0
- xgboost>=1.5.0
- scipy>=1.7.0
- matplotlib>=3.4.0

运行方式：
  python Q2_complete_model.py

作者：数学建模团队
日期：2026-05-30
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from scipy.signal import fftconvolve
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 全局参数
# ============================================================
PSO_PARTICLES = 8        # PSO粒子数（减少以加速）
PSO_ITERATIONS = 10      # PSO迭代次数（减少以加速）
CV_FOLDS = 3             # PSO内部用3折CV评估（加速）
FINAL_CV_FOLDS = 5       # 最终评估用5折CV

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 工具函数
# ============================================================
def print_flush(msg):
    """带flush的打印，确保实时输出"""
    print(msg, flush=True)


def compute_xcorr(x, y, max_lag=60):
    """
    计算两个时间序列的互相关函数（FFT加速）
    
    参数:
        x, y: 时间序列
        max_lag: 最大时滞步数
    
    返回:
        lags: 时滞数组 (-max_lag 到 +max_lag)
        cc: 互相关系数数组
    """
    xn = (x - np.mean(x)) / (np.std(x) * len(x) + 1e-8)
    yn = (y - np.mean(y)) / (np.std(y) + 1e-8)
    corr = fftconvolve(xn[::-1], yn, mode='full')
    center = len(xn) - 1
    lags = np.arange(-max_lag, max_lag + 1)
    cc = corr[center - max_lag: center + max_lag + 1]
    return lags, cc


# ============================================================
# Step 1: 数据加载与预处理
# ============================================================
def step1_load_data(filepath='data/processed_data.csv'):
    """加载数据并去除异常样本"""
    print_flush("="*70)
    print_flush("Step 1: 数据加载与预处理")
    print_flush("="*70)
    
    df = pd.read_csv(filepath)
    print_flush(f"原始数据: {df.shape[0]} 行, {df.shape[1]} 列")
    
    # 去除异常样本（索引1045-1061，传感器校准/恢复过渡期）
    abnormal = list(range(1045, 1062))
    df = df[~df.index.isin(abnormal)].reset_index(drop=True)
    print_flush(f"去除 {len(abnormal)} 个异常样本后: {len(df)} 行")
    
    return df


# ============================================================
# Step 2: 时滞对齐
# ============================================================
def step2_time_lag_alignment(df):
    """
    计算各变量与CO浓度的最优时滞，并对齐
    
    原理：
    - 烧结机18个风箱距离CO检测点远近不同
    - 远处的风箱影响需要更长时间传递到检测点
    - 用互相关分析找到每个变量与CO的最佳时滞
    
    返回:
        df: 对齐后的数据框
        lag_results: 各变量的最优时滞字典
    """
    print_flush("\n" + "="*70)
    print_flush("Step 2: 时滞对齐（互相关分析）")
    print_flush("="*70)
    
    co = df['CO浓度'].values
    
    # 需要分析的41个变量
    var_list = ['机速']
    for i in range(1, 19):
        var_list.extend([f'负压_{i}', f'温度_{i}'])
    var_list.extend(['大烟道负压_1', '大烟道负压_2', '大烟道温度_1', '大烟道温度_2'])
    
    # 计算每个变量的最优时滞
    lag_results = {}
    print_flush(f"\n{'变量':15s} {'最优时滞(步)':>12s} {'时滞(秒)':>10s} {'相关系数':>10s}")
    print_flush("-" * 50)
    
    for var in var_list:
        lags, cc = compute_xcorr(df[var].values, co, max_lag=60)
        best_idx = np.argmax(np.abs(cc))
        best_lag = int(lags[best_idx])
        best_corr = float(cc[best_idx])
        lag_results[var] = best_lag
        
        print_flush(f"{var:15s} {best_lag:12d} {best_lag*2:10d} {best_corr:10.4f}")
    
    # 应用时滞对齐
    df_aligned = df.copy()
    for var, lag in lag_results.items():
        if lag != 0:
            df_aligned[f'{var}_al'] = df_aligned[var].shift(-lag)
        else:
            df_aligned[f'{var}_al'] = df_aligned[var]
    
    # 删除因时滞产生的缺失值
    al_cols = [f'{var}_al' for var in var_list]
    before = len(df_aligned)
    df_aligned = df_aligned.dropna(subset=al_cols + ['CO浓度']).reset_index(drop=True)
    print_flush(f"\n时滞对齐后: {before} → {len(df_aligned)} 行")
    
    return df_aligned, lag_results


# ============================================================
# Step 3: 特征工程
# ============================================================
def step3_feature_engineering(df):
    """
    构建84个特征
    
    特征分组：
    1. 物理基础特征 (41个): 机速 + 18个负压 + 18个温度 + 4个大烟道参数
    2. 梯度特征 (34个): 17个负压梯度 + 17个温度梯度
    3. 统计特征 (4个): p_mean, p_mid, p_back, t_back
    4. CO自回归特征 (5个): co_lag1, co_lag2, co_lag5, co_ma5, co_diff1
    
    返回:
        X: 特征矩阵
        y: 标签数组
        all_features: 特征名列表
    """
    print_flush("\n" + "="*70)
    print_flush("Step 3: 特征工程")
    print_flush("="*70)
    
    # ---- 1. 物理基础特征 (41个) ----
    physical_features = ['机速_al']
    for i in range(1, 19):
        physical_features.extend([f'负压_{i}_al', f'温度_{i}_al'])
    physical_features.extend(['大烟道负压_1_al', '大烟道负压_2_al', 
                             '大烟道温度_1_al', '大烟道温度_2_al'])
    
    # ---- 2. 梯度特征 (34个) ----
    # 相邻风箱之间的压力/温度差值，反映空间变化
    gradient_features = []
    for i in range(1, 18):
        p_name = f'pgrad_{i}'
        t_name = f'tgrad_{i}'
        df[p_name] = df[f'负压_{i+1}_al'] - df[f'负压_{i}_al']
        df[t_name] = df[f'温度_{i+1}_al'] - df[f'温度_{i}_al']
        gradient_features.extend([p_name, t_name])
    
    # ---- 3. 统计特征 (4个) ----
    stat_features = []
    
    pcols = [f'负压_{i}_al' for i in range(1, 19)]
    df['p_mean'] = df[pcols].mean(axis=1)        # 全部负压均值
    stat_features.append('p_mean')
    
    df['p_mid'] = df[[f'负压_{i}_al' for i in range(6, 13)]].mean(axis=1)  # 中段负压均值
    stat_features.append('p_mid')
    
    df['p_back'] = df[[f'负压_{i}_al' for i in range(13, 19)]].mean(axis=1)  # 后段负压均值
    stat_features.append('p_back')
    
    df['t_back'] = df[[f'温度_{i}_al' for i in range(13, 19)]].mean(axis=1)  # 后段温度均值
    stat_features.append('t_back')
    
    # ---- 4. CO自回归特征 (5个) ----
    co_features = []
    
    df['co_lag1'] = df['CO浓度'].shift(1)     # CO滞后1步
    co_features.append('co_lag1')
    
    df['co_lag2'] = df['CO浓度'].shift(2)     # CO滞后2步
    co_features.append('co_lag2')
    
    df['co_lag5'] = df['CO浓度'].shift(5)     # CO滞后5步
    co_features.append('co_lag5')
    
    df['co_ma5'] = df['CO浓度'].rolling(5).mean()   # CO 5步滑动平均
    co_features.append('co_ma5')
    
    df['co_diff1'] = df['CO浓度'].diff(1)     # CO 1步差分
    co_features.append('co_diff1')
    
    # ---- 合并所有特征 ----
    all_features = physical_features + gradient_features + stat_features + co_features
    
    # 删除因shift/rolling产生的缺失值
    df_final = df.dropna(subset=all_features + ['CO浓度']).reset_index(drop=True)
    
    X = df_final[all_features].values
    y = df_final['CO浓度'].values
    
    print_flush(f"\n特征构建完成:")
    print_flush(f"  物理基础特征: {len(physical_features)} 个")
    print_flush(f"  梯度特征:     {len(gradient_features)} 个")
    print_flush(f"  统计特征:     {len(stat_features)} 个")
    print_flush(f"  CO自回归特征: {len(co_features)} 个")
    print_flush(f"  总特征数:     {len(all_features)} 个")
    print_flush(f"  样本数:       {len(y)}")
    
    return X, y, all_features


# ============================================================
# Step 4: PSO粒子群优化XGBoost超参数
# ============================================================
def step4_pso_tuning(X, y):
    """
    PSO粒子群优化XGBoost超参数
    
    优化目标：最大化3折时间序列交叉验证的R²
    
    PSO参数空间：
    - learning_rate: [0.01, 0.3]
    - max_depth: [3, 15]
    - n_estimators: [100, 500]
    - subsample: [0.5, 1.0]
    - colsample_bytree: [0.5, 1.0]
    - reg_alpha: [0, 10]
    - reg_lambda: [0, 10]
    
    返回:
        best_params: 最优参数字典
        pso_history: 每次迭代的最优CV R²
    """
    print_flush("\n" + "="*70)
    print_flush(f"Step 4: PSO粒子群优化超参数 ({PSO_PARTICLES}粒子 × {PSO_ITERATIONS}迭代)")
    print_flush("="*70)
    
    # ---- PSO参数空间定义 ----
    param_names = ['learning_rate', 'max_depth', 'n_estimators', 
                   'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda']
    param_bounds = {
        'learning_rate': (0.01, 0.3),
        'max_depth': (3, 15),
        'n_estimators': (100, 500),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0),
        'reg_alpha': (0, 10),
        'reg_lambda': (0, 10),
    }
    n_params = len(param_names)
    n_particles = PSO_PARTICLES
    n_iter = PSO_ITERATIONS
    
    # ---- 适应度函数：3折CV R² ----
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    
    def evaluate(params_dict):
        """评估一组XGBoost参数的CV R²"""
        model = XGBRegressor(**params_dict, random_state=42, n_jobs=-1)
        scores = []
        for tr_idx, va_idx in tscv.split(X):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X[tr_idx])
            X_va = sc.transform(X[va_idx])
            model.fit(X_tr, y[tr_idx], verbose=False)
            pred = model.predict(X_va)
            scores.append(r2_score(y[va_idx], pred))
        return np.mean(scores)
    
    # ---- PSO初始化 ----
    np.random.seed(42)
    
    # 粒子位置（在参数空间中随机初始化）
    positions = np.zeros((n_particles, n_params))
    for j, name in enumerate(param_names):
        lo, hi = param_bounds[name]
        positions[:, j] = np.random.uniform(lo, hi, n_particles)
    
    # 粒子速度
    velocities = np.random.uniform(-0.1, 0.1, (n_particles, n_params))
    
    # 个体最优
    pbest_pos = positions.copy()
    pbest_val = np.full(n_particles, -np.inf)
    
    # 全局最优
    gbest_pos = positions[0].copy()
    gbest_val = -np.inf
    
    # PSO参数
    w = 0.7    # 惯性权重（会自适应衰减）
    c1 = 1.5   # 个体学习因子
    c2 = 1.5   # 群体学习因子
    
    pso_history = []
    start_time = time.time()
    
    print_flush(f"\n开始PSO优化...")
    print_flush(f"  每次评估需要训练 {CV_FOLDS} 个XGBoost模型")
    print_flush(f"  总共需要评估 {n_particles * n_iter} 组参数\n")
    
    # ---- PSO主循环 ----
    for iteration in range(n_iter):
        iter_start = time.time()
        
        for p_idx in range(n_particles):
            # 将位置转换为XGBoost参数字典
            params_dict = {}
            for j, name in enumerate(param_names):
                val = positions[p_idx, j]
                if name in ['max_depth', 'n_estimators']:
                    params_dict[name] = int(round(val))
                else:
                    params_dict[name] = float(val)
            
            # 评估适应度
            score = evaluate(params_dict)
            
            # 更新个体最优
            if score > pbest_val[p_idx]:
                pbest_val[p_idx] = score
                pbest_pos[p_idx] = positions[p_idx].copy()
            
            # 更新全局最优
            if score > gbest_val:
                gbest_val = score
                gbest_pos = positions[p_idx].copy()
        
        pso_history.append(gbest_val)
        iter_time = time.time() - iter_start
        
        # 打印进度
        if (iteration + 1) % 5 == 0 or iteration == 0:
            elapsed = time.time() - start_time
            print_flush(f"  迭代 {iteration+1:2d}/{n_iter}: "
                       f"最优CV R² = {gbest_val:.4f} "
                       f"(耗时 {iter_time:.1f}s, 累计 {elapsed:.0f}s)")
        
        # ---- 更新速度和位置 ----
        r1 = np.random.random((n_particles, n_params))
        r2 = np.random.random((n_particles, n_params))
        
        velocities = (w * velocities 
                     + c1 * r1 * (pbest_pos - positions) 
                     + c2 * r2 * (gbest_pos - positions))
        
        positions = positions + velocities
        
        # 约束到参数范围
        for j, name in enumerate(param_names):
            lo, hi = param_bounds[name]
            positions[:, j] = np.clip(positions[:, j], lo, hi)
        
        # 惯性权重自适应衰减
        w = max(0.3, w * 0.97)
    
    total_time = time.time() - start_time
    
    # ---- 输出最优参数 ----
    best_params = {}
    for j, name in enumerate(param_names):
        val = gbest_pos[j]
        if name in ['max_depth', 'n_estimators']:
            best_params[name] = int(round(val))
        else:
            best_params[name] = float(val)
    
    print_flush(f"\n✓ PSO优化完成!")
    print_flush(f"  总耗时: {total_time:.0f}s")
    print_flush(f"  最优CV R²: {gbest_val:.4f}")
    print_flush(f"\n  最优参数:")
    for name, val in best_params.items():
        print_flush(f"    {name}: {val}")
    
    return best_params, pso_history


# ============================================================
# Step 5: 用最优参数训练模型并评估
# ============================================================
def step5_train_and_evaluate(X, y, best_params):
    """
    用PSO找到的最优参数训练最终模型
    
    评估方式:
    1. 70/30划分（按时间顺序）
    2. 5折时间序列交叉验证
    
    返回:
        results: 评估结果
        final_model: 最终模型
        final_scaler: 标准化器
    """
    print_flush("\n" + "="*70)
    print_flush("Step 5: 训练最终模型并评估")
    print_flush("="*70)
    
    results = {}
    
    # ========== 1. 70/30划分 ==========
    print_flush("\n[1/2] 70/30 划分评估")
    print_flush("-" * 40)
    
    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 标准化（fit在训练集，transform在测试集）
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # 训练
    model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    model.fit(X_train_s, y_train, verbose=False)
    
    # 预测
    y_pred = model.predict(X_test_s)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results['split_7030'] = {
        'r2': float(r2),
        'mae': float(mae),
        'rmse': float(rmse),
        'train_size': int(split),
        'test_size': int(len(X) - split)
    }
    
    print_flush(f"  训练集: {split} 个样本")
    print_flush(f"  测试集: {len(X) - split} 个样本")
    print_flush(f"  R²  = {r2:.4f}")
    print_flush(f"  MAE = {mae:.2f} mg/m³")
    print_flush(f"  RMSE= {rmse:.2f} mg/m³")
    
    # ========== 2. 5折交叉验证 ==========
    print_flush(f"\n[2/2] {FINAL_CV_FOLDS}折时间序列交叉验证")
    print_flush("-" * 40)
    
    tscv = TimeSeriesSplit(n_splits=FINAL_CV_FOLDS)
    cv_scores = []
    cv_details = []
    
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_va_s = sc.transform(X_va)
        
        m = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
        m.fit(X_tr_s, y_tr, verbose=False)
        pred = m.predict(X_va_s)
        
        fold_r2 = r2_score(y_va, pred)
        fold_mae = mean_absolute_error(y_va, pred)
        cv_scores.append(fold_r2)
        cv_details.append({
            'fold': fold,
            'r2': round(float(fold_r2), 4),
            'mae': round(float(fold_mae), 2),
            'train_size': int(len(tr_idx)),
            'test_size': int(len(va_idx))
        })
        print_flush(f"  Fold {fold}: R² = {fold_r2:.4f}, MAE = {fold_mae:.2f} "
                    f"(训练{len(tr_idx)}, 测试{len(va_idx)})")
    
    results['cv_5fold'] = {
        'fold_details': cv_details,
        'mean_r2': float(np.mean(cv_scores)),
        'std_r2': float(np.std(cv_scores)),
        'fold_scores': [float(s) for s in cv_scores]
    }
    
    print_flush(f"\n  平均 R² = {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # ========== 3. 全量训练最终模型 ==========
    print_flush(f"\n[3/3] 全量训练最终模型")
    print_flush("-" * 40)
    
    final_scaler = StandardScaler()
    X_all_s = final_scaler.fit_transform(X)
    
    final_model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    final_model.fit(X_all_s, y, verbose=False)
    
    y_pred_all = final_model.predict(X_all_s)
    full_r2 = r2_score(y, y_pred_all)
    print_flush(f"  全量 R² = {full_r2:.4f}")
    
    results['full_model'] = {
        'r2': float(full_r2),
        'n_samples': int(len(y))
    }
    
    return results, final_model, final_scaler


# ============================================================
# Step 6: 特征重要性分析
# ============================================================
def step6_feature_importance(model, feature_names):
    """
    分析特征重要性
    
    使用XGBoost内置的Gain重要性
    """
    print_flush("\n" + "="*70)
    print_flush("Step 6: 特征重要性分析")
    print_flush("="*70)
    
    importance = model.feature_importances_
    
    # 按重要性排序
    sorted_idx = np.argsort(importance)[::-1]
    
    print_flush(f"\nTop 20 最重要特征:")
    print_flush(f"{'排名':>4s} {'特征名':25s} {'重要性':>10s} {'类别':>10s}")
    print_flush("-" * 55)
    
    top_features = []
    for rank, idx in enumerate(sorted_idx[:20], 1):
        name = feature_names[idx]
        imp = importance[idx]
        
        # 判断类别
        if name.startswith('co_'):
            cat = 'CO自回归'
        elif name.startswith('pgrad') or name.startswith('tgrad'):
            cat = '梯度'
        elif name in ['p_mean', 'p_mid', 'p_back', 't_back']:
            cat = '统计'
        else:
            cat = '物理基础'
        
        top_features.append({'rank': rank, 'name': name, 'importance': float(imp), 'category': cat})
        print_flush(f"  {rank:2d}  {name:25s} {imp:10.4f} {cat:>10s}")
    
    # 各类别总重要性
    co_imp = sum(importance[i] for i in range(len(feature_names)) if feature_names[i].startswith('co_'))
    phys_imp = sum(importance[i] for i in range(len(feature_names)) 
                   if not feature_names[i].startswith('co_') 
                   and not feature_names[i].startswith('pgrad') 
                   and not feature_names[i].startswith('tgrad')
                   and feature_names[i] not in ['p_mean', 'p_mid', 'p_back', 't_back'])
    grad_imp = sum(importance[i] for i in range(len(feature_names)) 
                   if feature_names[i].startswith('pgrad') or feature_names[i].startswith('tgrad'))
    stat_imp = sum(importance[i] for i in range(len(feature_names)) 
                   if feature_names[i] in ['p_mean', 'p_mid', 'p_back', 't_back'])
    
    print_flush(f"\n各类别总重要性:")
    print_flush(f"  CO自回归:  {co_imp:.4f} ({co_imp*100:.1f}%)")
    print_flush(f"  物理基础:  {phys_imp:.4f} ({phys_imp*100:.1f}%)")
    print_flush(f"  梯度:      {grad_imp:.4f} ({grad_imp*100:.1f}%)")
    print_flush(f"  统计:      {stat_imp:.4f} ({stat_imp*100:.1f}%)")
    
    importance_dict = {
        'top20': top_features,
        'by_category': {
            'co_autoregressive': round(float(co_imp), 4),
            'physical': round(float(phys_imp), 4),
            'gradient': round(float(grad_imp), 4),
            'statistical': round(float(stat_imp), 4)
        }
    }
    
    return importance_dict, sorted_idx


# ============================================================
# Step 7: 可视化
# ============================================================
def step7_visualize(results, y_test, y_pred, importance_dict, sorted_idx, 
                    feature_names, model, pso_history):
    """生成可视化图表"""
    print_flush("\n" + "="*70)
    print_flush("Step 7: 生成可视化图表")
    print_flush("="*70)
    
    os.makedirs('figures', exist_ok=True)
    
    # ---- 图1: 预测 vs 真实 散点图 ----
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.5, s=15, c='steelblue', edgecolors='none')
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Ideal (y=x)')
    ax.axhline(y=2800, color='green', ls='-', lw=1.5, alpha=0.7, label='Standard: 2800')
    ax.set_xlabel('Actual CO (mg/m³)', fontsize=12)
    ax.set_ylabel('Predicted CO (mg/m³)', fontsize=12)
    r2 = results['split_7030']['r2']
    ax.set_title(f'Prediction vs Actual (R²={r2:.4f})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/Q2_scatter.png', dpi=200)
    plt.close()
    print_flush("  ✓ figures/Q2_scatter.png")
    
    # ---- 图2: 误差分布 ----
    errors = y_pred - y_test
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(x=0, color='red', ls='--', lw=2, label='Zero Error')
    ax.set_xlabel('Prediction Error (mg/m³)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Error Distribution (mean={errors.mean():.1f}, std={errors.std():.1f})', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/Q2_error_dist.png', dpi=200)
    plt.close()
    print_flush("  ✓ figures/Q2_error_dist.png")
    
    # ---- 图3: 特征重要性 Top 20 ----
    importance = model.feature_importances_
    top_n = 20
    top_idx = sorted_idx[:top_n]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    names = [feature_names[i] for i in top_idx]
    imps = [importance[i] for i in top_idx]
    
    # 按类别着色
    colors = []
    for name in names:
        if name.startswith('co_'):
            colors.append('#F44336')  # 红色 - CO自回归
        elif name.startswith('pgrad') or name.startswith('tgrad'):
            colors.append('#FF9800')  # 橙色 - 梯度
        elif name in ['p_mean', 'p_mid', 'p_back', 't_back']:
            colors.append('#9C27B0')  # 紫色 - 统计
        else:
            colors.append('#2196F3')  # 蓝色 - 物理基础
    
    ax.barh(range(top_n), imps[::-1], color=colors[::-1], alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
    ax.set_title('Top 20 Feature Importance', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#F44336', alpha=0.8, label='CO Autoregressive'),
        Patch(facecolor='#2196F3', alpha=0.8, label='Physical'),
        Patch(facecolor='#FF9800', alpha=0.8, label='Gradient'),
        Patch(facecolor='#9C27B0', alpha=0.8, label='Statistical'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('figures/Q2_importance.png', dpi=200)
    plt.close()
    print_flush("  ✓ figures/Q2_importance.png")
    
    # ---- 图4: 5折CV结果 ----
    fig, ax = plt.subplots(figsize=(10, 6))
    fold_scores = results['cv_5fold']['fold_scores']
    folds = range(1, len(fold_scores) + 1)
    
    bars = ax.bar(folds, fold_scores, color='steelblue', alpha=0.8, edgecolor='navy', width=0.5)
    ax.axhline(y=results['cv_5fold']['mean_r2'], color='red', ls='--', lw=2, 
               label=f'Mean: {results["cv_5fold"]["mean_r2"]:.4f}')
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title(f'{FINAL_CV_FOLDS}-Fold Cross-Validation (Mean={results["cv_5fold"]["mean_r2"]:.4f} ± {results["cv_5fold"]["std_r2"]:.4f})', fontsize=13)
    ax.set_xticks(folds)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    
    for bar, score in zip(bars, fold_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.4f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/Q2_cv_results.png', dpi=200)
    plt.close()
    print_flush("  ✓ figures/Q2_cv_results.png")
    
    # ---- 图5: PSO收敛曲线 ----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(pso_history) + 1), pso_history, 'b-o', lw=2, markersize=4, alpha=0.8)
    ax.set_xlabel('PSO Iteration', fontsize=12)
    ax.set_ylabel('Best CV R²', fontsize=12)
    ax.set_title(f'PSO Convergence (Final: {pso_history[-1]:.4f})', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 标注关键点
    ax.annotate(f'Best: {pso_history[-1]:.4f}', 
                xy=(len(pso_history), pso_history[-1]),
                xytext=(len(pso_history)*0.7, pso_history[-1]-0.02),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=11, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('figures/Q2_pso_convergence.png', dpi=200)
    plt.close()
    print_flush("  ✓ figures/Q2_pso_convergence.png")


# ============================================================
# Step 8: 保存结果
# ============================================================
def step8_save_results(results, importance_dict, best_params, pso_history, lag_results):
    """保存所有结果到JSON"""
    print_flush("\n" + "="*70)
    print_flush("Step 8: 保存结果")
    print_flush("="*70)
    
    os.makedirs('results', exist_ok=True)
    
    final_results = {
        'model_info': {
            'algorithm': 'XGBoost',
            'n_features': results['full_model']['n_samples'],
            'best_params': {k: float(v) if isinstance(v, (float, np.floating)) else int(v) 
                           for k, v in best_params.items()},
        },
        'pso_tuning': {
            'n_particles': PSO_PARTICLES,
            'n_iterations': PSO_ITERATIONS,
            'internal_cv_folds': CV_FOLDS,
            'best_cv_r2': round(float(pso_history[-1]), 4),
            'convergence_history': [round(float(h), 4) for h in pso_history]
        },
        'evaluation': {
            'split_7030': results['split_7030'],
            'cv_5fold': results['cv_5fold'],
            'full_model': results['full_model']
        },
        'feature_importance': importance_dict,
        'lag_results': {k: int(v) for k, v in lag_results.items()}
    }
    
    with open('results/Q2_complete_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    print_flush("  ✓ results/Q2_complete_results.json")


# ============================================================
# 主程序
# ============================================================
def main():
    """主程序：完整的问题2建模流程"""
    
    total_start = time.time()
    
    print_flush("="*70)
    print_flush("问题2：CO浓度预测模型 - 完整版（含PSO调参）")
    print_flush("="*70)
    
    # Step 1: 数据加载
    df = step1_load_data()
    
    # Step 2: 时滞对齐
    df_aligned, lag_results = step2_time_lag_alignment(df)
    
    # Step 3: 特征工程
    X, y, all_features = step3_feature_engineering(df_aligned)
    
    # Step 4: PSO超参数优化
    best_params, pso_history = step4_pso_tuning(X, y)
    
    # Step 5: 训练最终模型
    results, final_model, final_scaler = step5_train_and_evaluate(X, y, best_params)
    
    # Step 6: 特征重要性
    importance_dict, sorted_idx = step6_feature_importance(final_model, all_features)
    
    # Step 7: 可视化
    # 获取测试集数据
    split = int(len(X) * 0.7)
    X_test = X[split:]
    y_test = y[split:]
    X_test_s = final_scaler.transform(X_test)
    y_pred = final_model.predict(X_test_s)
    
    step7_visualize(results, y_test, y_pred, importance_dict, sorted_idx,
                    all_features, final_model, pso_history)
    
    # Step 8: 保存结果
    step8_save_results(results, importance_dict, best_params, pso_history, lag_results)
    
    # ============================================================
    # 最终总结
    # ============================================================
    total_time = time.time() - total_start
    
    print_flush("\n" + "="*70)
    print_flush("模型构建完成!")
    print_flush("="*70)
    
    print_flush(f"\n📊 最终结果:")
    print_flush(f"  PSO调参最优CV R²: {pso_history[-1]:.4f}")
    print_flush(f"  70/30 划分:")
    print_flush(f"    R²  = {results['split_7030']['r2']:.4f}")
    print_flush(f"    MAE = {results['split_7030']['mae']:.2f} mg/m³")
    print_flush(f"    RMSE= {results['split_7030']['rmse']:.2f} mg/m³")
    print_flush(f"  {FINAL_CV_FOLDS}折交叉验证:")
    print_flush(f"    平均 R² = {results['cv_5fold']['mean_r2']:.4f} ± {results['cv_5fold']['std_r2']:.4f}")
    
    print_flush(f"\n⏱️  总耗时: {total_time:.0f}s ({total_time/60:.1f}分钟)")
    
    print_flush(f"\n📁 输出文件:")
    print_flush(f"  results/Q2_complete_results.json")
    print_flush(f"  figures/Q2_scatter.png")
    print_flush(f"  figures/Q2_error_dist.png")
    print_flush(f"  figures/Q2_importance.png")
    print_flush(f"  figures/Q2_cv_results.png")
    print_flush(f"  figures/Q2_pso_convergence.png")


if __name__ == '__main__':
    main()
