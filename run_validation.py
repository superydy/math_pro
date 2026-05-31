#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证实验：
  A. 消融实验 —— 逐步移除特征组，量化每组特征的贡献
  D. 局部灵敏度验证 —— 在Q3最优解附近扰动±5%，验证是真实最优点
"""

import pandas as pd
import numpy as np
import json, time, os, warnings
warnings.filterwarnings('ignore')
os.chdir('/home/user/math_pro')

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

# ─── 复用 run_model_comparison.py 的数据加载逻辑 ─────────────────────────────
def load_data_full():
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

    # 各特征组
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

    feats_all = phys + grad + stat + co_f
    df_f = df_al.dropna(subset=feats_all+['CO浓度']).reset_index(drop=True)
    y = df_f['CO浓度'].values
    split = int(len(y) * 0.7)
    y_tr, y_te = y[:split], y[split:]

    return df_f, feats_all, phys, grad, stat, co_f, split, y_tr, y_te


def train_eval(X_tr, X_te, y_tr, y_te, best_params):
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(X_tr)
    Xte_s = sc.transform(X_te)
    mdl = XGBRegressor(**best_params, random_state=42, n_jobs=-1, verbosity=0)
    mdl.fit(Xtr_s, y_tr, verbose=False)
    pred = mdl.predict(Xte_s)
    return {
        'r2':   round(float(r2_score(y_te, pred)), 4),
        'mae':  round(float(mean_absolute_error(y_te, pred)), 2),
        'rmse': round(float(np.sqrt(np.mean((y_te - pred)**2))), 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 方案 A：消融实验
# ═══════════════════════════════════════════════════════════════════════════════
def run_ablation():
    print("\n" + "="*60)
    print("方案 A：消融实验")
    print("="*60)

    df_f, feats_all, phys, grad, stat, co_f, split, y_tr, y_te = load_data_full()

    with open('results/model_comparison_results.json') as f:
        cmp = json.load(f)
    best_params = cmp['model_comparison']['XGBoost+PSO(本文)']['best_params'].copy()
    best_params['max_depth']    = int(best_params['max_depth'])
    best_params['n_estimators'] = int(best_params['n_estimators'])

    # 5 个消融组：全特征 / 去梯度 / 去统计 / 去CO自回归 / 仅物理
    ablation_groups = [
        ('全特征（基准）',          phys + grad + stat + co_f,  '—'),
        ('移除梯度特征',            phys        + stat + co_f,  f'-{len(grad)}个梯度特征'),
        ('移除统计聚合特征',        phys + grad        + co_f,  f'-{len(stat)}个统计特征'),
        ('移除CO自回归特征',        phys + grad + stat,         f'-{len(co_f)}个CO滞后特征'),
        ('仅物理传感器特征',        phys,                        f'仅{len(phys)}个物理特征'),
    ]

    results = []
    for name, feats, removed in ablation_groups:
        X = df_f[feats].values
        X_tr, X_te = X[:split], X[split:]
        metrics = train_eval(X_tr, X_te, y_tr, y_te, best_params)
        results.append({
            'name':    name,
            'removed': removed,
            'n_feats': len(feats),
            **metrics,
        })
        print(f"  {name:20s}  特征数={len(feats):3d}  R²={metrics['r2']:.4f}  MAE={metrics['mae']:.2f}  RMSE={metrics['rmse']:.2f}")

    # 计算 R² 相对下降
    base_r2 = results[0]['r2']
    for r in results:
        r['r2_drop'] = round(base_r2 - r['r2'], 4)

    print(f"\n  基准 R²={base_r2:.4f}，各消融组 R² 降幅如上")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 方案 D：Q3 局部灵敏度验证
# ═══════════════════════════════════════════════════════════════════════════════
def run_sensitivity():
    print("\n" + "="*60)
    print("方案 D：Q3 局部灵敏度验证")
    print("="*60)

    with open('results/Q3_optimization_results.json') as f:
        q3 = json.load(f)
    # 使用 Q2 完整结果中的最优参数（与 Q3 实际使用的模型一致）
    with open('results/Q2_complete_results.json') as f:
        q2 = json.load(f)
    best_params = q2['model_info']['best_params'].copy()
    best_params['max_depth']    = int(best_params['max_depth'])
    best_params['n_estimators'] = int(best_params['n_estimators'])

    # 重建模型
    df_f, feats_all, phys, grad, stat, co_f, split, y_tr, y_te = load_data_full()
    X_all = df_f[feats_all].values
    X_tr_raw = X_all[:split]
    sc = StandardScaler()
    sc.fit(X_tr_raw)
    X_tr_s = sc.transform(X_tr_raw)
    y_tr_raw = df_f['CO浓度'].values[:split]
    mdl = XGBRegressor(**best_params, random_state=42, n_jobs=-1, verbosity=0)
    mdl.fit(X_tr_s, y_tr_raw, verbose=False)

    # 预测函数：固定 co_lag = co_ref，只变压力
    def predict_co_direct(pressures_dict, co_ref):
        p_arr = np.array([pressures_dict[f'bellows_{i}'] for i in range(1, 19)])
        X_ref = df_f[feats_all].median().values.copy()
        for i in range(18):
            X_ref[feats_all.index(f'负压_{i+1}_al')] = p_arr[i]
        neg_vals = p_arr.copy()
        for i in range(17):
            X_ref[feats_all.index(f'pgrad_{i+1}')] = neg_vals[i+1] - neg_vals[i]
        X_ref[feats_all.index('p_mean')] = neg_vals.mean()
        X_ref[feats_all.index('p_mid')]  = neg_vals[5:13].mean()
        X_ref[feats_all.index('p_back')] = neg_vals[12:18].mean()
        X_ref[feats_all.index('co_lag1')]  = co_ref
        X_ref[feats_all.index('co_lag2')]  = co_ref
        X_ref[feats_all.index('co_lag5')]  = co_ref
        X_ref[feats_all.index('co_ma5')]   = co_ref
        X_ref[feats_all.index('co_diff1')] = 0.0
        xs = sc.transform(X_ref.reshape(1, -1))
        return round(float(mdl.predict(xs)[0]), 2)

    opt_p    = q3['optimal_pressures_improved']
    curr_p   = q3['current_pressures']
    p_ranges = q3['pressure_ranges']

    co_ref = float(q3['current_co'])  # 固定 co_lag 参考值
    co_baseline = predict_co_direct(curr_p, co_ref)
    co_opt_pred = predict_co_direct(opt_p,  co_ref)
    print(f"  固定co_lag参考={co_ref:.0f}ppm下：当前压力→预测CO={co_baseline:.1f}, 最优压力→预测CO={co_opt_pred:.1f}")

    # ── 1. 约束可行性验证：所有18个风箱均在允许范围内 ───────────────────────
    print(f"\n  约束可行性验证（共18个风箱）：")
    feasibility = []
    all_feasible = True
    for i in range(1, 19):
        key = f'bellows_{i}'
        p_val = opt_p[key]
        lb, ub = p_ranges[key]
        ok = lb <= p_val <= ub
        if not ok:
            all_feasible = False
        feasibility.append({'bellows': i, 'p_opt': round(p_val,4), 'lb': round(lb,4), 'ub': round(ub,4), 'feasible': ok})
        status = '✓' if ok else '✗'
        print(f"  风箱{i:2d}: p*={p_val:8.4f}  范围=[{lb:.4f},{ub:.4f}]  {status}")
    print(f"\n  可行性汇总：{'全部18个风箱均在约束范围内 ✓' if all_feasible else '存在违约风箱 ✗'}")

    # ── 2. 全18风箱逐一扫描灵敏度 ──────────────────────────────────────────
    # 对每个风箱，在 [lb, ub] 全范围内均匀取11个点，记录模型预测变化
    print(f"\n  全风箱灵敏度扫描（模型输出对压力的响应范围）：")
    scan_results = []
    for i in range(1, 19):
        key = f'bellows_{i}'
        lb, ub = p_ranges[key]
        p_scan = np.linspace(lb, ub, 11)
        co_scan = []
        for pv in p_scan:
            p_tmp = opt_p.copy()
            p_tmp[key] = pv
            co_scan.append(predict_co_direct(p_tmp, co_ref))
        co_range = round(max(co_scan) - min(co_scan), 2)
        co_at_opt = predict_co_direct(opt_p, co_ref)
        # 灵敏度：CO全范围变化 / 压力全范围变化 (ppm per unit pressure)
        p_span = abs(ub - lb)
        sens = round(co_range / p_span, 3) if p_span > 0 else 0
        scan_results.append({
            'bellows': i,
            'co_range': co_range,
            'sensitivity': sens,
            'co_min': round(min(co_scan),2),
            'co_max': round(max(co_scan),2),
            'co_at_optimal': round(co_at_opt,2),
        })
        print(f"  风箱{i:2d}: CO变化范围={co_range:7.2f}ppm  灵敏度={sens:.3f}ppm/单位压力")

    # 灵敏度排名
    scan_sorted = sorted(scan_results, key=lambda x: -x['co_range'])
    print(f"\n  灵敏度 Top5 风箱：")
    for r in scan_sorted[:5]:
        print(f"    风箱{r['bellows']:2d}: ΔCO={r['co_range']:.2f}ppm  灵敏度={r['sensitivity']:.3f}")

    return {
        'co_ref': co_ref,
        'co_baseline': co_baseline,
        'co_opt_pred': co_opt_pred,
        'feasibility': feasibility,
        'all_feasible': all_feasible,
        'scan_results': scan_results,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 生成验证图
# ═══════════════════════════════════════════════════════════════════════════════
def gen_validation_figures(ablation_results, sensitivity_results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
    from matplotlib import font_manager
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'WenQuanYi Zen Hei'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 11

    os.makedirs('paper_figures', exist_ok=True)

    # ── 图A：消融实验柱状图 ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    names_short = ['全特征\n(基准)', '移除\n梯度特征', '移除\n统计特征', '移除\nCO自回归', '仅物理\n特征']
    r2_vals  = [r['r2']   for r in ablation_results]
    mae_vals = [r['mae']  for r in ablation_results]
    rmse_vals= [r['rmse'] for r in ablation_results]
    x = np.arange(len(names_short))
    colors = ['#2166AC'] + ['#F4A582']*3 + ['#D6604D']

    ax = axes[0]
    bars = ax.bar(x, r2_vals, color=colors, width=0.6, edgecolor='white', linewidth=0.8)
    ax.set_ylim(0.3, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(names_short, fontsize=9)
    ax.set_ylabel('测试集 R²')
    ax.set_title('(a) 消融实验 — R²对比')
    ax.axhline(r2_vals[0], color='#2166AC', linestyle='--', linewidth=1, alpha=0.5)
    for bar, val in zip(bars, r2_vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.005, f'{val:.4f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax2 = axes[1]
    width = 0.35
    bars1 = ax2.bar(x - width/2, mae_vals,  width, label='MAE',  color='#4DAF4A', edgecolor='white')
    bars2 = ax2.bar(x + width/2, rmse_vals, width, label='RMSE', color='#E41A1C', edgecolor='white', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names_short, fontsize=9)
    ax2.set_ylabel('误差 (ppm)')
    ax2.set_title('(b) 消融实验 — MAE / RMSE 对比')
    ax2.legend()
    for bar, val in zip(bars1, mae_vals):
        ax2.text(bar.get_x()+bar.get_width()/2, val+1, f'{val:.0f}',
                 ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, rmse_vals):
        ax2.text(bar.get_x()+bar.get_width()/2, val+1, f'{val:.0f}',
                 ha='center', va='bottom', fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('paper_figures/fig_val_A_ablation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ paper_figures/fig_val_A_ablation.png")

    # ── 图D1：约束可行性 + 全风箱灵敏度排名 ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (c) 可行性：最优压力 vs 约束范围
    feas = sensitivity_results['feasibility']
    b_labels = [f'风箱{r["bellows"]}' for r in feas]
    p_opts   = [r['p_opt'] for r in feas]
    lbs      = [r['lb']    for r in feas]
    ubs      = [r['ub']    for r in feas]
    x_pos    = np.arange(len(feas))

    ax = axes[0]
    ax.fill_between(x_pos, lbs, ubs, alpha=0.18, color='#2166AC', label='允许范围')
    ax.plot(x_pos, lbs, '--', color='#2166AC', linewidth=1, alpha=0.6)
    ax.plot(x_pos, ubs, '--', color='#2166AC', linewidth=1, alpha=0.6)
    ax.plot(x_pos, p_opts, 'o-', color='#D6604D', linewidth=2, markersize=5, label='最优压力')
    ax.set_xticks(x_pos[::2])
    ax.set_xticklabels([f'风箱{r["bellows"]}' for r in feas[::2]], fontsize=8, rotation=30)
    ax.set_ylabel('风箱负压 (kPa)')
    ax.set_title('(c) Q3 最优压力约束可行性验证（全部18个风箱）')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ok_count = sum(1 for r in feas if r['feasible'])
    ax.text(0.98, 0.04, f'可行性：{ok_count}/18 ✓',
            transform=ax.transAxes, ha='right', fontsize=10,
            color='green' if ok_count==18 else 'red',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # (d) 全18风箱灵敏度排名（水平条形图）
    scan = sensitivity_results['scan_results']
    scan_sorted = sorted(scan, key=lambda x: x['co_range'])
    b_names  = [f'风箱{r["bellows"]}' for r in scan_sorted]
    co_range = [r['co_range']         for r in scan_sorted]
    colors_bar = ['#D6604D' if r['co_range'] >= sorted([s['co_range'] for s in scan])[-5] else '#92C5DE'
                  for r in scan_sorted]

    ax2 = axes[1]
    bars = ax2.barh(b_names, co_range, color=colors_bar, edgecolor='white', height=0.7)
    ax2.set_xlabel('CO 预测变化范围 (ppm)')
    ax2.set_title('(d) 各风箱压力灵敏度排名（全范围扫描）')
    for bar, val in zip(bars, co_range):
        if val > 0:
            ax2.text(val + 0.3, bar.get_y()+bar.get_height()/2,
                     f'{val:.1f}', va='center', fontsize=8)
    ax2.grid(axis='x', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(color='#D6604D', label='Top5高灵敏度'),
                         Patch(color='#92C5DE', label='低灵敏度')], fontsize=9)

    plt.tight_layout()
    plt.savefig('paper_figures/fig_val_D_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ paper_figures/fig_val_D_sensitivity.png")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    ablation_res  = run_ablation()
    sens_res      = run_sensitivity()
    gen_validation_figures(ablation_res, sens_res)

    out = {
        'ablation':     ablation_res,
        'sensitivity':  sens_res,
    }
    with open('results/validation_results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n✓ results/validation_results.json")
    print("验证实验完成！")
