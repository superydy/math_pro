#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q3 v4b: Pressure + Machine Speed Joint Optimization (19 variables)
  Decision variables: 18 wind-box pressures + 1 machine speed
  Bounds: 5%-95% quantiles (wider than Q3_improved's 10%-90%)
  Algorithm: Differential Evolution (best1bin) + reliability penalty
  Steady-state: honest — co_lag features fixed at historical CO median

Why add machine speed:
  Speed directly controls material residence time in each combustion zone.
  Slower speed → longer combustion → lower CO. This is a primary physical lever.
  Feature importance: speed contributes ~5.7% of total model importance.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.optimize import differential_evolution
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import json, os, time, warnings
warnings.filterwarnings('ignore')

BOUND_LO = 0.05
BOUND_HI = 0.95
DE_POPSIZE = 12      # 12×19 = 228 individuals
DE_MAXITER = 100     # ~22,800 evaluations total
PENALTY_ALPHA = 150

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

def pf(msg):
    print(msg, flush=True)


def compute_xcorr(x, y, max_lag=60):
    xn = (x - np.mean(x)) / (np.std(x) * len(x) + 1e-8)
    yn = (y - np.mean(y)) / (np.std(y) + 1e-8)
    corr = fftconvolve(xn[::-1], yn, mode='full')
    center = len(xn) - 1
    cc = corr[center - max_lag: center + max_lag + 1]
    lags = np.arange(-max_lag, max_lag + 1)
    return lags, cc


def main():
    total_t = time.time()
    pf("=" * 70)
    pf("Q3 v4b: Pressure + Speed Joint Optimization (19 vars)")
    pf(f"  Bounds: {BOUND_LO*100:.0f}%-{BOUND_HI*100:.0f}% quantiles")
    pf(f"  DE: popsize={DE_POPSIZE}×19, maxiter={DE_MAXITER}")
    pf("=" * 70)

    # ── Data preparation ────────────────────────────────────
    pf("\nStep 1: Load data & train model")
    df = pd.read_csv('data/processed_data.csv')
    abnormal = list(range(1045, 1062))
    df = df[~df.index.isin(abnormal)].reset_index(drop=True)

    co = df['CO浓度'].values
    var_list = ['机速']
    for i in range(1, 19): var_list.extend([f'负压_{i}', f'温度_{i}'])
    var_list.extend(['大烟道负压_1', '大烟道负压_2', '大烟道温度_1', '大烟道温度_2'])

    lag_results = {}
    for var in var_list:
        lags, cc = compute_xcorr(df[var].values, co, 60)
        lag_results[var] = int(lags[np.argmax(np.abs(cc))])

    df_al = df.copy()
    for var, lag in lag_results.items():
        df_al[f'{var}_al'] = df_al[var].shift(-lag) if lag != 0 else df_al[var]

    al_cols = [f'{var}_al' for var in var_list]
    df_al = df_al.dropna(subset=al_cols + ['CO浓度']).reset_index(drop=True)

    physical = ['机速_al']
    for i in range(1, 19): physical.extend([f'负压_{i}_al', f'温度_{i}_al'])
    physical.extend(['大烟道负压_1_al','大烟道负压_2_al','大烟道温度_1_al','大烟道温度_2_al'])

    grad = []
    for i in range(1, 18):
        df_al[f'pgrad_{i}'] = df_al[f'负压_{i+1}_al'] - df_al[f'负压_{i}_al']
        df_al[f'tgrad_{i}'] = df_al[f'温度_{i+1}_al'] - df_al[f'温度_{i}_al']
        grad.extend([f'pgrad_{i}', f'tgrad_{i}'])

    pcols = [f'负压_{i}_al' for i in range(1, 19)]
    df_al['p_mean'] = df_al[pcols].mean(axis=1)
    df_al['p_mid']  = df_al[[f'负压_{i}_al' for i in range(6,13)]].mean(axis=1)
    df_al['p_back'] = df_al[[f'负压_{i}_al' for i in range(13,19)]].mean(axis=1)
    df_al['t_back'] = df_al[[f'温度_{i}_al' for i in range(13,19)]].mean(axis=1)
    stat = ['p_mean','p_mid','p_back','t_back']

    co_feat = ['co_lag1','co_lag2','co_lag5','co_ma5','co_diff1']
    df_al['co_lag1']  = df_al['CO浓度'].shift(1)
    df_al['co_lag2']  = df_al['CO浓度'].shift(2)
    df_al['co_lag5']  = df_al['CO浓度'].shift(5)
    df_al['co_ma5']   = df_al['CO浓度'].rolling(5).mean()
    df_al['co_diff1'] = df_al['CO浓度'].diff(1)

    all_features = physical + grad + stat + co_feat
    df_f = df_al.dropna(subset=all_features + ['CO浓度']).reset_index(drop=True)

    X = df_f[all_features].values
    y = df_f['CO浓度'].values

    bp = dict(learning_rate=0.2, max_depth=3, n_estimators=262,
              subsample=0.51, colsample_bytree=0.99,
              reg_alpha=0.88, reg_lambda=1.76, random_state=42, n_jobs=-1)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = XGBRegressor(**bp)
    model.fit(X_s, y, verbose=False)
    pf(f"  R^2 (full): {r2_score(y, model.predict(X_s)):.4f}")

    co_init = float(df_f['CO浓度'].median())
    pf(f"  co_init (median): {co_init:.1f} mg/m3")

    # Fixed values (medians for everything except decision vars)
    fixed_t  = {f'温度_{i}_al': float(df_f[f'温度_{i}_al'].median()) for i in range(1, 19)}
    fixed_t.update({nm: float(df_f[nm].median())
                    for nm in ['大烟道负压_1_al','大烟道负压_2_al','大烟道温度_1_al','大烟道温度_2_al']})
    fixed_tg = {f'tgrad_{i}': float(df_f[f'tgrad_{i}'].median()) for i in range(1, 18)}
    fixed_tb  = float(df_f['t_back'].median())

    feat_idx = {nm: i for i, nm in enumerate(all_features)}
    n_feat   = len(all_features)

    # ── Bounds ──────────────────────────────────────────────
    pf("\nStep 2: Build bounds (18 pressures + 1 speed)")
    bounds_list = []
    pressure_medians = []

    for i in range(1, 19):
        col = f'负压_{i}_al'
        lo = float(df_f[col].quantile(BOUND_LO))
        hi = float(df_f[col].quantile(BOUND_HI))
        bounds_list.append((lo, hi))
        pressure_medians.append(float(df_f[col].median()))

    speed_col = '机速_al'
    s_lo = float(df_f[speed_col].quantile(BOUND_LO))
    s_hi = float(df_f[speed_col].quantile(BOUND_HI))
    s_med = float(df_f[speed_col].median())
    bounds_list.append((s_lo, s_hi))

    pf(f"  Speed: [{s_lo:.3f}, {s_hi:.3f}]  median={s_med:.3f}")
    pf(f"  Pressure width range: {min(b[1]-b[0] for b in bounds_list[:-1]):.2f} ~ "
       f"{max(b[1]-b[0] for b in bounds_list[:-1]):.2f} KPa")

    # ── Predictor ───────────────────────────────────────────
    def predict_steady(x):
        pressures = x[:18]
        speed     = x[18]

        fv = np.zeros(n_feat)
        fv[feat_idx['机速_al']] = speed
        for i in range(18):
            fv[feat_idx[f'负压_{i+1}_al']] = pressures[i]
        for k, v in fixed_t.items():
            fv[feat_idx[k]] = v
        for i in range(17):
            fv[feat_idx[f'pgrad_{i+1}']] = pressures[i+1] - pressures[i]
        for k, v in fixed_tg.items():
            fv[feat_idx[k]] = v
        fv[feat_idx['p_mean']] = np.mean(pressures)
        fv[feat_idx['p_mid']]  = np.mean(pressures[5:12])
        fv[feat_idx['p_back']] = np.mean(pressures[12:18])
        fv[feat_idx['t_back']] = fixed_tb

        X_g = co_init
        for _ in range(20):
            fv[feat_idx['co_lag1']]  = X_g
            fv[feat_idx['co_lag2']]  = X_g
            fv[feat_idx['co_lag5']]  = X_g
            fv[feat_idx['co_ma5']]   = X_g
            fv[feat_idx['co_diff1']] = 0.0
            fv_s = scaler.transform(fv.reshape(1, -1))
            X_new = float(model.predict(fv_s)[0])
            if abs(X_new - X_g) < 1.0:
                return X_new
            X_g = 0.5 * X_g + 0.5 * X_new
        return X_g

    def penalty(x):
        pen = 0.0
        for i, (lo, hi) in enumerate(bounds_list):
            rng = hi - lo
            if rng < 1e-6: continue
            d = min(x[i] - lo, hi - x[i]) / rng
            pen += np.exp(-10 * d)
        return pen

    # ── Current operating point ─────────────────────────────
    curr_x  = np.array(pressure_medians + [s_med])
    curr_co = predict_steady(curr_x)
    pf(f"\n  Current operating CO: {curr_co:.1f} mg/m3")

    # ── DE Optimization ────────────────────────────────────
    pf(f"\nStep 3: Differential Evolution (19 vars)")
    pf(f"  Population: {DE_POPSIZE}×19 = {DE_POPSIZE*19} individuals")
    pf(f"  Max evaluations: {DE_POPSIZE*19*(DE_MAXITER+1):,}")

    eval_n = [0]
    best_co = [curr_co]
    t0 = time.time()

    def objective(x):
        co  = predict_steady(x)
        pen_v = penalty(x)
        obj = co + PENALTY_ALPHA * pen_v
        eval_n[0] += 1
        if co < best_co[0]:
            best_co[0] = co
            pf(f"    [{eval_n[0]:5d}]  CO={co:.1f}  pen={pen_v:.2f}  speed={x[18]:.3f}  t={time.time()-t0:.0f}s")
        return obj

    result = differential_evolution(
        objective, bounds_list,
        strategy='best1bin',
        maxiter=DE_MAXITER,
        popsize=DE_POPSIZE,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        polish=True,
        workers=1,
        updating='deferred',
    )
    de_elapsed = time.time() - t0

    opt_x   = result.x
    opt_co  = predict_steady(opt_x)
    opt_pen = penalty(opt_x)
    opt_speed = opt_x[18]
    opt_p   = opt_x[:18]
    reduction = (curr_co - opt_co) / curr_co * 100

    pf(f"\n  DE done in {de_elapsed:.0f}s ({de_elapsed/60:.1f} min)")
    pf(f"  Optimal CO:    {opt_co:.1f} mg/m3")
    pf(f"  Reduction:     {reduction:.1f}%")
    pf(f"  Optimal speed: {s_med:.3f} → {opt_speed:.3f} (Δ={opt_speed-s_med:+.3f})")
    pf(f"  Meets 2800:    {'YES' if opt_co<=2800 else 'NO'}")
    bcount = sum(1 for i,(lo,hi) in enumerate(bounds_list)
                 if abs(opt_x[i]-lo) < 0.05*(hi-lo) or abs(opt_x[i]-hi) < 0.05*(hi-lo))
    pf(f"  Near boundary (5%): {bcount}/19")

    # ── Comparison ──────────────────────────────────────────
    pressure_only_co = 2741.8
    pf(f"\nStep 4: Comparison")
    pf(f"  {'Method':35s} {'CO':>10s} {'Reduction':>12s}")
    pf("  " + "-" * 62)
    pf(f"  {'Current (medians)':35s} {curr_co:10.1f} {'baseline':>12s}")
    pf(f"  {'Pressure-only PSO (v3)':35s} {pressure_only_co:10.1f} {((curr_co-pressure_only_co)/curr_co*100):11.1f}%")
    pf(f"  {'Pressure+Speed DE (v4b)':35s} {opt_co:10.1f} {reduction:11.1f}%")

    # ── Breakdown ───────────────────────────────────────────
    pf(f"\nStep 5: Optimal pressure adjustments")
    pf(f"  {'Zone':>6s}  {'Current':>10s} {'Optimal':>10s} {'Delta':>10s}")
    pf("  " + "-" * 42)
    for i in range(18):
        pf(f"  {i+1:2d}#    {pressure_medians[i]:10.2f} {opt_p[i]:10.2f} {opt_p[i]-pressure_medians[i]:+10.2f}")
    pf(f"  Speed   {s_med:10.3f} {opt_speed:10.3f} {opt_speed-s_med:+10.3f}")

    # ── Figures ─────────────────────────────────────────────
    pf(f"\nStep 6: Figures")

    # Fig 1: CO comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ['Current\nOperating', 'Pressure-only\nPSO (v3)', 'Pressure+Speed\nDE (v4b)']
    vals   = [curr_co, pressure_only_co, opt_co]
    cols   = ['#95a5a6', '#e67e22', '#2980b9']
    bars = ax.bar(labels, vals, color=cols, alpha=0.85, width=0.5)
    ax.axhline(2800, color='red', ls='--', lw=2, label='Emission standard: 2800 mg/m³')
    ax.set_ylabel('Steady-state CO (mg/m³)', fontsize=12)
    ax.set_title('Q3: CO Reduction Comparison', fontsize=14)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, vals):
        r = (curr_co - v) / curr_co * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 40,
                f'{v:.0f}\n({r:.1f}%↓)', ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/Q3v4b_co_comparison.png', dpi=200); plt.close()
    pf("  Q3v4b_co_comparison.png")

    # Fig 2: Pressure adjustments + speed
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [4, 1]})
    ax = axes[0]
    x = np.arange(1, 19)
    p_delta = [opt_p[i] - pressure_medians[i] for i in range(18)]
    cc = ['#e74c3c' if d < -0.05 else '#2ecc71' if d > 0.05 else '#95a5a6' for d in p_delta]
    bars2 = ax.bar(x, p_delta, color=cc, alpha=0.8)
    ax.axhline(0, color='black', lw=1)
    ax.set_xlabel('Bellows Number', fontsize=11)
    ax.set_ylabel('Pressure Change (KPa)', fontsize=11)
    ax.set_title('Optimal Pressure Adjustments (v4b)', fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels([f'{i}#' for i in x])
    ax.grid(True, alpha=0.3, axis='y')
    for bar2, d in zip(bars2, p_delta):
        if abs(d) > 0.1:
            ax.text(bar2.get_x() + bar2.get_width()/2,
                    bar2.get_height() + (0.04 if d >= 0 else -0.15),
                    f'{d:+.2f}', ha='center', fontsize=8)

    ax = axes[1]
    s_delta = opt_speed - s_med
    bar_col = '#e74c3c' if s_delta < 0 else '#2ecc71'
    ax.bar(['Speed'], [s_delta], color=bar_col, alpha=0.8)
    ax.axhline(0, color='black', lw=1)
    ax.set_ylabel('Speed Change (m/min)', fontsize=11)
    ax.set_title('Speed', fontsize=12)
    ax.text(0, s_delta + (0.001 if s_delta >= 0 else -0.003),
            f'{s_delta:+.3f}', ha='center', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('figures/Q3v4b_adjustments.png', dpi=200); plt.close()
    pf("  Q3v4b_adjustments.png")

    # Fig 3: Boundary distance
    fig, ax = plt.subplots(figsize=(14, 5))
    dists = []
    for i, (lo, hi) in enumerate(bounds_list):
        rng = hi - lo
        d = min(opt_x[i] - lo, hi - opt_x[i]) / rng if rng > 1e-6 else 0.5
        dists.append(d)
    labels_x = [f'P{i+1}' for i in range(18)] + ['Speed']
    cc2 = ['#F44336' if d < 0.1 else '#FF9800' if d < 0.2 else '#4CAF50' for d in dists]
    ax.bar(range(19), dists, color=cc2, alpha=0.8)
    ax.axhline(0.1, color='red', ls='--', lw=1, label='Warning: 10%')
    ax.axhline(0.2, color='orange', ls='--', lw=1, label='Caution: 20%')
    ax.set_xticks(range(19)); ax.set_xticklabels(labels_x, rotation=45, fontsize=9)
    ax.set_ylabel('Normalized distance to boundary', fontsize=11)
    ax.set_title('v4b Reliability: Boundary Distance (19 variables)', fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim(0, 0.6)
    plt.tight_layout()
    plt.savefig('figures/Q3v4b_boundary_dist.png', dpi=200); plt.close()
    pf("  Q3v4b_boundary_dist.png")

    # ── Save ────────────────────────────────────────────────
    pf(f"\nStep 7: Save results")
    results = {
        'method': 'Pressure+Speed DE (19 vars, 5%-95% bounds, honest steady-state)',
        'current_co': float(curr_co),
        'pressure_only_co': pressure_only_co,
        'joint_de_co': float(opt_co),
        'reduction_pct': float(reduction),
        'meets_2800': bool(opt_co <= 2800),
        'penalty': float(opt_pen),
        'vars_near_boundary': int(bcount),
        'optimal_speed': float(opt_speed),
        'speed_change': float(opt_speed - s_med),
        'speed_pct': (s_med - opt_speed) / s_med * 100 if s_med > 0 else 0,
        'optimal_pressures': {f'bellows_{i+1}': float(opt_p[i]) for i in range(18)},
        'current_pressures': {f'bellows_{i+1}': float(pressure_medians[i]) for i in range(18)},
        'pressure_ranges': {f'bellows_{i+1}': [float(bounds_list[i][0]),
                                                float(bounds_list[i][1])] for i in range(18)},
    }
    with open('results/Q3_v4b_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    pf("  results/Q3_v4b_results.json")

    rows = []
    for i in range(18):
        lo, hi = bounds_list[i]
        d = min(opt_p[i]-lo, hi-opt_p[i])/(hi-lo)
        rows.append({
            'Bellows': f'{i+1}#',
            'Current_P': round(pressure_medians[i], 3),
            'Optimal_P': round(float(opt_p[i]), 3),
            'Delta_P': round(float(opt_p[i]) - pressure_medians[i], 3),
            'Lower': round(lo, 3),
            'Upper': round(hi, 3),
            'BoundaryDist': round(d, 3),
        })
    rows.append({'Bellows': 'Speed', 'Current_P': round(s_med, 4),
                 'Optimal_P': round(float(opt_speed), 4),
                 'Delta_P': round(float(opt_speed)-s_med, 4),
                 'Lower': round(s_lo, 4), 'Upper': round(s_hi, 4),
                 'BoundaryDist': round(min(opt_speed-s_lo, s_hi-opt_speed)/(s_hi-s_lo), 3)})
    pd.DataFrame(rows).to_csv('results/Q3_v4b_decisions.csv', index=False, encoding='utf-8-sig')
    pf("  results/Q3_v4b_decisions.csv")

    elapsed = time.time() - total_t
    pf(f"\n{'='*70}")
    pf(f"Q3 v4b done!  Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    pf(f"RESULT: {curr_co:.0f} → {opt_co:.0f} mg/m3  =  {reduction:.1f}% reduction")
    pf(f"{'='*70}")


if __name__ == '__main__':
    main()
