#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q3 v4: Constrained Joint Optimization — 37 variables
  Decision variables:
    - 18 wind-box pressures (5%-95% quantile bounds, wide range)
    - 18 zone temperatures  (25%-75% quantile bounds, IQR only — operationally achievable)
    - 1 machine speed       (5%-95% quantile bounds)
  Algorithm: Differential Evolution (best1bin) + reliability penalty
  Steady-state: honest — co_lag features fixed at historical CO median

Rationale for tight temperature bounds (25%-75%):
  Temperature in each zone genuinely varies by ±25-38°C (IQR) in real operation.
  Using the IQR ensures optimized temperatures stay within observed operational space,
  avoiding extrapolation to unrealistic physical states.
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

# Bounds configuration
P_LO, P_HI = 0.05, 0.95   # pressures: wide range (5%-95%)
T_LO, T_HI = 0.25, 0.75   # temperatures: IQR (25%-75%) — operationally achievable
S_LO, S_HI = 0.05, 0.95   # speed: full range

# DE configuration
DE_POPSIZE = 10      # 10×37 = 370 individuals
DE_MAXITER = 60      # 370 × 60 = 22,200 evaluations
PENALTY_ALPHA = 50   # lower alpha since temperature bounds already tight

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


def train_model(df_f):
    """Train XGBoost on full 84-feature set."""
    best_params = dict(learning_rate=0.2, max_depth=3, n_estimators=262,
                       subsample=0.51, colsample_bytree=0.99,
                       reg_alpha=0.88, reg_lambda=1.76,
                       random_state=42, n_jobs=-1)
    scaler = StandardScaler()
    X = df_f['_X'].values if '_X' in df_f else None
    raise NotImplementedError("call prepare_data first")


def prepare_data():
    df = pd.read_csv('data/processed_data.csv')
    abnormal = list(range(1045, 1062))
    df = df[~df.index.isin(abnormal)].reset_index(drop=True)

    co = df['CO浓度'].values
    var_list = ['机速']
    for i in range(1, 19):
        var_list.extend([f'负压_{i}', f'温度_{i}'])
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
    for i in range(1, 19):
        physical.extend([f'负压_{i}_al', f'温度_{i}_al'])
    physical.extend(['大烟道负压_1_al', '大烟道负压_2_al',
                     '大烟道温度_1_al', '大烟道温度_2_al'])

    grad = []
    for i in range(1, 18):
        df_al[f'pgrad_{i}'] = df_al[f'负压_{i+1}_al'] - df_al[f'负压_{i}_al']
        df_al[f'tgrad_{i}'] = df_al[f'温度_{i+1}_al'] - df_al[f'温度_{i}_al']
        grad.extend([f'pgrad_{i}', f'tgrad_{i}'])

    pcols = [f'负压_{i}_al' for i in range(1, 19)]
    df_al['p_mean'] = df_al[pcols].mean(axis=1)
    df_al['p_mid']  = df_al[[f'负压_{i}_al' for i in range(6, 13)]].mean(axis=1)
    df_al['p_back'] = df_al[[f'负压_{i}_al' for i in range(13, 19)]].mean(axis=1)
    df_al['t_back'] = df_al[[f'温度_{i}_al' for i in range(13, 19)]].mean(axis=1)
    stat = ['p_mean', 'p_mid', 'p_back', 't_back']

    co_feat = ['co_lag1', 'co_lag2', 'co_lag5', 'co_ma5', 'co_diff1']
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
    pf(f"  Model R^2 (full): {r2_score(y, model.predict(X_s)):.4f}")

    co_init = float(df_f['CO浓度'].median())
    pf(f"  Honest co_init (median): {co_init:.1f} mg/m3")

    fixed_vals = {}
    for nm in ['大烟道负压_1_al', '大烟道负压_2_al', '大烟道温度_1_al', '大烟道温度_2_al']:
        fixed_vals[nm] = float(df_f[nm].median())

    feat_idx = {nm: i for i, nm in enumerate(all_features)}
    n_feat = len(all_features)

    return model, scaler, feat_idx, n_feat, fixed_vals, co_init, df_f, all_features


def build_bounds(df_f):
    """Build 37-variable bounds: 18P (5-95%) + 18T (25-75%) + 1 speed (5-95%)."""
    bounds = []
    info = []

    for i in range(1, 19):
        col = f'负压_{i}_al'
        lo = float(df_f[col].quantile(P_LO))
        hi = float(df_f[col].quantile(P_HI))
        bounds.append((lo, hi))
        info.append(('P', i, lo, hi, float(df_f[col].median())))

    for i in range(1, 19):
        col = f'温度_{i}_al'
        lo = float(df_f[col].quantile(T_LO))
        hi = float(df_f[col].quantile(T_HI))
        bounds.append((lo, hi))
        info.append(('T', i, lo, hi, float(df_f[col].median())))

    col = '机速_al'
    lo = float(df_f[col].quantile(S_LO))
    hi = float(df_f[col].quantile(S_HI))
    bounds.append((lo, hi))
    info.append(('S', 0, lo, hi, float(df_f[col].median())))

    return bounds, info


def make_predictor(model, scaler, feat_idx, n_feat, fixed_vals, co_init):
    """Return predict_steady(x) and reliability_penalty(x, bounds) functions."""

    def predict_steady(x):
        pressures = x[:18]
        temps     = x[18:36]
        speed     = x[36]

        fv = np.zeros(n_feat)
        fv[feat_idx['机速_al']] = speed
        for i in range(18):
            fv[feat_idx[f'负压_{i+1}_al']] = pressures[i]
            fv[feat_idx[f'温度_{i+1}_al']] = temps[i]
        for nm, v in fixed_vals.items():
            fv[feat_idx[nm]] = v
        for i in range(17):
            fv[feat_idx[f'pgrad_{i+1}']] = pressures[i+1] - pressures[i]
            fv[feat_idx[f'tgrad_{i+1}']] = temps[i+1] - temps[i]
        fv[feat_idx['p_mean']] = np.mean(pressures)
        fv[feat_idx['p_mid']]  = np.mean(pressures[5:12])
        fv[feat_idx['p_back']] = np.mean(pressures[12:18])
        fv[feat_idx['t_back']] = np.mean(temps[12:18])

        # Honest steady-state: initialize from historical median
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

    return predict_steady


def main():
    total_t = time.time()

    pf("=" * 70)
    pf("Q3 v4: Constrained Joint Optimization (37 variables)")
    pf(f"  Pressures:    {P_LO*100:.0f}%-{P_HI*100:.0f}%  (wide — full operational range)")
    pf(f"  Temperatures: {T_LO*100:.0f}%-{T_HI*100:.0f}%  (IQR — operationally achievable)")
    pf(f"  Speed:        {S_LO*100:.0f}%-{S_HI*100:.0f}%")
    pf("=" * 70)

    # ── Data & model ──────────────────────────────────────────
    pf("\nStep 1: Train model")
    model, scaler, feat_idx, n_feat, fixed_vals, co_init, df_f, all_features = prepare_data()

    # ── Bounds ────────────────────────────────────────────────
    pf("\nStep 2: Build bounds")
    bounds, info = build_bounds(df_f)
    predict_steady = make_predictor(model, scaler, feat_idx, n_feat, fixed_vals, co_init)

    # Current operating point (all medians)
    current_x = np.array(
        [float(df_f[f'负压_{i}_al'].median()) for i in range(1, 19)] +
        [float(df_f[f'温度_{i}_al'].median()) for i in range(1, 19)] +
        [float(df_f['机速_al'].median())]
    )
    current_co = predict_steady(current_x)
    pf(f"  Current operating CO: {current_co:.1f} mg/m3")

    pf(f"\n  {'Var':>6s} {'Lower':>10s} {'Median':>10s} {'Upper':>10s} {'Width':>10s}")
    pf("  " + "-" * 50)
    for vtype, idx, lo, hi, med in info:
        label = f"{vtype}{idx}" if idx > 0 else "Speed"
        pf(f"  {label:>6s} {lo:10.2f} {med:10.2f} {hi:10.2f} {hi-lo:10.2f}")

    # ── Reliability penalty ──────────────────────────────────
    def reliability_penalty(x):
        pen = 0.0
        for i, (lo, hi) in enumerate(bounds):
            rng = hi - lo
            if rng < 1e-6:
                continue
            d = min(x[i] - lo, hi - x[i]) / rng
            pen += np.exp(-10 * d)
        return pen

    # ── Objective ────────────────────────────────────────────
    eval_n = [0]
    best_co = [current_co]
    t0 = time.time()
    co_log = []

    def objective(x):
        co  = predict_steady(x)
        pen = reliability_penalty(x)
        obj = co + PENALTY_ALPHA * pen
        eval_n[0] += 1
        co_log.append(co)
        if co < best_co[0]:
            best_co[0] = co
            dt = time.time() - t0
            pf(f"    [{eval_n[0]:5d}]  CO={co:.1f}  pen={pen:.2f}  obj={obj:.0f}  t={dt:.0f}s")
        return obj

    # ── Differential Evolution ────────────────────────────────
    pf(f"\nStep 3: Differential Evolution")
    pf(f"  pop={DE_POPSIZE}×37={DE_POPSIZE*37}, maxiter={DE_MAXITER}")
    pf(f"  penalty_alpha={PENALTY_ALPHA}")
    pf(f"  (reporting only when new best CO found)")

    de_t = time.time()
    result = differential_evolution(
        objective, bounds,
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
    de_elapsed = time.time() - de_t

    opt_x = result.x
    opt_co  = predict_steady(opt_x)
    opt_pen = reliability_penalty(opt_x)
    reduction = (current_co - opt_co) / current_co * 100

    pf(f"\n  DE finished in {de_elapsed:.0f}s ({de_elapsed/60:.1f} min)")
    pf(f"  Optimal CO:   {opt_co:.1f} mg/m3")
    pf(f"  Reduction:    {reduction:.1f}%")
    pf(f"  Penalty:      {opt_pen:.2f}")
    pf(f"  Meets 2800:   {'YES' if opt_co <= 2800 else 'NO'}")

    boundary_n = sum(
        1 for i, (lo, hi) in enumerate(bounds)
        if abs(opt_x[i] - lo) < 0.05*(hi-lo) or abs(opt_x[i] - hi) < 0.05*(hi-lo)
    )
    pf(f"  Near boundary (5%): {boundary_n}/37 variables")

    # ── Comparison ────────────────────────────────────────────
    pressure_only_co = 2741.8   # from Q3_improved
    pf(f"\nStep 4: Comparison")
    pf(f"  {'Method':30s} {'CO':>10s} {'Reduction':>10s}")
    pf("  " + "-" * 55)
    pf(f"  {'Current operating':30s} {current_co:10.1f} {'baseline':>10s}")
    pf(f"  {'Pressure-only PSO (v3)':30s} {pressure_only_co:10.1f} {(current_co-pressure_only_co)/current_co*100:9.1f}%")
    pf(f"  {'Joint DE v4 (P+T+speed)':30s} {opt_co:10.1f} {reduction:9.1f}%")

    # ── Breakdown ─────────────────────────────────────────────
    pf(f"\nStep 5: Optimal decision breakdown")
    opt_p = opt_x[:18]
    opt_t = opt_x[18:36]
    opt_s = opt_x[36]

    s_curr = float(df_f['机速_al'].median())
    pf(f"  Machine speed: {s_curr:.4f} → {opt_s:.4f} ({opt_s-s_curr:+.4f})")

    pf(f"\n  {'Zone':>6s}  {'P_curr':>8s} {'P_opt':>8s} {'P_Δ':>8s}   "
       f"{'T_curr':>8s} {'T_opt':>8s} {'T_Δ':>8s}")
    pf("  " + "-" * 60)
    for i in range(18):
        pc = float(df_f[f'负压_{i+1}_al'].median())
        tc = float(df_f[f'温度_{i+1}_al'].median())
        pf(f"  {i+1:2d}#    {pc:8.2f} {opt_p[i]:8.2f} {opt_p[i]-pc:+8.2f}   "
           f"{tc:8.1f} {opt_t[i]:8.1f} {opt_t[i]-tc:+8.1f}")

    # ── Figures ───────────────────────────────────────────────
    pf(f"\nStep 6: Generating figures")

    p_curr = [float(df_f[f'负压_{i}_al'].median()) for i in range(1, 19)]
    t_curr = [float(df_f[f'温度_{i}_al'].median()) for i in range(1, 19)]

    # Fig 1: CO comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ['Current\nOperating', 'Pressure-only\nPSO (v3)', 'Joint DE v4\n(P+T+speed)']
    vals   = [current_co, pressure_only_co, opt_co]
    cols   = ['#95a5a6', '#e67e22', '#2980b9']
    bars = ax.bar(labels, vals, color=cols, alpha=0.85, width=0.5)
    ax.axhline(2800, color='red', ls='--', lw=2, label='Emission standard: 2800')
    ax.set_ylabel('Steady-state CO (mg/m³)', fontsize=12)
    ax.set_title('Q3 v4: CO Reduction Comparison', fontsize=14)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, vals):
        r = (current_co - v) / current_co * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 40,
                f'{v:.0f}\n({r:.1f}%↓)', ha='center', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/Q3v4_co_comparison.png', dpi=200); plt.close()
    pf("  Q3v4_co_comparison.png")

    # Fig 2: Pressure & temperature adjustments (2-panel)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    x = np.arange(1, 19)

    ax = axes[0]
    p_delta = [opt_p[i] - p_curr[i] for i in range(18)]
    cc_p = ['#e74c3c' if d < -0.05 else '#2ecc71' if d > 0.05 else '#95a5a6' for d in p_delta]
    ax.bar(x, p_delta, color=cc_p, alpha=0.8)
    ax.axhline(0, color='black', lw=1)
    ax.set_ylabel('Pressure Change (KPa)', fontsize=11)
    ax.set_title('Optimal Pressure Adjustments (v4)', fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels([f'{i}#' for i in x])
    ax.grid(True, alpha=0.3, axis='y')
    for bar, d in zip(ax.patches, p_delta):
        if abs(d) > 0.1:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.05 if d >= 0 else -0.15),
                    f'{d:+.2f}', ha='center', fontsize=8)

    ax = axes[1]
    t_delta = [opt_t[i] - t_curr[i] for i in range(18)]
    cc_t = ['#e74c3c' if d < -2 else '#2ecc71' if d > 2 else '#95a5a6' for d in t_delta]
    ax.bar(x, t_delta, color=cc_t, alpha=0.8)
    ax.axhline(0, color='black', lw=1)
    ax.set_ylabel('Temperature Change (°C)', fontsize=11)
    ax.set_title('Optimal Temperature Adjustments (v4)', fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels([f'{i}#' for i in x])
    ax.grid(True, alpha=0.3, axis='y')
    for bar, d in zip(ax.patches, t_delta):
        if abs(d) > 3:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (1 if d >= 0 else -4),
                    f'{d:+.1f}', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('figures/Q3v4_adjustments.png', dpi=200); plt.close()
    pf("  Q3v4_adjustments.png")

    # Fig 3: Temperature profile
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(range(1, 19), t_curr, 'o-', color='#7f8c8d', lw=2, label='Current (median)')
    ax.plot(range(1, 19), opt_t,  's--', color='#2980b9', lw=2, label='Optimal (v4)')
    # shade changes
    for i in range(18):
        c = '#2ecc71' if opt_t[i] > t_curr[i] else '#e74c3c'
        ax.fill_between([i+1, i+1], [t_curr[i]], [opt_t[i]], alpha=0.25, color=c)
    # IQR bounds
    t_q25 = [float(df_f[f'温度_{i}_al'].quantile(0.25)) for i in range(1, 19)]
    t_q75 = [float(df_f[f'温度_{i}_al'].quantile(0.75)) for i in range(1, 19)]
    ax.fill_between(range(1, 19), t_q25, t_q75, alpha=0.1, color='blue',
                    label='IQR boundary (25%-75%)')
    ax.set_xlabel('Zone Number', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title('Temperature Profile: Current vs Optimal (within IQR bounds)', fontsize=14)
    ax.set_xticks(range(1, 19)); ax.set_xticklabels([f'{i}#' for i in range(1, 19)])
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/Q3v4_temperature_profile.png', dpi=200); plt.close()
    pf("  Q3v4_temperature_profile.png")

    # Fig 4: Boundary distance
    fig, ax = plt.subplots(figsize=(16, 5))
    dists = []
    for i, (lo, hi) in enumerate(bounds):
        rng = hi - lo
        d = min(opt_x[i] - lo, hi - opt_x[i]) / rng if rng > 1e-6 else 0.5
        dists.append(d)
    cc_d = ['#F44336' if d < 0.1 else '#FF9800' if d < 0.2 else '#4CAF50' for d in dists]
    ax.bar(range(37), dists, color=cc_d, alpha=0.8)
    ax.axhline(0.1, color='red',    ls='--', lw=1, label='Warning 10%')
    ax.axhline(0.2, color='orange', ls='--', lw=1, label='Caution 20%')
    ax.axvline(17.5, color='navy', ls=':', lw=1.5)
    ax.axvline(35.5, color='navy', ls=':', lw=1.5)
    ax.text(9, 0.55, 'Pressures\n(0-17)', ha='center', fontsize=10, color='navy')
    ax.text(26, 0.55, 'Temperatures\n(18-35)', ha='center', fontsize=10, color='navy')
    ax.text(36, 0.55, 'Speed', ha='center', fontsize=9, color='navy')
    ax.set_xlabel('Variable index', fontsize=11)
    ax.set_ylabel('Normalized distance to boundary', fontsize=11)
    ax.set_title('v4 Reliability: Boundary Distance (all 37 variables)', fontsize=13)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim(0, 0.65)
    plt.tight_layout()
    plt.savefig('figures/Q3v4_boundary_distance.png', dpi=200); plt.close()
    pf("  Q3v4_boundary_distance.png")

    # ── Save ──────────────────────────────────────────────────
    pf(f"\nStep 7: Save results")
    results = {
        'method': 'Joint 37-var DE: pressures(5-95%) + temperatures(25-75% IQR) + speed(5-95%)',
        'steady_state': 'honest — co_lag fixed at historical CO median',
        'current_co': float(current_co),
        'pressure_only_co': pressure_only_co,
        'joint_de_co': float(opt_co),
        'reduction_pct': float(reduction),
        'meets_2800': bool(opt_co <= 2800),
        'penalty': float(opt_pen),
        'vars_near_boundary': int(boundary_n),
        'optimal_speed': float(opt_s),
        'speed_change': float(opt_s - s_curr),
        'optimal_pressures': {f'bellows_{i+1}': float(opt_p[i]) for i in range(18)},
        'optimal_temperatures': {f'zone_{i+1}': float(opt_t[i]) for i in range(18)},
        'de_config': {'popsize': DE_POPSIZE, 'maxiter': DE_MAXITER,
                      'P_bounds': f'{P_LO*100:.0f}-{P_HI*100:.0f}%',
                      'T_bounds': f'{T_LO*100:.0f}-{T_HI*100:.0f}%',
                      'S_bounds': f'{S_LO*100:.0f}-{S_HI*100:.0f}%',
                      'penalty_alpha': PENALTY_ALPHA},
    }
    with open('results/Q3_v4_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    pf("  results/Q3_v4_results.json")

    rows = []
    for i in range(18):
        lo_p = float(df_f[f'负压_{i+1}_al'].quantile(P_LO))
        hi_p = float(df_f[f'负压_{i+1}_al'].quantile(P_HI))
        lo_t = float(df_f[f'温度_{i+1}_al'].quantile(T_LO))
        hi_t = float(df_f[f'温度_{i+1}_al'].quantile(T_HI))
        rng_p, rng_t = hi_p - lo_p, hi_t - lo_t
        dp = min(opt_p[i]-lo_p, hi_p-opt_p[i])/rng_p if rng_p > 0 else 0.5
        dt = min(opt_t[i]-lo_t, hi_t-opt_t[i])/rng_t if rng_t > 0 else 0.5
        rows.append({
            'Zone': f'{i+1}#',
            'P_current': round(p_curr[i], 3),
            'P_optimal': round(float(opt_p[i]), 3),
            'P_delta': round(float(opt_p[i]) - p_curr[i], 3),
            'P_boundary_dist': round(dp, 3),
            'T_current': round(t_curr[i], 1),
            'T_optimal': round(float(opt_t[i]), 1),
            'T_delta': round(float(opt_t[i]) - t_curr[i], 1),
            'T_boundary_dist': round(dt, 3),
        })
    pd.DataFrame(rows).to_csv('results/Q3_v4_decisions.csv', index=False, encoding='utf-8-sig')
    pf("  results/Q3_v4_decisions.csv")

    elapsed = time.time() - total_t
    pf(f"\n{'='*70}")
    pf(f"Q3 v4 done!  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    pf(f"RESULT: {current_co:.0f} → {opt_co:.0f} mg/m3  =  {reduction:.1f}% reduction")
    pf(f"{'='*70}")


if __name__ == '__main__':
    main()
