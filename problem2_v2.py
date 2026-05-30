"""
问题2：CO浓度实时预测模型
方法：时滞对齐 + 84特征工程 + PSO-XGBoost + 70/30 + 5折CV
"""
import warnings
warnings.filterwarnings('ignore')

import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.signal import fftconvolve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

# ── 中文字体 ──────────────────────────────────────────────────────────────────
def setup_font():
    for p in ['/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
              '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']:
        if os.path.exists(p):
            fm.fontManager.addfont(p)
            prop = fm.FontProperties(fname=p)
            plt.rcParams['font.family'] = prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False
    return 'DejaVu Sans'

FONT = setup_font()
OUT  = '/home/user/math_pro'

# ─────────────────────────────────────────────────────────────────────────────
# 合成数据（真实文件缺失时）
# ─────────────────────────────────────────────────────────────────────────────
def generate_synthetic_data(n=2442, seed=42):
    rng = np.random.default_rng(seed)
    t   = np.arange(n)

    neg_mean = np.linspace(-5.5, -12.4, 18)
    neg = rng.normal(neg_mean, 1.2, (n, 18))
    for i in range(18):
        neg[:, i] = pd.Series(neg[:, i]).ewm(span=8).mean().values

    temp_n = rng.normal(380, 25, (n, 18)) + 12 * np.sin(t[:, None] / 40)
    temp_s = temp_n + rng.normal(0, 8, (n, 18))
    speed  = 1.5 + 0.2 * np.sin(t / 120) + 0.05 * rng.standard_normal(n)
    flue_neg  = np.column_stack([-3.2 - 0.4*np.sin(t/90) + 0.1*rng.standard_normal(n),
                                  -3.5 - 0.3*np.sin(t/100) + 0.1*rng.standard_normal(n)])
    flue_temp = np.column_stack([165 + 8*np.sin(t/80) + 2*rng.standard_normal(n),
                                  162 + 7*np.sin(t/85) + 2*rng.standard_normal(n)])

    co = 3000 + 400*np.sin(t/32) + 300*np.sin(t/55) + 150*rng.standard_normal(n)
    for lag, coef, pi in [(25,-35,0),(18,-22,3),(12,-15,7),(8,-18,1),(35,-10,11)]:
        co[lag:] += coef * neg[:-lag, pi]
        co[:lag] += coef * neg[0, pi]
    co += 0.3*speed*100
    co = np.clip(co, 600, 5500)

    # 注入校准异常（1045-1061）
    co[1045:1054] = [3215, 1.4, 9998, 9998, 9998, 9998, 9998, 9998, 9998]
    co[1054:1062] = np.linspace(1087, 3387, 8)

    cols_neg  = [f'{i+1}#风箱负压'    for i in range(18)]
    cols_tn   = [f'{i+1}#风箱北侧温度' for i in range(18)]
    cols_ts   = [f'{i+1}#风箱南侧温度' for i in range(18)]

    df = pd.DataFrame(neg, columns=cols_neg)
    for i, c in enumerate(cols_tn): df[c] = temp_n[:, i]
    for i, c in enumerate(cols_ts): df[c] = temp_s[:, i]
    df['台车速度'] = speed
    df['1#大烟道负压'] = flue_neg[:, 0]
    df['2#大烟道负压'] = flue_neg[:, 1]
    df['1#大烟道温度'] = flue_temp[:, 0]
    df['2#大烟道温度'] = flue_temp[:, 1]
    df['烧结大烟道外排CO浓度'] = co
    return df

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  问题2：CO浓度实时预测（XGBoost + PSO + 时滞对齐）")
print("=" * 60)

# ── 数据加载 ──────────────────────────────────────────────────────────────────
DATA_PATH = f'{OUT}/附件1_原始数据.xlsx'
if os.path.exists(DATA_PATH):
    df_raw = pd.read_excel(DATA_PATH, header=1)
    print(f"  读取真实数据: {df_raw.shape}")
    DATA_REAL = True
else:
    print("  ⚠ 真实数据未找到，使用合成数据演示")
    df_raw = generate_synthetic_data()
    DATA_REAL = False

CO_COL  = '烧结大烟道外排CO浓度'
NEG_COLS = sorted([c for c in df_raw.columns if '风箱负压' in c and '大烟道' not in c],
                  key=lambda x: int(x.split('#')[0]))[:18]
TN_COLS  = sorted([c for c in df_raw.columns if '北侧温度' in c],
                  key=lambda x: int(x.split('#')[0]))[:18]
TS_COLS  = sorted([c for c in df_raw.columns if '南侧温度' in c],
                  key=lambda x: int(x.split('#')[0]))[:18]
SPEED_COL    = '台车速度'
FLUE_NEG_COLS  = ['1#大烟道负压', '2#大烟道负压']
FLUE_TEMP_COLS = ['1#大烟道温度', '2#大烟道温度']

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: 数据清洗（去除 1045-1061 校准异常）
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 1] 数据清洗...")
abnormal = list(range(1045, 1062))
df = df_raw.drop(index=[i for i in abnormal if i < len(df_raw)]).reset_index(drop=True)
df.ffill(inplace=True); df.bfill(inplace=True)
print(f"  去除校准异常索引 1045-1061: {len(df_raw)} → {len(df)} 条")

# 南北温度取平均，得到18个温度特征
TEMP_COLS = []
for i in range(len(NEG_COLS)):
    cname = f'{i+1}#风箱温度'
    tn = TN_COLS[i] if i < len(TN_COLS) else None
    ts = TS_COLS[i] if i < len(TS_COLS) else None
    if tn and ts:
        df[cname] = (df[tn] + df[ts]) / 2
    elif tn:
        df[cname] = df[tn]
    elif ts:
        df[cname] = df[ts]
    TEMP_COLS.append(cname)

# 41个原始输入变量
RAW_VARS = ([SPEED_COL] + NEG_COLS + TEMP_COLS +
            FLUE_NEG_COLS + FLUE_TEMP_COLS)
RAW_VARS = [c for c in RAW_VARS if c in df.columns]
print(f"  原始输入变量: {len(RAW_VARS)} 个")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: 时滞对齐（FFT互相关，max_lag=60）
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 2] 时滞对齐（FFT互相关，max_lag=60步）...")

def compute_xcorr(x, y, max_lag=60):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xn = (x - x.mean()) / (x.std() * len(x) + 1e-8)
    yn = (y - y.mean()) / (y.std()           + 1e-8)
    corr  = fftconvolve(xn[::-1], yn, mode='full')
    center = len(xn) - 1
    lags   = np.arange(-max_lag, max_lag + 1)
    cc     = corr[center - max_lag: center + max_lag + 1]
    return lags, cc

co_arr = df[CO_COL].values.astype(float)
lag_results = {}
for var in RAW_VARS:
    lags, cc = compute_xcorr(df[var].values, co_arr, max_lag=60)
    best_lag = int(lags[np.argmax(np.abs(cc))])
    lag_results[var] = best_lag

print("  时滞结果（前10个变量）:")
for v in RAW_VARS[:10]:
    print(f"    {v}: lag={lag_results[v]:+d} 步 ({lag_results[v]*2:+d} 秒)")

# 对齐：shift(-lag) 使变量提前 lag 步，与CO对齐
df_al = df.copy()
for var in RAW_VARS:
    lag = lag_results[var]
    df_al[f'{var}_al'] = df[var].shift(-lag)

AL_VARS = [f'{v}_al' for v in RAW_VARS]

# 去掉shift引入的NaN
df_al.dropna(subset=AL_VARS + [CO_COL], inplace=True)
df_al.reset_index(drop=True, inplace=True)
print(f"  对齐后剩余: {len(df_al)} 条")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: 特征工程（84个特征）
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 3] 特征工程（目标84个特征）...")

# 对齐后的负压/温度列
NEG_AL   = [f'{c}_al' for c in NEG_COLS   if f'{c}_al' in df_al.columns]
TEMP_AL  = [f'{c}_al' for c in TEMP_COLS  if f'{c}_al' in df_al.columns]

# (1) 物理基础特征：41个（直接用对齐列）
feat_base = [c for c in AL_VARS if c in df_al.columns]  # ≤41

# (2) 梯度特征：负压差 17个 + 温度差 17个 = 34个
for i in range(len(NEG_AL) - 1):
    df_al[f'pgrad_{i+1}'] = df_al[NEG_AL[i+1]] - df_al[NEG_AL[i]]
for i in range(len(TEMP_AL) - 1):
    df_al[f'tgrad_{i+1}'] = df_al[TEMP_AL[i+1]] - df_al[TEMP_AL[i]]

pgrad_cols = [f'pgrad_{i+1}' for i in range(len(NEG_AL)-1)]
tgrad_cols = [f'tgrad_{i+1}' for i in range(len(TEMP_AL)-1)]

# (3) 统计特征：4个
df_al['p_mean'] = df_al[NEG_AL].mean(axis=1)
mid_neg  = [NEG_AL[i] for i in range(5, 12) if i < len(NEG_AL)]
back_neg = [NEG_AL[i] for i in range(12, 18) if i < len(NEG_AL)]
back_tmp = [TEMP_AL[i] for i in range(12, 18) if i < len(TEMP_AL)]
df_al['p_mid']  = df_al[mid_neg].mean(axis=1)  if mid_neg  else 0.0
df_al['p_back'] = df_al[back_neg].mean(axis=1) if back_neg else 0.0
df_al['t_back'] = df_al[back_tmp].mean(axis=1) if back_tmp else 0.0
stat_cols = ['p_mean', 'p_mid', 'p_back', 't_back']

# (4) CO自回归特征：5个（全部基于历史值，不含当前CO，避免数据泄露）
df_al['co_lag1']  = df_al[CO_COL].shift(1)           # CO(t-1)
df_al['co_lag2']  = df_al[CO_COL].shift(2)           # CO(t-2)
df_al['co_lag5']  = df_al[CO_COL].shift(5)           # CO(t-5)
df_al['co_ma5']   = df_al[CO_COL].shift(1).rolling(5).mean()  # mean(CO[t-5..t-1])，不含CO(t)
df_al['co_diff1'] = df_al[CO_COL].shift(1).diff(1)           # CO(t-1)-CO(t-2)，不含CO(t)
co_ar_cols = ['co_lag1', 'co_lag2', 'co_lag5', 'co_ma5', 'co_diff1']

# 去掉rolling/shift引入的NaN
df_al.dropna(inplace=True)
df_al.reset_index(drop=True, inplace=True)

# 汇总所有特征
ALL_FEAT = feat_base + pgrad_cols + tgrad_cols + stat_cols + co_ar_cols
ALL_FEAT = [c for c in dict.fromkeys(ALL_FEAT) if c in df_al.columns]

print(f"  物理基础特征: {len(feat_base)}")
print(f"  压力梯度特征: {len(pgrad_cols)}")
print(f"  温度梯度特征: {len(tgrad_cols)}")
print(f"  统计特征:     {len(stat_cols)}")
print(f"  CO自回归特征: {len(co_ar_cols)}")
print(f"  合计特征数:   {len(ALL_FEAT)}")
print(f"  最终样本数:   {len(df_al)}")

X_all = df_al[ALL_FEAT].values.astype(float)
y_all = df_al[CO_COL].values.astype(float)

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: PSO优化XGBoost超参数（8粒子，10迭代，3折TimeSeriesCV）
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 4] PSO超参数搜索（8粒子 × 10迭代，3折时序CV）...")

N_PART, N_ITER = 8, 10
LB = np.array([0.01,  3, 100, 0.5, 0.5,  0,  0], dtype=float)
UB = np.array([0.30, 15, 500, 1.0, 1.0, 10, 10], dtype=float)
DIM   = len(LB)
V_MAX = (UB - LB) * 0.2

def decode_xgb(x):
    return {
        'learning_rate':    float(np.clip(x[0], 0.005, 0.5)),
        'max_depth':        int(np.clip(round(x[1]), 2, 15)),
        'n_estimators':     int(np.clip(round(x[2]), 50, 600)),
        'subsample':        float(np.clip(x[3], 0.4, 1.0)),
        'colsample_bytree': float(np.clip(x[4], 0.4, 1.0)),
        'reg_alpha':        float(np.clip(x[5], 0, 15)),
        'reg_lambda':       float(np.clip(x[6], 0, 15)),
        'early_stopping_rounds': 20,
    }

tscv3 = TimeSeriesSplit(n_splits=3)

def pso_fitness(x):
    params = decode_xgb(x)
    params.update({'random_state': 42, 'tree_method': 'hist', 'device': 'cpu',
                   'verbosity': 0, 'n_jobs': -1})
    scores = []
    for tr_idx, va_idx in tscv3.split(X_all):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_all[tr_idx])
        Xva = sc.transform(X_all[va_idx])
        m = XGBRegressor(**params)
        m.fit(Xtr, y_all[tr_idx], eval_set=[(Xva, y_all[va_idx])],
              verbose=False)
        scores.append(r2_score(y_all[va_idx], m.predict(Xva)))
    return float(np.mean(scores))

rng = np.random.default_rng(7)
pos    = rng.uniform(LB, UB, (N_PART, DIM))
vel    = rng.uniform(-V_MAX, V_MAX, (N_PART, DIM))
pbest  = pos.copy()
pfit   = np.array([pso_fitness(pos[i]) for i in range(N_PART)])
gi     = int(np.argmax(pfit))
gbest  = pbest[gi].copy()
gfit   = pfit[gi]
pso_cv = []

print(f"  初始最优3折CV R²: {gfit:.4f}")

w = 0.9
for it in range(1, N_ITER + 1):
    for i in range(N_PART):
        r1, r2 = rng.random(DIM), rng.random(DIM)
        vel[i] = w * vel[i] + 2*r1*(pbest[i]-pos[i]) + 2*r2*(gbest-pos[i])
        vel[i] = np.clip(vel[i], -V_MAX, V_MAX)
        pos[i] = np.clip(pos[i] + vel[i], LB, UB)
        f = pso_fitness(pos[i])
        if f > pfit[i]:
            pfit[i] = f; pbest[i] = pos[i].copy()
        if f > gfit:
            gfit = f; gbest = pos[i].copy()
    w = max(0.3, w * 0.97)
    pso_cv.append(gfit)
    print(f"  迭代 {it:2d}/{N_ITER}  最优3折CV R² = {gfit:.4f}")

best_params = decode_xgb(gbest)
best_params.update({'random_state': 42, 'tree_method': 'hist', 'device': 'cpu',
                    'verbosity': 0, 'n_jobs': -1})
print(f"\n  PSO最优超参数:")
for k, v in best_params.items():
    if k not in ['random_state', 'tree_method', 'device', 'verbosity', 'n_jobs']:
        print(f"    {k}: {v}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 5 & 6: 模型训练 + 评估（70/30 时序划分）
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 5&6] 模型训练与评估...")

split = int(len(X_all) * 0.7)
X_tr, X_te = X_all[:split], X_all[split:]
y_tr, y_te = y_all[:split], y_all[split:]

sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr)
X_te_s  = sc.transform(X_te)

model = XGBRegressor(**best_params)
model.fit(X_tr_s, y_tr, eval_set=[(X_te_s, y_te)],
          verbose=False)

y_pred_tr = model.predict(X_tr_s)
y_pred_te = model.predict(X_te_s)

r2_tr   = r2_score(y_tr, y_pred_tr)
r2_te   = r2_score(y_te, y_pred_te)
mae_te  = mean_absolute_error(y_te, y_pred_te)
rmse_te = np.sqrt(mean_squared_error(y_te, y_pred_te))

print(f"  70/30划分结果:")
print(f"    训练集: R²={r2_tr:.4f}")
print(f"    测试集: R²={r2_te:.4f}  MAE={mae_te:.2f} mg/m³  RMSE={rmse_te:.2f} mg/m³")

# 5折时间序列CV
print("\n  5折时间序列交叉验证:")
tscv5  = TimeSeriesSplit(n_splits=5)
cv_r2s = []
for fold, (tri, vai) in enumerate(tscv5.split(X_all), 1):
    sc5    = StandardScaler()
    Xtr5   = sc5.fit_transform(X_all[tri])
    Xva5   = sc5.transform(X_all[vai])
    m5     = XGBRegressor(**best_params)
    m5.fit(Xtr5, y_all[tri], eval_set=[(Xva5, y_all[vai])],
           verbose=False)
    r5 = r2_score(y_all[vai], m5.predict(Xva5))
    cv_r2s.append(r5)
    print(f"    Fold {fold}: R²={r5:.4f}  (训练[0:{tri[-1]+1}] 验证[{vai[0]}:{vai[-1]+1}])")

cv_mean = np.mean(cv_r2s)
cv_std  = np.std(cv_r2s)
print(f"    平均 R² = {cv_mean:.4f} ± {cv_std:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 7: 特征重要性（XGBoost Gain）
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 7] 特征重要性分析（Gain）...")

imp_gain = model.get_booster().get_score(importance_type='gain')
feat_imp = pd.DataFrame([{'feature': ALL_FEAT[int(k[1:])], 'gain': v}
                          for k, v in imp_gain.items()])
total_gain = feat_imp['gain'].sum()
feat_imp['importance'] = feat_imp['gain'] / total_gain
feat_imp = feat_imp.sort_values('importance', ascending=False).reset_index(drop=True)

# 类别标签
def get_category(fname):
    if fname in co_ar_cols:     return 'CO自回归'
    if fname in stat_cols:      return '统计特征'
    if fname in pgrad_cols + tgrad_cols: return '梯度特征'
    return '物理基础'

feat_imp['category'] = feat_imp['feature'].apply(get_category)
print("  Top 10 重要特征:")
for _, row in feat_imp.head(10).iterrows():
    print(f"    {row['feature']:30s}  {row['importance']:.4f}  [{row['category']}]")

# 各类别汇总
cat_imp = feat_imp.groupby('category')['importance'].sum().sort_values(ascending=False)
print("\n  各类别总重要性:")
for cat, val in cat_imp.items():
    print(f"    {cat}: {val*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# Step 8: 可视化（5张图）
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 8] 生成图表...")

cat_colors = {'CO自回归': '#e74c3c', '物理基础': '#3498db',
              '梯度特征': '#2ecc71', '统计特征': '#f39c12'}

# 图1: 散点图（预测 vs 真实）
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_te, y_pred_te, s=8, alpha=0.5, color='steelblue', label='测试集')
lo, hi = min(y_te.min(), y_pred_te.min()), max(y_te.max(), y_pred_te.max())
ax.plot([lo, hi], [lo, hi], 'r--', lw=1.5, label='理想线')
ax.set_xlabel('实测CO浓度 (mg/m³)', fontname=FONT)
ax.set_ylabel('预测CO浓度 (mg/m³)', fontname=FONT)
ax.set_title(f'预测值 vs 真实值  R²={r2_te:.4f}', fontname=FONT)
ax.legend(prop={'family': FONT})
plt.tight_layout()
fig.savefig(f'{OUT}/Q2_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Q2_scatter.png")

# 图2: 误差分布直方图
errors = y_pred_te - y_te
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(errors, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(0, color='red', lw=1.5, ls='--')
ax.set_xlabel('预测误差 (mg/m³)', fontname=FONT)
ax.set_ylabel('频次', fontname=FONT)
ax.set_title(f'误差分布  MAE={mae_te:.2f}  RMSE={rmse_te:.2f}', fontname=FONT)
plt.tight_layout()
fig.savefig(f'{OUT}/Q2_error_dist.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Q2_error_dist.png")

# 图3: 特征重要性（Top 20，按类别着色）
top20 = feat_imp.head(20)
colors = [cat_colors.get(c, 'gray') for c in top20['category']]
fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(range(len(top20)), top20['importance'].values[::-1], color=colors[::-1])
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20['feature'].values[::-1], fontname=FONT, fontsize=9)
ax.set_xlabel('重要性 (Gain归一化)', fontname=FONT)
ax.set_title('XGBoost特征重要性 Top 20（Gain）', fontname=FONT)
# 图例
from matplotlib.patches import Patch
handles = [Patch(color=v, label=k) for k, v in cat_colors.items()]
ax.legend(handles=handles, prop={'family': FONT}, loc='lower right')
plt.tight_layout()
fig.savefig(f'{OUT}/Q2_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Q2_importance.png")

# 图4: 5折CV R²结果
fig, ax = plt.subplots(figsize=(7, 5))
bar_colors = ['#e74c3c' if r < 0.8 else '#2ecc71' for r in cv_r2s]
ax.bar(range(1, 6), cv_r2s, color=bar_colors, edgecolor='white', width=0.6)
ax.axhline(cv_mean, color='navy', lw=2, ls='--',
           label=f'平均 R²={cv_mean:.4f}±{cv_std:.4f}')
ax.set_xticks(range(1, 6))
ax.set_xticklabels([f'Fold {i}' for i in range(1, 6)], fontname=FONT)
ax.set_ylabel('R²', fontname=FONT)
ax.set_ylim(0, 1.05)
ax.set_title('5折时间序列交叉验证结果', fontname=FONT)
ax.legend(prop={'family': FONT})
plt.tight_layout()
fig.savefig(f'{OUT}/Q2_cv_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Q2_cv_results.png")

# 图5: PSO收敛曲线
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, len(pso_cv)+1), pso_cv, 'b-o', ms=6, lw=2)
ax.set_xlabel('迭代次数', fontname=FONT)
ax.set_ylabel('最优3折CV R²', fontname=FONT)
ax.set_title('PSO收敛曲线', fontname=FONT)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(f'{OUT}/Q2_pso_convergence.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Q2_pso_convergence.png")

# ─────────────────────────────────────────────────────────────────────────────
# 最终结果汇总
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  最终结果汇总")
print("=" * 60)
print(f"  数据来源: {'真实数据' if DATA_REAL else '合成数据（演示）'}")
print()
print("【数据处理】")
print(f"  原始样本: {len(df_raw)} → 去异常后: {len(df)} → 对齐后: {len(df_al)} → 特征构造后: {len(df_al)}")
print(f"  最终特征数: {len(ALL_FEAT)}")
print(f"    物理基础: {len(feat_base)} | 梯度: {len(pgrad_cols)+len(tgrad_cols)} | 统计: {len(stat_cols)} | CO自回归: {len(co_ar_cols)}")
print()
print("【时滞结果（关键变量）】")
key_vars = RAW_VARS[:6]
for v in key_vars:
    print(f"  {v}: lag={lag_results[v]:+d} 步（{lag_results[v]*2:+d} 秒）")
print()
print("【PSO最优超参数】")
for k, v in best_params.items():
    if k not in ['random_state', 'tree_method', 'device', 'verbosity', 'n_jobs']:
        print(f"  {k}: {v}")
print()
print("【模型性能（70/30时序划分）】")
print(f"  训练集: R²={r2_tr:.4f}")
print(f"  测试集: R²={r2_te:.4f}  MAE={mae_te:.2f} mg/m³  RMSE={rmse_te:.2f} mg/m³")
print()
print("【5折时间序列CV】")
for i, r in enumerate(cv_r2s, 1):
    print(f"  Fold {i}: R²={r:.4f}")
print(f"  平均: R²={cv_mean:.4f} ± {cv_std:.4f}")
print()
print("【特征重要性各类别】")
for cat, val in cat_imp.items():
    print(f"  {cat}: {val*100:.1f}%")
print()
print("【Top 6重要特征】")
for _, row in feat_imp.head(6).iterrows():
    print(f"  {row['feature']:30s} {row['importance']:.4f} [{row['category']}]")
print()
print("  ===全部完成===")
