"""
烧结工艺问题2：CO浓度预测与影响规律分析
PSO-LightGBM + SHAP分析 + 偏依赖图
"""
import warnings
warnings.filterwarnings('ignore')

import os, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.signal import correlate
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

# ── 中文字体 ──────────────────────────────────────────────────────────────────
def setup_font():
    candidates = [
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    ]
    for p in candidates:
        if os.path.exists(p):
            fm.fontManager.addfont(p)
            prop = fm.FontProperties(fname=p)
            plt.rcParams['font.family'] = prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False
    return 'DejaVu Sans'

FONT_NAME = setup_font()
OUT_DIR = '/home/user/math_pro'

# ─────────────────────────────────────────────────────────────────────────────
# 0. 合成数据（真实文件不可用时）
# ─────────────────────────────────────────────────────────────────────────────
def generate_synthetic_data(n=2442, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    neg_mean = np.linspace(-5.5, -12.4, 18)
    neg_std  = np.full(18, 1.2)
    neg_pressures = rng.normal(neg_mean, neg_std, (n, 18))
    for i in range(18):
        neg_pressures[:, i] = np.clip(
            pd.Series(neg_pressures[:, i]).ewm(span=8).mean().values,
            neg_mean[i] - 3, neg_mean[i] + 3
        )

    temp_north = rng.normal(380, 25, (n, 18)) + 12 * np.sin(t[:, None] / 40)
    temp_south = temp_north + rng.normal(0, 8, (n, 18))

    speed      = 1.5 + 0.2 * np.sin(t / 120) + 0.05 * rng.standard_normal(n)
    main_neg   = np.array([-3.2 - 0.4 * np.sin(t / 90) + 0.1 * rng.standard_normal(n),
                            -3.5 - 0.3 * np.sin(t / 100) + 0.1 * rng.standard_normal(n)]).T
    main_temp  = np.array([165 + 8 * np.sin(t / 80) + 2 * rng.standard_normal(n),
                            162 + 7 * np.sin(t / 85) + 2 * rng.standard_normal(n)]).T
    moisture   = 7.0 + 0.5 * np.sin(t / 60) + 0.1 * rng.standard_normal(n)
    basicity   = 2.0 + 0.05 * rng.standard_normal(n)
    fuel_rate  = 4.5 + 0.3 * np.sin(t / 130) + 0.05 * rng.standard_normal(n)

    # CO with causal lags from pressure
    co = 3000 + 400 * np.sin(t / 32) + 300 * np.sin(t / 55) + 150 * rng.standard_normal(n)
    lags_used = [25, 18, 12, 8, 35]  # indices 0,3,7,1,11
    coeffs    = [-35, -22, -15, -18, -10]
    pidx      = [0, 3, 7, 1, 11]
    for lag, coef, pi in zip(lags_used, coeffs, pidx):
        co[lag:] += coef * neg_pressures[:-lag, pi]
        co[:lag] += coef * neg_pressures[0, pi]
    co += 0.3 * speed * 100 + 30 * moisture
    co = np.clip(co, 600, 5500)

    # 列名
    neg_cols  = [f'{i+1}#风箱负压' for i in range(18)]
    tnc       = [f'{i+1}#风箱北侧温度' for i in range(18)]
    tsc       = [f'{i+1}#风箱南侧温度' for i in range(18)]

    df = pd.DataFrame(neg_pressures, columns=neg_cols)
    for i, c in enumerate(tnc): df[c] = temp_north[:, i]
    for i, c in enumerate(tsc): df[c] = temp_south[:, i]
    df['台车速度'] = speed
    df['1#大烟道负压'] = main_neg[:, 0]
    df['2#大烟道负压'] = main_neg[:, 1]
    df['1#大烟道温度'] = main_temp[:, 0]
    df['2#大烟道温度'] = main_temp[:, 1]
    df['混合料水分'] = moisture
    df['碱度'] = basicity
    df['固体燃料配比'] = fuel_rate
    df['烧结大烟道外排CO浓度'] = co
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: 数据加载与多级清洗
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  烧结工艺问题2：CO浓度预测与影响规律分析")
print("=" * 65)
print("\n[Step 1] 数据加载与多级清洗...")

DATA_PATH = '/home/user/math_pro/附件1_原始数据.xlsx'
if os.path.exists(DATA_PATH):
    df_raw = pd.read_excel(DATA_PATH, header=1)
    print(f"  读取真实数据: {df_raw.shape}")
    DATA_REAL = True
else:
    print("  ⚠ 真实数据文件未找到，使用合成数据演示完整流程")
    df_raw = generate_synthetic_data()
    DATA_REAL = False

# 列识别
CO_COL   = '烧结大烟道外排CO浓度'
NEG_COLS = [c for c in df_raw.columns if '风箱负压' in c and '#' in c and '大烟道' not in c]
TN_COLS  = [c for c in df_raw.columns if '北侧温度' in c]
TS_COLS  = [c for c in df_raw.columns if '南侧温度' in c]

# 仅保留18个风箱
NEG_COLS = sorted([c for c in NEG_COLS if int(c.split('#')[0]) <= 18],
                  key=lambda x: int(x.split('#')[0]))[:18]
TN_COLS  = sorted(TN_COLS, key=lambda x: int(x.split('#')[0]))[:18]
TS_COLS  = sorted(TS_COLS, key=lambda x: int(x.split('#')[0]))[:18]

print(f"  CO列: {CO_COL}")
print(f"  风箱负压列({len(NEG_COLS)}): {NEG_COLS[:3]}...")
n0 = len(df_raw)

# 1a. 故障时段剔除 (1045-1061)
fault_start, fault_end = 1045, 1061
df = df_raw.drop(index=range(fault_start, min(fault_end + 1, len(df_raw)))).reset_index(drop=True)
print(f"  故障时段剔除 [{fault_start},{fault_end}]: {n0} → {len(df)}")

# 1b. 前向填充缺失值
df.ffill(inplace=True)
df.bfill(inplace=True)

# 1c. 滑动窗口Z-score异常剔除（CO列）
def sliding_zscore_mask(series, window=50, threshold=3.0, buffer=5):
    keep = np.ones(len(series), dtype=bool)
    arr  = series.values.astype(float)
    for i in range(len(arr)):
        lo = max(0, i - window)
        seg = arr[lo:i + 1]
        if len(seg) < 5:
            continue
        mu, sd = seg.mean(), seg.std()
        if sd > 0 and abs(arr[i] - mu) > threshold * sd:
            lo2 = max(0, i - buffer)
            hi2 = min(len(arr), i + buffer + 1)
            keep[lo2:hi2] = False
    return keep

keep_mask = sliding_zscore_mask(df[CO_COL], window=50, threshold=3.0, buffer=5)
n_before = len(df)
df = df[keep_mask].reset_index(drop=True)
print(f"  滑动窗口Z-score剔除: {n_before} → {len(df)}")

# 1d. 1%-99%分位数截断
q1_co = df[CO_COL].quantile(0.01)
q99_co = df[CO_COL].quantile(0.99)
n_before = len(df)
df = df[(df[CO_COL] >= q1_co) & (df[CO_COL] <= q99_co)].reset_index(drop=True)
print(f"  CO 1%-99%分位数截断: {n_before} → {len(df)}")

# 1e. 南北温度均值合并
AVG_TEMP_COLS = []
for i in range(len(TN_COLS)):
    col_name = f'{i+1}#风箱温度均值'
    if i < len(TN_COLS) and i < len(TS_COLS):
        df[col_name] = (df[TN_COLS[i]] + df[TS_COLS[i]]) / 2
    else:
        df[col_name] = df[TN_COLS[i]] if i < len(TN_COLS) else df[TS_COLS[i]]
    AVG_TEMP_COLS.append(col_name)

n_clean = len(df)
print(f"  Step1完成: {n_clean} 行")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: 时滞分析 + 全相关分析
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 2] 时滞分析 + 相关性分析...")

co_arr = df[CO_COL].values.astype(float)

# 互相关时滞
MAX_LAG = 300
lags_dict = {}
for col in NEG_COLS:
    x = df[col].values.astype(float)
    x_z = (x - x.mean()) / (x.std() + 1e-9)
    y_z = (co_arr - co_arr.mean()) / (co_arr.std() + 1e-9)
    corr = correlate(y_z, x_z, mode='full')
    mid  = len(corr) // 2
    corr_pos = corr[mid: mid + MAX_LAG + 1]
    best_lag = int(np.argmax(np.abs(corr_pos)))
    lags_dict[col] = best_lag

print("  时滞结果（步数，间隔2秒）:")
for col in NEG_COLS:
    print(f"    {col}: {lags_dict[col]} 步 = {lags_dict[col]*2} 秒")

# Pearson / Spearman 相关系数
print("\n  Pearson相关系数（|r|前5）:")
pearson_corr  = {}
spearman_corr = {}
all_feat_cols = NEG_COLS + AVG_TEMP_COLS
for col in all_feat_cols:
    try:
        from scipy.stats import spearmanr
        pr  = df[col].corr(df[CO_COL])
        sr, _ = spearmanr(df[col], df[CO_COL])
        pearson_corr[col]  = pr
        spearman_corr[col] = sr
    except Exception:
        pearson_corr[col]  = 0.0
        spearman_corr[col] = 0.0

top5 = sorted(pearson_corr.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
for col, r in top5:
    print(f"    {col}: r={r:.4f}")

# 区域分组：前段(1-6)、中段(7-12)、后段(13-18)
region_labels = {'前段(1-6)': list(range(0,6)), '中段(7-12)': list(range(6,12)), '后段(13-18)': list(range(12,18))}
print("\n  区域负压均值 vs CO 相关分析:")
for rname, idxs in region_labels.items():
    rcols = [NEG_COLS[i] for i in idxs if i < len(NEG_COLS)]
    r_mean = df[rcols].mean(axis=1)
    rc = r_mean.corr(df[CO_COL])
    print(f"    {rname}: r={rc:.4f}")

print("  Step2完成")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: 特征工程（特征集A & B）
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 3] 特征工程（特征集A & B）...")

# 应用时滞对齐
df_feat = df.copy()
for col in NEG_COLS:
    lag = lags_dict[col]
    if lag > 0:
        shifted = np.empty(len(df_feat))
        shifted[:lag] = df_feat[col].values[0]
        shifted[lag:] = df_feat[col].values[:-lag]
        df_feat[f'{col}_lagged'] = shifted
    else:
        df_feat[f'{col}_lagged'] = df_feat[col].values

LAGGED_NEG_COLS = [f'{c}_lagged' for c in NEG_COLS]

# 滚动统计（风箱温度均值 rolling mean/std）
for col in AVG_TEMP_COLS:
    df_feat[f'{col}_roll5m']  = df_feat[col].rolling(5, min_periods=1).mean()
    df_feat[f'{col}_roll5s']  = df_feat[col].rolling(5, min_periods=1).std().fillna(0)

# 一阶差分（负压）
for col in NEG_COLS:
    df_feat[f'{col}_diff1'] = df_feat[col].diff().fillna(0)

# 区域均值
for rname, idxs in region_labels.items():
    rcols = [NEG_COLS[i] for i in idxs if i < len(NEG_COLS)]
    col_safe = rname.replace('(', '').replace(')', '').replace('-', '_')
    df_feat[f'region_neg_{col_safe}'] = df_feat[rcols].mean(axis=1)

# 其他工艺列
OTHER_COLS = [c for c in df.columns if c not in NEG_COLS + TN_COLS + TS_COLS + AVG_TEMP_COLS + [CO_COL]
              and df[c].dtype in [np.float64, np.int64, float, int]]

# CO历史滞后特征（特征集A用）
CO_LAG_NAMES = []
for lg in [1, 3, 5, 10, 20]:
    df_feat[f'CO_lag{lg}'] = df_feat[CO_COL].shift(lg).ffill()
    CO_LAG_NAMES.append(f'CO_lag{lg}')
df_feat['CO_roll5m'] = df_feat[CO_COL].rolling(5, min_periods=1).mean()
df_feat['CO_roll5s'] = df_feat[CO_COL].rolling(5, min_periods=1).std().fillna(0)
CO_LAG_NAMES += ['CO_roll5m', 'CO_roll5s']

df_feat.dropna(inplace=True)
df_feat.reset_index(drop=True, inplace=True)

# 基础特征（B不含CO历史）
BASE_FEAT = (LAGGED_NEG_COLS + AVG_TEMP_COLS +
             [f'{c}_roll5m' for c in AVG_TEMP_COLS] +
             [f'{c}_roll5s' for c in AVG_TEMP_COLS] +
             [f'{c}_diff1'  for c in NEG_COLS] +
             [c for c in ['前段(1-6)', '中段(7-12)', '后段(13-18)']
              if f'region_neg_{c.replace("(","").replace(")","").replace("-","_")}' in df_feat.columns] +
             [f'region_neg_{rname.replace("(","").replace(")","").replace("-","_")}' for rname in region_labels] +
             [c for c in OTHER_COLS if c in df_feat.columns])
BASE_FEAT = [c for c in dict.fromkeys(BASE_FEAT) if c in df_feat.columns]

# 特征集B（无CO历史）
FEAT_B = BASE_FEAT.copy()
# 特征集A（含CO历史）
FEAT_A = BASE_FEAT + CO_LAG_NAMES

print(f"  特征集A（含CO历史）: {len(FEAT_A)} 列")
print(f"  特征集B（无CO历史）: {len(FEAT_B)} 列")
print(f"  Step3完成: {len(df_feat)} 行")

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: PCA降维（18个时滞对齐负压）
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 4] PCA降维（18个时滞对齐负压 → 主成分）...")

X_pca_raw = df_feat[LAGGED_NEG_COLS].values
scaler_pca = StandardScaler()
X_pca_scaled = scaler_pca.fit_transform(X_pca_raw)

pca_model = PCA(n_components=None)
pca_model.fit(X_pca_scaled)
cumvar = np.cumsum(pca_model.explained_variance_ratio_)
n_comp = int(np.searchsorted(cumvar, 0.95)) + 1
n_comp = max(2, min(n_comp, 18))

pca_final = PCA(n_components=n_comp)
pca_scores = pca_final.fit_transform(X_pca_scaled)
print(f"  保留主成分数: {n_comp}（累计解释方差 >= 95%）")
for i, v in enumerate(pca_final.explained_variance_ratio_):
    print(f"    PC{i+1}: {v*100:.2f}%  (累计 {cumvar[i]*100:.2f}%)")

PCA_COL_NAMES = [f'PC{i+1}' for i in range(n_comp)]
for i, cname in enumerate(PCA_COL_NAMES):
    df_feat[cname] = pca_scores[:, i]

# 从基础特征中移除原始时滞负压列，改用PCA
FEAT_B_PCA = [c for c in FEAT_B if c not in LAGGED_NEG_COLS] + PCA_COL_NAMES
FEAT_A_PCA = [c for c in FEAT_A if c not in LAGGED_NEG_COLS] + PCA_COL_NAMES
print(f"  PCA替换后: 特征集B={len(FEAT_B_PCA)}列, 特征集A={len(FEAT_A_PCA)}列")

# 图1: PCA解释方差
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
evr_all = pca_model.explained_variance_ratio_[:10]
axes[0].bar(range(1, len(evr_all)+1), evr_all * 100, color='steelblue')
axes[0].set_xlabel('主成分编号', fontname=FONT_NAME)
axes[0].set_ylabel('解释方差比例(%)', fontname=FONT_NAME)
axes[0].set_title('各主成分解释方差', fontname=FONT_NAME)
axes[1].plot(range(1, len(evr_all)+1), np.cumsum(evr_all)*100, 'o-', color='tomato')
axes[1].axhline(95, ls='--', color='gray', label='95%')
axes[1].set_xlabel('主成分数量', fontname=FONT_NAME)
axes[1].set_ylabel('累计解释方差(%)', fontname=FONT_NAME)
axes[1].set_title('累计解释方差', fontname=FONT_NAME)
axes[1].legend(prop={'family': FONT_NAME})
plt.tight_layout()
fig.savefig(f'{OUT_DIR}/P2_Q2_fig1_PCA.png', dpi=150, bbox_inches='tight')
plt.close()
print("  图1已保存: P2_Q2_fig1_PCA.png")

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: 相关性过滤（阈值0.95，保护PCA列）
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 5] 相关性过滤（阈值=0.95，PCA列受保护）...")

def corr_filter(df_in, feat_list, threshold=0.95, protected=None):
    protected = protected or []
    filter_cols = [c for c in feat_list if c not in protected and c in df_in.columns]
    if not filter_cols:
        return feat_list
    corr_mat  = df_in[filter_cols].corr().abs()
    upper_tri = corr_mat.where(np.triu(np.ones(corr_mat.shape, dtype=bool), k=1))
    drop_cols = {col for col in upper_tri.columns if (upper_tri[col] > threshold).any()}
    kept = [c for c in feat_list if c not in drop_cols]
    return kept, drop_cols

FEAT_B_filtered, drop_B = corr_filter(df_feat, FEAT_B_PCA, 0.95, PCA_COL_NAMES)
FEAT_A_filtered, drop_A = corr_filter(df_feat, FEAT_A_PCA, 0.95, PCA_COL_NAMES + CO_LAG_NAMES)
print(f"  特征集B: 剔除 {len(drop_B)} 个高相关特征 → {len(FEAT_B_filtered)} 列")
print(f"  特征集A: 剔除 {len(drop_A)} 个高相关特征 → {len(FEAT_A_filtered)} 列")

# 确保列存在
FEAT_B_filtered = [c for c in FEAT_B_filtered if c in df_feat.columns]
FEAT_A_filtered = [c for c in FEAT_A_filtered if c in df_feat.columns]

# 图2: 相关性热图（Top 20特征）
top20_cols = sorted(pearson_corr, key=lambda x: abs(pearson_corr.get(x,0)), reverse=True)[:10]
top20_cols = [c for c in top20_cols if c in df_feat.columns]
if len(top20_cols) >= 3:
    corr_sub = df_feat[top20_cols + [CO_COL]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_sub.values, cmap='RdYlBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(corr_sub.columns)))
    ax.set_yticks(range(len(corr_sub.columns)))
    ax.set_xticklabels(corr_sub.columns, rotation=45, ha='right', fontname=FONT_NAME, fontsize=8)
    ax.set_yticklabels(corr_sub.columns, fontname=FONT_NAME, fontsize=8)
    ax.set_title('主要特征与CO相关性热图', fontname=FONT_NAME)
    for i in range(len(corr_sub)):
        for j in range(len(corr_sub)):
            ax.text(j, i, f'{corr_sub.values[i,j]:.2f}', ha='center', va='center', fontsize=7)
    plt.tight_layout()
    fig.savefig(f'{OUT_DIR}/P2_Q2_fig2_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  图2已保存: P2_Q2_fig2_correlation.png")

print("  Step5完成")

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: 严格时序划分
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 6] 严格时序划分...")

N = len(df_feat)
# 根据实际数据量自适应
TRAIN_END  = min(1300, int(N * 0.55))
VAL_END    = min(1500, int(N * 0.63))
GAP_END    = min(1600, int(N * 0.67))
TEST_START = GAP_END

print(f"  总样本数: {N}")
print(f"  训练集: 0-{TRAIN_END-1} ({TRAIN_END} 条)")
print(f"  验证集: {TRAIN_END}-{VAL_END-1} ({VAL_END-TRAIN_END} 条)")
print(f"  间隔区: {VAL_END}-{GAP_END-1} ({GAP_END-VAL_END} 条，不用于训练/测试）")
print(f"  测试集: {TEST_START}-{N-1} ({N-TEST_START} 条)")

y_all = df_feat[CO_COL].values

def make_splits(feat_list):
    X_all = df_feat[feat_list].values.astype(float)
    # Scaler只在训练集上拟合
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_all[:TRAIN_END])
    X_va = sc.transform(X_all[TRAIN_END:VAL_END])
    X_te = sc.transform(X_all[TEST_START:])
    y_tr = y_all[:TRAIN_END]
    y_va = y_all[TRAIN_END:VAL_END]
    y_te = y_all[TEST_START:]
    return X_tr, X_va, X_te, y_tr, y_va, y_te, sc

X_tr_B, X_va_B, X_te_B, y_tr, y_va, y_te, sc_B = make_splits(FEAT_B_filtered)
X_tr_A, X_va_A, X_te_A, _,    _,    _,    sc_A = make_splits(FEAT_A_filtered)

print("  Step6完成")

# ─────────────────────────────────────────────────────────────────────────────
# Step 7: PSO超参数搜索（特征集B，验证集RMSE）
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 7] PSO超参数搜索（25粒子 × 30迭代，验证集RMSE适应度）...")

N_PART = 25
N_ITER = 30
W_MAX, W_MIN = 0.9, 0.4
C1 = C2 = 2.0

# 搜索空间: [num_leaves, max_depth, lr, n_estimators, min_child_samples, subsample, colsample, reg_alpha, reg_lambda]
LB = [16,   2, 0.01,  50,  5, 0.5, 0.5,  0.0, 0.0]
UB = [256,  8, 0.2,  800, 80, 1.0, 1.0,  2.0, 2.0]
DIM = len(LB)
LB_arr = np.array(LB, dtype=float)
UB_arr = np.array(UB, dtype=float)
V_MAX  = (UB_arr - LB_arr) * 0.2

def decode_params(x):
    return {
        'num_leaves':        int(np.clip(round(x[0]), 16, 256)),
        'max_depth':         int(np.clip(round(x[1]), 2, 8)),
        'learning_rate':     float(np.clip(x[2], 0.005, 0.3)),
        'n_estimators':      int(np.clip(round(x[3]), 50, 1000)),
        'min_child_samples': int(np.clip(round(x[4]), 5, 80)),
        'subsample':         float(np.clip(x[5], 0.4, 1.0)),
        'colsample_bytree':  float(np.clip(x[6], 0.4, 1.0)),
        'reg_alpha':         float(np.clip(x[7], 0.0, 5.0)),
        'reg_lambda':        float(np.clip(x[8], 0.0, 5.0)),
    }

def fitness_fn(x):
    params = decode_params(x)
    params.update({'random_state': 42, 'verbose': -1, 'n_jobs': -1,
                   'subsample_freq': 1})
    m = lgb.LGBMRegressor(**params)
    m.fit(X_tr_B, y_tr,
          eval_set=[(X_va_B, y_va)],
          callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
    pred = m.predict(X_va_B)
    return np.sqrt(mean_squared_error(y_va, pred))

# 初始化粒子
rng_pso = np.random.default_rng(0)
pos = rng_pso.uniform(LB_arr, UB_arr, (N_PART, DIM))
vel = rng_pso.uniform(-V_MAX, V_MAX, (N_PART, DIM))

pbest      = pos.copy()
pbest_fit  = np.array([fitness_fn(pos[i]) for i in range(N_PART)])
gbest_idx  = int(np.argmin(pbest_fit))
gbest      = pbest[gbest_idx].copy()
gbest_fit  = pbest_fit[gbest_idx]
pso_curve  = []

print(f"  初始最优验证RMSE: {gbest_fit:.4f}")

for it in range(1, N_ITER + 1):
    w = W_MAX - (W_MAX - W_MIN) * it / N_ITER
    for i in range(N_PART):
        r1, r2 = rng_pso.random(DIM), rng_pso.random(DIM)
        vel[i] = (w * vel[i]
                  + C1 * r1 * (pbest[i] - pos[i])
                  + C2 * r2 * (gbest   - pos[i]))
        vel[i] = np.clip(vel[i], -V_MAX, V_MAX)
        pos[i] = np.clip(pos[i] + vel[i], LB_arr, UB_arr)
        fit    = fitness_fn(pos[i])
        if fit < pbest_fit[i]:
            pbest_fit[i] = fit
            pbest[i]     = pos[i].copy()
        if fit < gbest_fit:
            gbest_fit = fit
            gbest     = pos[i].copy()
    pso_curve.append(gbest_fit)
    if it % 10 == 0:
        print(f"  迭代 {it:3d}/{N_ITER}  最优验证RMSE = {gbest_fit:.4f}")

best_params = decode_params(gbest)
best_params.update({'random_state': 42, 'verbose': -1, 'n_jobs': -1, 'subsample_freq': 1})
print(f"\n  PSO最优超参数:")
for k, v in best_params.items():
    if k not in ['random_state', 'verbose', 'n_jobs', 'subsample_freq']:
        print(f"    {k}: {v}")

# 图3: PSO收敛曲线
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(pso_curve)+1), pso_curve, 'b-o', ms=4)
ax.set_xlabel('迭代次数', fontname=FONT_NAME)
ax.set_ylabel('验证集 RMSE', fontname=FONT_NAME)
ax.set_title('PSO超参数搜索收敛曲线', fontname=FONT_NAME)
plt.tight_layout()
fig.savefig(f'{OUT_DIR}/P2_Q2_fig3_PSO_convergence.png', dpi=150, bbox_inches='tight')
plt.close()
print("  图3已保存: P2_Q2_fig3_PSO_convergence.png")

# ─────────────────────────────────────────────────────────────────────────────
# Step 8: 训练模型A & B
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 8] 训练模型A（PSO超参 + early stopping）& 模型B（固定超参）...")

# 模型A：PSO超参，特征集A（含CO历史）
model_A = lgb.LGBMRegressor(**best_params)
model_A.fit(X_tr_A, y_tr,
            eval_set=[(X_va_A, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])

pred_A_tr = model_A.predict(X_tr_A)
pred_A_te = model_A.predict(X_te_A)

rmse_A_tr = np.sqrt(mean_squared_error(y_tr, pred_A_tr))
rmse_A_te = np.sqrt(mean_squared_error(y_te, pred_A_te))
mae_A_te  = mean_absolute_error(y_te, pred_A_te)
r2_A_tr   = r2_score(y_tr, pred_A_tr)
r2_A_te   = r2_score(y_te, pred_A_te)
mape_A_te = np.mean(np.abs((y_te - pred_A_te) / (y_te + 1e-9))) * 100

print(f"  模型A（含CO历史）:")
print(f"    训练集: RMSE={rmse_A_tr:.2f}  R²={r2_A_tr:.4f}")
print(f"    测试集: RMSE={rmse_A_te:.2f}  MAE={mae_A_te:.2f}  R²={r2_A_te:.4f}  MAPE={mape_A_te:.2f}%")

# 模型B：固定超参，特征集B（无CO历史，物理可解释）
fixed_params_B = {
    'num_leaves': 63, 'max_depth': 6, 'learning_rate': 0.05,
    'n_estimators': 500, 'min_child_samples': 20,
    'subsample': 0.8, 'subsample_freq': 1, 'colsample_bytree': 0.8,
    'reg_alpha': 0.1, 'reg_lambda': 0.1,
    'random_state': 42, 'verbose': -1, 'n_jobs': -1
}
model_B = lgb.LGBMRegressor(**fixed_params_B)
model_B.fit(X_tr_B, y_tr,
            eval_set=[(X_va_B, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])

pred_B_tr = model_B.predict(X_tr_B)
pred_B_te = model_B.predict(X_te_B)

rmse_B_tr = np.sqrt(mean_squared_error(y_tr, pred_B_tr))
rmse_B_te = np.sqrt(mean_squared_error(y_te, pred_B_te))
mae_B_te  = mean_absolute_error(y_te, pred_B_te)
r2_B_tr   = r2_score(y_tr, pred_B_tr)
r2_B_te   = r2_score(y_te, pred_B_te)
mape_B_te = np.mean(np.abs((y_te - pred_B_te) / (y_te + 1e-9))) * 100

print(f"  模型B（物理特征）:")
print(f"    训练集: RMSE={rmse_B_tr:.2f}  R²={r2_B_tr:.4f}")
print(f"    测试集: RMSE={rmse_B_te:.2f}  MAE={mae_B_te:.2f}  R²={r2_B_te:.4f}  MAPE={mape_B_te:.2f}%")

# 图4: 预测时序对比（模型A测试集）
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
idx_te = np.arange(len(y_te))
axes[0].plot(idx_te, y_te, 'k-', lw=0.8, label='实测值', alpha=0.7)
axes[0].plot(idx_te, pred_A_te, 'r-', lw=0.8, label=f'模型A预测 R²={r2_A_te:.3f}', alpha=0.8)
axes[0].set_title('模型A（PSO超参 + CO历史特征）— 测试集预测', fontname=FONT_NAME)
axes[0].set_ylabel('CO浓度 (ppm)', fontname=FONT_NAME)
axes[0].legend(prop={'family': FONT_NAME})
axes[1].plot(idx_te, y_te, 'k-', lw=0.8, label='实测值', alpha=0.7)
axes[1].plot(idx_te, pred_B_te, 'b-', lw=0.8, label=f'模型B预测 R²={r2_B_te:.3f}', alpha=0.8)
axes[1].set_title('模型B（固定超参 + 物理特征）— 测试集预测', fontname=FONT_NAME)
axes[1].set_ylabel('CO浓度 (ppm)', fontname=FONT_NAME)
axes[1].set_xlabel('测试集时间步', fontname=FONT_NAME)
axes[1].legend(prop={'family': FONT_NAME})
plt.tight_layout()
fig.savefig(f'{OUT_DIR}/P2_Q2_fig4_prediction.png', dpi=150, bbox_inches='tight')
plt.close()
print("  图4已保存: P2_Q2_fig4_prediction.png")

# 图5: 模型A vs B 对比（散点）
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, pred, r2, label, color in [
    (axes[0], pred_A_te, r2_A_te, '模型A', 'tomato'),
    (axes[1], pred_B_te, r2_B_te, '模型B', 'steelblue'),
]:
    ax.scatter(y_te, pred, s=5, alpha=0.5, color=color)
    lo = min(y_te.min(), pred.min())
    hi = max(y_te.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
    ax.set_xlabel('实测CO (ppm)', fontname=FONT_NAME)
    ax.set_ylabel('预测CO (ppm)', fontname=FONT_NAME)
    ax.set_title(f'{label} 散点图  R²={r2:.4f}', fontname=FONT_NAME)
plt.tight_layout()
fig.savefig(f'{OUT_DIR}/P2_Q2_fig5_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  图5已保存: P2_Q2_fig5_model_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# Step 9: SHAP分析（模型B, 200测试样本）+ 偏依赖图
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 9] SHAP分析（模型B, 200测试样本）...")

import shap

N_SHAP = min(200, len(X_te_B))
X_shap = X_te_B[:N_SHAP]

explainer  = shap.TreeExplainer(model_B)
shap_vals  = explainer.shap_values(X_shap)

# 特征重要度（平均|SHAP|）
mean_abs_shap = np.abs(shap_vals).mean(axis=0)
feat_imp_df   = pd.DataFrame({'feature': FEAT_B_filtered, 'shap_mean': mean_abs_shap})
feat_imp_df   = feat_imp_df.sort_values('shap_mean', ascending=False)
top20_feat    = feat_imp_df.head(20)

# 图6: SHAP特征重要度（Top 20）
fig, ax = plt.subplots(figsize=(10, 7))
colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(top20_feat)))
bars = ax.barh(range(len(top20_feat)), top20_feat['shap_mean'].values[::-1], color=colors[::-1])
ax.set_yticks(range(len(top20_feat)))
ax.set_yticklabels(top20_feat['feature'].values[::-1], fontname=FONT_NAME, fontsize=9)
ax.set_xlabel('平均|SHAP值|', fontname=FONT_NAME)
ax.set_title('模型B特征重要度（SHAP，Top 20）', fontname=FONT_NAME)
plt.tight_layout()
fig.savefig(f'{OUT_DIR}/P2_Q2_fig6_SHAP_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  图6已保存: P2_Q2_fig6_SHAP_importance.png")

# 区域贡献分析
region_map = {}
for col in FEAT_B_filtered:
    for rname, idxs in region_labels.items():
        region_neg_cols_set = {NEG_COLS[i] for i in idxs if i < len(NEG_COLS)}
        if any(rc.split('_lagged')[0] in region_neg_cols_set or rc in col for rc in [col]):
            pass
    # 简单分组：按风箱编号
    if '#风箱' in col:
        num_str = col.split('#')[0]
        try:
            num = int(num_str)
            if 1 <= num <= 6:
                region_map[col] = '前段(1-6#)'
            elif 7 <= num <= 12:
                region_map[col] = '中段(7-12#)'
            else:
                region_map[col] = '后段(13-18#)'
        except ValueError:
            region_map[col] = '其他'
    elif 'PC' in col:
        region_map[col] = 'PCA主成分'
    elif 'CO_' in col:
        region_map[col] = 'CO历史'
    else:
        region_map[col] = '其他工艺'

region_contrib = {}
for i, col in enumerate(FEAT_B_filtered):
    rg = region_map.get(col, '其他')
    region_contrib[rg] = region_contrib.get(rg, 0) + mean_abs_shap[i]

print(f"  区域贡献分析:")
for rg, val in sorted(region_contrib.items(), key=lambda x: -x[1]):
    print(f"    {rg}: {val:.3f}")

# 偏依赖图（6个关键特征）
# 选取：3#/10#/17#风箱温度, PC1/PC2, 1#大烟道负压/温度
pdp_candidates = [
    '3#风箱温度均值', '10#风箱温度均值', '17#风箱温度均值',
    'PC1', 'PC2',
    '1#大烟道负压', '1#大烟道温度',
]
pdp_feats = [f for f in pdp_candidates if f in FEAT_B_filtered]
# 补充至6个
if len(pdp_feats) < 6:
    for f in top20_feat['feature'].values:
        if f not in pdp_feats:
            pdp_feats.append(f)
        if len(pdp_feats) >= 6:
            break
pdp_feats = pdp_feats[:6]

print(f"\n[Step 9b] 偏依赖分析（{len(pdp_feats)} 个特征）...")

X_pdp_base = X_te_B.copy()
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()
for pi, fname in enumerate(pdp_feats):
    if fname not in FEAT_B_filtered:
        axes[pi].axis('off')
        continue
    fi = FEAT_B_filtered.index(fname)
    grid = np.linspace(X_te_B[:, fi].min(), X_te_B[:, fi].max(), 30)
    pdp_vals = []
    for gv in grid:
        X_tmp = X_pdp_base.copy()
        X_tmp[:, fi] = gv
        pdp_vals.append(model_B.predict(X_tmp).mean())
    axes[pi].plot(grid, pdp_vals, 'b-o', ms=3)
    axes[pi].set_xlabel(f'{fname}（标准化）', fontname=FONT_NAME, fontsize=9)
    axes[pi].set_ylabel('平均预测CO (ppm)', fontname=FONT_NAME, fontsize=9)
    axes[pi].set_title(f'PDP: {fname}', fontname=FONT_NAME, fontsize=9)
    axes[pi].grid(alpha=0.3)
plt.suptitle('偏依赖分析（部分依赖图）', fontname=FONT_NAME, fontsize=12)
plt.tight_layout()
fig.savefig(f'{OUT_DIR}/P2_Q2_fig7_PDP.png', dpi=150, bbox_inches='tight')
plt.close()
print("  图7已保存: P2_Q2_fig7_PDP.png")

# 大烟道特征专项分析
flue_cols = [c for c in df_feat.columns if '大烟道' in c and c != CO_COL]
print(f"\n  大烟道特征: {flue_cols}")

# 图8: 汇总表格图
metrics_data = {
    '模型': ['模型A\n(PSO超参+CO历史)', '模型B\n(固定超参+物理)'],
    '训练R²':  [f'{r2_A_tr:.4f}', f'{r2_B_tr:.4f}'],
    '测试R²':  [f'{r2_A_te:.4f}', f'{r2_B_te:.4f}'],
    '测试RMSE': [f'{rmse_A_te:.2f}', f'{rmse_B_te:.2f}'],
    '测试MAE':  [f'{mae_A_te:.2f}', f'{mae_B_te:.2f}'],
    'MAPE(%)': [f'{mape_A_te:.2f}', f'{mape_B_te:.2f}'],
}
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')
cols = list(metrics_data.keys())
rows = [[metrics_data[c][i] for c in cols] for i in range(2)]
tbl = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.2, 2.0)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#4472C4')
        cell.set_text_props(color='white', fontproperties=fm.FontProperties(family=FONT_NAME))
    else:
        cell.set_facecolor('#DCE6F1' if r % 2 == 0 else 'white')
        cell.set_text_props(fontproperties=fm.FontProperties(family=FONT_NAME))
ax.set_title('模型性能汇总', fontname=FONT_NAME, fontsize=13, pad=20)
plt.tight_layout()
fig.savefig(f'{OUT_DIR}/P2_Q2_fig8_summary_table.png', dpi=150, bbox_inches='tight')
plt.close()
print("  图8已保存: P2_Q2_fig8_summary_table.png")

# ─────────────────────────────────────────────────────────────────────────────
# 论文写作数据汇总
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("  论文写作数据汇总")
print("=" * 65)
print(f"  数据来源: {'真实数据' if DATA_REAL else '合成数据（演示）'}")
print()
print("【数据清洗结果】")
print(f"  原始样本数: {n0}")
print(f"  故障时段剔除后: {n0 - (fault_end - fault_start + 1)}")
print(f"  清洗后最终: {n_clean} 行")
print()
print("【时滞分析结果（前6个风箱）】")
for col in NEG_COLS[:6]:
    print(f"  {col}: {lags_dict[col]} 步（{lags_dict[col]*2} 秒）")
print()
print("【PCA降维结果】")
print(f"  保留主成分数: {n_comp}")
print(f"  累计解释方差: {cumvar[n_comp-1]*100:.2f}%")
for i in range(n_comp):
    print(f"  PC{i+1}: {pca_final.explained_variance_ratio_[i]*100:.2f}%")
print()
print("【数据集划分】")
print(f"  训练集: {TRAIN_END} 条")
print(f"  验证集: {VAL_END - TRAIN_END} 条")
print(f"  间隔区: {GAP_END - VAL_END} 条")
print(f"  测试集: {N - TEST_START} 条")
print()
print("【模型A（PSO超参 + CO历史特征，在线1步预测）】")
print(f"  特征数: {len(FEAT_A_filtered)}")
print(f"  训练集: RMSE={rmse_A_tr:.2f}  R²={r2_A_tr:.4f}")
print(f"  测试集: RMSE={rmse_A_te:.2f}  MAE={mae_A_te:.2f}  R²={r2_A_te:.4f}  MAPE={mape_A_te:.2f}%")
print(f"  ΔR²（训练-测试）= {r2_A_tr - r2_A_te:.4f}")
print()
print("【模型B（固定超参 + 物理特征，机理可解释）】")
print(f"  特征数: {len(FEAT_B_filtered)}")
print(f"  训练集: RMSE={rmse_B_tr:.2f}  R²={r2_B_tr:.4f}")
print(f"  测试集: RMSE={rmse_B_te:.2f}  MAE={mae_B_te:.2f}  R²={r2_B_te:.4f}  MAPE={mape_B_te:.2f}%")
print(f"  ΔR²（训练-测试）= {r2_B_tr - r2_B_te:.4f}")
print()
print("【SHAP特征重要度（Top 10）】")
for _, row in feat_imp_df.head(10).iterrows():
    print(f"  {row['feature']}: {row['shap_mean']:.4f}")
print()
print("【区域贡献（SHAP）】")
for rg, val in sorted(region_contrib.items(), key=lambda x: -x[1]):
    print(f"  {rg}: {val:.4f}")
print()
print("【PSO最优超参数（模型A用）】")
for k, v in best_params.items():
    if k not in ['random_state', 'verbose', 'n_jobs', 'subsample_freq']:
        print(f"  {k}: {v}")
print()
print("=" * 65)
print("  ===全部完成===")
print("=" * 65)
