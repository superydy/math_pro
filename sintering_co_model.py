#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
烧结大烟道外排CO浓度预测与风箱负压优化
问题2: PSO-LightGBM预测模型
问题3: 约束PSO风箱负压优化
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.signal import correlate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 中文字体配置
# ============================================================
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

OUTPUT_DIR = '/home/user/math_pro'
DATA_FILE = os.path.join(OUTPUT_DIR, '附件1_原始数据.xlsx')

# ============================================================
# Step 0: 合成数据生成（真实文件不存在时使用）
# ============================================================

def generate_synthetic_data(n=2442, seed=42):
    """生成与真实数据结构一致的合成数据"""
    np.random.seed(seed)
    t = np.arange(n)

    speed = 1.8 + 0.3 * np.sin(t / 200) + 0.05 * np.random.randn(n)

    # 18个风箱负压（kPa，负值，沿炉床位置递变）
    neg_pressures = np.zeros((n, 18))
    for i in range(18):
        trend = -6.0 - i * 0.4
        wave = 2.5 * np.sin(t / 250 + i * 0.4)
        neg_pressures[:, i] = trend + wave + 0.4 * np.random.randn(n)

    # 18个风箱南北侧温度（°C）
    temps_n = np.zeros((n, 18))
    temps_s = np.zeros((n, 18))
    for i in range(18):
        base = 70 + i * 18 + 25 * np.sin(t / 400 + i * 0.25)
        temps_n[:, i] = base + 3 * np.random.randn(n)
        temps_s[:, i] = base + 4 + 3 * np.random.randn(n)

    # 大烟道负压和温度
    flue_neg1 = -11.5 + 1.8 * np.sin(t / 200) + 0.25 * np.random.randn(n)
    flue_neg2 = -10.8 + 1.8 * np.sin(t / 200 + 0.4) + 0.25 * np.random.randn(n)
    flue_tmp1 = 155 + 28 * np.sin(t / 320) + 4 * np.random.randn(n)
    flue_tmp2 = 148 + 28 * np.sin(t / 320 + 0.3) + 4 * np.random.randn(n)

    # CO浓度（ppm）：与前几段负压有时滞相关
    co = (2600
          + 180 * np.sin(t / 550)
          - 35 * np.roll(neg_pressures[:, 0], 25)
          - 22 * np.roll(neg_pressures[:, 3], 18)
          - 15 * np.roll(neg_pressures[:, 7], 12)
          + 0.3 * speed * 100
          + 55 * np.random.randn(n))
    co = np.clip(co, 600, 5500)

    # 组合列名
    cols = ['序列', '烧结机机速L1设定']
    cols += [f'{i}#风箱负压' for i in range(1, 19)]
    cols += [f'{i}#风箱北侧温度' for i in range(1, 19)]
    cols += [f'{i}#风箱南侧温度' for i in range(1, 19)]
    cols += ['1#大烟道负压', '2#大烟道负压', '1#大烟道温度', '2#大烟道温度']
    cols.append('烧结大烟道外排CO浓度')

    data = np.column_stack([
        np.arange(1, n + 1),
        speed,
        neg_pressures,
        temps_n,
        temps_s,
        flue_neg1, flue_neg2,
        flue_tmp1, flue_tmp2,
        co
    ])
    return pd.DataFrame(data, columns=cols)


# ============================================================
# Step 1: 数据预处理
# ============================================================
print("=" * 65)
print("  烧结CO预测与风箱负压优化（PSO-LightGBM + 约束PSO）")
print("=" * 65)
print("\n[Step 1] 数据预处理...")

if os.path.exists(DATA_FILE):
    print(f"  读取真实数据: {DATA_FILE}")
    df_raw = pd.read_excel(DATA_FILE, header=1)
    print(f"  原始数据形状: {df_raw.shape}")
    using_real = True
else:
    print("  ⚠ 真实数据文件未找到，使用合成数据演示完整流程")
    df_raw = generate_synthetic_data()
    print(f"  合成数据形状: {df_raw.shape}")
    using_real = False

# -- 识别关键列 --
def find_col(df, keywords, exclude=None):
    """按关键词列表查找列名（所有关键词均命中）"""
    result = []
    for c in df.columns:
        cs = str(c)
        if all(k in cs for k in keywords):
            if exclude is None or not any(e in cs for e in exclude):
                result.append(c)
    return result

# CO列
co_candidates = find_col(df_raw, ['CO']) + find_col(df_raw, ['co'])
co_col = co_candidates[0] if co_candidates else df_raw.columns[-1]

# 风箱负压列（排除大烟道）
neg_cols = find_col(df_raw, ['风箱', '负压'])
if not neg_cols:
    neg_cols = find_col(df_raw, ['负压'], exclude=['大烟道'])

# 温度列
temp_n_cols = find_col(df_raw, ['北侧', '温度'])
temp_s_cols = find_col(df_raw, ['南侧', '温度'])

# 机速列
speed_cols = find_col(df_raw, ['机速']) or find_col(df_raw, ['速'])

print(f"  CO列: {co_col}")
print(f"  风箱负压列({len(neg_cols)}): {neg_cols}")
print(f"  北侧温度列({len(temp_n_cols)}): {temp_n_cols[:3]}...")
print(f"  南侧温度列({len(temp_s_cols)}): {temp_s_cols[:3]}...")

df = df_raw.copy()
# 保证数值类型
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df.dropna(subset=[co_col], inplace=True)
df.reset_index(drop=True, inplace=True)

# 1%~99% 分位数剔除CO异常值
q01 = df[co_col].quantile(0.01)
q99 = df[co_col].quantile(0.99)
n_before = len(df)
df = df[(df[co_col] >= q01) & (df[co_col] <= q99)].copy().reset_index(drop=True)
print(f"  CO异常值剔除: {n_before} -> {len(df)} (剔除 {n_before - len(df)} 条)")

# 南北侧温度取均值合并
avg_temp_cols = []
for i, (nc, sc) in enumerate(zip(temp_n_cols, temp_s_cols), 1):
    new_col = f'风箱{i}温度均值'
    df[new_col] = (df[nc] + df[sc]) / 2
    avg_temp_cols.append(new_col)
print(f"  南北温度取均值合并: {len(avg_temp_cols)} 列")
print(f"  Step1完成，数据形状: {df.shape}")


# ============================================================
# Step 2: 特征工程
# ============================================================
print("\n[Step 2] 特征工程...")

MAX_LAG_SEARCH = 300
co_series = df[co_col].values
lags_result = {}

print(f"  互相关时滞搜索（最大 {MAX_LAG_SEARCH} 步）...")
for col in neg_cols:
    x = df[col].ffill().values.astype(float)
    x_norm = (x - x.mean()) / (x.std() + 1e-8)
    co_norm = (co_series - co_series.mean()) / (co_series.std() + 1e-8)
    # 计算全互相关，只看正向（负压领先CO方向）
    full_corr = correlate(co_norm, x_norm, mode='full')
    mid = len(full_corr) // 2
    # 取 lag = 0 ~ MAX_LAG_SEARCH
    search_corr = full_corr[mid: mid + MAX_LAG_SEARCH + 1]
    best_lag = int(np.argmax(np.abs(search_corr)))
    lags_result[col] = best_lag

print("  时滞结果（步数，间隔2秒）:")
for c, lg in lags_result.items():
    print(f"    {c}: {lg} 步 = {lg * 2} 秒")

# 在 df 上构建特征
feat_df = df.copy()

# 时滞配准
for col, lag in lags_result.items():
    shifted_col = f'{col}_lagged'
    feat_df[shifted_col] = feat_df[col].shift(lag)

# CO自身历史特征
for lag_n in [1, 3, 5, 10, 20, 30]:
    feat_df[f'CO_lag{lag_n}'] = feat_df[co_col].shift(lag_n)
feat_df['CO_roll5_mean'] = feat_df[co_col].rolling(5).mean()
feat_df['CO_roll10_mean'] = feat_df[co_col].rolling(10).mean()
feat_df['CO_roll5_std'] = feat_df[co_col].rolling(5).std()
feat_df['CO_roll10_std'] = feat_df[co_col].rolling(10).std()

# 各风箱负压1步差分
for col in neg_cols:
    feat_df[f'{col}_diff1'] = feat_df[col].diff(1)

# 区域汇总特征
n_boxes = len(neg_cols)
if n_boxes >= 18:
    front_c = neg_cols[:5]
    mid_c   = neg_cols[5:15]
    rear_c  = neg_cols[15:18]
else:
    # 按比例划分
    f_end = max(1, n_boxes // 4)
    r_start = max(f_end + 1, n_boxes - max(1, n_boxes // 6))
    front_c = neg_cols[:f_end]
    mid_c   = neg_cols[f_end:r_start]
    rear_c  = neg_cols[r_start:]

feat_df['neg_front_mean'] = feat_df[front_c].mean(axis=1)
feat_df['neg_mid_mean']   = feat_df[mid_c].mean(axis=1)
feat_df['neg_rear_mean']  = feat_df[rear_c].mean(axis=1)
feat_df['neg_range']      = feat_df[neg_cols].max(axis=1) - feat_df[neg_cols].min(axis=1)

# 加入均值温度列和机速列
extra_cols = avg_temp_cols + speed_cols
for c in extra_cols:
    if c in df.columns and c not in feat_df.columns:
        feat_df[c] = df[c]

feat_df.dropna(inplace=True)
feat_df.reset_index(drop=True, inplace=True)
print(f"  Step2完成，特征工程后 {feat_df.shape[1]} 列，{len(feat_df)} 行")


# ============================================================
# Step 3: PCA 降维（仅对18个风箱负压）
# ============================================================
print("\n[Step 3] PCA降维（18个风箱负压 -> 主成分）...")

neg_data = feat_df[neg_cols].values
scaler_pca = StandardScaler()
neg_scaled = scaler_pca.fit_transform(neg_data)

pca_probe = PCA()
pca_probe.fit(neg_scaled)
cumvar = np.cumsum(pca_probe.explained_variance_ratio_)
n_components = int(np.searchsorted(cumvar, 0.95)) + 1
print(f"  保留主成分数: {n_components}（累计解释方差 >= 95%）")
for i in range(n_components):
    print(f"    PC{i+1}: {pca_probe.explained_variance_ratio_[i]*100:.2f}%  "
          f"(累计 {cumvar[i]*100:.2f}%)")

pca_model = PCA(n_components=n_components)
neg_pca = pca_model.fit_transform(neg_scaled)
pca_cols = [f'PC{i+1}' for i in range(n_components)]

# 图1：PCA解释方差比
fig1, ax1 = plt.subplots(figsize=(10, 5))
evr = pca_model.explained_variance_ratio_ * 100
cumevr = np.cumsum(evr)
bars = ax1.bar(range(1, n_components + 1), evr,
               color='#2E86AB', alpha=0.85, edgecolor='white', linewidth=0.5)
ax1.plot(range(1, n_components + 1), cumevr, 'o-',
         color='#E84855', linewidth=2, markersize=6, label='累计方差比')
ax1.axhline(95, color='#555', linestyle='--', alpha=0.6, label='95% 阈值')
for bar, v in zip(bars, evr):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.4,
             f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
ax1.set_xlabel('主成分编号', fontsize=12)
ax1.set_ylabel('解释方差比 (%)', fontsize=12)
ax1.set_title('图1：PCA各主成分解释方差比（风箱负压）', fontsize=13, fontweight='bold', pad=10)
ax1.legend(fontsize=10)
ax1.set_ylim(0, max(cumevr[-1], 100) * 1.08)
ax1.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, 'P2_fig1_PCA.png'), dpi=150, bbox_inches='tight')
plt.close(fig1)
print("  图1已保存: P2_fig1_PCA.png")

# 将PCA主成分替代原始负压列
pca_df = pd.DataFrame(neg_pca, columns=pca_cols, index=feat_df.index)

# 确定保留的列：去掉原始负压列、南北侧原始温度、时滞配准重复列，保留target
lagged_neg_cols = [f'{c}_lagged' for c in neg_cols]
drop_set = set(neg_cols) | set(temp_n_cols) | set(temp_s_cols)
keep_base = [c for c in feat_df.columns if c not in drop_set and c != co_col]

feat_combined = pd.concat(
    [feat_df[keep_base].reset_index(drop=True), pca_df.reset_index(drop=True)],
    axis=1
)
target_arr = feat_df[co_col].values

print(f"  Step3完成，PCA替换后特征数: {feat_combined.shape[1]}")


# ============================================================
# Step 4: 相关性过滤（阈值 0.95）
# ============================================================
print("\n[Step 4] 相关性过滤（相关系数阈值 = 0.95）...")

feat_combined = feat_combined.select_dtypes(include=[np.number])
feat_combined.dropna(axis=1, how='all', inplace=True)
feat_combined.fillna(feat_combined.mean(), inplace=True)

corr_mat = feat_combined.corr().abs()
upper_tri = corr_mat.where(
    np.triu(np.ones(corr_mat.shape, dtype=bool), k=1)
)
drop_corr = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
feat_combined.drop(columns=drop_corr, inplace=True, errors='ignore')
print(f"  剔除高相关特征 {len(drop_corr)} 个")
print(f"  Step4完成，最终特征数: {feat_combined.shape[1]}")

feature_names = feat_combined.columns.tolist()
X = feat_combined.values
y = target_arr[:len(feat_combined)]


# ============================================================
# Step 5: PSO 自动调参（30粒子 × 40迭代）
# ============================================================
print("\n[Step 5] PSO超参数搜索（30粒子，40迭代）...")

param_names_pso = [
    'num_leaves', 'max_depth', 'learning_rate', 'n_estimators',
    'min_child_samples', 'subsample', 'colsample_bytree',
    'reg_alpha', 'reg_lambda'
]
lb_pso = np.array([20,  3, 0.01, 100,  5, 0.6, 0.6, 0.0, 0.0])
ub_pso = np.array([200, 12, 0.30, 800, 50, 1.0, 1.0, 1.0, 1.0])
n_dim_pso = len(lb_pso)


def decode_params(vec):
    """将PSO向量解码为LightGBM参数字典"""
    return {
        'num_leaves':        max(5, int(round(vec[0]))),
        'max_depth':         max(2, int(round(vec[1]))),
        'learning_rate':     float(vec[2]),
        'n_estimators':      max(10, int(round(vec[3]))),
        'min_child_samples': max(1, int(round(vec[4]))),
        'subsample':         float(np.clip(vec[5], 0.6, 1.0)),
        'colsample_bytree':  float(np.clip(vec[6], 0.6, 1.0)),
        'reg_alpha':         float(np.clip(vec[7], 0.0, 1.0)),
        'reg_lambda':        float(np.clip(vec[8], 0.0, 1.0)),
    }


def lgb_cv_rmse(vec):
    """5折时序CV RMSE（适应度函数，越小越好）"""
    params = decode_params(vec)
    model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1, n_jobs=-1)
    kf = KFold(n_splits=5, shuffle=False)
    rmses = []
    for tr_idx, val_idx in kf.split(X):
        model.fit(X[tr_idx], y[tr_idx])
        pred = model.predict(X[val_idx])
        rmses.append(np.sqrt(mean_squared_error(y[val_idx], pred)))
    return float(np.mean(rmses))


# PSO 初始化
N_P = 30
N_ITER = 40
W_MAX, W_MIN = 0.9, 0.4
C1 = C2 = 2.0
np.random.seed(42)

pos = lb_pso + np.random.rand(N_P, n_dim_pso) * (ub_pso - lb_pso)
vel = np.zeros_like(pos)
v_max = (ub_pso - lb_pso) * 0.2  # 速度限制

pbest_pos = pos.copy()
print("  初始化粒子适应度...")
pbest_val = np.array([lgb_cv_rmse(pos[i]) for i in range(N_P)])
gbest_idx = int(np.argmin(pbest_val))
gbest_pos = pbest_pos[gbest_idx].copy()
gbest_val = float(pbest_val[gbest_idx])
print(f"  初始最优RMSE: {gbest_val:.4f}")

pso_convergence = [gbest_val]

for it in range(1, N_ITER + 1):
    w = W_MAX - (W_MAX - W_MIN) * it / N_ITER
    r1 = np.random.rand(N_P, n_dim_pso)
    r2 = np.random.rand(N_P, n_dim_pso)
    vel = w * vel + C1 * r1 * (pbest_pos - pos) + C2 * r2 * (gbest_pos - pos)
    vel = np.clip(vel, -v_max, v_max)
    pos = np.clip(pos + vel, lb_pso, ub_pso)

    for i in range(N_P):
        val = lgb_cv_rmse(pos[i])
        if val < pbest_val[i]:
            pbest_val[i] = val
            pbest_pos[i] = pos[i].copy()
        if val < gbest_val:
            gbest_val = val
            gbest_pos = pos[i].copy()

    pso_convergence.append(gbest_val)
    if it % 10 == 0:
        print(f"  迭代 {it:3d} / {N_ITER}  最优RMSE = {gbest_val:.4f}")

best_params = decode_params(gbest_pos)
print(f"\n  PSO最优超参数:")
for k, v in best_params.items():
    print(f"    {k}: {v}")

# 图2：PSO收敛曲线
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(range(N_ITER + 1), pso_convergence, 'b-', linewidth=2)
ax2.fill_between(range(N_ITER + 1), pso_convergence, alpha=0.12, color='blue')
ax2.scatter([np.argmin(pso_convergence)], [min(pso_convergence)],
            color='red', s=80, zorder=5, label=f'最优 RMSE={min(pso_convergence):.2f}')
ax2.set_xlabel('迭代次数', fontsize=12)
ax2.set_ylabel('5折CV RMSE', fontsize=12)
ax2.set_title('图2：PSO收敛曲线（LightGBM超参数搜索）', fontsize=13, fontweight='bold', pad=10)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)
plt.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, 'P2_fig2_PSO_convergence.png'), dpi=150, bbox_inches='tight')
plt.close(fig2)
print("  图2已保存: P2_fig2_PSO_convergence.png")


# ============================================================
# Step 6: 最终模型训练与评估
# ============================================================
print("\n[Step 6] 最终模型训练与评估...")

train_size = int(len(X) * 0.8)
X_tr, X_te = X[:train_size], X[train_size:]
y_tr, y_te = y[:train_size], y[train_size:]

final_model = lgb.LGBMRegressor(**best_params, random_state=42, verbose=-1, n_jobs=-1)
final_model.fit(X_tr, y_tr)

y_tr_pred = final_model.predict(X_tr)
y_te_pred = final_model.predict(X_te)


def metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)
    return rmse, mae, r2, mape


tr_rmse, tr_mae, tr_r2, tr_mape = metrics(y_tr, y_tr_pred)
te_rmse, te_mae, te_r2, te_mape = metrics(y_te, y_te_pred)

print(f"  训练集: RMSE={tr_rmse:.2f}  MAE={tr_mae:.2f}  R²={tr_r2:.4f}  MAPE={tr_mape:.2f}%")
print(f"  测试集: RMSE={te_rmse:.2f}  MAE={te_mae:.2f}  R²={te_r2:.4f}  MAPE={te_mape:.2f}%")
print(f"  ΔR²（训练-测试）= {tr_r2 - te_r2:.4f}  "
      f"{'（轻微过拟合）' if tr_r2 - te_r2 > 0.05 else '（泛化正常）'}")

# 图3：CO浓度时序预测
fig3, ax3 = plt.subplots(figsize=(14, 5))
x_axis = range(len(y_te))
ax3.plot(x_axis, y_te,     color='#1f77b4', linewidth=1.0, alpha=0.85, label='真实值')
ax3.plot(x_axis, y_te_pred, color='#d62728', linewidth=1.0, alpha=0.75, label='预测值')
ax3.axhline(2800, color='#2ca02c', linestyle='--', linewidth=1.5, label='限值 2800 ppm')
ax3.set_xlabel('测试集时间步', fontsize=12)
ax3.set_ylabel('CO浓度 (ppm)', fontsize=12)
ax3.set_title('图3：CO浓度预测效果（测试集真实值 vs 预测值）', fontsize=13, fontweight='bold', pad=10)
ax3.legend(fontsize=10, loc='upper right')
ax3.grid(alpha=0.25)
plt.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, 'P2_fig3_CO_prediction.png'), dpi=150, bbox_inches='tight')
plt.close(fig3)
print("  图3已保存: P2_fig3_CO_prediction.png")

# 图4：预测值散点图
fig4, ax4 = plt.subplots(figsize=(6.5, 6.5))
ax4.scatter(y_te, y_te_pred, alpha=0.35, s=14, color='#1f77b4', edgecolors='none')
lim_lo = min(y_te.min(), y_te_pred.min()) * 0.97
lim_hi = max(y_te.max(), y_te_pred.max()) * 1.03
ax4.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'r--', linewidth=1.8, label='y = x')
ax4.text(0.06, 0.90, f'R² = {te_r2:.4f}',
         transform=ax4.transAxes, fontsize=13,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))
ax4.set_xlabel('真实CO浓度 (ppm)', fontsize=12)
ax4.set_ylabel('预测CO浓度 (ppm)', fontsize=12)
ax4.set_title('图4：预测值 vs 真实值散点图', fontsize=13, fontweight='bold', pad=10)
ax4.legend(fontsize=10)
ax4.grid(alpha=0.25)
plt.tight_layout()
fig4.savefig(os.path.join(OUTPUT_DIR, 'P2_fig4_scatter.png'), dpi=150, bbox_inches='tight')
plt.close(fig4)
print("  图4已保存: P2_fig4_scatter.png")

# 图5：特征重要性 Top20
importances = final_model.feature_importances_
top_n = min(20, len(feature_names))
top_idx = np.argsort(importances)[::-1][:top_n]
top_feats = [feature_names[i] for i in top_idx]
top_imp   = importances[top_idx]

fig5, ax5 = plt.subplots(figsize=(11, 6))
colors_imp = ['#E84855' if i == 0 else '#2E86AB' for i in range(top_n)]
ax5.barh(range(top_n), top_imp[::-1], color=colors_imp[::-1], alpha=0.85)
ax5.set_yticks(range(top_n))
ax5.set_yticklabels(top_feats[::-1], fontsize=8.5)
ax5.set_xlabel('特征重要性得分', fontsize=12)
ax5.set_title(f'图5：特征重要性 Top{top_n}', fontsize=13, fontweight='bold', pad=10)
ax5.grid(axis='x', alpha=0.3)
plt.tight_layout()
fig5.savefig(os.path.join(OUTPUT_DIR, 'P2_fig5_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close(fig5)
print("  图5已保存: P2_fig5_feature_importance.png")


# ============================================================
# Step 8: 确定各风箱负压调节范围（5%~95% 分位数）
# ============================================================
print("\n[Step 8] 确定风箱负压调节范围...")

neg_lb_opt = np.array([df[c].quantile(0.05) for c in neg_cols])
neg_ub_opt = np.array([df[c].quantile(0.95) for c in neg_cols])
neg_mean_opt = np.array([df[c].mean() for c in neg_cols])

print(f"  {'风箱':<18} {'下界(5%)':<12} {'均值':<12} {'上界(95%)'}")
for c, lb, m, ub in zip(neg_cols, neg_lb_opt, neg_mean_opt, neg_ub_opt):
    print(f"  {c:<18} {lb:<12.3f} {m:<12.3f} {ub:.3f}")


# ============================================================
# Step 9: 约束PSO优化（60粒子 × 200迭代，最小化CO）
# ============================================================
print("\n[Step 9] 约束PSO优化风箱负压（60粒子，200迭代）...")

# 用于从18个负压值构造特征向量
feat_mean_row = feat_combined.mean()  # 其他特征的均值基线

# PCA列在最终特征中的位置
pca_feat_indices = [feature_names.index(pc) for pc in pca_cols if pc in feature_names]
print(f"  PCA列在特征矩阵中的位置: {pca_feat_indices}")


def neg_to_feature_row(neg_values):
    """将18个风箱负压值映射到最终特征向量（其余取均值）"""
    row = feat_mean_row.copy().values.astype(float)
    neg_arr = np.array(neg_values, dtype=float).reshape(1, -1)
    # 长度对齐
    if neg_arr.shape[1] < len(neg_cols):
        pad = np.zeros((1, len(neg_cols) - neg_arr.shape[1]))
        neg_arr = np.hstack([neg_arr, pad])
    elif neg_arr.shape[1] > len(neg_cols):
        neg_arr = neg_arr[:, :len(neg_cols)]
    neg_scaled_val = scaler_pca.transform(neg_arr)
    pca_val = pca_model.transform(neg_scaled_val)[0]
    for k, idx in enumerate(pca_feat_indices):
        if k < len(pca_val):
            row[idx] = pca_val[k]
    return row.reshape(1, -1)


LAMBDA_ADJ = 100.0  # 相邻差约束惩罚系数
LAMBDA_AMP  = 100.0  # 调整幅度约束惩罚系数


def opt_fitness(neg_vals):
    """约束适应度函数：CO预测 + 软约束惩罚"""
    penalty = 0.0
    # 约束2：相邻风箱负压差 > 3 kPa
    for k in range(len(neg_vals) - 1):
        diff = abs(float(neg_vals[k + 1]) - float(neg_vals[k]))
        if diff > 3.0:
            penalty += LAMBDA_ADJ * (diff - 3.0)
    # 约束3：单个调整幅度 > 20% 当前均值
    for k, (v, m) in enumerate(zip(neg_vals, neg_mean_opt)):
        ratio = abs(v - m) / (abs(m) + 1e-8)
        if ratio > 0.20:
            penalty += LAMBDA_AMP * (ratio - 0.20)

    feat_row = neg_to_feature_row(neg_vals)
    co_pred = float(final_model.predict(feat_row)[0])
    return co_pred + penalty


N_OPT_P = 60
N_OPT_ITER = 200
n_dim_opt = len(neg_cols)
np.random.seed(42)

opt_pos = neg_lb_opt + np.random.rand(N_OPT_P, n_dim_opt) * (neg_ub_opt - neg_lb_opt)
opt_vel = np.zeros_like(opt_pos)
opt_vmax = (neg_ub_opt - neg_lb_opt) * 0.25

opt_pbest_pos = opt_pos.copy()
opt_pbest_val = np.array([opt_fitness(opt_pos[i]) for i in range(N_OPT_P)])
opt_gbest_idx = int(np.argmin(opt_pbest_val))
opt_gbest_pos = opt_pbest_pos[opt_gbest_idx].copy()
opt_gbest_val = float(opt_pbest_val[opt_gbest_idx])
opt_conv_curve = [opt_gbest_val]
print(f"  初始最优目标值: {opt_gbest_val:.2f}")

for it in range(1, N_OPT_ITER + 1):
    w = 0.9 - 0.5 * it / N_OPT_ITER
    r1 = np.random.rand(N_OPT_P, n_dim_opt)
    r2 = np.random.rand(N_OPT_P, n_dim_opt)
    opt_vel = (w * opt_vel
               + 2.0 * r1 * (opt_pbest_pos - opt_pos)
               + 2.0 * r2 * (opt_gbest_pos - opt_pos))
    opt_vel = np.clip(opt_vel, -opt_vmax, opt_vmax)
    opt_pos = np.clip(opt_pos + opt_vel, neg_lb_opt, neg_ub_opt)

    for i in range(N_OPT_P):
        val = opt_fitness(opt_pos[i])
        if val < opt_pbest_val[i]:
            opt_pbest_val[i] = val
            opt_pbest_pos[i] = opt_pos[i].copy()
        if val < opt_gbest_val:
            opt_gbest_val = val
            opt_gbest_pos = opt_pos[i].copy()

    opt_conv_curve.append(opt_gbest_val)
    if it % 50 == 0:
        print(f"  迭代 {it:4d} / {N_OPT_ITER}  当前最优CO = {opt_gbest_val:.2f} ppm")

# 计算纯CO预测（不含惩罚）
co_before_opt = float(final_model.predict(neg_to_feature_row(neg_mean_opt))[0])
co_after_opt  = float(final_model.predict(neg_to_feature_row(opt_gbest_pos))[0])
print(f"\n  优化前CO（均值工况）: {co_before_opt:.2f} ppm")
print(f"  优化后CO（最优负压）: {co_after_opt:.2f} ppm")
print(f"  CO降低: {co_before_opt - co_after_opt:.2f} ppm "
      f"({(co_before_opt - co_after_opt)/max(co_before_opt, 1)*100:.1f}%)")

# 图6：优化PSO收敛曲线
fig6, ax6 = plt.subplots(figsize=(10, 5))
ax6.plot(range(N_OPT_ITER + 1), opt_conv_curve, color='#2ca02c', linewidth=1.8)
ax6.axhline(2800, color='#d62728', linestyle='--', linewidth=1.5, label='限值 2800 ppm')
ax6.set_xlabel('迭代次数', fontsize=12)
ax6.set_ylabel('当前最优目标值 (ppm)', fontsize=12)
ax6.set_title('图6：约束PSO优化收敛曲线（风箱负压优化）', fontsize=13, fontweight='bold', pad=10)
ax6.legend(fontsize=10)
ax6.grid(alpha=0.3)
plt.tight_layout()
fig6.savefig(os.path.join(OUTPUT_DIR, 'P3_fig6_opt_convergence.png'), dpi=150, bbox_inches='tight')
plt.close(fig6)
print("  图6已保存: P3_fig6_opt_convergence.png")

# 图7：风箱负压调整量对比
short_labels = [c.replace('风箱负压', '#').replace('#', f'{i+1}#')
                if '风箱负压' not in c else f'{i+1}#'
                for i, c in enumerate(neg_cols)]
# 简化标签为 "1#", "2#", ...
short_labels = [f'{i+1}#' for i in range(len(neg_cols))]

x_idx = np.arange(len(neg_cols))
w_bar = 0.36
fig7, ax7 = plt.subplots(figsize=(14, 5))
ax7.bar(x_idx - w_bar/2, neg_mean_opt, w_bar, label='优化前均值', color='#1f77b4', alpha=0.85)
ax7.bar(x_idx + w_bar/2, opt_gbest_pos,  w_bar, label='优化后',   color='#d62728', alpha=0.85)
ax7.set_xticks(x_idx)
ax7.set_xticklabels(short_labels, fontsize=9)
ax7.set_xlabel('风箱编号', fontsize=12)
ax7.set_ylabel('负压值 (kPa)', fontsize=12)
ax7.set_title('图7：各风箱负压优化前后对比', fontsize=13, fontweight='bold', pad=10)
ax7.legend(fontsize=10)
ax7.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig7.savefig(os.path.join(OUTPUT_DIR, 'P3_fig7_neg_adjustment.png'), dpi=150, bbox_inches='tight')
plt.close(fig7)
print("  图7已保存: P3_fig7_neg_adjustment.png")

# 图8：优化前后CO对比
fig8, ax8 = plt.subplots(figsize=(7, 5))
cat = ['优化前', '优化后']
vals = [co_before_opt, co_after_opt]
bar_colors = ['#1f77b4', '#d62728']
bars8 = ax8.bar(cat, vals, color=bar_colors, alpha=0.85, width=0.4, edgecolor='white')
ax8.axhline(2800, color='#2ca02c', linestyle='--', linewidth=2, label='限值 2800 ppm')
for bar, val in zip(bars8, vals):
    ax8.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + max(vals) * 0.01,
             f'{val:.1f} ppm', ha='center', va='bottom', fontsize=13, fontweight='bold')
ax8.set_ylabel('CO浓度 (ppm)', fontsize=12)
ax8.set_ylim(0, max(vals) * 1.25)
ax8.set_title('图8：优化前后CO浓度对比', fontsize=13, fontweight='bold', pad=10)
ax8.legend(fontsize=10)
ax8.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig8.savefig(os.path.join(OUTPUT_DIR, 'P3_fig8_CO_comparison.png'), dpi=150, bbox_inches='tight')
plt.close(fig8)
print("  图8已保存: P3_fig8_CO_comparison.png")

# 图9：敏感性分析
print("\n  敏感性分析（各风箱单独扫描，固定其余为均值）...")
sensitivity_arr = []
SCAN_POINTS = 25
for i, col in enumerate(neg_cols):
    scan = np.linspace(neg_lb_opt[i], neg_ub_opt[i], SCAN_POINTS)
    co_scan = []
    for sv in scan:
        nv = neg_mean_opt.copy()
        nv[i] = sv
        co_scan.append(float(final_model.predict(neg_to_feature_row(nv))[0]))
    sensitivity_arr.append(max(co_scan) - min(co_scan))

sensitivity_arr = np.array(sensitivity_arr)
most_sens_idx = int(np.argmax(sensitivity_arr))
print(f"  最敏感风箱: {neg_cols[most_sens_idx]}，CO响应幅度: {sensitivity_arr[most_sens_idx]:.2f} ppm")

fig9, ax9 = plt.subplots(figsize=(14, 5))
bar_colors9 = ['#E84855' if i == most_sens_idx else '#2E86AB' for i in range(len(neg_cols))]
ax9.bar(short_labels, sensitivity_arr, color=bar_colors9, alpha=0.85)
ax9.text(most_sens_idx, sensitivity_arr[most_sens_idx] + sensitivity_arr.max() * 0.02,
         '最敏感', ha='center', va='bottom', color='#E84855', fontsize=10, fontweight='bold')
ax9.set_xlabel('风箱编号', fontsize=12)
ax9.set_ylabel('CO响应幅度 (ppm)', fontsize=12)
ax9.set_title('图9：各风箱负压敏感性分析（CO响应幅度）', fontsize=13, fontweight='bold', pad=10)
ax9.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig9.savefig(os.path.join(OUTPUT_DIR, 'P3_fig9_sensitivity.png'), dpi=150, bbox_inches='tight')
plt.close(fig9)
print("  图9已保存: P3_fig9_sensitivity.png")


# ============================================================
# 论文写作数据汇总
# ============================================================
print("\n" + "=" * 65)
print("  论文写作数据汇总")
print("=" * 65)
print(f"  数据来源: {'真实数据' if using_real else '合成数据（演示）'}")
print(f"\n【数据规模】")
print(f"  原始样本数: {n_before}，清洗后: {len(df)}")
print(f"  最终特征维度: {feat_combined.shape[1]}")
print(f"  训练样本: {train_size}，测试样本: {len(X_te)}")

print(f"\n【时滞分析结果（前5个风箱）】")
for c, lg in list(lags_result.items())[:5]:
    print(f"  {c}: {lg} 步（{lg * 2} 秒）")

print(f"\n【PCA降维结果】")
print(f"  保留主成分数: {n_components}")
print(f"  累计解释方差: {cumvar[n_components-1]*100:.2f}%")
for i in range(n_components):
    print(f"  PC{i+1}: {pca_model.explained_variance_ratio_[i]*100:.2f}%")

print(f"\n【PSO-LightGBM模型性能】")
print(f"  训练集 RMSE={tr_rmse:.2f}  MAE={tr_mae:.2f}  R²={tr_r2:.4f}  MAPE={tr_mape:.2f}%")
print(f"  测试集 RMSE={te_rmse:.2f}  MAE={te_mae:.2f}  R²={te_r2:.4f}  MAPE={te_mape:.2f}%")
print(f"  过拟合指标 ΔR² = {tr_r2 - te_r2:.4f}")

print(f"\n【约束PSO优化结果】")
print(f"  优化前CO: {co_before_opt:.2f} ppm")
print(f"  优化后CO: {co_after_opt:.2f} ppm")
print(f"  降低量: {co_before_opt - co_after_opt:.2f} ppm "
      f"({(co_before_opt - co_after_opt) / max(co_before_opt, 1) * 100:.1f}%)")
print(f"  是否达标(<2800 ppm): {'是 ✓' if co_after_opt < 2800 else '否 (仍需调整)'}")

print(f"\n【最敏感风箱】")
print(f"  {neg_cols[most_sens_idx]}（CO响应幅度 {sensitivity_arr[most_sens_idx]:.2f} ppm）")

print(f"\n【最优风箱负压设定值（优化后）】")
for c, v, m in zip(neg_cols, opt_gbest_pos, neg_mean_opt):
    change = v - m
    print(f"  {c}: {v:.3f} kPa  （原均值 {m:.3f}，调整 {change:+.3f}）")

print("\n" + "=" * 65)
print("  全部完成！共生成9张图表。")
print("=" * 65)
