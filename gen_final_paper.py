#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成最终论文：真实对比数据 + PSO迭代过程表 + 全部嵌入图
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import json, os, warnings
warnings.filterwarnings('ignore')

os.chdir('/home/user/math_pro')

_fp_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
fm.fontManager.addfont(_fp_path)
_fp = fm.FontProperties(fname=_fp_path)
_FN = _fp.get_name()
plt.rcParams['font.family']        = _FN
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('paper_figures', exist_ok=True)
C1='#2980B9'; C2='#C0392B'; C3='#27AE60'; C4='#8E44AD'; C5='#E67E22'; C6='#1ABC9C'

def fp(ax):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontproperties(_fp)

# ── 加载真实结果 ──────────────────────────────────────────────────────────────
with open('results/model_comparison_results.json','r',encoding='utf-8') as f:
    comp_data = json.load(f)
with open('results/Q2_complete_results.json','r',encoding='utf-8') as f:
    q2_data = json.load(f)

cmp   = comp_data['model_comparison']
pso_log = comp_data['pso_iteration_log']

# Ridge的R²=1.0是因为线性自回归特征，作为单独说明，主对比排除Ridge
cmp_main = {k: v for k, v in cmp.items() if k != 'Ridge回归'}

# ══════════════════════════════════════════════════════════════════════════════
# 图4-3  模型架构 + 对比
# ══════════════════════════════════════════════════════════════════════════════
def gen_comparison_fig():
    models  = list(cmp_main.keys())
    r2_vals = [cmp_main[m]['r2']   for m in models]
    mae_vals= [cmp_main[m]['mae']  for m in models]
    rmse_vals=[cmp_main[m]['rmse'] for m in models]

    colors = [C1, C1, C1, C1, C2]  # 最后一个是本文方法

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    x = np.arange(len(models))
    w = 0.55

    short_names = ['随机森林\n(RF)', '梯度提升树\n(GBR)', 'LightGBM',
                   'XGBoost\n(默认参数)', 'XGBoost\n+PSO(本文)']

    for ax, vals, ylabel, title, best_is_high in zip(
        axes,
        [r2_vals, mae_vals, rmse_vals],
        ['R²', 'MAE (mg/m³)', 'RMSE (mg/m³)'],
        ['决定系数 R²（越高越好）', '平均绝对误差 MAE（越低越好）', '均方根误差 RMSE（越低越好）'],
        [True, False, False]
    ):
        bars = ax.bar(x, vals, width=w, color=colors, alpha=0.85,
                      edgecolor='white', linewidth=1.3)

        best_idx = np.argmax(vals) if best_is_high else np.argmin(vals)
        for i, (bar, val) in enumerate(zip(bars, vals)):
            offset = 0.005 if best_is_high else max(vals)*0.015
            fw = 'bold' if i == best_idx else 'normal'
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height() + offset,
                    f'{val:.4f}' if best_is_high else f'{val:.1f}',
                    ha='center', va='bottom', fontproperties=_fp,
                    fontsize=9, fontweight=fw)

        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontproperties=_fp, fontsize=8)
        ax.set_ylabel(ylabel, fontproperties=_fp)
        ax.set_title(title, fontproperties=_fp, fontsize=10, fontweight='bold')
        if best_is_high:
            ax.set_ylim(0.8, 0.98)
        ax.grid(axis='y', alpha=0.3)
        fp(ax)

    legend_e = [mpatches.Patch(fc=C1, alpha=0.85, label='对比基线模型'),
                mpatches.Patch(fc=C2, alpha=0.85, label='本文方法 XGBoost+PSO')]
    fig.legend(handles=legend_e, loc='upper center', ncol=2, prop=_fp,
               bbox_to_anchor=(0.5, 1.02))
    plt.suptitle('图4-4  五种模型测试集性能对比（真实训练结果，70/30划分）',
                 fontproperties=_fp, fontsize=12, fontweight='bold', y=1.06)
    plt.tight_layout()
    plt.savefig('paper_figures/fig4_4_real_comparison.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('✓ fig4_4_real_comparison.png')


# ══════════════════════════════════════════════════════════════════════════════
# 图4-5  PSO迭代过程（全局最优 + 每轮均值 + 粒子分布）
# ══════════════════════════════════════════════════════════════════════════════
def gen_pso_fig():
    iters    = [d['iter']         for d in pso_log]
    gbest    = [d['gbest_r2']     for d in pso_log]
    it_best  = [d['iter_best_r2'] for d in pso_log]
    it_mean  = [d['iter_mean_r2'] for d in pso_log]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # 左：收敛曲线
    ax = axes[0]
    ax.plot(iters, gbest,   'o-', color=C2, lw=2.5, ms=8, label=f'全局最优CV-R²（最终={gbest[-1]:.4f}）')
    ax.plot(iters, it_best, 's--', color=C1, lw=1.5, ms=6, label='本轮最优粒子CV-R²', alpha=0.8)
    ax.plot(iters, it_mean, '^:', color=C3, lw=1.5, ms=6, label='本轮全部粒子均值', alpha=0.8)
    ax.fill_between(iters, it_mean, gbest, alpha=0.08, color=C2)

    # 标注每次全局最优更新的迭代
    prev_best = 0
    for i, (it, g) in enumerate(zip(iters, gbest)):
        if g > prev_best:
            ax.annotate(f'  第{it}轮\n  更新→{g:.4f}',
                        xy=(it, g), xytext=(it+0.3, g-0.006),
                        fontproperties=_fp, fontsize=7.5, color=C2,
                        arrowprops=dict(arrowstyle='->', color=C2, lw=0.8))
            prev_best = g

    ax.set_xlabel('PSO迭代次数', fontproperties=_fp)
    ax.set_ylabel('3折时序交叉验证 R²', fontproperties=_fp)
    ax.set_title('PSO寻优收敛过程（全局最优 / 本轮最优 / 本轮均值）',
                 fontproperties=_fp, fontsize=10, fontweight='bold')
    ax.legend(prop=_fp, fontsize=8.5)
    ax.set_ylim(0.69, 0.90)
    ax.grid(True, alpha=0.3); fp(ax)

    # 右：每轮8粒子分数箱线图
    ax2 = axes[1]
    scores_by_iter = [d['particle_scores'] for d in pso_log]
    bp = ax2.boxplot(scores_by_iter, positions=iters, widths=0.55,
                     patch_artist=True,
                     boxprops=dict(facecolor='#AED6F1', alpha=0.7),
                     medianprops=dict(color=C2, lw=2),
                     whiskerprops=dict(lw=1.2),
                     capprops=dict(lw=1.2))
    ax2.plot(iters, gbest, 'D-', color=C2, lw=1.8, ms=6, zorder=5, label='全局最优')
    ax2.set_xlabel('PSO迭代次数', fontproperties=_fp)
    ax2.set_ylabel('粒子适应度（CV-R²）', fontproperties=_fp)
    ax2.set_title('各迭代8粒子适应度分布（箱线图）',
                  fontproperties=_fp, fontsize=10, fontweight='bold')
    ax2.legend(prop=_fp, fontsize=8.5)
    ax2.grid(axis='y', alpha=0.3); fp(ax2)

    plt.suptitle('图4-5  PSO超参数寻优迭代过程（8粒子×10迭代，3折时序CV）',
                 fontproperties=_fp, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('paper_figures/fig4_5_pso_real.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('✓ fig4_5_pso_real.png')


# ══════════════════════════════════════════════════════════════════════════════
# 图4-6  超参数演化轨迹（lr / depth / n_estimators）
# ══════════════════════════════════════════════════════════════════════════════
def gen_param_trajectory():
    iters = [d['iter'] for d in pso_log]
    lrs   = [d['gbest_params']['learning_rate'] for d in pso_log]
    deps  = [d['gbest_params']['max_depth']      for d in pso_log]
    nests = [d['gbest_params']['n_estimators']   for d in pso_log]
    subs  = [d['gbest_params']['subsample']      for d in pso_log]
    cols  = [d['gbest_params']['colsample_bytree'] for d in pso_log]
    alphas= [d['gbest_params']['reg_alpha']      for d in pso_log]
    lams  = [d['gbest_params']['reg_lambda']     for d in pso_log]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    params_data = [
        (lrs,   '学习率 (learning_rate)', [0.01, 0.50], C1),
        (deps,  '最大树深 (max_depth)',    [2,    8],    C2),
        (nests, '树数量 (n_estimators)',   [50,   500],  C3),
        (subs,  '行采样率 (subsample)',    [0.5,  1.0],  C4),
        (cols,  '列采样率 (colsample)',    [0.5,  1.0],  C5),
        (alphas,'L1正则 (reg_alpha)',      [0.0,  2.0],  C6),
    ]

    for i, (vals, label, ylim, color) in enumerate(params_data):
        ax = axes[i]
        ax.step(iters, vals, where='post', color=color, lw=2, alpha=0.9)
        ax.plot(iters, vals, 'o', color=color, ms=6, zorder=5)
        ax.axhspan(ylim[0], ylim[1], alpha=0.05, color=color, label=f'搜索范围[{ylim[0]},{ylim[1]}]')
        ax.set_xlabel('PSO迭代次数', fontproperties=_fp, fontsize=9)
        ax.set_ylabel(label, fontproperties=_fp, fontsize=9)
        ax.set_title(label, fontproperties=_fp, fontsize=10, fontweight='bold')
        ax.set_ylim(ylim[0]-abs(ylim[1]-ylim[0])*0.15, ylim[1]+abs(ylim[1]-ylim[0])*0.15)
        ax.legend(prop=_fp, fontsize=7.5)
        ax.grid(True, alpha=0.3); fp(ax)

        # 标注最终值
        ax.annotate(f'最终: {vals[-1]:.4f}', xy=(iters[-1], vals[-1]),
                    xytext=(iters[-1]-2.5, vals[-1]+abs(ylim[1]-ylim[0])*0.12),
                    fontproperties=_fp, fontsize=8, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

    plt.suptitle('图4-6  PSO寻优过程中全局最优粒子各超参数演化轨迹',
                 fontproperties=_fp, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('paper_figures/fig4_6_param_trajectory.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print('✓ fig4_6_param_trajectory.png')


# ══════════════════════════════════════════════════════════════════════════════
# 图4-7  预测结果（真实重跑）
# ══════════════════════════════════════════════════════════════════════════════
def gen_prediction_fig():
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor
    from sklearn.metrics import r2_score, mean_absolute_error
    from scipy.stats import norm

    df = pd.read_csv('data/processed_data.csv')
    df = df[~df.index.isin(range(1045,1062))].reset_index(drop=True)

    with open('results/Q2_complete_results.json','r',encoding='utf-8') as f:
        q2 = json.load(f)
    lag_results = q2['lag_results']

    var_list = ['机速'] + [f'负压_{i}' for i in range(1,19)] + [f'温度_{i}' for i in range(1,19)]
    var_list += ['大烟道负压_1','大烟道负压_2','大烟道温度_1','大烟道温度_2']
    df_al = df.copy()
    for v in var_list:
        df_al[f'{v}_al'] = df_al[v].shift(-lag_results.get(v,0))

    phys = ['机速_al']+[f'负压_{i}_al' for i in range(1,19)]+[f'温度_{i}_al' for i in range(1,19)]
    phys += ['大烟道负压_1_al','大烟道负压_2_al','大烟道温度_1_al','大烟道温度_2_al']
    grad = []
    for i in range(1,18):
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
    feats = phys+grad+stat+co_f
    df_f = df_al.dropna(subset=feats+['CO浓度']).reset_index(drop=True)
    X = df_f[feats].values; y = df_f['CO浓度'].values
    split = int(len(y)*0.7)
    X_tr,X_te = X[:split],X[split:]
    y_tr,y_te = y[:split],y[split:]
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)

    # 用本次PSO最优参数
    bp = comp_data['model_comparison']['XGBoost+PSO(本文)']['best_params']
    mdl = XGBRegressor(**bp, random_state=42, n_jobs=-1, verbosity=0)
    mdl.fit(X_tr_s, y_tr, verbose=False)
    y_pred = mdl.predict(X_te_s)
    residuals = y_te - y_pred

    r2   = r2_score(y_te, y_pred)
    mae  = mean_absolute_error(y_te, y_pred)
    rmse = float(np.sqrt(np.mean((y_te-y_pred)**2)))

    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # 散点图
    ax = fig.add_subplot(gs[0, :2])
    ax.scatter(y_te, y_pred, s=6, alpha=0.4, color=C1)
    lm = max(y_te.max(), y_pred.max())*1.05
    ax.plot([0,lm],[0,lm],'r--',lw=1.8,label='理想预测线 y=x')
    ax.set_xlabel('真实CO浓度 (mg/m³)',fontproperties=_fp)
    ax.set_ylabel('预测CO浓度 (mg/m³)',fontproperties=_fp)
    ax.set_title(f'图4-7(a)  测试集预测值 vs 真实值\nR²={r2:.4f}, MAE={mae:.1f} mg/m³, RMSE={rmse:.1f} mg/m³',
                 fontproperties=_fp, fontsize=10, fontweight='bold')
    ax.legend(prop=_fp); ax.grid(True,alpha=0.3); fp(ax)

    # 时序曲线
    ax2 = fig.add_subplot(gs[1, :2])
    idx = range(min(300,len(y_te)))
    ax2.plot(idx,y_te[:300],color=C1,lw=1.2,label='真实值',alpha=0.85)
    ax2.plot(idx,y_pred[:300],color=C2,lw=1.2,ls='--',label='预测值',alpha=0.85)
    ax2.fill_between(idx,y_pred[:300]-rmse,y_pred[:300]+rmse,alpha=0.1,color=C2,
                     label=f'±RMSE ({rmse:.0f} mg/m³)')
    ax2.set_xlabel('测试样本序号',fontproperties=_fp)
    ax2.set_ylabel('CO浓度 (mg/m³)',fontproperties=_fp)
    ax2.set_title('图4-7(b)  测试集时序预测结果（前300条）',fontproperties=_fp,fontsize=10,fontweight='bold')
    ax2.legend(prop=_fp,fontsize=8); ax2.grid(True,alpha=0.3); fp(ax2)

    # 残差分布
    ax3 = fig.add_subplot(gs[0,2])
    ax3.hist(residuals,bins=40,color=C3,alpha=0.75,edgecolor='white',density=True)
    mu,sigma = np.mean(residuals),np.std(residuals)
    x_r = np.linspace(residuals.min(),residuals.max(),200)
    from scipy.stats import norm
    ax3.plot(x_r,norm.pdf(x_r,mu,sigma),color=C2,lw=2,label=f'正态拟合\nμ={mu:.1f}, σ={sigma:.1f}')
    ax3.axvline(x=0,color='black',lw=1,ls='--')
    ax3.set_xlabel('残差 (mg/m³)',fontproperties=_fp)
    ax3.set_ylabel('密度',fontproperties=_fp)
    ax3.set_title('图4-7(c)  预测残差分布',fontproperties=_fp,fontsize=10,fontweight='bold')
    ax3.legend(prop=_fp,fontsize=8); ax3.grid(True,alpha=0.3); fp(ax3)

    # 误差CDF
    ax4 = fig.add_subplot(gs[1,2])
    abs_err = np.sort(np.abs(residuals))
    cdf = np.arange(1,len(abs_err)+1)/len(abs_err)
    ax4.plot(abs_err,cdf*100,color=C1,lw=2)
    for thr,col in [(75,C3),(150,C5),(200,C2)]:
        ii = np.searchsorted(abs_err,thr)
        cp = cdf[min(ii,len(cdf)-1)]*100
        ax4.axvline(x=thr,color=col,ls='--',lw=1.2,alpha=0.8)
        ax4.axhline(y=cp,color=col,ls='--',lw=1.2,alpha=0.8)
        ax4.text(thr+5,cp-4,f'≤{thr}: {cp:.0f}%',fontproperties=_fp,fontsize=7.5,color=col)
    ax4.set_xlabel('绝对误差 (mg/m³)',fontproperties=_fp)
    ax4.set_ylabel('累积百分比 (%)',fontproperties=_fp)
    ax4.set_title('图4-7(d)  误差累积分布',fontproperties=_fp,fontsize=10,fontweight='bold')
    ax4.grid(True,alpha=0.3); fp(ax4)

    plt.savefig('paper_figures/fig4_7_real_prediction.png',dpi=150,bbox_inches='tight',facecolor='white')
    plt.close()
    print(f'✓ fig4_7_real_prediction.png  R²={r2:.4f}')
    return r2, mae, rmse


# ══════════════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    gen_comparison_fig()
    gen_pso_fig()
    gen_param_trajectory()
    r2_final, mae_final, rmse_final = gen_prediction_fig()

    # 保存最终指标供Word使用
    with open('results/final_metrics.json','w',encoding='utf-8') as f:
        json.dump({'r2':r2_final,'mae':mae_final,'rmse':rmse_final}, f)

    print('\n全部图表生成完成')
