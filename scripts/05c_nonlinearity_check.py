"""STEP 05c（穩健性）｜延伸模型的線性假設檢查

使用時數題為級距不等的代碼，因此以兩種方式檢查線性設定（皆為納入控制變數的模型）：
  1. 依 usage_composite 的分位數分組，估計各組的調整後平均（不假設函數形狀）
  2. 於模型中加入 usage_composite 的二次項，並比較二次與線性模型的預測差距
樣本數大時二次項容易顯著，判讀應以分組平均是否貼近線性預測線、以及預測差距的實質大小為主。

執行：python scripts/05c_nonlinearity_check.py

輸入：
  data/TIGPS_quadrant_clean.dta   03 的輸出

輸出：
  results/step05c_nonlinearity_quadratic.csv
  results/step05c_nonlinearity_binned.csv
  results/05c_nonlinearity_online.png
  results/05c_nonlinearity_offline.png
"""
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# 字型清單涵蓋不同作業系統（Windows、macOS、Linux），
# 找不到的字型只會產生 findfont 警告，不影響出圖，因此關閉該警告
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
RESULTS = BASE / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)

SOURCE = DATA / 'TIGPS_quadrant_clean.dta'

# 必須跟 05c_usage_on_identity.py / 05_all_models.py 的 CONTROLS 保持一致
CONTROLS = ['as1_clean', 'as4d1_clean', 'as10_clean', 'as5_clean', 'as8_any_clean', 'school_type_w1_clean', 'digi_level_w1_clean']
N_BINS = 6  # 分位數切幾組；因為 usage_composite 是整數、同分的人很多，實際組數可能會少一點

plt.rcParams['font.family'] = ['Microsoft JhengHei', 'PingFang TC', 'Noto Sans CJK TC',
                                'Heiti TC', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

if not SOURCE.exists():
    raise FileNotFoundError(f'找不到 {SOURCE}，請先執行 03_quadrant.py')

df = pd.read_stata(SOURCE, convert_categoricals=False)
df = df.copy()  # 從Stata讀回來的欄位一多就容易被判定成碎片化，讀完先整理一次，避免後面逐欄賦值時跳PerformanceWarning

needed = ['online_score', 'offline_score', 'bs21c_clean', 'bs21e_clean'] + CONTROLS
missing_needed = sorted(set(needed) - set(df.columns))
if missing_needed:
    raise KeyError(f'缺少必要欄位：{missing_needed}；請確認 02_clean_data.py / 03_quadrant.py 有正常跑完。')

df['usage_composite'] = df['bs21c_clean'] + df['bs21e_clean']
# 控制變數一律當類別變數（C()），跟主模型 05_all_models.py 一致
for c in CONTROLS:
    df[c] = df[c].astype(object).where(pd.notna(df[c]), np.nan)
df = df.copy()
CONTROLS_FORMULA = ' + '.join(f'C({c})' for c in CONTROLS)


def design_matrix(model, data):
    """依照模型公式把 data 轉成設計矩陣（含 C() 類別虛擬變數、I(x**2) 等轉換），
    欄位順序跟 model.params 一致。同時支援新舊版 statsmodels。"""
    try:
        X, _ = model._transform_predict_exog(data, transform=True)
    except AttributeError:  # 較舊版本的 statsmodels
        from patsy import build_design_matrices
        X = build_design_matrices([model.model.data.design_info], data, return_type='matrix')[0]
    return np.asarray(X, dtype=float)


def average_prediction(model, data, var, values, z=1.96):
    """平均調整後預測：每個值都把全樣本的 var 設成該值、其他變數維持實際值，取設計矩陣的
    平均列 x̄；預測值 = x̄·β，標準誤 = sqrt(x̄ V x̄')（V 為 HC3 共變異數矩陣）。"""
    beta = model.params.to_numpy()
    V = np.asarray(model.cov_params())
    rows = []
    for v in values:
        d = data.copy()
        d[var] = v
        xbar = design_matrix(model, d).mean(axis=0)
        pred = float(xbar @ beta)
        se = float(np.sqrt(xbar @ V @ xbar))
        rows.append({'value': v, 'mean': pred, 'ci_low': pred - z * se, 'ci_high': pred + z * se})
    return pd.DataFrame(rows)


SPECS = [
    ('online_score', 'offline_score', '線上自我認同', 'online', '#4C72B0'),
    ('offline_score', 'online_score', '線下自我認同', 'offline', '#DD8452'),
]

quad_rows, bin_rows = [], []

for dv, other, dv_label, tag, color in SPECS:
    sub = df[[dv, 'usage_composite', other] + CONTROLS].dropna().copy()
    cov_str = f'{other} + {CONTROLS_FORMULA}'

    # ---------- 線性模型（= 05c 的 A2 / B2）與二次模型 ----------
    m_lin = smf.ols(f'{dv} ~ usage_composite + {cov_str}', data=sub).fit(cov_type='HC3')
    m_quad = smf.ols(f'{dv} ~ usage_composite + I(usage_composite ** 2) + {cov_str}',
                     data=sub).fit(cov_type='HC3')

    lo, hi = np.percentile(sub['usage_composite'], [5, 95])
    grid = np.linspace(lo, hi, 60)
    pred_lin = average_prediction(m_lin, sub, 'usage_composite', grid)['mean'].to_numpy()
    pred_quad = average_prediction(m_quad, sub, 'usage_composite', grid)['mean'].to_numpy()
    max_gap = float(np.max(np.abs(pred_quad - pred_lin)))

    b1 = m_quad.params['usage_composite']
    b2 = m_quad.params['I(usage_composite ** 2)']
    turning = -b1 / (2 * b2) if b2 != 0 else np.nan
    turning_in_range = bool(lo <= turning <= hi) if np.isfinite(turning) else False

    quad_rows.append({
        'DV': dv_label, 'N': len(sub),
        '線性模型_usage係數': m_lin.params['usage_composite'],
        '二次模型_一次項係數': b1,
        '二次模型_二次項係數': b2,
        '二次項_p值': m_quad.pvalues['I(usage_composite ** 2)'],
        'R2_線性': m_lin.rsquared,
        'R2_二次': m_quad.rsquared,
        'R2_增加量': m_quad.rsquared - m_lin.rsquared,
        '5th~95th範圍內_二次與線性預測最大差距(分)': max_gap,
        '線性預測_整段變化量(分)': float(pred_lin[-1] - pred_lin[0]),
        '轉折點(usage_composite)': turning,
        '轉折點是否落在5th~95th範圍內': turning_in_range,
    })

    # ---------- 分組調整後平均（不假設形狀）----------
    sub['usage_bin'] = pd.qcut(sub['usage_composite'], q=N_BINS, duplicates='drop')
    bins = sub['usage_bin'].cat.categories
    sub['usage_bin'] = sub['usage_bin'].astype(str)
    bin_labels = [str(b) for b in bins]
    m_bin = smf.ols(f'{dv} ~ C(usage_bin) + {cov_str}', data=sub).fit(cov_type='HC3')

    pred_bin = average_prediction(m_bin, sub, 'usage_bin', bin_labels)
    for i, b in enumerate(bins):
        in_bin = sub['usage_bin'] == str(b)
        bin_rows.append({
            'DV': dv_label, '組別(usage_composite範圍)': str(b),
            'N': int(in_bin.sum()),
            '組內usage_composite平均': sub.loc[in_bin, 'usage_composite'].mean(),
            '調整後預測平均': pred_bin['mean'].iloc[i],
            'CI下界': pred_bin['ci_low'].iloc[i],
            'CI上界': pred_bin['ci_high'].iloc[i],
        })

    # ---------- 畫圖 ----------
    this_bins = pd.DataFrame([r for r in bin_rows if r['DV'] == dv_label])
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    ax.plot(grid, pred_lin, color=color, linewidth=2, label='線性模型預測（05c）')
    ax.plot(grid, pred_quad, color='#555555', linewidth=1.5, linestyle='--', label='二次模型預測')
    ax.errorbar(this_bins['組內usage_composite平均'], this_bins['調整後預測平均'],
                yerr=[this_bins['調整後預測平均'] - this_bins['CI下界'],
                      this_bins['CI上界'] - this_bins['調整後預測平均']],
                fmt='o', color=color, ecolor=color, markersize=8, capsize=4, elinewidth=1.5,
                markeredgecolor='white', markeredgewidth=1.5, zorder=3,
                label='分組調整後平均（95% CI，不假設形狀）')
    for _, r in this_bins.iterrows():
        ax.annotate(f"n={r['N']:,}", (r['組內usage_composite平均'], r['CI下界']),
                    textcoords='offset points', xytext=(0, -14), ha='center', fontsize=8, color='#666666')
    ax.margins(y=0.12)  # 留空間給每組下方的 n= 標籤，避免被x軸切到
    ax.set_xlabel('usage_composite（社群媒體使用程度代理變數，2～24）', fontsize=11)
    ax.set_ylabel(f'預測 {dv_label}分數（{dv}）', fontsize=11)
    ax.set_title(f'延伸二線性假設檢查：usage_composite 與 {dv_label}\n'
                 f'（adjusted模型，{other} 與7個控制變數依樣本實際分布平均）', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#e0e0e0', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc='best', fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / f'05c_nonlinearity_{tag}.png')
    plt.close(fig)

quad_df = pd.DataFrame(quad_rows)
bin_df_all = pd.DataFrame(bin_rows)
quad_df.to_csv(RESULTS / 'step05c_nonlinearity_quadratic.csv', index=False, encoding='utf-8-sig')
bin_df_all.to_csv(RESULTS / 'step05c_nonlinearity_binned.csv', index=False, encoding='utf-8-sig')

pd.set_option('display.width', 200)
print('===== 二次項檢定（adjusted模型）=====')
print(quad_df[['DV', 'N', '二次模型_二次項係數', '二次項_p值', 'R2_增加量',
               '5th~95th範圍內_二次與線性預測最大差距(分)', '線性預測_整段變化量(分)',
               '轉折點(usage_composite)', '轉折點是否落在5th~95th範圍內']].round(5).to_string(index=False))
print('\n===== 分組調整後平均 =====')
print(bin_df_all.round(3).to_string(index=False))
print('\n已輸出 results/step05c_nonlinearity_quadratic.csv、results/step05c_nonlinearity_binned.csv、')
print('       results/05c_nonlinearity_online.png、results/05c_nonlinearity_offline.png')
print('\n判讀提醒：樣本很大，二次項 p 值顯著不代表彎曲有實質意義；主要看分組平均的點是否大致')
print('落在線性預測線上，以及「最大差距」跟「整段變化量」相比是不是很小。')
