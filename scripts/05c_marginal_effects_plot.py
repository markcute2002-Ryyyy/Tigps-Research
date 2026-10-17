"""STEP 05c（圖）｜延伸模型的平均預測曲線

以納入控制變數的延伸模型，計算 usage_composite 由第 5 至第 95 百分位變化時，
線上與線下自我認同的平均調整後預測值（其他變數維持樣本的實際分布）及 95% 信賴區間。

執行：python scripts/05c_marginal_effects_plot.py

輸入：
  data/TIGPS_quadrant_clean.dta   03 的輸出

輸出：
  results/05c_marginal_effects_online.png
  results/05c_marginal_effects_offline.png
  results/step05c_marginal_effects_predictions.csv
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

# 必須跟 05_all_models.py / 05c_usage_on_identity.py 的 CONTROLS 保持一致
CONTROLS = ['as1_clean', 'as4d1_clean', 'as10_clean', 'as5_clean', 'as8_any_clean', 'school_type_w1_clean', 'digi_level_w1_clean']

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

all_pred_rows = []

for dv, other, dv_label, tag, color in SPECS:
    sub = df[[dv, 'usage_composite', other] + CONTROLS].dropna().copy()
    model = smf.ols(f'{dv} ~ usage_composite + {other} + {CONTROLS_FORMULA}', data=sub).fit(cov_type='HC3')
    print(f'{dv_label}（adjusted）N={int(model.nobs)}，usage_composite 係數={model.params["usage_composite"]:+.4f}')

    lo, hi = np.percentile(sub['usage_composite'], [5, 95])
    grid = np.linspace(lo, hi, 50)
    pred = average_prediction(model, sub, 'usage_composite', grid)
    pred = pred.rename(columns={'value': 'usage_composite'})
    pred['DV'] = dv_label
    all_pred_rows.append(pred)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.plot(pred['usage_composite'], pred['mean'], color=color, linewidth=2.5)
    ax.fill_between(pred['usage_composite'], pred['ci_low'], pred['ci_high'], color=color, alpha=0.2,
                    label='95% 信賴區間')
    ax.set_xlabel('usage_composite（社群媒體使用程度代理變數，2～24）', fontsize=11)
    ax.set_ylabel(f'預測 {dv_label}分數（{dv}）', fontsize=11)
    ax.set_title(f'延伸二：usage_composite 對 {dv_label} 的邊際效果\n'
                 f'（adjusted模型，{other} 與7個控制變數依樣本實際分布平均）', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#e0e0e0', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc='best', fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / f'05c_marginal_effects_{tag}.png')
    plt.close(fig)
    print(f'已輸出 results/05c_marginal_effects_{tag}.png')

pd.concat(all_pred_rows, ignore_index=True).to_csv(
    RESULTS / 'step05c_marginal_effects_predictions.csv', index=False, encoding='utf-8-sig')
print('已輸出 results/step05c_marginal_effects_predictions.csv')
