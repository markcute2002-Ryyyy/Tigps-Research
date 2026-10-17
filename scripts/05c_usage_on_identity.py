"""STEP 05c｜延伸模型：社群媒體使用程度與線上、線下自我認同

社群媒體使用程度 usage_composite = bs21c + bs21e（兩題時數代碼加總，範圍 2–24）。
以線上、線下自我認同分數（未中心化）分別為依變數，並將另一種認同分數納入控制：
    online_score  ~ usage_composite + offline_score (+ 7 個背景控制變數)
    offline_score ~ usage_composite + online_score  (+ 7 個背景控制變數)
各估計未調整與調整後兩種版本；控制變數以類別變數納入，採 HC3 穩健標準誤，
usage_composite 的 p 值以 Benjamini-Hochberg 法進行 FDR 校正。

注意：使用時數與自我認同皆為第二波同一時間點的測量，結果僅代表關聯，不代表因果。

執行：python scripts/05c_usage_on_identity.py

輸入：
  data/TIGPS_quadrant_clean.dta   03 的輸出

輸出：
  results/step05c_usage_on_identity_coefficients.csv   係數、標準誤、p 值、R squared
  results/step05c_usage_on_identity_samples.csv        各模型樣本數
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
RESULTS = BASE / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)

SOURCE = DATA / 'TIGPS_quadrant_clean.dta'

# 必須跟 05_all_models.py 的 CONTROLS 保持一致
CONTROLS = ['as1_clean', 'as4d1_clean', 'as10_clean', 'as5_clean', 'as8_any_clean', 'school_type_w1_clean', 'digi_level_w1_clean']

if not SOURCE.exists():
    raise FileNotFoundError(f'找不到 {SOURCE}，請先執行 03_quadrant.py')

df = pd.read_stata(SOURCE, convert_categoricals=False)
df = df.copy()  # 從Stata讀回來的欄位一多就容易被判定成碎片化，讀完先整理一次，避免後面逐欄賦值時跳PerformanceWarning

needed = ['online_score', 'offline_score', 'bs21c_clean', 'bs21e_clean'] + CONTROLS
missing_needed = sorted(set(needed) - set(df.columns))
if missing_needed:
    raise KeyError(f'缺少必要欄位：{missing_needed}；請確認 02_clean_data.py / 03_quadrant.py 有正常跑完。')

# ----------------------------------------
# 社群媒體使用程度代理變數：兩題直接相加
# ----------------------------------------
df['usage_composite'] = df['bs21c_clean'] + df['bs21e_clean']
df = df.copy()

# 控制變數一律當「類別變數」放進模型（公式裡用 C()），跟主模型 05_all_models.py 完全一致。
# 例如 as4d1 的 -7（不清楚）是獨立類別、as10 是次序選項，都不能當連續數值直接放進迴歸。
for c in CONTROLS:
    df[c] = df[c].astype(object).where(pd.notna(df[c]), np.nan)
CONTROLS_FORMULA = [f'C({c})' for c in CONTROLS]

corr = df[['bs21c_clean', 'bs21e_clean']].corr().iloc[0, 1]
print(f'bs21c與bs21e的相關係數：{corr:.3f}（僅供參考，這裡已經確定用「相加」而不是分開放）')
print('usage_composite（bs21c_clean + bs21e_clean）描述統計：')
print(df['usage_composite'].describe().round(2).to_string())

# ----------------------------------------
# 4個模型：2個DV（online_score / offline_score）x unadjusted / adjusted
# ----------------------------------------
SPECS = [
    ('online_score', 'offline_score', '線上自我認同'),
    ('offline_score', 'online_score', '線下自我認同'),
]

coef_rows, sample_rows = [], []

for dv, other, dv_label in SPECS:
    for adjusted in (False, True):
        model_name = f'{dv}_{"adjusted" if adjusted else "unadjusted"}'
        rhs = ['usage_composite', other] + (CONTROLS_FORMULA if adjusted else [])
        formula = f'{dv} ~ ' + ' + '.join(rhs)
        use_cols = [dv, 'usage_composite', other] + (CONTROLS if adjusted else [])
        sub = df[use_cols].dropna()

        model = smf.ols(formula, data=sub).fit(cov_type='HC3')

        sample_rows.append({
            '模型': model_name, 'DV': dv_label, '是否調整背景變數': adjusted, 'N': len(sub),
        })
        coef_rows.append({
            '模型': model_name, 'DV': dv_label, '是否調整背景變數': adjusted, 'N': len(sub),
            'usage_composite_係數': model.params['usage_composite'],
            '標準誤': model.bse['usage_composite'],
            't值': model.tvalues['usage_composite'],
            'p值': model.pvalues['usage_composite'],
            f'對方認同分數({other})_係數': model.params[other],
            f'對方認同分數({other})_標準誤': model.bse[other],
            f'對方認同分數({other})_p值': model.pvalues[other],
            'R_squared': model.rsquared,
        })

coef_df = pd.DataFrame(coef_rows)
sample_df = pd.DataFrame(sample_rows)

# FDR校正：對「usage_composite係數的p值」做校正；unadjusted兩個DV一組、
# adjusted兩個DV另一組，各自校正（概念跟04/04b對多個outcome做FDR一致）
coef_df['p值_FDR_BH'] = np.nan
for adjusted_flag in (False, True):
    mask = coef_df['是否調整背景變數'] == adjusted_flag
    coef_df.loc[mask, 'p值_FDR_BH'] = multipletests(coef_df.loc[mask, 'p值'], method='fdr_bh')[1]

coef_df.to_csv(RESULTS / 'step05c_usage_on_identity_coefficients.csv', index=False, encoding='utf-8-sig')
sample_df.to_csv(RESULTS / 'step05c_usage_on_identity_samples.csv', index=False, encoding='utf-8-sig')

print('\n===== 延伸二：社群媒體使用程度 對 線上/線下自我認同 的關聯（OLS, robust HC3 SE）=====')
print(coef_df[['模型', 'DV', '是否調整背景變數', 'N', 'usage_composite_係數', 'p值', 'p值_FDR_BH']]
      .round(5).to_string(index=False))
print('\n已輸出 results/step05c_usage_on_identity_coefficients.csv、results/step05c_usage_on_identity_samples.csv')
print('\n提醒：這是橫斷面資料（使用程度跟認同題目同一波測量），係數只能解讀成「關聯」，')
print('不能解讀成「使用程度影響/導致了認同」這種因果語氣，寫論文時請注意用詞。')
