"""STEP 04b｜延伸分析：四象限的社群媒體使用時數差異

比較四種類型在玩線上遊戲（bs21c）與和他人聊天／傳訊息（bs21e）兩項使用時數上的差異，
方法同 04（ANOVA、Kruskal-Wallis、eta squared、FDR 校正）。兩題各自計算有效樣本。

執行：python scripts/04b_extension_social_media_behavior.py

輸入：
  data/TIGPS_quadrant_clean.dta   03 的輸出

輸出：
  results/step04b_ext_by_quadrant_long.csv     各類型的平均數與標準差
  results/step04b_ext_group_diff_summary.csv   檢定結果與 FDR 校正
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal
from statsmodels.stats.multitest import multipletests

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
RESULTS = BASE / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)

SOURCE = DATA / 'TIGPS_quadrant_clean.dta'
GROUPS = ['雙高型', '線上優勢型', '線下優勢型', '雙低型']

# 必須跟 02_clean_data.py 的 EXT_ITEMS 保持一致
EXT_ITEMS = ['bs21c', 'bs21e']
EXT_ITEM_LABELS = {
    'bs21c': '玩線上遊戲(時數代碼1-12)',
    'bs21e': '和他人聊天/傳訊息(時數代碼1-12)',
}

if not SOURCE.exists():
    raise FileNotFoundError(f'找不到 {SOURCE}，請先執行 03_quadrant.py')

df = pd.read_stata(SOURCE, convert_categoricals=False)
df = df.copy()  # 從Stata讀回來的欄位一多就容易被判定成碎片化，讀完先整理一次，避免後面逐欄賦值時跳PerformanceWarning

if pd.api.types.is_numeric_dtype(df['quadrant_label']):
    labels = pd.read_stata(SOURCE, convert_categoricals=True, columns=['quadrant_label'])
    df['quadrant_label'] = labels['quadrant_label'].astype(str)
else:
    df['quadrant_label'] = df['quadrant_label'].astype(str)

df = df.loc[df['quadrant_label'].isin(GROUPS)].copy()
print('四象限有效樣本數（基礎，尚未扣掉bs21c/bs21e各自的缺失）：', len(df))
print(df['quadrant_label'].value_counts().reindex(GROUPS).to_string())

missing_ext = sorted(set(f'{i}_clean' for i in EXT_ITEMS) - set(df.columns))
if missing_ext:
    raise KeyError(
        f'缺少延伸變數清理後欄位：{missing_ext}；請確認 02_clean_data.py 有正常跑完，'
        ' 且 02 的 EXT_ITEMS 不是空的（之前曾經因為欄位名稱沒確認好而暫時停用過）。'
    )

# ----------------------------------------
# 各象限的敘述統計 + 組間差異檢定
# ----------------------------------------
long_rows, test_rows = [], []
for item in EXT_ITEMS:
    col = f'{item}_clean'
    valid = df[['quadrant_label', col]].dropna()
    samples = [valid.loc[valid['quadrant_label'] == g, col].to_numpy() for g in GROUPS]

    for group, sample in zip(GROUPS, samples):
        long_rows.append({
            '變項': item, '變項說明': EXT_ITEM_LABELS[item], '象限': group,
            '有效N': len(sample),
            '平均數': np.mean(sample) if len(sample) else np.nan,
            '標準差': np.std(sample, ddof=1) if len(sample) > 1 else np.nan,
        })

    if all(len(s) >= 2 for s in samples):
        f_stat, p_anova = f_oneway(*samples)
        kw_stat, p_kw = kruskal(*samples)
        grand = valid[col].mean()
        ss_between = sum(len(s) * (np.mean(s) - grand) ** 2 for s in samples)
        ss_total = ((valid[col] - grand) ** 2).sum()
        eta2 = ss_between / ss_total if ss_total > 0 else np.nan
    else:
        f_stat = p_anova = kw_stat = p_kw = eta2 = np.nan

    test_rows.append({
        '變項': item, '變項說明': EXT_ITEM_LABELS[item], '總有效N': len(valid),
        'ANOVA_F': f_stat, 'ANOVA_p': p_anova,
        'Kruskal_Wallis_H': kw_stat, 'KW_p': p_kw, 'eta_squared': eta2,
    })

long = pd.DataFrame(long_rows)
tests = pd.DataFrame(test_rows)

for source, target in [('ANOVA_p', 'ANOVA_FDR_BH'), ('KW_p', 'KW_FDR_BH')]:
    ok = tests[source].notna()
    tests[target] = np.nan
    if ok.any():
        tests.loc[ok, target] = multipletests(tests.loc[ok, source], method='fdr_bh')[1]

wide = long.pivot(index=['變項', '變項說明'], columns='象限', values=['有效N', '平均數', '標準差'])
wide.columns = [f'{group}_{metric}' for metric, group in wide.columns]
wide = wide.reset_index()
summary = tests.merge(wide, on=['變項', '變項說明'], how='left', validate='one_to_one')

front = ['變項', '變項說明', '總有效N']
for g in GROUPS:
    front += [f'{g}_有效N', f'{g}_平均數', f'{g}_標準差']
front += ['ANOVA_F', 'ANOVA_p', 'ANOVA_FDR_BH', 'Kruskal_Wallis_H', 'KW_p', 'KW_FDR_BH', 'eta_squared']
summary = summary[front]

long.to_csv(RESULTS / 'step04b_ext_by_quadrant_long.csv', index=False, encoding='utf-8-sig')
summary.to_csv(RESULTS / 'step04b_ext_group_diff_summary.csv', index=False, encoding='utf-8-sig')

print('\n===== 延伸一：社群媒體使用行為的四象限組間差異檢定 =====')
print(summary[['變項說明', '總有效N', 'ANOVA_p', 'ANOVA_FDR_BH', 'eta_squared']].round(5).to_string(index=False))
print('\n已輸出 results/step04b_ext_by_quadrant_long.csv、results/step04b_ext_group_diff_summary.csv')
print('提醒：這是未調整背景變數的組間差異，只有2個檢定，FDR校正效果有限；如果要調整後/')
print('兩兩比較版本，可以仿照 05_all_models.py 的做法另外寫一支（例如 05b）。')
