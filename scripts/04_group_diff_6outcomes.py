"""STEP 04｜六項身心狀態指標的四象限組間差異

計算四種類型在六項身心狀態指標的平均數與標準差，並以單因子變異數分析（ANOVA）
與 Kruskal-Wallis 檢定比較組間差異，計算效果量 eta squared，p 值以 Benjamini-Hochberg
法進行 FDR 校正。此為未納入控制變數的組間比較；控制背景變數後的比較見 05。

六項指標：
    loneliness         孤獨感（bs56l）
    wellbeing_who5     WHO-5 幸福感（bs58a–e 平均）
    self_esteem        自尊（bs53a–c 平均）
    life_satisfaction  生活滿意度（bs50）
    cesd12             CESD-12 心理困擾（bs56a–d、f–h、j–n 平均）
    friend_support     好友支持（bs29a、c–e 平均）

執行：python scripts/04_group_diff_6outcomes.py

輸入：
  data/TIGPS_quadrant_clean.dta   03 的輸出

輸出：
  results/step04_outcome_by_quadrant_long.csv   各類型各指標的平均數與標準差
  results/step04_group_diff_summary.csv         ANOVA / Kruskal-Wallis 檢定結果與 FDR 校正
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

# 必須跟 02_clean_data.py / 05_all_models.py 的 OUTCOME_ITEMS 保持一致
OUTCOME_ITEMS = {
    'loneliness': ['bs56l'],
    'wellbeing_who5': ['bs58a', 'bs58b', 'bs58c', 'bs58d', 'bs58e'],
    'self_esteem': ['bs53a', 'bs53b', 'bs53c'],
    'life_satisfaction': ['bs50'],
    'cesd12': ['bs56a', 'bs56b', 'bs56c', 'bs56d', 'bs56f', 'bs56g', 'bs56h', 'bs56j', 'bs56k', 'bs56l', 'bs56m', 'bs56n'],
    'friend_support': ['bs29a', 'bs29c', 'bs29d', 'bs29e'],
}
OUTCOME_LABELS = {
    'loneliness': '孤獨感',
    'wellbeing_who5': 'WHO-5幸福感',
    'self_esteem': '自我價值/自尊',
    'life_satisfaction': '生活滿意度',
    'cesd12': 'CESD-12心理困擾',
    'friend_support': '好友支持',
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
print('四象限有效樣本數：', len(df))
print(df['quadrant_label'].value_counts().reindex(GROUPS).to_string())

# ----------------------------------------
# 用 02 清理好的題目建立 6 個outcome的合成分數
# ----------------------------------------
for outcome, items in OUTCOME_ITEMS.items():
    clean_items = [f'{i}_clean' for i in items]
    missing = sorted(set(clean_items) - set(df.columns))
    if missing:
        raise KeyError(f'{outcome} 缺少清理後欄位：{missing}；請確認 02_clean_data.py 有正常跑完。')
    df[outcome] = df[clean_items].mean(axis=1, skipna=False)
df = df.copy()  # 消除逐欄賦值造成的 DataFrame 碎片化（pandas PerformanceWarning），只是效能提醒不是錯誤

# ----------------------------------------
# 各象限的敘述統計 + 組間差異檢定
# ----------------------------------------
long_rows, test_rows = [], []
for outcome in OUTCOME_ITEMS:
    valid = df[['quadrant_label', outcome]].dropna()
    samples = [valid.loc[valid['quadrant_label'] == g, outcome].to_numpy() for g in GROUPS]

    for group, sample in zip(GROUPS, samples):
        long_rows.append({
            'Outcome': outcome, 'Outcome說明': OUTCOME_LABELS[outcome], '象限': group,
            '有效N': len(sample),
            '平均數': np.mean(sample) if len(sample) else np.nan,
            '標準差': np.std(sample, ddof=1) if len(sample) > 1 else np.nan,
        })

    if all(len(s) >= 2 for s in samples):
        f_stat, p_anova = f_oneway(*samples)
        kw_stat, p_kw = kruskal(*samples)
        grand = valid[outcome].mean()
        ss_between = sum(len(s) * (np.mean(s) - grand) ** 2 for s in samples)
        ss_total = ((valid[outcome] - grand) ** 2).sum()
        eta2 = ss_between / ss_total if ss_total > 0 else np.nan
    else:
        f_stat = p_anova = kw_stat = p_kw = eta2 = np.nan

    test_rows.append({
        'Outcome': outcome, 'Outcome說明': OUTCOME_LABELS[outcome], '總有效N': len(valid),
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

wide = long.pivot(index=['Outcome', 'Outcome說明'], columns='象限', values=['有效N', '平均數', '標準差'])
wide.columns = [f'{group}_{metric}' for metric, group in wide.columns]
wide = wide.reset_index()
summary = tests.merge(wide, on=['Outcome', 'Outcome說明'], how='left', validate='one_to_one')

front = ['Outcome', 'Outcome說明', '總有效N']
for g in GROUPS:
    front += [f'{g}_有效N', f'{g}_平均數', f'{g}_標準差']
front += ['ANOVA_F', 'ANOVA_p', 'ANOVA_FDR_BH', 'Kruskal_Wallis_H', 'KW_p', 'KW_FDR_BH', 'eta_squared']
summary = summary[front]

long.to_csv(RESULTS / 'step04_outcome_by_quadrant_long.csv', index=False, encoding='utf-8-sig')
summary.to_csv(RESULTS / 'step04_group_diff_summary.csv', index=False, encoding='utf-8-sig')

print('\n===== 6個核心outcome的四象限組間差異檢定 =====')
print(summary[['Outcome說明', '總有效N', 'ANOVA_p', 'ANOVA_FDR_BH', 'eta_squared']].round(5).to_string(index=False))
print('\n已輸出 results/step04_outcome_by_quadrant_long.csv、results/step04_group_diff_summary.csv')
print('提醒：這是未調整背景變數的組間差異（跟 05_all_models.py 的 A_unadjusted 模型概念一致，')
print('但這裡只用 ANOVA/KW 做整體組間差異檢定，沒有逐對比較跟調整後的版本，那些留給 05 處理）。')
