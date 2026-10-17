"""STEP 03｜計算自我認同分數與四象限分類

線上自我認同分數 = bs25a–c 三題平均；線下自我認同分數 = bs25d–f 三題平均。
六題須全部有效回答才納入分析（任一題缺失即排除），再依兩個分數的中位數，
將樣本分為雙高型、線上優勢型、線下優勢型、雙低型四類。

執行：python scripts/03_quadrant.py

輸入：
  data/TIGPS_clean.dta   02 的輸出

輸出：
  data/TIGPS_quadrant_clean.dta         分析樣本（含 online_score、offline_score、quadrant_label），
                                        03a、04、04b、05、05c 皆讀取此檔
  results/step03_quadrant_summary.csv   四象限人數與占比
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
RESULTS = BASE / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)

SOURCE = DATA / 'TIGPS_clean.dta'
TARGET = DATA / 'TIGPS_quadrant_clean.dta'

ONLINE_ITEMS_CLEAN = ['bs25a_clean', 'bs25b_clean', 'bs25c_clean']
OFFLINE_ITEMS_CLEAN = ['bs25d_clean', 'bs25e_clean', 'bs25f_clean']

QUADRANT_LABELS = {1: '雙高型', 2: '線上優勢型', 3: '線下優勢型', 4: '雙低型'}

if not SOURCE.exists():
    raise FileNotFoundError(f'找不到 {SOURCE}，請先執行 02_clean_data.py')

df = pd.read_stata(SOURCE, convert_categoricals=False)
df = df.copy()  # 從Stata讀回來的欄位一多就容易被判定成碎片化，讀完先整理一次，避免後面逐欄賦值時跳PerformanceWarning
print('清理後資料筆數（篩選前）：', len(df))

missing = sorted(set(ONLINE_ITEMS_CLEAN + OFFLINE_ITEMS_CLEAN) - set(df.columns))
if missing:
    raise KeyError(f'缺少線上/線下清理後欄位：{missing}；請確認 02_clean_data.py 有正常跑完。')

# ----------------------------------------
# 1. 計算合成分數（skipna=False：六題只要有一題缺，分數就是 NaN）
# ----------------------------------------
df['online_score'] = df[ONLINE_ITEMS_CLEAN].mean(axis=1, skipna=False)
df['offline_score'] = df[OFFLINE_ITEMS_CLEAN].mean(axis=1, skipna=False)

df['n_missing_identity_items'] = df[ONLINE_ITEMS_CLEAN + OFFLINE_ITEMS_CLEAN].isna().sum(axis=1)
df = df.copy()  # 消除逐欄賦值造成的 DataFrame 碎片化（pandas PerformanceWarning），只是效能提醒不是錯誤

print('\n六題自我認同題目的遺漏題數分布（篩選前）：')
print(df['n_missing_identity_items'].value_counts().sort_index().to_string())

# ----------------------------------------
# 2. 篩選：六題皆有效回答
# ----------------------------------------
df = df.loc[df['online_score'].notna() & df['offline_score'].notna()].copy()
print('\n六題皆有效回答的樣本數（篩選後）：', len(df))

print('\n線上/線下自我認同敘述統計：')
print(df[['online_score', 'offline_score']].describe().to_string())

# ----------------------------------------
# 3. 中位數切高低、分四象限
# ----------------------------------------
med_online = df['online_score'].median()
med_offline = df['offline_score'].median()
print(f'\n線上自我認同中位數：{med_online}')
print(f'線下自我認同中位數：{med_offline}')

online_high = df['online_score'] >= med_online
offline_high = df['offline_score'] >= med_offline

df['quadrant'] = 0
df.loc[online_high & offline_high, 'quadrant'] = 1
df.loc[online_high & ~offline_high, 'quadrant'] = 2
df.loc[~online_high & offline_high, 'quadrant'] = 3
df.loc[~online_high & ~offline_high, 'quadrant'] = 4
df['quadrant_label'] = df['quadrant'].map(QUADRANT_LABELS)

summary = df['quadrant_label'].value_counts().reindex(QUADRANT_LABELS.values())
summary_pct = (summary / len(df) * 100).round(2)
result = pd.DataFrame({'樣本數': summary, '占比(%)': summary_pct})
print('\n===== 四象限分組結果 =====')
print(result.to_string())
result.to_csv(RESULTS / 'step03_quadrant_summary.csv', encoding='utf-8-sig')

# ----------------------------------------
# 4. 儲存（保留原始欄位 + 清理欄位 + 新增的象限欄位，04/05 都從這份接）
# ----------------------------------------
for col in df.columns:
    if pd.api.types.is_object_dtype(df[col]):
        df[col] = df[col].where(pd.notna(df[col]), None)

df.to_stata(TARGET, write_index=False, version=118)
print(f'\n已輸出 {TARGET}（03a 視覺化、04 組間差異檢定、05 迴歸模型都讀這個檔案）')
print('已輸出 results/step03_quadrant_summary.csv')
