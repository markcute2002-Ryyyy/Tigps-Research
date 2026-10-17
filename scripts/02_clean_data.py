"""STEP 02｜資料清理

將問卷的缺失代碼（-4、-5、-6、-7、-8、-9、-99、-999）轉為缺失值，分四個區塊清理：
  A. 線上／線下自我認同題目（bs25a–f）
  B. 家庭背景控制變數；第一位主要照顧者學歷（as4d1）的「-7 不清楚」保留為獨立類別，
     並將 as8_1、as8_2 合併為 as8_any（與至少一位親生父母同住：任一為是即為是）
  C. 六項身心狀態指標的題目
  D. 社群媒體使用時數題目（bs21c 玩線上遊戲、bs21e 和他人聊天／傳訊息，時數代碼 1–12）
     註：官方編碼簿將此題組的變項名稱印為 as35a–f，經核對實際資料檔，正確名稱為 bs21a–f。

此步驟只清理題目層級的值，不計算合成分數，也不刪除任何樣本。

執行：python scripts/02_clean_data.py

輸入：
  data/TIGPSw1w2_merged_raw.dta   01 的輸出

輸出：
  data/TIGPS_clean.dta                         原始欄位 + 清理後的 *_clean 欄位
  results/step02_control_cleaning_summary.csv  控制變數清理前後的缺失人數
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
RESULTS = BASE / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)

SOURCE = DATA / 'TIGPSw1w2_merged_raw.dta'
TARGET = DATA / 'TIGPS_clean.dta'

MISSING_CODES = {-4, -5, -6, -7, -8, -9, -99, -999}

ONLINE_ITEMS = ['bs25a', 'bs25b', 'bs25c']
OFFLINE_ITEMS = ['bs25d', 'bs25e', 'bs25f']

# 必須跟 01_merge_family_vars.py 的 FAMILY_VARS 保持一致
FAMILY_VARS = ['as1', 'as4d1', 'as10', 'as5', 'as8_1', 'as8_2', 'school_type', 'digi_level']

# 必須跟 04_group_diff_6outcomes.py / 05_all_models.py 的 OUTCOME_ITEMS 保持一致
OUTCOME_ITEMS = {
    'loneliness': ['bs56l'],
    'wellbeing_who5': ['bs58a', 'bs58b', 'bs58c', 'bs58d', 'bs58e'],
    'self_esteem': ['bs53a', 'bs53b', 'bs53c'],
    'life_satisfaction': ['bs50'],
    'cesd12': ['bs56a', 'bs56b', 'bs56c', 'bs56d', 'bs56f', 'bs56g', 'bs56h', 'bs56j', 'bs56k', 'bs56l', 'bs56m', 'bs56n'],
    'friend_support': ['bs29a', 'bs29c', 'bs29d', 'bs29e'],
}

# 延伸一：不同線上/線下自我認同類型的社群媒體使用行為差異
# 目標題目：s21c（玩線上遊戲）、s21e（和他人聊天/傳訊息）。
# 編碼簿（TIGPSw2_cb_s.pdf）上這題組印的變項名稱是 as35a-f，但那是官方編碼簿誤植——
# 已經確認實際資料檔裡正確的欄位名稱是 bs21a-f，這裡要用的是 bs21c / bs21e。
# （跟上一版誤用 bs35c/bs35e 不是同一件事：bs35 那組其實是 s35 同儕霸凌題組，數字只是碰巧
# 兩個題組都叫「35」，bs21 才是真正跟 s21 這個題號對得上的前綴+題號組合。）
EXT_ITEMS = ['bs21c', 'bs21e']
EXT_ITEM_LABELS = {
    'bs21c': '玩線上遊戲(時數代碼1-12)',
    'bs21e': '和他人聊天/傳訊息(時數代碼1-12)',
}
EXT_VALID_CODES = set(range(1, 13))  # 1=沒有 ... 12=5小時以上，共12個有效選項

if not SOURCE.exists():
    raise FileNotFoundError(f'找不到 {SOURCE}，請先執行 01_merge_family_vars.py')

df = pd.read_stata(SOURCE, convert_categoricals=False)
print('合併原始資料筆數：', len(df))


def find_column(df, name):
    """01 的合併只對 school_type / digi_level 加了 _w1/_w2 後綴，其他 6 個家庭變數不會被加後綴。"""
    choices = [name] if name.endswith('_w1') else [f'{name}_w1', name]
    found = [c for c in choices if c in df.columns]
    if not found:
        raise KeyError(f'找不到 {name}；候選欄位：{choices}')
    return found[0]


def clean_numeric(s, valid=None):
    x = pd.to_numeric(s, errors='coerce')
    x = x.mask(x.isin(MISSING_CODES))
    if valid is not None:
        x = x.where(x.isin(valid))
    else:
        x = x.where(x >= 0)
    return x.astype('float64')


def clean_category(s):
    x = s.astype('string').str.strip()
    bad = {str(v) for v in MISSING_CODES} | {'', '<NA>', 'nan', 'None'}
    x = x.mask(x.isin(bad))
    return x.astype(object).where(x.notna(), None)


# ==========================================================
# A. 線上 / 線下自我認同題目（bs25a-f）—— 只清理，不算合成分數
# ==========================================================
online_offline_items = ONLINE_ITEMS + OFFLINE_ITEMS
missing_oo = sorted(set(online_offline_items) - set(df.columns))
if missing_oo:
    raise KeyError(f'缺少線上/線下題目欄位：{missing_oo}')

for item in online_offline_items:
    df[f'{item}_clean'] = clean_numeric(df[item])
df = df.copy()  # 消除逐欄賦值造成的 DataFrame 碎片化（pandas PerformanceWarning），只是效能提醒不是錯誤，但順手清掉

print('\n===== A. 線上/線下題目清理後缺失情況 =====')
print(df[[f'{i}_clean' for i in online_offline_items]].isna().sum().rename('遺漏人數').to_string())

# ==========================================================
# B. 家庭控制變數（8 個）—— 清理 + 敘述統計確認清理成功
# ==========================================================
control_rows = []
for var in FAMILY_VARS:
    source_col = find_column(df, var)
    raw = df[source_col]
    target_col = f'{source_col}_clean'

    if var == 'as4d1':
        # -7（不清楚）保留成獨立類別，其餘缺失代碼才轉成 NaN
        num = pd.to_numeric(raw, errors='coerce')
        valid = num.notna() & (~num.isin(MISSING_CODES - {-7})) & ((num >= 0) | (num == -7))
        df[target_col] = num.where(valid).astype('float64')
    elif var == 'as5':
        df[target_col] = clean_numeric(raw, {1, 2})
    elif var in {'as8_1', 'as8_2'}:
        df[target_col] = clean_numeric(raw, {0, 1})
    elif var == 'school_type':
        df[target_col] = clean_numeric(raw, {1, 2})
    elif var == 'digi_level':
        df[target_col] = clean_category(raw)
    else:  # as1, as10
        df[target_col] = clean_numeric(raw)

    nonmissing = int(df[target_col].notna().sum())
    control_rows.append({
        'variable': var,
        'source_column': source_col,
        'clean_column': target_col,
        'total_N': len(df),
        'raw_missing_N(含合併失敗)': int(raw.isna().sum()) if raw.dtype != object else int(pd.isna(raw).sum()),
        'clean_nonmissing_N': nonmissing,
        'clean_missing_N': len(df) - nonmissing,
    })

df = df.copy()  # 同上，消除碎片化

# ----------------------------------------------------------
# 是否與親生父母同住（合併 as8_1、as8_2）：跟論文 2.2 節的定義一致
#   as8_1（與親生父親同住）或 as8_2（與親生母親同住）任一為 1 → 1（是）
#   兩題都是 0 → 0（否）
#   其他情況（一題 0、另一題缺失，或兩題都缺失）無法判斷 → NaN
# 05 / 05c 的控制變數用的是這個合併後的 as8_any_clean，不再分別放 as8_1、as8_2。
# ----------------------------------------------------------
_f = df['as8_1_clean']
_m = df['as8_2_clean']
df['as8_any_clean'] = np.select([(_f == 1) | (_m == 1), (_f == 0) & (_m == 0)], [1.0, 0.0], default=np.nan)
df = df.copy()
_any_nonmissing = int(df['as8_any_clean'].notna().sum())
control_rows.append({
    'variable': 'as8_any（as8_1/as8_2 合併）',
    'source_column': 'as8_1_clean, as8_2_clean',
    'clean_column': 'as8_any_clean',
    'total_N': len(df),
    'raw_missing_N(含合併失敗)': np.nan,
    'clean_nonmissing_N': _any_nonmissing,
    'clean_missing_N': len(df) - _any_nonmissing,
})

control_summary = pd.DataFrame(control_rows)
print('\n===== B. 家庭控制變數清理前後比較（確認清理是否成功） =====')
print(control_summary.to_string(index=False))
control_summary.to_csv(RESULTS / 'step02_control_cleaning_summary.csv', index=False, encoding='utf-8-sig')

print('\n各控制變數清理後的值分布：')
for var in FAMILY_VARS:
    source_col = find_column(df, var)
    target_col = f'{source_col}_clean'
    print(f'\n-- {target_col} --')
    print(df[target_col].value_counts(dropna=False).sort_index().to_string())

print('\n-- as8_1_clean x as8_2_clean 交叉表（列=父親、欄=母親；確認合併規則用）--')
print(pd.crosstab(df['as8_1_clean'].fillna(-1), df['as8_2_clean'].fillna(-1)).rename(
    index={-1: '缺失'}, columns={-1: '缺失'}).to_string())
print('\n-- as8_any_clean（與至少一位親生父母同住：1=是、0=否）--')
print(df['as8_any_clean'].value_counts(dropna=False).sort_index().to_string())

# ==========================================================
# C. 6 個核心 outcome 的題目 —— 只清理題目層級的值
# ==========================================================
all_outcome_items = sorted({item for items in OUTCOME_ITEMS.values() for item in items})
missing_outcome_items = sorted(set(all_outcome_items) - set(df.columns))
if missing_outcome_items:
    raise KeyError(f'缺少 outcome 題目欄位：{missing_outcome_items}')

for item in all_outcome_items:
    df[f'{item}_clean'] = clean_numeric(df[item])
df = df.copy()  # 同上，消除碎片化

print('\n===== C. 6 個核心 outcome 題目清理後缺失情況 =====')
print(df[[f'{i}_clean' for i in all_outcome_items]].isna().sum().rename('遺漏人數').to_string())

# ==========================================================
# D. 延伸變數：社群媒體使用行為（s21c 線上遊戲、s21e 和他人聊天 —— 正確欄位名稱待確認）
#    只清理，不算合成分數；時數代碼 1-12 以外的值（含問卷缺失代碼）一律轉 NaN
# ==========================================================
if not EXT_ITEMS:
    print('\n===== D. 延伸變數（社群媒體使用行為）=====')
    print('目前 EXT_ITEMS 是空的，跳過這個區塊：s21c/s21e 在實際資料檔裡的正確欄位名稱還沒確認，')
    print('先不清理，避免誤用到名稱相似但題目內容完全不同的欄位（例如 bs35c/bs35e 其實是同儕霸凌題組）。')
else:
    missing_ext = sorted(set(EXT_ITEMS) - set(df.columns))
    if missing_ext:
        raise KeyError(f'缺少延伸變數欄位：{missing_ext}；請確認欄位名稱是否正確。')

    for item in EXT_ITEMS:
        df[f'{item}_clean'] = clean_numeric(df[item], EXT_VALID_CODES)
    df = df.copy()  # 消除逐欄賦值造成的 DataFrame 碎片化（pandas PerformanceWarning），只是效能提醒不是錯誤

    print('\n===== D. 延伸變數（社群媒體使用行為）清理後分布 =====')
    for item in EXT_ITEMS:
        print(f'\n-- {item}_clean（{EXT_ITEM_LABELS[item]}） --')
        print(df[f'{item}_clean'].value_counts(dropna=False).sort_index().to_string())

# ==========================================================
# 儲存（Stata 不支援 pandas StringDtype / object 裡的 pd.NA）
# ==========================================================
for col in df.columns:
    if pd.api.types.is_object_dtype(df[col]):
        df[col] = df[col].where(pd.notna(df[col]), None)

df.to_stata(TARGET, write_index=False, version=118)
print(f'\n已輸出 {TARGET}（下一步 03_quadrant.py 會讀這個檔案）')
print('已輸出 results/step02_control_cleaning_summary.csv')
