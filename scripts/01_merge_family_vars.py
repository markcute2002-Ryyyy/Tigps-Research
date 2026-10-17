"""STEP 01｜跨波資料合併

以第二波（W2）學生問卷的全部樣本為主，用學生識別碼 nstudent_id 將第一波（W1）的
8 個家庭背景變數合併進來（left join），並檢查配對情形。此步驟不刪除任何樣本；
未配對成功的學生，其家庭背景變數為缺失。

合併的家庭背景變數：
    as1          性別
    as4d1        第一位主要照顧者學歷
    as10         主觀家庭經濟狀況
    as5          是否有兄弟姊妹
    as8_1        是否與親生父親同住
    as8_2        是否與親生母親同住
    school_type  學校類型（兩波皆有，合併後為 school_type_w1 / school_type_w2）
    digi_level   數位發展區域（同上）

執行：python scripts/01_merge_family_vars.py

輸入：
  data/TIGPSw1_s.dta   第一波學生問卷
  data/TIGPSw2_s.dta   第二波學生問卷

輸出：
  data/TIGPSw1w2_merged_raw.dta                   合併後資料（含 _merge 配對指標）
  results/step01_merge_check.csv                  配對情形
  results/step01_family_var_missing_after_merge.csv  各家庭變數合併後的缺失人數
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
RESULTS = BASE / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)

W1_FILE = DATA / 'TIGPSw1_s.dta'
W2_FILE = DATA / 'TIGPSw2_s.dta'
OUT_FILE = DATA / 'TIGPSw1w2_merged_raw.dta'

# 05_all_models.py 的 CONTROLS 清單去掉 "_clean" / "_w1" 後對應的原始欄位名稱。
# 這是唯一一個之後要跟 05 同步維護的清單。
FAMILY_VARS = ['as1', 'as4d1', 'as10', 'as5', 'as8_1', 'as8_2', 'school_type', 'digi_level']

if not W1_FILE.exists():
    raise FileNotFoundError(f'找不到 {W1_FILE}；請把 Wave 1 原始檔放到這個路徑，或修改 W1_FILE。')
if not W2_FILE.exists():
    raise FileNotFoundError(f'找不到 {W2_FILE}；請把 Wave 2 原始檔放到這個路徑，或修改 W2_FILE。')

# ----------------------------------------
# 1. 讀取兩波原始資料
# ----------------------------------------
w1_full = pd.read_stata(W1_FILE, convert_categoricals=False)
w2 = pd.read_stata(W2_FILE, convert_categoricals=False)

print('W1 原始樣本數：', len(w1_full))
print('W2 原始樣本數（全量，尚未篩選）：', len(w2))

missing_in_w1 = sorted(set(FAMILY_VARS) - set(w1_full.columns))
if missing_in_w1:
    raise KeyError(f'W1 檔案裡缺少這些家庭變數，請檢查欄位名稱：{missing_in_w1}')

# 只留 nstudent_id + 8 個家庭變數，避免把 W1 全部欄位無差別併進來
w1 = w1_full[['nstudent_id'] + FAMILY_VARS].copy()

# ----------------------------------------
# 2. 合併前的 ID 檢查
# ----------------------------------------
assert 'nstudent_id' in w1.columns and 'nstudent_id' in w2.columns, '兩波都必須有 nstudent_id'
assert w1['nstudent_id'].notna().all(), 'W1 的 nstudent_id 有遺漏'
assert w2['nstudent_id'].notna().all(), 'W2 的 nstudent_id 有遺漏'
assert w1['nstudent_id'].is_unique, 'W1 的 nstudent_id 有重複，請先確認原始檔'
assert w2['nstudent_id'].is_unique, 'W2 的 nstudent_id 有重複，請先確認原始檔'
print('\n兩波 ID 檢查通過（無遺漏、無重複）。')

# ----------------------------------------
# 3. 合併：以 W2 全量為主（left），接上 W1 的家庭變數
# ----------------------------------------
merged = w2.merge(
    w1,
    on='nstudent_id',
    how='left',
    suffixes=('_w2', '_w1'),
    validate='one_to_one',
    indicator=True,
)

# how='left' 以 W2 為主，理論上合併後筆數必須跟 W2 原始筆數一致
assert len(merged) == len(w2), '合併後筆數跟 W2 原始筆數不一致，請檢查 merge 設定'

# ----------------------------------------
# 4. 檢查合併結果（算在全部 W2 樣本上）
# ----------------------------------------
print('\n===== 合併結果（_merge，以全部 W2 樣本為分母） =====')
merge_counts = merged['_merge'].value_counts()
print(merge_counts.to_string())

matched = (merged['_merge'] == 'both').sum()
match_rate = matched / len(w2) * 100
print(f'\n成功合併（both）人數：{matched}')
print(f'合併成功率（以全部 W2 樣本為分母）：{match_rate:.2f}%')

# 8 個家庭變數各自的合併後缺失情況（合併成功 != 這題有填）
family_cols_after_merge = []
for var in FAMILY_VARS:
    col = f'{var}_w1' if f'{var}_w1' in merged.columns else var
    family_cols_after_merge.append(col)
missing_summary = merged[family_cols_after_merge].isna().sum().rename('遺漏人數(含合併失敗)')
print('\n===== 8 個家庭變數合併後的遺漏概況 =====')
print(missing_summary.to_string())

check_df = pd.concat([
    merge_counts.rename('人數'),
], axis=0).to_frame()
missing_summary.to_frame().to_csv(RESULTS / 'step01_family_var_missing_after_merge.csv', encoding='utf-8-sig')
merge_counts.to_csv(RESULTS / 'step01_merge_check.csv', encoding='utf-8-sig', header=['人數'])

# ----------------------------------------
# 5. 儲存
# ----------------------------------------
merged.to_stata(OUT_FILE, write_index=False, version=118)
print(f'\n已輸出 {OUT_FILE}（下一步 02_clean_data.py 會讀這個檔案）')
print('已輸出 results/step01_merge_check.csv、results/step01_family_var_missing_after_merge.csv')
