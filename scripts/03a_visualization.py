"""STEP 03a｜四象限分布圖

繪製線上與線下自我認同分數的分布熱力圖，並標示中位數分界線。

執行：python scripts/03a_visualization.py

輸入：
  data/TIGPS_quadrant_clean.dta   03 的輸出

輸出：
  results/quadrant_heatmap.png
  results/quadrant_heatmap.svg
"""
from pathlib import Path
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

# 字型清單涵蓋不同作業系統，找不到的字型只會產生 findfont 警告，不影響出圖
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import seaborn as sns

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
RESULTS = BASE / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)

SOURCE = DATA / 'TIGPS_quadrant_clean.dta'

if not SOURCE.exists():
    raise FileNotFoundError(f'找不到 {SOURCE}，請先執行 03_quadrant.py')

df = pd.read_stata(SOURCE, convert_categoricals=False)
df = df.copy()  # 從Stata讀回來的欄位一多就容易被判定成碎片化，讀完先整理一次，避免後面逐欄賦值時跳PerformanceWarning
print('有效樣本數：', len(df))

# ----------------------------------------
# 建立 10 x 10 人數矩陣
# ----------------------------------------
df['online_idx'] = np.rint((df['online_score'] - 1) * 3).astype(int)
df['offline_idx'] = np.rint((df['offline_score'] - 1) * 3).astype(int)

heatmap_data = pd.crosstab(df['offline_idx'], df['online_idx'])
heatmap_data = heatmap_data.reindex(index=range(10), columns=range(10), fill_value=0)
assert heatmap_data.to_numpy().sum() == len(df)

# ----------------------------------------
# 繪圖風格（Windows 優先字型）
# ----------------------------------------
sns.set_theme(style='white')
plt.rcParams['font.family'] = ['Microsoft JhengHei', 'PingFang TC', 'Noto Sans CJK TC',
                                'Heiti TC', 'Microsoft YaHei', 'sans-serif']

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    heatmap_data, cmap='Blues', annot=True, fmt='d',
    linewidths=1.2, linecolor='white', square=True,
    cbar_kws={'label': '樣本數', 'shrink': 0.75},
    annot_kws={'fontsize': 9}, ax=ax,
)

scores = [f'{1 + i/3:.2f}' for i in range(10)]
ax.set_xticklabels(scores, rotation=0)
ax.set_yticklabels(scores, rotation=0)
ax.invert_yaxis()
ax.set_xlabel('線上自我認同分數', fontsize=14, labelpad=12)
ax.set_ylabel('線下自我認同分數', fontsize=14, labelpad=12)

ax.axvline(x=5, color='#333333', linestyle='--', linewidth=2)
ax.axhline(y=6, color='#333333', linestyle='--', linewidth=2)

quadrant_info = [
    (1, '雙高型', 7.5, 8.3),
    (2, '線上優勢型', 7.5, 2.5),
    (3, '線下優勢型', 2.5, 8.3),
    (4, '雙低型', 2.5, 2.5),
]
for q, name, x, y in quadrant_info:
    n = (df['quadrant'] == q).sum()
    pct = n / len(df) * 100
    ax.text(
        x, y, f'{name}\nN={n:,} ({pct:.1f}%)',
        ha='center', va='center', fontsize=11, fontweight='bold', color='#26364A',
        bbox={'facecolor': 'white', 'alpha': 0.9, 'edgecolor': 'none', 'boxstyle': 'round,pad=0.4'},
    )

ax.set_title('青少年線上與線下自我認同四象限分布', fontsize=20, fontweight='bold', pad=22)
fig.text(
    0.5, 0.025,
    f'註：線上自我認同中位數 = {df["online_score"].median():.2f}、'
    f'線下自我認同中位數 = {df["offline_score"].median():.2f}，高認同組包含等於中位數者。',
    ha='center', fontsize=11, color='#555555',
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(RESULTS / 'quadrant_heatmap.png', dpi=300, bbox_inches='tight')
plt.savefig(RESULTS / 'quadrant_heatmap.svg', bbox_inches='tight')
plt.show()

print('\n熱力圖已成功儲存至 results 資料夾！')
print('若圖中中文顯示為方框，請確認系統已安裝中文字型（Windows：微軟正黑體；macOS：蘋方）。')
