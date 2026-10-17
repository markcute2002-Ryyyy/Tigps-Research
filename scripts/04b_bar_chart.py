"""STEP 04b（圖）｜四象限 × 社群媒體使用時數長條圖

讀取 04b 輸出的描述統計繪製長條圖，誤差線為標準誤。不需讀取原始資料。

執行：python scripts/04b_bar_chart.py

輸入：
  results/step04b_ext_by_quadrant_long.csv   04b 的輸出

輸出：
  results/04b_quadrant_bar_chart.png
  results/04b_quadrant_bar_chart.svg
"""
from pathlib import Path
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

# 字型清單涵蓋不同作業系統（Windows、macOS、Linux），
# 找不到的字型只會產生 findfont 警告，不影響出圖，因此關閉該警告
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
SOURCE = RESULTS / 'step04b_ext_by_quadrant_long.csv'

plt.rcParams['font.family'] = ['Microsoft JhengHei', 'PingFang TC', 'Noto Sans CJK TC',
                                'Heiti TC', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

GROUPS = ['雙高型', '線上優勢型', '線下優勢型', '雙低型']
ITEMS = ['bs21c', 'bs21e']
ITEM_LABELS = {
    'bs21c': '玩線上遊戲',
    'bs21e': '和他人聊天/傳訊息',
}

if not SOURCE.exists():
    raise FileNotFoundError(f'找不到 {SOURCE}，請先執行 04b_extension_social_media_behavior.py')

long = pd.read_csv(SOURCE)

means = {item: [long.loc[(long['變項'] == item) & (long['象限'] == g), '平均數'].iloc[0] for g in GROUPS]
         for item in ITEMS}
sds = {item: [long.loc[(long['變項'] == item) & (long['象限'] == g), '標準差'].iloc[0] for g in GROUPS]
       for item in ITEMS}
ns = {item: [long.loc[(long['變項'] == item) & (long['象限'] == g), '有效N'].iloc[0] for g in GROUPS]
      for item in ITEMS}

x = np.arange(len(GROUPS))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

colors = ['#4C72B0', '#DD8452']
for i, item in enumerate(ITEMS):
    sem = [sds[item][j] / np.sqrt(ns[item][j]) for j in range(len(GROUPS))]
    offset = (i - 0.5) * width
    bars = ax.bar(x + offset, means[item], width, yerr=sem, capsize=4,
                   label=f'{item} {ITEM_LABELS[item]}', color=colors[i])
    for bar, val in zip(bars, means[item]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.12, f'{val:.2f}',
                 ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(GROUPS, fontsize=11)
ax.set_ylabel('平均時數代碼（1=沒有 … 12=5小時以上）', fontsize=11)
ax.set_title('四象限 x 社群媒體使用行為（延伸一）\n誤差線為標準誤（SEM）', fontsize=13)
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(0, max(max(means['bs21c']), max(means['bs21e'])) + 1.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig(RESULTS / '04b_quadrant_bar_chart.png')
fig.savefig(RESULTS / '04b_quadrant_bar_chart.svg')
print('已輸出 results/04b_quadrant_bar_chart.png、results/04b_quadrant_bar_chart.svg')
