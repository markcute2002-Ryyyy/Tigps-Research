"""STEP 05c（圖）｜延伸模型係數森林圖

讀取 05c 的係數與標準誤，繪製四個模型中 usage_composite 係數的 95% 信賴區間。

執行：python scripts/05c_coef_forest_plot.py

輸入：
  results/step05c_usage_on_identity_coefficients.csv   05c 的輸出

輸出：
  results/05c_coef_forest_plot.png
  results/05c_coef_forest_plot.svg
"""
from pathlib import Path
import pandas as pd
import logging
import matplotlib.pyplot as plt

# 字型清單涵蓋不同作業系統（Windows、macOS、Linux），
# 找不到的字型只會產生 findfont 警告，不影響出圖，因此關閉該警告
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
SOURCE = RESULTS / 'step05c_usage_on_identity_coefficients.csv'

plt.rcParams['font.family'] = ['Microsoft JhengHei', 'PingFang TC', 'Noto Sans CJK TC',
                                'Heiti TC', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

if not SOURCE.exists():
    raise FileNotFoundError(f'找不到 {SOURCE}，請先執行 05c_usage_on_identity.py')

df = pd.read_csv(SOURCE)

# 顯示順序（由上到下）：先線上認同（adjusted在上、unadjusted在下），再線下認同
ORDER = [
    ('online_score_adjusted', '線上自我認同 — adjusted（含8項控制變數）'),
    ('online_score_unadjusted', '線上自我認同 — unadjusted'),
    ('offline_score_adjusted', '線下自我認同 — adjusted（含8項控制變數）'),
    ('offline_score_unadjusted', '線下自我認同 — unadjusted'),
]
COLOR_ONLINE = '#4C72B0'   # 跟延伸一長條圖、整份文件維持一致：藍=線上
COLOR_OFFLINE = '#DD8452'  # 橙=線下

rows = []
for model_key, row_label in ORDER:
    r = df.loc[df['模型'] == model_key].iloc[0]
    ci_half = 1.96 * r['標準誤']
    rows.append({
        'label': row_label,
        'coef': r['usage_composite_係數'],
        'lo': r['usage_composite_係數'] - ci_half,
        'hi': r['usage_composite_係數'] + ci_half,
        'N': int(r['N']),
        'color': COLOR_ONLINE if model_key.startswith('online') else COLOR_OFFLINE,
    })

fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
y_pos = list(range(len(rows), 0, -1))

ax.axvline(0, color='#888888', linewidth=1, linestyle='--', zorder=1)

for y, row in zip(y_pos, rows):
    ax.errorbar(row['coef'], y, xerr=[[row['coef'] - row['lo']], [row['hi'] - row['coef']]],
                 fmt='o', color=row['color'], ecolor=row['color'], elinewidth=2,
                 capsize=4, markersize=9, zorder=3)
    ax.text(row['hi'] + 0.0015, y, f"{row['coef']:+.4f}  (N={row['N']:,})",
             va='center', ha='left', fontsize=9)

ax.set_yticks(y_pos)
ax.set_yticklabels([r['label'] for r in rows], fontsize=10)
ax.set_xlabel('usage_composite 的迴歸係數（95% CI）', fontsize=11)
ax.set_title('延伸二：社群媒體使用程度對線上/線下自我認同的關聯\n（迴歸係數森林圖）', fontsize=13)
ax.set_ylim(0.3, len(rows) + 0.7)

handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_ONLINE, markersize=9, label='線上自我認同（DV=online_score）'),
           plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_OFFLINE, markersize=9, label='線下自我認同（DV=offline_score）')]
ax.legend(handles=handles, loc='lower right', fontsize=9, frameon=False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(left=False)
ax.grid(axis='x', color='#e0e0e0', linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(RESULTS / '05c_coef_forest_plot.png')
fig.savefig(RESULTS / '05c_coef_forest_plot.svg')
print('已輸出 results/05c_coef_forest_plot.png、results/05c_coef_forest_plot.svg')
