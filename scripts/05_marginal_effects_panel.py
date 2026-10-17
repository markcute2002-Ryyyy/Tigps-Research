"""STEP 05（圖）｜線上自我認同之邊際效果（2×3 小圖）

呈現模型 B（納入控制變數）中，線上自我認同的斜率如何隨線下自我認同改變，
六項指標排成 2×3 小圖。讀取 05 的輸出，不需重跑模型。

執行：python scripts/05_marginal_effects_panel.py

輸入：
  results/step05_marginal_effect_curves.csv   05 的輸出

輸出：
  results/05_marginal_effects_panel.pdf
  results/05_marginal_effects_panel.png
"""
from pathlib import Path
import logging
import pandas as pd
import matplotlib.pyplot as plt

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
SOURCE = RESULTS / 'step05_marginal_effect_curves.csv'

plt.rcParams['font.family'] = ['Microsoft JhengHei', 'PingFang TC', 'Noto Sans CJK TC',
                                'Heiti TC', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9

# 上排：數值越高越好；下排：好友支持（越高越好）+ 兩項數值越高越差的指標
PANELS = [
    ('wellbeing_who5', '幸福感（WHO-5）'),
    ('self_esteem', '自尊'),
    ('life_satisfaction', '生活滿意度'),
    ('friend_support', '好友支持'),
    ('loneliness', '孤獨感（越高越差）'),
    ('cesd12', '心理困擾 CESD-12（越高越差）'),
]
COLOR = '#4C72B0'

if not SOURCE.exists():
    raise FileNotFoundError(f'找不到 {SOURCE}，請先執行 05_all_models.py')

curves = pd.read_csv(SOURCE)
curves = curves[curves['direction'] == 'online_given_offline']

fig, axes = plt.subplots(2, 3, figsize=(6.9, 3.9), sharex=True)
for ax, (outcome, title) in zip(axes.flat, PANELS):
    d = curves[curves['outcome'] == outcome].sort_values('conditioning_score')
    if d.empty:
        raise ValueError(f'{outcome} 沒有邊際效果資料')
    ax.fill_between(d['conditioning_score'], d['ci_low'], d['ci_high'], color=COLOR, alpha=0.2, linewidth=0)
    ax.plot(d['conditioning_score'], d['marginal_effect'], color=COLOR, linewidth=1.8)
    ax.axhline(0, color='#777777', linewidth=0.8, linestyle='--')
    ax.set_title(title, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)
    lo, hi = min(d['ci_low'].min(), 0), max(d['ci_high'].max(), 0)
    pad = (hi - lo) * 0.08
    ax.set_ylim(lo - pad, hi + pad)

for ax in axes[1]:
    ax.set_xlabel('線下自我認同分數', fontsize=8.5)
for ax in axes[:, 0]:
    ax.set_ylabel('線上認同的斜率', fontsize=8.5)

fig.tight_layout(h_pad=0.8, w_pad=0.6)
fig.savefig(RESULTS / '05_marginal_effects_panel.pdf')
fig.savefig(RESULTS / '05_marginal_effects_panel.png', dpi=200)
plt.close(fig)
print('已輸出 results/05_marginal_effects_panel.pdf、results/05_marginal_effects_panel.png')
