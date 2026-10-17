"""STEP 06｜產生報告用的 LaTeX 表格

讀取 04、05、05c 輸出的 CSV，產生報告第 3 節的三張表格，不重跑任何模型。
表格格式：係數加顯著性星號（依 FDR 校正後的 p 值），括號內為 HC3 穩健標準誤。

執行：python scripts/06_report_tables.py

輸入（皆在 results/）：
  step04_group_diff_summary.csv
  step05_pairwise_comparisons.csv
  step05_model_coefficients.csv
  step05c_usage_on_identity_coefficients.csv
  step05c_marginal_effects_predictions.csv

輸出（皆在 results/tables/）：
  table_main.tex   各類型平均數（標準差）與調整後組間差異
  table_reg.tex    主要模型迴歸結果
  table_ext.tex    延伸模型迴歸結果
"""
from pathlib import Path
import pandas as pd
from scipy.stats import norm

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
OUT = RESULTS / 'tables'
OUT.mkdir(parents=True, exist_ok=True)

# 欄位順序：先列數值越高越好的四項，再列數值越高越差的兩項
OUTCOMES = [
    ('wellbeing_who5', '幸福感'),
    ('self_esteem', '自尊'),
    ('life_satisfaction', '生活滿意度'),
    ('friend_support', '好友支持'),
    ('loneliness', r'孤獨感$^{\dagger}$'),
    ('cesd12', r'心理困擾$^{\dagger}$'),
]
QTERM = 'C(quadrant_label, Treatment(reference="雙低型"))'
GROUPS = ['雙高型', '線上優勢型', '線下優勢型', '雙低型']
NOTE_STARS = r'$^{*}p<.05$，$^{**}p<.01$，$^{***}p<.001$'
NOTE_DAGGER = r'$^{\dagger}$分數越高代表狀態越差'


def stars(p):
    return '^{***}' if p < .001 else '^{**}' if p < .01 else '^{*}' if p < .05 else ''


def coef_cell(b, p, nd=3):
    return f'${b:.{nd}f}{stars(p)}$'


def se_cell(se, nd=3):
    return f'({se:.{nd}f})'


def r2_cell(r2):
    return f'{r2:.3f}'.lstrip('0')


def table(caption, label, colspec, header_lines, body_lines, note, tabcolsep='5pt'):
    return '\n'.join([
        r'\begin{table}[htbp]', r'\centering',
        rf'\caption{{{caption}}}', rf'\label{{{label}}}',
        r'\footnotesize', rf'\setlength{{\tabcolsep}}{{{tabcolsep}}}', r'\renewcommand{\arraystretch}{1.12}',
        rf'\begin{{tabular}}{{{colspec}}}', r'\toprule', *header_lines, r'\midrule', *body_lines,
        r'\bottomrule', r'\end{tabular}', '', r'\vspace{0.4em}',
        rf'\parbox{{\linewidth}}{{\scriptsize {note}}}', r'\end{table}', '',
    ])


# ======================================================================
# 表：各類型平均數（標準差）與調整後兩兩比較
# ======================================================================
desc = pd.read_csv(RESULTS / 'step04_group_diff_summary.csv').set_index('Outcome')
pair = pd.read_csv(RESULTS / 'step05_pairwise_comparisons.csv')
pair = pair[pair['model'] == 'A_adjusted']
CONTRASTS = [('線上優勢型', '雙低型'), ('線上優勢型', '線下優勢型'), ('雙高型', '線上優勢型'), ('雙高型', '線下優勢型')]

body = []
for key, label in OUTCOMES:
    d = desc.loc[key]
    means = [f'{d[f"{g}_平均數"]:.2f}' for g in GROUPS]
    sds = [f'({d[f"{g}_標準差"]:.2f})' for g in GROUPS]
    eta = f'{d["eta_squared"]:.3f}'.lstrip('0')
    cs, ss = [], []
    for a, b in CONTRASTS:
        r = pair[(pair['outcome'] == key) & (pair['group_A'] == a) & (pair['group_B'] == b)].iloc[0]
        cs.append(coef_cell(r['difference_A_minus_B'], r['p_fdr_six'], 2))
        ss.append(se_cell(r['se_HC3'], 3))
    body.append(f'{label} & ' + ' & '.join(means) + f' & {eta} & ' + ' & '.join(cs) + r' \\')
    body.append(' & ' + ' & '.join(sds) + ' & & ' + ' & '.join(ss) + r' \\[2pt]')
header = [
    r'& \multicolumn{4}{c}{各類型平均數（標準差）} & & \multicolumn{4}{c}{調整後組間差異（四象限分類模型）} \\',
    r'\cmidrule(lr){2-5} \cmidrule(lr){7-10}',
    r'身心狀態指標 & 雙高 & 線上優勢 & 線下優勢 & 雙低 & $\eta^2$ & 線上－雙低 & 線上－線下 & 雙高－線上 & 雙高－線下 \\',
]
note = ('註：平均數為原始分數，六項指標的 ANOVA 與 Kruskal-Wallis 檢定經 FDR 校正後皆 $p<.001$。'
        '組間差異為納入背景控制變數之四象限分類模型的係數線性組合，括號內為 HC3 穩健標準誤，$p$ 值經 FDR 校正；'
        '「線上」「線下」分別指線上優勢型與線下優勢型。' + NOTE_STARS + '。' + NOTE_DAGGER + '。')
(OUT / 'table_main.tex').write_text(
    table('各類型身心狀態平均數與調整後組間差異', 'tab:main', 'l rrrr r rrrr', header, body, note, '4pt'),
    encoding='utf-8')

# ======================================================================
# 表：主要模型迴歸結果（納入背景控制變數）
# ======================================================================
coef = pd.read_csv(RESULTS / 'step05_model_coefficients.csv')


def reg_rows(label, model, term):
    cs, ss = [], []
    for key, _ in OUTCOMES:
        r = coef[(coef['model'] == model) & (coef['term'] == term) & (coef['outcome'] == key)].iloc[0]
        cs.append(coef_cell(r['coef'], r['p_fdr_six']))
        ss.append(se_cell(r['se_HC3']))
    return [rf'\quad {label} & ' + ' & '.join(cs) + r' \\', ' & ' + ' & '.join(ss) + r' \\[1pt]']


def model_stats(model):
    r2, n = [], []
    for key, _ in OUTCOMES:
        r = coef[(coef['model'] == model) & (coef['outcome'] == key)].iloc[0]
        r2.append(r2_cell(r['r_squared']))
        n.append(f'{int(r["nobs"]):,}')
    return r2, n


r2a, na = model_stats('A_adjusted')
r2b, nb = model_stats('B_adjusted')
if na != nb:
    raise ValueError('模型 A 與模型 B 的樣本數不一致，請檢查 05_all_models.py 的輸出')
body = [r'\multicolumn{7}{l}{\textbf{A. 四象限分類模型}（對照組：雙低型）} \\']
body += reg_rows('線上優勢型', 'A_adjusted', f'{QTERM}[T.線上優勢型]')
body += reg_rows('線下優勢型', 'A_adjusted', f'{QTERM}[T.線下優勢型]')
body += reg_rows('雙高型', 'A_adjusted', f'{QTERM}[T.雙高型]')
body += [r'\quad $R^2$ & ' + ' & '.join(r2a) + r' \\', r'\midrule',
         r'\multicolumn{7}{l}{\textbf{B. 連續分數交互作用模型}} \\']
body += reg_rows('線上自我認同', 'B_adjusted', 'online_c')
body += reg_rows('線下自我認同', 'B_adjusted', 'offline_c')
body += reg_rows(r'線上 $\times$ 線下', 'B_adjusted', 'online_c:offline_c')
body += [r'\quad $R^2$ & ' + ' & '.join(r2b) + r' \\', r'\midrule',
         r'背景控制變數 & ' + ' & '.join(['是'] * len(OUTCOMES)) + r' \\',
         r'$N$ & ' + ' & '.join(na) + r' \\']
header = ['& ' + ' & '.join(l for _, l in OUTCOMES) + r' \\']
note = ('註：括號內為 HC3 穩健標準誤，$p$ 值依六項指標進行 FDR 校正；背景控制變數之係數未列出。'
        '模型 B 的線上與線下自我認同分數經平均數中心化。' + NOTE_STARS + '。' + NOTE_DAGGER + '。')
(OUT / 'table_reg.tex').write_text(
    table('主要模型迴歸結果', 'tab:reg', 'l rrrrrr', header, body, note, '6pt'), encoding='utf-8')

# ======================================================================
# 表：延伸模型
# ======================================================================
ext = pd.read_csv(RESULTS / 'step05c_usage_on_identity_coefficients.csv').set_index('模型')
pred = pd.read_csv(RESULTS / 'step05c_marginal_effects_predictions.csv')
COLS = ['online_score_unadjusted', 'online_score_adjusted', 'offline_score_unadjusted', 'offline_score_adjusted']


def other_term(model):
    return 'offline_score' if model.startswith('online') else 'online_score'


def other_se(r, model):
    """另一種認同分數的標準誤：05c 新版會直接存；舊版輸出沒有存，就用係數與 p 值反推
    （HC3 推論採常態分配，SE = |b| / z_{p/2}，跟 statsmodels 的計算完全一致）。"""
    o = other_term(model)
    col = f'對方認同分數({o})_標準誤'
    if col in r.index and pd.notna(r[col]):
        return r[col]
    b, p = r[f'對方認同分數({o})_係數'], r[f'對方認同分數({o})_p值']
    return abs(b) / norm.isf(p / 2)


usage_c, usage_s, oth_c, oth_s, r2, n, ctrl = [], [], [], [], [], [], []
for m in COLS:
    r = ext.loc[m]
    o = other_term(m)
    # 使用程度每增加 1 分的係數很小（約 0.02），取到小數點後 4 位才看得出標準誤
    usage_c.append(coef_cell(r['usage_composite_係數'], r['p值_FDR_BH'], 4))
    usage_s.append(se_cell(r['標準誤'], 4))
    oth_c.append(coef_cell(r[f'對方認同分數({o})_係數'], r[f'對方認同分數({o})_p值']))
    oth_s.append(se_cell(other_se(r, m)))
    r2.append(r2_cell(r['R_squared']))
    n.append(f'{int(r["N"]):,}')
    ctrl.append('是' if 'adjusted' in m and 'unadjusted' not in m else '否')
change = {}
for dv, g in pred.groupby('DV'):
    g = g.sort_values('usage_composite')
    change[dv] = g['mean'].iloc[-1] - g['mean'].iloc[0]
pred_row = ['--', f'${change["線上自我認同"]:+.2f}$', '--', f'${change["線下自我認同"]:+.2f}$']

body = [
    '社群媒體使用程度 & ' + ' & '.join(usage_c) + r' \\', ' & ' + ' & '.join(usage_s) + r' \\[1pt]',
    '另一種自我認同分數 & ' + ' & '.join(oth_c) + r' \\', ' & ' + ' & '.join(oth_s) + r' \\[1pt]',
    r'\midrule',
    '背景控制變數 & ' + ' & '.join(ctrl) + r' \\',
    '預測變化（P5 至 P95） & ' + ' & '.join(pred_row) + r' \\',
    r'$N$ & ' + ' & '.join(n) + r' \\',
    r'$R^2$ & ' + ' & '.join(r2) + r' \\',
]
header = [
    r'& \multicolumn{2}{c}{依變數：線上自我認同} & \multicolumn{2}{c}{依變數：線下自我認同} \\',
    r'\cmidrule(lr){2-3} \cmidrule(lr){4-5}',
    r'& 未調整 & 調整後 & 未調整 & 調整後 \\',
]
note = ('註：括號內為 HC3 穩健標準誤；社群媒體使用程度之 $p$ 值經 FDR 校正。'
        '「另一種自我認同分數」在線上認同模型中為線下認同，在線下認同模型中為線上認同。'
        '預測變化為使用程度由第 5 百分位提高至第 95 百分位時，依樣本實際分布平均之預測值變化。' + NOTE_STARS + '。')
(OUT / 'table_ext.tex').write_text(
    table('社群媒體使用程度與線上、線下自我認同之迴歸結果', 'tab:ext', 'l cccc', header, body, note, '8pt'),
    encoding='utf-8')

print('已輸出 results/tables/table_main.tex、table_reg.tex、table_ext.tex')
