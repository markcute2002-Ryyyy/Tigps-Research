"""STEP 05｜主要模型

對六項身心狀態指標分別估計兩組 OLS 模型，皆採 HC3 穩健標準誤：
  模型 A（四象限分類模型）：以雙低型為對照組的類型虛擬變數
  模型 B（連續分數交互作用模型）：平均數中心化後的線上、線下自我認同分數及其交互作用項
每個模型估計三種版本：未納入控制變數、納入控制變數、以及以納入控制變數後的樣本重新估計的
未調整模型（用以區分係數變化來自控制變數或樣本差異）。控制變數皆以類別變數納入。

另計算：模型 A 的組間兩兩比較（係數線性組合）、模型 B 中線上認同在不同線下認同水準下的
簡單斜率與邊際效果曲線。p 值依六項指標以 Benjamini-Hochberg 法進行 FDR 校正。

執行：python scripts/05_all_models.py

輸入：
  data/TIGPS_quadrant_clean.dta   03 的輸出

輸出：
  results/step05_model_coefficients.csv     所有模型的係數、HC3 標準誤、信賴區間、R squared
  results/step05_pairwise_comparisons.csv   模型 A 的組間兩兩比較
  results/step05_model_samples.csv          各模型樣本數
  results/step05_online_marginal_slopes.csv 線上認同的簡單斜率（線下認同第 25、50、75 百分位）
  results/step05_marginal_effect_curves.csv 邊際效果曲線
  results/step05_marginal_*.png             各指標的邊際效果圖
  results/step05_interaction_coefficients.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
OUT = BASE / 'results'
OUT.mkdir(parents=True, exist_ok=True)

SOURCE = DATA / 'TIGPS_quadrant_clean.dta'
QUADS = ['雙低型', '線上優勢型', '線下優勢型', '雙高型']

# 必須跟 02_clean_data.py / 04_group_diff_6outcomes.py 的 OUTCOME_ITEMS 保持一致
OUTCOME_ITEMS = {
    'loneliness': ['bs56l'],
    'wellbeing_who5': ['bs58a', 'bs58b', 'bs58c', 'bs58d', 'bs58e'],
    'self_esteem': ['bs53a', 'bs53b', 'bs53c'],
    'life_satisfaction': ['bs50'],
    'cesd12': ['bs56a', 'bs56b', 'bs56c', 'bs56d', 'bs56f', 'bs56g', 'bs56h', 'bs56j', 'bs56k', 'bs56l', 'bs56m', 'bs56n'],
    'friend_support': ['bs29a', 'bs29c', 'bs29d', 'bs29e'],
}
OUTCOME_LABELS = {
    'loneliness': 'Loneliness',
    'wellbeing_who5': 'WHO-5 wellbeing',
    'self_esteem': 'Self-esteem',
    'life_satisfaction': 'Life satisfaction',
    'cesd12': 'CESD-12 distress',
    'friend_support': 'Friend support',
}

# 必須跟 01_merge_family_vars.py 的 FAMILY_VARS / 02_clean_data.py 產生的 *_clean 欄位一致
CONTROLS = [
    'as1_clean', 'as4d1_clean', 'as10_clean', 'as5_clean',
    'as8_any_clean', 'school_type_w1_clean', 'digi_level_w1_clean',
]


def load_data():
    if not SOURCE.exists():
        raise FileNotFoundError(f'找不到 {SOURCE}；請先依序執行 01→02→03。')
    df = pd.read_stata(SOURCE, convert_categoricals=False)
    df = df.copy()  # 從Stata讀回來的欄位一多就容易被判定成碎片化，讀完先整理一次，避免後面逐欄賦值時跳PerformanceWarning

    if not df['nstudent_id'].is_unique:
        raise ValueError('nstudent_id 有重複，無法把每一列當成獨立學生處理')

    missing_controls = sorted(set(CONTROLS) - set(df.columns))
    if missing_controls:
        raise KeyError(f'缺少清理後的控制變數：{missing_controls}；請確認 02_clean_data.py 有正常跑完。')

    if pd.api.types.is_numeric_dtype(df['quadrant_label']):
        labels = pd.read_stata(SOURCE, convert_categoricals=True, columns=['quadrant_label'])
        df['quadrant_label'] = labels['quadrant_label'].astype(str)
    else:
        df['quadrant_label'] = df['quadrant_label'].astype(str)

    observed = sorted(df['quadrant_label'].dropna().unique().tolist())
    if not set(QUADS).issubset(set(observed)):
        raise ValueError(f'四象限標籤不齊：{observed}；請檢查 03_quadrant.py 的輸出。')
    df = df[df['quadrant_label'].isin(QUADS)].copy()

    for outcome, items in OUTCOME_ITEMS.items():
        clean_items = [f'{i}_clean' for i in items]
        missing = sorted(set(clean_items) - set(df.columns))
        if missing:
            raise KeyError(f'{outcome} 缺少清理後欄位：{missing}')
        df[outcome] = df[clean_items].mean(axis=1, skipna=False)
    df = df.copy()  # 消除逐欄賦值造成的 DataFrame 碎片化（pandas PerformanceWarning），只是效能提醒不是錯誤

    for c in CONTROLS:
        df[c] = df[c].astype(object).where(pd.notna(df[c]), np.nan)

    return df


def marginal_effects_and_plots(model, sub, outcome, online_mean, offline_mean):
    """Two continuous simple-slope curves from the SAME adjusted interaction OLS.

    Uses model.t_test (HC3 covariance) for each linear combination.
    Plots only the 5th--95th percentile of the observed conditioning score.
    """
    output_rows = []
    interaction = 'online_c:offline_c'
    for direction, varying, conditioning, main_term, center in [
        ('online_given_offline', 'online_score', 'offline_score', 'online_c', offline_mean),
        ('offline_given_online', 'offline_score', 'online_score', 'offline_c', online_mean),
    ]:
        observed = sub[conditioning].dropna()
        lo, hi = (float(observed.quantile(q)) for q in (0.05, 0.95))
        if lo == hi:
            print(f'WARNING: {outcome} {direction}: 5th and 95th percentiles coincide; skipped')
            continue
        grid = np.linspace(lo, hi, 101)
        for score in grid:
            L = np.zeros(len(model.params))
            L[model.params.index.get_loc(main_term)] = 1.0
            L[model.params.index.get_loc(interaction)] = score - center
            test = model.t_test(L)
            ci = np.asarray(test.conf_int(alpha=0.05)).reshape(-1)
            output_rows.append({
                'outcome': outcome, 'model': 'B_adjusted', 'direction': direction,
                'conditioning_variable': conditioning,
                'conditioning_score': float(score),
                'marginal_effect': float(np.asarray(test.effect).item()),
                'se_HC3': float(np.asarray(test.sd).item()),
                'p': float(np.asarray(test.pvalue).item()),
                'ci_low': float(ci[0]), 'ci_high': float(ci[1]),
                'N': int(model.nobs), 'range_p05': lo, 'range_p95': hi,
            })
        part = pd.DataFrame(output_rows).query('direction == @direction and outcome == @outcome')
        fig, ax = plt.subplots(figsize=(8, 5))
        x = part['conditioning_score'].to_numpy(dtype=float)
        effect = part['marginal_effect'].to_numpy(dtype=float)
        lower = part['ci_low'].to_numpy(dtype=float)
        upper = part['ci_high'].to_numpy(dtype=float)
        ax.plot(x, effect, lw=2, label='Marginal effect')
        ax.fill_between(x, lower, upper, alpha=0.2, label='95% HC3 CI')
        ax.axhline(0, color='gray', ls='--', lw=1)
        condition_label = 'Offline identity' if conditioning == 'offline_score' else 'Online identity'
        effect_label = 'Online identity' if varying == 'online_score' else 'Offline identity'
        ax.set_xlabel(f'{condition_label} score (observed P5-P95)')
        ax.set_ylabel(f'Marginal effect of {effect_label} (+1 score point)')
        ax.set_title(f'{OUTCOME_LABELS[outcome]} | B adjusted | {direction}')
        ax.legend(loc='best')
        fig.tight_layout()
        target = OUT / f'step05_marginal_{direction}_{outcome}.png'
        fig.savefig(target, dpi=180)
        plt.close(fig)
        print('Saved:', target)
    return output_rows


def fit_and_save(df):
    coef, comparisons, samples, prediction, marginal_curves = [], [], [], [], []
    from patsy.contrasts import Treatment  # noqa: F401  (patsy formula string below needs this in scope)
    qterm = 'C(quadrant_label, Treatment(reference="雙低型"))'
    controls_formula = ' + '.join(f'C({c})' for c in CONTROLS)

    for outcome in OUTCOME_ITEMS:
        base_cols = [outcome, 'quadrant_label', 'online_score', 'offline_score']
        base = df[base_cols + CONTROLS].dropna(subset=base_cols).copy()
        adjusted = base.dropna(subset=CONTROLS).copy()
        if len(base) < 100 or len(adjusted) < 100:
            print(f'WARNING: {outcome}: unadjusted N={len(base)}, adjusted N={len(adjusted)}; skipping')
            continue

        specs = {
            'A_unadjusted': (f'{outcome} ~ {qterm}', base),
            'A_adjusted': (f'{outcome} ~ {qterm} + {controls_formula}', adjusted),
            'B_unadjusted': (f'{outcome} ~ online_c * offline_c', base),
            'B_adjusted': (f'{outcome} ~ online_c * offline_c + {controls_formula}', adjusted),
            'A_unadjusted_same_sample': (f'{outcome} ~ {qterm}', adjusted),
            'B_unadjusted_same_sample': (f'{outcome} ~ online_c * offline_c', adjusted),
        }
        online_mean = base['online_score'].mean()
        offline_mean = base['offline_score'].mean()

        for model_name, (formula, sub0) in specs.items():
            sub = sub0.copy()
            sub['quadrant_label'] = pd.Categorical(sub['quadrant_label'], categories=QUADS)
            sub['online_c'] = sub['online_score'] - online_mean
            sub['offline_c'] = sub['offline_score'] - offline_mean
            model = smf.ols(formula, data=sub).fit(cov_type='HC3')

            samples.append({
                'outcome': outcome, 'model': model_name, 'regression_N': int(model.nobs),
                'online_mean': sub['online_score'].mean(), 'offline_mean': sub['offline_score'].mean(),
                **{f'N_{q}': int((sub['quadrant_label'] == q).sum()) for q in QUADS},
            })

            ci = model.conf_int()
            for term in model.params.index:
                coef.append({
                    'outcome': outcome, 'model': model_name, 'term': term,
                    'coef': model.params[term], 'se_HC3': model.bse[term],
                    'p': model.pvalues[term], 'ci_low': ci.loc[term, 0], 'ci_high': ci.loc[term, 1],
                    'nobs': int(model.nobs), 'r_squared': model.rsquared,
                })

            if model_name.startswith('A'):
                for a, b in [('線上優勢型', '雙低型'), ('線上優勢型', '線下優勢型'),
                             ('雙高型', '線上優勢型'), ('雙高型', '線下優勢型')]:
                    def qcoef(q):
                        return None if q == '雙低型' else f'{qterm}[T.{q}]'
                    L = np.zeros(len(model.params))
                    if qcoef(a):
                        L[model.params.index.get_loc(qcoef(a))] += 1
                    if qcoef(b):
                        L[model.params.index.get_loc(qcoef(b))] -= 1
                    test = model.t_test(L)
                    ci2 = np.asarray(test.conf_int()).reshape(-1)
                    comparisons.append({
                        'outcome': outcome, 'model': model_name, 'group_A': a, 'group_B': b,
                        'difference_A_minus_B': float(np.asarray(test.effect).item()),
                        'se_HC3': float(np.asarray(test.sd).item()),
                        'p': float(np.asarray(test.pvalue).item()),
                        'ci_low': ci2[0], 'ci_high': ci2[1], 'N': int(model.nobs),
                    })

            if model_name == 'B_adjusted':
                marginal_curves.extend(marginal_effects_and_plots(model, sub, outcome, online_mean, offline_mean))
                for pct in [25, 50, 75]:
                    offline = float(sub['offline_score'].quantile(pct / 100))
                    offline_c = offline - offline_mean
                    L = np.zeros(len(model.params))
                    L[model.params.index.get_loc('online_c')] = 1
                    L[model.params.index.get_loc('online_c:offline_c')] = offline_c
                    test = model.t_test(L)
                    ci3 = np.asarray(test.conf_int()).reshape(-1)
                    prediction.append({
                        'outcome': outcome, 'offline_percentile': pct, 'offline_score': offline,
                        'online_slope': float(np.asarray(test.effect).item()),
                        'p': float(np.asarray(test.pvalue).item()),
                        'ci_low': ci3[0], 'ci_high': ci3[1], 'N': int(model.nobs),
                    })

        print(f'{outcome}: unadjusted N={len(base)} | adjusted N={len(adjusted)} | six models complete')

    coef = pd.DataFrame(coef)
    comparisons = pd.DataFrame(comparisons)
    samples = pd.DataFrame(samples)
    prediction = pd.DataFrame(prediction)
    marginal_curves = pd.DataFrame(marginal_curves)

    if coef.empty:
        raise RuntimeError('沒有任何迴歸結果；請檢查資料缺失狀況跟欄位名稱。')

    coef['p_fdr_six'] = np.nan
    focus = coef['term'].str.contains('quadrant_label|online_c|offline_c', regex=True)
    for (_, _), idx in coef.loc[focus].groupby(['model', 'term']).groups.items():
        coef.loc[idx, 'p_fdr_six'] = multipletests(coef.loc[idx, 'p'], method='fdr_bh')[1]

    comparisons['p_fdr_six'] = np.nan
    for (_, _, _), idx in comparisons.groupby(['model', 'group_A', 'group_B']).groups.items():
        comparisons.loc[idx, 'p_fdr_six'] = multipletests(comparisons.loc[idx, 'p'], method='fdr_bh')[1]

    prediction['p_fdr_six'] = np.nan
    for _, idx in prediction.groupby('offline_percentile').groups.items():
        prediction.loc[idx, 'p_fdr_six'] = multipletests(prediction.loc[idx, 'p'], method='fdr_bh')[1]

    for name, table in [
        ('step05_marginal_effect_curves.csv', marginal_curves),
        ('step05_model_coefficients.csv', coef),
        ('step05_pairwise_comparisons.csv', comparisons),
        ('step05_model_samples.csv', samples),
        ('step05_online_marginal_slopes.csv', prediction),
    ]:
        path = OUT / name
        table.to_csv(path, index=False, encoding='utf-8-sig')
        print('Saved:', path)

    fig, ax = plt.subplots(figsize=(9, 5))
    plot = coef[(coef.model == 'B_adjusted') & (coef.term == 'online_c:offline_c')].copy()
    y = np.arange(len(plot))
    ax.errorbar(plot.coef, y, xerr=[plot.coef - plot.ci_low, plot.ci_high - plot.coef], fmt='o', capsize=4)
    ax.axvline(0, ls='--', lw=1, color='gray')
    ax.set_yticks(y, plot.outcome)
    ax.invert_yaxis()
    ax.set_xlabel('Online x Offline interaction coefficient (95% HC3 CI)')
    ax.set_title('Model B: adjusted interaction estimates')
    fig.tight_layout()
    fig.savefig(OUT / 'step05_interaction_coefficients.png', dpi=180)
    plt.close(fig)
    print('Saved:', OUT / 'step05_interaction_coefficients.png')


if __name__ == '__main__':
    d = load_data()
    print('可分析樣本數：', len(d))
    print('四象限人數：', d['quadrant_label'].value_counts().to_dict())
    fit_and_save(d)
