import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.spatial.distance import correlation
from statannotations.Annotator import Annotator
import statsmodels.api as sm
from statsmodels.formula.api import ols

# https://github.com/trevismd/statannotations
# seaborn >= 0.9,<0.12

# 二対比較検定
def t_test(ax, pairs, data: pd.DataFrame, x="variable", y="value",
           hue=None, order=None, test='t-test_ind'):
    annotator = Annotator(ax, pairs, x=x, y=y, hue=hue, order=order, data=data, sym="")
    annotator.configure(test=test, text_format='star', loc='inside')
    annotator.apply_and_annotate()


# 多重比較検定
def t_test_multi(ax, pairs, data: pd.DataFrame, x="variable", y="value",
                 test='t-test_ind', comparisons_correction="Bonferroni"):
    # comparisons_correction="BH", "Bonferroni"
    # Bonferroni
    # Holm-Bonferroni
    # Benjamini-Hochberg
    # Benjamini-Yekutieli

    annotator = Annotator(ax, pairs, x=x, y=y, data=data, sym="")
    annotator.configure(test=test, text_format='star', loc='inside')
    # annotator.apply_and_annotate()

    annotator.new_plot(ax=ax, x=x, y=y, data=data)
    annotator.configure(comparisons_correction=comparisons_correction, verbose=2)  # 補正
    test_results = annotator.apply_test().annotate()


# ANOVA
def anova(data: pd.DataFrame, variable='variable', value='value'):
    data.rename(columns={variable: 'variable', value: 'value'}, inplace=True)
    formula = 'value ~ variable'
    model = ols(formula=formula, data=data).fit()
    model.summary()
    anova = sm.stats.anova_lm(model)
    print('ANOVA table:')
    print(anova)

    ## Partial Eta_squared
    partial_eta_squared = anova['sum_sq'][0] / (anova['sum_sq'][0] + anova['sum_sq'][1])
    print(f"Partial Eta squared: {partial_eta_squared}")

    ## Eta_squared
    n_groups = len(anova.index)
    ss_treatment = anova['sum_sq'][0]
    ss_total = anova['sum_sq'][n_groups - 1]
    eta_squared = ss_treatment / ss_total
    print(f"Eta squared: {eta_squared}")


# 決定係数
# def r2_score(data1, data2):
#     sy2 = np.var(data2)  # データの分散
#     error = data1 - data2  # 誤差
#     syx2 = np.mean(error ** 2)  # 誤差の二乗平均
#     sr2 = sy2 - syx2  # 回帰の分散
#     r2 = sr2 / sy2  # 決定係数
#
#     return r2

def r2_score(actual, predicted):
    ssr = np.sum((actual - predicted) ** 2)  # 回帰変動
    sst = np.sum((actual - np.mean(actual)) ** 2)  # 全変動
    r2 = 1 - (ssr / sst)  # 決定係数

    return r2

# ブートストラップ
# data: nparray
# R: pick up number
def bootstrap(self, data, R):
    lenth = len(data[0])
    data_all = data.reshape(-1)
    results = np.zeros(R)

    for i in range(R):
        sample = np.random.choice(data_all, lenth)
        results[i] = np.mean(sample)

    return results


if __name__ == '__main__':
    # 　テストデータ作成
    Result_A = np.arange(5, 10, 0.1)
    Result_B = np.arange(8, 13, 0.1)
    Result_C = np.arange(10, 15, 0.1)

    # 実験条件の名前
    Condition = ['A', 'B', 'C']

    # データフレーム作成
    xlabel = 'Condition'
    ylabel = 'Error'

    df = pd.DataFrame({
        Condition[0]: Result_A,
        Condition[1]: Result_B,
        Condition[2]: Result_C,
    })
    print(df)
    df_melt = pd.melt(df)
    print(df_melt)
    df_melt.rename(columns={'variable': xlabel, 'value': ylabel}, inplace=True)
    print(df_melt)

    # プロット
    fig, sub = plt.subplots(1, 1, figsize=(10, 10), dpi=150)
    # 箱ひげ図
    ax = sns.boxplot(x=xlabel, y=ylabel, data=df_melt, ax=sub, sym="")

    # 検定に使うペアを指定
    pairs = [
        {Condition[0], Condition[1]},
        {Condition[0], Condition[2]},
        {Condition[1], Condition[2]},
    ]

    # 多重比較検定
    t_test_multi(ax, pairs, df_melt, x=xlabel, y=ylabel, test='t-test_ind', comparisons_correction="Bonferroni")
    # ANOVA
    anova(df_melt, variable=xlabel, value=ylabel)

    plt.show()
