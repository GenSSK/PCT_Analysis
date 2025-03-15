import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# from minepy import MINE
from scipy.spatial.distance import correlation
from statannotations.Annotator import Annotator
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
import pingouin as pg

import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import joblib

from mypackage.mystatistics import myHilbertTransform as HT
from mypackage.mystatistics import mySTFT as STFT
from mypackage.mystatistics import myhistogram as hist
from mypackage.mystatistics import myFilter as Filter
from mypackage.mystatistics import statistics as mystat
from mypackage import ParallelExecutor


if __name__ == '__main__':
    df = pd.read_csv("Avatar_Survey.csv")

    condition = ['ALL', 'ALONE', 'OTHER', 'NONE']

    df.sort_values(['name', 'condition'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    # print(df)

    # 数値を文字に置き換える辞書を作成
    replacement_dict = {1: 'ALL', 2: 'ALONE', 3: 'OTHER', 4: 'NONE'}
    df['condition'] = df['condition'].replace(replacement_dict)
    replacement_dict = {1: 'Dyad', 2: 'Triad', 3: 'Tetrad'}
    df['join'] = df['join'].replace(replacement_dict)

    # 数値を反転させる辞書を作成
    replacement_dict = {1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1}
    inverse_item = ['Q5', 'Q6', 'Q7', 'Q8', 'Q11', 'Q12', 'Q13', 'Q14']
    for item in inverse_item:
        df[item] = df[item].replace(replacement_dict)


    label = ['Social Presence', 'Co-presence', 'Perceived Attentional Engagement', 'Perceived Behavioral Interdependence']

    df[label[0]] = df[['Q'+str(_ + 1) for _ in range(20)]].mean(axis=1)
    df[label[1]] = df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8']].mean(axis=1)
    df[label[2]] = df[['Q9', 'Q10', 'Q11', 'Q12', 'Q13', 'Q14']].mean(axis=1)
    df[label[3]] = df[['Q15', 'Q16', 'Q17', 'Q18', 'Q19', 'Q20']].mean(axis=1)



    pairs = [
        [
            {condition[0], condition[1]},
            # {condition[0], condition[2]},
            {condition[0], condition[3]},
            {condition[1], condition[2]},
            # {condition[1], condition[3]},
            {condition[2], condition[3]},
        ],
        [
            {condition[0], condition[1]},
            {condition[0], condition[2]},
            {condition[0], condition[3]},
            {condition[1], condition[2]},
            # {condition[1], condition[3]},
            {condition[2], condition[3]},
        ],
        [
            {condition[0], condition[1]},
            # {condition[0], condition[2]},
            {condition[0], condition[3]},
            {condition[1], condition[2]},
            # {condition[1], condition[3]},
            {condition[2], condition[3]},
        ],
        [
            # {condition[0], condition[1]},
            # {condition[0], condition[2]},
            {condition[0], condition[3]},
            # {condition[1], condition[2]},
            # {condition[1], condition[3]},
            # {condition[2], condition[3]},
        ]
    ]

    # for i in range(len(condition)):
    #     print(f'count {condition[i]} {df[df['condition'] == condition[i]]['Q1'].count()}')

    pd.set_option('display.max_columns', 500)

    # シャピロウィルク検定
    for i in range(len(label)):
        for j in range(len(condition)):
            W, shapiro_p_value = stats.shapiro(df[df['condition'] == condition[j]][label[i]])
            print(f'{label[i]} - {condition[j]} Shapiro-Wilk test statistic: {W}, p-value: {shapiro_p_value}')

    # 反復測定ANOVA
    for i in range(len(label)):
        # mystat.anova_RM(df, subject='name', variable='condition', value=label[i])
        # aovrm = AnovaRM(df, label[i], 'name', within=['condition'])
        # res = aovrm.fit()
        # print(res)
        #
        # # 球面性検定（マウクリーの検定）
        # mauchly_result = sm.stats.anova_lm(res, typ=2)
        # print("Mauchly's test of sphericity:", mauchly_result)

        # 反復測定ANOVAの実行
        aov = pg.rm_anova(data=df, dv=label[i], within='condition', subject='name', detailed=True, correction=False)

        # 球面性検定
        sphericity_test = pg.sphericity(df, dv=label[i], within='condition', subject='name')

        # 結果の表示
        print(f'{label[i]}')
        print("反復測定ANOVAの結果:\n", aov)
        print("球面性検定の結果:\n", sphericity_test)

    # 修正して反復測定ANOVAの実行
    aov = pg.rm_anova(data=df, dv=label[0], within='condition', subject='name', detailed=True, correction=True)
    print(f'{label[0]}')
    print("反復測定ANOVAの結果:\n", aov)
    aov = pg.rm_anova(data=df, dv=label[1], within='condition', subject='name', detailed=True, correction=True)
    print(f'{label[1]}')
    print("反復測定ANOVAの結果:\n", aov)


    fig, axs = plt.subplots(1, 4, figsize=(18, 5))
    states_palette = states_palette = sns.color_palette("YlGnBu_r", n_colors=4, desat=1.0)
    for i, ax in enumerate(axs):
        print(f'{label[i]}')
        sns.boxplot(x='condition', y=label[i], data=df, ax=ax, palette=states_palette,
                    flierprops={"marker": "o", "markerfacecolor": "w"},)
        mystat.t_test_multi(ax, pairs[i], df, 'condition', label[i], test='t-test_paired', comparisons_correction="Bonferroni")

        tukey_result = pairwise_tukeyhsd(endog=df[label[i]], groups=df['condition'], alpha=0.05)
        print(tukey_result)

        ax.set_yticks(np.arange(1, 8, 1))
        ax.set_ylim(0, 11)
        ax.set_xlabel('')

    os.makedirs('fig/Survey', exist_ok=True)
    plt.savefig('fig/Survey/Survey_result_separate.pdf')

    df_csv = [
        df[df['condition'] == 'ALL'],
        df[df['condition'] == 'ALONE'],
        df[df['condition'] == 'OTHER'],
        df[df['condition'] == 'NONE'],
    ]


    for c in range(len(condition)):
        df_csv[c] = df_csv[c].loc[:, ['name', label[0], label[1], label[2], label[3]]]
        df_csv[c].rename(columns={label[0]: label[0]+'_'+condition[c],
                                  label[1]: label[1]+'_'+condition[c],
                                  label[2]: label[2]+'_'+condition[c],
                                  label[3]: label[3]+'_'+condition[c]}, inplace=True)
        df_csv[c].reset_index(drop=True, inplace=True)

    df_csv_con = pd.concat([_ for _ in df_csv], axis=1)

    # print(df_csv_con)

    # df_csv_con.to_csv('Avatar_Survey_result.csv', index=False)


    # pairs = [
    #     ((label[0], condition[0]), (label[0], condition[1])),
    #     ((label[0], condition[0]), (label[0], condition[2])),
    #     ((label[0], condition[0]), (label[0], condition[3])),
    #     ((label[0], condition[1]), (label[0], condition[2])),
    #     ((label[0], condition[1]), (label[0], condition[3])),
    #     ((label[0], condition[2]), (label[0], condition[3])),
    #
    #     ((label[1], condition[0]), (label[1], condition[1])),
    #     ((label[1], condition[0]), (label[1], condition[2])),
    #     ((label[1], condition[0]), (label[1], condition[3])),
    #     ((label[1], condition[1]), (label[1], condition[2])),
    #     ((label[1], condition[1]), (label[1], condition[3])),
    #     ((label[1], condition[2]), (label[1], condition[3])),
    #
    #     ((label[2], condition[0]), (label[2], condition[1])),
    #     ((label[2], condition[0]), (label[2], condition[2])),
    #     ((label[2], condition[0]), (label[2], condition[3])),
    #     ((label[2], condition[1]), (label[2], condition[2])),
    #     ((label[2], condition[1]), (label[2], condition[3])),
    #     ((label[2], condition[2]), (label[2], condition[3])),
    #
    #     ((label[3], condition[0]), (label[3], condition[1])),
    #     ((label[3], condition[0]), (label[3], condition[2])),
    #     ((label[3], condition[0]), (label[3], condition[3])),
    #     ((label[3], condition[1]), (label[3], condition[2])),
    #     ((label[3], condition[1]), (label[3], condition[3])),
    #     ((label[3], condition[2]), (label[3], condition[3])),
    #     ]
    #
    # df_melt = df.loc[:, ['name', 'condition', label[0], label[1], label[2], label[3]]]
    # df_melt = pd.melt(df, value_vars=label, id_vars=['name', 'condition'], value_name='score', var_name='SP')
    #
    # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # sns.boxplot(x='SP', y='score', hue='condition', data=df_melt, ax=ax, palette=states_palette,
    #             flierprops={"marker": "o", "markerfacecolor": "w"},)
    # mystat.t_test_multi(ax, pairs, df_melt, x='SP', y='score', hue='condition', test='t-test_ind', comparisons_correction="Bonferroni")
    # ax.set_yticks(np.arange(1, 8, 1))
    # ax.set_ylim(0, 8)
    # ax.set_xlabel('')
    # ax.legend(loc='lower left', ncol=4, framealpha=0.0,)

    plt.show()