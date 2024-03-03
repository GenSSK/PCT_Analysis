import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import CFO_Analysis_2gen.CFO_analysis as CFO_analysis
from scipy.spatial.distance import correlation
from statannotations.Annotator import Annotator
from statsmodels.formula.api import ols
import statsmodels.api as sm

from mypackage.mystatistics import myHilbertTransform as HT
from mypackage.mystatistics import mySTFT as STFT
from mypackage.mystatistics import myhistogram as hist
from mypackage.mystatistics import myFilter as Filter
from mypackage.mystatistics import statistics as mystat
from mypackage import ParallelExecutor


def plot_scatter(x, y, color, **kwargs):
    ax = plt.gca()
    ax.scatter(x, y, c=color, **kwargs)


class combine:
    def __init__(self, dyad_npz, triad_npz, tetrad_npz):
        self.dyad_cfo: CFO_analysis.CFO = CFO_analysis.CFO(dyad_npz, 'dyad')
        self.triad_cfo: CFO_analysis.CFO = CFO_analysis.CFO(triad_npz, 'triad')
        self.tetrad_cfo: CFO_analysis.CFO = CFO_analysis.CFO(tetrad_npz, 'tetrad')

        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.default'] = 'regular'
        plt.rcParams['xtick.top'] = 'True'
        plt.rcParams['ytick.right'] = 'True'
        # plt.rcParams['axes.grid'] = 'True'
        plt.rcParams['xtick.direction'] = 'in'  # x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'  # y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['xtick.major.width'] = 1.0  # x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 1.0  # y軸主目盛り線の線幅
        plt.rcParams['font.size'] = 12  # フォントの大きさ
        plt.rcParams['axes.linewidth'] = 0.5  # 軸の線幅edge linewidth。囲みの太さ

        plt.rcParams["legend.fancybox"] = False  # 丸角
        plt.rcParams["legend.framealpha"] = 0  # 透明度の指定、0で塗りつぶしなし
        plt.rcParams["legend.edgecolor"] = 'black'  # edgeの色を変更
        plt.rcParams["legend.handlelength"] = 2  # 凡例の線の長さを調節
        plt.rcParams["legend.labelspacing"] = 0.1  # 垂直方向（縦）の距離の各凡例の距離
        plt.rcParams["legend.handletextpad"] = .4  # 凡例の線と文字の距離の長さ

        plt.rcParams["legend.markerscale"] = 2  # 点がある場合のmarker scale
        plt.rcParams['axes.xmargin'] = '0'  # '.05'
        plt.rcParams['axes.ymargin'] = '0'
        plt.rcParams['savefig.facecolor'] = 'None'
        plt.rcParams['savefig.edgecolor'] = 'None'
        # plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['pdf.fonttype'] = 42  # PDFにフォントを埋め込むためのパラメータ

    def summation_cfo(self, graph=False, mode='no_abs'):
        dyad_pp, dyad_rp, dyad_pf, dyad_rf = self.dyad_cfo.summation_cfo_3sec(mode)
        triad_pp, triad_rp, triad_pf, triad_rf = self.triad_cfo.summation_cfo_3sec(mode)
        tetrad_pp, tetrad_rp, tetrad_pf, tetrad_rf = self.tetrad_cfo.summation_cfo_3sec(mode)
        summation_3sec_datas = [
            [dyad_pp, triad_pp, tetrad_pp],
            [dyad_rp, triad_rp, tetrad_rp],
            [dyad_pf, triad_pf, tetrad_pf],
            [dyad_rf, triad_rf, tetrad_rf],
        ]

        if graph == True:
            # sns.set()
            # sns.set_style('whitegrid')
            # sns.set_palette('Set3')

            size = ['Dyad', 'Triad', 'Tetrad']

            types = ['Summation Pitch PCFO (Avg)',
                     'Summation Roll PCFO (Avg)',
                     'Summation Pitch FCFO (Avg)',
                     'Summation Roll FCFO (Avg)']
            ranges = [0.05, 0.05, 0.3, 0.3]
            ticks = [[0.0, 0.1, 0.3, 0.3],
                     [0.0, 0.1, 0.2, 0.3],
                     [0.0, 1.0, 2.0, 3.0],
                     [0.0, 0.1, 2.0, 3.0],
                     ]

            if mode == 'b_abs':
                types = ['Before abs.\nSummation Pitch PCFO (Avg)',
                         'Before abs.\nSummation Roll PCFO (Avg)',
                         'Before abs.\nSummation Pitch FCFO (Avg)',
                         'Before abs.\nSummation Roll FCFO (Avg)']
                ranges = [0.3, 0.3, 4.0, 4.0]
                ticks = [[0.0, 0.1, 0.2, 0.3],
                         [0.0, 0.1, 0.2, 0.3],
                         [0.0, 2.0, 4.0],
                         [0.0, 2.0, 4.0],
                         ]

            if mode == 'a_abs':
                types = ['After abs.\nSummation Pitch PCFO (Avg)',
                         'After abs.\nSummation Roll PCFO (Avg)',
                         'After abs.\nSummation Pitch FCFO (Avg)',
                         'After abs.\nSummation Roll FCFO (Avg)']
                ranges = [0.3, 0.3, 1.2, 1.2]
                ticks = [[0.0, 0.1, 0.2, 0.3],
                         [0.0, 0.1, 0.2, 0.3],
                         [0.0, 0.6, 1.2],
                         [0.0, 0.6, 1.2],
                         ]

            fig = plt.figure(figsize=(10, 7), dpi=150)

            plot = [
                fig.add_subplot(2, 2, 1),
                fig.add_subplot(2, 2, 2),
                fig.add_subplot(2, 2, 3),
                fig.add_subplot(2, 2, 4),
            ]

            xlabel = 'Group size'
            ylabel = 'CFO'

            for j in range(len(plot)):
                dfpp = []
                dfpp_melt = []
                for i in range(len(dyad_pp)):
                    dfpp.append(pd.DataFrame({
                        'Dyad': summation_3sec_datas[j][0][i],
                        'Triad': summation_3sec_datas[j][1][i],
                        'Tetrad': summation_3sec_datas[j][2][i],
                    })
                    )

                    dfpp_melt.append(pd.melt(dfpp[i]))
                    dfpp_melt[i]['Group'] = 'Group' + str(i + 1)

                df = pd.concat([i for i in dfpp_melt], axis=0)
                df.rename(columns={'variable': xlabel, 'value': ylabel}, inplace=True)

                ax = sns.boxplot(data=df, x=xlabel, y=ylabel, ax=plot[j], sym="")
                # sns.stripplot(data=df, x=xlabel, y=ylabel, hue='Group', dodge=True,
                #               jitter=0.2, color='black', palette='Paired', ax=plot[j])

                plot[j].legend_ = None
                plot[j].set_ylabel(types[j])
                plot[j].set_yticks(ticks[j])
                # plot[j].axes.xaxis.set_visible(False)
                plot[j].set_ylim(-ranges[j], ranges[j])
                if mode == 'b_abs' or mode == 'a_abs':
                    plot[j].set_ylim(0, ranges[j])

                pairs = [
                    {size[0], size[1]},
                    {size[0], size[2]},
                    {size[1], size[2]},
                ]

                mystat.t_test_multi(ax, pairs, df, x=xlabel, y=ylabel, test='t-test_ind',)
                mystat.anova(df, variable=xlabel, value=ylabel)

            if mode == 'no_abs':
                os.makedirs('fig/CFO/Summation/NoABS/Comparison/', exist_ok=True)
                plt.savefig('fig/CFO/Summation/NoABS/Comparison/SummationCFO_NoABS_3sec_comparison.png')
            elif mode == 'b_abs':
                os.makedirs('fig/CFO/Summation/BeforeABS/Comparison/', exist_ok=True)
                plt.savefig('fig/CFO/Summation/BeforeABS/Comparison/SummationCFO_BeforeABS_3sec_comparison.png')
            elif mode == 'a_abs':
                os.makedirs('fig/CFO/Summation/AfterABS/Comparison/', exist_ok=True)
                plt.savefig('fig/CFO/Summation/AfterABS/Comparison/SummationCFO_AfterABS_3sec_comparison.png')
            plt.show()

        return summation_3sec_datas

    def summation_cfo_combine(self, graph=False, mode='no_abs'):
        dyad_p, dyad_f = self.dyad_cfo.summation_cfo_3sec_combine(mode)
        triad_p, triad_f = self.triad_cfo.summation_cfo_3sec_combine(mode)
        tetrad_p, tetrad_f = self.tetrad_cfo.summation_cfo_3sec_combine(mode)
        summation_3sec_datas = [
            [dyad_p, triad_p, tetrad_p],
            [dyad_f, triad_f, tetrad_f],
        ]

        if graph == True:
            # sns.set()
            # sns.set_style('whitegrid')
            # sns.set_palette('Set3')

            size = ['Dyad', 'Triad', 'Tetrad']

            types = ['Summation PCFO (Avg)',
                     'Summation FCFO (Avg)']
            ranges = [0.1, 0.2]
            ticks = [[-0.1, 0.0, 0.1],
                     [-0.2, 0.0, 0.2],
                     ]

            if mode == 'b_abs':
                types = ['Before abs. Summation PCFO (Avg)',
                         'Before abs. Summation FCFO (Avg)']
                ranges = [0.4, 3.0]
                ticks = [[0.0, 0.2, 0.4],
                         [0.0, 1.0, 2.0, 3.0],
                         ]

            if mode == 'a_abs':
                types = ['After abs. Summation PCFO (Avg)',
                         'After abs. Summation FCFO (Avg)']
                ranges = [0.3, 1.0]
                ticks = [[0.0, 0.1, 0.2, 0.3],
                         [0.0, 0.5, 1.0],
                         ]

            fig = plt.figure(figsize=(10, 6), dpi=150)

            plot = [
                fig.add_subplot(1, 2, 1),
                fig.add_subplot(1, 2, 2),
            ]

            xlabel = 'Group size'
            ylabel = 'CFO'

            for j in range(len(plot)):
                dfpp = []
                dfpp_melt = []
                for i in range(len(dyad_p)):
                    dfpp.append(pd.DataFrame({
                        'Dyad': summation_3sec_datas[j][0][i],
                        'Triad': summation_3sec_datas[j][1][i],
                        'Tetrad': summation_3sec_datas[j][2][i],
                    })
                    )

                    dfpp_melt.append(pd.melt(dfpp[i]))
                    dfpp_melt[i]['Group'] = 'Group' + str(i + 1)

                df = pd.concat([i for i in dfpp_melt], axis=0)
                df.rename(columns={'variable': xlabel, 'value': ylabel}, inplace=True)

                ax = sns.boxplot(data=df, x=xlabel, y=ylabel, ax=plot[j], sym="")
                # sns.stripplot(data=df, x=xlabel, y=ylabel, hue='Group', dodge=True,
                #               jitter=0.2, color='black', palette='Paired', ax=plot[j])

                plot[j].legend_ = None
                plot[j].set_ylabel(types[j])
                plot[j].set_yticks(ticks[j])
                # plot[j].axes.xaxis.set_visible(False)
                plot[j].set_ylim(-ranges[j], ranges[j])
                if mode == 'b_abs' or mode == 'a_abs':
                    plot[j].set_ylim(0, ranges[j])

                pairs = [
                    {size[0], size[1]},
                    {size[0], size[2]},
                    {size[1], size[2]},
                ]

                mystat.t_test_multi(ax, pairs, df, x=xlabel, y=ylabel, test='t-test_ind',)
                mystat.anova(df, variable=xlabel, value=ylabel)

            if mode == 'no_abs':
                os.makedirs('fig/CFO/Summation/NoABS/Comparison/Combine/', exist_ok=True)
                plt.savefig('fig/CFO/Summation/NoABS/Comparison/Combine/SummationCFO_NoABS_3sec_combine_comparison.png')
            elif mode == 'b_abs':
                os.makedirs('fig/CFO/Summation/BeforeABS/Comparison/Combine/', exist_ok=True)
                plt.savefig('fig/CFO/Summation/BeforeABS/Comparison/Combine/SummationCFO_BeforeABS_3sec_combine_comparison.png')
            elif mode == 'a_abs':
                os.makedirs('fig/CFO/Summation/AfterABS/Comparison/Combine/', exist_ok=True)
                plt.savefig('fig/CFO/Summation/AfterABS/Comparison/Combine/SummationCFO_AfterABS_3sec_combine_comparison.png')
            plt.show()

        return summation_3sec_datas

    def subtraction_cfo(self, graph=False):
        dyad_pp, dyad_rp, dyad_pf, dyad_rf = self.dyad_cfo.subtraction_cfo_3sec()
        triad_pp, triad_rp, triad_pf, triad_rf = self.triad_cfo.subtraction_cfo_3sec()
        tetrad_pp, tetrad_rp, tetrad_pf, tetrad_rf = self.tetrad_cfo.subtraction_cfo_3sec()
        subtraction_3sec_datas = [
            [dyad_pp, triad_pp, tetrad_pp],
            [dyad_rp, triad_rp, tetrad_rp],
            [dyad_pf, triad_pf, tetrad_pf],
            [dyad_pf, triad_pf, tetrad_pf],
            [dyad_rf, triad_rf, tetrad_rf],
        ]

        if graph == True:
            # sns.set()
            # sns.set_style('whitegrid')
            # sns.set_palette('Set3')

            types = ['Subtraction Pitch PCFO (Avg)', 'Subtraction Roll PCFO (Avg)', 'Subtraction Pitch FCFO (Avg)',
                     'Subtraction Roll FCFO (Avg)']
            ranges = [0.5, 0.5, 10.0, 10.0]

            size = ['Dyad', 'Triad', 'Tetrad']

            fig = plt.figure(figsize=(10, 7), dpi=150)

            plot = [
                fig.add_subplot(2, 2, 1),
                fig.add_subplot(2, 2, 2),
                fig.add_subplot(2, 2, 3),
                fig.add_subplot(2, 2, 4),
            ]

            xlabel = 'Group size'
            ylabel = 'CFO'

            for j in range(len(plot)):
                dfpp = []
                dfpp_melt = []
                for i in range(len(dyad_pp)):
                    dfpp.append(pd.DataFrame({
                        'Dyad': subtraction_3sec_datas[j][0][i],
                        'Triad': subtraction_3sec_datas[j][1][i],
                        'Tetrad': subtraction_3sec_datas[j][2][i],
                    })
                    )

                    dfpp_melt.append(pd.melt(dfpp[i]))
                    dfpp_melt[i]['Group'] = 'Group' + str(i + 1)

                    print(dfpp_melt[i])

                df = pd.concat([i for i in dfpp_melt], axis=0)
                df.rename(columns={'variable': xlabel, 'value': ylabel}, inplace=True)

                ax = sns.boxplot(data=df, x=xlabel, y=ylabel, ax=plot[j], sym="")
                # sns.stripplot(data=df, x=xlabel, y=ylabel, ax=plot[j], hue='Group', dodge=True,
                #               jitter=0.2, color='black', palette='Paired')

                plot[j].legend_ = None
                plot[j].set_ylabel(types[j])
                # plot[j].axes.xaxis.set_visible(False)
                plot[j].set_ylim(0.0, ranges[j])

                pairs = [
                    {size[0], size[1]},
                    {size[0], size[2]},
                    {size[1], size[2]},
                ]

                mystat.t_test_multi(ax, pairs, df, x=xlabel, y=ylabel, test='t-test_ind',)
                mystat.anova(df, variable=xlabel, value=ylabel)

            os.makedirs('fig/CFO/Subtraction/Comparison', exist_ok=True)
            plt.savefig('fig/CFO/Subtraction/Comparison/SubtractionCFO_3sec_comparison.png')
            plt.show()

        return subtraction_3sec_datas

    def subtraction_cfo_combine(self, graph=False):
        dyad_p, dyad_f = self.dyad_cfo.subtraction_cfo_3sec_combine()
        triad_p, triad_f = self.triad_cfo.subtraction_cfo_3sec_combine()
        tetrad_p, tetrad_f = self.tetrad_cfo.subtraction_cfo_3sec_combine()
        subtraction_3sec_datas = [
            [dyad_p, triad_p, tetrad_p],
            [dyad_f, triad_f, tetrad_f],
        ]

        if graph == True:
            # sns.set()
            # sns.set_style('whitegrid')
            # sns.set_palette('Set3')

            types = ['Subtraction PCFO (Avg)', 'Subtraction FCFO (Avg)']
            ranges = [0.6, 10.0]
            ticks = [[0.0, 0.2, 0.4, 0.6],
                     [0.0, 5.0, 10.0]]

            size = ['Dyad', 'Triad', 'Tetrad']

            fig = plt.figure(figsize=(10, 6), dpi=150)

            plot = [
                fig.add_subplot(1, 2, 1),
                fig.add_subplot(1, 2, 2),
            ]

            xlabel = 'Group size'
            ylabel = 'CFO'

            for j in range(len(plot)):
                dfpp = []
                dfpp_melt = []
                for i in range(len(dyad_p)):
                    dfpp.append(pd.DataFrame({
                        'Dyad': subtraction_3sec_datas[j][0][i],
                        'Triad': subtraction_3sec_datas[j][1][i],
                        'Tetrad': subtraction_3sec_datas[j][2][i],
                    })
                    )

                    dfpp_melt.append(pd.melt(dfpp[i]))
                    dfpp_melt[i]['Group'] = 'Group' + str(i + 1)

                    print(dfpp_melt[i])

                df = pd.concat([i for i in dfpp_melt], axis=0)
                df.rename(columns={'variable': xlabel, 'value': ylabel}, inplace=True)

                ax = sns.boxplot(data=df, x=xlabel, y=ylabel, ax=plot[j], sym="")
                # sns.stripplot(data=df, x=xlabel, y=ylabel, ax=plot[j], hue='Group', dodge=True,
                #               jitter=0.2, color='black', palette='Paired')

                plot[j].legend_ = None
                plot[j].set_ylabel(types[j])
                # plot[j].axes.xaxis.set_visible(False)
                plot[j].set_ylim(0.0, ranges[j])

                pairs = [
                    {size[0], size[1]},
                    {size[0], size[2]},
                    {size[1], size[2]},
                ]

                mystat.t_test_multi(ax, pairs, df, x=xlabel, y=ylabel, test='t-test_ind',)
                mystat.anova(df, variable=xlabel, value=ylabel)

            os.makedirs('fig/CFO/Subtraction/Comparison/Combine', exist_ok=True)
            plt.savefig('fig/CFO/Subtraction/Comparison/Combine/SubtractionCFO_3sec_combine_comparison.png')
            plt.show()

        return subtraction_3sec_datas

    def performance_show(self):
        error_period_dyad, spend_period_dyad = self.dyad_cfo.period_performance_cooperation()
        error_period_triad, spend_period_triad = self.triad_cfo.period_performance_cooperation()
        error_period_tetrad, spend_period_tetrad = self.tetrad_cfo.period_performance_cooperation()

        error_periode = [error_period_dyad, error_period_triad, error_period_tetrad]
        spend_periode = [spend_period_dyad, spend_period_triad, spend_period_tetrad]
        performance = [error_periode, spend_periode]

        types = ['dyad', 'triad', 'tetrad']
        performance_type = ['Cooperative Error (m)', 'Cooperative Time (s)']
        labels = ['Error', 'Time']
        ylim = [[-0.02, 0.02], [-1.0, 1.0]]

        for i, p in enumerate(performance):
            fig, axs = plt.subplots(3, 1, figsize=(10, 7), dpi=150, sharex=True, sharey=True)
            for j, ax in enumerate(axs):
                ax.title.set_text(types[j])
                for k, p_ in enumerate(p[j]):
                    ax.plot(np.arange(1, len(p_) + 1, 1), p_, label='Group' + str(k + 1))
                    ax.scatter(np.arange(1, len(p_) + 1, 1), p_, s=10, marker='x')
                ax.legend(ncol=10, columnspacing=1)

            axs[1].set_ylabel(performance_type[i])
            axs[2].set_xlabel('Period')
            axs[2].set_xlim(0, len(error_period_dyad[0]) + 1)
            axs[2].set_xticks(np.arange(1, len(error_period_dyad[0]) + 1, 1))
            axs[2].set_ylim(ylim[i][0], ylim[i][1])
            os.makedirs('fig/Performance/TimeSeries/', exist_ok=True)
            plt.savefig('fig/Performance/TimeSeries/Performance_TimeSeries_Cooperation_' + labels[i] + '.png')
        plt.show()

    def performance_comparison(self, mode='h-m'):
        if mode == 'h-m':
            error_period_dyad, spend_period_dyad = self.dyad_cfo.period_performance_cooperation()
            error_period_triad, spend_period_triad = self.triad_cfo.period_performance_cooperation()
            error_period_tetrad, spend_period_tetrad = self.tetrad_cfo.period_performance_cooperation()
        elif mode == 'h-h':
            error_period_dyad, spend_period_dyad = self.dyad_cfo.period_performance(mode='H-H')
            error_period_triad, spend_period_triad = self.triad_cfo.period_performance(mode='H-H')
            error_period_tetrad, spend_period_tetrad = self.tetrad_cfo.period_performance(mode='H-H')
        elif mode == 'm-m':
            error_period_dyad, spend_period_dyad = self.dyad_cfo.period_performance(mode='M-M')
            error_period_triad, spend_period_triad = self.triad_cfo.period_performance(mode='M-M')
            error_period_tetrad, spend_period_tetrad = self.tetrad_cfo.period_performance(mode='M-M')

        type = ['Dyad', 'Triad', 'Tetrad']
        xlabel = 'Group size'
        ylabel = ['Error', 'Time']
        performance_label = ['Error (m)', 'Time (s)']

        ep = []
        ep_melt = []
        for i in range(len(error_period_dyad)):
            ep.append(pd.DataFrame({
                type[0]: error_period_dyad[i],
                type[1]: error_period_triad[i],
                type[2]: error_period_tetrad[i],
            })
            )

            ep_melt.append(pd.melt(ep[i]))
            ep_melt[i]['Group'] = 'Group' + str(i + 1)

        df_ep = pd.concat([i for i in ep_melt], axis=0)
        df_ep.rename(columns={'variable': xlabel, 'value': ylabel[0]}, inplace=True)

        sp = []
        sp_melt = []
        for i in range(len(spend_period_dyad)):
            sp.append(pd.DataFrame({
                'Dyad': spend_period_dyad[i],
                'Triad': spend_period_triad[i],
                'Tetrad': spend_period_tetrad[i],
            })
            )

            sp_melt.append(pd.melt(sp[i]))
            sp_melt[i]['Group'] = 'Group' + str(i + 1)

        df_sp = pd.concat([i for i in sp_melt], axis=0)
        df_sp.rename(columns={'variable': xlabel, 'value': ylabel[1]}, inplace=True)

        performance = [df_ep, df_sp]
        ylim = [[[-0.02, 0.03], [-0.2, 0.8]], #h-m
                [[0.01, 0.06], [1.4, 3.0]], #h-h
                [[0.01, 0.06], [1.4, 3.0]]] #m-m

        ytixs = [[(-0.02, 0.01, 0.00, 0.01, 0.02, 0.03), (-0.2, 0.0, 0.2, 0.4, 0.6, 0.8)],
                 [(0.2, 0.3, 0.4), (1.0, 2.0, 3.0, 4.0, 5.0)],
                 [(-1.0, 0.0, 1.0, 2.0, 3.0), (-1.0, 0.0, 1.0, 2.0, 3.0)]]

        pairs = [
            {type[0], type[1]},
            {type[0], type[2]},
            {type[1], type[2]},
        ]

        for i, p in enumerate(performance):
            fig, ax = plt.subplots(1, 1, figsize=(10, 7), dpi=150)
            g = sns.boxplot(data=p, x=xlabel, y=ylabel[i], sym="", ax=ax)

            base_dir = 'fig/Performance/Comparison/'
            os.makedirs(base_dir, exist_ok=True)
            if mode == 'h-m':
                ax.set_title('Cooperation')
                ax.set_ylabel('Cooperative ' + performance_label[i])
                ax.set_ylim(ylim[0][i][0], ylim[0][i][1])
                plt.savefig(base_dir + 'Performance_Cooperation_' + ylabel[i] + '_Comparison.png')
            elif mode == 'h-h':
                ax.set_title('Human-Human')
                ax.set_ylabel(performance_label[i])
                ax.set_ylim(ylim[1][i][0], ylim[1][i][1])
                plt.savefig(base_dir + 'Performance_Human-Human_' + ylabel[i] + '_Comparison.png')
            elif mode == 'm-m':
                ax.set_title('Model-Model')
                ax.set_ylabel(performance_label[i])
                ax.set_ylim(ylim[2][i][0], ylim[2][i][1])
                plt.savefig(base_dir + 'Performance_Model-Model_' + ylabel[i] + '_Comparison.png')

            mystat.t_test_multi(ax=g, pairs=pairs, data=p, x=xlabel, y=ylabel[i], test="t-test_ind")
            mystat.anova(data=p, variable=xlabel, value=ylabel[i])

        plt.show()

    def performance_relation(self):
        error_period_dyad, spend_period_dyad = self.dyad_cfo.period_performance_cooperation()
        error_period_triad, spend_period_triad = self.triad_cfo.period_performance_cooperation()
        error_period_tetrad, spend_period_tetrad = self.tetrad_cfo.period_performance_cooperation()

        error_period_dyad = error_period_dyad.reshape(-1)
        error_period_triad = error_period_triad.reshape(-1)
        error_period_tetrad = error_period_tetrad.reshape(-1)
        spend_period_dyad = spend_period_dyad.reshape(-1)
        spend_period_triad = spend_period_triad.reshape(-1)
        spend_period_tetrad = spend_period_tetrad.reshape(-1)

        error_period = np.concatenate((error_period_dyad, error_period_triad, error_period_tetrad))
        spend_period = np.concatenate((spend_period_dyad, spend_period_triad, spend_period_tetrad))

        print(1 - correlation(error_period, spend_period))

        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=150)

        s = 10

        ax.scatter(error_period_dyad, spend_period_dyad, label='Dyad', color='blue', s=s)
        ax.scatter(error_period_triad, spend_period_triad, label='Triad', color='red', s=s)
        ax.scatter(error_period_tetrad, spend_period_tetrad, label='Tetrad', color='green', s=s)

        r2_dyad = np.corrcoef(error_period_dyad, spend_period_dyad)
        r2_triad = np.corrcoef(error_period_triad, spend_period_triad)
        r2_tetrad = np.corrcoef(error_period_tetrad, spend_period_tetrad)
        plt.text
        ax.text(0.01, 0.12, '$Dyad: r = {:.2f}$'.format(r2_dyad[0][1]), horizontalalignment='left',
                transform=ax.transAxes, fontsize="medium")
        ax.text(0.01, 0.07, '$Triad: r = {:.2f}$'.format(r2_triad[0][1]), horizontalalignment='left',
                transform=ax.transAxes, fontsize="medium")
        ax.text(0.01, 0.02, '$Tetrad: r = {:.2f}$'.format(r2_tetrad[0][1]), horizontalalignment='left',
                transform=ax.transAxes, fontsize="medium")

        ax.set_xlim(-0.03, 0.02)
        ax.set_ylim(-0.5, 1.0)
        ax.set_xlabel('Cooperative Error (m)')
        ax.set_ylabel('Cooperative Time (sec)')
        ax.legend()
        os.makedirs('fig/Performance/', exist_ok=True)
        plt.savefig('fig/Performance/performance_relation.png')
        plt.show()

    def performance_hist(self):
        error_period_dyad, spend_period_dyad = self.dyad_cfo.period_performance_cooperation()
        error_period_triad, spend_period_triad = self.triad_cfo.period_performance_cooperation()
        error_period_tetrad, spend_period_tetrad = self.tetrad_cfo.period_performance_cooperation()

        error_period_dyad = error_period_dyad.reshape(-1)
        error_period_triad = error_period_triad.reshape(-1)
        error_period_tetrad = error_period_tetrad.reshape(-1)
        spend_period_dyad = spend_period_dyad.reshape(-1)
        spend_period_triad = spend_period_triad.reshape(-1)
        spend_period_tetrad = spend_period_tetrad.reshape(-1)

        error = pd.DataFrame({
            'Dyad': error_period_dyad,
            'Triad': error_period_triad,
            'Tetrad': error_period_tetrad,
        }
        )

        spend = pd.DataFrame({
            'Dyad': spend_period_dyad,
            'Triad': spend_period_triad,
            'Tetrad': spend_period_tetrad,
        }
        )

        performence = [error, spend]
        label = ['Cooperative Error (m)', 'Cooperative Time (s)']
        outlabel = ['Error', 'Time']

        xlim = [
            (-0.02, 0.02),
            (-1, 1),
        ]
        xtics = [
            (np.arange(-5, 5, 1)),
            (np.arange(-1, 1, 1)),
        ]

        mode = ['Dyad', 'Triad', 'Tetrad']
        kwargs = dict(hist_kws={'alpha': .6}, kde_kws={'linewidth': 2})

        for i in range(len(performence)):
            fig, ax = plt.subplots(3, 1, figsize=(10, 10), dpi=150, sharey=True, sharex=True)
            ax[1].set_ylabel('Counts')
            plt.xlabel(label[i])
            for j in range(3):
                # sns.histplot(performence[i][mode[j]], kde=True, bins=10, label="Counts", ax=ax[j])
                # y, edge = hist.get_histogram_normalize(performence[i][mode[j]], bin=10)
                step = 0.1 * xlim[i][1]
                freq, edge = hist.calc_frequency(performence[i][mode[j]], np.arange(xlim[i][0], xlim[i][1], step))
                ax[j].bar(edge, freq, width=step, align='edge', label="Counts", color="gray", edgecolor="black", linewidth=0.2)
                ax[j].set_ylim(0, 30)
                ax[j].set_title('Group size: ' + mode[j])
                ax[j].set_xlim(xlim[i])
            plt.tight_layout()
            os.makedirs('fig/Performance/Hist/', exist_ok=True)
            plt.savefig('fig/Performance/Hist/CooperativePerformance_' + str(outlabel[i]) + '_hist.png')
        plt.show()

    def performance_bootstrap(self):
        R = 100000
        error_period_dyad, spend_period_dyad = self.dyad_cfo.period_performance_cooperation()
        error_period_triad, spend_period_triad = self.triad_cfo.period_performance_cooperation()
        error_period_tetrad, spend_period_tetrad = self.tetrad_cfo.period_performance_cooperation()

        data = [
            error_period_dyad,
            error_period_triad,
            error_period_tetrad,
            spend_period_dyad,
            spend_period_triad,
            spend_period_tetrad,
        ]

        bs_data = []

        for i in range(len(data)):
            bs_data.append(combine.bootstrap(self, data[i], R))

        reshape_data = [data[_].reshape(-1) for _ in range(6)]
        bs_reshape_data = [bs_data[_].reshape(-1) for _ in range(6)]

        error = pd.DataFrame({
            'Dyad': reshape_data[0],
            'Triad': reshape_data[1],
            'Tetrad': reshape_data[2],
        }
        )

        spend = pd.DataFrame({
            'Dyad': reshape_data[3],
            'Triad': reshape_data[4],
            'Tetrad': reshape_data[5],
        }
        )
        performance = [error, spend]

        bs_error = pd.DataFrame({
            'Dyad': bs_reshape_data[0],
            'Triad': bs_reshape_data[1],
            'Tetrad': bs_reshape_data[2],
        }
        )

        bs_spend = pd.DataFrame({
            'Dyad': bs_reshape_data[3],
            'Triad': bs_reshape_data[4],
            'Tetrad': bs_reshape_data[5],
        }
        )
        bs_performance = [bs_error, bs_spend]
        label = ['Error', 'Spend']

        xlim = [
            (-0.1, 0.1),
            (-1, 1),
        ]
        xtics = [
            (np.arange(-5, 5, 1)),
            (np.arange(-1, 1, 1)),
        ]

        mode = ['Dyad', 'Triad', 'Tetrad']
        kwargs = dict(hist_kws={'alpha': .6}, kde_kws={'linewidth': 2})

        fig, ax_box = plt.subplots(1, 2, figsize=(10, 7), dpi=150)
        for i in range(2):
            fig, ax = plt.subplots(3, 1, figsize=(10, 7), dpi=150)
            for j in range(3):
                sns.histplot(performance[i][mode[j]], kde=True, bins=10, label="Counts", ax=ax[j], stat='probability')
                sns.histplot(bs_performance[i][mode[j]], kde=True, bins=10, label="Counts", ax=ax[j],
                             stat='probability')

                ax[j].set_title(mode[j])
                ax[j].set_xlim(xlim[i])

            plt.title(label[i])
            plt.tight_layout()
            plt.savefig('fig/performance_bootstrap_hist_' + str(label[i]) + '.png')

            df = pd.melt(performance[0])
            bs_df = pd.melt(bs_performance[0])
            # sns.boxplot(x="variable", y="value", data=df, ax=ax_box[i], sym="")
            sns.boxplot(x="variable", y="value", data=bs_df, ax=ax_box[i], sym="")

            plt.tight_layout()
        plt.savefig('fig/performance_bootstrap.png')
        plt.show()

    def bootstrap(self, data, R):
        lenth = len(data[0])
        data_all = data.reshape(-1)
        results = np.zeros(R)

        for i in range(R):
            sample = np.random.choice(data_all, lenth)
            results[i] = np.mean(sample)

        return results

    def summation_ave_cfo(self, graph=False, mode='no_abs'):
        dyad_p, dyad_f, dyad_pa, dyad_fa = self.dyad_cfo.summation_ave_cfo_3sec(mode)
        triad_p, triad_f, triad_pa, triad_fa = self.triad_cfo.summation_ave_cfo_3sec(mode)
        tetrad_p, tetrad_f, tetrad_pa, tetrad_fa = self.tetrad_cfo.summation_ave_cfo_3sec(mode)
        summation_3sec_datas = [
            [dyad_p, triad_p, tetrad_p],
            [dyad_f, triad_f, tetrad_f],
            [dyad_pa, triad_pa, tetrad_pa],
            [dyad_fa, triad_fa, tetrad_fa],
        ]

        if graph == True:
            sns.set()
            # sns.set_style('whitegrid')
            sns.set_palette('Set3')

            types = ['Summation PCFO (Avg)',
                     'Summation FCFO (Avg)',
                     'Summation abs. PCFO (Avg)',
                     'Summation abs. FCFO (Avg)']
            ranges = [0.06, 0.5, 0.3, 4.0]

            if mode == 'b_abs':
                types = ['Before abs.\nSummation PCFO (Avg)',
                         'Before abs.\nSummation FCFO (Avg)',
                         'Before abs.\nSummation abs. PCFO (Avg)',
                         'Before abs.\nSummation abs. FCFO (Avg)']
                ranges = [0.25, 3.0, 0.25, 3.0]

            if mode == 'a_abs':
                types = ['After abs.\nSummation PCFO (Avg)',
                         'After abs.\nSummation FCFO (Avg)',
                         'After abs.\nSummation abs. PCFO (Avg)',
                         'After abs.\nSummation abs. FCFO (Avg)']
                ranges = [0.2, 1.0, 0.25, 3.0]

            fig = plt.figure(figsize=(10, 7), dpi=150)

            plot = [
                fig.add_subplot(2, 2, 1),
                fig.add_subplot(2, 2, 2),
                fig.add_subplot(2, 2, 3),
                fig.add_subplot(2, 2, 4),
            ]

            for j in range(len(plot)):
                dfpp = []
                dfpp_melt = []
                for i in range(len(dyad_p)):
                    dfpp.append(pd.DataFrame({
                        'Dyad': summation_3sec_datas[j][0][i],
                        'Triad': summation_3sec_datas[j][1][i],
                        'Tetrad': summation_3sec_datas[j][2][i],
                    })
                    )

                    dfpp_melt.append(pd.melt(dfpp[i]))
                    dfpp_melt[i]['Group'] = 'Group' + str(i + 1)

                df = pd.concat([i for i in dfpp_melt], axis=0)

                sns.boxplot(x="variable", y="value", data=df, ax=plot[j], sym="")
                sns.stripplot(x='variable', y='value', data=df, hue='Group', dodge=True,
                              jitter=0.2, color='black', palette='Paired', ax=plot[j])

                plot[j].legend_ = None
                plot[j].set_ylabel(types[j])
                # plot[j].axes.xaxis.set_visible(False)
                plot[j].set_ylim(-ranges[j], ranges[j])
                if mode == 'b_abs' or mode == 'a_abs':
                    plot[j].set_ylim(0, ranges[j])

            plt.tight_layout()
            plt.savefig('fig/summation_ave_cfo_' + str(mode) + '.png')
            plt.show()

        return summation_3sec_datas

    def subtraction_ave_cfo(self, graph=False):
        dyad_p, dyad_f, dyad_pa, dyad_fa = self.dyad_cfo.subtraction_ave_cfo_3sec()
        triad_p, triad_f, triad_pa, triad_fa = self.triad_cfo.subtraction_ave_cfo_3sec()
        tetrad_p, tetrad_f, tetrad_pa, tetrad_fa = self.tetrad_cfo.subtraction_ave_cfo_3sec()
        subtraction_3sec_datas = [
            [dyad_p, triad_p, tetrad_p],
            [dyad_f, triad_f, tetrad_f],
            [dyad_pa, triad_pa, tetrad_pa],
            [dyad_fa, triad_fa, tetrad_fa],
        ]

        if graph == True:
            sns.set()
            # sns.set_style('whitegrid')
            sns.set_palette('Set3')

            types = ['Subtraction PCFO (Avg)',
                     'Subtraction FCFO (Avg)',
                     'Subtraction abs. PCFO (Avg)',
                     'Subtraction abs. FCFO (Avg)'
                     ]
            ranges = [0.4, 8.0, 0.4, 6.0]

            fig = plt.figure(figsize=(10, 7), dpi=150)

            plot = [
                fig.add_subplot(2, 2, 1),
                fig.add_subplot(2, 2, 2),
                fig.add_subplot(2, 2, 3),
                fig.add_subplot(2, 2, 4),
            ]

            for j in range(len(plot)):
                dfpp = []
                dfpp_melt = []
                for i in range(len(dyad_p)):
                    dfpp.append(pd.DataFrame({
                        'Dyad': subtraction_3sec_datas[j][0][i],
                        'Triad': subtraction_3sec_datas[j][1][i],
                        'Tetrad': subtraction_3sec_datas[j][2][i],
                    })
                    )

                    dfpp_melt.append(pd.melt(dfpp[i]))
                    dfpp_melt[i]['Group'] = 'Group' + str(i + 1)

                df = pd.concat([i for i in dfpp_melt], axis=0)

                sns.boxplot(x="variable", y="value", data=df, ax=plot[j], sym="")
                sns.stripplot(x='variable', y='value', data=df, hue='Group', dodge=True,
                              jitter=0.2, color='black', palette='Paired', ax=plot[j])

                plot[j].legend_ = None
                plot[j].set_ylabel(types[j])
                # plot[j].axes.xaxis.set_visible(False)
                plot[j].set_ylim(0.0, ranges[j])

            plt.tight_layout()
            plt.savefig('fig/subtraction_ave_cfo.png')
            # plt.show()

        return subtraction_3sec_datas

    def summation_ave_cfo_bs(self, graph=False, mode='no_abs'):
        dyad_p, dyad_f, dyad_pa, dyad_fa = self.dyad_cfo.summation_ave_cfo_3sec(mode)
        triad_p, triad_f, triad_pa, triad_fa = self.triad_cfo.summation_ave_cfo_3sec(mode)
        tetrad_p, tetrad_f, tetrad_pa, tetrad_fa = self.tetrad_cfo.summation_ave_cfo_3sec(mode)
        summation_3sec_datas = [
            [dyad_p, triad_p, tetrad_p],
            [dyad_f, triad_f, tetrad_f],
            [dyad_pa, triad_pa, tetrad_pa],
            [dyad_fa, triad_fa, tetrad_fa],
        ]

        R = 10000
        summation_3sec_datas_bs = []

        for i in range(len(summation_3sec_datas)):
            summation_3sec_datas_bs.append([])
            for j in range(len(summation_3sec_datas[i])):
                summation_3sec_datas_bs[i].append(combine.bootstrap(self, summation_3sec_datas[i][j], R))

        if graph == True:
            sns.set()
            # sns.set_style('whitegrid')
            sns.set_palette('Set3')

            types = ['Summation PCFO (Avg)',
                     'Summation FCFO (Avg)',
                     'Summation abs. PCFO (Avg)',
                     'Summation abs. FCFO (Avg)']
            ranges = [0.06, 0.5, 0.3, 4.0]

            if mode == 'b_abs':
                types = ['Before abs.\nSummation PCFO (Avg)',
                         'Before abs.\nSummation FCFO (Avg)',
                         'Before abs.\nSummation abs. PCFO (Avg)',
                         'Before abs.\nSummation abs. FCFO (Avg)']
                ranges = [0.25, 3.0, 0.25, 3.0]

            if mode == 'a_abs':
                types = ['After abs.\nSummation PCFO (Avg)',
                         'After abs.\nSummation FCFO (Avg)',
                         'After abs.\nSummation abs. PCFO (Avg)',
                         'After abs.\nSummation abs. FCFO (Avg)']
                ranges = [0.2, 1.0, 0.25, 3.0]

            fig = plt.figure(figsize=(10, 7), dpi=150)

            plot = [
                fig.add_subplot(2, 2, 1),
                fig.add_subplot(2, 2, 2),
                fig.add_subplot(2, 2, 3),
                fig.add_subplot(2, 2, 4),
            ]

            for j in range(len(plot)):
                dfpp = pd.DataFrame({
                    'Dyad': summation_3sec_datas_bs[j][0],
                    'Triad': summation_3sec_datas_bs[j][1],
                    'Tetrad': summation_3sec_datas_bs[j][2],
                })

                df = pd.melt(dfpp)

                sns.boxplot(x="variable", y="value", data=df, ax=plot[j], sym="")

                plot[j].legend_ = None
                plot[j].set_ylabel(types[j])
                # plot[j].axes.xaxis.set_visible(False)
                plot[j].set_ylim(-ranges[j], ranges[j])
                if mode == 'b_abs' or mode == 'a_abs':
                    plot[j].set_ylim(0, ranges[j])

            plt.tight_layout()
            plt.savefig('fig/summation_ave_cfo_bs' + str(mode) + '.png')
            # plt.show()

        return summation_3sec_datas

    def subtraction_ave_cfo_bs(self, graph=False):
        dyad_p, dyad_f, dyad_pa, dyad_fa = self.dyad_cfo.subtraction_ave_cfo_3sec()
        triad_p, triad_f, triad_pa, triad_fa = self.triad_cfo.subtraction_ave_cfo_3sec()
        tetrad_p, tetrad_f, tetrad_pa, tetrad_fa = self.tetrad_cfo.subtraction_ave_cfo_3sec()
        subtraction_3sec_datas = [
            [dyad_p, triad_p, tetrad_p],
            [dyad_f, triad_f, tetrad_f],
            [dyad_pa, triad_pa, tetrad_pa],
            [dyad_fa, triad_fa, tetrad_fa],
        ]

        R = 10000
        subtraction_3sec_datas_bs = []

        for i in range(len(subtraction_3sec_datas)):
            subtraction_3sec_datas_bs.append([])
            for j in range(len(subtraction_3sec_datas[i])):
                subtraction_3sec_datas_bs[i].append(combine.bootstrap(self, subtraction_3sec_datas[i][j], R))

        if graph == True:
            sns.set()
            # sns.set_style('whitegrid')
            sns.set_palette('Set3')

            types = ['Subtraction PCFO (Avg)',
                     'Subtraction FCFO (Avg)',
                     'Subtraction abs. PCFO (Avg)',
                     'Subtraction abs. FCFO (Avg)'
                     ]
            ranges = [0.4, 8.0, 0.4, 6.0]

            fig = plt.figure(figsize=(10, 7), dpi=150)

            plot = [
                fig.add_subplot(2, 2, 1),
                fig.add_subplot(2, 2, 2),
                fig.add_subplot(2, 2, 3),
                fig.add_subplot(2, 2, 4),
            ]

            for j in range(len(plot)):
                df = pd.DataFrame({
                    'Dyad': subtraction_3sec_datas_bs[j][0],
                    'Triad': subtraction_3sec_datas_bs[j][1],
                    'Tetrad': subtraction_3sec_datas_bs[j][2],
                })

                df_melt = pd.melt(df)

                sns.boxplot(x="variable", y="value", data=df_melt, ax=plot[j], sym="")

                plot[j].legend_ = None
                plot[j].set_ylabel(types[j])
                # plot[j].axes.xaxis.set_visible(False)
                plot[j].set_ylim(0.0, ranges[j])

            plt.tight_layout()
            plt.savefig('fig/subtraction_ave_cfo_bs.png')
            # plt.show()

        return subtraction_3sec_datas

    def performance_deviation(self):
        error_period_dyad, spend_period_dyad = self.dyad_cfo.period_performance_cooperation()
        error_period_triad, spend_period_triad = self.triad_cfo.period_performance_cooperation()
        error_period_tetrad, spend_period_tetrad = self.tetrad_cfo.period_performance_cooperation()

        data = [
            error_period_dyad.T,
            error_period_triad.T,
            error_period_tetrad.T,
            spend_period_dyad.T,
            spend_period_triad.T,
            spend_period_tetrad.T,
        ]

        # data_bs = []
        #
        # R = 10000
        # for i in range(len(data)):
        #     data_bs.append([])
        #     for j in range(len(data[i])):
        #         data_bs[i].append(combine.bootstrap(self, data[i][j], R))

        df_error_raw_ = []
        df_spend_raw_ = []
        for i in range(len(data[0])):
            df_error_ = pd.DataFrame({
                'Dyad': data[0][i],
                'Triad': data[1][i],
                'Tetrad': data[2][i],
            }
            )
            df_error_raw_.append(df_error_)

            df_spend_ = pd.DataFrame({
                'Dyad': data[3][i],
                'Triad': data[4][i],
                'Tetrad': data[5][i],
            }
            )
            df_spend_raw_.append(df_spend_)

        df_error_raw = pd.concat(df_error_raw_[i] for i in range(len(df_error_raw_)))
        df_error_raw_melt = pd.melt(df_error_raw)
        Period = [i for i in range(1, len(data[0]) + 1)] * len(data[0][0]) * 3
        df_error = pd.concat([df_error_raw_melt, pd.DataFrame({'Period': Period})], axis=1)

        df_error.rename(columns={'variable': 'Group size'}, inplace=True)
        df_error.rename(columns={'value': 'Value'}, inplace=True)

        df_spend_raw = pd.concat(df_spend_raw_[i] for i in range(len(df_spend_raw_)))
        df_spend_raw_melt = pd.melt(df_spend_raw)
        df_spend = pd.concat([df_spend_raw_melt, pd.DataFrame({'Period': Period})], axis=1)

        df_spend.rename(columns={'variable': 'Group size'}, inplace=True)
        df_spend.rename(columns={'value': 'Value'}, inplace=True)

        df = [df_error, df_spend]
        label = ['Error', 'Spend']
        ylim = [
            (-0.01, 0.01),
            (-0.1, 0.3)
        ]

        kwargs = dict(
            height=10,
            aspect=1.5,
            scatter=True,
            n_boot=1000,
            x_ci='sd',
        )

        # for i in range(2):
        #     sns.set(font_scale=2)
        #     sns.set_context("poster")
        #     # sns.set_style("whitegrid", {'grid.linestyle': '--'})
        #     sns.set_style("white")
        #     p = sns.lmplot(data=df[i], x='Period', y='Value', hue='Group size', order=2, **kwargs)
        #     p.set(title=label[i])
        #     p.set(xlim=(0, 20))
        #     p.set(ylim=ylim[i])
        # plt.show()

        std = []
        mean = []
        data_use = data
        for i in range(len(data_use)):
            std.append([])
            mean.append([])
            for j in range(len(data_use[i])):
                std[i].append(np.std(data_use[i][j]))
                mean[i].append(np.mean(data_use[i][j]))

        period = [i + 1 for i in range(len(std[0]))] * 3
        df_period = pd.DataFrame(period, columns=['Period'])

        df_error_ = pd.DataFrame({
            'Dyad': mean[0],
            'Triad': mean[1],
            'Tetrad': mean[2],
        }
        )

        df_melt_error = pd.concat([pd.melt(df_error_), df_period], axis=1)
        df_melt_error.rename(columns={'variable': 'Group size'}, inplace=True)
        df_melt_error.rename(columns={'value': 'Deviation'}, inplace=True)

        df_spend_melt_ = pd.DataFrame({
            'Dyad': mean[3],
            'Triad': mean[4],
            'Tetrad': mean[5],
        }
        )
        df_melt_spend = pd.concat([pd.melt(df_spend_melt_), df_period], axis=1)
        df_melt_spend.rename(columns={'variable': 'Group size'}, inplace=True)
        df_melt_spend.rename(columns={'value': 'Deviation'}, inplace=True)

        df = [df_melt_error, df_melt_spend]
        label = ['Error', 'Spend']
        ylim = [
            (-0.01, 0.01),
            (-0.1, 0.3)
        ]

        kwargs = dict(
            height=10,
            aspect=1.5,
            scatter=True,
            n_boot=1000,
            x_ci='sd',
        )

        for i in range(2):
            sns.set(font_scale=2)
            sns.set_context("poster")
            # sns.set_style("whitegrid", {'grid.linestyle': '--'})
            sns.set_style("white")
            p = sns.lmplot(data=df[i], x='Period', y='Deviation', hue='Group size', order=2, **kwargs)
            p.set(title=label[i])
            p.set(xlim=(0, None))
            p.set(ylim=ylim[i])
        plt.show()

    def variance_analysis(self, mode='no_abs'):
        dyad_valiance, dyad_valiance_period = self.dyad_cfo.fcfo_valiance()
        triad_valiance, triad_valiance_period = self.triad_cfo.fcfo_valiance()
        tetrad_valiance, tetrad_valiance_period = self.tetrad_cfo.fcfo_valiance()

        error_period_dyad, spend_period_dyad = self.dyad_cfo.period_performance_cooperation()
        error_period_triad, spend_period_triad = self.triad_cfo.period_performance_cooperation()
        error_period_tetrad, spend_period_tetrad = self.tetrad_cfo.period_performance_cooperation()

        dyad_pp, dyad_rp, dyad_pf, dyad_rf = self.dyad_cfo.summation_cfo_3sec(mode)
        triad_pp, triad_rp, triad_pf, triad_rf = self.triad_cfo.summation_cfo_3sec(mode)
        tetrad_pp, tetrad_rp, tetrad_pf, tetrad_rf = self.tetrad_cfo.summation_cfo_3sec(mode)

        errorx_period_dyad, errory_period_dyad, spendx_period_dyad, spendy_period_dyad = self.dyad_cfo.period_performance_cooperation_each_axis()
        errorx_period_triad, errory_period_triad, spendx_period_triad, spendy_period_triad = self.triad_cfo.period_performance_cooperation_each_axis()
        errorx_period_tetrad, errory_period_tetrad, spendx_period_tetrad, spendy_period_tetrad = self.tetrad_cfo.period_performance_cooperation_each_axis()

        label = ['Dyad', 'Triad', 'Tetrad']
        variance_label = [
            'Variance of pitch FCFO(Nm)',
            'Variance of roll FCFO(Nm)',
        ]
        performance_label = [
            'error',
            'spend',
        ]
        summation_label = [
            'Summation of pitch FCFO(Nm)',
            'Summation of roll FCFO(Nm)',
        ]
        performance_ea_label = [
            ['error_x', 'error_y'],
            ['spend_x', 'spend_y'],
        ]
        summation_label = [
            'Summation Pitch FCFO (Avg)',
            'Summation Roll FCFO (Avg)']
        ranges = [0.05, 0.05, 0.3, 0.3]

        if mode == 'b_abs':
            summation_label = [
                'Before abs.\nSummation Pitch FCFO (Avg)',
                'Before abs.\nSummation Roll FCFO (Avg)']
            ranges = [0.2, 0.2, 2.0, 2.0]

        if mode == 'a_abs':
            summation_label = [
                'After abs.\nSummation Pitch FCFO (Avg)',
                'After abs.\nSummation Roll FCFO (Avg)']
            ranges = [0.2, 0.2, 0.8, 0.8]

        variance_period = [dyad_valiance_period, triad_valiance_period, tetrad_valiance_period]
        error = [error_period_dyad, error_period_triad, error_period_tetrad]
        spend = [spend_period_dyad, spend_period_triad, spend_period_tetrad]
        errorx = [errorx_period_dyad, errorx_period_triad, errorx_period_tetrad]
        errory = [errory_period_dyad, errory_period_triad, errory_period_tetrad]
        spendx = [spendx_period_dyad, spendx_period_triad, spendx_period_tetrad]
        spendy = [spendy_period_dyad, spendy_period_triad, spendy_period_tetrad]

        summation = [
            [dyad_pf, triad_pf, tetrad_pf],
            [dyad_rf, triad_rf, tetrad_rf],
        ]

        periods = []
        for l in range(len(label)):
            period = []
            for i, j, k, p, r, a, b, c, d in zip(variance_period[l], error[l], spend[l], summation[0][l],
                                                 summation[1][l], errorx[l], errory[l], spendx[l], spendy[l]):
                period_ = pd.DataFrame({
                    variance_label[0]: i[0],
                    variance_label[1]: i[1],
                    performance_label[0]: j,
                    performance_label[1]: k,
                    summation_label[0]: p,
                    summation_label[1]: r,
                    performance_ea_label[0][0]: a,
                    performance_ea_label[0][1]: b,
                    performance_ea_label[1][0]: c,
                    performance_ea_label[1][1]: d,
                    'Group size': label[l],
                }
                )

                period.append(period_)
            periods.append(pd.concat([_ for _ in period], axis=0).reset_index(drop=True))

        df = pd.concat([_ for _ in periods], axis=0).reset_index(drop=True)
        # print(df)

        # ##histogram
        # fig = plt.figure(figsize=(10, 10), dpi=300)
        # subplot = [fig.add_subplot(2, 1, i+1) for i in range(2)]
        # for i in range(2):
        #     for j in range(3):
        #         df_ = df[df['Group size'] == label[j]]
        #         sns.histplot(df_[variance_label[i]], kde=True, bins=10, label=label[j], ax=subplot[i], stat='probability')
        #
        #     plt.legend()
        # plt.tight_layout()
        # plt.show()

        # ## variance-performance
        # fig = plt.figure(figsize=(10, 10), dpi=300)
        # subplot = [fig.add_subplot(2, 2, i+1) for i in range(4)]
        # for i in range(2):
        #     for j in range(2):
        #         g = sns.scatterplot(data=df, x=variance_label[j], y=performance_label[i], hue='Group size', ax=subplot[i*2+j], s=10)
        #         for lh in g.legend_.legendHandles:
        #             lh.set_alpha(1)
        #             lh._sizes = [10]
        #
        # plt.tight_layout()
        # plt.show()

        # ## variance-each_axis_performance
        # fig = plt.figure(figsize=(10, 10), dpi=300)
        # subplot = [fig.add_subplot(2, 2, i + 1) for i in range(4)]
        # for i in range(2):
        #     for j in range(2):
        #         g = sns.scatterplot(data=df, x=variance_label[j], y=performance_ea_label[i][j], hue='Group size',
        #                             ax=subplot[i * 2 + j], s=10)
        #         for lh in g.legend_.legendHandles:
        #             lh.set_alpha(1)
        #             lh._sizes = [10]
        #
        # plt.tight_layout()
        # plt.show()

        # ##summatio-variance
        # fig = plt.figure(figsize=(10, 10), dpi=300)
        # subplot = [fig.add_subplot(1, 2, i+1) for i in range(2)]
        # for j in range(2):
        #     g = sns.scatterplot(data=df, x=variance_label[j], y=summation_label[j], hue='Group size', ax=subplot[j], s=10)
        #     for lh in g.legend_.legendHandles:
        #         lh.set_alpha(1)
        #         lh._sizes = [10]
        #
        # # plt.tight_layout()
        # plt.show()

        # # ##summatio-variance with regression
        # kwargs = dict(
        #     height=10,
        #     aspect=1.5,
        #     scatter=True,
        #     # n_boot=1000,
        #     # ci='sd',
        #     ci=None,
        #     order=2,
        # )
        # sns.set(font_scale=2)
        # for i in range(2):
        #     g = sns.lmplot(data=df, x=variance_label[i], y=summation_label[i], hue='Group size', **kwargs)
        #     plt.savefig('fig/'+str(variance_label[i])+'_'+str(summation_label[i])+'_lmplot.png')
        #
        # # plt.tight_layout()
        # plt.show()

        # ##summatio-variance with performance
        # # cmap = ['Blues_r', 'Blues']
        # cmap = ['plasma_r', 'plasma']
        # fig = plt.figure(figsize=(10, 7), dpi=300)
        # subplot = [fig.add_subplot(2, 2, i+1) for i in range(4)]
        # for i in range(2):
        #     for j in range(2):
        #         points = subplot[i*2+j].scatter(x=df[variance_label[j]], y=df[summation_label[j]], c=df[performance_label[i]], cmap=cmap[i], s=1)
        #         plt.colorbar(points, ax=subplot[i*2+j], label=performance_label[i])
        #         subplot[i*2+j].set_xlabel(variance_label[j])
        #         subplot[i*2+j].set_ylabel(summation_label[j])
        #
        # # plt.tight_layout()
        # # plt.savefig('fig/variance_summation-'+str(mode)+'_performance.png')
        # plt.show()

        ##summatio-variance with performance each axis
        # cmap = ['Blues_r', 'Blues']
        cmap = ['plasma_r', 'plasma']
        fig = plt.figure(figsize=(10, 7), dpi=300)
        subplot = [fig.add_subplot(2, 2, i + 1) for i in range(4)]
        for i in range(2):
            for j in range(2):
                points = subplot[i * 2 + j].scatter(x=df[variance_label[j]], y=df[summation_label[j]],
                                                    c=df[performance_ea_label[i][j]], cmap=cmap[i], s=1)
                plt.colorbar(points, ax=subplot[i * 2 + j], label=performance_ea_label[i][j])
                subplot[i * 2 + j].set_xlabel(variance_label[j])
                subplot[i * 2 + j].set_ylabel(summation_label[j])

        # plt.tight_layout()
        # plt.savefig('fig/variance_summation-'+str(mode)+'_performance.png')
        plt.show()

    def correlation_ratio(self, categories, values):
        categories = np.array(categories)
        values = np.array(values)

        ssw = 0
        ssb = 0
        for category in set(categories):
            subgroup = values[np.where(categories == category)[0]]
            ssw += sum((subgroup - np.mean(subgroup)) ** 2)
            ssb += len(subgroup) * (np.mean(subgroup) - np.mean(values)) ** 2

        return (ssb / (ssb + ssw)) ** .5

    def ftr_3sec(self, graph=True, source='human'):
        dyad_p, dyad_r = self.dyad_cfo.get_ftr_3sec(source=source)
        triad_p, triad_r = self.triad_cfo.get_ftr_3sec(source=source)
        tetrad_p, tetrad_r = self.tetrad_cfo.get_ftr_3sec(source=source)

        ratio_3sec_datas = [
            [dyad_p, triad_p, tetrad_p],
            [dyad_r, triad_r, tetrad_r],
        ]

        if graph == True:
            sns.set()
            # sns.set_style('whitegrid')
            sns.set_palette('Set3')

            size = ['Dyad', 'Triad', 'Tetrad']

            types = ['Pitch FTR (Avg)',
                     'Roll FTR (Avg)',
                     ]
            ranges = [1.0, 1.0]

            sns.set(font='Times New Roman', font_scale=1.0)
            sns.set_style('ticks')
            sns.set_context("paper",
                            rc={
                                "axes.linewidth": 0.5,
                                "legend.fancybox": False,
                                'pdf.fonttype': 42,
                                'xtick.direction': 'in',
                                'ytick.major.width': 1.0,
                                'xtick.major.width': 1.0,
                            })

            fig = plt.figure(figsize=(10, 5), dpi=150)

            plot = [
                fig.add_subplot(1, 2, 1),
                fig.add_subplot(1, 2, 2),
            ]

            for j in range(len(plot)):
                dfpp = []
                dfpp_melt = []
                for i in range(len(dyad_p)):
                    dfpp.append(pd.DataFrame({
                        'Dyad': ratio_3sec_datas[j][0][i],
                        'Triad': ratio_3sec_datas[j][1][i],
                        'Tetrad': ratio_3sec_datas[j][2][i],
                    })
                    )

                    dfpp_melt.append(pd.melt(dfpp[i]))
                    dfpp_melt[i]['Group'] = 'Group' + str(i + 1)

                df = pd.concat([i for i in dfpp_melt], axis=0)

                xlabel = 'Group size'
                ylabel = 'FTR (Avg)'

                df.rename(columns={'variable': xlabel, 'value': ylabel}, inplace=True)

                ax = sns.boxplot(x=xlabel, y=ylabel, data=df, ax=plot[j], sym="")
                # sns.stripplot(x='variable', y='value', data=df, hue='Group', dodge=True,
                #               jitter=0.2, color='black', palette='Paired', ax=plot[j])

                plot[j].legend_ = None
                plot[j].set_ylabel(types[j])
                # plot[j].axes.xaxis.set_visible(False)
                plot[j].set_ylim(0.0, ranges[j])

                pairs = [
                    {size[1], size[2]},
                    {size[0], size[1]},
                    {size[0], size[2]},
                ]

                combine.t_test_multi(self, ax=ax, pairs=pairs, data=df, x=xlabel, y=ylabel)
                # combine.anova(self, data=df, variable='variable', value='value')

            plt.tight_layout()
            os.makedirs('fig/FTR', exist_ok=True)
            if source == 'human':
                plt.savefig('fig/FTR/Compare_FTR_3sec.png')
            else:
                plt.savefig('fig/FTR/Compare_FTR_3sec_model.png')
            plt.show()

        return ratio_3sec_datas

    def ftr_3sec_combine(self, graph=True, source='human'):
        dyad_ftr = self.dyad_cfo.get_ftr_combine_3sec(source=source)
        triad_ftr = self.triad_cfo.get_ftr_combine_3sec(source=source)
        tetrad_ftr = self.tetrad_cfo.get_ftr_combine_3sec(source=source)

        ftr_3sec_datas = [
            dyad_ftr, triad_ftr, tetrad_ftr,
        ]

        if graph == True:
            sns.set()
            # sns.set_style('whitegrid')
            sns.set_palette('Set3')

            size = ['Dyad', 'Triad', 'Tetrad']

            plt.figure(figsize=(10, 10), dpi=150)

            sns.set(font='Times New Roman', font_scale=1.0)
            sns.set_style('ticks')
            sns.set_context("paper",
                            rc={
                                "axes.linewidth": 0.5,
                                "legend.fancybox": False,
                                'pdf.fonttype': 42,
                                'xtick.direction': 'in',
                                'ytick.major.width': 1.0,
                                'xtick.major.width': 1.0,
                            })

            dfpp = []
            dfpp_melt = []
            for i in range(len(dyad_ftr)):
                dfpp.append(pd.DataFrame({
                    'Dyad': ftr_3sec_datas[0][i],
                    'Triad': ftr_3sec_datas[1][i],
                    'Tetrad': ftr_3sec_datas[2][i],
                })
                )

                dfpp_melt.append(pd.melt(dfpp[i]))
                dfpp_melt[i]['Group'] = 'Group' + str(i + 1)

            df = pd.concat([i for i in dfpp_melt], axis=0)

            xlabel = 'Group size'
            ylabel = 'FTR (Avg)'

            df.rename(columns={'variable': xlabel, 'value': ylabel}, inplace=True)

            ax = sns.boxplot(x=xlabel, y=ylabel, data=df, sym="")
            # sns.stripplot(x='variable', y='value', data=df, hue='Group', dodge=True,
            #               jitter=0.2, color='black', palette='Paired', ax=plot[j])

            plt.legend = None
            plt.ylabel('FTR (Avg)')
            # plot[j].axes.xaxis.set_visible(False)
            plt.ylim(0.0, 1.0)

            pairs = [
                {size[1], size[2]},
                {size[0], size[1]},
                {size[0], size[2]},
            ]

            combine.t_test_multi(self, ax=ax, pairs=pairs, data=df, x=xlabel, y=ylabel)
            # combine.anova(self, data=df, variable=xlabel, value=ylabel)

            plt.tight_layout()
            os.makedirs('fig/FTR', exist_ok=True)
            if source == 'human':
                plt.savefig('fig/FTR/Compare_FTR_3sec_combine.png')
            else:
                plt.savefig('fig/FTR/Compare_FTR_3sec_combine_model.png')
            plt.show()

        return ftr_3sec_datas

    def ftr_3sec_diff(self, graph=True):
        dyad_p, dyad_r = self.dyad_cfo.get_ftr_3sec_diff()
        triad_p, triad_r = self.triad_cfo.get_ftr_3sec_diff()
        tetrad_p, tetrad_r = self.tetrad_cfo.get_ftr_3sec_diff()

        ftr_3sec_datas = [
            [dyad_p, triad_p, tetrad_p],
            [dyad_r, triad_r, tetrad_r],
        ]

        if graph == True:
            sns.set()
            # sns.set_style('whitegrid')
            sns.set_palette('Set3')

            size = ['Dyad', 'Triad', 'Tetrad']

            types = ['Pitch FTR (Avg)',
                     'Roll FTR (Avg)',
                     ]
            ranges = [1.0, 1.0]

            sns.set(font='Times New Roman', font_scale=1.0)
            sns.set_style('ticks')
            sns.set_context("paper",
                            rc={
                                "axes.linewidth": 0.5,
                                "legend.fancybox": False,
                                'pdf.fonttype': 42,
                                'xtick.direction': 'in',
                                'ytick.major.width': 1.0,
                                'xtick.major.width': 1.0,
                            })

            fig = plt.figure(figsize=(10, 5), dpi=150)

            plot = [
                fig.add_subplot(1, 2, 1),
                fig.add_subplot(1, 2, 2),
            ]

            for j in range(len(plot)):
                dfpp = []
                dfpp_melt = []
                for i in range(len(dyad_p)):
                    dfpp.append(pd.DataFrame({
                        'Dyad': ftr_3sec_datas[j][0][i],
                        'Triad': ftr_3sec_datas[j][1][i],
                        'Tetrad': ftr_3sec_datas[j][2][i],
                    })
                    )

                    dfpp_melt.append(pd.melt(dfpp[i]))
                    dfpp_melt[i]['Group'] = 'Group' + str(i + 1)

                df = pd.concat([i for i in dfpp_melt], axis=0)

                xlabel = 'Group size'
                ylabel = 'Cooperative FTR (Avg)'

                df.rename(columns={'variable': xlabel, 'value': ylabel}, inplace=True)

                ax = sns.boxplot(x=xlabel, y=ylabel, data=df, ax=plot[j], sym="")
                # sns.stripplot(x='variable', y='value', data=df, hue='Group', dodge=True,
                #               jitter=0.2, color='black', palette='Paired', ax=plot[j])

                plot[j].legend_ = None
                plot[j].set_ylabel(types[j])
                # plot[j].axes.xaxis.set_visible(False)
                plot[j].set_ylim(-1.0, ranges[j])

                pairs = [
                    {size[1], size[2]},
                    {size[0], size[1]},
                    {size[0], size[2]},
                ]

                combine.t_test_multi(self, ax=ax, pairs=pairs, data=df, x=xlabel, y=ylabel)
                # combine.anova(self, data=df, variable='variable', value='value')

            plt.tight_layout()
            os.makedirs('fig/FTR', exist_ok=True)
            plt.savefig('fig/FTR/Compare_Cooperative_FTR_3sec.png')

            plt.show()

        return ftr_3sec_datas

    def ftr_3sec_combine_diff(self, graph=True):
        dyad_ftr = self.dyad_cfo.get_ftr_combine_3sec_diff()
        triad_ftr = self.triad_cfo.get_ftr_combine_3sec_diff()
        tetrad_ftr = self.tetrad_cfo.get_ftr_combine_3sec_diff()

        ftr_3sec_datas = [
            dyad_ftr, triad_ftr, tetrad_ftr,
        ]

        if graph == True:
            sns.set()
            # sns.set_style('whitegrid')
            sns.set_palette('Set3')

            size = ['Dyad', 'Triad', 'Tetrad']

            plt.figure(figsize=(10, 10), dpi=150)

            sns.set(font='Times New Roman', font_scale=1.0)
            sns.set_style('ticks')
            sns.set_context("paper",
                            rc={
                                "axes.linewidth": 0.5,
                                "legend.fancybox": False,
                                'pdf.fonttype': 42,
                                'xtick.direction': 'in',
                                'ytick.major.width': 1.0,
                                'xtick.major.width': 1.0,
                            })

            dfpp = []
            dfpp_melt = []
            for i in range(len(dyad_ftr)):
                dfpp.append(pd.DataFrame({
                    'Dyad': ftr_3sec_datas[0][i],
                    'Triad': ftr_3sec_datas[1][i],
                    'Tetrad': ftr_3sec_datas[2][i],
                })
                )

                dfpp_melt.append(pd.melt(dfpp[i]))
                dfpp_melt[i]['Group'] = 'Group' + str(i + 1)

            df = pd.concat([i for i in dfpp_melt], axis=0)

            xlabel = 'Group size'
            ylabel = 'Cooperative FTR (Avg)'

            df.rename(columns={'variable': xlabel, 'value': ylabel}, inplace=True)

            ax = sns.boxplot(x=xlabel, y=ylabel, data=df, sym="")
            # sns.stripplot(x='variable', y='value', data=df, hue='Group', dodge=True,
            #               jitter=0.2, color='black', palette='Paired', ax=plot[j])

            plt.legend = None
            plt.ylabel('Cooperative FTR (Avg)')
            # plot[j].axes.xaxis.set_visible(False)
            plt.ylim(-1.0, 0.4)

            pairs = [
                {size[1], size[2]},
                {size[0], size[1]},
                {size[0], size[2]},
            ]

            combine.t_test_multi(self, ax=ax, pairs=pairs, data=df, x=xlabel, y=ylabel)
            # combine.anova(self, data=df, variable=xlabel, value=ylabel)

            plt.tight_layout()
            os.makedirs('fig/FTR', exist_ok=True)
            plt.savefig('fig/FTR/Compare_Cooperative_FTR_3sec_combine.png')
            plt.show()

        return ftr_3sec_datas

    def performance_ftr(self, mode='h-m'):
        ftr_dyad = self.dyad_cfo.get_ftr_combine_3sec(self)
        ftr_triad = self.triad_cfo.get_ftr_combine_3sec(self)
        ftr_tetrad = self.tetrad_cfo.get_ftr_combine_3sec(self)

        error_period_dyad, spend_period_dyad = self.dyad_cfo.period_performance_cooperation()
        error_period_triad, spend_period_triad = self.triad_cfo.period_performance_cooperation()
        error_period_tetrad, spend_period_tetrad = self.tetrad_cfo.period_performance_cooperation()
        if mode == 'h-h':
            error_period_dyad, spend_period_dyad = self.dyad_cfo.period_performance_human()
            error_period_triad, spend_period_triad = self.triad_cfo.period_performance_human()
            error_period_tetrad, spend_period_tetrad = self.tetrad_cfo.period_performance_human()

        ftr = [
            ftr_dyad, ftr_triad, ftr_tetrad
        ]

        performance = [
            [error_period_dyad, spend_period_dyad],
            [error_period_triad, spend_period_triad],
            [error_period_tetrad, spend_period_tetrad],
        ]

        size = ['Dyad', 'Triad', 'Tetrad']

        performance_label = ['RMSE', 'Time']

        df_ = []
        for i in range(len(size)):
            for j in range(len(performance_label)):
                for k in range(len(performance[0][0])):
                    df_.append(pd.DataFrame({
                        'FTR': ftr[i][k],
                        'Performance': performance[i][j][k],
                        'Performance_type': performance_label[j],
                        'Group Size': size[i],
                        'Group': str(k + 1)
                    })
                    )

        df = pd.concat([i for i in df_], axis=0)
        df.reset_index(drop=True, inplace=True)

        # FacetGridを作成してグラフを設定
        g = sns.FacetGrid(df, col="Performance_type", row="Group Size", hue="Group", height=4, aspect=1.2, sharey=False,
                          sharex=False)
        g.map(plot_scatter, "Performance", "FTR")

        ylims = [(0, 1)]

        xlims = [(-0.015, 0.015), (-0.1, 0.5),
                 (-0.015, 0.015), (-0.1, 0.5),
                 (-0.015, 0.015), (-0.1, 0.5)]

        if mode == 'h-h':
            xlims = [(0.0, 0.1), (0.0, 3.0),
                     (0.0, 0.1), (0.0, 3.0),
                     (0.0, 0.1), (0.0, 3.0)]

        # Iterate over the AxesSubplots in the FacetGrid object, and set the y limit individually
        for num, ax in enumerate(g.axes.flatten()):
            # col_var = ax.get_title().split(" ")[-1]  # get the column variable from the title
            ax.set_xlim(xlims[num])
            ax.set_ylim(ylims[0])

        if mode == 'h-h':
            os.makedirs("fig/FTR-Performance/h-h/", exist_ok=True)
            plt.savefig("fig/FTR-Performance/h-h/FTR-Performance_h-h.png")
        else:
            os.makedirs("fig/FTR-Performance/", exist_ok=True)
            plt.savefig("fig/FTR-Performance/FTR-Performance_.png")

        plt.show()

    def ief_3sec(self, graph=True, source='human'):
        dyad_p, dyad_r = self.dyad_cfo.get_ief_3sec(source=source)
        triad_p, triad_r = self.triad_cfo.get_ief_3sec(source=source)
        tetrad_p, tetrad_r = self.tetrad_cfo.get_ief_3sec(source=source)

        ratio_3sec_datas = [
            [dyad_p, triad_p, tetrad_p],
            [dyad_r, triad_r, tetrad_r],
        ]

        if graph == True:
            sns.set()
            # sns.set_style('whitegrid')
            sns.set_palette('Set3')

            size = ['Dyad', 'Triad', 'Tetrad']

            types = ['Pitch IEF (Avg)',
                     'Roll IEF (Avg)',
                     ]
            ranges = [6.0, 6.0]
            if source == 'model':
                ranges = [1.0, 1.0]

            sns.set(font='Times New Roman', font_scale=1.0)
            sns.set_style('ticks')
            sns.set_context("paper",
                            rc={
                                "axes.linewidth": 0.5,
                                "legend.fancybox": False,
                                'pdf.fonttype': 42,
                                'xtick.direction': 'in',
                                'ytick.major.width': 1.0,
                                'xtick.major.width': 1.0,
                            })

            fig = plt.figure(figsize=(10, 5), dpi=150)

            plot = [
                fig.add_subplot(1, 2, 1),
                fig.add_subplot(1, 2, 2),
            ]

            for j in range(len(plot)):
                dfpp = []
                dfpp_melt = []
                for i in range(len(dyad_p)):
                    dfpp.append(pd.DataFrame({
                        'Dyad': ratio_3sec_datas[j][0][i],
                        'Triad': ratio_3sec_datas[j][1][i],
                        'Tetrad': ratio_3sec_datas[j][2][i],
                    })
                    )

                    dfpp_melt.append(pd.melt(dfpp[i]))
                    dfpp_melt[i]['Group'] = 'Group' + str(i + 1)

                df = pd.concat([i for i in dfpp_melt], axis=0)

                xlabel = 'Group size'
                ylabel = 'IEF (Avg)'

                df.rename(columns={'variable': xlabel, 'value': ylabel}, inplace=True)

                ax = sns.boxplot(x=xlabel, y=ylabel, data=df, ax=plot[j], sym="")
                # sns.stripplot(x='variable', y='value', data=df, hue='Group', dodge=True,
                #               jitter=0.2, color='black', palette='Paired', ax=plot[j])

                plot[j].legend_ = None
                plot[j].set_ylabel(types[j])
                # plot[j].axes.xaxis.set_visible(False)
                plot[j].set_ylim(0.0, ranges[j])

                pairs = [
                    {size[1], size[2]},
                    {size[0], size[1]},
                    {size[0], size[2]},
                ]

                combine.t_test_multi(self, ax=ax, pairs=pairs, data=df, x=xlabel, y=ylabel)
                # combine.anova(self, data=df, variable='variable', value='value')

            plt.tight_layout()
            os.makedirs('fig/IEF', exist_ok=True)
            if source == 'human':
                plt.savefig('fig/IEF/Compare_IEF_3sec.png')
            else:
                plt.savefig('fig/IEF/Compare_IEF_3sec_model.png')

            plt.show()

        return ratio_3sec_datas

    def ief_3sec_diff(self, graph=True, source='human'):
        dyad_p, dyad_r = self.dyad_cfo.get_ief_3sec_diff()
        triad_p, triad_r = self.triad_cfo.get_ief_3sec_diff()
        tetrad_p, tetrad_r = self.tetrad_cfo.get_ief_3sec_diff()

        ratio_3sec_datas = [
            [dyad_p, triad_p, tetrad_p],
            [dyad_r, triad_r, tetrad_r],
        ]

        if graph == True:
            sns.set()
            # sns.set_style('whitegrid')
            sns.set_palette('Set3')

            size = ['Dyad', 'Triad', 'Tetrad']

            types = ['Pitch IEF (Avg)',
                     'Roll IEF (Avg)',
                     ]
            ranges = [8.0, 8.0]

            sns.set(font='Times New Roman', font_scale=1.0)
            sns.set_style('ticks')
            sns.set_context("paper",
                            rc={
                                "axes.linewidth": 0.5,
                                "legend.fancybox": False,
                                'pdf.fonttype': 42,
                                'xtick.direction': 'in',
                                'ytick.major.width': 1.0,
                                'xtick.major.width': 1.0,
                            })

            fig = plt.figure(figsize=(10, 5), dpi=150)

            plot = [
                fig.add_subplot(1, 2, 1),
                fig.add_subplot(1, 2, 2),
            ]

            for j in range(len(plot)):
                dfpp = []
                dfpp_melt = []
                for i in range(len(dyad_p)):
                    dfpp.append(pd.DataFrame({
                        'Dyad': ratio_3sec_datas[j][0][i],
                        'Triad': ratio_3sec_datas[j][1][i],
                        'Tetrad': ratio_3sec_datas[j][2][i],
                    })
                    )

                    dfpp_melt.append(pd.melt(dfpp[i]))
                    dfpp_melt[i]['Group'] = 'Group' + str(i + 1)

                df = pd.concat([i for i in dfpp_melt], axis=0)

                xlabel = 'Group size'
                ylabel = 'Cooperative IEF (Avg)'

                df.rename(columns={'variable': xlabel, 'value': ylabel}, inplace=True)

                ax = sns.boxplot(x=xlabel, y=ylabel, data=df, ax=plot[j], sym="")
                # sns.stripplot(x='variable', y='value', data=df, hue='Group', dodge=True,
                #               jitter=0.2, color='black', palette='Paired', ax=plot[j])

                plot[j].legend_ = None
                plot[j].set_ylabel(types[j])
                # plot[j].axes.xaxis.set_visible(False)
                plot[j].set_ylim(-1.0, ranges[j])

                pairs = [
                    {size[1], size[2]},
                    {size[0], size[1]},
                    {size[0], size[2]},
                ]

                combine.t_test_multi(self, ax=ax, pairs=pairs, data=df, x=xlabel, y=ylabel)
                # combine.anova(self, data=df, variable='variable', value='value')

            plt.tight_layout()
            os.makedirs('fig/IEF', exist_ok=True)
            plt.savefig('fig/IEF/Compare_Cooperative_IEF_3sec.png')

            plt.show()

        return ratio_3sec_datas

    def ief_3sec_combine(self, graph=True, source='human'):
        dyad_ief = self.dyad_cfo.get_ief_combine_3sec(source=source)
        triad_ief = self.triad_cfo.get_ief_combine_3sec(source=source)
        tetrad_ief = self.tetrad_cfo.get_ief_combine_3sec(source=source)

        ief_3sec_datas = [
            dyad_ief, triad_ief, tetrad_ief,
        ]

        if graph == True:
            sns.set()
            # sns.set_style('whitegrid')
            sns.set_palette('Set3')

            size = ['Dyad', 'Triad', 'Tetrad']

            plt.figure(figsize=(10, 10), dpi=300)

            sns.set(font='Times New Roman', font_scale=2.0)
            sns.set_style('ticks')
            sns.set_context("poster",
                            rc={
                                "axes.linewidth": 0.5,
                                "legend.fancybox": False,
                                'pdf.fonttype': 42,
                                'xtick.direction': 'in',
                                'ytick.major.width': 1.0,
                                'xtick.major.width': 1.0,
                            })

            dfpp = []
            dfpp_melt = []
            for i in range(len(dyad_ief)):
                dfpp.append(pd.DataFrame({
                    'Dyad': ief_3sec_datas[0][i],
                    'Triad': ief_3sec_datas[1][i],
                    'Tetrad': ief_3sec_datas[2][i],
                })
                )

                dfpp_melt.append(pd.melt(dfpp[i]))
                dfpp_melt[i]['Group'] = 'Group' + str(i + 1)

            df = pd.concat([i for i in dfpp_melt], axis=0)

            xlabel = 'Group size'
            ylabel = 'IEF (Avg)'

            df.rename(columns={'variable': xlabel, 'value': ylabel}, inplace=True)

            ax = sns.boxplot(x=xlabel, y=ylabel, data=df, sym="")
            # sns.stripplot(x='variable', y='value', data=df, hue='Group', dodge=True,
            #               jitter=0.2, color='black', palette='Paired', ax=plot[j])

            plt.legend = None
            plt.ylabel('IEF (Avg)')
            # plot[j].axes.xaxis.set_visible(False)
            plt.ylim(0.0, 8.0)
            if source == 'model':
                plt.ylim(0.0, 1.4)

            pairs = [
                {size[1], size[2]},
                {size[0], size[1]},
                {size[0], size[2]},
            ]

            combine.t_test_multi(self, ax=ax, pairs=pairs, data=df, x=xlabel, y=ylabel)
            # combine.anova(self, data=df, variable=xlabel, value=ylabel)

            plt.tight_layout()
            os.makedirs('fig/IEF', exist_ok=True)
            if source == 'human':
                plt.savefig('fig/IEF/Compare_IEF_3sec_combine.png')
            else:
                plt.savefig('fig/IEF/Compare_IEF_3sec_combine_model.png')
            plt.show()

        return ief_3sec_datas

    def ief_3sec_combine_diff(self, graph=True):
        dyad_ief = self.dyad_cfo.get_ief_combine_3sec_diff()
        triad_ief = self.triad_cfo.get_ief_combine_3sec_diff()
        tetrad_ief = self.tetrad_cfo.get_ief_combine_3sec_diff()

        ief_3sec_datas = [
            dyad_ief, triad_ief, tetrad_ief,
        ]

        if graph == True:
            sns.set()
            # sns.set_style('whitegrid')
            sns.set_palette('Set3')

            size = ['Dyad', 'Triad', 'Tetrad']

            plt.figure(figsize=(10, 10), dpi=150)

            sns.set(font='Times New Roman', font_scale=1.0)
            sns.set_style('ticks')
            sns.set_context("paper",
                            rc={
                                "axes.linewidth": 0.5,
                                "legend.fancybox": False,
                                'pdf.fonttype': 42,
                                'xtick.direction': 'in',
                                'ytick.major.width': 1.0,
                                'xtick.major.width': 1.0,
                            })

            dfpp = []
            dfpp_melt = []
            for i in range(len(dyad_ief)):
                dfpp.append(pd.DataFrame({
                    'Dyad': ief_3sec_datas[0][i],
                    'Triad': ief_3sec_datas[1][i],
                    'Tetrad': ief_3sec_datas[2][i],
                })
                )

                dfpp_melt.append(pd.melt(dfpp[i]))
                dfpp_melt[i]['Group'] = 'Group' + str(i + 1)

            df = pd.concat([i for i in dfpp_melt], axis=0)

            xlabel = 'Group size'
            ylabel = 'Cooperative IEF (Avg)'

            df.rename(columns={'variable': xlabel, 'value': ylabel}, inplace=True)

            ax = sns.boxplot(x=xlabel, y=ylabel, data=df, sym="")
            # sns.stripplot(x='variable', y='value', data=df, hue='Group', dodge=True,
            #               jitter=0.2, color='black', palette='Paired', ax=plot[j])

            plt.legend = None
            plt.ylabel('Cooperative IEF (Avg)')
            # plot[j].axes.xaxis.set_visible(False)
            plt.ylim(-1.0, 6.0)

            pairs = [
                {size[1], size[2]},
                {size[0], size[1]},
                {size[0], size[2]},
            ]

            combine.t_test_multi(self, ax=ax, pairs=pairs, data=df, x=xlabel, y=ylabel)
            # combine.anova(self, data=df, variable=xlabel, value=ylabel)

            plt.tight_layout()
            os.makedirs('fig/IEF', exist_ok=True)
            plt.savefig('fig/IEF/Compare_Cooperative_IEF_3sec_combine.png')
            plt.show()

        return ief_3sec_datas

    def t_test(self, ax, pairs, data, x="variable", y="value", test='t-test_ind'):
        annotator = Annotator(ax, pairs, x=x, y=y, data=data, sym="")
        annotator.configure(test=test, text_format='star', loc='inside')
        annotator.apply_and_annotate()

    def t_test_multi(self, ax, pairs, data, x="variable", y="value",
                     test='t-test_ind', comparisons_correction="Bonferroni"):
        # comparisons_correction="BH", "Bonferroni"

        annotator = Annotator(ax, pairs, x=x, y=y, data=data, sym="")
        annotator.configure(test=test, text_format='star', loc='inside')
        # annotator.apply_and_annotate()

        annotator.new_plot(ax=ax, x=x, y=y, data=data)
        annotator.configure(comparisons_correction=comparisons_correction, verbose=2)  # 補正
        test_results = annotator.apply_test().annotate()

    def anova(self, data, variable='variable', value='value'):
        formula = value + ' ~ ' + variable
        model = ols(formula=formula, data=data).fit()
        model.summary()
        anova = sm.stats.anova_lm(model, typ=1)
        print(anova)

        ## Partial Eta_squared
        partial_eta_squared = anova['sum_sq'][0] / (anova['sum_sq'][0] + anova['sum_sq'][1])
        print(f"Partial Eta squared: {partial_eta_squared}")

        ## Eta_squared
        n_groups = len(anova.index)
        ss_treatment = anova['sum_sq'][0]
        ss_total = anova['sum_sq'][n_groups]
        eta_squared = ss_treatment / ss_total
        print(f"Eta squared: {eta_squared}")
