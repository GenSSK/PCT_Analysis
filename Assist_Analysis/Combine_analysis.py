import os

import matplotlib.pyplot as plt
from matplotlib import cbook, cm
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D
from matplotlib import colors
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
from scipy.interpolate import griddata
# from minepy import MINE
import CFO_Analysis_2gen.CFO_analysis as CFO_analysis
from scipy.spatial.distance import correlation
from statannotations.Annotator import Annotator
from statsmodels.formula.api import ols
import statsmodels.api as sm
import pickle
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


def plot_scatter(x, y, color, **kwargs):
    ax = plt.gca()
    ax.scatter(x, y, c=color, **kwargs)


class combine:
    def __init__(self, dyad_npz, triad_npz, tetrad_npz, trajectory_type):
        self.dyad_cfo: CFO_analysis.CFO = CFO_analysis.CFO(dyad_npz, 'dyad', trajectory_type)
        self.triad_cfo: CFO_analysis.CFO = CFO_analysis.CFO(triad_npz, 'triad', trajectory_type)
        self.tetrad_cfo: CFO_analysis.CFO = CFO_analysis.CFO(tetrad_npz, 'tetrad', trajectory_type)

        self.trajectory_type = trajectory_type

        self.trajectory_dir = ""
        if str(self.trajectory_type) == 'Circle':
            self.trajectory_dir = 'Circle/'
        elif str(self.trajectory_type) == 'Lemniscate':
            self.trajectory_dir = 'Lemniscate/'
        elif str(self.trajectory_type) == 'RoseCurve':
            self.trajectory_dir = 'RoseCurve/'
        elif str(self.trajectory_type) == 'Random':
            self.trajectory_dir = 'Random/'
        elif str(self.trajectory_type) == 'Discrete_Random':
            self.trajectory_dir = 'DiscreteRandom/'

        # plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
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
        plt.rcParams['figure.subplot.bottom'] = 0.15
        plt.rcParams['figure.subplot.left'] = 0.15
        # plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['pdf.fonttype'] = 42  # PDFにフォントを埋め込むためのパラメータ

        # パスチャンクの最大サイズを設定
        plt.rcParams['agg.path.chunksize'] = 10000

        # パスの単純化閾値を設定（必要に応じて調整）
        plt.rcParams['path.simplify_threshold'] = 0.1

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

                mystat.t_test_multi(ax, pairs, df, x=xlabel, y=ylabel, test='t-test_ind', )
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

                mystat.t_test_multi(ax, pairs, df, x=xlabel, y=ylabel, test='t-test_ind', )
                mystat.anova(df, variable=xlabel, value=ylabel)

            if mode == 'no_abs':
                os.makedirs('fig/CFO/Summation/NoABS/Comparison/Combine/', exist_ok=True)
                plt.savefig('fig/CFO/Summation/NoABS/Comparison/Combine/SummationCFO_NoABS_3sec_combine_comparison.png')
            elif mode == 'b_abs':
                os.makedirs('fig/CFO/Summation/BeforeABS/Comparison/Combine/', exist_ok=True)
                plt.savefig(
                    'fig/CFO/Summation/BeforeABS/Comparison/Combine/SummationCFO_BeforeABS_3sec_combine_comparison.png')
            elif mode == 'a_abs':
                os.makedirs('fig/CFO/Summation/AfterABS/Comparison/Combine/', exist_ok=True)
                plt.savefig(
                    'fig/CFO/Summation/AfterABS/Comparison/Combine/SummationCFO_AfterABS_3sec_combine_comparison.png')
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

                mystat.t_test_multi(ax, pairs, df, x=xlabel, y=ylabel, test='t-test_ind', )
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

                mystat.t_test_multi(ax, pairs, df, x=xlabel, y=ylabel, test='t-test_ind', )
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
        performance_type = ['Cooperative Error (m)', 'Cooperative Time (s/DT)']
        labels = ['Error', 'Time']
        ylim = [[-0.05, 0.05], [-2.0, 2.0]]

        fig, axs = plt.subplots(3, 2, figsize=(12, 7), dpi=150, sharex=True)
        axs[0, 0].text(1.10, 1.20, self.trajectory_type, ha='center', transform=axs[0, 0].transAxes, fontsize=16)
        for i, p in enumerate(performance):
            for j in range(len(types)):
                axs[j, i].title.set_text(types[j])
                for k, p_ in enumerate(p[j]):
                    axs[j, i].plot(np.arange(1, len(p_) + 1, 1), p_, label='Group' + str(k + 1))
                    axs[j, i].scatter(np.arange(1, len(p_) + 1, 1), p_, s=10, marker='x')
                axs[j, i].legend(ncol=10, columnspacing=1)
                axs[j, i].set_ylim(ylim[i][0], ylim[i][1])

            axs[1, i].set_ylabel(performance_type[i])
            axs[2, i].set_xlabel('Period')
            axs[2, i].set_xticks(np.arange(1, len(error_period_dyad[0]) + 1, 2))
            axs[2, i].set_xlim(0, len(error_period_dyad[0]) + 1)
        os.makedirs('fig/Performance/Period/', exist_ok=True)
        plt.savefig('fig/Performance/Period/Performance_Period_Cooperation_' + self.trajectory_type + '.png')
        # plt.show()

    def time_series_performance_show(self, sigma: int = 'none'):
        error_ts_dyad, error_dot_ts_dyad = self.dyad_cfo.time_series_performance_cooperation(sigma=sigma)
        error_ts_triad, error_dot_ts_triad = self.triad_cfo.time_series_performance_cooperation(sigma=sigma)
        error_ts_tetrad, error_dot_ts_tetrad = self.tetrad_cfo.time_series_performance_cooperation(sigma=sigma)

        error_ts = [error_ts_dyad, error_ts_triad, error_ts_tetrad]
        error_dot_ts = [error_dot_ts_dyad, error_dot_ts_triad, error_dot_ts_tetrad]
        performance = [error_ts, error_dot_ts]

        types = ['dyad', 'triad', 'tetrad']
        performance_type = ['Cooperative Error (m)', 'Cooperative Error Speed (m/$^2$)']
        ylim = [[-0.05, 0.05], [-0.05, 0.05]]

        time = self.dyad_cfo.get_time()
        start_time = self.dyad_cfo.get_starttime()
        end_time = self.dyad_cfo.get_endtime()

        fig, axs = plt.subplots(3, 2, figsize=(12, 7), dpi=150, sharex=True)
        axs[0, 0].text(1.10, 1.20, self.trajectory_type, ha='center', transform=axs[0, 0].transAxes, fontsize=16)
        for i, p in enumerate(performance):
            for j in range(len(types)):
                axs[j, i].title.set_text(types[j])
                for k, p_ in enumerate(p[j]):
                    axs[j, i].plot(time[::10], p_[::10], label='Group' + str(k + 1))
                axs[j, i].legend(ncol=10, columnspacing=1)
                axs[j, i].set_ylim(ylim[i][0], ylim[i][1])

            axs[1, i].set_ylabel(performance_type[i])
            axs[2, i].set_xlabel('Time (s)')
            axs[2, i].set_xlim([start_time, end_time])
        os.makedirs('fig/Performance/TimeSeries/', exist_ok=True)
        plt.savefig('fig/Performance/TimeSeries/Performance_TimeSeries_Cooperation_' + self.trajectory_type + '.png')
        # plt.show()

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
        ylim = [[[-0.02, 0.03], [-0.2, 0.8]],  # h-m
                [[0.01, 0.06], [1.4, 3.0]],  # h-h
                [[0.01, 0.06], [1.4, 3.0]]]  # m-m

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
                ax[j].bar(edge, freq, width=step, align='edge', label="Counts", color="gray", edgecolor="black",
                          linewidth=0.2)
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

    def time_series_performance_subtraction_ave_cfo(self, graph=False, sigma: int = 'none', dec=1):
        dyad_p, dyad_f, dyad_pa, dyad_fa = self.dyad_cfo.subtraction_ave_cfo()
        triad_p, triad_f, triad_pa, triad_fa = self.triad_cfo.subtraction_ave_cfo()
        tetrad_p, tetrad_f, tetrad_pa, tetrad_fa = self.tetrad_cfo.subtraction_ave_cfo()
        summation_3sec_datas = [
            [dyad_p, triad_p, tetrad_p],
            [dyad_f, triad_f, tetrad_f],
            [dyad_pa, triad_pa, tetrad_pa],
            [dyad_fa, triad_fa, tetrad_fa],
        ]

        error_ts_dyad, error_dot_ts_dyad = self.dyad_cfo.time_series_performance_cooperation(sigma=sigma)
        error_ts_triad, error_dot_ts_triad = self.triad_cfo.time_series_performance_cooperation(sigma=sigma)
        error_ts_tetrad, error_dot_ts_tetrad = self.tetrad_cfo.time_series_performance_cooperation(sigma=sigma)

        performance = [
            [error_ts_dyad, error_dot_ts_dyad],
            [error_ts_triad, error_dot_ts_triad],
            [error_ts_tetrad, error_dot_ts_tetrad],
        ]

        size = ['Dyad', 'Triad', 'Tetrad']

        performance_label = ['Error', 'Speed Error']

        types = ['Subtraction PCFO (Avg)',
                 'Subtraction FCFO (Avg)',
                 'Subtraction abs. PCFO (Avg)',
                 'Subtraction abs. FCFO (Avg)']
        ranges = [0.06, 0.5, 0.3, 4.0]

        df_ = []
        for i in range(len(size)):
            for j in range(len(performance_label)):
                for k in range(len(performance[i][j])):
                    for l in range(len(summation_3sec_datas)):
                        df_.append(pd.DataFrame({
                            'CFO': summation_3sec_datas[l][i][k][::dec],
                            'CFO_type': types[l],
                            'Performance': performance[i][j][k][::dec],
                            'Performance_type': performance_label[j],
                            'Group Size': size[i],
                            'Group': str(k + 1)
                        })
                        )

        df = pd.concat([i for i in df_], axis=0)
        df.reset_index(drop=True, inplace=True)

        if graph:
            kwargs = dict(
                s=0.02,
                alpha=0.6,
            )

            # FacetGridを作成してグラフを設定
            # g = sns.FacetGrid(df, col="Performance_type", row="CFO_type", hue="Group Size", height=4, aspect=1.2, sharey=False,
            #                   sharex=False, )
            # g.map(plot_scatter, "Performance", "CFO", **kwargs)
            # df = df[df['CFO'] > 0.01]
            df = df[~df['CFO'].between(-0.01, 0.01)]

            sns.lmplot(data=df, x='CFO', y='Performance', hue='Group Size', col='Performance_type', row='CFO_type',
                       scatter_kws={'s': 0.05, 'alpha': 0.1}, height=4, aspect=1.2, order=5,
                       sharey=False, sharex=False, line_kws={'lw': 2})

            ylims = [(0, 1)]

            xlims = [(-0.015, 0.015), (-0.1, 0.5),
                     (-0.015, 0.015), (-0.1, 0.5),
                     (-0.015, 0.015), (-0.1, 0.5)]

            # Iterate over the AxesSubplots in the FacetGrid object, and set the y limit individually
            # for num, ax in enumerate(g.axes.flatten()):
            #     # col_var = ax.get_title().split(" ")[-1]  # get the column variable from the title
            #     ax.set_xlim(xlims[num])
            #     ax.set_ylim(ylims[0])

            # base_dir = 'fig/CFO-Performance/TimeSeries/Subtraction/Combine/'
            # os.makedirs(base_dir, exist_ok=True)
            # plt.savefig(base_dir + 'SubtractionCFO-Performance_Combine_' + self.trajectory_type + '.png')
            plt.show()

        return df


    def lrm_size(self, sigma: int = 'none'):
        dyad_p_tot, dyad_f_tot, dyad_pa_tot, dyad_fa_tot = self.dyad_cfo.summation_ave_cfo(mode='b_abs')
        triad_p_tot, triad_f_tot, triad_pa_tot, triad_fa_tot = self.triad_cfo.summation_ave_cfo(mode='b_abs')
        tetrad_p_tot, tetrad_f_tot, tetrad_pa_tot, tetrad_fa_tot = self.tetrad_cfo.summation_ave_cfo(mode='b_abs')

        dyad_p_sum, dyad_f_sum, dyad_pa_sum, dyad_fa_sum = self.dyad_cfo.summation_ave_cfo(mode='a_abs')
        triad_p_sum, triad_f_sum, triad_pa_sum, triad_fa_sum = self.triad_cfo.summation_ave_cfo(mode='a_abs')
        tetrad_p_sum, tetrad_f_sum, tetrad_pa_sum, tetrad_fa_sum = self.tetrad_cfo.summation_ave_cfo(mode='a_abs')

        dyad_p_sub, dyad_f_sub, dyad_pa_sub, dyad_fa_sub = self.dyad_cfo.subtraction_ave_cfo()
        triad_p_sub, triad_f_sub, triad_pa_sub, triad_fa_sub = self.triad_cfo.subtraction_ave_cfo()
        tetrad_p_sub, tetrad_f_sub, tetrad_pa_sub, tetrad_fa_sub = self.tetrad_cfo.subtraction_ave_cfo()

        error_ts_dyad, error_dot_ts_dyad = self.dyad_cfo.time_series_performance_cooperation(sigma=sigma)
        error_ts_triad, error_dot_ts_triad = self.triad_cfo.time_series_performance_cooperation(sigma=sigma)
        error_ts_tetrad, error_dot_ts_tetrad = self.tetrad_cfo.time_series_performance_cooperation(sigma=sigma)

        performance = [
            error_ts_dyad,
            error_ts_triad,
            error_ts_tetrad,
        ]

        size = ['Dyad', 'Triad', 'Tetrad']

        cfo_f = [
            [dyad_f_tot, triad_f_tot, tetrad_f_tot],
            [dyad_f_sum, triad_f_sum, tetrad_f_sum],
            [dyad_f_sub, triad_f_sub, tetrad_f_sub],
        ]

        cfo_p = [
            [dyad_p_tot, triad_p_tot, tetrad_p_tot],
            [dyad_p_sum, triad_p_sum, tetrad_p_sum],
            [dyad_p_sub, triad_p_sub, tetrad_p_sub],
        ]

        cfo_fa = [
            [dyad_fa_tot, triad_fa_tot, tetrad_fa_tot],
            [dyad_fa_sum, triad_fa_sum, tetrad_fa_sum],
            [dyad_fa_sub, triad_fa_sub, tetrad_fa_sub],
        ]

        cfo_pa = [
            [dyad_pa_tot, triad_pa_tot, tetrad_pa_tot],
            [dyad_pa_sum, triad_pa_sum, tetrad_pa_sum],
            [dyad_pa_sub, triad_pa_sub, tetrad_pa_sub],
        ]

        cfo_labels = ['Total FCFO', 'Summation FCFO', 'Subtraction FCFO']
        input_labels = ['$\sigma_F^{tot}$', '$\sigma_F^{sum}$', '$\sigma_F^{sub}$',
                        '$\sigma_P^{tot}$', '$\sigma_P^{sum}$', '$\sigma_P^{sub}$',
                        # '$\sigma_{FA}^{tot}$', '$\sigma_{FA}^{sum}$', '$\sigma_{FA}^{sub}$',
                        # '$\sigma_{PA}^{tot}$', '$\sigma_{PA}^{sum}$', '$\sigma_{PA}^{sub}$'
                        ]

        feature_importances = []
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200)
        for i in range(len(size)):
            cfo_f_tot_re = cfo_f[0][i].reshape(1, -1)
            cfo_f_sum_re = cfo_f[1][i].reshape(1, -1)
            cfo_f_sub_re = cfo_f[2][i].reshape(1, -1)
            cfo_p_tot_re = cfo_p[0][i].reshape(1, -1)
            cfo_p_sum_re = cfo_p[1][i].reshape(1, -1)
            cfo_p_sub_re = cfo_p[2][i].reshape(1, -1)
            cfo_fa_tot_re = cfo_fa[0][i].reshape(1, -1)
            cfo_fa_sum_re = cfo_fa[1][i].reshape(1, -1)
            cfo_fa_sub_re = cfo_fa[2][i].reshape(1, -1)
            cfo_pa_tot_re = cfo_pa[0][i].reshape(1, -1)
            cfo_pa_sum_re = cfo_pa[1][i].reshape(1, -1)
            cfo_pa_sub_re = cfo_pa[2][i].reshape(1, -1)

            performance_re = performance[i].reshape(1, -1)

            input = np.column_stack((cfo_f_tot_re[0],
                                     cfo_f_sum_re[0],
                                     cfo_f_sub_re[0],
                                     cfo_p_tot_re[0],
                                     cfo_p_sum_re[0],
                                     cfo_p_sub_re[0],
                                     # cfo_fa_tot_re[0],
                                     # cfo_fa_sum_re[0],
                                     # cfo_fa_sub_re[0],
                                     # cfo_pa_tot_re[0],
                                     # cfo_pa_sum_re[0],
                                     # cfo_pa_sub_re[0],
                                     cfo_f_tot_re[0] * cfo_f_sum_re[0],
                                     cfo_f_tot_re[0] * cfo_f_sub_re[0],
                                     cfo_f_tot_re[0] * cfo_p_tot_re[0],
                                     cfo_f_tot_re[0] * cfo_p_sum_re[0],
                                     cfo_f_tot_re[0] * cfo_p_sub_re[0],

                                     cfo_f_sum_re[0] * cfo_f_sub_re[0],
                                     cfo_f_sum_re[0] * cfo_p_tot_re[0],
                                     cfo_f_sum_re[0] * cfo_p_sum_re[0],
                                     cfo_f_sum_re[0] * cfo_p_sub_re[0],

                                     cfo_f_sub_re[0] * cfo_p_tot_re[0],
                                     cfo_f_sub_re[0] * cfo_p_sum_re[0],
                                     cfo_f_sub_re[0] * cfo_p_sub_re[0],

                                     cfo_p_tot_re[0] * cfo_p_sum_re[0],
                                     cfo_p_tot_re[0] * cfo_p_sub_re[0],

                                     cfo_p_sum_re[0] * cfo_p_sub_re[0],

                                     ))

            # 訓練データとテストデータに分割
            X_train, X_test, y_train, y_test = train_test_split(input[::100], performance_re[0][::100], test_size=0.2,
                                                                random_state=42, shuffle=True)


            # 説明変数行列に定数項を追加
            X = sm.add_constant(X_train)

            # 回帰モデルの作成
            model = sm.OLS(y_train, X).fit()
            print(model.summary())

            # モデルをファイルに保存
            os.makedirs('lrm/' + self.trajectory_dir, exist_ok=True)
            joblib.dump(model, 'lrm/' + self.trajectory_dir + 'lrm_model_' + size[i] + '.pkl')

            # 保存されたモデルをロード
            model: RandomForestRegressor = joblib.load(
                'lrm/' + self.trajectory_dir + 'lrm_model_' + size[i] + '.pkl')


            X_ = sm.add_constant(X_test)

            predicted_values = model.predict(X_)
            # sns.regplot(x=predicted_values[::100], y=performance_re[0][::100], ax=ax,
            #             scatter_kws={'s': 0.05, 'alpha': 0.1}, line_kws={'lw': 2},
            #             )
            ax.scatter(predicted_values, y_test, s=0.05, alpha=0.2)

            # r2 = np.corrcoef(predicted_values, performance_re[0])
            ax.text(0.02, 0.89-i*0.05, size[i]+': $R^2 = {:.2f}$'.format(model.rsquared), horizontalalignment='left',
                    transform=ax.transAxes, fontsize="small")

            ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='black', linestyle='--')
            ax.set_xlabel('Predicted Cooperative RMSE')
            ax.set_ylabel('Cooperative RMSE')
            ax.set_ylim(-0.03, 0.03)
            ax.set_xlim(-0.03, 0.03)

        plt.subplots_adjust(wspace=1.0)  # 横方向の余白を調整
        os.makedirs('fig/LRM/' + self.trajectory_dir, exist_ok=True)
        plt.savefig('fig/LRM/' + self.trajectory_dir + 'performance_predict_LRM_size.png')


    def lrm_check(self, sigma: int = 'none', mode: str = 'all'):
        if mode not in ['all', 'size', 'each']:
            raise ValueError('mode must be all or size or each')

        time = self.dyad_cfo.get_time()

        dyad_p_tot, dyad_f_tot, dyad_pa_tot, dyad_fa_tot = self.dyad_cfo.summation_ave_cfo(mode='b_abs')
        triad_p_tot, triad_f_tot, triad_pa_tot, triad_fa_tot = self.triad_cfo.summation_ave_cfo(mode='b_abs')
        tetrad_p_tot, tetrad_f_tot, tetrad_pa_tot, tetrad_fa_tot = self.tetrad_cfo.summation_ave_cfo(mode='b_abs')

        dyad_p_sum, dyad_f_sum, dyad_pa_sum, dyad_fa_sum = self.dyad_cfo.summation_ave_cfo(mode='a_abs')
        triad_p_sum, triad_f_sum, triad_pa_sum, triad_fa_sum = self.triad_cfo.summation_ave_cfo(mode='a_abs')
        tetrad_p_sum, tetrad_f_sum, tetrad_pa_sum, tetrad_fa_sum = self.tetrad_cfo.summation_ave_cfo(mode='a_abs')

        dyad_p_sub, dyad_f_sub, dyad_pa_sub, dyad_fa_sub = self.dyad_cfo.subtraction_ave_cfo()
        triad_p_sub, triad_f_sub, triad_pa_sub, triad_fa_sub = self.triad_cfo.subtraction_ave_cfo()
        tetrad_p_sub, tetrad_f_sub, tetrad_pa_sub, tetrad_fa_sub = self.tetrad_cfo.subtraction_ave_cfo()

        error_ts_dyad, error_dot_ts_dyad = self.dyad_cfo.time_series_performance_cooperation(sigma=sigma)
        error_ts_triad, error_dot_ts_triad = self.triad_cfo.time_series_performance_cooperation(sigma=sigma)
        error_ts_tetrad, error_dot_ts_tetrad = self.tetrad_cfo.time_series_performance_cooperation(sigma=sigma)

        performance = [
            error_ts_dyad,
            error_ts_triad,
            error_ts_tetrad,
        ]

        size = ['Dyad', 'Triad', 'Tetrad']

        cfo_f = [
            [dyad_f_tot, triad_f_tot, tetrad_f_tot],
            [dyad_f_sum, triad_f_sum, tetrad_f_sum],
            [dyad_f_sub, triad_f_sub, tetrad_f_sub],
        ]

        cfo_p = [
            [dyad_p_tot, triad_p_tot, tetrad_p_tot],
            [dyad_p_sum, triad_p_sum, tetrad_p_sum],
            [dyad_p_sub, triad_p_sub, tetrad_p_sub],
        ]

        cfo_fa = [
            [dyad_fa_tot, triad_fa_tot, tetrad_fa_tot],
            [dyad_fa_sum, triad_fa_sum, tetrad_fa_sum],
            [dyad_fa_sub, triad_fa_sub, tetrad_fa_sub],
        ]

        cfo_pa = [
            [dyad_pa_tot, triad_pa_tot, tetrad_pa_tot],
            [dyad_pa_sum, triad_pa_sum, tetrad_pa_sum],
            [dyad_pa_sub, triad_pa_sub, tetrad_pa_sub],
        ]

        model: sm.OLS
        if mode == 'all':
            model: sm.OLS = joblib.load(
                'lrm/' + self.trajectory_dir + 'lrm_model_all.pkl')

        for i in range(3):
            if mode == 'size':
                model: sm.OLS = joblib.load(
                    'lrm/' + self.trajectory_dir + 'lrm_model_' + size[i] + '.pkl')
            fig_scat, ax_scat = plt.subplots(1, 1, figsize=(7, 5), dpi=200)
            for j in range(len(cfo_f[0][i])):
                if mode == 'each':
                    model: sm.OLS = joblib.load(
                        'lrm/' + self.trajectory_dir + 'lrm_model_' + size[i] + '_Group' + str(
                            j + 1) + '.pkl')

                fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=200)
                cfo_f_tot_re = cfo_f[0][i][j]
                cfo_f_sum_re = cfo_f[1][i][j]
                cfo_f_sub_re = cfo_f[2][i][j]
                cfo_p_tot_re = cfo_p[0][i][j]
                cfo_p_sum_re = cfo_p[1][i][j]
                cfo_p_sub_re = cfo_p[2][i][j]
                cfo_fa_tot_re = cfo_fa[0][i][j]
                cfo_fa_sum_re = cfo_fa[1][i][j]
                cfo_fa_sub_re = cfo_fa[2][i][j]
                cfo_pa_tot_re = cfo_pa[0][i][j]
                cfo_pa_sum_re = cfo_pa[1][i][j]
                cfo_pa_sub_re = cfo_pa[2][i][j]

                performance_re = performance[i][j]

                input = np.column_stack((
                    cfo_f_tot_re,
                    cfo_f_sum_re,
                    cfo_f_sub_re,
                    cfo_p_tot_re,
                    cfo_p_sum_re,
                    cfo_p_sub_re,
                    # cfo_fa_tot_re,
                    # cfo_fa_sum_re,
                    # cfo_fa_sub_re,
                    # cfo_pa_tot_re,
                    # cfo_pa_sum_re,
                    # cfo_pa_sub_re,
                    cfo_f_tot_re * cfo_f_sum_re,
                    cfo_f_tot_re * cfo_f_sub_re,
                    cfo_f_tot_re * cfo_p_tot_re,
                    cfo_f_tot_re * cfo_p_sum_re,
                    cfo_f_tot_re * cfo_p_sub_re,

                    cfo_f_sum_re * cfo_f_sub_re,
                    cfo_f_sum_re * cfo_p_tot_re,
                    cfo_f_sum_re * cfo_p_sum_re,
                    cfo_f_sum_re * cfo_p_sub_re,

                    cfo_f_sub_re * cfo_p_tot_re,
                    cfo_f_sub_re * cfo_p_sum_re,
                    cfo_f_sub_re * cfo_p_sub_re,

                    cfo_p_tot_re * cfo_p_sum_re,
                    cfo_p_tot_re * cfo_p_sub_re,

                    cfo_p_sum_re * cfo_p_sub_re,
                ))

                X_ = sm.add_constant(input)

                pre_performance = model.predict(X_)

                ax.plot(time[::100], performance_re[::100], label='$e^{coop}$', lw=2)
                ax.plot(time[::100], pre_performance[::100], label='$\hat{e}^{coop}$', linestyle='--', lw=0.5,
                        alpha=0.8)
                ax.set_xlabel('Time (sec)')
                ax.set_ylabel('Cooperative RMSE')
                ax.legend()
                ax.set_title(size[i] + ' Group' + str(j + 1))
                os.makedirs('fig/LRM/' + self.trajectory_dir + 'TimeSeries', exist_ok=True)
                # fig.savefig('fig/random_forest/' + self.trajectory_dir + 'TimeSeries/Performance_predict_' + size[i] + '_Group' + str(j + 1) + '_' + mode + '.png')
                fig.savefig('fig/LRM/' + self.trajectory_dir + 'TimeSeries/Performance_predict_' + size[
                    i] + '_Group' + str(j + 1) + '_' + mode + '.pdf')


                # 訓練データとテストデータに分割
                X_train, X_test, y_train, y_test = train_test_split(input[::100], performance_re[::100], test_size=0.2,
                                                                    random_state=42, shuffle=True)

                X_ = sm.add_constant(X_test)

                # テストデータを用いて予測
                predicted_values = np.zeros(len(y_test))
                for k in range(len(X_)):
                    predicted_values[k] = model.predict(X_[k].reshape(1, -1))

                ax_scat.scatter(predicted_values[::1], y_test[::1], s=3.0, alpha=0.2)

                r2 = mystat.r2_score(predicted_values, y_test)

                ax_scat.text(0.02, 0.89 - j * 0.05, 'Group' + str(j + 1) + ': $R^2 = {:.2f}$'.format(r2),
                             horizontalalignment='left',
                             transform=ax_scat.transAxes, fontsize="small")

                ax_scat.set_xlabel('Predicted Cooperative RMSE')
                ax_scat.set_ylabel('Cooperative RMSE')
                ax_scat.set_ylim(-0.03, 0.03)
                ax_scat.set_xlim(-0.03, 0.03)

            ax_scat.plot([0, 1], [0, 1], transform=ax_scat.transAxes, color='black', linestyle='--')
            fig_scat.subplots_adjust(wspace=1.0)  # 横方向の余白を調整
            os.makedirs('fig/LRM/' + self.trajectory_dir, exist_ok=True)
            # fig_scat.savefig('fig/LRM/' + self.trajectory_dir + 'performance_predict_LRM_group_' + size[i] +'.pdf')
            fig_scat.savefig('fig/LRM/' + self.trajectory_dir + 'performance_predict_LRM_group_' + size[i] +'.png')

    def random_forest_all(self, sigma: int = 'none'):
        dyad_p_tot, dyad_f_tot, dyad_pa_tot, dyad_fa_tot = self.dyad_cfo.summation_ave_cfo(mode='b_abs')
        triad_p_tot, triad_f_tot, triad_pa_tot, triad_fa_tot = self.triad_cfo.summation_ave_cfo(mode='b_abs')
        tetrad_p_tot, tetrad_f_tot, tetrad_pa_tot, tetrad_fa_tot = self.tetrad_cfo.summation_ave_cfo(mode='b_abs')

        dyad_p_sum, dyad_f_sum, dyad_pa_sum, dyad_fa_sum = self.dyad_cfo.summation_ave_cfo(mode='a_abs')
        triad_p_sum, triad_f_sum, triad_pa_sum, triad_fa_sum = self.triad_cfo.summation_ave_cfo(mode='a_abs')
        tetrad_p_sum, tetrad_f_sum, tetrad_pa_sum, tetrad_fa_sum = self.tetrad_cfo.summation_ave_cfo(mode='a_abs')

        dyad_p_sub, dyad_f_sub, dyad_pa_sub, dyad_fa_sub = self.dyad_cfo.subtraction_ave_cfo()
        triad_p_sub, triad_f_sub, triad_pa_sub, triad_fa_sub = self.triad_cfo.subtraction_ave_cfo()
        tetrad_p_sub, tetrad_f_sub, tetrad_pa_sub, tetrad_fa_sub = self.tetrad_cfo.subtraction_ave_cfo()

        error_ts_dyad, error_dot_ts_dyad = self.dyad_cfo.time_series_performance_cooperation(sigma=sigma)
        error_ts_triad, error_dot_ts_triad = self.triad_cfo.time_series_performance_cooperation(sigma=sigma)
        error_ts_tetrad, error_dot_ts_tetrad = self.tetrad_cfo.time_series_performance_cooperation(sigma=sigma)

        performance = np.array([
            np.vstack([error_ts_dyad, error_ts_triad, error_ts_tetrad]),
            np.vstack([error_ts_dyad, error_ts_triad, error_ts_tetrad]),
            np.vstack([error_ts_dyad, error_ts_triad, error_ts_tetrad]),
        ])

        size = ['Dyad', 'Triad', 'Tetrad']

        cfo_f = np.array([
            np.vstack([dyad_f_tot, triad_f_tot, tetrad_f_tot]),
            np.vstack([dyad_f_sum, triad_f_sum, tetrad_f_sum]),
            np.vstack([dyad_f_sub, triad_f_sub, tetrad_f_sub]),
        ])

        cfo_p = np.array([
            np.vstack([dyad_p_tot, triad_p_tot, tetrad_p_tot]),
            np.vstack([dyad_p_sum, triad_p_sum, tetrad_p_sum]),
            np.vstack([dyad_p_sub, triad_p_sub, tetrad_p_sub]),
        ])

        cfo_fa = np.array([
            np.vstack([dyad_fa_tot, triad_fa_tot, tetrad_fa_tot]),
            np.vstack([dyad_fa_sum, triad_fa_sum, tetrad_fa_sum]),
            np.vstack([dyad_fa_sub, triad_fa_sub, tetrad_fa_sub]),
        ])

        cfo_pa = np.array([
            np.vstack([dyad_pa_tot, triad_pa_tot, tetrad_pa_tot]),
            np.vstack([dyad_pa_sum, triad_pa_sum, tetrad_pa_sum]),
            np.vstack([dyad_pa_sub, triad_pa_sub, tetrad_pa_sub]),
        ])
        cfo_f = cfo_f.reshape(3, -1)
        cfo_p = cfo_p.reshape(3, -1)
        cfo_fa = cfo_fa.reshape(3, -1)
        cfo_pa = cfo_pa.reshape(3, -1)
        performance = performance.reshape(3, -1)

        cfo_labels = ['Total FCFO', 'Summation FCFO', 'Subtraction FCFO']
        input_labels = ['$\sigma_F^{tot}$', '$\sigma_F^{sum}$', '$\sigma_F^{sub}$',
                        '$\sigma_P^{tot}$', '$\sigma_P^{sum}$', '$\sigma_P^{sub}$',
                        # '$\sigma_{FA}^{tot}$', '$\sigma_{FA}^{sum}$', '$\sigma_{FA}^{sub}$',
                        # '$\sigma_{PA}^{tot}$', '$\sigma_{PA}^{sum}$', '$\sigma_{PA}^{sub}$'
                        ]

        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200)

        input = np.column_stack((cfo_f[0],
                                 cfo_f[1],
                                 cfo_f[2],
                                 cfo_p[0],
                                 cfo_p[1],
                                 cfo_p[2],
                                 # cfo_fa[0],
                                 # cfo_fa[1],
                                 # cfo_fa[2],
                                 # cfo_pa[0],
                                 # cfo_pa[1],
                                 # cfo_pa[2],
                                 ))

        # 訓練データとテストデータに分割
        X_train, X_test, y_train, y_test = train_test_split(input[::100], performance[0][::100], test_size=0.2,
                                                            random_state=42, shuffle=True)

        # 各入力の最大値を保存
        max_values = np.max(X_train, axis=0)
        joblib.dump(max_values, 'random_forest/' + self.trajectory_dir + 'max_values.pkl')

        # ランダムフォレストモデルの構築
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        # モデルをファイルに保存
        os.makedirs('random_forest/' + self.trajectory_dir, exist_ok=True)
        joblib.dump(model, 'random_forest/' + self.trajectory_dir + 'random_forest_model_all.pkl')

        # 保存されたモデルをロード
        model: RandomForestRegressor = joblib.load(
            'random_forest/' + self.trajectory_dir + 'random_forest_model_all.pkl')

        # # SHAP値の計算
        # explainer = shap.Explainer(model, X_train)
        # shap_values = explainer(X_test[:100])
        # print(shap_values)

        # 特徴量の重要度を取得
        feature_importances = model.feature_importances_

        # テストデータを用いて予測
        # predicted_values = model.predict(X_test)
        predicted_values = np.zeros(len(y_test))
        for j in range(len(X_test)):
            # print(X_test[j].reshape(1, -1).shape)
            predicted_values[j] = model.predict(X_test[j].reshape(1, -1))

        # sns.regplot(x=predicted_values, y=y_test, ax=ax,
        #             scatter_kws={'s': 0.05, 'alpha': 0.1}, line_kws={'lw': 2},
        #             )
        ax.scatter(predicted_values, y_test, s=0.05, alpha=0.2)
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='black', linestyle='--')

        # r = np.corrcoef(predicted_values, y_test)
        # r = np.corrcoef(np.log10(cfo_re[0]), performance_re[0])
        r2 = mystat.r2_score(predicted_values, y_test)

        ax.text(0.02, 0.89, '$R^2 = {:.2f}$'.format(r2), horizontalalignment='left',
                transform=ax.transAxes, fontsize="small")

        ax.set_xlabel('Predicted Cooperative RMSE')
        ax.set_ylabel('Cooperative RMSE')
        ax.set_ylim(-0.03, 0.03)
        ax.set_xlim(-0.03, 0.03)

        # plt.subplots_adjust(wspace=1.0)  # 横方向の余白を調整
        os.makedirs('fig/random_forest/' + self.trajectory_dir, exist_ok=True)
        plt.savefig('fig/random_forest/' + self.trajectory_dir + 'performance_predict_random_forest_all.png')
        # plt.show()

        # 特徴量の重要度を可視化
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=200, sharex=True, sharey=True)

        f = np.zeros((1, len(input_labels)))
        f[0] = feature_importances
        df = pd.DataFrame(f, columns=input_labels)
        sns.barplot(data=df, ax=ax, color='dodgerblue')

        ax.set_ylim(0, 0.5)
        ax.set_ylabel('Feature Importance')
        ax.set_xlabel('Feature')
        # plt.subplots_adjust(wspace=0.4)  # 横方向の余白を調整
        plt.savefig('fig/random_forest/' + self.trajectory_dir + 'feature_importance_random_forest_all.png')
        # plt.show()

    def random_forest_size(self, sigma: int = 'none'):
        dyad_p_tot, dyad_f_tot, dyad_pa_tot, dyad_fa_tot = self.dyad_cfo.summation_ave_cfo(mode='b_abs')
        triad_p_tot, triad_f_tot, triad_pa_tot, triad_fa_tot = self.triad_cfo.summation_ave_cfo(mode='b_abs')
        tetrad_p_tot, tetrad_f_tot, tetrad_pa_tot, tetrad_fa_tot = self.tetrad_cfo.summation_ave_cfo(mode='b_abs')

        dyad_p_sum, dyad_f_sum, dyad_pa_sum, dyad_fa_sum = self.dyad_cfo.summation_ave_cfo(mode='a_abs')
        triad_p_sum, triad_f_sum, triad_pa_sum, triad_fa_sum = self.triad_cfo.summation_ave_cfo(mode='a_abs')
        tetrad_p_sum, tetrad_f_sum, tetrad_pa_sum, tetrad_fa_sum = self.tetrad_cfo.summation_ave_cfo(mode='a_abs')

        dyad_p_sub, dyad_f_sub, dyad_pa_sub, dyad_fa_sub = self.dyad_cfo.subtraction_ave_cfo()
        triad_p_sub, triad_f_sub, triad_pa_sub, triad_fa_sub = self.triad_cfo.subtraction_ave_cfo()
        tetrad_p_sub, tetrad_f_sub, tetrad_pa_sub, tetrad_fa_sub = self.tetrad_cfo.subtraction_ave_cfo()

        error_ts_dyad, error_dot_ts_dyad = self.dyad_cfo.time_series_performance_cooperation(sigma=sigma)
        error_ts_triad, error_dot_ts_triad = self.triad_cfo.time_series_performance_cooperation(sigma=sigma)
        error_ts_tetrad, error_dot_ts_tetrad = self.tetrad_cfo.time_series_performance_cooperation(sigma=sigma)

        performance = [
            error_ts_dyad,
            error_ts_triad,
            error_ts_tetrad,
        ]

        size = ['Dyad', 'Triad', 'Tetrad']

        cfo_f = [
            [dyad_f_tot, triad_f_tot, tetrad_f_tot],
            [dyad_f_sum, triad_f_sum, tetrad_f_sum],
            [dyad_f_sub, triad_f_sub, tetrad_f_sub],
        ]

        cfo_p = [
            [dyad_p_tot, triad_p_tot, tetrad_p_tot],
            [dyad_p_sum, triad_p_sum, tetrad_p_sum],
            [dyad_p_sub, triad_p_sub, tetrad_p_sub],
        ]

        cfo_fa = [
            [dyad_fa_tot, triad_fa_tot, tetrad_fa_tot],
            [dyad_fa_sum, triad_fa_sum, tetrad_fa_sum],
            [dyad_fa_sub, triad_fa_sub, tetrad_fa_sub],
        ]

        cfo_pa = [
            [dyad_pa_tot, triad_pa_tot, tetrad_pa_tot],
            [dyad_pa_sum, triad_pa_sum, tetrad_pa_sum],
            [dyad_pa_sub, triad_pa_sub, tetrad_pa_sub],
        ]

        cfo_labels = ['Total FCFO', 'Summation FCFO', 'Subtraction FCFO']
        input_labels = ['$\sigma_F^{tot}$', '$\sigma_F^{sum}$', '$\sigma_F^{sub}$',
                        '$\sigma_P^{tot}$', '$\sigma_P^{sum}$', '$\sigma_P^{sub}$',
                        # '$\sigma_{FA}^{tot}$', '$\sigma_{FA}^{sum}$', '$\sigma_{FA}^{sub}$',
                        # '$\sigma_{PA}^{tot}$', '$\sigma_{PA}^{sum}$', '$\sigma_{PA}^{sub}$'
                        ]

        feature_importances = []
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200)
        for i in range(len(size)):
            cfo_f_tot_re = cfo_f[0][i].reshape(1, -1)
            cfo_f_sum_re = cfo_f[1][i].reshape(1, -1)
            cfo_f_sub_re = cfo_f[2][i].reshape(1, -1)
            cfo_p_tot_re = cfo_p[0][i].reshape(1, -1)
            cfo_p_sum_re = cfo_p[1][i].reshape(1, -1)
            cfo_p_sub_re = cfo_p[2][i].reshape(1, -1)
            cfo_fa_tot_re = cfo_fa[0][i].reshape(1, -1)
            cfo_fa_sum_re = cfo_fa[1][i].reshape(1, -1)
            cfo_fa_sub_re = cfo_fa[2][i].reshape(1, -1)
            cfo_pa_tot_re = cfo_pa[0][i].reshape(1, -1)
            cfo_pa_sum_re = cfo_pa[1][i].reshape(1, -1)
            cfo_pa_sub_re = cfo_pa[2][i].reshape(1, -1)

            performance_re = performance[i].reshape(1, -1)

            input = np.column_stack((cfo_f_tot_re[0],
                                     cfo_f_sum_re[0],
                                     cfo_f_sub_re[0],
                                     cfo_p_tot_re[0],
                                     cfo_p_sum_re[0],
                                     cfo_p_sub_re[0],
                                     # cfo_fa_tot_re[0],
                                     # cfo_fa_sum_re[0],
                                     # cfo_fa_sub_re[0],
                                     # cfo_pa_tot_re[0],
                                     # cfo_pa_sum_re[0],
                                     # cfo_pa_sub_re[0],
                                     ))

            # 訓練データとテストデータに分割
            X_train, X_test, y_train, y_test = train_test_split(input[::100], performance_re[0][::100], test_size=0.2,
                                                                random_state=42, shuffle=True)

            # 各入力の最大値を保存
            max_values = np.max(X_train, axis=0)
            joblib.dump(max_values, 'random_forest/' + self.trajectory_dir + 'max_values_' + size[i] + '.pkl')

            # ランダムフォレストモデルの構築
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            # モデルをファイルに保存
            os.makedirs('random_forest/' + self.trajectory_dir, exist_ok=True)
            joblib.dump(model, 'random_forest/' + self.trajectory_dir + 'random_forest_model_' + size[i] + '.pkl')

            # 保存されたモデルをロード
            model: RandomForestRegressor = joblib.load(
                'random_forest/' + self.trajectory_dir + 'random_forest_model_' + size[i] + '.pkl')

            # # SHAP値の計算
            # explainer = shap.Explainer(model, X_train)
            # shap_values = explainer(X_test[:100])
            # print(shap_values)

            # 特徴量の重要度を取得
            feature_importances.append(model.feature_importances_)

            # テストデータを用いて予測
            # predicted_values = model.predict(X_test)
            predicted_values = np.zeros(len(y_test))
            for j in range(len(X_test)):
                # print(X_test[j].reshape(1, -1).shape)
                predicted_values[j] = model.predict(X_test[j].reshape(1, -1))

            # sns.regplot(x=predicted_values, y=y_test, ax=ax,
            #             scatter_kws={'s': 0.05, 'alpha': 0.1}, line_kws={'lw': 2},
            #             )
            ax.scatter(predicted_values, y_test, s=0.05, alpha=0.2)
            ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='black', linestyle='--')

            # r = np.corrcoef(predicted_values, y_test)
            # r = np.corrcoef(np.log10(cfo_re[0]), performance_re[0])
            r2 = mystat.r2_score(predicted_values, y_test)

            ax.text(0.02, 0.89 - i * 0.1, size[i] + ': $R^2 = {:.2f}$'.format(r2), horizontalalignment='left',
                    transform=ax.transAxes, fontsize="small")

            ax.set_xlabel('Predicted Cooperative RMSE')
            ax.set_ylabel('Cooperative RMSE')
            ax.set_ylim(-0.03, 0.03)
            ax.set_xlim(-0.03, 0.03)

        plt.subplots_adjust(wspace=1.0)  # 横方向の余白を調整
        os.makedirs('fig/random_forest/' + self.trajectory_dir, exist_ok=True)
        plt.savefig('fig/random_forest/' + self.trajectory_dir + 'performance_predict_random_forest_size.png')

        # 特徴量の重要度を可視化
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=200, sharex=True, sharey=True)
        df_ = []
        for i in range(len(size)):
            f = np.zeros((1, len(input_labels)))
            f[0] = feature_importances[i]
            df_.append(pd.DataFrame(f, columns=input_labels, index=[size[i]]))
        df = pd.concat([i for i in df_], axis=0)
        # indexを列に移動
        df.reset_index(inplace=True)
        # index列名を設定（任意）
        df.rename(columns={'index': 'Group size'}, inplace=True)
        df = df.melt(id_vars='Group size')
        df.rename(columns={'variable': 'Feature', 'value': 'Feature Importance'}, inplace=True)
        sns.barplot(data=df, x='Feature', y='Feature Importance', hue='Group size', ax=ax)

        # width = 0.9
        # ax.bar(np.arange(len(input_labels)) - width/3, feature_importances[0], tick_label=input_labels, width=width/2, label='dyad')
        # ax.bar(np.arange(len(input_labels)) + width/3, feature_importances[2], tick_label=input_labels, width=width/2, label='tetrad')
        # ax.bar(np.arange(len(input_labels)) , feature_importances[1], tick_label=input_labels, width=width/2, label='triad')
        ax.set_ylim(0, 0.5)
        # ax.set_xlim(-width, len(input_labels) - width/2)
        ax.legend()

        ax.set_ylabel('Feature Importance')
        ax.set_xlabel('Feature')
        plt.subplots_adjust(wspace=0.4)  # 横方向の余白を調整
        plt.savefig('fig/random_forest/' + self.trajectory_dir + 'feature_importance_random_forest_size.png')
        # plt.show()

    def random_forest_each(self, sigma: int = 'none'):
        dyad_p_tot, dyad_f_tot, dyad_pa_tot, dyad_fa_tot = self.dyad_cfo.summation_ave_cfo(mode='b_abs')
        triad_p_tot, triad_f_tot, triad_pa_tot, triad_fa_tot = self.triad_cfo.summation_ave_cfo(mode='b_abs')
        tetrad_p_tot, tetrad_f_tot, tetrad_pa_tot, tetrad_fa_tot = self.tetrad_cfo.summation_ave_cfo(mode='b_abs')

        dyad_p_sum, dyad_f_sum, dyad_pa_sum, dyad_fa_sum = self.dyad_cfo.summation_ave_cfo(mode='a_abs')
        triad_p_sum, triad_f_sum, triad_pa_sum, triad_fa_sum = self.triad_cfo.summation_ave_cfo(mode='a_abs')
        tetrad_p_sum, tetrad_f_sum, tetrad_pa_sum, tetrad_fa_sum = self.tetrad_cfo.summation_ave_cfo(mode='a_abs')

        dyad_p_sub, dyad_f_sub, dyad_pa_sub, dyad_fa_sub = self.dyad_cfo.subtraction_ave_cfo()
        triad_p_sub, triad_f_sub, triad_pa_sub, triad_fa_sub = self.triad_cfo.subtraction_ave_cfo()
        tetrad_p_sub, tetrad_f_sub, tetrad_pa_sub, tetrad_fa_sub = self.tetrad_cfo.subtraction_ave_cfo()

        error_ts_dyad, error_dot_ts_dyad = self.dyad_cfo.time_series_performance_cooperation(sigma=sigma)
        error_ts_triad, error_dot_ts_triad = self.triad_cfo.time_series_performance_cooperation(sigma=sigma)
        error_ts_tetrad, error_dot_ts_tetrad = self.tetrad_cfo.time_series_performance_cooperation(sigma=sigma)

        performance = [
            error_ts_dyad,
            error_ts_triad,
            error_ts_tetrad,
        ]

        size = ['Dyad', 'Triad', 'Tetrad']

        cfo_f = [
            [dyad_f_tot, triad_f_tot, tetrad_f_tot],
            [dyad_f_sum, triad_f_sum, tetrad_f_sum],
            [dyad_f_sub, triad_f_sub, tetrad_f_sub],
        ]

        cfo_p = [
            [dyad_p_tot, triad_p_tot, tetrad_p_tot],
            [dyad_p_sum, triad_p_sum, tetrad_p_sum],
            [dyad_p_sub, triad_p_sub, tetrad_p_sub],
        ]

        cfo_fa = [
            [dyad_fa_tot, triad_fa_tot, tetrad_fa_tot],
            [dyad_fa_sum, triad_fa_sum, tetrad_fa_sum],
            [dyad_fa_sub, triad_fa_sub, tetrad_fa_sub],
        ]

        cfo_pa = [
            [dyad_pa_tot, triad_pa_tot, tetrad_pa_tot],
            [dyad_pa_sum, triad_pa_sum, tetrad_pa_sum],
            [dyad_pa_sub, triad_pa_sub, tetrad_pa_sub],
        ]

        cfo_labels = ['Total FCFO', 'Summation FCFO', 'Subtraction FCFO']
        input_labels = [
            '$\sigma_F^{tot}$', '$\sigma_F^{sum}$', '$\sigma_F^{sub}$',
            '$\sigma_P^{tot}$', '$\sigma_P^{sum}$', '$\sigma_P^{sub}$',
            # '$\sigma_{FA}^{tot}$', '$\sigma_{FA}^{sum}$', '$\sigma_{FA}^{sub}$',
            # '$\sigma_{PA}^{tot}$', '$\sigma_{PA}^{sum}$', '$\sigma_{PA}^{sub}$'
        ]

        feature_importances = []
        fig, axs = plt.subplots(3, 1, figsize=(5, 12), dpi=200, sharex=True, sharey=True)
        for i in range(len(size)):
            ax = axs[i]
            feature_importances.append([])
            for j in range(len(cfo_f[0][i])):
                cfo_f_tot_re = cfo_f[0][i][j]
                cfo_f_sum_re = cfo_f[1][i][j]
                cfo_f_sub_re = cfo_f[2][i][j]
                cfo_p_tot_re = cfo_p[0][i][j]
                cfo_p_sum_re = cfo_p[1][i][j]
                cfo_p_sub_re = cfo_p[2][i][j]
                cfo_fa_tot_re = cfo_fa[0][i][j]
                cfo_fa_sum_re = cfo_fa[1][i][j]
                cfo_fa_sub_re = cfo_fa[2][i][j]
                cfo_pa_tot_re = cfo_pa[0][i][j]
                cfo_pa_sum_re = cfo_pa[1][i][j]
                cfo_pa_sub_re = cfo_pa[2][i][j]

                performance_re = performance[i][j]

                input = np.column_stack((
                    cfo_f_tot_re,
                    cfo_f_sum_re,
                    cfo_f_sub_re,
                    cfo_p_tot_re,
                    cfo_p_sum_re,
                    cfo_p_sub_re,
                    # cfo_fa_tot_re,
                    # cfo_fa_sum_re,
                    # cfo_fa_sub_re,
                    # cfo_pa_tot_re,
                    # cfo_pa_sum_re,
                    # cfo_pa_sub_re,
                ))

                # 訓練データとテストデータに分割
                X_train, X_test, y_train, y_test = train_test_split(input[::100], performance_re[::100], test_size=0.2,
                                                                    random_state=42, shuffle=True)

                # 各入力の最大値を保存
                max_values = np.max(X_train, axis=0)
                joblib.dump(max_values,
                            'random_forest/' + self.trajectory_dir + 'max_values_' + size[i] + '_Group' + str(
                                j + 1) + '.pkl')

                # ランダムフォレストモデルの構築
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                # モデルをファイルに保存
                os.makedirs('random_forest/' + self.trajectory_dir, exist_ok=True)
                joblib.dump(model,
                            'random_forest/' + self.trajectory_dir + 'random_forest_model_' + size[i] + '_Group' + str(
                                j + 1) + '.pkl')

                # 保存されたモデルをロード
                model: RandomForestRegressor = joblib.load(
                    'random_forest/' + self.trajectory_dir + 'random_forest_model_' + size[i] + '_Group' + str(
                        j + 1) + '.pkl')

                # # SHAP値の計算
                # explainer = shap.Explainer(model, X_train)
                # shap_values = explainer(X_test[:100])
                # print(shap_values)

                # 特徴量の重要度を取得
                feature_importances[i].append(model.feature_importances_)

                # テストデータを用いて予測
                # predicted_values = model.predict(X_test)
                predicted_values = np.zeros(len(y_test))
                for k in range(len(X_test)):
                    # print(X_test[k].reshape(1, -1).shape)
                    predicted_values[k] = model.predict(X_test[k].reshape(1, -1))

                ax.scatter(predicted_values, y_test, s=0.05, alpha=0.2, label='Group' + str(j + 1))

                # r = np.corrcoef(predicted_values, y_test)
                # r = np.corrcoef(np.log10(cfo_re[0]), performance_re[0])
                r2 = mystat.r2_score(predicted_values, y_test)
                ax.text(0.02, 0.89 - j * 0.1, 'Group' + str(j + 1) + ': $R^2 = {:.2f}$'.format(r2),
                        horizontalalignment='left',
                        transform=ax.transAxes, fontsize="small")
                ax.set_title(size[i])
            ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='black', linestyle='--')
            ax.set_ylim(-0.03, 0.03)
            ax.set_xlim(-0.03, 0.03)
        axs[2].set_xlabel('Predicted Cooperative RMSE')
        axs[1].set_ylabel('Cooperative RMSE')

        os.makedirs('fig/random_forest/' + self.trajectory_dir, exist_ok=True)
        plt.savefig('fig/random_forest/' + self.trajectory_dir + 'performance_predict_random_forest_each.png')
        plt.savefig('fig/random_forest/' + self.trajectory_dir + 'performance_predict_random_forest_each.pdf')

        # plt.show()

        # 特徴量の重要度を可視化
        fig, axs = plt.subplots(3, 1, figsize=(6, 12), dpi=200, sharex=True, sharey=True)
        for i in range(len(size)):
            ax = axs[i]
            df_ = []
            for j in range(len(cfo_f[0][i])):
                f = np.zeros((1, len(input_labels)))
                f[0] = feature_importances[i][j]
                df_.append(pd.DataFrame(f, columns=input_labels, index=['Group' + str(j + 1)]))

            df = pd.concat([i for i in df_], axis=0)
            df.reset_index(drop=True, inplace=True)

            sns.barplot(data=df, ax=ax, color='dodgerblue',
                        ci='sd', errwidth=1, capsize=0.1)

            ax.set_ylim(0, 0.5)
            ax.set_title(size[i])
        axs[1].set_ylabel('Feature Importance')
        axs[2].set_xlabel('Feature')
        # plt.subplots_adjust(wspace=0.4)  # 横方向の余白を調整
        plt.savefig('fig/random_forest/feature_importance_random_forest_each.png')
        # plt.show()

        # 特徴量の重要度を可視化
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=200, sharex=True, sharey=True)
        df_melt = []
        for i in range(len(size)):
            df__ = []
            for j in range(len(cfo_f[0][i])):
                f = np.zeros((1, len(input_labels)))
                f[0] = feature_importances[i][j]
                df__.append(pd.DataFrame(f, columns=input_labels, index=['Group' + str(j + 1)]))

            df_ = pd.concat([i for i in df__], axis=0)
            # indexを列に移動
            df_.reset_index(inplace=True)
            # index列名を設定（任意）
            df_.rename(columns={'index': 'Group'}, inplace=True)
            df_ = df_.melt(id_vars='Group')
            df_['Group size'] = size[i]
            df_.rename(columns={'variable': 'Feature', 'value': 'Feature Importance'}, inplace=True)
            df_melt.append(df_)

        df = pd.concat([i for i in df_melt], axis=0)
        df.reset_index(drop=True, inplace=True)
        sns.barplot(data=df, x='Feature', y='Feature Importance', hue='Group size', ax=ax,
                    ci='sd', errwidth=1, capsize=0.1)

        ax.set_ylim(0, 0.5)
        ax.set_ylabel('Feature Importance')
        ax.set_xlabel('Feature')
        # plt.subplots_adjust(wspace=0.4)  # 横方向の余白を調整
        plt.savefig('fig/random_forest/' + self.trajectory_dir + 'feature_importance_random_forest_each_comp.png')
        plt.show()

    def simulation_all(self):
        # 各入力の最大値をロード
        max_values = joblib.load('random_forest/' + self.trajectory_dir + 'max_values.pkl')
        model: RandomForestRegressor = joblib.load(
            'random_forest/' + self.trajectory_dir + 'random_forest_model_all.pkl')

        self.simulation(model, max_values, 'All/')

    def simulation_size(self, size: str):
        if size not in ['Dyad', 'Triad', 'Tetrad']:
            raise ValueError('size is not Dyad, Triad, Tetrad')
        # 各入力の最大値をロード
        max_values = joblib.load('random_forest/' + self.trajectory_dir + 'max_values_' + size + '.pkl')
        model: RandomForestRegressor = joblib.load(
            'random_forest/' + self.trajectory_dir + 'random_forest_model_' + size + '.pkl')

        self.simulation(model, max_values, size + '/')

    def simulation(self, model: RandomForestRegressor, max_values: np.ndarray, save_dir):
        rewrite = False
        cfo_labels = ['$\sigma_1^{F}$', '$\sigma_2^{F}$', '$\sigma_1^{P}$', '$\sigma_2^{P}$']
        input_labels = ['$\sigma^{F,tot}$', '$\sigma^{F,sum}$', '$\sigma^{F,sub}$',
                        '$\sigma^{P,tot}$', '$\sigma^{P,sum}$', '$\sigma^{P,sub}$',
                        # '$\sigma_{FA}^{tot}$', '$\sigma_{FA}^{sum}$', '$\sigma_{FA}^{sub}$',
                        # '$\sigma_{PA}^{tot}$', '$\sigma_{PA}^{sum}$', '$\sigma_{PA}^{sub}$'
                        ]
        num = len(input_labels)
        smp = 0.01
        pattern = 3  # fix
        level = 40
        steps = (pattern * level - level + 1)
        counts = steps ** 4
        during_time = 0.01
        time = np.arange(0, during_time * counts, smp)

        f = [1.0] * num
        g_noise = [0.0] * num  # 0.2
        g_triangle = [0.2] * num
        offset = np.full((num, len(time)), 0.0)
        # generate time series random signal

        noise = np.zeros((num, len(time)))
        triangle = np.zeros((num, len(time)))
        # num種類の長さnのランダムノイズを生成
        for _ in range(num):
            noise[_] = (np.random.rand(len(time)) - 0.5) * 2
            triangle[_] = signal.sawtooth(2 * np.pi * f[_] * time, width=0.5)

        cfo_pattern = np.zeros((4, counts))
        for i in range(steps):
            for j in range(steps):
                for k in range(steps):
                    for l in range(steps):
                        cfo_pattern[3][steps ** 3 * i + steps ** 2 * j + steps * k + l] = i
                        cfo_pattern[2][steps ** 3 * i + steps ** 2 * j + steps * k + l] = j
                        cfo_pattern[1][steps ** 3 * i + steps ** 2 * j + steps * k + l] = k
                        cfo_pattern[0][steps ** 3 * i + steps ** 2 * j + steps * k + l] = l

        g_cfo_f = 2.0 / level
        g_cfo_p = 0.2 / level
        g_cfo = [
            g_cfo_f, g_cfo_f, g_cfo_p, g_cfo_p
        ]
        cfo_pattern = (cfo_pattern - level)
        for i in range(4):
            cfo_pattern[i] = cfo_pattern[i] * g_cfo[i]
        cfo = np.repeat(cfo_pattern, during_time / smp, axis=1)

        for i in range(4):
            cfo[i] = cfo[i] + g_noise[i] * g_cfo[i] * noise[i]

        cfo_f = np.array([
            # g * np.sin(2 * np.pi * f * time),
            # g * np.sin(2 * np.pi * f * time),
            # g * np.sin(2 * np.pi * f * time),

            # g_noise * noise[0],
            # g_noise * noise[1],
            # g_noise * noise[2],

            # np.abs(g_triangle[0] * triangle[0]+ g_noise[0] * noise[0] + offset[0]),
            # g_triangle[1] * triangle[1]+ g_noise[1] * noise[1] + offset[1],
            # g_triangle[2] * triangle[2]+ g_noise[2] * noise[2] + offset[2],

            np.abs(cfo[0]) + np.abs(cfo[1]),
            np.abs(cfo[0] + cfo[1]),
            np.abs(cfo[0] - cfo[1]),
        ])

        cfo_p = np.array([
            # g * np.sin(2 * np.pi * f * time),
            # g * np.sin(2 * np.pi * f * time),
            # g * np.sin(2 * np.pi * f * time),

            # g_noise * noise[3],
            # g_noise * noise[4],
            # g_noise * noise[5],

            # np.abs(g_triangle[3] * triangle[3] + g_noise[3] * noise[3] + offset[3]),
            # g_triangle[4] * triangle[4] + g_noise[4] * noise[4] + offset[4],
            # g_triangle[5] * triangle[5] + g_noise[5] * noise[5] + offset[5],

            np.abs(cfo[2]) + np.abs(cfo[3]),
            np.abs(cfo[2] + cfo[3]),
            np.abs(cfo[2] - cfo[3]),
        ])

        input = np.column_stack((
            cfo_f[0],
            cfo_f[1],
            cfo_f[2],
            cfo_p[0],
            cfo_p[1],
            cfo_p[2],
        ))

        for i in range(len(input_labels)):
            cfo = cfo[:, input[:, i] < max_values[i]]
            cfo_p = cfo_p[:, input[:, i] < max_values[i]]
            cfo_f = cfo_f[:, input[:, i] < max_values[i]]
            input = input[input[:, i] < max_values[i]]

        # print(input.shape)
        # print(len(time))
        time = np.arange(0, len(input) * smp, smp)

        # modelを使って予測
        os.makedirs('simulation/' + self.trajectory_dir + save_dir, exist_ok=True)
        sim_name = 'simulation/' + self.trajectory_dir + save_dir + 'sim_performance_' + str(level)
        inp_name = 'simulation/' + self.trajectory_dir + save_dir + 'input_' + str(level)
        if os.path.exists(sim_name) & os.path.exists(inp_name) & (rewrite == False):
            print('Load simulation performance')
            with open(sim_name, 'rb') as f:
                sim_performance = pickle.load(f)
            with open(inp_name, 'rb') as f:
                input = pickle.load(f)
        else:
            print('Simulation')
            sim_performance = model.predict(input)
            with open(sim_name, 'wb') as f:
                pickle.dump(sim_performance, f, protocol=5)
            with open(inp_name, 'wb') as f:
                pickle.dump(input, f, protocol=5)

        counts = np.arange(1, len(sim_performance) + 1, 1)
        # シミュレーション結果の一覧表示
        fig, axs = plt.subplots(1 + len(input_labels) + len(cfo_labels), 1, figsize=(20, 10), dpi=150, sharex=True)
        axs[0].plot(counts, sim_performance)
        axs[0].set_ylabel('Simulated $e^{coop}$')
        axs[0].set_ylim(sim_performance.min() - 0.01, sim_performance.max() + 0.01)
        # # パフォーマンスいいとこに色つける
        # max_performance_ = 100.0
        # max_p_count = 0
        # for i in range(counts):
        #     start_datetime = i * during_time
        #     end_datetime = (i + 1) * during_time
        #     p = np.average(sim_performance[i * int(during_time / smp): (i + 1) * int(during_time / smp)])
        #     if p < -0.0:
        #         if max_performance_> p:
        #             max_p_count = i
        #             max_performance_ = p
        #         for j in range(len(axs)):
        #             axs[j].axvspan(start_datetime, end_datetime, color="red", alpha=0.1)
        # for i in range(len(axs)):
        #     axs[i].axvspan(max_p_count * during_time, (max_p_count + 1) * during_time, color="red", alpha=0.3)


        for i, label in enumerate(input_labels):
            axs[i+1].plot(counts, input[:, i], label=label)
            axs[i+1].set_ylabel(label)
            if i <= 3:
                axs[i+1].set_ylim(input[:, i].min() - 1.0 * g_cfo[0],
                                  input[:, i].max() + 1.0 * g_cfo[0])
            else:
                axs[i+1].set_ylim(input[:, i].min() - 1.0 * g_cfo[2],
                                  input[:, i].max() + 1.0 * g_cfo[2])
        # axs[1].legend(loc='upper right', fontsize='x-small', ncol=100)


        for i, label in enumerate(cfo_labels):
            axs[i+len(input_labels)+1].plot(counts, cfo[i])
            axs[i+len(input_labels)+1].set_ylabel(label)
            # axs[i+len(input_labels)+1].set_yticks(np.arange(-level, level + 1) * g_cfo[i])
            axs[i+len(input_labels)+1].set_ylim((-level - 0.5) * g_cfo[i], (level + 0.5) * g_cfo[i])

        axs[len(axs) - 1].set_xlabel('Time (sec)')

        os.makedirs('fig/simulation/' + self.trajectory_dir + save_dir, exist_ok=True)
        plt.savefig('fig/simulation/' + self.trajectory_dir + save_dir + 'simulation.png')
        # plt.show()
        return 0

        # パフォーマンスと各CFOの関係
        # fig, ax = plt.subplots(1, len(input_labels), figsize=(60, 10), dpi=200)
        # for i, label in enumerate(input_labels):
        #     ax[i].scatter(input[:, i], sim_performance, s=0.1, alpha=0.5)
        #     sns.regplot(x=input[:, i][::100], y=sim_performance[::100], ax=ax[i],
        #                 scatter=False, color='darkorange', order=10, x_ci='ci', ci=95,
        #                 line_kws={'lw': 2})
        #     ax[i].set_ylim(sim_performance.min() - 0.01, sim_performance.max() + 0.01)
        #     ax[i].set_ylabel('Simulated Cooperative RMSE')
        #     ax[i].set_xlabel(input_labels[i])
        #
        # plt.savefig('fig/simulation/' + self.trajectory_dir + save_dir + 'simulation_scatter.png')

        # 各CFOとパフォーマンスの3者関係を3Dプロットで表示
        dec = 1
        x = input[:, 1][::dec]
        y = input[:, 2][::dec]
        xi = np.linspace(min(x), max(x), 100)
        yi = np.linspace(min(y), max(y), 100)
        xi, yi = np.meshgrid(xi, yi)
        z = sim_performance[::dec]
        zi = griddata((x, y), z, (xi, yi), method='cubic')
        fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=200)
        im = ax.imshow(zi, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='gnuplot', aspect='auto')
        fig.colorbar(im, ax=ax, orientation='vertical')
        # fig.colorbar(im, ax=ax, orientation='horizontal')

        # fig, ax = plt.subplots(2, 1, figsize=(10, 20), dpi=100)
        # mappable = ax[0].scatter(x, y, c=z, s=5.0, alpha=1.0, cmap='gnuplot')
        # fig.colorbar(mappable, ax=ax[0], orientation='horizontal')
        # im = ax[1].imshow(zi, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='gnuplot')
        # fig.colorbar(im, ax=ax[1], orientation='horizontal')

        ax.set_xlabel(input_labels[1])
        ax.set_ylabel(input_labels[2])

        plt.savefig('fig/simulation/' + self.trajectory_dir + save_dir + 'simulation_3d_part_' + str(level) + '.pdf')
        plt.show()

        lb = [[1, 2, 3, 4, 5],
              [2, 3, 4, 5],
              [3, 4, 5],
              [4, 5],
              [5]]

        lb = [[5, 4, 3, 2, 1],
              [5, 4, 3, 2, 0],
              [5, 4, 3, 1, 0],
              [5, 4, 2, 1, 0],
              [5, 3, 2, 1, 0]]

        # fig, ax = plt.subplots(len(lb), len(lb[0]), figsize=(20, 20), dpi=200)
        # for i in range(len(input_labels) - 1):
        #     for j in range(len(lb[i])):
        #         x = input[:, i][::dec]
        #         y = input[:, lb[i][j]][::dec]
        #         # y = input[:, j][::dec]
        #         xi = np.linspace(min(x), max(x), 100)
        #         yi = np.linspace(min(y), max(y), 100)
        #         xi, yi = np.meshgrid(xi, yi)
        #         z = sim_performance[::dec]
        #         zi = griddata((x, y), z, (xi, yi), method='cubic')
        #         im = ax[i, j].imshow(zi, extent=(min(x), max(x), min(y), max(y)), origin='lower', cmap='gnuplot', aspect='auto')
        #
        #         # ax[i, j].set_xlabel(input_labels[i])
        #         # ax[i, j].set_ylabel(input_labels[lb[i][j]])
        #         # ax[i, j].set_ylabel(input_labels[j])
        #         ax[i, j].set_xticks((min(x), max(x)))
        #         ax[i, j].set_yticks((min(y), max(y)))
        #         ax[i, j].set_xlim(min(x), max(x))
        #         ax[i, j].set_ylim(min(y), max(y))
        #
        #         print(i, j)

        # # カラーバーを作成し、figの右側に配置
        # cbar = fig.colorbar(im, ax=ax[0,0], orientation='horizontal')
        # # cbar.set_label('Velocity (m/s)')
        # # カラーバーの目盛りの間隔を設定
        # cbar.locator = MaxNLocator(nbins=6)  # 例として5つの目盛りを設定
        # cbar.update_ticks()
        # plt.tight_layout()
        plt.subplots_adjust(wspace=0.4, hspace=0.4)  # 横方向の余白を調整
        plt.savefig('fig/simulation/' + self.trajectory_dir + save_dir + 'simulation_CFO-Performance_color' + str(level) + '.pdf')
        plt.show()


        # fig, ax = plt.subplots(len(input_labels), len(input_labels) - 1, figsize=(20, 20), dpi=200)
        # for i in range(len(input_labels)):
        #     for j in range(len(input_labels) - 1):
        #         x = input[:, i][::dec]
        #         y = input[:, lb[i][j]][::dec]
        #         z = sim_performance[::dec]
        #         im = ax[i, j].scatter(x, y, c=z, s=5.0, alpha=1.0, cmap='gnuplot')
        #
        #         ax[i, j].set_xlabel(input_labels[i])
        #         ax[i, j].set_ylabel(input_labels[lb[i][j]])
        #         ax[i, j].set_xticks((min(x), max(x)))
        #         ax[i, j].set_yticks((min(y), max(y)))
        #         ax[i, j].set_xlim(min(x), max(x))
        #         ax[i, j].set_ylim(min(y), max(y))
        #
        #         print(i, j)
        # plt.subplots_adjust(wspace=0.4, hspace=0.4)  # 横方向の余白を調整
        # plt.savefig('fig/simulation/' + self.trajectory_dir + save_dir + 'simulation_CFO-Performance_scatter' + str(level) + '.png')
        # plt.show()

        # パフォーマンスと各CFOの関係からいいパフォーマンスの部分を選択する
        # # Test 1
        # selected_p = sim_performance[(2.4 < cfo_f[0])]
        # selected_input = input[(2.4 < cfo_f[0])]
        # # Test 2
        # # selected_p = sim_performance[(0.4 < cfo_f[2]) & (cfo_f[2] < 0.6)]
        # # selected_input = input[(0.4 < cfo_f[2]) & (cfo_f[2] < 0.6)]
        # # Test 3
        # # selected_p = sim_performance[(cfo_f[2] < 0.6)]
        # # selected_input = input[(cfo_f[2] < 0.6)]
        #
        # but_p = selected_p[selected_p > 0.0]
        # but_input = selected_input[selected_p > 0.0]
        # good_p = selected_p[selected_p < -0.01]
        # good_input = selected_input[selected_p < -0.01]
        # fig, ax = plt.subplots(1, len(input_labels), figsize=(60, 10), dpi=200)
        # for i, label in enumerate(input_labels):
        #     ax[i].scatter(input[:, i], sim_performance, s=0.1, alpha=0.5)
        #     # ax[i].scatter(selected_input[:, i], selected_p, s=0.1, alpha=0.5, color='red')
        #     ax[i].scatter(but_input[:, i], but_p, s=0.1, alpha=0.5, color='red')
        #     ax[i].scatter(good_input[:, i], good_p, s=0.1, alpha=0.5, color='darkorange')
        #     ax[i].set_ylim(sim_performance.min() - 0.01, sim_performance.max() + 0.01)
        #     ax[i].set_ylabel('Simulated Cooperative RMSE')
        #     ax[i].set_xlabel(input_labels[i])
        # plt.savefig('fig/simulation/' + self.trajectory_dir + save_dir + 'simulation_scatter_selected_#1.png')

        # costを探すためのパフォーマンス探索
        # # Test 1
        # selected_p = sim_performance[2.4 < cfo_f[2]]
        # selected_input = input[2.4 < cfo_f[0]]
        # selected_p_sub = cfo_p[2][2.4 < cfo_f[0]]
        # selected_p = selected_p[(0.09 < selected_p_sub) & (selected_p_sub < 0.115)]
        # selected_input = selected_input[(0.09 < selected_p_sub) & (selected_p_sub < 0.115)]
        # # Test 2
        # # selected_p = sim_performance[(0.4 < cfo_f[2]) & (cfo_f[2] < 0.6)]
        # # selected_input = input[(0.4 < cfo_f[2]) & (cfo_f[2] < 0.6)]
        # # selected_f_sum = cfo_f[1][(0.4 < cfo_f[2]) & (cfo_f[2] < 0.6)]
        # # selected_p = selected_p[(0.4 < selected_f_sum) & (selected_f_sum < 0.6)]
        # # selected_input = selected_input[(0.4 < selected_f_sum) & (selected_f_sum < 0.6)]
        #
        # fig, ax = plt.subplots(1, len(input_labels), figsize=(60, 10), dpi=200)
        # for i, label in enumerate(input_labels):
        #     ax[i].scatter(input[:, i], sim_performance, s=0.1, alpha=0.5)
        #     ax[i].scatter(selected_input[:, i], selected_p, s=0.1, alpha=0.5, color='darkorange')
        #     # ax[i].set_ylim(sim_performance.min() - 0.01, sim_performance.max() + 0.01)
        #     ax[i].set_ylabel('Simulated Cooperative RMSE')
        #     ax[i].set_xlabel(input_labels[i])
        # plt.savefig('fig/simulation/' + self.trajectory_dir + save_dir + 'simulation_scatter_selected_cost_search_#1.png')

        # costの有効性確認（相関）
        # # Test 1
        # cost = np.sqrt(np.abs((np.abs(0.1 - cfo_p[2]) / 0.25) + ((2.6 - cfo_f[0]) / 2.6)))
        # # Test 2
        # # cost = np.sqrt(np.abs(0.5 - cfo_f[1]) * (0.5 - cfo_f[2]))
        #
        # # 3次の多項式でフィット
        # X = np.column_stack((cost**3, cost**2, cost))  # 3次の多項式の基底関数を作成
        # X = sm.add_constant(X)  # 定数項を追加
        # model = sm.OLS(sim_performance, X)  # 最小二乗法でモデル化
        # result = model.fit()
        # print(result.summary())
        # # フィットされたパラメータを取得
        # params = result.params
        # # フィットされたパラメータを用いて予測
        # predicted = params[0] + params[1]*cost**3 + params[2]*cost**2 + params[3]*cost
        # # costで昇順にソートするためのインデックスを取得
        # sorted_indices = np.argsort(cost)
        # # インデックスを使ってcostとpredictedをソート
        # sorted_cost = cost[sorted_indices]
        # sorted_predicted = predicted[sorted_indices]
        #
        # fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=200)
        # ax.scatter(cost[::1], sim_performance[::1], s=0.02, alpha=0.1)
        # # sns.regplot(x=cost[::1], y=sim_performance[::1], ax=ax, scatter=False, color='darkorange', order=3, x_ci='ci', ci=95,
        # #             line_kws={'lw': 2})
        # ax.plot(sorted_cost[::100], sorted_predicted[::100], color='red', lw=2)
        # ax.set_xlabel('Cost')
        # ax.set_ylabel('Simulated Cooperative RMSE')
        # plt.savefig('fig/simulation/' + self.trajectory_dir + save_dir + 'simulation_cost_#1.png')

        # costの分布
        # fig, ax = plt.subplots(1, len(input_labels), figsize=(60, 10), dpi=200)
        # for i, label in enumerate(input_labels):
        #     ax[i].scatter(input[:, i], sim_performance, c=cost, s=0.1, alpha=0.5, cmap='cividis_r')
        #     # ax[i].set_ylim(sim_performance.min() - 0.01, sim_performance.max() + 0.01)
        #     ax[i].set_ylabel('Simulated Cooperative RMSE')
        #     ax[i].set_xlabel(input_labels[i])
        # norm = plt.Normalize(cost.min(), cost.max())
        # cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='cividis_r'), ax=ax.ravel().tolist(), pad=0.025)
        # cbar.ax.set_title("Cost", y=1.01)
        #
        # plt.savefig('fig/simulation/' + self.trajectory_dir + save_dir + 'simulation_scatter_cost_#1.png')
        #
        #
        # fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=200)
        # ax.scatter(predicted[::1], sim_performance[::1], s=0.02, alpha=0.1)
        # ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='black', linestyle='--')
        # ax.set_xlabel('Fitted cost')
        # ax.set_ylabel('Simulated Cooperative RMSE')
        # plt.savefig('fig/simulation/' + self.trajectory_dir + save_dir + 'simulation_cost_fit_#1.png')

        # plt.show()

    def random_forest_check(self, sigma: int = 'none', mode: str = 'all'):
        if mode not in ['all', 'size', 'each']:
            raise ValueError('mode must be all or size or each')

        time = self.dyad_cfo.get_time()

        dyad_p_tot, dyad_f_tot, dyad_pa_tot, dyad_fa_tot = self.dyad_cfo.summation_ave_cfo(mode='b_abs')
        triad_p_tot, triad_f_tot, triad_pa_tot, triad_fa_tot = self.triad_cfo.summation_ave_cfo(mode='b_abs')
        tetrad_p_tot, tetrad_f_tot, tetrad_pa_tot, tetrad_fa_tot = self.tetrad_cfo.summation_ave_cfo(mode='b_abs')

        dyad_p_sum, dyad_f_sum, dyad_pa_sum, dyad_fa_sum = self.dyad_cfo.summation_ave_cfo(mode='a_abs')
        triad_p_sum, triad_f_sum, triad_pa_sum, triad_fa_sum = self.triad_cfo.summation_ave_cfo(mode='a_abs')
        tetrad_p_sum, tetrad_f_sum, tetrad_pa_sum, tetrad_fa_sum = self.tetrad_cfo.summation_ave_cfo(mode='a_abs')

        dyad_p_sub, dyad_f_sub, dyad_pa_sub, dyad_fa_sub = self.dyad_cfo.subtraction_ave_cfo()
        triad_p_sub, triad_f_sub, triad_pa_sub, triad_fa_sub = self.triad_cfo.subtraction_ave_cfo()
        tetrad_p_sub, tetrad_f_sub, tetrad_pa_sub, tetrad_fa_sub = self.tetrad_cfo.subtraction_ave_cfo()

        error_ts_dyad, error_dot_ts_dyad = self.dyad_cfo.time_series_performance_cooperation(sigma=sigma)
        error_ts_triad, error_dot_ts_triad = self.triad_cfo.time_series_performance_cooperation(sigma=sigma)
        error_ts_tetrad, error_dot_ts_tetrad = self.tetrad_cfo.time_series_performance_cooperation(sigma=sigma)

        performance = [
            error_ts_dyad,
            error_ts_triad,
            error_ts_tetrad,
        ]

        size = ['Dyad', 'Triad', 'Tetrad']

        cfo_f = [
            [dyad_f_tot, triad_f_tot, tetrad_f_tot],
            [dyad_f_sum, triad_f_sum, tetrad_f_sum],
            [dyad_f_sub, triad_f_sub, tetrad_f_sub],
        ]

        cfo_p = [
            [dyad_p_tot, triad_p_tot, tetrad_p_tot],
            [dyad_p_sum, triad_p_sum, tetrad_p_sum],
            [dyad_p_sub, triad_p_sub, tetrad_p_sub],
        ]

        cfo_fa = [
            [dyad_fa_tot, triad_fa_tot, tetrad_fa_tot],
            [dyad_fa_sum, triad_fa_sum, tetrad_fa_sum],
            [dyad_fa_sub, triad_fa_sub, tetrad_fa_sub],
        ]

        cfo_pa = [
            [dyad_pa_tot, triad_pa_tot, tetrad_pa_tot],
            [dyad_pa_sum, triad_pa_sum, tetrad_pa_sum],
            [dyad_pa_sub, triad_pa_sub, tetrad_pa_sub],
        ]

        model: RandomForestRegressor
        if mode == 'all':
            model: RandomForestRegressor = joblib.load(
                'random_forest/' + self.trajectory_dir + 'random_forest_model_all.pkl')

        for i in range(3):
            if mode == 'size':
                model: RandomForestRegressor = joblib.load(
                    'random_forest/' + self.trajectory_dir + 'random_forest_model_' + size[i] + '.pkl')
            fig_scat, ax_scat = plt.subplots(1, 1, figsize=(7, 5), dpi=200)
            for j in range(len(cfo_f[0][i])):
                if mode == 'each':
                    model: RandomForestRegressor = joblib.load(
                        'random_forest/' + self.trajectory_dir + 'random_forest_model_' + size[i] + '_Group' + str(
                            j + 1) + '.pkl')

                fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=200)
                cfo_f_tot_re = cfo_f[0][i][j]
                cfo_f_sum_re = cfo_f[1][i][j]
                cfo_f_sub_re = cfo_f[2][i][j]
                cfo_p_tot_re = cfo_p[0][i][j]
                cfo_p_sum_re = cfo_p[1][i][j]
                cfo_p_sub_re = cfo_p[2][i][j]
                cfo_fa_tot_re = cfo_fa[0][i][j]
                cfo_fa_sum_re = cfo_fa[1][i][j]
                cfo_fa_sub_re = cfo_fa[2][i][j]
                cfo_pa_tot_re = cfo_pa[0][i][j]
                cfo_pa_sum_re = cfo_pa[1][i][j]
                cfo_pa_sub_re = cfo_pa[2][i][j]

                performance_re = performance[i][j]

                input = np.column_stack((
                    cfo_f_tot_re,
                    cfo_f_sum_re,
                    cfo_f_sub_re,
                    cfo_p_tot_re,
                    cfo_p_sum_re,
                    cfo_p_sub_re,
                    # cfo_fa_tot_re,
                    # cfo_fa_sum_re,
                    # cfo_fa_sub_re,
                    # cfo_pa_tot_re,
                    # cfo_pa_sum_re,
                    # cfo_pa_sub_re,
                ))

                pre_performance = model.predict(input)

                ax.plot(time[::100], performance_re[::100], label='$e^{coop}$', lw=2)
                ax.plot(time[::100], pre_performance[::100], label='$\hat{e}^{coop}$', linestyle='--', lw=0.5,
                        alpha=0.8)
                ax.set_xlabel('Time (sec)')
                ax.set_ylabel('Cooperative RMSE')
                ax.legend()
                ax.set_title(size[i] + ' Group' + str(j + 1))
                os.makedirs('fig/random_forest/' + self.trajectory_dir + 'TimeSeries', exist_ok=True)
                # fig.savefig('fig/random_forest/' + self.trajectory_dir + 'TimeSeries/Performance_predict_' + size[i] + '_Group' + str(j + 1) + '_' + mode + '.png')
                fig.savefig('fig/random_forest/' + self.trajectory_dir + 'TimeSeries/Performance_predict_' + size[
                    i] + '_Group' + str(j + 1) + '_' + mode + '.pdf')


                # 訓練データとテストデータに分割
                X_train, X_test, y_train, y_test = train_test_split(input[::100], performance_re[::100], test_size=0.2,
                                                                    random_state=42, shuffle=True)

                # テストデータを用いて予測
                predicted_values = np.zeros(len(y_test))
                for k in range(len(X_test)):
                    predicted_values[k] = model.predict(X_test[k].reshape(1, -1))

                ax_scat.scatter(predicted_values[::1], y_test[::1], s=3.0, alpha=0.2)

                r2 = mystat.r2_score(predicted_values, y_test)

                ax_scat.text(0.02, 0.89 - j * 0.05, 'Group' + str(j + 1) + ': $R^2 = {:.2f}$'.format(r2),
                             horizontalalignment='left',
                             transform=ax_scat.transAxes, fontsize="small")

                ax_scat.set_xlabel('Predicted Cooperative RMSE')
                ax_scat.set_ylabel('Cooperative RMSE')
                ax_scat.set_ylim(-0.03, 0.03)
                ax_scat.set_xlim(-0.03, 0.03)

            ax_scat.plot([0, 1], [0, 1], transform=ax_scat.transAxes, color='black', linestyle='--')
            fig_scat.subplots_adjust(wspace=1.0)  # 横方向の余白を調整
            os.makedirs('fig/random_forest/' + self.trajectory_dir, exist_ok=True)
            fig_scat.savefig('fig/random_forest/' + self.trajectory_dir + 'performance_predict_random_forest_group_' + size[i] +'.pdf')

        # plt.show()

    def simulation_verification_all(self, sigma: int = 'none'):
        dyad_p_tot, dyad_f_tot, dyad_pa_tot, dyad_fa_tot = self.dyad_cfo.summation_ave_cfo(mode='b_abs')
        triad_p_tot, triad_f_tot, triad_pa_tot, triad_fa_tot = self.triad_cfo.summation_ave_cfo(mode='b_abs')
        tetrad_p_tot, tetrad_f_tot, tetrad_pa_tot, tetrad_fa_tot = self.tetrad_cfo.summation_ave_cfo(mode='b_abs')

        dyad_p_sum, dyad_f_sum, dyad_pa_sum, dyad_fa_sum = self.dyad_cfo.summation_ave_cfo(mode='a_abs')
        triad_p_sum, triad_f_sum, triad_pa_sum, triad_fa_sum = self.triad_cfo.summation_ave_cfo(mode='a_abs')
        tetrad_p_sum, tetrad_f_sum, tetrad_pa_sum, tetrad_fa_sum = self.tetrad_cfo.summation_ave_cfo(mode='a_abs')

        dyad_p_sub, dyad_f_sub, dyad_pa_sub, dyad_fa_sub = self.dyad_cfo.subtraction_ave_cfo()
        triad_p_sub, triad_f_sub, triad_pa_sub, triad_fa_sub = self.triad_cfo.subtraction_ave_cfo()
        tetrad_p_sub, tetrad_f_sub, tetrad_pa_sub, tetrad_fa_sub = self.tetrad_cfo.subtraction_ave_cfo()

        error_ts_dyad, error_dot_ts_dyad = self.dyad_cfo.time_series_performance_cooperation(sigma=sigma)
        error_ts_triad, error_dot_ts_triad = self.triad_cfo.time_series_performance_cooperation(sigma=sigma)
        error_ts_tetrad, error_dot_ts_tetrad = self.tetrad_cfo.time_series_performance_cooperation(sigma=sigma)

        performance = [
            error_ts_dyad,
            error_ts_triad,
            error_ts_tetrad,
        ]

        size = ['Dyad', 'Triad', 'Tetrad']

        cfo_f = [
            [dyad_f_tot, triad_f_tot, tetrad_f_tot],
            [dyad_f_sum, triad_f_sum, tetrad_f_sum],
            [dyad_f_sub, triad_f_sub, tetrad_f_sub],
        ]

        cfo_p = [
            [dyad_p_tot, triad_p_tot, tetrad_p_tot],
            [dyad_p_sum, triad_p_sum, tetrad_p_sum],
            [dyad_p_sub, triad_p_sub, tetrad_p_sub],
        ]

        for i in range(3):
            for j in range(len(cfo_f[0][i])):
                fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200)

                cost = np.sqrt(np.abs((np.abs(0.1 - cfo_p[2][i][j]) / 0.25) + ((2.6 - cfo_f[0][i][j]) / 2.6)))

                p = performance[i][j]

                ax.scatter(cost, p, s=0.05, alpha=0.1)
                ax.set_ylabel('Cooperative RMSE')
                ax.set_xlabel('Cost')

                os.makedirs('fig/simulation/' + self.trajectory_dir + '/Check/', exist_ok=True)
                plt.savefig('fig/simulation/' + self.trajectory_dir + '/Check/check_cost_' + size[i] + '_Group' + str(
                    j + 1) + '_all.png')
        plt.show()

    def robomech2024(self, sigma: int = 'none'):
        dyad_p_tot, dyad_f_tot, dyad_pa_tot, dyad_fa_tot = self.dyad_cfo.summation_ave_cfo(mode='b_abs')
        triad_p_tot, triad_f_tot, triad_pa_tot, triad_fa_tot = self.triad_cfo.summation_ave_cfo(mode='b_abs')
        tetrad_p_tot, tetrad_f_tot, tetrad_pa_tot, tetrad_fa_tot = self.tetrad_cfo.summation_ave_cfo(mode='b_abs')

        dyad_p_sum, dyad_f_sum, dyad_pa_sum, dyad_fa_sum = self.dyad_cfo.summation_ave_cfo(mode='a_abs')
        triad_p_sum, triad_f_sum, triad_pa_sum, triad_fa_sum = self.triad_cfo.summation_ave_cfo(mode='a_abs')
        tetrad_p_sum, tetrad_f_sum, tetrad_pa_sum, tetrad_fa_sum = self.tetrad_cfo.summation_ave_cfo(mode='a_abs')

        dyad_p_sub, dyad_f_sub, dyad_pa_sub, dyad_fa_sub = self.dyad_cfo.subtraction_ave_cfo()
        triad_p_sub, triad_f_sub, triad_pa_sub, triad_fa_sub = self.triad_cfo.subtraction_ave_cfo()
        tetrad_p_sub, tetrad_f_sub, tetrad_pa_sub, tetrad_fa_sub = self.tetrad_cfo.subtraction_ave_cfo()

        error_ts_dyad, error_dot_ts_dyad = self.dyad_cfo.time_series_performance_cooperation(sigma=sigma)
        error_ts_triad, error_dot_ts_triad = self.triad_cfo.time_series_performance_cooperation(sigma=sigma)
        error_ts_tetrad, error_dot_ts_tetrad = self.tetrad_cfo.time_series_performance_cooperation(sigma=sigma)

        performance = [
            error_ts_dyad,
            error_ts_triad,
            error_ts_tetrad,
        ]

        size = ['Dyad', 'Triad', 'Tetrad']

        cfo_f = [
            [dyad_f_tot, triad_f_tot, tetrad_f_tot],
            [dyad_f_sum, triad_f_sum, tetrad_f_sum],
            [dyad_f_sub, triad_f_sub, tetrad_f_sub],
        ]

        cfo_p = [
            [dyad_p_tot, triad_p_tot, tetrad_p_tot],
            [dyad_p_sum, triad_p_sum, tetrad_p_sum],
            [dyad_p_sub, triad_p_sub, tetrad_p_sub],
        ]

        cfo_fa = [
            [dyad_fa_tot, triad_fa_tot, tetrad_fa_tot],
            [dyad_fa_sum, triad_fa_sum, tetrad_fa_sum],
            [dyad_fa_sub, triad_fa_sub, tetrad_fa_sub],
        ]

        cfo_pa = [
            [dyad_pa_tot, triad_pa_tot, tetrad_pa_tot],
            [dyad_pa_sum, triad_pa_sum, tetrad_pa_sum],
            [dyad_pa_sub, triad_pa_sub, tetrad_pa_sub],
        ]

        # cfo_labels = ['Total FCFO', 'Summation FCFO', 'Subtraction FCFO']
        cfo_labels = ['Total PCFO', 'Summation PCFO', 'Subtraction PCFO']
        input_labels = ['$\sigma_F^{tot}$', '$\sigma_F^{sum}$', '$\sigma_F^{sub}$',
                        '$\sigma_P^{tot}$', '$\sigma_P^{sum}$', '$\sigma_P^{sub}$',
                        # '$\sigma_{FA}^{tot}$', '$\sigma_{FA}^{sum}$', '$\sigma_{FA}^{sub}$',
                        # '$\sigma_{PA}^{tot}$', '$\sigma_{PA}^{sum}$', '$\sigma_{PA}^{sub}$'
                        ]

        # クロス相関
        # fig, axs = plt.subplots(3, 1, figsize=(5, 10), dpi=150)
        # dp = sns.color_palette()
        # for i, ax in enumerate(axs):
        #     lines = []
        #     for j in range(len(size)):
        #         cfo_f_re = cfo_f[i][j].reshape(1, -1)
        #         cfo_p_re = cfo_p[i][j].reshape(1, -1)
        #         cfo_fa_re = cfo_fa[i][j].reshape(1, -1)
        #         cfo_pa_re = cfo_pa[i][j].reshape(1, -1)
        #
        #         performance_re = performance[j].reshape(1, -1)
        #
        #
        #         # クロス相関関数を計算
        #         n = len(cfo_f_re[0][::100])
        #         x = cfo_f_re[0][::100] - np.mean(cfo_p_re[0][::100])
        #         y = performance_re[0][::100] - np.mean(performance_re[0][::100])
        #         cross_corr = np.correlate(x, y, mode='full')
        #         cross_corr /= (np.linalg.norm(x, ord=2) * np.linalg.norm(y, ord=2))
        #         lags = np.arange(-n + 1, n)
        #         ax.plot(lags, cross_corr, label=size[j])
        #         ax.set_title(cfo_labels[i] + ' vs Cooperative RMSE')
        #         ax.set_xlabel('Lag')
        #         ax.set_ylabel('Cross Correlation')
        #         ax.legend()
        #
        # plt.tight_layout()
        # os.makedirs('fig/CFO-Performance/cross_correlation/' + self.trajectory_dir, exist_ok=True)
        # plt.savefig('fig/CFO-Performance/cross_correlation/' + self.trajectory_dir + 'cross_correlation_PCFO.png')
        # plt.show()

        # sns.regplot(x=cfo_f_re[0][::100], y=performance_re[0][::100], ax=ax,
        #             scatter_kws={'s': 0.05, 'alpha': 0.1}, line_kws={'lw': 2}, logx=True
        #             # label=size[j]
        #             )
        #
        # # 回帰直線の凡例を手動で作成
        # line = Line2D([0], [0], color=dp[j], lw=2, label=size[j])
        # lines.append(line)
        # r = np.corrcoef(cfo_f_re[0], performance_re[0])
        # # r = np.corrcoef(np.log10(cfo_re[0]), performance_re[0])
        # # r2 = np.corrcoef(cfo_re[0], performance_re[0])
        #
        # ax.text(0.02, 0.89-j*0.1, size[j]+': $r = {:.2f}$'.format(r2), horizontalalignment='left',
        #         transform=ax.transAxes, fontsize="small")
        # ax.set(xscale='log')
        # ax.set_xlabel(cfo_labels[i])
        # ax.set_ylabel('Cooperative RMSE')

        # ax.legend(handles=lines, loc='upper right')

        feature_importances = []
        model_list = []
        # fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=200)
        for i in range(len(size)):
            cfo_f_tot_re = cfo_f[0][i].reshape(1, -1)
            cfo_f_sum_re = cfo_f[1][i].reshape(1, -1)
            cfo_f_sub_re = cfo_f[2][i].reshape(1, -1)
            cfo_p_tot_re = cfo_p[0][i].reshape(1, -1)
            cfo_p_sum_re = cfo_p[1][i].reshape(1, -1)
            cfo_p_sub_re = cfo_p[2][i].reshape(1, -1)
            cfo_fa_tot_re = cfo_fa[0][i].reshape(1, -1)
            cfo_fa_sum_re = cfo_fa[1][i].reshape(1, -1)
            cfo_fa_sub_re = cfo_fa[2][i].reshape(1, -1)
            cfo_pa_tot_re = cfo_pa[0][i].reshape(1, -1)
            cfo_pa_sum_re = cfo_pa[1][i].reshape(1, -1)
            cfo_pa_sub_re = cfo_pa[2][i].reshape(1, -1)

            performance_re = performance[i].reshape(1, -1)

            input = np.column_stack((cfo_f_tot_re[0],
                                     cfo_f_sum_re[0],
                                     cfo_f_sub_re[0],
                                     cfo_p_tot_re[0],
                                     cfo_p_sum_re[0],
                                     cfo_p_sub_re[0],
                                     # cfo_fa_tot_re[0],
                                     # cfo_fa_sum_re[0],
                                     # cfo_fa_sub_re[0],
                                     # cfo_pa_tot_re[0],
                                     # cfo_pa_sum_re[0],
                                     # cfo_pa_sub_re[0],
                                     ))

            input_log = np.column_stack((np.log(cfo_f_tot_re[0]),
                                         np.log(cfo_f_sum_re[0]),
                                         np.log(cfo_f_sub_re[0]),
                                         np.log(cfo_p_tot_re[0]),
                                         np.log(cfo_p_sum_re[0]),
                                         np.log(cfo_p_sub_re[0]),
                                         # np.log(cfo_fa_tot_re[0]),
                                         # np.log(cfo_fa_sum_re[0]),
                                         # np.log(cfo_fa_sub_re[0]),
                                         # np.log(cfo_pa_tot_re[0]),
                                         # np.log(cfo_pa_sum_re[0]),
                                         # np.log(cfo_pa_sub_re[0]),
                                         ))

            n = 2
            input_tim = np.column_stack((cfo_f_tot_re[0] ** n,
                                         cfo_f_sum_re[0] ** n,
                                         cfo_f_sub_re[0] ** n,
                                         cfo_p_tot_re[0] ** n,
                                         cfo_p_sum_re[0] ** n,
                                         cfo_p_sub_re[0] ** n,
                                         # cfo_fa_tot_re[0] ** n,
                                         # cfo_fa_sum_re[0] ** n,
                                         # cfo_fa_sub_re[0] ** n,
                                         # cfo_pa_tot_re[0] ** n,
                                         # cfo_pa_sum_re[0] ** n,
                                         # cfo_pa_sub_re[0] ** n,
                                         ))

            X_train, X_test, y_train, y_test = train_test_split(input[::100], performance_re[0][::100], test_size=0.2,
                                                                random_state=42)

            # 説明変数行列に定数項を追加
            X = sm.add_constant(X_train)

            # 回帰モデルの作成
            model = sm.OLS(y_train, X).fit()
            model_list.append(model)
            print(model.summary())

            # X_ = sm.add_constant(X_test)
            #
            # predicted_values = model.predict(X_)
            # # sns.regplot(x=predicted_values[::100], y=performance_re[0][::100], ax=ax,
            # #             scatter_kws={'s': 0.05, 'alpha': 0.1}, line_kws={'lw': 2},
            # #             )
            # ax.scatter(predicted_values, y_test, s=0.05, alpha=0.2)
            # ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='black', linestyle='--')
            #
            # # r2 = np.corrcoef(predicted_values, performance_re[0])
            # ax.text(0.02, 0.89-i*0.05, size[i]+': $R^2 = {:.2f}$'.format(model.rsquared), horizontalalignment='left',
            #         transform=ax.transAxes, fontsize="small")
            #
            # ax.set_xlabel('Predicted Cooperative RMSE')
            # ax.set_ylabel('Cooperative RMSE')
            # ax.set_ylim(-0.03, 0.03)
            # ax.set_xlim(-0.03, 0.03)

        fig, axs = plt.subplots(3, 1, figsize=(5, 12), dpi=200, sharex=True, sharey=True)
        for i in range(len(size)):
            ax = axs[i]
            for j in range(len(cfo_f[0][i])):
                cfo_f_tot_re = cfo_f[0][i]
                cfo_f_sum_re = cfo_f[1][i]
                cfo_f_sub_re = cfo_f[2][i]
                cfo_p_tot_re = cfo_p[0][i]
                cfo_p_sum_re = cfo_p[1][i]
                cfo_p_sub_re = cfo_p[2][i]

                input = np.column_stack((
                    cfo_f[0][i][j],
                    cfo_f[1][i][j],
                    cfo_f[2][i][j],
                    cfo_p[0][i][j],
                    cfo_p[1][i][j],
                    cfo_p[2][i][j],
                ))

                X_train, X_test, y_train, y_test = train_test_split(input[::100], performance[i][j][::100],
                                                                    test_size=0.2, random_state=42, shuffle=True)

                X = sm.add_constant(X_test)
                predicted = model_list[i].predict(X)
                ax.scatter(predicted, y_test, s=0.05, alpha=0.2)
                ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='black', linestyle='--')
                r2 = r2_score(y_test, predicted)
                ax.text(0.02, 0.89 - j * 0.05, 'Group' + str(j + 1) + ': $R^2 = {:.2f}$'.format(r2),
                        horizontalalignment='left',
                        transform=ax.transAxes, fontsize="small")
                ax.set_title(size[i])
                ax.set_ylim(-0.03, 0.03)
                ax.set_xlim(-0.03, 0.03)
        axs[2].set_xlabel('Predicted Cooperative RMSE')
        axs[1].set_ylabel('Cooperative RMSE')

        os.makedirs('fig/LRM/' + self.trajectory_dir, exist_ok=True)
        plt.savefig('fig/LRM/' + self.trajectory_dir + 'performance_predict.png')
        plt.savefig('fig/LRM/' + self.trajectory_dir + 'performance_predict.pdf')

        #     # 訓練データとテストデータに分割
        #     X_train, X_test, y_train, y_test = train_test_split(input[::100], performance_re[0][::100], test_size=0.2, random_state=42)
        #
        #     # # ランダムフォレストモデルの構築
        #     # model = RandomForestRegressor(n_estimators=100, random_state=42)
        #     # model.fit(X_train, y_train)
        #     # # モデルをファイルに保存
        #     # os.makedirs('random_forest', exist_ok=True)
        #     # joblib.dump(model, 'random_forest/random_forest_model_' + size[i] + '.pkl')
        #
        #     # 保存されたモデルをロード
        #     model = joblib.load('random_forest/random_forest_model_' + size[i] + '.pkl')
        #
        #     # # SHAP値の計算
        #     # explainer = shap.Explainer(model, X_train)
        #     # shap_values = explainer(X_test[:100])
        #     # print(shap_values)
        #
        #
        #     # 特徴量の重要度を取得
        #     feature_importances.append(model.feature_importances_)
        #
        #     # テストデータを用いて予測
        #     # predicted_values = model.predict(X_test)
        #     predicted_values = np.zeros(len(y_test))
        #     for j in range(len(X_test)):
        #         # print(X_test[j].reshape(1, -1).shape)
        #         predicted_values[j] = model.predict(X_test[j].reshape(1, -1))
        #
        #     # sns.regplot(x=predicted_values, y=y_test, ax=ax,
        #     #             scatter_kws={'s': 0.05, 'alpha': 0.1}, line_kws={'lw': 2},
        #     #             )
        #     ax.scatter(predicted_values, y_test, s=0.05, alpha=0.2)
        #     ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='black', linestyle='--')
        #
        #     # r = np.corrcoef(predicted_values, y_test)
        #     # r = np.corrcoef(np.log10(cfo_re[0]), performance_re[0])
        #     r2 = mystat.r2_score(predicted_values, y_test)
        #
        #     ax.text(0.02, 0.89-i*0.1, size[i]+': $R^2 = {:.2f}$'.format(r2), horizontalalignment='left',
        #             transform=ax.transAxes, fontsize="small")
        #
        #     ax.set_xlabel('Predicted Cooperative RMSE')
        #     ax.set_ylabel('Cooperative RMSE')
        #     ax.set_ylim(-0.03, 0.03)
        #     ax.set_xlim(-0.03, 0.03)
        #
        # plt.subplots_adjust(wspace=1.0)  # 横方向の余白を調整
        # os.makedirs('fig/robomech2024', exist_ok=True)
        # plt.savefig('fig/robomech2024/performance_predict_random_forest.png')
        #
        # # 特徴量の重要度を可視化
        # fig, axs = plt.subplots(3, 1, figsize=(6, 15), dpi=200, sharex=True, sharey=True)
        # for i in range(len(size)):
        #     ax = axs[i]
        #     width=0.4
        #     ax.bar(np.arange(len(input_labels)) + width/2, feature_importances[i], tick_label=input_labels, width=width)
        #     # ax.yticks(range(len(feature_importances)), boston.feature_names)
        #     ax.set_title(size[i])
        #     ax.set_ylim(0, 0.5)
        #     ax.set_xlim(-width, len(input_labels) - width/2)
        #
        # axs[1].set_ylabel('Feature Importance')
        # axs[2].set_xlabel('Feature')
        # plt.subplots_adjust(wspace=0.4)  # 横方向の余白を調整
        # plt.savefig('fig/robomech2024/feature_importance_random_forest.png')
        plt.show()

    def robomech2024_axis(self, sigma: int = 'none'):
        dyad_pp_tot, dyad_pr_tot, dyad_fp_tot, dyad_fr_tot = self.dyad_cfo.summation_cfo(mode='b_abs')
        triad_pp_tot, triad_pr_tot, triad_fp_tot, triad_fr_tot = self.triad_cfo.summation_cfo(mode='b_abs')
        tetrad_pp_tot, tetrad_pr_tot, tetrad_fp_tot, tetrad_fr_tot = self.tetrad_cfo.summation_cfo(mode='b_abs')

        dyad_pp_sum, dyad_pr_sum, dyad_fp_sum, dyad_fr_sum = self.dyad_cfo.summation_cfo(mode='a_abs')
        triad_pp_sum, triad_pr_sum, triad_fp_sum, triad_fr_sum = self.triad_cfo.summation_cfo(mode='a_abs')
        tetrad_pp_sum, tetrad_pr_sum, tetrad_fp_sum, tetrad_fr_sum = self.tetrad_cfo.summation_cfo(mode='a_abs')

        dyad_pp_sub, dyad_pr_sub, dyad_fp_sub, dyad_fr_sub = self.dyad_cfo.subtraction_cfo()
        triad_pp_sub, triad_pr_sub, triad_fp_sub, triad_fr_sub = self.triad_cfo.subtraction_cfo()
        tetrad_pp_sub, tetrad_pr_sub, tetrad_fp_sub, tetrad_fr_sub = self.tetrad_cfo.subtraction_cfo()

        error_ts_dyad_p, error_ts_dyad_r, error_dot_ts_dyad_p, error_dot_ts_dyad_r = self.dyad_cfo.time_series_performance_cooperation_axis(
            sigma=sigma)
        error_ts_triad_p, error_ts_triad_r, error_dot_ts_triad_p, error_dot_ts_triad_r = self.triad_cfo.time_series_performance_cooperation_axis(
            sigma=sigma)
        error_ts_tetrad_p, error_ts_tetrad_r, error_dot_ts_tetrad_p, error_dot_ts_tetrad_r = self.tetrad_cfo.time_series_performance_cooperation_axis(
            sigma=sigma)

        performance = [
            [
                error_ts_dyad_p,
                error_ts_triad_p,
                error_ts_tetrad_p,
            ],
            [
                error_ts_dyad_r,
                error_ts_triad_r,
                error_ts_tetrad_r,
            ]
        ]

        size = ['Dyad', 'Triad', 'Tetrad']

        cfo_p = [
            [
                [dyad_pp_tot, triad_pp_tot, tetrad_pp_tot],
                [dyad_pp_sum, triad_pp_sum, tetrad_pp_sum],
                [dyad_pp_sub, triad_pp_sub, tetrad_pp_sub],
            ],
            [
                [dyad_pr_tot, triad_pr_tot, tetrad_pr_tot],
                [dyad_pr_sum, triad_pr_sum, tetrad_pr_sum],
                [dyad_pr_sub, triad_pr_sub, tetrad_pr_sub],
            ],

        ]
        cfo_f = [
            [
                [dyad_fp_tot, triad_fp_tot, tetrad_fp_tot],
                [dyad_fp_sum, triad_fp_sum, tetrad_fp_sum],
                [dyad_fp_sub, triad_fp_sub, tetrad_fp_sub],
            ],
            [
                [dyad_fr_tot, triad_fr_tot, tetrad_fr_tot],
                [dyad_fr_sum, triad_fr_sum, tetrad_fr_sum],
                [dyad_fr_sub, triad_fr_sub, tetrad_fr_sub],
            ]
        ]

        cfo_labels = ['Total FCFO', 'Summation FCFO', 'Subtraction FCFO']

        axis = ['Pitch', 'Roll']

        # fig, axs = plt.subplots(3, 2, figsize=(12, 10), dpi=150)
        # dp = sns.color_palette()
        # for i in range(len(cfo_labels)):
        #     for j in range(len(size)):
        #         lines = []
        #         for k in range(len(axis)):
        #             ax = axs[i][k]
        #             cfo_p_re = cfo_p[k][i][j].reshape(1, -1)
        #             cfo_f_re = cfo_f[k][i][j].reshape(1, -1)
        #
        #             performance_re = performance[k][j].reshape(1, -1)
        #
        #             # # クロス相関関数を計算
        #             # n = len(cfo_f_re[0][::100])
        #             # cross_corr = np.correlate(cfo_f_re[0][::100], performance_re[0][::100], mode='full')
        #             # lags = np.arange(-n + 1, n)
        #             # ax.plot(lags, cross_corr)
        #             # ax.set_title('Cross Correlation')
        #             # ax.set_xlabel('Lag')
        #             # ax.set_ylabel('Cross Correlation')
        #
        #             # 自己相関関数を計算
        #             autocorr1 = np.correlate(cfo_f_re[0][::100], cfo_f_re[0][::100], mode='full')
        #             autocorr2 = np.correlate(performance_re[0][::100], performance_re[0][::100], mode='full')
        #
        #             ax.plot(autocorr1 / np.max(autocorr1), label='CFO')
        #             ax.plot(autocorr2 / np.max(autocorr2), label='Performance')
        #             ax.set_title('Autocorrelation')
        #             ax.set_xlabel('Lag')
        #             ax.set_ylabel('Autocorrelation')
        #             ax.legend()
        #             # ax.show()
        #
        #
        #             # sns.regplot(x=cfo_f_re[0][::100], y=performance_re[0][::100], ax=ax,
        #             #             scatter_kws={'s': 0.05, 'alpha': 0.1}, line_kws={'lw': 2}, logx=True,
        #             #             fit_reg=True
        #             #             # label=size[j]
        #             #             )
        #             # # 回帰直線の凡例を手動で作成
        #             # line = Line2D([0], [0], color=dp[j], lw=2, label=size[j])
        #             # lines.append(line)
        #             # r = np.corrcoef(cfo_f_re[0], performance_re[0])
        #             # # r = np.corrcoef(np.log10(cfo_re[0]), performance_re[0])
        #             # # r2 = mystat.r2_score(cfo_re[0], performance_re[0])
        #             #
        #             # ax.text(0.02, 0.89-j*0.1, size[j]+': $r = {:.2f}$'.format(r[0][1]), horizontalalignment='left',
        #             #         transform=ax.transAxes, fontsize="small")
        #             # # ax.set(xscale='log')
        #             # ax.set_xlabel(cfo_labels[i])
        #             # ax.set_ylabel('Cooperative RMSE')
        #
        #         # ax.legend(handles=lines, loc='upper right')

        fig, axs = plt.subplots(1, 2, figsize=(8, 6), dpi=150)
        for i in range(len(size)):
            for j in range(len(axis)):
                ax = axs[j]
                cfo_tot_p_re = cfo_p[j][0][i].reshape(1, -1)
                cfo_sum_p_re = cfo_p[j][1][i].reshape(1, -1)
                cfo_sub_p_re = cfo_p[j][2][i].reshape(1, -1)
                cfo_tot_f_re = cfo_f[j][0][i].reshape(1, -1)
                cfo_sum_f_re = cfo_f[j][1][i].reshape(1, -1)
                cfo_sub_f_re = cfo_f[j][2][i].reshape(1, -1)

                performance_re = performance[j][i].reshape(1, -1)

                # 説明変数行列に定数項を追加
                X = sm.add_constant(np.column_stack((cfo_tot_p_re[0], cfo_sum_p_re[0], cfo_sub_p_re[0], cfo_tot_f_re[0],
                                                     cfo_sum_f_re[0], cfo_sub_f_re[0])))

                # 回帰モデルの作成
                model = sm.OLS(performance_re[0], X).fit()
                print(model.summary())

                predicted_values = model.predict(X)
                sns.regplot(x=predicted_values[::100], y=performance_re[0][::100], ax=ax,
                            scatter_kws={'s': 0.05, 'alpha': 0.1}, line_kws={'lw': 2},
                            )

        plt.show()

    def time_series_performance_summation_ave_cfo(self, graph=False, mode='no_abs', sigma: int = 'none', dec=1):
        dyad_p, dyad_f, dyad_pa, dyad_fa = self.dyad_cfo.summation_ave_cfo(mode=mode)
        triad_p, triad_f, triad_pa, triad_fa = self.triad_cfo.summation_ave_cfo(mode=mode)
        tetrad_p, tetrad_f, tetrad_pa, tetrad_fa = self.tetrad_cfo.summation_ave_cfo(mode=mode)
        summation_3sec_datas = [
            [dyad_p, triad_p, tetrad_p],
            [dyad_f, triad_f, tetrad_f],
            [dyad_pa, triad_pa, tetrad_pa],
            [dyad_fa, triad_fa, tetrad_fa],
        ]

        error_ts_dyad, error_dot_ts_dyad = self.dyad_cfo.time_series_performance_cooperation(sigma=sigma)
        error_ts_triad, error_dot_ts_triad = self.triad_cfo.time_series_performance_cooperation(sigma=sigma)
        error_ts_tetrad, error_dot_ts_tetrad = self.tetrad_cfo.time_series_performance_cooperation(sigma=sigma)

        performance = [
            [error_ts_dyad, error_dot_ts_dyad],
            [error_ts_triad, error_dot_ts_triad],
            [error_ts_tetrad, error_dot_ts_tetrad],
        ]

        size = ['Dyad', 'Triad', 'Tetrad']

        performance_label = ['Error', 'Speed Error']

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

        df_ = []
        for i in range(len(size)):
            for j in range(len(performance_label)):
                for k in range(len(performance[i][j])):
                    for l in range(len(summation_3sec_datas)):
                        df_.append(pd.DataFrame({
                            'CFO': summation_3sec_datas[l][i][k][::dec],
                            'CFO_type': types[l],
                            'Performance': performance[i][j][k][::dec],
                            'Performance_type': performance_label[j],
                            'Group Size': size[i],
                            'Group': str(k + 1)
                        })
                        )

        df = pd.concat([i for i in df_], axis=0)
        df.reset_index(drop=True, inplace=True)

        if graph:

            kwargs = dict(
                s=0.02,
                alpha=0.6,
            )

            # # FacetGridを作成してグラフを設定
            # g = sns.FacetGrid(df, col="Performance_type", row="CFO_type", hue="Group Size", height=4, aspect=1.2, sharey=False,
            #                   sharex=False, )
            # g.map(plot_scatter, "Performance", "CFO", **kwargs)
            # df = df[df['CFO'] > 0.01]
            # df = df.loc[~(abs(df) < 0.01).any(axis=1)]
            df = df[~df['CFO'].between(-0.01, 0.01)]

            sns.lmplot(data=df, x='CFO', y='Performance', hue='Group Size', col='Performance_type', row='CFO_type',
                       scatter_kws={'s': 0.05, 'alpha': 0.1}, height=4, aspect=1.2, order=5,
                       sharey=False, sharex=False, line_kws={'lw': 2})

            ylims = [(0, 1)]

            xlims = [(-0.015, 0.015), (-0.1, 0.5),
                     (-0.015, 0.015), (-0.1, 0.5),
                     (-0.015, 0.015), (-0.1, 0.5)]

            if mode == 'h-h':
                xlims = [(0.0, 0.1), (0.0, 3.0),
                         (0.0, 0.1), (0.0, 3.0),
                         (0.0, 0.1), (0.0, 3.0)]

            # Iterate over the AxesSubplots in the FacetGrid object, and set the y limit individually
            # for num, ax in enumerate(g.axes.flatten()):
            #     # col_var = ax.get_title().split(" ")[-1]  # get the column variable from the title
            #     ax.set_xlim(xlims[num])
            #     ax.set_ylim(ylims[0])

            # base_dir = 'fig/CFO-Performance/TimeSeries/Summation/Combine/'
            # if mode == 'no_abs':
            #     os.makedirs(base_dir + 'NoABS/', exist_ok=True)
            #     plt.savefig(base_dir + 'NoABS/SummationCFO-Performance_NoABS_Combine_' + self.trajectory_type + '.png')
            # elif mode == 'b_abs':
            #     os.makedirs(base_dir + 'BeforeABS/', exist_ok=True)
            #     plt.savefig(base_dir + 'BeforeABS/SummationCFO-Performance_BeforeABS_Combine_' + self.trajectory_type + '.png')
            # elif mode == 'a_abs':
            #     os.makedirs(base_dir + 'AfterABS/', exist_ok=True)
            #     plt.savefig(base_dir + 'AfterABS/SummationCFO-Performance_AfterABS_Combine_' + self.trajectory_type + '.png')
            plt.show()

        return df

    def time_series_performance_summation_ave_cfo_axis(self, graph=False, mode='no_abs', sigma: int = 'none', dec=1):
        dyad_pp, dyad_rp, dyad_pf, dyad_rf = self.dyad_cfo.summation_cfo(mode=mode)
        triad_pp, triad_rp, triad_pf, triad_rf = self.triad_cfo.summation_cfo(mode=mode)
        tetrad_pp, tetrad_rp, tetrad_pf, tetrad_rf = self.tetrad_cfo.summation_cfo(mode=mode)
        summation_3sec_datas = [
            [dyad_pp, triad_pp, tetrad_pp],
            [dyad_rp, triad_rp, tetrad_rp],
            [dyad_pf, triad_pf, tetrad_pf],
            [dyad_rf, triad_rf, tetrad_rf],
        ]

        error_ts_dyad_x, error_ts_dyad_y, error_dot_ts_dyad_x, error_dot_ts_dyad_y = self.dyad_cfo.time_series_performance_cooperation_axis(
            sigma=sigma)
        error_ts_triad_x, error_ts_triad_y, error_dot_ts_triad_x, error_dot_ts_triad_y = self.triad_cfo.time_series_performance_cooperation_axis(
            sigma=sigma)
        error_ts_tetrad_x, error_ts_tetrad_y, error_dot_ts_tetrad_x, error_dot_ts_tetrad_y = self.tetrad_cfo.time_series_performance_cooperation_axis(
            sigma=sigma)

        performance = [
            [error_ts_dyad_x, error_ts_dyad_y, error_dot_ts_dyad_x, error_dot_ts_dyad_y],
            [error_ts_triad_x, error_ts_triad_y, error_dot_ts_triad_x, error_dot_ts_triad_y],
            [error_ts_tetrad_x, error_ts_tetrad_y, error_dot_ts_tetrad_x, error_dot_ts_tetrad_y],
        ]

        size = ['Dyad', 'Triad', 'Tetrad']

        performance_label = ['X Error', 'Y Error', 'X SE', 'Y SE']

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

        df_ = []
        for i in range(len(size)):
            for j in range(len(performance_label)):
                for k in range(len(performance[i][j])):
                    for l in range(len(summation_3sec_datas)):
                        df_.append(pd.DataFrame({
                            'CFO': summation_3sec_datas[l][i][k][::dec],
                            'CFO_type': types[l],
                            'Performance': performance[i][j][k][::dec],
                            'Performance_type': performance_label[j],
                            'Group Size': size[i],
                            'Group': str(k + 1)
                        })
                        )

        df = pd.concat([i for i in df_], axis=0)
        df.reset_index(drop=True, inplace=True)

        if graph:

            kwargs = dict(
                s=0.02,
                alpha=0.6,
            )

            # # FacetGridを作成してグラフを設定
            # g = sns.FacetGrid(df, col="Performance_type", row="CFO_type", hue="Group Size", height=4, aspect=1.2, sharey=False,
            #                   sharex=False, )
            # g.map(plot_scatter, "Performance", "CFO", **kwargs)
            # df = df[df['CFO'] > 0.01]
            # df = df.loc[~(abs(df) < 0.01).any(axis=1)]
            df = df[~df['CFO'].between(-0.01, 0.01)]

            sns.lmplot(data=df, x='CFO', y='Performance', hue='Group Size', col='Performance_type', row='CFO_type',
                       scatter_kws={'s': 0.05, 'alpha': 0.1}, height=4, aspect=1.2, order=5,
                       sharey=False, sharex=False, line_kws={'lw': 2})

            ylims = [(0, 1)]

            xlims = [(-0.015, 0.015), (-0.1, 0.5),
                     (-0.015, 0.015), (-0.1, 0.5),
                     (-0.015, 0.015), (-0.1, 0.5)]

            if mode == 'h-h':
                xlims = [(0.0, 0.1), (0.0, 3.0),
                         (0.0, 0.1), (0.0, 3.0),
                         (0.0, 0.1), (0.0, 3.0)]

            # Iterate over the AxesSubplots in the FacetGrid object, and set the y limit individually
            # for num, ax in enumerate(g.axes.flatten()):
            #     # col_var = ax.get_title().split(" ")[-1]  # get the column variable from the title
            #     ax.set_xlim(xlims[num])
            #     ax.set_ylim(ylims[0])

            # base_dir = 'fig/CFO-Performance/TimeSeries/Summation/Combine/'
            # if mode == 'no_abs':
            #     os.makedirs(base_dir + 'NoABS/', exist_ok=True)
            #     plt.savefig(base_dir + 'NoABS/SummationCFO-Performance_NoABS_Combine_' + self.trajectory_type + '.png')
            # elif mode == 'b_abs':
            #     os.makedirs(base_dir + 'BeforeABS/', exist_ok=True)
            #     plt.savefig(base_dir + 'BeforeABS/SummationCFO-Performance_BeforeABS_Combine_' + self.trajectory_type + '.png')
            # elif mode == 'a_abs':
            #     os.makedirs(base_dir + 'AfterABS/', exist_ok=True)
            #     plt.savefig(base_dir + 'AfterABS/SummationCFO-Performance_AfterABS_Combine_' + self.trajectory_type + '.png')
            plt.show()

        return df

    def time_series_performance_subtraction_ave_cfo_axis(self, graph=False, sigma: int = 'none', dec=1):
        dyad_pp, dyad_rp, dyad_pf, dyad_rf = self.dyad_cfo.subtraction_cfo()
        triad_pp, triad_rp, triad_pf, triad_rf = self.triad_cfo.subtraction_cfo()
        tetrad_pp, tetrad_rp, tetrad_pf, tetrad_rf = self.tetrad_cfo.subtraction_cfo()
        subtraction_3sec_datas = [
            [dyad_pp, triad_pp, tetrad_pp],
            [dyad_rp, triad_rp, tetrad_rp],
            [dyad_pf, triad_pf, tetrad_pf],
            [dyad_rf, triad_rf, tetrad_rf],
        ]

        error_ts_dyad_x, error_ts_dyad_y, error_dot_ts_dyad_x, error_dot_ts_dyad_y = self.dyad_cfo.time_series_performance_cooperation_axis(
            sigma=sigma)
        error_ts_triad_x, error_ts_triad_y, error_dot_ts_triad_x, error_dot_ts_triad_y = self.triad_cfo.time_series_performance_cooperation_axis(
            sigma=sigma)
        error_ts_tetrad_x, error_ts_tetrad_y, error_dot_ts_tetrad_x, error_dot_ts_tetrad_y = self.tetrad_cfo.time_series_performance_cooperation_axis(
            sigma=sigma)

        performance = [
            [error_ts_dyad_x, error_ts_dyad_y, error_dot_ts_dyad_x, error_dot_ts_dyad_y],
            [error_ts_triad_x, error_ts_triad_y, error_dot_ts_triad_x, error_dot_ts_triad_y],
            [error_ts_tetrad_x, error_ts_tetrad_y, error_dot_ts_tetrad_x, error_dot_ts_tetrad_y],
        ]

        size = ['Dyad', 'Triad', 'Tetrad']

        performance_label = ['X Error', 'Y Error', 'X SE', 'Y SE']

        types = ['Subtraction Pitch PCFO (Avg)', 'Subtraction Roll PCFO (Avg)', 'Subtraction Pitch FCFO (Avg)',
                 'Subtraction Roll FCFO (Avg)']
        ranges = [0.5, 0.5, 10.0, 10.0]

        df_ = []
        for i in range(len(size)):
            for j in range(len(performance_label)):
                for k in range(len(performance[i][j])):
                    for l in range(len(subtraction_3sec_datas)):
                        df_.append(pd.DataFrame({
                            'CFO': subtraction_3sec_datas[l][i][k][::dec],
                            'CFO_type': types[l],
                            'Performance': performance[i][j][k][::dec],
                            'Performance_type': performance_label[j],
                            'Group Size': size[i],
                            'Group': str(k + 1)
                        })
                        )

        df = pd.concat([i for i in df_], axis=0)
        df.reset_index(drop=True, inplace=True)

        if graph:
            kwargs = dict(
                s=0.02,
                alpha=0.6,
            )

            # FacetGridを作成してグラフを設定
            # g = sns.FacetGrid(df, col="Performance_type", row="CFO_type", hue="Group Size", height=4, aspect=1.2, sharey=False,
            #                   sharex=False, )
            # g.map(plot_scatter, "Performance", "CFO", **kwargs)
            # df = df[df['CFO'] > 0.01]
            df = df[~df['CFO'].between(-0.01, 0.01)]

            sns.lmplot(data=df, x='CFO', y='Performance', hue='Group Size', col='Performance_type', row='CFO_type',
                       scatter_kws={'s': 0.05, 'alpha': 0.1}, height=4, aspect=1.2, order=5,
                       sharey=False, sharex=False, line_kws={'lw': 2})

            ylims = [(0, 1)]

            xlims = [(-0.015, 0.015), (-0.1, 0.5),
                     (-0.015, 0.015), (-0.1, 0.5),
                     (-0.015, 0.015), (-0.1, 0.5)]

            # Iterate over the AxesSubplots in the FacetGrid object, and set the y limit individually
            # for num, ax in enumerate(g.axes.flatten()):
            #     # col_var = ax.get_title().split(" ")[-1]  # get the column variable from the title
            #     ax.set_xlim(xlims[num])
            #     ax.set_ylim(ylims[0])

            # base_dir = 'fig/CFO-Performance/TimeSeries/Subtraction/Combine/'
            # os.makedirs(base_dir, exist_ok=True)
            # plt.savefig(base_dir + 'SubtractionCFO-Performance_Combine_' + self.trajectory_type + '.png')
            plt.show()

        return df

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
