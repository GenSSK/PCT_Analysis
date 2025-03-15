import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import friedmanchisquare
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

class group:
    def __init__(self, all_npz, alone_npz, other_npz, none_npz, group_type, trajectory_type):
        self.group_type = group_type
        self.trajectory_type = trajectory_type
        self.all = all_npz
        self.alone = alone_npz
        self.other = other_npz
        self.none = none_npz

        self.smp = 0.0001  # サンプリング時間
        self.duringtime = self.all[0]['duringtime'][0]  # ターゲットの移動時間
        remove_count_from_start = 0
        remove_count_from_end = 0
        if trajectory_type == 'Circle':
            remove_count_from_start = 1.0
            remove_count_from_end = 0.0
        elif trajectory_type == 'Lemniscate':
            remove_count_from_start = 2.0
            remove_count_from_end = 0.0
            self.duringtime = self.duringtime / 2.0
        elif trajectory_type == 'RoseCurve':
            remove_count_from_start = 0.0
            remove_count_from_end = 0.0
            self.duringtime = self.duringtime / 4.0
        elif trajectory_type == 'Random':
            remove_count_from_start =  10.0
            remove_count_from_end = 0.0
        elif trajectory_type == 'Discrete_Random':
            remove_count_from_start = 9.0
            remove_count_from_end = 0.0

        self.starttime = self.all[0]['starttime'][0] + remove_count_from_start * self.duringtime  # タスク開始時間
        self.endtime = self.all[0]['endtime'][0] - remove_count_from_end * self.duringtime  # タスク終了時間
        if self.trajectory_type == 'Discrete_Random':
            self.endtime -= 2.0
        self.tasktime = self.endtime - self.starttime  # タスクの時間
        self.period = int(self.tasktime / self.duringtime)  # 回数
        self.num = int(self.duringtime / self.smp)  # 1ピリオドにおけるデータ数
        self.start_num = int(self.starttime / self.smp)
        self.end_num = int(self.endtime / self.smp)
        self.join = self.all[0]['join'][0]
        self.tasktype = ''.join(chr(char) for char in self.all[0]['tasktype'])
        self.controltype = ''.join(chr(char) for char in self.all[0]['controltype'])
        # print(self.join)


        self.group_dir = ""
        if str(self.group_type) == 'dyad':
            self.group_dir = 'dyad/'
        elif str(self.group_type) == 'triad':
            self.group_dir = 'triad/'
        elif str(self.group_type) == 'tetrad':
            self.group_dir = 'tetrad/'

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
        plt.rcParams['font.family']= 'sans-serif'
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

    def get_time(self):
        data = self.all[0]
        return data['time'][self.start_num:self.end_num]-20.0

    def get_starttime(self):
        return self.starttime-20.0


    def get_endtime(self):
        return self.endtime-20.0


    def get_force(self, npz):
        force = np.zeros((len(npz), 2, self.join, self.end_num - self.start_num))
        for i in range(len(npz)):
            data = npz[i]
            for j in range(self.join):
                interfacenum = 'i' + str(j + 1)
                force[i][0][j] = data[interfacenum + '_p_text_10'][self.start_num:self.end_num]
                force[i][1][j] = data[interfacenum + '_r_text_10'][self.start_num:self.end_num]

        return force


    def get_force_all(self, npz):
        force = self.get_force(npz)
        force_all = np.zeros((len(force), 2, self.end_num - self.start_num))
        for i in range(len(force)):
            for j in range(2):
                for k in range(self.join):
                    force_all[i][j] += np.abs(force[i][j][k])
        return force_all

    def get_force_diff_dyad_only(self, npz):
        force = self.get_force(npz)
        force_diff = np.zeros((len(force), 2, self.end_num - self.start_num))
        for i in range(len(force)):
            for j in range(2):
                force_diff[i][j] = force[i][j][0] - force[i][j][1]
        return force_diff

    def get_force_all_rms(self, npz):
        force = self.get_force(npz)
        force_all_rms = np.zeros((len(force), self.end_num - self.start_num))
        for i in range(len(force)):
            for j in range(2):
                force_ = np.zeros((2, self.end_num - self.start_num))
                for k in range(self.join):
                    force_[j] += np.abs(force[i][j][k])
                force_all_rms[i] = np.sqrt(force_[0] **2 + force_[1] **2)
        return force_all_rms

    def get_plate_force(self, npz):
        force = self.get_force(npz)
        plate_force = np.zeros((len(force), 2, self.end_num - self.start_num))
        for i in range(len(force)):
            for j in range(2):
                for k in range(self.join):
                    plate_force[i][j] += force[i][j][k]
                plate_force[i][j] = np.abs(plate_force[i][j])
        return plate_force


    def get_force_effort(self, npz):
        force = self.get_force(npz)
        force_all = self.get_force_all(npz)
        force_plate = self.get_plate_force(npz)
        force_effort = force_all - force_plate
        return force_effort

    def plot_force(self):
        force = [
            self.get_force(self.all),
            self.get_force(self.alone),
            self.get_force(self.other),
            self.get_force(self.none)
        ]

        force_all = [
            self.get_force_all(self.all),
            self.get_force_all(self.alone),
            self.get_force_all(self.other),
            self.get_force_all(self.none)
        ]

        force_plate = [
            self.get_plate_force(self.all),
            self.get_plate_force(self.alone),
            self.get_plate_force(self.other),
            self.get_plate_force(self.none)
        ]

        force_effort = [
            self.get_force_effort(self.all),
            self.get_force_effort(self.alone),
            self.get_force_effort(self.other),
            self.get_force_effort(self.none)
        ]

        axis = ['Pitch', 'Roll']
        condition = ['All', 'Alone', 'Other', 'None']
        dec = 100
        time = self.get_time()
        for c in range(len(condition)):
            fig, ax = plt.subplots(2, 1, figsize=(10, 10), dpi=200, sharex=True)
            ax[0].set_title(condition[c])
            # ax[0].set_xlim(20.0, 30.0)
            for i in range(1):
            # for i in range(len(force_effort[0])):
                for j in range(2):
                    for k in range(self.join):
                        ax[j].plot(time[::dec], force[c][i][j][k][::dec], label='Subject ' + str(k + 1))
                    ax[j].plot(time[::dec], force_all[c][i][j][::dec], label='All')
                    ax[j].plot(time[::dec], force_plate[c][i][j][::dec], label='Plate')
                    ax[j].plot(time[::dec], force_effort[c][i][j][::dec], label='Effort')
                    ax[j].set_ylabel('Force (Nm)')
                    ax[j].set_xlabel('Time (s)')
                    ax[j].legend()
        plt.show()

    def csv_effort_and_force(self):
        force_effort = np.stack([
            self.get_force_effort(self.all),
            self.get_force_effort(self.alone),
            self.get_force_effort(self.other),
            self.get_force_effort(self.none)
        ])

        force_effort_com = np.sum(force_effort, axis=2)
        force_effort_ave = np.average(force_effort_com, axis=2)

        performance = np.stack([
            self.get_performance(self.all),
            self.get_performance(self.alone),
            self.get_performance(self.other),
            self.get_performance(self.none)
        ])

        performance_ave = np.average(performance, axis=2)


        condition = ['ALL', 'ALONE', 'OTHER', 'NONE']

        df_ = []
        for i in range(len(force_effort_ave[0])):
            df_.append(pd.DataFrame({
                'Effort_ALL': force_effort_ave[0][i] * 5,
                'Effort_ALONE': force_effort_ave[1][i] * 5,
                'Effort_OTHER': force_effort_ave[2][i] * 5,
                'Effort_NONE': force_effort_ave[3][i] * 5,
                'Performance_ALL': performance_ave[0][i],
                'Performance_ALONE': performance_ave[1][i],
                'Performance_OTHER': performance_ave[2][i],
                'Performance_NONE': performance_ave[3][i],
                'Group': 'Group ' + str(i + 1),
            }, index=[0])
            )


        df = pd.concat(df_, axis=0, ignore_index=True)
        # df.rename(columns={'value': 'Effort'}, inplace=True)
        # df.drop(columns='variable', inplace=True)

        print(df)

        df.to_csv('Avatar_effort_and_performance.csv', index=False)

    def plot_compare_effort(self):
        force_effort = np.stack([
            self.get_force_effort(self.all),
            self.get_force_effort(self.alone),
            self.get_force_effort(self.other),
            self.get_force_effort(self.none)
        ])

        force_effort_com = np.sum(force_effort, axis=2)
        force_effort_ave = np.average(force_effort_com, axis=2)


        condition = ['ALL', 'ALONE', 'OTHER', 'NONE']

        pairs = [
            {condition[0], condition[1]},
            {condition[0], condition[2]},
            {condition[0], condition[3]},
            {condition[1], condition[2]},
            {condition[1], condition[3]},
            {condition[2], condition[3]},
        ]

        df_ = []
        for i in range(len(force_effort_ave)):
            for j in range(len(force_effort_ave[i])):
                df_temp = pd.DataFrame({
                    'condition': condition[i],
                    'Effort': force_effort_ave[i][j] * 5,
                }, index=[0])
                df_.append(pd.melt(df_temp, id_vars='condition'))
                df_[i*len(force_effort_ave[0]) + j]['Group'] = 'Group ' + str(j + 1)

        df = pd.concat([i for i in df_], axis=0)
        df.rename(columns={'value': 'Effort'}, inplace=True)
        df.drop(columns='variable', inplace=True)

        # シャピロウィルク検定
        for i in range(len(condition)):
            W, shapiro_p_value = stats.shapiro(df['condition'] == condition[i])
            print(f'{condition[i]} Shapiro-Wilk test statistic: {W}, p-value: {shapiro_p_value}')

        # フリードマン検定を実行
        stat, p = friedmanchisquare(force_effort_ave[0], force_effort_ave[1], force_effort_ave[2], force_effort_ave[3])
        print(f'Friedman test statistic: {stat}, p-value: {p}')

        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=200, sharex=True)
        states_palette = states_palette = sns.color_palette("YlGnBu_r", n_colors=4)
        sns.boxplot(x='condition', y='Effort', data=df, ax=ax, palette=states_palette,
                    flierprops={"marker": "o", "markerfacecolor": "w"},)
        # mystat.t_test_multi(ax, pairs, df, 'condition', 'Effort', test='t-test_ind', comparisons_correction="Bonferroni")
        # mystat.anova(df, 'condition', 'Effort')

        ax.set_ylabel('Effort (N)')
        ax.set_xlabel('')
        ax.legend().set_visible(False)
        ax.set_yticks(np.arange(0, 30, 5))
        ax.set_ylim([0, 25.0])

        # os.makedirs('fig/' + self.group_dir + self.trajectory_dir, exist_ok=True)
        # plt.savefig('fig/' + self.group_dir + self.trajectory_dir + 'effort.pdf')

        plt.show()


    def get_performance(self, npz):
        performance = np.zeros((len(npz), self.end_num - self.start_num))
        for i in range(len(npz)):
            data = npz[i]
            performance[i] = np.sqrt(
                (data['targetx'][self.start_num:self.end_num] - data['ballx'][self.start_num:self.end_num]) ** 2
                + (data['targety'][self.start_num:self.end_num] - data['bally'][self.start_num:self.end_num]) ** 2
            )

        return performance

    def plot_compare_performance(self):
        performance = np.stack([
            self.get_performance(self.all),
            self.get_performance(self.alone),
            self.get_performance(self.other),
            self.get_performance(self.none)
        ])

        performance_ave = np.average(performance, axis=2)


        condition = ['ALL', 'ALONE', 'OTHER', 'NONE']

        pairs = [
            {condition[0], condition[1]},
            {condition[0], condition[2]},
            {condition[0], condition[3]},
            {condition[1], condition[2]},
            {condition[1], condition[3]},
            {condition[2], condition[3]},
        ]


        df_ = []
        for i in range(len(performance_ave)):
            for j in range(len(performance_ave[i])):
                df_temp = pd.DataFrame({
                    'condition': condition[i],
                    'Performance': performance_ave[i][j],
                }, index=[0])
                df_.append(pd.melt(df_temp, id_vars='condition'))
                df_[i*len(performance_ave[0]) + j]['Group'] = 'Group ' + str(j + 1)

        df = pd.concat([i for i in df_], axis=0)
        df.rename(columns={'value': 'Performance'}, inplace=True)
        df.drop(columns='variable', inplace=True)


        # シャピロウィルク検定
        for i in range(len(condition)):
            W, shapiro_p_value = stats.shapiro(df['condition'] == condition[i])
            print(f'{condition[i]} Shapiro-Wilk test statistic: {W}, p-value: {shapiro_p_value}')

        # フリードマン検定を実行
        stat, p = friedmanchisquare(performance_ave[0], performance_ave[1], performance_ave[2], performance_ave[3])
        print(f'Friedman test statistic: {stat}, p-value: {p}')



        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=200, sharex=True)
        states_palette = sns.color_palette("YlGnBu_r", n_colors=4)
        sns.boxplot(x='condition', y='Performance', data=df, ax=ax, palette=states_palette,
                    flierprops={"marker": "o", "markerfacecolor": "w"},)
        # mystat.t_test_multi(ax, pairs, df, 'Condition', 'Performance', test='t-test_ind', comparisons_correction="Bonferroni")
        # mystat.anova(df, 'Condition', 'Performance')

        ax.set_ylabel('Performance (m)')
        ax.set_xlabel('')
        ax.legend().set_visible(False)
        ax.set_yticks(np.arange(0.00, 0.06, 0.01))
        ax.set_ylim([0.00, 0.04])

        os.makedirs('fig/' + self.group_dir + self.trajectory_dir, exist_ok=True)
        plt.savefig('fig/' + self.group_dir + self.trajectory_dir + 'Performance.pdf')

        plt.show()

    def plot_compare_force_period(self):
        force_effort = np.stack([
            self.get_force(self.all),
            self.get_force(self.alone),
            self.get_force(self.other),
            self.get_force(self.none)
        ])

        force_effort_com = np.sum(force_effort, axis=2)
        force_effort_ave = np.average(force_effort_com, axis=2)

        condition = ['ALL', 'ALONE', 'OTHER', 'NONE']


        pairs = [
            {condition[0], condition[1]},
            {condition[0], condition[2]},
            {condition[0], condition[3]},
            {condition[1], condition[2]},
            {condition[1], condition[3]},
            {condition[2], condition[3]},
        ]

        df_ = []
        for i in range(len(force_effort_ave)):
            for j in range(len(force_effort_ave[i])):
                df_temp = pd.DataFrame({
                    'Condition': condition[i],
                    'Effort': force_effort_ave[i][j],
                }, index=[0])
                df_.append(pd.melt(df_temp, id_vars='Condition'))
                df_[i*len(force_effort_ave[0]) + j]['Group'] = 'Group ' + str(j + 1)

        df = pd.concat([i for i in df_], axis=0)
        df.rename(columns={'value': 'Effort'}, inplace=True)
        df.drop(columns='variable', inplace=True)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=200, sharex=True)
        sns.boxplot(x='Condition', y='Effort', data=df, ax=ax)
        mystat.t_test_multi(ax, pairs, df, 'Condition', 'Effort', test='t-test_ind', comparisons_correction="Bonferroni")
        mystat.anova(df, 'Condition', 'Effort')

        ax.set_ylabel('Effort (Nm)')
        ax.set_xlabel('')
        ax.legend().set_visible(False)
        ax.set_ylim([0, 6.0])

        plt.show()


    def plot_ts_force(self):
        force = np.stack([
            self.get_force(self.all),
            self.get_force(self.alone),
            self.get_force(self.other),
            self.get_force(self.none)
        ])

        axis = ['Pitch', 'Roll']

        print(force.shape)
        force = force.transpose(0, 2, 3, 1, 4)
        print(force.shape)
        force = force.reshape(len(force), len(axis), self.join, len(self.all), -1, self.num * 4)
        print(force.shape)
        force = force.reshape(len(force), len(axis), self.join, -1, self.num * 4)
        print(force.shape)

        force_reg = force.reshape(4, 2, 2, -1)

        fig, axs = plt.subplots(2, 1, figsize=(10, 10), dpi=200, sharex=True)
        palette = [
            sns.color_palette("ocean", n_colors=5),
            sns.color_palette("spring", n_colors=5)
        ]
        time = self.get_time()
        time_reg = np.tile(time[:self.num*4:1000], len(force[0][0][0]))
        for i, ax in enumerate(axs):
            for j in range(len(force)):
                for k in range(self.join):
                    for l in range(len(force[j][i][k])):
                        ax.scatter(time[:self.num*4:100], force[j][i][k][l][::100], color=palette[k][j], s=0.01, alpha=0.2)

                    sns.regplot(x=time_reg, y=force_reg[j][i][k][::1000], scatter=False, color=palette[k][j], ax=ax,
                                line_kws={'linewidth': 2.0}, order=10)

        plt.show()


    def plot_ine_improve(self):
        force_ine = np.stack([
            self.get_force_effort(self.all),
            self.get_force_effort(self.alone),
            self.get_force_effort(self.other),
            self.get_force_effort(self.none)
        ])

        condition = ['ALL', 'ALONE', 'OTHER', 'NONE']

        print(force_ine.shape)
        force_ine = np.sum(force_ine, axis=2)
        print(force_ine.shape)
        force_ine = force_ine.reshape(4, len(self.all), -1, self.num)
        print(force_ine.shape)
        force_ine = np.average(force_ine, axis=3)
        print(force_ine.shape)
        init_force_ine = force_ine[:, :, 0]
        force_ine_norm = force_ine / init_force_ine[:, :, np.newaxis]
        force_ine_norm_reg = force_ine_norm.reshape(4, -1)

        force_ine_norm_mean = np.average(force_ine_norm, axis=1)
        print(force_ine_norm_mean.shape)

        fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=200, sharex=True)
        period = np.arange(1, self.period + 1)
        period_reg = np.tile(period, len(force_ine_norm[0]))
        states_palette = sns.color_palette("YlGnBu_r", n_colors=4)
        lines = []
        for i in range(len(force_ine_norm_mean)):
            sns.regplot(x=period_reg, y=force_ine_norm_reg[i], scatter=False, ax=ax, line_kws={'linewidth': 2.0},
                        order=7, color=states_palette[i])

            line = Line2D([0], [0], color=states_palette[i], lw=2, label=condition[i])
            lines.append(line)

        ax.set_ylabel('Normalized ineffective force')
        ax.set_xlabel('Period')
        ax.set_ylim([0.0, 3.0])
        ax.set_xticks(np.arange(1, self.period + 1, 1))
        ax.legend(handles=lines, loc='lower left', ncol=4, framealpha=0.0,)

        os.makedirs('fig/' + self.group_dir + self.trajectory_dir, exist_ok=True)
        plt.savefig('fig/' + self.group_dir + self.trajectory_dir + 'Normalized_ine.pdf')

        plt.show()

    def plot_performance_improve(self):
        performance = np.stack([
            self.get_performance(self.all),
            self.get_performance(self.alone),
            self.get_performance(self.other),
            self.get_performance(self.none)
        ])

        condition = ['ALL', 'ALONE', 'OTHER', 'NONE']

        print(performance.shape)
        performance = performance.reshape(4, len(self.all), -1, self.num)
        print(performance.shape)
        performance = np.average(performance, axis=3)
        print(performance.shape)
        init_performance = performance[:, :, 0]
        performance_norm = performance / init_performance[:, :, np.newaxis]
        performance_norm_reg = performance_norm.reshape(4, -1)

        performance_norm_mean = np.average(performance_norm, axis=1)
        print(performance_norm_mean.shape)

        fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=200, sharex=True)
        period = np.arange(1, self.period + 1)
        period_reg = np.tile(period, len(performance_norm[0]))
        states_palette = sns.color_palette("YlGnBu_r", n_colors=4)
        lines = []
        for i in range(len(performance_norm_mean)):
            sns.regplot(x=period_reg, y=performance_norm_reg[i], scatter=False, ax=ax, line_kws={'linewidth': 2.0},
                        order=7, color=states_palette[i])

            line = Line2D([0], [0], color=states_palette[i], lw=2, label=condition[i])
            lines.append(line)

        ax.set_ylabel('Normalized Performance')
        ax.set_xlabel('Period')
        ax.set_ylim([0.0, 1.0])
        ax.set_xticks(np.arange(1, self.period + 1, 1))
        ax.legend(handles=lines, loc='lower left', ncol=4, framealpha=0.0,)

        os.makedirs('fig/' + self.group_dir + self.trajectory_dir, exist_ok=True)
        plt.savefig('fig/' + self.group_dir + self.trajectory_dir + 'Performance_improvement.pdf')

        plt.show()


    def plot_ts_force_rms(self):
        force = np.stack([
            self.get_force_all_rms(self.all),
            self.get_force_all_rms(self.alone),
            self.get_force_all_rms(self.other),
            self.get_force_all_rms(self.none)
        ])

        axis = ['Pitch', 'Roll']
        print(force.shape)
        force = np.abs(force)
        force = force.reshape(4, -1, self.num)
        print(force.shape)
        # force = force.transpose(0, 2, 1, 3, 4, 5)
        # print(force.shape)
        # force = force.reshape(4, 2, -1, self.num)
        # print(force.shape)
        # fig, axs = plt.subplots(2, 1, figsize=(10, 10), dpi=200, sharex=True)
        # states_palette = sns.color_palette("YlGnBu", n_colors=5)
        # time = self.get_time()
        # for i, ax in enumerate(axs):
        #     for j in range(len(force)):
        #         for k in range(len(force[j][i])):
        #             ax.scatter(time[:self.num:100], force[j][i][k][::100], color=states_palette[j], s=0.01, alpha=0.2)


        fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=200, sharex=True)
        states_palette = sns.color_palette("YlGnBu_r", n_colors=4)
        time = self.get_time()
        time_reg = np.tile(time[:self.num:100], len(force[0]))
        force_reg = force.reshape(4, -1)
        print(time_reg.shape)
        for j in range(len(force)):
            for k in range(len(force[j])):
                ax.scatter(time[:self.num:100], force[j][k][::100], color=states_palette[j], s=0.01, alpha=0.8)

            sns.regplot(x=time_reg, y=force_reg[j][::100], scatter=False, color=states_palette[j], ax=ax,
                        line_kws={'linewidth': 1.0}, order=5)

        plt.show()


    def plot_ts_force_diff(self):
        force = np.stack([
            self.get_force_diff_dyad_only(self.all),
            self.get_force_diff_dyad_only(self.alone),
            self.get_force_diff_dyad_only(self.other),
            self.get_force_diff_dyad_only(self.none)
        ])

        axis = ['Pitch', 'Roll']
        print(force.shape)
        force = force.reshape(4, 2, len(self.all), -1, self.num * 4)
        print(force.shape)

        force_reg = force.reshape(4, 2, -1)
        print(force_reg.shape)


        fig, axs = plt.subplots(2, 1, figsize=(10, 10), dpi=200, sharex=True)
        states_palette = sns.color_palette("Set1", n_colors=5)
        time = self.get_time()
        time_reg = np.tile(time[:self.num*4:1000], len(force[0][0]) * len(force[0][0][0]))
        for i, ax in enumerate(axs):
            for j in range(len(force)):
                for k in range(len(force[j][i])):
                    for l in range(len(force[j][i][k])):
                        ax.scatter(time[:self.num*4:100], force[j][i][k][l][::100], color=states_palette[j], s=0.01, alpha=0.2)

                sns.regplot(x=time_reg, y=force_reg[j][i][::1000], scatter=False, color=states_palette[j], ax=ax,
                            line_kws={'linewidth': 2.0}, order=10)

        plt.show()


    def plot_force_diff(self):
        force = np.stack([
            self.get_force_diff_dyad_only(self.all),
            self.get_force_diff_dyad_only(self.alone),
            self.get_force_diff_dyad_only(self.other),
            self.get_force_diff_dyad_only(self.none)
        ])

        print(force.shape)

        axis = ['Pitch', 'Roll']
        condition = ['ALL', 'ALONE', 'OTHER', 'NONE']
        dec = 100
        time = self.get_time()
        for c in range(len(condition)):
            fig, ax = plt.subplots(2, 1, figsize=(10, 10), dpi=200, sharex=True)
            ax[0].set_title(condition[c])
            # ax[0].set_xlim(20.0, 30.0)
            for i in range(len(force[0])):
                for j in range(2):
                    ax[j].plot(time[::dec], force[c][i][j][::dec])
                    ax[j].set_ylabel('Force (Nm)')
                    ax[j].set_xlabel('Time (s)')
                    ax[j].legend()
        plt.show()