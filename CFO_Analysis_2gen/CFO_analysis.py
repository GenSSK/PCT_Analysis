import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as path_effects
from matplotlib.colors import Normalize

import pandas as pd
import seaborn as sns
import os
import time

from scipy import optimize
from scipy import signal
from scipy.ndimage import gaussian_filter

import statsmodels.api as sm
from statsmodels.formula.api import ols

from sklearn.preprocessing import StandardScaler

from mypackage.mystatistics import myHilbertTransform as HT
from mypackage.mystatistics import mySTFT as STFT
from mypackage.mystatistics import myhistogram as hist
from mypackage.mystatistics import myFilter as Filter
from mypackage.mystatistics import statistics as mystat
from mypackage import ParallelExecutor

def fit_func_2nd(parameter, *args):
    accel, velo, y = args
    a = parameter[0]
    b = parameter[1]
    residual = y - (a * accel + b * velo)
    return residual


def plot_scatter(x, y, color, **kwargs):
    ax = plt.gca()
    ax.scatter(x, y, c=color, s=100, **kwargs)


class CFO:
    def __init__(self, cfo_data, group_type, trajectory_type):
        self.group_type = group_type
        self.trajectory_type = trajectory_type
        self.cfo = cfo_data

        self.smp = 0.0001  # サンプリング時間
        self.duringtime = self.cfo[0]['duringtime'][0]  # ターゲットの移動時間
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
            remove_count_from_start = 4.0
            remove_count_from_end = 0.0
            self.duringtime = self.duringtime / 4.0
        elif trajectory_type == 'Random':
            remove_count_from_start =  10.0
            remove_count_from_end = 0.0
        elif trajectory_type == 'Discrete_Random':
            remove_count_from_start = 9.0
            remove_count_from_end = 0.0

        self.starttime = self.cfo[0]['starttime'][0] + remove_count_from_start * self.duringtime  # タスク開始時間
        self.endtime = self.cfo[0]['endtime'][0] - remove_count_from_end * self.duringtime  # タスク終了時間
        if self.trajectory_type == 'Discrete_Random':
            self.endtime -= 2.0
        self.tasktime = self.endtime - self.starttime  # タスクの時間
        self.period = int(self.tasktime / self.duringtime)  # 回数
        self.num = int(self.duringtime / self.smp)  # 1ピリオドにおけるデータ数
        self.start_num = int((self.starttime - 20.0) / self.smp)
        self.end_num = int((self.endtime - 20.0) / self.smp)
        self.join = self.cfo[0]['join'][0]
        self.tasktype = ''.join(chr(char) for char in self.cfo[0]['tasktype'])
        self.controltype = ''.join(chr(char) for char in self.cfo[0]['controltype'])
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
        plt.rcParams['font.size'] = 8  # フォントの大きさ
        plt.rcParams['axes.linewidth'] = 1.5  # 軸の線幅edge linewidth。囲みの太さ

        plt.rcParams['lines.linewidth'] = 0.5  # 線の太さ

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

    def get_time(self):
        data = self.cfo[0]
        return data['time'][self.start_num:self.end_num]-20.0
    def get_starttime(self):
        return self.starttime-20.0

    def get_endtime(self):
        return self.endtime-20.0

    def show_prediction(self):
        variable_label = ['thm', 'text']
        rp_label = ['_r', '_p']
        # variable_option_label = ['', '_pre', '_pre_solo']
        variable_option_label = ['', '_pre']
        lws = [2.0, 1.0]
        lts = ['-', '--']

        ylims = [[-1.5, 1.5], [-6.0, 6.0]]
        yticks = [np.arange(-10, 10, 0.5), np.arange(-10, 10, 2.0)]

        ylabels = [['Roll angle (rad)', 'Pitch angle (rad)',],
                   ['Roll force (Nm)', 'Pitch force (Nm)']]

        for i in range(len(self.cfo)):
            data = self.cfo[i]
            fig, ax = plt.subplots(3, 2, figsize=(15, 10), dpi=150, sharex=True)

            plt.xticks(np.arange(self.starttime-20.0, (self.endtime-20.0) * 2, self.duringtime))
            plt.xlim([self.starttime-20.0, self.endtime-20.0])  # x軸の範囲
            ax[0, 1].set_xlabel("Time (sec)")
            ax[1, 1].set_xlabel("Time (sec)")

            for j in range(self.join):
                interfacenum = 'i' + str(j + 1)
                for k, v in enumerate(variable_label):
                    for l, rp in enumerate(rp_label):
                        ax[k, l].set_ylabel(ylabels[k][l])
                        ax[k, l].set_yticks(yticks[k])
                        ax[k, l].set_ylim(ylims[k][0], ylims[k][1])
                        # ax[k, l].set_xlim([50, 62])  # x軸の範囲
                        for m, vo in enumerate(variable_option_label):
                            lw = lws[m]
                            lt = lts[m]
                            disp_name = interfacenum  + rp + '_' + v + vo
                            ax[k, l].plot(data['time'][self.start_num:self.end_num:10]-20.0,
                                          data[disp_name][self.start_num:self.end_num:10],
                                          lt, lw=lw,
                                          label='P' + str(j + 1) + vo)
                        ax[k, l].legend(ncol=self.join, columnspacing=1, loc='upper left')

            axis_label = ['x', 'y']
            for j, axis in enumerate(axis_label):
                ax[2, j].set_ylabel(axis + "-axis Position (m)")
                # ax[2, j].set_yticks(np.arange(-0.2, 0.2, 0.05))
                ax[2, j].set_ylim([-0.5, 0.5])  # y軸の範囲
                ax[2, j].plot(data['time'][self.start_num:self.end_num:10]-20.0,
                              data['target' + axis][self.start_num:self.end_num:10],
                              '-', lw=2.0,
                              label='Target')
                ax[2, j].plot(data['time'][self.start_num:self.end_num:10]-20.0,
                              data['ball' + axis][self.start_num:self.end_num:10],
                              '-', lw=2.0,
                              label='H-H')
                for k in range(self.join):
                    interfacenum = 'i' + str(k + 1)
                    ax[2, j].plot(data['time'][self.start_num:self.end_num:10]-20.0,
                                  data[interfacenum + '_ball' + axis + '_pre'][self.start_num:self.end_num:10],
                                  '--', lw=1.0,
                                  label='Model' + str(k + 1))
                ax[2, j].legend(ncol=self.join, columnspacing=1, loc='upper left')


            plt.tight_layout()
            # plt.savefig("fig/response_check.png")
        plt.show()

    def task_show(self):
        for i in range(len(self.cfo)):
            data = self.cfo[i]

            fig, (x, y) = plt.subplots(2, 1, figsize=(5, 5), dpi=150, sharex=True)

            x.plot(data['time'][self.start_num:self.end_num:10]-20.0, data['targetx'][self.start_num:self.end_num:10],
                   label='Target')
            x.plot(data['time'][self.start_num:self.end_num:10]-20.0, data['ballx'][self.start_num:self.end_num:10],
                   label='Ball(H-H)')
            x.plot(data['time'][self.start_num:self.end_num:10]-20.0, data['i1_ballx_pre'][self.start_num:self.end_num:10],
                   label='Ball(M-M)')
            x.set_ylabel('X-axis Position (m)')
            x.legend(ncol=2, columnspacing=1, loc='upper left')
            x.set_ylim([-0.2, 0.2])  # y軸の範囲

            y.plot(data['time'][self.start_num:self.end_num:10]-20.0, data['targety'][self.start_num:self.end_num:10],
                   label='Target')
            y.plot(data['time'][self.start_num:self.end_num:10]-20.0, data['bally'][self.start_num:self.end_num:10],
                   label='Ball(H-H)')
            y.plot(data['time'][self.start_num:self.end_num:10]-20.0, data['i1_bally_pre'][self.start_num:self.end_num:10],
                   label='Ball(M-M)')
            y.set_ylabel('Y-axis Position (m)')
            y.legend(ncol=2, columnspacing=1, loc='upper left')
            y.set_ylim([-0.2, 0.2])  # y軸の範囲

            # plt.ylabel(r'Position (m)')
            # plt.legend()
            # plt.yticks(np.arange(-4, 4, 0.1))
            # plt.ylim([-0.4, 0.4])  # y軸の範囲
            # plt.xlim([data['starttime'], data['endtime']])  # x軸の範囲

            plt.xticks(np.arange(self.starttime, self.endtime * 2, self.duringtime * 2))
            plt.xlim([self.starttime, self.endtime])  # x軸の範囲
            plt.xlabel("Time (sec)")

            plt.tight_layout()
            # plt.savefig("First_time_target_movement.png")
        plt.show()

    def task_show_solo(self):
        for i in range(len(self.cfo)):
            data = self.cfo[i]

            fig, (x, y) = plt.subplots(2, 1, figsize=(5, 5), dpi=150, sharex=True)

            x.plot(data['time'][self.start_num:self.end_num:10]-20.0, data['targetx'][self.start_num:self.end_num:10],
                   label='Target')
            x.plot(data['time'][self.start_num:self.end_num:10]-20.0, data['ballx'][self.start_num:self.end_num:10],
                   label='Ball(H-H)')
            for j in range(self.join):
                x.plot(data['time'][self.start_num:self.end_num:10]-20.0,
                       data['i' + str(j + 1) + '_ballx_pre'][self.start_num:self.end_num:10],
                       label='Ball(solo' + str(j + 1) + ')')

            x.set_ylabel('X-axis Position (m)')
            x.legend(ncol=2, columnspacing=1, loc='upper left')
            x.set_ylim([-0.2, 0.2])  # y軸の範囲

            y.plot(data['time'][self.start_num:self.end_num:10]-20.0, data['targety'][self.start_num:self.end_num:10],
                   label='Target')
            y.plot(data['time'][self.start_num:self.end_num:10]-20.0, data['bally'][self.start_num:self.end_num:10],
                   label='Ball(H-H)')
            for j in range(self.join):
                y.plot(data['time'][self.start_num:self.end_num:10]-20.0,
                       data['i' + str(j + 1) + '_bally_pre'][self.start_num:self.end_num:10],
                       label='Ball(solo' + str(j + 1) + ')')
            # y.plot(data['pre_time'], data['pre_ball_x'], label='pre_bally')
            y.set_ylabel('Y-axis Position (m)')
            y.legend(ncol=2, columnspacing=1, loc='upper left')
            y.set_ylim([-0.2, 0.2])  # y軸の範囲

            # plt.ylabel(r'Position (m)')
            # plt.legend()
            # plt.yticks(np.arange(-4, 4, 0.1))
            # plt.ylim([-0.4, 0.4])  # y軸の範囲
            # plt.xlim([data['starttime'], data['endtime']])  # x軸の範囲

            plt.xticks(np.arange(self.starttime, self.endtime * 2, self.duringtime * 2))
            plt.xlim([self.starttime, self.endtime])  # x軸の範囲
            plt.xlabel("Time (sec)")

            plt.tight_layout()
            # plt.savefig("First_time_target_movement.png")
        plt.show()

    def task_show_sub(self, data):

        plt.ylabel(r'Position (m)')
        plt.yticks(np.arange(-4, 4, 0.1))
        plt.ylim([-0.4, 0.4])  # y軸の範囲
        plt.xlim([data[0]['starttime'], data[0]['endtime']])  # x軸の範囲
        plt.xlabel("Time (sec)")

        for i in range(len(data)):
            # plt.plot(data['time'], data['ballx'], label='ballx_'+str(i))
            # plt.plot(data['time'], data['bally'], label='bally_'+str(i))

            plt.plot(data[i]['time'][self.start_num:self.end_num:10],
                     data[i]['targetx'][self.start_num:self.end_num:10], label='targetx_' + str(i))
            plt.plot(data[i]['time'][self.start_num:self.end_num:10],
                     data[i]['targety'][self.start_num:self.end_num:10], label='targety_' + str(i))
            plt.legend()

        plt.tight_layout()
        # plt.savefig("First_time_target_movement.png")
        plt.show()

    def plot_time_series(self, mode='r'):
        rorp = 0  # 0:pitch, 1:roll
        if mode == 'p':
            rorp = 1
        data_name_r_p = [
            # [select_graph, 'data_name', 'label']
            [0, 'thm', 'Cooperation'],
            [0, 'thm_pre', 'Solo(Prediction)'],
            # [0, 'thm_pre_solo',     'Solo(Solo prediction)'],

            # [5, 'text',             'Cooperation'],
            # [5, 'text_pre',         'Solo(Prediction)'],
            # [5, 'text_pre_solo',    'Solo(Solo prediction)'],

            [1, 'pcfo', 'PCFO'],
            # [13, 'fcfo',            'FCFO'],

        ]

        data_name = [
            # [select_graph, 'data_name', 'label']
            [10, 'targetx', 'Target'],
            [10, 'targetx_act', 'Target'],
            [10, 'ballx', 'Ball(H-H)'],
            [10, 'ballx_pre', 'Ball(M-M)'],

            [11, 'targety', 'Target'],
            [11, 'targety_act', 'Target'],
            [11, 'bally', 'Ball(H-H)'],
            [11, 'bally_pre', 'Ball(M-M)'],
        ]

        ppcfo_summation_no_abs, rpcfo_summation_no_abs, pfcfo_summation_no_abs, rfcfo_summation_no_abs = CFO.summation_cfo(
            self, graph=False, mode='no_abs')
        ppcfo_summation_babs, rpcfo_summation_babs, pfcfo_summation_babs, rfcfo_summation_babs = CFO.summation_cfo(self,
                                                                                                                   graph=False,
                                                                                                                   mode='b_abs')
        ppcfo_summation_aabs, rpcfo_summation_aabs, pfcfo_summation_aabs, rfcfo_summation_aabs = CFO.summation_cfo(self,
                                                                                                                   graph=False,
                                                                                                                   mode='a_abs')
        ppcfo_subtraction, rpcfo_subtraction, pfcfo_subtraction, rfcfo_subtraction = CFO.subtraction_cfo(self)

        cfo_name = [
            # [select_graph, roll_data, pitch_data 'label']
            # [2, rpcfo_summation_no_abs,  ppcfo_summation_no_abs,  'PCFO_sum (no abs)'],
            [2, rpcfo_summation_babs, ppcfo_summation_babs, 'PCFO_sum (b abs)'],
            [3, rpcfo_summation_aabs, ppcfo_summation_aabs, 'PCFO_sum (a abs)'],
            [4, rpcfo_subtraction, ppcfo_subtraction, 'PCFO_sub'],

            # [6, rfcfo_summation_no_abs,  pfcfo_summation_no_abs,  'FCFO_sum (no abs)'],
            # [7, rfcfo_summation_babs,  pfcfo_summation_babs,    'FCFO_sum (b abs)'],
            # [8, rfcfo_summation_aabs,  pfcfo_summation_aabs,    'FCFO_sum (a abs)'],
            # [9, rfcfo_subtraction,     pfcfo_subtraction,       'FCFO_sub'],
        ]

        plot_info = [
            {'yticks': np.arange(-3, 3, 1.0), 'ylim': [-1.0, 1.0], 'ylabel': 'Position (rad)'},
            {'yticks': np.arange(-3, 3, 0.6), 'ylim': [-0.6, 0.6], 'ylabel': 'PCFO (rad)'},
            {'yticks': np.arange(-10, 10, 0.5), 'ylim': [0, 0.5], 'ylabel': 'PCFO_sum (b abs)'},
            {'yticks': np.arange(-10, 10, 0.5), 'ylim': [0, 0.5], 'ylabel': 'PCFO_sum (a abs)'},
            {'yticks': np.arange(-10, 10, 0.5), 'ylim': [0, 0.5], 'ylabel': 'PCFO_sub'},
            # {'yticks': np.arange(-10, 10, 0.5), 'ylim': [0, 1.5], 'ylabel': 'Position (rad)'},
            # {'yticks': np.arange(-10, 10, 0.5), 'ylim': [0, 1.5], 'ylabel': 'Position (rad)'},
        ]

        for j in range(len(self.cfo)):
            fig, ax = plt.subplots(5, 1, figsize=(10, 10), dpi=150, sharex=True)

            plt.xticks(np.arange(self.starttime, self.endtime * 2, self.duringtime))
            plt.xlim([self.starttime, self.endtime])  # x軸の範囲
            # plt.xlim([26, 35])  # x軸の範囲
            plt.xlabel("Time (sec)")

            data = self.cfo[j]
            for k in data_name_r_p:
                for i in range(self.join):
                    inum = 'i' + str(i + 1)
                    plot_name = inum + '_' + mode + '_' + k[1]
                    ax[k[0]].plot(data['time'][self.start_num:self.end_num:10]-20.0,
                                  data[plot_name][self.start_num:self.end_num:10],
                                  label=k[2] + ' (P' + str(i + 1) + ')')

            for k in cfo_name:
                ax[k[0]].plot(data['time'][self.start_num:self.end_num:10]-20.0,
                              k[rorp + 1][j][self.start_num:self.end_num:10], label=k[3])

            # for k in data_name:
            #     ax[k[0]].plot(data['time'][self.start_num:self.end_num:10]-20.0, data[k[1]][self.start_num:self.end_num:10], label=k[2])

            for k in range(len(ax)):
                ax[k].legend(ncol=2, loc='upper right', fontsize=6)
                ax[k].set_yticks(plot_info[k]['yticks'])
                ax[k].set_ylim(plot_info[k]['ylim'])
                ax[k].set_ylabel(plot_info[k]['ylabel'])

            plt.tight_layout()
            plt.savefig('fig/CFO/CFO_description_' + self.group_type + '.pdf')
            plt.show()

            # rthm.set_ylabel('Roll angle (rad)')
            # rthm.legend(ncol=2, columnspacing=1, loc='upper left')
            # rthm.set_yticks(np.arange(-10, 10, 0.5))
            # rthm.set_ylim([-1.5, 1.5])  # y軸の範囲

    def cfo_sub(self):

        for j in range(len(self.cfo)):
            data = self.cfo[j]

            fig, (ppcfo, rpcfo, pfcfo, rfcfo) = plt.subplots(4, 1, figsize=(5, 8), dpi=150, sharex=True)

            # plt.xlim([10, 60])  # x軸の範囲
            # plt.xlim([0.28, 0.89])  # x軸の範囲
            plt.xticks(np.arange(self.starttime-20.0, (self.endtime-20.0) * 2, self.duringtime))
            plt.xlim([self.starttime-20.0, self.endtime-20.0])  # x軸の範囲
            plt.xlabel("Time (sec)")

            for i in range(self.join):
                interfacenum = 'i' + str(i + 1)
                pcfoname = interfacenum + '_p_pcfo'
                fcfoname = interfacenum + '_p_fcfo'

                ppcfo.plot(data['time'][self.start_num:self.end_num:10]-20.0, data[pcfoname][self.start_num:self.end_num:10],
                           label='P' + str(i + 1))
                pfcfo.plot(data['time'][self.start_num:self.end_num:10]-20.0, data[fcfoname][self.start_num:self.end_num:10],
                           label='P' + str(i + 1))

                pcfoname = interfacenum + '_r_pcfo'
                fcfoname = interfacenum + '_r_fcfo'

                rpcfo.plot(data['time'][self.start_num:self.end_num:10]-20.0, data[pcfoname][self.start_num:self.end_num:10],
                           label='P' + str(i + 1))
                rfcfo.plot(data['time'][self.start_num:self.end_num:10]-20.0, data[fcfoname][self.start_num:self.end_num:10],
                           label='P' + str(i + 1))

            ppcfo.set_ylabel('Pitch PCFO (rad)')
            ppcfo.legend(ncol=2, columnspacing=1, loc='upper left')
            ppcfo.set_yticks(np.arange(-10, 10, 0.5))
            ppcfo.set_ylim([-1.5, 1.5])  # y軸の範囲

            rpcfo.set_ylabel('Roll PCFO (rad)')
            rpcfo.legend(ncol=2, columnspacing=1, loc='upper left')
            rpcfo.set_yticks(np.arange(-10, 10, 0.5))
            rpcfo.set_ylim([-1.5, 1.5])  # y軸の範囲

            pfcfo.set_ylabel('Pitch FCFO (Nm)')
            pfcfo.legend(ncol=2, columnspacing=1, loc='upper left')
            pfcfo.set_yticks(np.arange(-8.0, 8.0, 2.0))
            pfcfo.set_ylim([-6.0, 6.0])  # y軸の範囲

            rfcfo.set_ylabel('Roll FCFO (Nm)')
            rfcfo.legend(ncol=2, columnspacing=1, loc='upper left')
            rfcfo.set_yticks(np.arange(-8.0, 8.0, 2.0))
            rfcfo.set_ylim([-6.0, 6.0])  # y軸の範囲r

            plt.tight_layout()
            # plt.savefig("response.png")
        plt.show()

    def ocfo(self):

        plt.ylabel(r'Position (m)')
        plt.yticks(np.arange(-4, 4, 0.1))
        plt.ylim([-0.4, 0.4])  # y軸の範囲
        # plt.xlim([self.data[0]['starttime'], self.data[0]['endtime']])  # x軸の範囲
        plt.xlabel("Time (sec)")

        plt.plot(self.cfoo['time'][::10], self.cfoo['ecfo'][::10], label='ECFO')
        plt.plot(self.cfoo['time'][::10], self.cfoo['inecfo'][::10], label='InECFO')

        plt.ylabel('CFO')
        plt.legend(ncol=2, columnspacing=1, loc='upper left')
        plt.yticks(np.arange(-100, 100, 10.0))
        plt.ylim([-50, 50])  # y軸の範囲plt

        plt.tight_layout()
        plt.show()

    def summation_cfo(self, graph=False, mode='no_abs'):
        summation = self.cfo[0]['i1_p_pcfo'][self.start_num:self.end_num]
        types = ['_p_pcfo', '_r_pcfo', '_p_fcfo', '_r_fcfo']
        for type in types:
            for j in range(len(self.cfo)):
                data = self.cfo[j]
                summation_ = data['i1_p_pcfo'][self.start_num:self.end_num]
                for i in range(self.join):
                    interfacenum = 'i' + str(i + 1)
                    pcfoname = interfacenum + type

                    summation_ = np.vstack((summation_, data[pcfoname][self.start_num:self.end_num]))
                summation_ = np.delete(summation_, 0, 0)
                if mode == 'b_abs':
                    summation_ = np.abs(summation_)
                summation = np.vstack((summation, np.sum(summation_, axis=0)))

            # print(summation_)
        summation = np.delete(summation, 0, 0)
        # print(summation.shape)
        summation = summation.reshape([4, len(self.cfo), -1])
        summation = summation / self.cfo[0]['join'][0]
        if mode == 'a_abs':
            summation = np.abs(summation)

        if graph:
            fig, ax = plt.subplots(4, 1, figsize=(20, 10), dpi=150, sharex=True)

            # plt.xlim([10, 60])  # x軸の範囲
            # plt.xlim([0.28, 0.89])  # x軸の範囲
            plt.xlabel("Time (sec)")
            ylabels = ['Summation\nPitch PCFO (rad)', 'Summation\nRoll PCFO (rad)',
                       'Summation\nPitch FCFO (Nm)', 'Summation\nRoll FCFO (Nm)']
            ylims = [[0, 1.0], [0, 1.0], [0, 4.0], [0, 4.0]]
            ytics = [np.arange(-10, 10, 0.5), np.arange(-10, 10, 0.5),
                     np.arange(-8.0, 8.0, 1.0), np.arange(-8.0, 8.0, 1.0)]

            for i, cfo in enumerate(summation):
                for j in range(len(self.cfo)):
                    data = self.cfo[j]
                    ax[i].plot(data['time'][self.start_num:self.end_num:10]-20.0, cfo[j][::10], label='Group' + str(j + 1))
                    ax[i].set_ylabel(ylabels[i])
                    ax[i].legend(ncol=10, columnspacing=1, loc='upper left')
                    ax[i].set_yticks(ytics[i])
                    ax[i].set_ylim(ylims[i][0], ylims[i][1])  # y軸の範囲

            if mode == 'no_abs':
                os.makedirs('fig/CFO/Summation/NoABS/TimeSeries/', exist_ok=True)
                plt.savefig('fig/CFO/Summation/NoABS/TimeSeries/SummationCFO_NoABS_TimeSeries_' + str(self.group_type) + '.png')
            elif mode == 'b_abs':
                os.makedirs('fig/CFO/Summation/BeforeABS/TimeSeries/', exist_ok=True)
                plt.savefig('fig/CFO/Summation/BeforeABS/TimeSeries/SummationCFO_BeforeABS_TimeSeries_' + str(self.group_type) + '.png')
            elif mode == 'a_abs':
                os.makedirs('fig/CFO/Summation/AfterABS/TimeSeries/', exist_ok=True)
                plt.savefig('fig/CFO/Summation/AfterABS/TimeSeries/SummationCFO_AfterABS_TimeSeries_' + str(self.group_type) + '.png')

            plt.show()

        return summation[0], summation[1], summation[2], summation[3]

    def summation_cfo_3sec(self, mode='no_abs'):
        ppcfo_summation, rpcfo_summation, pfcfo_summation, rfcfo_summation = CFO.summation_cfo(self, graph=False,
                                                                                               mode=mode)

        ppcfo_summation_3sec = ppcfo_summation.reshape([len(self.cfo), -1, self.num])
        ppcfo_summation_3sec = np.average(ppcfo_summation_3sec, axis=2)

        rpcfo_summation_3sec = rpcfo_summation.reshape([len(self.cfo), -1, self.num])
        rpcfo_summation_3sec = np.average(rpcfo_summation_3sec, axis=2)

        pfcfo_summation_3sec = pfcfo_summation.reshape([len(self.cfo), -1, self.num])
        pfcfo_summation_3sec = np.average(pfcfo_summation_3sec, axis=2)

        rfcfo_summation_3sec = rfcfo_summation.reshape([len(self.cfo), -1, self.num])
        rfcfo_summation_3sec = np.average(rfcfo_summation_3sec, axis=2)

        return ppcfo_summation_3sec, rpcfo_summation_3sec, pfcfo_summation_3sec, rfcfo_summation_3sec

    def summation_cfo_combine(self, graph=False, mode='no_abs'):
        summation = self.cfo[0]['i1_p_pcfo'][self.start_num:self.end_num]
        types = [['_p_pcfo', '_r_pcfo'], ['_p_fcfo', '_r_fcfo']]
        for type in types:
            for j in range(len(self.cfo)):
                data = self.cfo[j]
                summation_ = data['i1_p_pcfo'][self.start_num:self.end_num]
                for i in range(self.join):
                    interfacenum = 'i' + str(i + 1)
                    pitch_cfoname = interfacenum + type[0]
                    roll_cfoname = interfacenum + type[1]

                    summation_ = np.vstack((summation_, data[pitch_cfoname][self.start_num:self.end_num] + data[roll_cfoname][self.start_num:self.end_num]))
                summation_ = np.delete(summation_, 0, 0)
                if mode == 'b_abs':
                    summation_ = np.abs(summation_)
                summation = np.vstack((summation, np.sum(summation_, axis=0)))

            # print(summation_)
        summation = np.delete(summation, 0, 0)
        # print(summation.shape)
        summation = summation.reshape([2, len(self.cfo), -1])
        summation = summation / self.cfo[0]['join'][0]
        if mode == 'a_abs':
            summation = np.abs(summation)

        if graph == True:
            fig, (pcfo, fcfo) = plt.subplots(2, 1, figsize=(20, 10), dpi=150, sharex=True)

            # plt.xlim([10, 60])  # x軸の範囲
            # plt.xlim([0.28, 0.89])  # x軸の範囲
            plt.xlabel("Time (sec)")

            for i in range(len(self.cfo)):
                data = self.cfo[i]

                pcfo.plot(data['time'][self.start_num:self.end_num:10]-20.0, summation[0][i][::10],
                           label='Group' + str(i + 1))
                fcfo.plot(data['time'][self.start_num:self.end_num:10]-20.0, summation[1][i][::10],
                           label='Group' + str(i + 1))

            pcfo.set_ylabel('Summation PCFO (rad)')
            fcfo.set_ylabel('Summation FCFO (Nm)')

            pcfo.legend(ncol=10, columnspacing=1, loc='upper left')
            fcfo.legend(ncol=10, columnspacing=1, loc='upper left')

            pcfo.set_yticks(np.arange(-10, 10, 0.5))
            fcfo.set_yticks(np.arange(-8.0, 8.0, 1.0))

            pcfo.set_ylim([0, 1.0])  # y軸の範囲
            fcfo.set_ylim([0, 4.0])  # y軸の範囲

            plt.tight_layout()
            if mode == 'no_abs':
                os.makedirs('fig/CFO/Summation/NoABS/TimeSeries/Combine/', exist_ok=True)
                plt.savefig('fig/CFO/Summation/NoABS/TimeSeries/Combine/SummationCFO_NoABS_TimeSeries_' + str(self.group_type) + '.png')
            elif mode == 'b_abs':
                os.makedirs('fig/CFO/Summation/BeforeABS/TimeSeries/Combine/', exist_ok=True)
                plt.savefig('fig/CFO/Summation/BeforeABS/TimeSeries/Combine/SummationCFO_BeforeABS_TimeSeries_' + str(self.group_type) + '.png')
            elif mode == 'a_abs':
                os.makedirs('fig/CFO/Summation/AfterABS/TimeSeries/Combine/', exist_ok=True)
                plt.savefig('fig/CFO/Summation/AfterABS/TimeSeries/Combine/SummationCFO_AfterABS_TimeSeries_' + str(self.group_type) + '.png')

        plt.show()

        return summation[0], summation[1]

    def summation_cfo_3sec_combine(self, mode='no_abs'):
        pcfo_summation, fcfo_summation = CFO.summation_cfo_combine(self, graph=False, mode=mode)

        pcfo_summation_3sec = pcfo_summation.reshape([len(self.cfo), -1, self.num])
        pcfo_summation_3sec = np.average(pcfo_summation_3sec, axis=2)

        fcfo_summation_3sec = fcfo_summation.reshape([len(self.cfo), -1, self.num])
        fcfo_summation_3sec = np.average(fcfo_summation_3sec, axis=2)

        return pcfo_summation_3sec, fcfo_summation_3sec

    def subtraction_cfo(self, graph=False):
        subtraction = self.cfo[0]['i1_p_pcfo'][self.start_num:self.end_num]
        types = ['p_pcfo', 'r_pcfo', 'p_fcfo', 'r_fcfo']
        for type in types:
            subtraction_ = self.cfo[0]['i1_p_pcfo'][self.start_num:self.end_num]
            for i in range(len(self.cfo)):
                data = self.cfo[i]
                if self.group_type == 'dyad':
                    sub_cfo1 = np.abs(np.subtract(data['i1_' + type][self.start_num:self.end_num],
                                                  data['i2_' + type][self.start_num:self.end_num]))
                    sub_cfo2 = np.abs(np.subtract(data['i2_' + type][self.start_num:self.end_num],
                                                  data['i1_' + type][self.start_num:self.end_num]))
                    sub_cfo_ave = np.add(np.abs(sub_cfo1), np.abs(sub_cfo2)) / 2
                    subtraction_ = np.vstack((subtraction_, sub_cfo_ave))

                elif self.group_type == 'triad':
                    sub_cfo1 = np.subtract(np.subtract(2 * data['i1_' + type][self.start_num:self.end_num],
                                                       data['i2_' + type][self.start_num:self.end_num]),
                                           data['i3_' + type][self.start_num:self.end_num])
                    sub_cfo2 = np.subtract(np.subtract(2 * data['i2_' + type][self.start_num:self.end_num],
                                                       data['i1_' + type][self.start_num:self.end_num]),
                                           data['i3_' + type][self.start_num:self.end_num])
                    sub_cfo3 = np.subtract(np.subtract(2 * data['i3_' + type][self.start_num:self.end_num],
                                                       data['i1_' + type][self.start_num:self.end_num]),
                                           data['i2_' + type][self.start_num:self.end_num])
                    sub_cfo_ave = np.add(np.add(np.abs(sub_cfo1), np.abs(sub_cfo2)), np.abs(sub_cfo3)) / 3
                    subtraction_ = np.vstack((subtraction_, sub_cfo_ave))

                elif self.group_type == 'tetrad':
                    sub_cfo1 = np.subtract(np.subtract(np.subtract(3 * data['i1_' + type][self.start_num:self.end_num],
                                                                   data['i2_' + type][self.start_num:self.end_num]),
                                                       data['i3_' + type][self.start_num:self.end_num]),
                                           data['i4_' + type][self.start_num:self.end_num])
                    sub_cfo2 = np.subtract(np.subtract(np.subtract(3 * data['i2_' + type][self.start_num:self.end_num],
                                                                   data['i1_' + type][self.start_num:self.end_num]),
                                                       data['i3_' + type][self.start_num:self.end_num]),
                                           data['i4_' + type][self.start_num:self.end_num])
                    sub_cfo3 = np.subtract(np.subtract(np.subtract(3 * data['i3_' + type][self.start_num:self.end_num],
                                                                   data['i1_' + type][self.start_num:self.end_num]),
                                                       data['i2_' + type][self.start_num:self.end_num]),
                                           data['i4_' + type][self.start_num:self.end_num])
                    sub_cfo4 = np.subtract(np.subtract(np.subtract(3 * data['i4_' + type][self.start_num:self.end_num],
                                                                   data['i1_' + type][self.start_num:self.end_num]),
                                                       data['i2_' + type][self.start_num:self.end_num]),
                                           data['i3_' + type][self.start_num:self.end_num])
                    sub_cfo_ave = np.add(np.add(np.add(np.abs(sub_cfo1), np.abs(sub_cfo2)), np.abs(sub_cfo3)),
                                         np.abs(sub_cfo4)) / 4
                    subtraction_ = np.vstack((subtraction_, sub_cfo_ave))

            subtraction_ = np.delete(subtraction_, 0, 0)
            subtraction = np.vstack((subtraction, subtraction_))
        subtraction = np.delete(subtraction, 0, 0)
        subtraction = subtraction.reshape([4, len(self.cfo), -1])

        if graph:
            fig, ax = plt.subplots(4, 1, figsize=(20, 10), dpi=150, sharex=True)

            # plt.xlim([10, 60])  # x軸の範囲
            # plt.xlim([0.28, 0.89])  # x軸の範囲
            plt.xlabel("Time (sec)")
            ylabels = ['Subtraction\nPitch PCFO (rad)', 'Subtraction\nRoll PCFO (rad)',
                       'Subtraction\nPitch FCFO (Nm)', 'Subtraction\nRoll FCFO (Nm)']
            ylims = [[0, 1.0], [0, 1.0], [0, 10.0], [0, 10.0]]
            ytics = [np.arange(0, 10, 0.5), np.arange(0, 10, 0.5),
                     np.arange(0, 20, 5.0), np.arange(0, 20, 5.0)]

            for i, cfo in enumerate(subtraction):
                for j in range(len(self.cfo)):
                    data = self.cfo[j]
                    ax[i].plot(data['time'][self.start_num:self.end_num:10]-20.0, cfo[j][::10], label='Group' + str(j + 1))
                    ax[i].set_ylabel(ylabels[i])
                    ax[i].legend(ncol=10, columnspacing=1, loc='upper left')
                    ax[i].set_yticks(ytics[i])
                    ax[i].set_ylim(ylims[i][0], ylims[i][1])  # y軸の範囲

            os.makedirs('fig/CFO/Subtraction/TimeSeries/', exist_ok=True)
            plt.savefig('fig/CFO/Subtraction/TimeSeries/SubtractionCFO_TimeSeries_' + str(self.group_type) + '.png')
            plt.show()

        return subtraction[0], subtraction[1], subtraction[2], subtraction[3]

    def subtraction_cfo_3sec(self):
        ppcfo_subtraction, rpcfo_subtraction, pfcfo_subtraction, rfcfo_subtraction = CFO.subtraction_cfo(self)

        ppcfo_subtraction_3sec = ppcfo_subtraction.reshape([len(self.cfo), -1, self.num])
        ppcfo_subtraction_3sec = np.average(ppcfo_subtraction_3sec, axis=2)

        rpcfo_subtraction_3sec = rpcfo_subtraction.reshape([len(self.cfo), -1, self.num])
        rpcfo_subtraction_3sec = np.average(rpcfo_subtraction_3sec, axis=2)

        pfcfo_subtraction_3sec = pfcfo_subtraction.reshape([len(self.cfo), -1, self.num])
        pfcfo_subtraction_3sec = np.average(pfcfo_subtraction_3sec, axis=2)

        rfcfo_subtraction_3sec = rfcfo_subtraction.reshape([len(self.cfo), -1, self.num])
        rfcfo_subtraction_3sec = np.average(rfcfo_subtraction_3sec, axis=2)

        return ppcfo_subtraction_3sec, rpcfo_subtraction_3sec, pfcfo_subtraction_3sec, rfcfo_subtraction_3sec

    def subtraction_cfo_combine(self, graph=False):
        subtraction = self.cfo[0]['i1_p_pcfo'][self.start_num:self.end_num]
        types = [['p_pcfo', 'r_pcfo'], ['p_fcfo', 'r_fcfo']]
        for type in types:
            subtraction_ = self.cfo[0]['i1_p_pcfo'][self.start_num:self.end_num]
            for i in range(len(self.cfo)):
                data = self.cfo[i]
                if self.group_type == 'dyad':
                    sub_cfo1 = np.abs(np.subtract(data['i1_' + type[0]][self.start_num:self.end_num] + data['i1_' + type[1]][self.start_num:self.end_num],
                                                  data['i2_' + type[0]][self.start_num:self.end_num] + data['i2_' + type[1]][self.start_num:self.end_num]))
                    sub_cfo2 = np.abs(np.subtract(data['i2_' + type[0]][self.start_num:self.end_num] + data['i2_' + type[1]][self.start_num:self.end_num],
                                                  data['i1_' + type[0]][self.start_num:self.end_num] + data['i1_' + type[1]][self.start_num:self.end_num]))
                    sub_cfo_ave = np.add(np.abs(sub_cfo1), np.abs(sub_cfo2)) / 2
                    subtraction_ = np.vstack((subtraction_, sub_cfo_ave))

                elif self.group_type == 'triad':
                    sub_cfo1 = np.subtract(np.subtract(2 * (data['i1_' + type[0]][self.start_num:self.end_num] + data['i1_' + type[1]][self.start_num:self.end_num]),
                                                       data['i2_' + type[0]][self.start_num:self.end_num] + data['i2_' + type[1]][self.start_num:self.end_num]),
                                           data['i3_' + type[0]][self.start_num:self.end_num] + data['i3_' + type[1]][self.start_num:self.end_num])
                    sub_cfo2 = np.subtract(np.subtract(2 * (data['i2_' + type[0]][self.start_num:self.end_num] + data['i2_' + type[1]][self.start_num:self.end_num]),
                                                       data['i1_' + type[0]][self.start_num:self.end_num] + data['i1_' + type[1]][self.start_num:self.end_num]),
                                           data['i3_' + type[0]][self.start_num:self.end_num] + data['i3_' + type[1]][self.start_num:self.end_num])
                    sub_cfo3 = np.subtract(np.subtract(2 * (data['i3_' + type[0]][self.start_num:self.end_num] + data['i3_' + type[1]][self.start_num:self.end_num]),
                                                       data['i1_' + type[0]][self.start_num:self.end_num] + data['i1_' + type[1]][self.start_num:self.end_num]),
                                           data['i2_' + type[0]][self.start_num:self.end_num] + data['i2_' + type[1]][self.start_num:self.end_num])
                    sub_cfo_ave = np.add(np.add(np.abs(sub_cfo1), np.abs(sub_cfo2)), np.abs(sub_cfo3)) / 3
                    subtraction_ = np.vstack((subtraction_, sub_cfo_ave))

                elif self.group_type == 'tetrad':
                    sub_cfo1 = np.subtract(np.subtract(np.subtract(3 * (data['i1_' + type[0]][self.start_num:self.end_num] + data['i1_' + type[1]][self.start_num:self.end_num]),
                                                                   data['i2_' + type[0]][self.start_num:self.end_num] + data['i2_' + type[1]][self.start_num:self.end_num]),
                                                       data['i3_' + type[0]][self.start_num:self.end_num] + data['i3_' + type[1]][self.start_num:self.end_num]),
                                           data['i4_' + type[0]][self.start_num:self.end_num] + data['i4_' + type[1]][self.start_num:self.end_num])
                    sub_cfo2 = np.subtract(np.subtract(np.subtract(3 * (data['i2_' + type[0]][self.start_num:self.end_num] + data['i2_' + type[1]][self.start_num:self.end_num]),
                                                                   data['i1_' + type[0]][self.start_num:self.end_num] + data['i1_' + type[1]][self.start_num:self.end_num]),
                                                       data['i3_' + type[0]][self.start_num:self.end_num] + data['i3_' + type[1]][self.start_num:self.end_num]),
                                           data['i4_' + type[0]][self.start_num:self.end_num] + data['i4_' + type[1]][self.start_num:self.end_num])
                    sub_cfo3 = np.subtract(np.subtract(np.subtract(3 * (data['i3_' + type[0]][self.start_num:self.end_num] + data['i3_' + type[1]][self.start_num:self.end_num]),
                                                                   data['i1_' + type[0]][self.start_num:self.end_num] + data['i1_' + type[1]][self.start_num:self.end_num]),
                                                       data['i2_' + type[0]][self.start_num:self.end_num] + data['i2_' + type[1]][self.start_num:self.end_num]),
                                           data['i4_' + type[0]][self.start_num:self.end_num] + data['i4_' + type[1]][self.start_num:self.end_num])
                    sub_cfo4 = np.subtract(np.subtract(np.subtract(3 * (data['i4_' + type[0]][self.start_num:self.end_num] + data['i4_' + type[1]][self.start_num:self.end_num]),
                                                                   data['i1_' + type[0]][self.start_num:self.end_num] + data['i1_' + type[1]][self.start_num:self.end_num]),
                                                       data['i2_' + type[0]][self.start_num:self.end_num] + data['i2_' + type[1]][self.start_num:self.end_num]),
                                           data['i3_' + type[0]][self.start_num:self.end_num] + data['i3_' + type[1]][self.start_num:self.end_num])
                    sub_cfo_ave = np.add(np.add(np.add(np.abs(sub_cfo1), np.abs(sub_cfo2)), np.abs(sub_cfo3)),
                                         np.abs(sub_cfo4)) / 4
                    subtraction_ = np.vstack((subtraction_, sub_cfo_ave))

            subtraction_ = np.delete(subtraction_, 0, 0)
            subtraction = np.vstack((subtraction, subtraction_))
        subtraction = np.delete(subtraction, 0, 0)
        subtraction = subtraction.reshape([2, len(self.cfo), -1])

        if graph == True:
            fig, (pcfo, fcfo) = plt.subplots(2, 1, figsize=(20, 10), dpi=150, sharex=True)

            # plt.xlim([10, 60])  # x軸の範囲
            # plt.xlim([0.28, 0.89])  # x軸の範囲
            plt.xlabel("Time (sec)")

            for i in range(len(self.cfo)):
                data = self.cfo[i]

                pcfo.plot(data['time'][self.start_num:self.end_num:10]-20.0, subtraction[0][i][::10],
                          label='Group' + str(i + 1))
                fcfo.plot(data['time'][self.start_num:self.end_num:10]-20.0, subtraction[1][i][::10],
                          label='Group' + str(i + 1))

            pcfo.set_ylabel('Subtraction PCFO (rad)')
            fcfo.set_ylabel('Subtraction FCFO (Nm)')

            pcfo.legend(ncol=10, columnspacing=1, loc='upper left')
            fcfo.legend(ncol=10, columnspacing=1, loc='upper left')

            pcfo.set_yticks(np.arange(-10, 10, 0.5))
            fcfo.set_yticks(np.arange(-8.0, 8.0, 1.0))

            pcfo.set_ylim([0, 1.0])  # y軸の範囲
            fcfo.set_ylim([0, 10.0])  # y軸の範囲

            plt.tight_layout()
            os.makedirs('fig/CFO/Subtraction/TimeSeries/Combine/', exist_ok=True)
            plt.savefig('fig/CFO/Subtraction/TimeSeries/Combine/SubtractionCFO_TimeSeries_' + str(self.group_type) + '.png')
            plt.show()

        return subtraction[0], subtraction[1]

    def subtraction_cfo_3sec_combine(self):
        pcfo_subtraction, fcfo_subtraction = CFO.subtraction_cfo_combine(self)

        pcfo_subtraction_3sec = pcfo_subtraction.reshape([len(self.cfo), -1, self.num])
        pcfo_subtraction_3sec = np.average(pcfo_subtraction_3sec, axis=2)

        fcfo_subtraction_3sec = fcfo_subtraction.reshape([len(self.cfo), -1, self.num])
        fcfo_subtraction_3sec = np.average(fcfo_subtraction_3sec, axis=2)

        return pcfo_subtraction_3sec, fcfo_subtraction_3sec


    def performance_calc(self, data, ballx, bally):
        error = np.sqrt(
            (data['targetx'][self.start_num:self.end_num] - ballx[self.start_num:self.end_num]) ** 2
            + (data['targety'][self.start_num:self.end_num] - bally[self.start_num:self.end_num]) ** 2)

        target_size = 0.03
        spent = np.where(error < target_size, 1, 0)
        # spent = numpy.where(error < self.data['targetsize'], 1, 0)

        return error, spent

    def time_series_performance_calc(self, data, ballx, bally, ballx_dot, bally_dot):
        error = np.sqrt(
            (data['targetx'][self.start_num:self.end_num] - ballx[self.start_num:self.end_num]) ** 2
            + (data['targety'][self.start_num:self.end_num] - bally[self.start_num:self.end_num]) ** 2)

        error_dot = np.sqrt(
            (data['targetx_dot'][self.start_num:self.end_num] - ballx_dot[self.start_num:self.end_num]) ** 2
            + (data['targety_dot'][self.start_num:self.end_num] - bally_dot[self.start_num:self.end_num]) ** 2)

        return error, error_dot


    def time_series_performance_calc_axis(self, data, ballx, bally, ballx_dot, bally_dot):
        error_x = np.abs(data['targetx'][self.start_num:self.end_num] - ballx[self.start_num:self.end_num])
        error_y = np.abs(data['targety'][self.start_num:self.end_num] - bally[self.start_num:self.end_num])

        error_dot_x = np.abs(data['targetx_dot'][self.start_num:self.end_num] - ballx_dot[self.start_num:self.end_num])
        error_dot_y = np.abs(data['targety_dot'][self.start_num:self.end_num] - bally_dot[self.start_num:self.end_num])

        return error_x, error_y, error_dot_x, error_dot_y

    def time_series_performance(self, mode='H-H', sigma: int = 'none', graph=False):
        error_ts = np.zeros((len(self.cfo), self.join, self.end_num-self.start_num))
        error_dot_ts = np.zeros((len(self.cfo), self.join, self.end_num-self.start_num))
        error_ts_ave = np.zeros((len(self.cfo), self.end_num-self.start_num))
        error_dot_ts_ave = np.zeros((len(self.cfo), self.end_num-self.start_num))
        error_ts_best = np.zeros((len(self.cfo), self.end_num-self.start_num))
        error_dot_ts_best = np.zeros((len(self.cfo), self.end_num-self.start_num))
        for i in range(len(self.cfo)):
            data = self.cfo[i]
            best_join = 0
            best_error = 100
            for j in range(self.join):
                interface = 'i' + str(j + 1)
                if mode == 'H-H':
                    error, error_dot = CFO.time_series_performance_calc(self, data,
                                                            data['ballx'], data['bally'],
                                                            data['ballx_dot'], data['bally_dot'])
                elif mode == 'M-M':
                    error, error_dot = CFO.time_series_performance_calc(self, data,
                                                            data[interface+'_ballx_pre'], data[interface+'_bally_pre'],
                                                            data[interface+'_ballx_pre_dot'], data[interface+'_bally_pre_dot'])
                if sigma == 'none':
                    error_ts[i][j] = error
                    error_dot_ts[i][j] = error_dot
                else:
                    error_ts[i][j] = gaussian_filter(error, sigma=sigma)
                    error_dot_ts[i][j] = gaussian_filter(error_dot, sigma=sigma)

                if np.average(error) < best_error:
                    best_error = np.average(error, axis=0)
                    best_join = j

            error_ts_ave[i] = np.average(error_ts[i], axis=0)
            error_dot_ts_ave[i] = np.average(error_dot_ts[i], axis=0)
            error_ts_best[i] = error_ts[i][best_join]
            error_dot_ts_best[i] = error_dot_ts[i][best_join]

            if graph:
                fig, ax = plt.subplots(2, 1, figsize=(10, 4), dpi=150, sharex=True)
                labels = ['Error (m)', 'Error speed (m/s$^2$)']
                performance = [error_ts[i], error_dot_ts[i]]
                performance_ave = [error_ts_ave[i], error_dot_ts_ave[i]]
                ax[0].set_title(str(self.group_type) + ' | ' + mode + ' | ' + self.trajectory_type + ' | Group' + str(i + 1))
                for j, label in enumerate(labels):
                    if mode == 'H-H':
                        ax[j].plot(data['time'][self.start_num:self.end_num:10]-20.0, performance_ave[j][::10])
                    else:
                        ax[j].plot(data['time'][self.start_num:self.end_num:10]-20.0, performance_ave[j][::10], label='Average', lw=2)
                        for k in range(self.join):
                            ax[j].plot(data['time'][self.start_num:self.end_num:10]-20.0, performance[j][k][::10], label='Model' + str(k + 1))
                        ax[j].legend(ncol=10, columnspacing=1)
                    ax[j].set_ylabel(label)
                    if j == 0:
                        ax[j].set_ylim([0, 0.1])
                    elif j == 1:
                        ax[j].set_ylim([0, 0.1])
                ax[1].set_xlim(self.starttime-20.0, self.endtime-20.0)
                ax[1].set_xlabel('Time (sec)')

                base_dir = 'fig/Performance/TimeSeries/Group/'

                if mode == 'H-H':
                    mode_dir = 'Human-Human/'
                elif mode == 'M-M':
                    mode_dir = 'Model-Model/'

                os.makedirs(base_dir + self.group_dir + self.trajectory_dir + mode_dir, exist_ok=True)
                plt.savefig(base_dir + self.group_dir + self.trajectory_dir + mode_dir + 'Performance_TimeSeries_'
                            + mode + '_' + self.trajectory_type + '_' + 'Group' + str(i + 1) + '_' + str(self.group_type) + '.png')

        if graph:
            fig, ax = plt.subplots(2, 1, figsize=(10, 4), dpi=150, sharex=True)
            labels = ['Error (m)', 'Error speed (m/s$^2$)']
            ax[0].set_title(str(self.group_type) + ' | ' + mode + ' | ' + self.trajectory_type)
            for i in range(len(self.cfo)):
                performance = [error_ts_ave[i], error_dot_ts_ave[i]]
                for j, label in enumerate(labels):
                    ax[j].plot(data['time'][self.start_num:self.end_num:10]-20.0, performance[j][::10], label='Group' + str(i + 1))
                    ax[j].set_ylabel(label)
                    ax[j].legend(ncol=10, columnspacing=1)
                    if j == 0:
                        ax[j].set_ylim([0, 0.1])
                    elif j == 1:
                        ax[j].set_ylim([0, 0.1])
                ax[1].set_xlim(self.starttime-20.0, self.endtime-20.0)
                ax[1].set_xlabel('Time (sec)')

                if mode == 'H-H':
                    mode_dir = 'Human-Human/'
                elif mode == 'M-M':
                    mode_dir = 'Model-Model/'

                base_dir = 'fig/Performance/TimeSeries/'
                os.makedirs(base_dir + self.group_dir + self.trajectory_dir + mode_dir, exist_ok=True)
                plt.savefig(base_dir + self.group_dir + self.trajectory_dir + mode_dir + 'Performance_TimeSeries_'
                            + mode + '_' + self.trajectory_type + '_' + str(self.group_type) + '.png')
            plt.show()

        # return error_ts_ave, error_dot_ts_ave
        return error_ts_best, error_dot_ts_best

    def time_series_performance_axis(self, mode='H-H', sigma: int = 'none', graph=False):
        error_ts_x = np.zeros((len(self.cfo), self.join, self.end_num-self.start_num))
        error_ts_y = np.zeros((len(self.cfo), self.join, self.end_num-self.start_num))
        error_dot_ts_x = np.zeros((len(self.cfo), self.join, self.end_num-self.start_num))
        error_dot_ts_y = np.zeros((len(self.cfo), self.join, self.end_num-self.start_num))
        error_ts_ave_x = np.zeros((len(self.cfo), self.end_num-self.start_num))
        error_ts_ave_y = np.zeros((len(self.cfo), self.end_num-self.start_num))
        error_dot_ts_ave_x = np.zeros((len(self.cfo), self.end_num-self.start_num))
        error_dot_ts_ave_y = np.zeros((len(self.cfo), self.end_num-self.start_num))
        error_ts_best_x = np.zeros((len(self.cfo), self.end_num-self.start_num))
        error_ts_best_y = np.zeros((len(self.cfo), self.end_num-self.start_num))
        error_dot_ts_best_x = np.zeros((len(self.cfo), self.end_num-self.start_num))
        error_dot_ts_best_y = np.zeros((len(self.cfo), self.end_num-self.start_num))
        for i in range(len(self.cfo)):
            data = self.cfo[i]
            best_join = 0
            best_error = 100
            for j in range(self.join):
                interface = 'i' + str(j + 1)
                if mode == 'H-H':
                    error_x, error_y, error_dot_x, error_dot_y = CFO.time_series_performance_calc_axis(self, data,
                                                                                                       data['ballx'], data['bally'],
                                                                                                       data['ballx_dot'], data['bally_dot'])
                elif mode == 'M-M':
                    error_x, error_y, error_dot_x, error_dot_y = CFO.time_series_performance_calc_axis(self, data,
                                                                                                       data[interface+'_ballx_pre'], data[interface+'_bally_pre'],
                                                                                                       data[interface+'_ballx_pre_dot'], data[interface+'_bally_pre_dot'])
                if sigma == 'none':
                    error_ts_x[i][j] = error_x
                    error_ts_y[i][j] = error_y
                    error_dot_ts_x[i][j] = error_dot_x
                    error_dot_ts_y[i][j] = error_dot_y
                else:
                    error_ts_x[i][j] = gaussian_filter(error_x, sigma=sigma)
                    error_ts_y[i][j] = gaussian_filter(error_y, sigma=sigma)
                    error_dot_ts_x[i][j] = gaussian_filter(error_dot_x, sigma=sigma)
                    error_dot_ts_y[i][j] = gaussian_filter(error_dot_y, sigma=sigma)

                if np.average(error_x) < best_error:
                    best_error = np.average(error_x, axis=0)
                    best_join = j

            error_ts_ave_x[i] = np.average(error_ts_x[i], axis=0)
            error_ts_ave_y[i] = np.average(error_ts_y[i], axis=0)
            error_dot_ts_ave_x[i] = np.average(error_dot_ts_x[i], axis=0)
            error_dot_ts_ave_y[i] = np.average(error_dot_ts_y[i], axis=0)
            error_ts_best_x[i] = error_ts_x[i][best_join]
            error_ts_best_y[i] = error_ts_y[i][best_join]
            error_dot_ts_best_x[i] = error_dot_ts_x[i][best_join]
            error_dot_ts_best_y[i] = error_dot_ts_y[i][best_join]

        # return error_ts_ave, error_dot_ts_ave
        return error_ts_best_x, error_ts_best_y, error_dot_ts_best_x, error_dot_ts_best_y

    def time_series_performance_cooperation(self, sigma: int = 'none', graph=False):
        error_ts_human, error_dot_ts_human = CFO.time_series_performance(self, mode='H-H', sigma=sigma)
        error_ts_model, error_dot_ts_model = CFO.time_series_performance(self, mode='M-M', sigma=sigma)

        error_ts = np.subtract(error_ts_human, error_ts_model)
        error_dot_ts = np.subtract(error_dot_ts_human, error_dot_ts_model)

        if graph:
            for i in range(len(self.cfo)):
                data = self.cfo[i]
                fig, ax = plt.subplots(2, 1, figsize=(10, 4), dpi=150, sharex=True)
                labels = ['Error (m)', 'Error speed (m/s$^2$)']
                performance_h = [error_ts_human[i], error_dot_ts_human[i]]
                performance_m = [error_ts_model[i], error_dot_ts_model[i]]
                ax[0].set_title(str(self.group_type) + ' | Cooperation | ' + self.trajectory_type + ' | Group' + str(i + 1))
                for j, label in enumerate(labels):
                    ax[j].plot(data['time'][self.start_num:self.end_num:10]-20.0, performance_h[j][::10], label='H-H')
                    ax[j].plot(data['time'][self.start_num:self.end_num:10]-20.0, performance_m[j][::10], label='M-M')
                    ax[j].legend(ncol=10, columnspacing=1)
                    ax[j].set_ylabel(label)
                    if j == 0:
                        ax[j].set_ylim([0, 0.1])
                    elif j == 1:
                        ax[j].set_ylim([0, 0.1])
                ax[1].set_xlim(self.starttime-20.0, self.endtime-20.0)
                ax[1].set_xlabel('Time (sec)')

                base_dir = 'fig/Performance/TimeSeries/Group/'

                os.makedirs(base_dir + self.group_dir + self.trajectory_dir + 'Cooperation/', exist_ok=True)
                plt.savefig(base_dir + self.group_dir + self.trajectory_dir + 'Cooperation/' + 'Performance_TimeSeries_'
                            + 'Group' + str(i + 1) + '_' + self.trajectory_type + '_' + str(self.group_type) + '.png')

            fig, ax = plt.subplots(2, 1, figsize=(10, 4), dpi=150, sharex=True)
            labels = ['Error (m)', 'Error speed (m/s$^2$)']
            ax[0].set_title(str(self.group_type) + ' | Cooperation | ' + self.trajectory_type)
            for i in range(len(self.cfo)):
                data = self.cfo[i]
                performance = [error_ts[i], error_dot_ts[i]]
                for j, label in enumerate(labels):
                    ax[j].plot(data['time'][self.start_num:self.end_num:10]-20.0, performance[j][::10], label='Group' + str(i + 1))
                    ax[j].set_ylabel(label)
                    ax[j].legend(ncol=10, columnspacing=1, loc='upper left')
                    if j == 0:
                        ax[j].set_ylim([-0.02, 0.02])
                    elif j == 1:
                        ax[j].set_ylim([-0.04, 0.04])
                ax[1].set_xlim(self.starttime-20.0, self.endtime-20.0)
                ax[1].set_xlabel('Time (sec)')

                plt.savefig(base_dir + self.group_dir + self.trajectory_dir + 'Cooperation/' + 'Performance_TimeSeries_'
                            + 'Cooperation_' + '_' + self.trajectory_type + '_' + str(self.group_type) + '.png')
            plt.show()

        return error_ts, error_dot_ts

    def time_series_performance_cooperation_axis(self, sigma: int = 'none', graph=False):
        error_ts_human_x, error_ts_human_y, error_dot_ts_human_x, error_dot_ts_human_y = CFO.time_series_performance_axis(self, mode='H-H', sigma=sigma)
        error_ts_model_x, error_ts_model_y, error_dot_ts_model_x, error_dot_ts_model_y = CFO.time_series_performance_axis(self, mode='M-M', sigma=sigma)

        error_ts_x = np.subtract(error_ts_human_x, error_ts_model_x)
        error_ts_y = np.subtract(error_ts_human_y, error_ts_model_y)
        error_dot_ts_x = np.subtract(error_dot_ts_human_x, error_dot_ts_model_x)
        error_dot_ts_y = np.subtract(error_dot_ts_human_y, error_dot_ts_model_y)

        return error_ts_x, error_ts_y, error_dot_ts_x, error_dot_ts_y


    def period_performance_New(self, mode='H-H', sigma: int = 'none', graph=False): #TODO will be marge with period_performance
        error_period = np.zeros((len(self.cfo), self.join, self.period))
        error_dot_period = np.zeros((len(self.cfo), self.join, self.period))
        error_period_ave = np.zeros((len(self.cfo), self.period))
        error_dot_period_ave = np.zeros((len(self.cfo), self.period))
        error_period_best = np.zeros((len(self.cfo), self.period))
        error_dot_period_best = np.zeros((len(self.cfo), self.period))
        for i in range(len(self.cfo)):
            data = self.cfo[i]
            best_join = 0
            best_error = 10000
            # best_error_dot = 10000
            for j in range(self.join):
                interface = 'i' + str(j + 1)
                if mode == 'H-H':
                    error, error_dot = CFO.time_series_performance_calc(self, data,
                                                                        data['ballx'], data['bally'],
                                                                        data['ballx_dot'], data['bally_dot'])
                elif mode == 'M-M':
                    error, error_dot = CFO.time_series_performance_calc(self, data,
                                                                        data[interface+'_ballx_pre'], data[interface+'_bally_pre'],
                                                                        data[interface+'_ballx_pre_dot'], data[interface+'_bally_pre_dot'])
                if sigma == 'none':
                    pass
                else:
                    error = gaussian_filter(error, sigma=sigma)
                    error_dot = gaussian_filter(error_dot, sigma=sigma)
                error_period[i][j] = error.reshape([self.period, -1]).mean(axis=1)
                error_dot_period[i][j] = error_dot.reshape([self.period, -1]).mean(axis=1)


                if np.average(error) < best_error:
                    best_error = np.average(error, axis=0)
                    best_join = j

            error_period_ave[i] = np.average(error_period[i], axis=0)
            error_dot_period_ave[i] = np.average(error_dot_period[i], axis=0)
            error_period_best[i] = error_period[i][best_join]
            error_dot_period_best[i] = error_dot_period[i][best_join]

            if graph:
                fig, ax = plt.subplots(2, 1, figsize=(10, 4), dpi=150, sharex=True)
                labels = ['Error (m)', 'Error speed (m/s$^2$)']
                performance = [error_period[i], error_dot_period[i]]
                performance_ave = [error_period_ave[i], error_dot_period_ave[i]]
                performance_best = [error_period_best[i], error_dot_period_best[i]]
                ax[0].set_title(str(self.group_type) + ' | ' + mode + ' | ' + self.trajectory_type+ ' | Group' + str(i + 1))
                for j, label in enumerate(labels):
                    if mode == 'H-H':
                        ax[j].plot(np.arange(self.period) + 1, performance_ave[j])
                        ax[j].scatter(np.arange(self.period) + 1, performance_ave[j], s=5, marker='x')
                    else:
                        ax[j].plot(np.arange(self.period) + 1, performance_ave[j], label='Average')
                        ax[j].scatter(np.arange(self.period) + 1, performance_ave[j], s=5, marker='x')
                        for k in range(self.join):
                            ax[j].plot(np.arange(self.period) + 1, performance[j][k], label='Model' + str(k + 1))
                            ax[j].scatter(np.arange(self.period) + 1, performance[j][k], s=5, marker='x')
                        ax[j].plot(np.arange(self.period) + 1, performance_best[j], label='Best')
                        ax[j].scatter(np.arange(self.period) + 1, performance_best[j], s=5, marker='x')
                        ax[j].legend(ncol=10, columnspacing=1)
                    ax[j].set_ylabel(label)
                    if j == 0:
                        ax[j].set_ylim([0, 0.1])
                    elif j == 1:
                        ax[j].set_ylim([0, 0.1])
                ax[1].set_xticks(np.arange(1, self.period + 1, 1))
                ax[1].set_xlim(0, self.period + 1)
                ax[1].set_xlabel('Period')

                base_dir = 'fig/Performance/Period/Group/'

                if mode == 'H-H':
                    mode_dir = 'Human-Human/'
                elif mode == 'M-M':
                    mode_dir = 'Model-Model/'

                # os.makedirs(base_dir + self.group_dir + self.trajectory_dir + mode_dir, exist_ok=True)
                # plt.savefig(base_dir + self.group_dir + self.trajectory_dir + mode_dir + 'Performance_Period_'
                #             + mode + '_' + self.trajectory_type + '_' + 'Group' + str(i + 1) + '_' + str(self.group_type) + '.png')

        if graph:
            fig, ax = plt.subplots(2, 1, figsize=(10, 4), dpi=150, sharex=True)
            labels = ['Error (m)', 'Error speed (m/s$^2$)']
            ax[0].set_title(str(self.group_type) + ' | ' + mode + ' | ' + self.trajectory_type)
            for i in range(len(self.cfo)):
                # performance = [error_period_ave[i], error_dot_period_ave[i]]
                performance = [error_period_best[i], error_dot_period_best[i]]
                for j, label in enumerate(labels):
                    ax[j].plot(np.arange(self.period) + 1, performance[j], label='Group' + str(i + 1))
                    ax[j].scatter(np.arange(self.period) + 1, performance[j], s=5, marker='x')
                    ax[j].set_ylabel(label)
                    ax[j].legend(ncol=10, columnspacing=1)
                    if j == 0:
                        ax[j].set_ylim([0, 0.1])
                    elif j == 1:
                        ax[j].set_ylim([0, 0.1])
                ax[1].set_xticks(np.arange(1, self.period + 1, 1))
                ax[1].set_xlim(0, self.period + 1)
                ax[1].set_xlabel('Period')

                if mode == 'H-H':
                    mode_dir = 'Human-Human/'
                elif mode == 'M-M':
                    mode_dir = 'Model-Model/'

                # base_dir = 'fig/Performance/Period/'
                # os.makedirs(base_dir + self.group_dir + self.trajectory_dir + mode_dir, exist_ok=True)
                # plt.savefig(base_dir + self.group_dir + self.trajectory_dir + mode_dir + 'Performance_Period_'
                #             + mode + '_' + self.trajectory_type + '_' + str(self.group_type) + '.png')
            plt.show()

        # return error_period_ave, error_dot_period_ave
        return error_period_best, error_dot_period_best


    def period_performance(self, mode='H-H', graph=False):
        error_period = np.zeros((len(self.cfo), self.join, self.period))
        error_period_ave = np.zeros((len(self.cfo), self.period))
        spent_period = np.zeros((len(self.cfo), self.join, self.period))
        spent_period_ave = np.zeros((len(self.cfo), self.period))
        for i in range(len(self.cfo)):
            data = self.cfo[i]
            for j in range(self.join):
                interface = 'i' + str(j + 1)
                if mode == 'H-H':
                    error, spent = CFO.performance_calc(self, data, data['ballx'], data['bally'])
                elif mode == 'M-M':
                    error, spent = CFO.performance_calc(self, data, data[interface+'_ballx_pre'], data[interface+'_bally_pre'])
                error_reshape = error.reshape([self.period, self.num])  # [回数][データ]にわける
                error_period[i][j] = np.sum(error_reshape, axis=1) / self.num

                spent_reshape = spent.reshape([self.period, self.num])
                spent_period_ = np.sum(spent_reshape, axis=1)
                spent_period[i][j] = spent_period_ * self.smp / self.duringtime

            spent_period_ave = np.average(spent_period, axis=1)
            error_period_ave = np.average(error_period, axis=1)

            if graph:
                fig, ax = plt.subplots(2, 1, figsize=(10, 4), dpi=150, sharex=True)
                labels = ['Error (m)', 'Time (sec)']
                performance = [error_period[i], spent_period[i]]
                performance_ave = [error_period_ave[i], spent_period_ave[i]]
                ax[0].set_title(str(self.group_type) + ' | ' + mode + ' | Group' + str(i + 1))
                for j, label in enumerate(labels):
                    if mode == 'H-H':
                        ax[j].plot(np.arange(self.period) + 1, performance_ave[j])
                        ax[j].scatter(np.arange(self.period) + 1, performance_ave[j], s=5, marker='x')
                    else:
                        ax[j].plot(np.arange(self.period) + 1, performance_ave[j], label='Average')
                        ax[j].scatter(np.arange(self.period) + 1, performance_ave[j], s=5, marker='x')
                        for k in range(self.join):
                            ax[j].plot(np.arange(self.period) + 1, performance[j][k], label='Model' + str(i + 1))
                            ax[j].scatter(np.arange(self.period) + 1, performance[j][k], s=5, marker='x')
                        ax[j].legend(ncol=10, columnspacing=1)
                    ax[j].set_ylabel(label)
                    if j == 0:
                        ax[j].set_ylim([0, 0.1])
                    elif j == 1:
                        ax[j].set_ylim([0, 3.0])
                ax[1].set_xticks(np.arange(1, self.period + 1, 1))
                ax[1].set_xlim(0, self.period + 1)
                ax[1].set_xlabel('Period')

                base_dir = 'fig/Performance/Period/Group/'

                if mode == 'H-H':
                    mode_dir = 'Human-Human/'
                elif mode == 'M-M':
                    mode_dir = 'Model-Model/'

                os.makedirs(base_dir + self.group_dir + self.trajectory_dir + mode_dir, exist_ok=True)
                plt.savefig(base_dir + self.group_dir + self.trajectory_dir + mode_dir + 'Performance_Period_'
                            + mode + '_' + self.trajectory_type + '_' + 'Group' + str(i + 1) + '_' + str(self.group_type) + '.png')

        if graph:
            fig, ax = plt.subplots(2, 1, figsize=(10, 4), dpi=150, sharex=True)
            labels = ['Error (m)', 'Time (sec)']
            ax[0].set_title(str(self.group_type) + ' | ' + mode + ' | ' + self.trajectory_type)
            for i in range(len(self.cfo)):
                performance = [error_period[i], spent_period[i]]
                for j, label in enumerate(labels):
                    ax[j].plot(np.arange(self.period) + 1, performance[j], label='Group' + str(i + 1))
                    ax[j].scatter(np.arange(self.period) + 1, performance[j], s=5, marker='x')
                    ax[j].set_ylabel(label)
                    ax[j].legend(ncol=10, columnspacing=1)
                    if j == 0:
                        ax[j].set_ylim([0, 0.1])
                    elif j == 1:
                        ax[j].set_ylim([0, 3.0])
                ax[1].set_xticks(np.arange(1, self.period + 1, 1))
                ax[1].set_xlim(0, self.period + 1)
                ax[1].set_xlabel('Period')

                if mode == 'H-H':
                    mode_dir = 'Human-Human/'
                elif mode == 'M-M':
                    mode_dir = 'Model-Model/'

                base_dir = 'fig/Performance/Period/'
                os.makedirs(base_dir + self.group_dir + self.trajectory_dir + mode_dir, exist_ok=True)
                plt.savefig(base_dir + self.group_dir + self.trajectory_dir + mode_dir + 'Performance_Period_'
                            + mode + '_' + self.trajectory_type + '_' + str(self.group_type) + '.png')
            plt.show()

        return error_period_ave, spent_period_ave


    def period_performance_cooperation(self, graph=False):
        error_period_human, spent_period_human = CFO.period_performance(self, mode='H-H')
        error_period_model, spent_period_model = CFO.period_performance(self, mode='M-M')

        error_period = np.subtract(error_period_human, error_period_model)
        spent_period = np.subtract(spent_period_human, spent_period_model)

        if graph:
            for i in range(len(self.cfo)):
                fig, ax = plt.subplots(2, 1, figsize=(10, 4), dpi=150, sharex=True)
                labels = ['Error (m)', 'Time (sec)']
                performance_h = [error_period_human[i], spent_period_human[i]]
                performance_m = [error_period_model[i], spent_period_model[i]]
                ax[0].set_title(str(self.group_type) + ' | Cooperation | Group' + str(i + 1))
                for j, label in enumerate(labels):
                    ax[j].plot(np.arange(self.period) + 1, performance_h[j], label='H-H')
                    ax[j].scatter(np.arange(self.period) + 1, performance_h[j], s=5, marker='x')
                    ax[j].plot(np.arange(self.period) + 1, performance_m[j], label='M-M')
                    ax[j].scatter(np.arange(self.period) + 1, performance_m[j], s=5, marker='x')
                    ax[j].legend(ncol=10, columnspacing=1)
                    ax[j].set_ylabel(label)
                    if j == 0:
                        ax[j].set_ylim([0, 0.1])
                    elif j == 1:
                        ax[j].set_ylim([0, 3.0])
                ax[1].set_xticks(np.arange(1, self.period + 1, 1))
                ax[1].set_xlim(0, self.period + 1)
                ax[1].set_xlabel('Period')

                base_dir = 'fig/Performance/Period/Group/'
                if str(self.group_type) == 'dyad':
                    group_dir = 'dyad/'
                elif str(self.group_type) == 'triad':
                    group_dir = 'triad/'
                elif str(self.group_type) == 'tetrad':
                    group_dir = 'tetrad/'

                os.makedirs(base_dir + group_dir + 'Cooperation/', exist_ok=True)
                plt.savefig(base_dir + group_dir + 'Cooperation/' + 'Performance_Period_'
                            + 'Group' + str(i + 1) + '_' + str(self.group_type) + '.png')

            fig, ax = plt.subplots(2, 1, figsize=(10, 4), dpi=150, sharex=True)
            labels = ['Error (m)', 'Time (sec)']
            ax[0].set_title(str(self.group_type) + ' | Cooperation')
            for i in range(len(self.cfo)):
                performance = [error_period[i], spent_period[i]]
                for j, label in enumerate(labels):
                    ax[j].plot(np.arange(self.period) + 1, performance[j], label='Group' + str(i + 1))
                    ax[j].scatter(np.arange(self.period) + 1, performance[j], s=5, marker='x')
                    ax[j].set_ylabel(label)
                    ax[j].legend(ncol=10, columnspacing=1, loc='upper left')
                    if j == 0:
                        ax[j].set_ylim([-0.02, 0.02])
                    elif j == 1:
                        ax[j].set_ylim([-1.0, 1.0])
                ax[1].set_xticks(np.arange(1, self.period + 1, 1))
                ax[1].set_xlim(0, self.period + 1)
                ax[1].set_xlabel('Period')

                plt.savefig(base_dir + group_dir + 'Cooperation/' + 'Performance_Period_'
                            + 'Cooperation_' + str(self.group_type) + '.png')
            plt.show()

        return error_period, spent_period

    def period_performance_cooperation_and_solo(self, graph=False):
        error_period_human, spent_period_human = CFO.period_performance(self, mode='H-H')
        error_period_model, spent_period_model = CFO.period_performance(self, mode='M-M')

        error_period_solo = []
        spent_period_solo = []
        for i in range(len(self.cfo)):
            data = self.cfo[i]
            error_period_solo.append([])
            spent_period_solo.append([])
            for j in range(self.join):
                interfacenum = 'i' + str(j + 1) + '_'

                error, spent = CFO.performance_calc(self, data, data[interfacenum + 'ballx_solo'], data[interfacenum + 'bally_solo'])
                error_reshape = error.reshape([self.period, self.num])  # [回数][データ]にわける
                # error_period = np.sum(error_reshape, axis=1) # 回数ごとに足す
                error_period_solo[i].append(np.sum(error_reshape, axis=1) / self.num)

                spent_reshape = spent.reshape([self.period, self.num])
                spent_period_ = np.sum(spent_reshape, axis=1)
                # spent_period = spent_period_ * self.smp
                spent_period_solo[i].append(spent_period_ * self.smp)

        if graph:
            for i in range(len(self.cfo)):
                fig, ax = plt.subplots(2, 1, figsize=(10, 4), dpi=150, sharex=True)
                labels = ['Error (m)', 'Time (sec)']
                performance_h = [error_period_human[i], spent_period_human[i]]
                performance_m = [error_period_model[i], spent_period_model[i]]
                performance_s = [error_period_solo[i], spent_period_solo[i]]
                performance_ave = [np.average(error_period_solo[i], axis=0), np.average(spent_period_solo[i], axis=0)]
                ax[0].set_title(str(self.group_type) + ' | Cooperation | Group' + str(i + 1))
                for j, label in enumerate(labels):
                    ax[j].plot(np.arange(self.period) + 1, performance_h[j], label='H-H')
                    ax[j].scatter(np.arange(self.period) + 1, performance_h[j], s=5, marker='x')
                    ax[j].plot(np.arange(self.period) + 1, performance_m[j], label='M-M')
                    ax[j].scatter(np.arange(self.period) + 1, performance_m[j], s=5, marker='x')
                    ax[j].plot(np.arange(self.period) + 1, performance_ave[j], label='SM_Ave', color='black')
                    ax[j].scatter(np.arange(self.period) + 1, performance_ave[j], s=5, marker='x', color='black')
                    ax[j].set_ylabel(label)
                    for k, ps in enumerate(performance_s[j]):
                        ax[j].plot(np.arange(self.period) + 1, ps, label='Solo' + str(k + 1))
                        ax[j].scatter(np.arange(self.period) + 1, ps, s=5, marker='x')

                    ax[j].legend(ncol=10, columnspacing=1)
                    if j == 0:
                        ax[j].set_ylim([0, 0.1])
                    elif j == 1:
                        ax[j].set_ylim([0, 3.0])
                ax[1].set_xticks(np.arange(1, self.period + 1, 1))
                ax[1].set_xlim(0, self.period + 1)
                ax[1].set_xlabel('Period')

                base_dir = 'fig/Performance/Period/Group/'
                if str(self.group_type) == 'dyad':
                    group_dir = 'dyad/'
                elif str(self.group_type) == 'triad':
                    group_dir = 'triad/'
                elif str(self.group_type) == 'tetrad':
                    group_dir = 'tetrad/'

                # os.makedirs(base_dir + group_dir + 'Cooperation/', exist_ok=True)
                # plt.savefig(base_dir + group_dir + 'Cooperation/' + 'Performance_Period_'
                #             + 'Group' + str(i + 1) + '_' + str(self.group_type) + '.png')

            plt.show()


    def performance_calc_each_axis(self, data, ballx, bally):
        errorx = np.abs(data['targetx'][self.start_num:self.end_num] - ballx[self.start_num:self.end_num])
        errory = np.abs(data['targety'][self.start_num:self.end_num] - bally[self.start_num:self.end_num])

        target_size = 0.03
        spentx = np.where(errorx < target_size, 1, 0)
        spenty = np.where(errory < target_size, 1, 0)
        # spent = numpy.where(error < self.data['targetsize'], 1, 0)

        return errorx, errory, spentx, spenty

    def period_performance_each_axis(self, mode='H-H', graph=False):
        errorx_period = []
        errory_period = []
        spentx_period = []
        spenty_period = []
        for i in range(len(self.cfo)):
            data = self.cfo[i]
            if mode == 'H-H':
                errorx, errory, spentx, spenty = CFO.performance_calc_each_axis(self, data, data['ballx'], data['bally'])
            elif mode == 'M-M':
                errorx, errory, spentx, spenty = CFO.performance_calc_each_axis(self, data, data['ballx_pre'],
                                                                                data['bally_pre'])

            errorx_reshape = errorx.reshape([self.period, self.num])  # [回数][データ]にわける
            errory_reshape = errory.reshape([self.period, self.num])  # [回数][データ]にわける
            # error_period = np.sum(error_reshape, axis=1) # 回数ごとに足す
            errorx_period.append(np.sum(errorx_reshape, axis=1) / self.num)
            errory_period.append(np.sum(errory_reshape, axis=1) / self.num)

            spentx_reshape = spentx.reshape([self.period, self.num])
            spenty_reshape = spenty.reshape([self.period, self.num])
            spentx_period_ = np.sum(spentx_reshape, axis=1)
            spenty_period_ = np.sum(spenty_reshape, axis=1)
            # spent_period = spent_period_ * self.smp
            spentx_period.append(spentx_period_ * self.smp)
            spenty_period.append(spenty_period_ * self.smp)

            if graph == True:
                fig, ax = plt.subplots(4, 1, figsize=(10, 6), dpi=150, sharex=True)
                labels = ['Error (m)', 'Time (sec)']
                axis = ['x-axis', 'y-axis']
                performance = [[errorx_period[i], errory_period[i]], [spentx_period[i], spenty_period[i]]]
                ax[0].set_title(str(self.group_type) + ' | ' + mode + ' | Group' + str(i + 1))
                for j, label in enumerate(labels):
                    for k, p in enumerate(performance[j]):
                        ax[j*2+k].plot(np.arange(self.period) + 1, p)
                        ax[j*2+k].scatter(np.arange(self.period) + 1, p, s=5, marker='x')
                        ax[j*2+k].set_ylabel(axis[k] + ' ' + label)
                        if j == 0:
                            ax[j*2+k].set_ylim([0, 0.1])
                        elif j == 1:
                            ax[j*2+k].set_ylim([0, 3.0])
                ax[3].set_xticks(np.arange(1, self.period + 1, 1))
                ax[3].set_xlim(0, self.period + 1)
                ax[3].set_xlabel('Period')

                base_dir = 'fig/Performance/Period/EachAxis/Group/'
                if str(self.group_type) == 'dyad':
                    group_dir = 'dyad/'
                elif str(self.group_type) == 'triad':
                    group_dir = 'triad/'
                elif str(self.group_type) == 'tetrad':
                    group_dir = 'tetrad/'
                if mode == 'H-H':
                    mode_dir = 'Human-Human/'
                elif mode == 'M-M':
                    mode_dir = 'Model-Model/'

                os.makedirs(base_dir + group_dir + mode_dir, exist_ok=True)
                plt.savefig(base_dir + group_dir + mode_dir + 'Performance_Period_eachaxis_'
                            + mode + '_' + 'Group' + str(i + 1) + '_' + str(self.group_type) + '.png')

        if graph == True:
            fig, ax = plt.subplots(4, 1, figsize=(10, 6), dpi=150, sharex=True)
            labels = ['Error (m)', 'Time (sec)']
            axis = ['x-axis', 'y-axis']
            ax[0].set_title(str(self.group_type) + ' | ' + mode)
            for i in range(len(self.cfo)):
                performance = [[errorx_period[i], errory_period[i]], [spentx_period[i], spenty_period[i]]]
                for j, label in enumerate(labels):
                    for k, p in enumerate(performance[j]):
                        ax[j*2+k].plot(np.arange(self.period) + 1, p, label='Group' + str(i + 1))
                        ax[j*2+k].scatter(np.arange(self.period) + 1, p, s=5, marker='x')
                        ax[j*2+k].legend(ncol=10, columnspacing=1)
                        if j == 0:
                            ax[j*2+k].set_ylim([0, 0.1])
                        elif j == 1:
                            ax[j*2+k].set_ylim([0, 3.0])
                ax[3].set_xticks(np.arange(1, self.period + 1, 1))
                ax[3].set_xlim(0, self.period + 1)
                ax[3].set_xlabel('Period')
                plt.savefig(base_dir + group_dir + mode_dir + 'Performance_Period_eachaxis_'
                            + mode + '_' + str(self.group_type) + '.png')
            plt.show()

        return errorx_period, errory_period, spentx_period, spenty_period

    def period_performance_cooperation_each_axis(self, graph=False):
        errorx_period_human, errory_period_human, spentx_period_human, spenty_period_human = CFO.period_performance_each_axis(self, mode='H-H')
        errorx_period_model, errory_period_model, spentx_period_model, spenty_period_model = CFO.period_performance_each_axis(self, mode='M-M')

        errorx_period = np.subtract(errorx_period_human, errorx_period_model)
        errory_period = np.subtract(errory_period_human, errory_period_model)
        spentx_period = np.subtract(spentx_period_human, spentx_period_model)
        spenty_period = np.subtract(spenty_period_human, spenty_period_model)

        if graph:
            for i in range(len(self.cfo)):
                fig, ax = plt.subplots(4, 1, figsize=(10, 6), dpi=150, sharex=True)
                labels = ['Error (m)', 'Time (sec)']
                axis = ['x-axis', 'y-axis']
                performance_human = [[errorx_period_human[i], errory_period_human[i]], [spentx_period_human[i], spenty_period_human[i]]]
                performance_model = [[errorx_period_model[i], errory_period_model[i]], [spentx_period_model[i], spenty_period_model[i]]]
                ax[0].set_title(str(self.group_type) + ' | Cooperation | Group' + str(i + 1))
                for j, label in enumerate(labels):
                    k = 0
                    for p_h, p_m in zip(performance_human[j], performance_model[j]):
                        ax[j*2+k].plot(np.arange(self.period) + 1, p_h, label='H-H')
                        ax[j*2+k].scatter(np.arange(self.period) + 1, p_h, s=5, marker='x')
                        ax[j*2+k].plot(np.arange(self.period) + 1, p_m, label='M-M')
                        ax[j*2+k].scatter(np.arange(self.period) + 1, p_m, s=5, marker='x')
                        ax[j*2+k].set_ylabel(axis[k] + ' ' + label)
                        ax[j*2+k].legend(ncol=10, columnspacing=1)
                        if j == 0:
                            ax[j*2+k].set_ylim([0, 0.1])
                        elif j == 1:
                            ax[j*2+k].set_ylim([0, 3.0])
                        k += 1
                ax[3].set_xticks(np.arange(1, self.period + 1, 1))
                ax[3].set_xlim(0, self.period + 1)
                ax[3].set_xlabel('Period')

                base_dir = 'fig/Performance/Period/EachAxis/Group/'
                if str(self.group_type) == 'dyad':
                    group_dir = 'dyad/'
                elif str(self.group_type) == 'triad':
                    group_dir = 'triad/'
                elif str(self.group_type) == 'tetrad':
                    group_dir = 'tetrad/'

                os.makedirs(base_dir + group_dir + 'Cooperation/', exist_ok=True)
                plt.savefig(base_dir + group_dir + 'Cooperation/' + 'Performance_Period_eachaxis_'
                            + 'Cooperation_' + 'Group' + str(i + 1) + '_' + str(self.group_type) + '.png')

            fig, ax = plt.subplots(4, 1, figsize=(10, 6), dpi=150, sharex=True)
            labels = ['Error (m)', 'Time (sec)']
            axis = ['x-axis', 'y-axis']
            ax[0].set_title(str(self.group_type) + ' | Cooperation')
            for i in range(len(self.cfo)):
                performance = [[errorx_period[i], errory_period[i]], [spentx_period[i], spenty_period[i]]]
                for j, label in enumerate(labels):
                    for k, p in enumerate(performance[j]):
                        ax[j*2+k].plot(np.arange(self.period) + 1, p, label='Group' + str(i + 1))
                        ax[j*2+k].scatter(np.arange(self.period) + 1, p, s=5, marker='x')
                        ax[j*2+k].legend(ncol=10, columnspacing=1, loc='upper left')
                        ax[j*2+k].set_ylabel(axis[k] + ' ' + label)
                        if j == 0:
                            ax[j*2+k].set_ylim([-0.02, 0.02])
                        elif j == 1:
                            ax[j*2+k].set_ylim([-1.0, 1.0])
                ax[3].set_xticks(np.arange(1, self.period + 1, 1))
                ax[3].set_xlim(0, self.period + 1)
                ax[3].set_xlabel('Period')
                plt.savefig(base_dir + group_dir + 'Cooperation/' + 'Performance_Period_eachaxis_'
                            + 'Cooperation_' + str(self.group_type) + '.png')
            plt.show()

        return errorx_period, errory_period, spentx_period, spenty_period

    def each_ocfo_performance(self, mode='normal'):
        error_period, spend_period = CFO.period_performance_cooperation(self)
        ecfo = CFO.period_ecfo(self, mode)
        inecfo = CFO.period_inecfo(self, mode)

        df = []
        for i in range(len(self.cfo)):
            df.append(pd.DataFrame({
                'error': error_period[i],
                'spend': spend_period[i],
                'ECFO': ecfo[i],
                'Ineffective CFO': inecfo[i],
            }))

            df[i]['Group'] = 'Group' + str(i + 1)

        df_all = pd.concat([i for i in df], axis=0)
        # print(df_all)

        marker_size = 5

        fig = plt.figure(figsize=(4, 4), dpi=200)

        xdata = ['ECFO', 'Ineffective CFO']
        ydata = ['error', 'spend']

        xlim = [[-1.5, 1.5],  # Ecfo
                [0.0, 4.0]]  # InECFO
        ylim = [[-0.015, 0.015],  # error
                [-0.3, 0.4]]  # spend

        for i in range(2):
            for j in range(2):
                ax = fig.add_subplot(2, 2, i * 2 + j + 1)
                ax.set_xlim(xlim[i][0], xlim[i][1])
                ax.set_ylim(ylim[j][0], ylim[j][1])
                g = sns.scatterplot(data=df_all, x=xdata[i], y=ydata[j], hue='Group', s=marker_size)
                for lh in g.legend_.legendHandles:
                    lh.set_alpha(1)
                    lh._sizes = [10]
        plt.tight_layout()
        plt.savefig('fig/each_ocfo_performance_' + str(mode) + '_' + str(self.group_type) + '.png')
        plt.show()

    # def calc_ocfo(self):
    #     for i in range(len(self.cfo)):
    #         data = self.cfo[i]
    #         task_angle = np.arctan2(data['targety'][self.start_num:self.end_num] - data['bally'][self.start_num:self.end_num],
    #                                 data['targetx'][self.start_num:self.end_num] - data['ballx'][self.start_num:self.end_num])
    #         for j in range(self.join):
    #
    #




    def ocfo_performance(self):
        error_period, spend_period = CFO.period_performance_cooperation(self)
        ecfo = CFO.period_ecfo(self)
        inecfo = CFO.period_inecfo(self)

        df = []
        for i in range(len(self.cfo)):
            df.append(pd.DataFrame({
                'error': error_period[i],
                'spend': spend_period[i],
                'Effective CFO': ecfo[i],
                'Ineffective CFO': inecfo[i],
            }))

            df[i]['Group'] = 'Group' + str(i + 1)

        df_all = pd.concat([i for i in df], axis=0)
        # print(df_all)

        marker_size = 5

        fig = plt.figure(figsize=(6, 3), dpi=300)

        performance = ['error', 'spend']
        label = ['Error (m)', 'Spent time (sec)']

        for i in range(2):
            ax = fig.add_subplot(1, 2, i + 1)
            ax.set_xlim(0.0, 3.0)
            ax.set_ylim(-3.0, 3.0)
            points = ax.scatter(df_all['Ineffective CFO'], df_all['Effective CFO'], c=df_all[performance[i]],
                                s=marker_size, cmap="Blues")
            plt.colorbar(points, ax=ax, label=label[i])
            ax.set_xlabel('Ineffective CFO')
            ax.set_ylabel('Effective CFO')

        plt.tight_layout()
        plt.show()

    def sum_sub_performance(self):
        error_period, spend_period = CFO.period_performance_cooperation(self)
        ppcfo_summation_3sec, rpcfo_summation_3sec, pfcfo_summation_3sec, rfcfo_summation_3sec = CFO.summation_cfo_3sec(
            self)
        ppcfo_babs_summation_3sec, rpcfo_babs_summation_3sec, pfcfo_babs_summation_3sec, rfcfo_babs_summation_3sec = CFO.summation_cfo_3sec(
            self, 'b_abs')
        ppcfo_aabs_summation_3sec, rpcfo_aabs_summation_3sec, pfcfo_aabs_summation_3sec, rfcfo_aabs_summation_3sec = CFO.summation_cfo_3sec(
            self, 'a_abs')
        ppcfo_subtraction_3sec, rpcfo_subtraction_3sec, pfcfo_subtraction_3sec, rfcfo_subtraction_3sec = CFO.subtraction_cfo_3sec(
            self)

        df = []
        for i in range(len(self.cfo)):
            performance = ['Cooperative Error', 'Cooperative Time']
            df.append(pd.DataFrame({
                performance[0]: error_period[i],
                performance[1]: spend_period[i],
                'Pitch Summation PCFO': ppcfo_summation_3sec[i],
                'Roll Summation PCFO': rpcfo_summation_3sec[i],
                'Pitch Summation FCFO': pfcfo_summation_3sec[i],
                'Roll Summation FCFO': rfcfo_summation_3sec[i],
                'Before abs. Pitch Summation PCFO': ppcfo_babs_summation_3sec[i],
                'Before abs. Roll Summation PCFO': rpcfo_babs_summation_3sec[i],
                'Before abs. Pitch Summation FCFO': pfcfo_babs_summation_3sec[i],
                'Before abs. Roll Summation FCFO': rfcfo_babs_summation_3sec[i],
                'After abs. Pitch Summation PCFO': ppcfo_aabs_summation_3sec[i],
                'After abs. Roll Summation PCFO': rpcfo_aabs_summation_3sec[i],
                'After abs. Pitch Summation FCFO': pfcfo_aabs_summation_3sec[i],
                'After abs. Roll Summation FCFO': rfcfo_aabs_summation_3sec[i],
                'Pitch Subtraction PCFO': ppcfo_subtraction_3sec[i],
                'Roll Subtraction PCFO': rpcfo_subtraction_3sec[i],
                'Pitch Subtraction FCFO': pfcfo_subtraction_3sec[i],
                'Roll Subtraction FCFO': rfcfo_subtraction_3sec[i],
            }))

            df[i]['Group'] = 'Group' + str(i + 1)

        df_all = pd.concat([i for i in df], axis=0)
        # print(df_all)

        marker_size = 20

        label = ['normal', 'Before abs.', 'After abs.']
        outputlabel = ['NoABS', 'BeforeABS', 'AfterABS']

        cmap = ['Blues_r', 'Blues']
        ylabel = [
            ['Pitch Summation PCFO', 'Roll Summation PCFO', 'Pitch Summation FCFO', 'Roll Summation FCFO'],
            ['Before abs. Pitch Summation PCFO', 'Before abs. Roll Summation PCFO', 'Before abs. Pitch Summation FCFO',
             'Before abs. Roll Summation FCFO'],
            ['After abs. Pitch Summation PCFO', 'After abs. Roll Summation PCFO', 'After abs. Pitch Summation FCFO',
             'After abs. Roll Summation FCFO'],
        ]
        xlabel = ['Pitch Subtraction PCFO', 'Roll Subtraction PCFO', 'Pitch Subtraction FCFO', 'Roll Subtraction FCFO']

        xlim = [0.2, 0.2, 4.0, 3.0]

        ylim = [
            [
                [-0.04, 0.04],
                [-0.04, 0.04],
                [-0.2, 0.2],
                [-0.2, 0.2],
            ],
            [
                [0.0, 0.15],
                [0.0, 0.15],
                [0.0, 2.0],
                [0.0, 1.25],
            ],
            [
                [0.0, 0.12],
                [0.0, 0.12],
                [0.0, 0.7],
                [0.0, 0.6],
            ]
        ]

        for i in range(3):
            fig = plt.figure(figsize=(15, 7), dpi=100)
            for j in range(2):
                for k in range(4):
                    ax = fig.add_subplot(2, 4, 4 * j + k + 1)
                    ax.set_xlim(0, xlim[k])
                    ax.set_ylim(ylim[i][k][0], ylim[i][k][1])
                    points = ax.scatter(df_all[xlabel[k]], df_all[ylabel[i][k]], c=df_all[performance[j]],
                                        s=marker_size, cmap=cmap[j])
                    plt.colorbar(points, ax=ax, label=performance[j])
                    ax.set_xlabel(xlabel[k])
                    ax.set_ylabel(ylabel[i][k])

            plt.tight_layout()
            os.makedirs('fig/CFO-Performance/SummationCFO_SubtractionCFO', exist_ok=True)
            plt.savefig('fig/CFO-Performance/SummationCFO_SubtractionCFO/SummationCFO_SubtractionCFO_Performance_' + str(outputlabel[i]) + '_' + str(self.group_type) + '.png')
        plt.show()

    def sum_sub(self):
        ppcfo_summation_3sec, rpcfo_summation_3sec, pfcfo_summation_3sec, rfcfo_summation_3sec = CFO.summation_cfo_3sec(
            self)
        ppcfo_babs_summation_3sec, rpcfo_babs_summation_3sec, pfcfo_babs_summation_3sec, rfcfo_babs_summation_3sec = CFO.summation_cfo_3sec(
            self, 'b_abs')
        ppcfo_aabs_summation_3sec, rpcfo_aabs_summation_3sec, pfcfo_aabs_summation_3sec, rfcfo_aabs_summation_3sec = CFO.summation_cfo_3sec(
            self, 'a_abs')
        ppcfo_subtraction_3sec, rpcfo_subtraction_3sec, pfcfo_subtraction_3sec, rfcfo_subtraction_3sec = CFO.subtraction_cfo_3sec(
            self)

        df = []
        for i in range(len(self.cfo)):
            df.append(pd.DataFrame({
                'Pitch Summation PCFO': ppcfo_summation_3sec[i],
                'Roll Summation PCFO': rpcfo_summation_3sec[i],
                'Pitch Summation FCFO': pfcfo_summation_3sec[i],
                'Roll Summation FCFO': rfcfo_summation_3sec[i],
                'Before abs. Pitch Summation PCFO': ppcfo_babs_summation_3sec[i],
                'Before abs. Roll Summation PCFO': rpcfo_babs_summation_3sec[i],
                'Before abs. Pitch Summation FCFO': pfcfo_babs_summation_3sec[i],
                'Before abs. Roll Summation FCFO': rfcfo_babs_summation_3sec[i],
                'After abs. Pitch Summation PCFO': ppcfo_aabs_summation_3sec[i],
                'After abs. Roll Summation PCFO': rpcfo_aabs_summation_3sec[i],
                'After abs. Pitch Summation FCFO': pfcfo_aabs_summation_3sec[i],
                'After abs. Roll Summation FCFO': rfcfo_aabs_summation_3sec[i],
                'Pitch Subtraction PCFO': ppcfo_subtraction_3sec[i],
                'Roll Subtraction PCFO': rpcfo_subtraction_3sec[i],
                'Pitch Subtraction FCFO': pfcfo_subtraction_3sec[i],
                'Roll Subtraction FCFO': rfcfo_subtraction_3sec[i],
            }))

            df[i]['Group'] = 'Group' + str(i + 1)

        df_all = pd.concat([i for i in df], axis=0)
        # print(df_all)

        marker_size = 5

        label = ['NoABS', 'BeforeABS', 'AfterABS']

        cmap = ['Blues']
        ylabel = [
            ['Pitch Summation PCFO', 'Roll Summation PCFO', 'Pitch Summation FCFO', 'Roll Summation FCFO'],
            ['Before abs. Pitch Summation PCFO', 'Before abs. Roll Summation PCFO', 'Before abs. Pitch Summation FCFO',
             'Before abs. Roll Summation FCFO'],
            ['After abs. Pitch Summation PCFO', 'After abs. Roll Summation PCFO', 'After abs. Pitch Summation FCFO',
             'After abs. Roll Summation FCFO'],
        ]
        xlabel = ['Pitch Subtraction PCFO', 'Roll Subtraction PCFO', 'Pitch Subtraction FCFO', 'Roll Subtraction FCFO']

        xlim = [0.3, 0.3, 5.0, 5.0]

        ylim = [
            [
                [-0.04, 0.04],
                [-0.04, 0.04],
                [-0.2, 0.2],
                [-0.2, 0.2],
            ],
            [
                [0.0, 0.2],
                [0.0, 0.2],
                [0.0, 2.0],
                [0.0, 2.0],
            ],
            [
                [0.0, 0.2],
                [0.0, 0.2],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        ]

        for i in range(3):
            fig = plt.figure(figsize=(8, 8), dpi=100)
            for k in range(4):
                ax = fig.add_subplot(2, 2, k + 1)
                ax.set_xlim(0, xlim[k])
                ax.set_ylim(ylim[i][k][0], ylim[i][k][1])
                points = ax.scatter(df_all[xlabel[k]], df_all[ylabel[i][k]], s=marker_size, cmap=cmap)
                ax.set_xlabel(xlabel[k])
                ax.set_ylabel(ylabel[i][k])
                r2 = np.corrcoef(df_all[xlabel[k]], df_all[ylabel[i][k]])
                ax.text(0.99, 0.02, '$r = {:.2f}$'.format(r2[0][1]), horizontalalignment='right',
                        transform=ax.transAxes, fontsize="medium")

                plt.tight_layout()
            # os.makedirs('fig/CFO/Summation-Subtraction/', exist_ok=True)
            # plt.savefig('fig/CFO/Summation-Subtraction/SummationCFO-SubtractionCFO_' + str(label[i]) + '_' + str(self.group_type) + '.png')
        plt.show()

    def period_ecfo(self, mode='normal'):
        ecfo = []
        for i in range(len(self.cfo)):
            data = self.cfo[i]
            if mode == 'error':
                error, spent = CFO.performance_calc(self, data, data['ballx'], data['bally'])
                ecfo.append(
                    CFO.period_calculation_consider_error(self, data['ecfo'][self.start_num:self.end_num], spent))
            else:
                ecfo.append(CFO.calc_period_calculator(self, data['ecfo'][self.start_num:self.end_num]))

        return ecfo

    def period_inecfo(self, mode='normal'):
        inecfo = []
        for i in range(len(self.cfo)):
            data = self.cfo[i]
            if mode == 'error':
                error, spent = CFO.performance_calc(self, data, data['ballx'], data['bally'])
                inecfo.append(
                    CFO.period_calculation_consider_error(self, np.abs(data['inecfo'][self.start_num:self.end_num]),
                                                          spent))
            else:
                inecfo.append(CFO.calc_period_calculator(self, np.abs(data['inecfo'][self.start_num:self.end_num])))
        return inecfo

    def calc_period_calculator(self, data):
        if data.shape[0] == self.period * self.num:
            data_reshape = data.reshape([self.period, self.num])  # [ピリオド][データ]にわける
            data_period = np.sum(data_reshape, axis=1) / self.num

        if data.shape[0] == len(self.cfo):
            data_reshape = data.reshape([-1, self.period, self.num])  # [グループ][ピリオド][データ]にわける
            # print(data_reshape.shape)
            data_period = np.sum(data_reshape, axis=2) / self.num
            # print(data_period.shape)

        return data_period

    def period_calculation_consider_error(self, data, spend):
        spent_inv = np.where(spend == 0, 1, 0)
        data_consider = data * spent_inv
        data_reshape = data_consider.reshape([self.period, self.num])  # [回数][データ]にわける
        count = np.zeros(len(data_reshape))
        for i in range(len(data_reshape)):
            count[i] = self.num - np.count_nonzero(data_reshape[i])

        data_period = np.sum(data_reshape, axis=1) / count

        return data_period

    def summation_performance(self, mode='no_abs'):
        error_period, spend_period = CFO.period_performance_cooperation(self)
        ppcfo_summation_3sec, rpcfo_summation_3sec, pfcfo_summation_3sec, rfcfo_summation_3sec = CFO.summation_cfo_3sec(
            self, mode)
        if mode == 'no_abs':
            xlabel = ['Pitch Summation PCFO', 'Roll Summation PCFO', 'Pitch Summation FCFO', 'Roll Summation FCFO']
            xlim = [
                [-0.04, 0.04],
                [-0.04, 0.04],
                [-0.2, 0.2],
                [-0.2, 0.2],
            ]
        elif mode == 'b_abs':
            xlabel = ['Before abs. Pitch Summation PCFO', 'Before abs. Roll Summation PCFO',
                      'Before abs. Pitch Summation FCFO',
                      'Before abs. Roll Summation FCFO']
            xlim = [
                [0.0, 0.15],
                [0.0, 0.15],
                [0.0, 2.0],
                [0.0, 1.25],
            ]
        elif mode == 'a_abs':
            xlabel = ['After abs. Pitch Summation PCFO', 'After abs. Roll Summation PCFO',
                      'After abs. Pitch Summation FCFO',
                      'After abs. Roll Summation FCFO']
            xlim = [
                [0.0, 0.12],
                [0.0, 0.12],
                [0.0, 0.7],
                [0.0, 0.6],
            ]

        performance = ['RMSE', 'Time']

        df = []
        for i in range(len(self.cfo)):
            df.append(pd.DataFrame({
                performance[0]: error_period[i],
                performance[1]: spend_period[i],
                xlabel[0]: ppcfo_summation_3sec[i],
                xlabel[1]: rpcfo_summation_3sec[i],
                xlabel[2]: pfcfo_summation_3sec[i],
                xlabel[3]: rfcfo_summation_3sec[i],
            }))

            df[i]['Group'] = 'Group' + str(i + 1)

        df_all = pd.concat([i for i in df], axis=0)
        # print(df_all)

        marker_size = 40

        cmap = ['Blues_r', 'Blues']

        fig = plt.figure(figsize=(20, 10), dpi=100)
        for i in range(2):
            for j in range(4):
                ax = fig.add_subplot(2, 4, 4 * i + j + 1)
                ax.set_xlim(xlim[j][0], xlim[j][1])
                # ax.set_ylim(ylim[i][k][0], ylim[i][k][1])
                g = sns.scatterplot(data=df_all, x=xlabel[j], y=performance[i], hue='Group', s=marker_size)
                # g.set(xscale="log")
                for lh in g.legend_.legendHandles:
                    lh.set_alpha(1)
                    lh._sizes = [10]

                ax.set_xlabel(xlabel[j])
                ax.set_ylabel(performance[i])

        plt.tight_layout()
        if mode == 'no_abs':
            os.makedirs('fig/CFO-Performance/Summation/', exist_ok=True)
            plt.savefig('fig/CFO-Performance/Summation/SummationCFO-Performance_NoABS_' + str(self.group_type) + '.png')
        elif mode == 'b_abs':
            os.makedirs('fig/CFO-Performance/Summation/', exist_ok=True)
            plt.savefig('fig/CFO-Performance/Summation/SummationCFO-Performance_BeforeABS_' + str(self.group_type) + '.png')
        elif mode == 'a_abs':
            os.makedirs('fig/CFO-Performance/Summation/', exist_ok=True)
            plt.savefig('fig/CFO-Performance/Summation/SummationCFO-Performance_AfterABS_' + str(self.group_type) + '.png')
        plt.show()

    def summation_performance_each_axis(self, mode='no_abs'):
        errorx_period, errory_period, spendx_period, spendy_period = CFO.period_performance_cooperation_each_axis(self)
        ppcfo_summation_3sec, rpcfo_summation_3sec, pfcfo_summation_3sec, rfcfo_summation_3sec = CFO.summation_cfo_3sec(
            self, mode)
        if mode == 'no_abs':
            xlabel = [
                'Pitch Summation PCFO',
                'Roll Summation PCFO',
                'Pitch Summation FCFO',
                'Roll Summation FCFO']
            xlim = [
                [-0.04, 0.04],
                [-0.04, 0.04],
                [-0.2, 0.2],
                [-0.2, 0.2],
            ]
        elif mode == 'b_abs':
            xlabel = [
                'Before abs. Pitch Summation PCFO',
                'Before abs. Roll Summation PCFO',
                'Before abs. Pitch Summation FCFO',
                'Before abs. Roll Summation FCFO']
            xlim = [
                [0.0, 0.15],
                [0.0, 0.15],
                [0.0, 2.0],
                [0.0, 1.25],
            ]
        elif mode == 'a_abs':
            xlabel = [
                'After abs. Pitch Summation PCFO',
                'After abs. Roll Summation PCFO',
                'After abs. Pitch Summation FCFO',
                'After abs. Roll Summation FCFO']
            xlim = [
                [0.0, 0.12],
                [0.0, 0.12],
                [0.0, 0.7],
                [0.0, 0.6],
            ]

        performance = [['Time at x-axis', 'RMSE at x-axis'], ['Time at y-axis', 'RMESE at y-axis']]

        df = []
        for i in range(len(self.cfo)):
            df.append(pd.DataFrame({
                performance[0][1]: errorx_period[i],
                performance[1][1]: errory_period[i],
                performance[0][0]: spendx_period[i],
                performance[1][0]: spendy_period[i],
                xlabel[0]: ppcfo_summation_3sec[i],
                xlabel[1]: rpcfo_summation_3sec[i],
                xlabel[2]: pfcfo_summation_3sec[i],
                xlabel[3]: rfcfo_summation_3sec[i],
            }))

            df[i]['Group'] = 'Group' + str(i + 1)

        df_all = pd.concat([i for i in df], axis=0)
        # print(df_all)

        marker_size = 40

        cmap = ['Blues_r', 'Blues']

        fig = plt.figure(figsize=(20, 10), dpi=100)
        for i in range(2):
            for j in range(4):
                ax = fig.add_subplot(2, 4, 4 * i + j + 1)
                ax.set_xlim(xlim[j][0], xlim[j][1])
                # ax.set_ylim(ylim[i][k][0], ylim[i][k][1])
                g = sns.scatterplot(data=df_all, x=xlabel[j], y=performance[i][j % 2], hue='Group', s=marker_size)
                # g.set(yscale="log")
                for lh in g.legend_.legendHandles:
                    lh.set_alpha(1)
                    lh._sizes = [10]

                ax.set_xlabel(xlabel[j])
                ax.set_ylabel(performance[i][j % 2])

        plt.tight_layout()
        if mode == 'no_abs':
            os.makedirs('fig/CFO-Performance/Summation/EachAxis/', exist_ok=True)
            plt.savefig('fig/CFO-Performance/Summation/EachAxis/SummationCFO-Performance_NoABS_EachAxis_' + str(self.group_type) + '.png')
        elif mode == 'b_abs':
            os.makedirs('fig/CFO-Performance/Summation/EachAxis/', exist_ok=True)
            plt.savefig('fig/CFO-Performance/Summation/EachAxis/SummationCFO-Performance_BeforeABS_EachAxis_' + str(self.group_type) + '.png')
        elif mode == 'a_abs':
            os.makedirs('fig/CFO-Performance/Summation/EachAxis/', exist_ok=True)
            plt.savefig('fig/CFO-Performance/Summation/EachAxis/SummationCFO-Performance_AfterABS_EachAxis_' + str(self.group_type) + '.png')
        plt.show()

    def summation_performance_combine(self, mode='no_abs'):
        error_period, spend_period = CFO.period_performance_cooperation(self)
        pcfo_summation_3sec, fcfo_summation_3sec = CFO.summation_cfo_3sec_combine(self, mode)
        if mode == 'no_abs':
            xlabel = ['Summation PCFO', 'Summation FCFO']
            xlim = [
                [-0.08, 0.08],
                [-0.4, 0.4],
            ]
        elif mode == 'b_abs':
            xlabel = ['Before abs. Summation PCFO', 'Before abs. Summation FCFO']
            xlim = [
                [0.0, 0.3],
                [0.0, 2.5],
            ]
        elif mode == 'a_abs':
            xlabel = ['After abs. Summation PCFO', 'After abs. Summation FCFO']
            xlim = [
                [0.0, 0.4],
                [0.0, 1.0],
            ]

        performance = ['RMSE', 'Time']

        df = []
        for i in range(len(self.cfo)):
            df.append(pd.DataFrame({
                performance[0]: error_period[i],
                performance[1]: spend_period[i],
                xlabel[0]: pcfo_summation_3sec[i],
                xlabel[1]: fcfo_summation_3sec[i],
            }))

            df[i]['Group'] = 'Group' + str(i + 1)

        df_all = pd.concat([i for i in df], axis=0)
        # print(df_all)

        marker_size = 40

        cmap = ['Blues_r', 'Blues']

        fig = plt.figure(figsize=(12, 10), dpi=100)
        for i in range(2):
            for j in range(2):
                ax = fig.add_subplot(2, 2, 2 * i + j + 1)
                ax.set_xlim(xlim[j][0], xlim[j][1])
                # ax.set_ylim(ylim[i][k][0], ylim[i][k][1])
                g = sns.scatterplot(data=df_all, x=xlabel[j], y=performance[i], hue='Group', s=marker_size)
                # g.set(xscale="log")
                r2 = np.corrcoef(df_all[xlabel[j]].values, df_all[performance[i]].values)
                ax.text(0.99, 0.02, '$r = {:.2f}$'.format(r2[0][1]), horizontalalignment='right', transform=ax.transAxes, fontsize="large")

                for lh in g.legend_.legendHandles:
                    lh.set_alpha(1)
                    lh._sizes = [10]

                ax.set_xlabel(xlabel[j])
                ax.set_ylabel(performance[i])

        plt.tight_layout()
        if mode == 'no_abs':
            os.makedirs('fig/CFO-Performance/Summation/Combine/', exist_ok=True)
            plt.savefig('fig/CFO-Performance/Summation/Combine/SummationCFO-Performance_NoABS_' + str(self.group_type) + '.png')
        elif mode == 'b_abs':
            os.makedirs('fig/CFO-Performance/Summation/Combine/', exist_ok=True)
            plt.savefig('fig/CFO-Performance/Summation/Combine/SummationCFO-Performance_BeforeABS_' + str(self.group_type) + '.png')
        elif mode == 'a_abs':
            os.makedirs('fig/CFO-Performance/Summation/Combine/', exist_ok=True)
            plt.savefig('fig/CFO-Performance/Summation/Combine/SummationCFO-Performance_AfterABS_' + str(self.group_type) + '.png')
        plt.show()

    def subtraction_performance(self):
        error_period, spend_period = CFO.period_performance_cooperation(self)
        ppcfo_subtraction_3sec, rpcfo_subtraction_3sec, pfcfo_subtraction_3sec, rfcfo_subtraction_3sec = CFO.subtraction_cfo_3sec(
            self)

        performance = ['RMSE', 'Time']

        df = []
        for i in range(len(self.cfo)):
            df.append(pd.DataFrame({
                performance[0]: error_period[i],
                performance[1]: spend_period[i],
                'Pitch Subtraction PCFO': ppcfo_subtraction_3sec[i],
                'Roll Subtraction PCFO': rpcfo_subtraction_3sec[i],
                'Pitch Subtraction FCFO': pfcfo_subtraction_3sec[i],
                'Roll Subtraction FCFO': rfcfo_subtraction_3sec[i],
            }))

            df[i]['Group'] = 'Group' + str(i + 1)

        df_all = pd.concat([i for i in df], axis=0)
        # print(df_all)

        marker_size = 40

        cmap = ['Blues_r', 'Blues']
        xlabel = ['Pitch Subtraction PCFO', 'Roll Subtraction PCFO', 'Pitch Subtraction FCFO', 'Roll Subtraction FCFO']

        if self.group_type == 'dyad':
            xlim = [0.2, 0.2, 3.0, 3.0]
        elif self.group_type == 'triad':
            xlim = [0.3, 0.3, 4.0, 5.0]
        else:
            xlim = [0.4, 0.4, 5.0, 5.0]

        fig = plt.figure(figsize=(20, 10), dpi=100)
        for i in range(2):
            for j in range(4):
                ax = fig.add_subplot(2, 4, 4 * i + j + 1)
                ax.set_xlim(0, xlim[j])
                # ax.set_ylim(ylim[i][k][0], ylim[i][k][1])
                g = sns.scatterplot(data=df_all, x=xlabel[j], y=performance[i], hue='Group', s=marker_size)
                for lh in g.legend_.legendHandles:
                    lh.set_alpha(1)
                    lh._sizes = [10]

                ax.set_xlabel(xlabel[j])
                ax.set_ylabel(performance[i])

        plt.tight_layout()
        os.makedirs('fig/CFO-Performance/Subtraction/', exist_ok=True)
        plt.savefig('fig/CFO-Performance/Subtraction/SubtractionCFO_Performance_' + str(self.group_type) + '.png')
        plt.show()

    def subtraction_performance_each_axis(self):
        errorx_period, errory_period, spendx_period, spendy_period = CFO.period_performance_cooperation_each_axis(self)
        ppcfo_subtraction_3sec, rpcfo_subtraction_3sec, pfcfo_subtraction_3sec, rfcfo_subtraction_3sec = CFO.subtraction_cfo_3sec(
            self)

        df = []
        for i in range(len(self.cfo)):
            df.append(pd.DataFrame({
                'errorx': errorx_period[i],
                'errory': errory_period[i],
                'spendx': spendx_period[i],
                'spendy': spendy_period[i],
                'Pitch Subtraction PCFO': ppcfo_subtraction_3sec[i],
                'Roll Subtraction PCFO': rpcfo_subtraction_3sec[i],
                'Pitch Subtraction FCFO': pfcfo_subtraction_3sec[i],
                'Roll Subtraction FCFO': rfcfo_subtraction_3sec[i],
            }))

            df[i]['Group'] = 'Group' + str(i + 1)

        df_all = pd.concat([i for i in df], axis=0)
        # print(df_all)

        marker_size = 5

        performance = [['errorx', 'errory'], ['spendx', 'spendy']]
        cmap = ['Blues_r', 'Blues']
        xlabel = ['Pitch Subtraction PCFO', 'Roll Subtraction PCFO', 'Pitch Subtraction FCFO', 'Roll Subtraction FCFO']

        if self.group_type == 'dyad':
            xlim = [0.2, 0.2, 3.0, 3.0]
        elif self.group_type == 'triad':
            xlim = [0.3, 0.3, 4.0, 5.0]
        else:
            xlim = [0.4, 0.4, 5.0, 5.0]

        fig = plt.figure(figsize=(8, 4), dpi=300)
        for i in range(2):
            for j in range(4):
                ax = fig.add_subplot(2, 4, 4 * i + j + 1)
                ax.set_xlim(0, xlim[j])
                # ax.set_ylim(ylim[i][k][0], ylim[i][k][1])
                g = sns.scatterplot(data=df_all, x=xlabel[j], y=performance[i][j % 2], hue='Group', s=marker_size)
                for lh in g.legend_.legendHandles:
                    lh.set_alpha(1)
                    lh._sizes = [10]

                ax.set_xlabel(xlabel[j])
                ax.set_ylabel(performance[i][j % 2])

        plt.tight_layout()
        plt.savefig('fig/subtraction_performance_each_axis_' + str(self.group_type) + '.png')
        plt.show()

    def subtraction_performance_combine(self):
        error_period, spend_period = CFO.period_performance_cooperation(self)
        pcfo_subtraction_3sec, fcfo_subtraction_3sec = CFO.subtraction_cfo_3sec_combine(self)

        performance = ['RMSE', 'Time']

        df = []
        for i in range(len(self.cfo)):
            df.append(pd.DataFrame({
                performance[0]: error_period[i],
                performance[1]: spend_period[i],
                'Subtraction PCFO': pcfo_subtraction_3sec[i],
                'Subtraction FCFO': fcfo_subtraction_3sec[i],
            }))

            df[i]['Group'] = 'Group' + str(i + 1)

        df_all = pd.concat([i for i in df], axis=0)
        # print(df_all)

        marker_size = 40

        cmap = ['Blues_r', 'Blues']
        xlabel = ['Subtraction PCFO', 'Subtraction FCFO']

        if self.group_type == 'dyad':
            xlim = [0.4, 8.0]
        elif self.group_type == 'triad':
            xlim = [0.6, 10.0]
        else:
            xlim = [0.8, 10.0]

        fig = plt.figure(figsize=(12, 10), dpi=100)
        for i in range(2):
            for j in range(2):
                ax = fig.add_subplot(2, 2, 2 * i + j + 1)
                ax.set_xlim(0, xlim[j])
                # ax.set_ylim(ylim[i][k][0], ylim[i][k][1])
                g = sns.scatterplot(data=df_all, x=xlabel[j], y=performance[i], hue='Group', s=marker_size)
                r2 = np.corrcoef(df_all[xlabel[j]].values, df_all[performance[i]].values)
                ax.text(0.99, 0.02, '$r = {:.2f}$'.format(r2[0][1]), horizontalalignment='right', transform=ax.transAxes, fontsize="large")
                for lh in g.legend_.legendHandles:
                    lh.set_alpha(1)
                    lh._sizes = [10]

                ax.set_xlabel(xlabel[j])
                ax.set_ylabel(performance[i])

        plt.tight_layout()
        os.makedirs('fig/CFO-Performance/Subtraction/Combine/', exist_ok=True)
        plt.savefig('fig/CFO-Performance/Subtraction/Combine/SubtractionCFO_Performance_' + str(self.group_type) + '.png')
        plt.show()

    def CFO_relation_axis(self):
        df_list = []
        for i in range(len(self.cfo)):
            data = self.cfo[i]
            for j in range(self.join):
                interface = 'i' + str(j + 1)
                df_each = pd.DataFrame({
                    'Pitch PCFO': data[interface + '_p_pcfo'][self.start_num:self.end_num:1000],
                    'Roll PCFO': data[interface + '_r_pcfo'][self.start_num:self.end_num:1000],
                    'Pitch FCFO': data[interface + '_p_fcfo'][self.start_num:self.end_num:1000],
                    'Roll FCFO': data[interface + '_r_fcfo'][self.start_num:self.end_num:1000],
                })
                df_list.append(df_each)
        df = pd.concat([i for i in df_list], axis=0)
        df = df.reset_index()

        # plt.figure(figsize=(8, 4), dpi=300)

        kwargs = dict(
            ratio=10,
            height=10,
            kind='hist',
            # kind="kde",
            # kind="scatter",
        )

        kwargs_PCFO = dict(
            xlim=(-0.25, 0.25),
            ylim=(-0.25, 0.25),
            # gridsize=500,
        )

        sns.set(font_scale=1)
        sns.jointplot(data=df, x='Pitch PCFO', y='Roll PCFO', **kwargs, **kwargs_PCFO)

        plt.tight_layout()
        plt.savefig('fig/CFO_relation_axis_P-P_R-P_' + str(self.group_type) + '.png')

        kwargs_FCFO = dict(
            xlim=(-2.5, 2.5),
            ylim=(-2.5, 2.5),
            # gridsize=300,
        )

        sns.jointplot(data=df, x='Pitch FCFO', y='Roll FCFO', **kwargs, **kwargs_FCFO)

        plt.tight_layout()
        plt.savefig('fig/CFO_relation_axis_P-F_R-F_' + str(self.group_type) + '.png')

        kwargs_PCFO_FCFO = dict(
            xlim=(-0.25, 0.25),
            ylim=(-2.5, 2.5),
            # gridsize=500,
        )

        sns.set(font_scale=1)
        sns.jointplot(data=df, x='Pitch PCFO', y='Pitch FCFO', **kwargs, **kwargs_PCFO_FCFO)

        plt.tight_layout()
        plt.savefig('fig/CFO_relation_axis_P-P_P-F_' + str(self.group_type) + '.png')

        sns.jointplot(data=df, x='Roll PCFO', y='Roll FCFO', **kwargs, **kwargs_PCFO_FCFO)

        plt.tight_layout()
        plt.savefig('fig/CFO_relation_axis_R-P_R-F_' + str(self.group_type) + '.png')

        plt.tight_layout()
        plt.show()

    def CFO_relation_axis_3sec(self):
        df_list = []
        for i in range(len(self.cfo)):
            data = self.cfo[i]
            for j in range(self.join):
                interface = 'i' + str(j + 1)
                period_ppcfo = CFO.calc_period_calculator(self,
                                                          data[interface + '_p_pcfo'][self.start_num:self.end_num])
                period_rpcfo = CFO.calc_period_calculator(self,
                                                          data[interface + '_r_pcfo'][self.start_num:self.end_num])
                period_pfcfo = CFO.calc_period_calculator(self,
                                                          data[interface + '_p_fcfo'][self.start_num:self.end_num])
                period_rfcfo = CFO.calc_period_calculator(self,
                                                          data[interface + '_r_fcfo'][self.start_num:self.end_num])

                df_each = pd.DataFrame({
                    'Pitch PCFO': period_ppcfo,
                    'Roll PCFO': period_rpcfo,
                    'Pitch FCFO': period_pfcfo,
                    'Roll FCFO': period_rfcfo,
                })
                df_list.append(df_each)
        df = pd.concat([i for i in df_list], axis=0)
        df = df.reset_index()

        # plt.figure(figsize=(8, 4), dpi=300)

        kwargs = dict(
            ratio=10,
            height=10,
            kind='hist',
            # kind="kde",
            # kind="scatter",
            # kind='hex'
        )

        kwargs_PCFO = dict(
            xlim=(-0.25, 0.25),
            ylim=(-0.25, 0.25),
            # gridsize=20,
        )

        sns.set(font_scale=1)
        sns.jointplot(data=df, x='Pitch PCFO', y='Roll PCFO', **kwargs, **kwargs_PCFO)

        plt.tight_layout()
        plt.savefig('fig/CFO_relation_axis_P-P_P-F_3sec_' + str(self.group_type) + '.png')

        kwargs_FCFO = dict(
            xlim=(-2.5, 2.5),
            ylim=(-2.5, 2.5),
            # gridsize=15,
        )

        sns.jointplot(data=df, x='Pitch FCFO', y='Roll FCFO', **kwargs, **kwargs_FCFO)

        plt.tight_layout()
        plt.savefig('fig/CFO_relation_axis_P-F_R-F_3sec_' + str(self.group_type) + '.png')

        kwargs_PCFO_FCFO = dict(
            xlim=(-0.25, 0.25),
            ylim=(-2.5, 2.5),
            # gridsize=20,
        )

        sns.set(font_scale=1)
        sns.jointplot(data=df, x='Pitch PCFO', y='Pitch FCFO', **kwargs, **kwargs_PCFO_FCFO)

        plt.tight_layout()
        plt.savefig('fig/CFO_relation_axis_P-P_P-F_3sec_' + str(self.group_type) + '.png')

        sns.jointplot(data=df, x='Roll PCFO', y='Roll FCFO', **kwargs, **kwargs_PCFO_FCFO)

        plt.tight_layout()
        plt.savefig('fig/CFO_relation_axis_R-P_R-F_3sec_' + str(self.group_type) + '.png')

        plt.tight_layout()
        plt.show()

    def summation_ave_performance(self, mode='no_abs'):
        error_period, spend_period = CFO.period_performance_cooperation(self)
        pcfo_summation_3sec, fcfo_summation_3sec, pcfo_abs_summation_3sec, fcfo_abs_summation_3sec = CFO.summation_cfo_3sec(
            self, mode)
        if mode == 'no_abs':
            xlabel = ['Summation PCFO',
                      'Summation FCFO',
                      'Summation abs. PCFO',
                      'Summation abs. FCFO',
                      ]
            xlim = [
                [-0.04, 0.04],
                [-0.04, 0.04],
                [-0.2, 0.2],
                [-0.2, 0.2],
            ]
        elif mode == 'b_abs':
            xlabel = ['Before abs. Summation PCFO',
                      'Before abs. Summation FCFO',
                      'Before abs. Summation abs. PCFO',
                      'Before abs. Summation abs. FCFO',
                      ]
            xlim = [
                [0.0, 0.2],
                [0.0, 0.2],
                [0.0, 2.0],
                [0.0, 2.0],
            ]
        elif mode == 'a_abs':
            xlabel = ['After abs. Summation PCFO',
                      'After abs. Summation FCFO',
                      'After abs. Summation abs. PCFO',
                      'After abs. Summation abs. FCFO',
                      ]
            xlim = [
                [0.0, 0.15],
                [0.0, 0.15],
                [0.0, 0.6],
                [0.0, 0.6],
            ]

        performance = ['error', 'spend']

        df = []
        for i in range(len(self.cfo)):
            df.append(pd.DataFrame({
                performance[0]: error_period[i],
                performance[1]: spend_period[i],
                xlabel[0]: pcfo_summation_3sec[i],
                xlabel[1]: fcfo_summation_3sec[i],
                xlabel[2]: pcfo_abs_summation_3sec[i],
                xlabel[3]: fcfo_abs_summation_3sec[i],
            }))

            df[i]['Group'] = 'Group' + str(i + 1)

        df_all = pd.concat([i for i in df], axis=0)
        # print(df_all)

        marker_size = 5

        cmap = ['Blues_r', 'Blues']

        fig = plt.figure(figsize=(8, 4), dpi=300)
        for i in range(2):
            for j in range(4):
                ax = fig.add_subplot(2, 4, 4 * i + j + 1)
                ax.set_xlim(xlim[j][0], xlim[j][1])
                # ax.set_ylim(ylim[i][k][0], ylim[i][k][1])
                g = sns.scatterplot(data=df_all, x=xlabel[j], y=performance[i], hue='Group', s=marker_size)
                for lh in g.legend_.legendHandles:
                    lh.set_alpha(1)
                    lh._sizes = [10]

                ax.set_xlabel(xlabel[j])
                ax.set_ylabel(performance[i])

        plt.tight_layout()
        plt.savefig('fig/summation_ave_performance_' + str(mode) + '_' + str(self.group_type) + '.png')
        plt.show()

    def subtraction_ave_performance(self):
        error_period, spend_period = CFO.period_performance_cooperation(self)
        pcfo_subtraction_3sec, fcfo_subtraction_3sec, pcfo_abs_subtraction_3sec, fcfo_abs_subtraction_3sec = CFO.subtraction_ave_cfo_3sec(
            self)

        xlabel = ['Subtraction PCFO', 'Subtraction PCFO', 'Subtraction abs. FCFO', 'Subtraction abs. FCFO']

        df = []
        for i in range(len(self.cfo)):
            df.append(pd.DataFrame({
                'error': error_period[i],
                'spend': spend_period[i],
                xlabel[0]: pcfo_subtraction_3sec[i],
                xlabel[1]: fcfo_subtraction_3sec[i],
                xlabel[2]: pcfo_abs_subtraction_3sec[i],
                xlabel[3]: fcfo_abs_subtraction_3sec[i],
            }))

            df[i]['Group'] = 'Group' + str(i + 1)

        df_all = pd.concat([i for i in df], axis=0)
        # print(df_all)

        marker_size = 5

        performance = ['error', 'spend']
        cmap = ['Blues_r', 'Blues']

        if self.group_type == 'dyad':
            xlim = [4.0, 4.0, 2.0, 2.0]
        elif self.group_type == 'triad':
            xlim = [7.0, 7.0, 4.0, 4.0]
        elif self.group_type == 'tetrad':
            xlim = [10.0, 10.0, 6.0, 6.0]

        fig = plt.figure(figsize=(8, 4), dpi=300)
        for i in range(2):
            for j in range(4):
                ax = fig.add_subplot(2, 4, 4 * i + j + 1)
                ax.set_xlim(0, xlim[j])
                # ax.set_ylim(ylim[i][k][0], ylim[i][k][1])
                g = sns.scatterplot(data=df_all, x=xlabel[j], y=performance[i], hue='Group', s=marker_size)
                for lh in g.legend_.legendHandles:
                    lh.set_alpha(1)
                    lh._sizes = [10]

                ax.set_xlabel(xlabel[j])
                ax.set_ylabel(performance[i])

        plt.tight_layout()
        plt.savefig('fig/subtraction_ave_performance_' + str(self.group_type) + '.png')
        plt.show()

    def summation_ave_cfo(self, graph=False, mode='no_abs'):
        summation = self.cfo[0]['i1_pcfo_sum'][self.start_num:self.end_num]
        types = ['_pcfo_sum', '_fcfo_sum', '_pcfo_sum_abs', '_fcfo_sum_abs']
        for type in types:
            for j in range(len(self.cfo)):
                data = self.cfo[j]
                summation_ = data['i1_pcfo_sum'][self.start_num:self.end_num]
                for i in range(self.join):
                    interfacenum = 'i' + str(i + 1)
                    pcfoname = interfacenum + type

                    summation_ = np.vstack((summation_, data[pcfoname][self.start_num:self.end_num]))
                summation_ = np.delete(summation_, 0, 0)
                if mode == 'b_abs':
                    summation_ = np.abs(summation_)
                summation = np.vstack((summation, np.sum(summation_, axis=0)))

            # print(summation_)
        summation = np.delete(summation, 0, 0)
        # print(summation.shape)
        summation = summation.reshape([4, len(self.cfo), -1])
        summation = summation / self.cfo[0]['join'][0]
        if mode == 'a_abs':
            summation = np.abs(summation)

        if graph == True:
            fig, (pcfo, fcfo, pcfoave, fcfoave) = plt.subplots(4, 1, figsize=(5, 7), dpi=150, sharex=True)

            # plt.xlim([10, 60])  # x軸の範囲
            # plt.xlim([0.28, 0.89])  # x軸の範囲
            plt.xlabel("Time (sec)")

            for i in range(len(self.cfo)):
                data = self.cfo[i]

                pcfo.plot(data['time'][self.start_num:self.end_num:10]-20.0, summation[0][i][::10],
                          label='Group' + str(i + 1))
                fcfo.plot(data['time'][self.start_num:self.end_num:10]-20.0, summation[1][i][::10],
                          label='Group' + str(i + 1))
                pcfoave.plot(data['time'][self.start_num:self.end_num:10]-20.0, summation[2][i][::10],
                             label='Group' + str(i + 1))
                fcfoave.plot(data['time'][self.start_num:self.end_num:10]-20.0, summation[3][i][::10],
                             label='Group' + str(i + 1))

            pcfo.set_ylabel('Summation\nPCFO (rad)')
            pcfo.set_ylabel('Summation\nFCFO (Nm)')
            fcfoave.set_ylabel('Summation\nabs. PCFO (rad)')
            fcfoave.set_ylabel('Summation\nabs. FCFO (Nm)')

            pcfo.legend(ncol=10, columnspacing=1, loc='upper left')
            fcfo.legend(ncol=10, columnspacing=1, loc='upper left')
            pcfoave.legend(ncol=10, columnspacing=1, loc='upper left')
            fcfoave.legend(ncol=10, columnspacing=1, loc='upper left')

            pcfo.set_yticks(np.arange(-10, 10, 0.5))
            fcfo.set_yticks(np.arange(-10, 10, 0.5))
            pcfoave.set_yticks(np.arange(-8.0, 8.0, 1.0))
            fcfoave.set_yticks(np.arange(-8.0, 8.0, 1.0))

            pcfo.set_ylim([0, 1.0])  # y軸の範囲
            fcfo.set_ylim([0, 1.0])  # y軸の範囲
            pcfoave.set_ylim([0, 4.0])  # y軸の範囲
            fcfoave.set_ylim([0, 4.0])  # y軸の範囲

            plt.tight_layout()
            # plt.savefig(savename)
            plt.show()

        return summation[0], summation[1], summation[2], summation[3]

    def subtraction_ave_cfo(self, graph=False):
        subtraction = self.cfo[0]['i1_pcfo_sum'][self.start_num:self.end_num]
        types = ['pcfo_sum', 'fcfo_sum', 'pcfo_sum_abs', 'fcfo_sum_abs']
        for type in types:
            subtraction_ = self.cfo[0]['i1_pcfo_sum'][self.start_num:self.end_num]
            for i in range(len(self.cfo)):
                data = self.cfo[i]
                if self.group_type == 'dyad':
                    sub_cfo1 = np.abs(np.subtract(data['i1_' + type][self.start_num:self.end_num],
                                                  data['i2_' + type][self.start_num:self.end_num]))
                    sub_cfo2 = np.abs(np.subtract(data['i2_' + type][self.start_num:self.end_num],
                                                  data['i1_' + type][self.start_num:self.end_num]))
                    sub_cfo_ave = np.add(np.abs(sub_cfo1), np.abs(sub_cfo2)) / 2
                    subtraction_ = np.vstack((subtraction_, sub_cfo_ave))

                elif self.group_type == 'triad':
                    sub_cfo1 = np.subtract(np.subtract(2 * data['i1_' + type][self.start_num:self.end_num],
                                                       data['i2_' + type][self.start_num:self.end_num]),
                                           data['i3_' + type][self.start_num:self.end_num])
                    sub_cfo2 = np.subtract(np.subtract(2 * data['i2_' + type][self.start_num:self.end_num],
                                                       data['i1_' + type][self.start_num:self.end_num]),
                                           data['i3_' + type][self.start_num:self.end_num])
                    sub_cfo3 = np.subtract(np.subtract(2 * data['i3_' + type][self.start_num:self.end_num],
                                                       data['i1_' + type][self.start_num:self.end_num]),
                                           data['i2_' + type][self.start_num:self.end_num])
                    sub_cfo_ave = np.add(np.add(np.abs(sub_cfo1), np.abs(sub_cfo2)), np.abs(sub_cfo3)) / 3
                    subtraction_ = np.vstack((subtraction_, sub_cfo_ave))

                elif self.group_type == 'tetrad':
                    sub_cfo1 = np.subtract(np.subtract(np.subtract(3 * data['i1_' + type][self.start_num:self.end_num],
                                                                   data['i2_' + type][self.start_num:self.end_num]),
                                                       data['i3_' + type][self.start_num:self.end_num]),
                                           data['i4_' + type][self.start_num:self.end_num])
                    sub_cfo2 = np.subtract(np.subtract(np.subtract(3 * data['i2_' + type][self.start_num:self.end_num],
                                                                   data['i1_' + type][self.start_num:self.end_num]),
                                                       data['i3_' + type][self.start_num:self.end_num]),
                                           data['i4_' + type][self.start_num:self.end_num])
                    sub_cfo3 = np.subtract(np.subtract(np.subtract(3 * data['i3_' + type][self.start_num:self.end_num],
                                                                   data['i1_' + type][self.start_num:self.end_num]),
                                                       data['i2_' + type][self.start_num:self.end_num]),
                                           data['i4_' + type][self.start_num:self.end_num])
                    sub_cfo4 = np.subtract(np.subtract(np.subtract(3 * data['i4_' + type][self.start_num:self.end_num],
                                                                   data['i1_' + type][self.start_num:self.end_num]),
                                                       data['i2_' + type][self.start_num:self.end_num]),
                                           data['i3_' + type][self.start_num:self.end_num])
                    sub_cfo_ave = np.add(np.add(np.add(np.abs(sub_cfo1), np.abs(sub_cfo2)), np.abs(sub_cfo3)),
                                         np.abs(sub_cfo4)) / 4
                    subtraction_ = np.vstack((subtraction_, sub_cfo_ave))

            subtraction_ = np.delete(subtraction_, 0, 0)
            subtraction = np.vstack((subtraction, subtraction_))
        subtraction = np.delete(subtraction, 0, 0)
        subtraction = subtraction.reshape([4, len(self.cfo), -1])

        if graph == True:
            fig, (pcfo, fcfo, pcfoave, fcfoave) = plt.subplots(4, 1, figsize=(5, 7), dpi=150, sharex=True)

            # plt.xlim([10, 60])  # x軸の範囲
            # plt.xlim([0.28, 0.89])  # x軸の範囲
            plt.xlabel("Time (sec)")

            for i in range(len(self.cfo)):
                data = self.cfo[i]

                pcfo.plot(data['time'][self.start_num:self.end_num:10]-20.0, subtraction[0][i][::10],
                          label='Group' + str(i + 1))
                fcfo.plot(data['time'][self.start_num:self.end_num:10]-20.0, subtraction[1][i][::10],
                          label='Group' + str(i + 1))
                pcfoave.plot(data['time'][self.start_num:self.end_num:10]-20.0, subtraction[2][i][::10],
                             label='Group' + str(i + 1))
                fcfoave.plot(data['time'][self.start_num:self.end_num:10]-20.0, subtraction[3][i][::10],
                             label='Group' + str(i + 1))

            pcfo.set_ylabel('Subtraction\nPCFO (rad)')
            fcfo.set_ylabel('Subtraction\nFCFO (Nm)')
            pcfoave.set_ylabel('Subtraction\nabs. PCFO (rad)')
            fcfoave.set_ylabel('Subtraction\nabs. FCFO (Nm)')

            pcfo.legend(ncol=10, columnspacing=1, loc='upper left')
            fcfo.legend(ncol=10, columnspacing=1, loc='upper left')
            pcfoave.legend(ncol=10, columnspacing=1, loc='upper left')
            fcfoave.legend(ncol=10, columnspacing=1, loc='upper left')

            pcfo.set_yticks(np.arange(-10, 10, 0.5))
            fcfo.set_yticks(np.arange(-10, 10, 0.5))
            pcfoave.set_yticks(np.arange(-8.0, 8.0, 1.0))
            fcfoave.set_yticks(np.arange(-8.0, 8.0, 1.0))

            pcfo.set_ylim([0, 1.0])  # y軸の範囲
            fcfo.set_ylim([0, 1.0])  # y軸の範囲
            pcfoave.set_ylim([0, 10.0])  # y軸の範囲
            fcfoave.set_ylim([0, 10.0])  # y軸の範囲

            plt.tight_layout()
            # plt.savefig(savename)
            plt.show()

        return subtraction[0], subtraction[1], subtraction[2], subtraction[3]

    def summation_ave_cfo_3sec(self, mode='no_abs'):
        pcfo_summation, fcfo_summation, pcfo_abs_summation, fcfo_abs_summation = CFO.summation_ave_cfo(self,
                                                                                                       graph=False,
                                                                                                       mode=mode)

        pcfo_summation_3sec = pcfo_summation.reshape([len(self.cfo), -1, self.num])
        pcfo_summation_3sec = np.average(pcfo_summation_3sec, axis=2)

        fcfo_summation_3sec = fcfo_summation.reshape([len(self.cfo), -1, self.num])
        fcfo_summation_3sec = np.average(fcfo_summation_3sec, axis=2)

        pcfo_abs_summation_3sec = pcfo_abs_summation.reshape([len(self.cfo), -1, self.num])
        pcfo_abs_summation_3sec = np.average(pcfo_abs_summation_3sec, axis=2)

        fcfo_abs_summation_3sec = fcfo_abs_summation.reshape([len(self.cfo), -1, self.num])
        fcfo_abs_summation_3sec = np.average(fcfo_abs_summation_3sec, axis=2)

        return pcfo_summation_3sec, fcfo_summation_3sec, pcfo_abs_summation_3sec, fcfo_abs_summation_3sec

    def subtraction_ave_cfo_3sec(self):
        pcfo_subtraction, fcfo_subtraction, pcfo_abs_subtraction, fcfo_abs_subtraction = CFO.subtraction_ave_cfo(self)

        pcfo_subtraction_3sec = pcfo_subtraction.reshape([len(self.cfo), -1, self.num])
        pcfo_subtraction_3sec = np.average(pcfo_subtraction_3sec, axis=2)

        fcfo_subtraction_3sec = fcfo_subtraction.reshape([len(self.cfo), -1, self.num])
        fcfo_subtraction_3sec = np.average(fcfo_subtraction_3sec, axis=2)

        pcfo_abs_subtraction_3sec = pcfo_abs_subtraction.reshape([len(self.cfo), -1, self.num])
        pcfo_abs_subtraction_3sec = np.average(pcfo_abs_subtraction_3sec, axis=2)

        fcfo_abs_subtraction_3sec = fcfo_abs_subtraction.reshape([len(self.cfo), -1, self.num])
        fcfo_abs_subtraction_3sec = np.average(fcfo_abs_subtraction_3sec, axis=2)

        return pcfo_subtraction_3sec, fcfo_subtraction_3sec, pcfo_abs_subtraction_3sec, fcfo_abs_subtraction_3sec

    def fcfo_valiance(self, graph=False):
        label = ['_p_fcfo', '_r_fcfo']
        ylabel = ['pitch', 'roll']
        valiance = []
        valiance_period = []
        for i in range(len(self.cfo)):
            valiance.append([])
            valiance_period.append([])
            data = self.cfo[i]
            for j in range(len(label)):
                valiance[i].append([])
                valiance_period[i].append([])
                stack = np.stack(
                    [data['i' + str(_ + 1) + label[j]][self.start_num:self.end_num] for _ in range(self.join)], axis=0)
                valiance_ = CFO.variance_calculation(self, stack)
                valiance[i][j] = valiance_

                valiance_period_ = CFO.calc_period_calculator(self, valiance_)
                valiance_period[i][j] = valiance_period_

        if graph == True:
            for i in range(len(self.cfo)):
                fig = plt.figure(figsize=(5, 3), dpi=300)
                for j in range(len(label)):
                    ax = fig.add_subplot(2, 1, j + 1)
                    ax.set_ylim(0, 4)
                    ax.plot(data['time'][self.start_num:self.end_num:10]-20.0, valiance[i][j][::10])
                    ax.set_ylabel('Variance of ' + ylabel[j] + ' FCFO (Nm)')
                    ax.set_xlabel('time (s)')

                plt.tight_layout()
            plt.show()

        return valiance, valiance_period

    def variance_calculation(self, data):
        variance = np.var(data, axis=0)
        return variance

    def tf_graph_sub(self):
        for j in range(len(self.cfo)):
            fig, (rthm, pthm, rtext, ptext) = plt.subplots(4, 1, figsize=(5, 5), dpi=150, sharex=True)

            plt.xticks(np.arange(self.starttime, self.endtime * 2, self.duringtime * 2))
            plt.xlim([self.starttime, self.endtime])  # x軸の範囲
            plt.xlabel("Time (sec)")

            data = self.cfo[j]
            for i in range(self.join):
                interfacenum = 'i' + str(i + 1)

                thmname = interfacenum + '_r_thm_tf'
                thm_prename = interfacenum + '_r_thm_pre_tf'
                thm_prename_solo = interfacenum + '_r_thm_pre_solo_tf'
                rthm.plot(data['time'][self.start_num:self.end_num:10]-20.0, data[thmname][self.start_num:self.end_num:10],
                          label='P' + str(i + 1) + '_act')
                rthm.plot(data['time'][self.start_num:self.end_num:10]-20.0,
                          data[thm_prename][self.start_num:self.end_num:10], label='P' + str(i + 1) + '_pre')
                # rthm.plot(data['time'][self.start_num:self.end_num:10]-20.0, data[thm_prename_solo][self.start_num:self.end_num:10], label='P' + str(i + 1) + '_pre_solo')

                thmname = interfacenum + '_p_thm_tf'
                thm_prename = interfacenum + '_p_thm_pre_tf'
                thm_prename_solo = interfacenum + '_p_thm_pre_solo_tf'
                pthm.plot(data['time'][self.start_num:self.end_num:10]-20.0, data[thmname][self.start_num:self.end_num:10],
                          label='P' + str(i + 1) + '_act')
                pthm.plot(data['time'][self.start_num:self.end_num:10]-20.0,
                          data[thm_prename][self.start_num:self.end_num:10], label='P' + str(i + 1) + '_pre')
                # pthm.plot(data['time'][self.start_num:self.end_num:10]-20.0, data[thm_prename_solo][self.start_num:self.end_num:10], label='P'+str(i+1)+'_pre_solo')

                textname = interfacenum + '_r_text_tf'
                text_prename = interfacenum + '_r_text_pre_tf'
                text_prename_solo = interfacenum + '_r_text_pre_solo_tf'
                rtext.plot(data['time'][self.start_num:self.end_num:10]-20.0, data[textname][self.start_num:self.end_num:10],
                           label='P' + str(i + 1) + '_act')
                rtext.plot(data['time'][self.start_num:self.end_num:10]-20.0,
                           data[text_prename][self.start_num:self.end_num:10], label='P' + str(i + 1) + '_pre')
                # rtext.plot(data['time'][self.start_num:self.end_num:10]-20.0, data[text_prename_solo][self.start_num:self.end_num:10], label='P' + str(i + 1) + '_pre_solo')

                textname = interfacenum + '_p_text_tf'
                text_prename = interfacenum + '_p_text_pre_tf'
                text_prename_solo = interfacenum + '_p_text_pre_solo_tf'
                ptext.plot(data['time'][self.start_num:self.end_num:10]-20.0, data[textname][self.start_num:self.end_num:10],
                           label='P' + str(i + 1) + '_act')
                ptext.plot(data['time'][self.start_num:self.end_num:10]-20.0,
                           data[text_prename][self.start_num:self.end_num:10], label='P' + str(i + 1) + '_pre')
                # ptext.plot(data['time'][self.start_num:self.end_num:10]-20.0, data[text_prename_solo][self.start_num:self.end_num:10], label='P'+str(i+1)+'_pre_solo')

            rthm.set_ylabel('Roll angle (rad)')
            rthm.legend(ncol=2, columnspacing=1, loc='upper left')
            rthm.set_yticks(np.arange(-10, 10, 0.5))
            rthm.set_ylim([-1.5, 1.5])  # y軸の範囲

            pthm.set_ylabel('Pitch angle (rad)')
            pthm.legend(ncol=2, columnspacing=1, loc='upper left')
            pthm.set_yticks(np.arange(-10, 10, 0.5))
            pthm.set_ylim([-1.5, 1.5])  # y軸の範囲

            rtext.set_ylabel('Roll force (Nm)')
            rtext.legend(ncol=2, columnspacing=1, loc='upper left')
            rtext.set_yticks(np.arange(-8.0, 8.0, 2.0))
            rtext.set_ylim([-6.0, 6.0])  # y軸の範囲

            ptext.set_ylabel('Pitch force (Nm)')
            ptext.legend(ncol=2, columnspacing=1, loc='upper left')
            ptext.set_yticks(np.arange(-8.0, 8.0, 2.0))
            ptext.set_ylim([-6.0, 6.0])  # y軸の範囲

            plt.tight_layout()
            # plt.savefig("response.png")
        plt.show()

    def tf_cfo_sub(self):

        for j in range(len(self.cfo)):
            data = self.cfo[j]

            fig, (ppcfo, rpcfo, pfcfo, rfcfo) = plt.subplots(4, 1, figsize=(5, 8), dpi=150, sharex=True)

            # plt.xlim([10, 60])  # x軸の範囲
            # plt.xlim([0.28, 0.89])  # x軸の範囲
            plt.xlabel("Time (sec)")

            for i in range(self.join):
                interfacenum = 'i' + str(i + 1)
                pcfoname = interfacenum + '_p_pcfo_tf'
                fcfoname = interfacenum + '_p_fcfo_tf'

                ppcfo.plot(data['time'][self.start_num:self.end_num:10]-20.0, data[pcfoname][self.start_num:self.end_num:10],
                           label='P' + str(i + 1))
                pfcfo.plot(data['time'][self.start_num:self.end_num:10]-20.0, data[fcfoname][self.start_num:self.end_num:10],
                           label='P' + str(i + 1))

                pcfoname = interfacenum + '_r_pcfo_tf'
                fcfoname = interfacenum + '_r_fcfo_tf'

                rpcfo.plot(data['time'][self.start_num:self.end_num:10]-20.0, data[pcfoname][self.start_num:self.end_num:10],
                           label='P' + str(i + 1))
                rfcfo.plot(data['time'][self.start_num:self.end_num:10]-20.0, data[fcfoname][self.start_num:self.end_num:10],
                           label='P' + str(i + 1))

            ppcfo.set_ylabel('Pitch PCFO (rad)')
            ppcfo.legend(ncol=2, columnspacing=1, loc='upper left')
            ppcfo.set_yticks(np.arange(-10, 10, 0.5))
            ppcfo.set_ylim([-1.5, 1.5])  # y軸の範囲

            rpcfo.set_ylabel('Roll PCFO (rad)')
            rpcfo.legend(ncol=2, columnspacing=1, loc='upper left')
            rpcfo.set_yticks(np.arange(-10, 10, 0.5))
            rpcfo.set_ylim([-1.5, 1.5])  # y軸の範囲

            pfcfo.set_ylabel('Pitch FCFO (Nm)')
            pfcfo.legend(ncol=2, columnspacing=1, loc='upper left')
            pfcfo.set_yticks(np.arange(-8.0, 8.0, 2.0))
            pfcfo.set_ylim([-6.0, 6.0])  # y軸の範囲

            rfcfo.set_ylabel('Roll FCFO (Nm)')
            rfcfo.legend(ncol=2, columnspacing=1, loc='upper left')
            rfcfo.set_yticks(np.arange(-8.0, 8.0, 2.0))
            rfcfo.set_ylim([-6.0, 6.0])  # y軸の範囲r

            plt.tight_layout()
            # plt.savefig("response.png")
        plt.show()

    def tf_summation_cfo_3sec(self, mode='no_abs'):
        ppcfo_summation, rpcfo_summation, pfcfo_summation, rfcfo_summation = CFO.tf_summation_cfo(self, graph=False,
                                                                                                  mode=mode)

        ppcfo_summation_3sec = ppcfo_summation.reshape([len(self.cfo), -1, self.num])
        ppcfo_summation_3sec = np.average(ppcfo_summation_3sec, axis=2)

        rpcfo_summation_3sec = rpcfo_summation.reshape([len(self.cfo), -1, self.num])
        rpcfo_summation_3sec = np.average(rpcfo_summation_3sec, axis=2)

        pfcfo_summation_3sec = pfcfo_summation.reshape([len(self.cfo), -1, self.num])
        pfcfo_summation_3sec = np.average(pfcfo_summation_3sec, axis=2)

        rfcfo_summation_3sec = rfcfo_summation.reshape([len(self.cfo), -1, self.num])
        rfcfo_summation_3sec = np.average(rfcfo_summation_3sec, axis=2)

        return ppcfo_summation_3sec, rpcfo_summation_3sec, pfcfo_summation_3sec, rfcfo_summation_3sec

    def tf_summation_cfo(self, graph=False, mode='no_abs'):
        summation = self.cfo[0]['i1_p_pcfo'][self.start_num:self.end_num]
        types = ['_p_pcfo_tf', '_r_pcfo_tf', '_p_fcfo_tf', '_r_fcfo_tf']
        for type in types:
            for j in range(len(self.cfo)):
                data = self.cfo[j]
                summation_ = data['i1_p_pcfo'][self.start_num:self.end_num]
                for i in range(self.join):
                    interfacenum = 'i' + str(i + 1)
                    pcfoname = interfacenum + type

                    summation_ = np.vstack((summation_, data[pcfoname][self.start_num:self.end_num]))
                summation_ = np.delete(summation_, 0, 0)
                if mode == 'b_abs':
                    summation_ = np.abs(summation_)
                summation = np.vstack((summation, np.sum(summation_, axis=0)))

            # print(summation_)
        summation = np.delete(summation, 0, 0)
        # print(summation.shape)
        summation = summation.reshape([4, len(self.cfo), -1])
        summation = summation / self.cfo[0]['join'][0]
        if mode == 'a_abs':
            summation = np.abs(summation)

        if graph == True:
            fig, (ppcfo, rpcfo, pfcfo, rfcfo) = plt.subplots(4, 1, figsize=(5, 7), dpi=150, sharex=True)

            # plt.xlim([10, 60])  # x軸の範囲
            # plt.xlim([0.28, 0.89])  # x軸の範囲
            plt.xlabel("Time (sec)")

            for i in range(len(self.cfo)):
                data = self.cfo[i]

                ppcfo.plot(data['time'][self.start_num:self.end_num:10]-20.0, summation[0][i][::10],
                           label='Group' + str(i + 1))
                rpcfo.plot(data['time'][self.start_num:self.end_num:10]-20.0, summation[1][i][::10],
                           label='Group' + str(i + 1))
                pfcfo.plot(data['time'][self.start_num:self.end_num:10]-20.0, summation[2][i][::10],
                           label='Group' + str(i + 1))
                rfcfo.plot(data['time'][self.start_num:self.end_num:10]-20.0, summation[3][i][::10],
                           label='Group' + str(i + 1))

            ppcfo.set_ylabel('Summation\nPitch PCFO (rad)')
            rpcfo.set_ylabel('Summation\nRoll PCFO (rad)')
            pfcfo.set_ylabel('Summation\nPitch FCFO (Nm)')
            rfcfo.set_ylabel('Summation\nRoll FCFO (Nm)')

            ppcfo.legend(ncol=10, columnspacing=1, loc='upper left')
            rpcfo.legend(ncol=10, columnspacing=1, loc='upper left')
            pfcfo.legend(ncol=10, columnspacing=1, loc='upper left')
            rfcfo.legend(ncol=10, columnspacing=1, loc='upper left')

            ppcfo.set_yticks(np.arange(-10, 10, 0.5))
            rpcfo.set_yticks(np.arange(-10, 10, 0.5))
            pfcfo.set_yticks(np.arange(-8.0, 8.0, 1.0))
            rfcfo.set_yticks(np.arange(-8.0, 8.0, 1.0))

            ppcfo.set_ylim([0, 1.0])  # y軸の範囲
            rpcfo.set_ylim([0, 1.0])  # y軸の範囲
            pfcfo.set_ylim([0, 4.0])  # y軸の範囲
            rfcfo.set_ylim([0, 4.0])  # y軸の範囲

            plt.tight_layout()
            plt.savefig('fig/tf_summation_cfo_' + str(self.group_type) + '.png')
            plt.show()

        return summation[0], summation[1], summation[2], summation[3]

    def work_calc(self, mode='human'):
        flabel = ['_p_text_tf', '_r_text_tf']
        plabel = ['_p_thm_tf', '_r_thm_tf']
        flabel = ['_p_text', '_r_text']
        plabel = ['_p_thm', '_r_thm']

        if mode == 'model':
            flabel = ['_p_text_pre_tf', '_r_text_pre_tf']
            plabel = ['_p_thm_pre_tf', '_r_thm_pre_tf']
            flabel = ['_p_text_pre', '_r_text_pre']
            plabel = ['_p_thm_pre', '_r_thm_pre']

        omegac = 10.0
        Ts = 0.0001

        work = []
        for i in range(len(self.cfo)):
            data = self.cfo[i]
            for j in range(self.join):
                interfacenum = 'i' + str(j + 1)
                for k in range(len(flabel)):
                    thm_data = data[interfacenum + plabel[k]][self.start_num:self.end_num]
                    thm_data_lpf = CFO.calc_lpf_1st_order(self, thm_data, omegac, Ts)
                    thm_data_ = np.append(np.zeros(1), thm_data_lpf)
                    thm_diff = np.diff(thm_data_)

                    text_data = data[interfacenum + flabel[k]][self.start_num:self.end_num]
                    text_data_lpf = CFO.calc_lpf_1st_order(self, text_data, omegac, Ts)

                    diff = np.abs(thm_diff * text_data_lpf)

                    work.append(diff)

        work_np_ = np.concatenate([work[_] for _ in range(len(work))])
        work_np = work_np_.reshape([len(self.cfo), self.join, len(flabel), -1])
        return work_np

    def work_calc_rs(self, mode='human'):
        flabel = ['_p_text_tf', '_r_text_tf']
        plabel = ['_p_thm_tf', '_r_thm_tf']
        flabel = ['_p_text', '_r_text']
        plabel = ['_p_thm', '_r_thm']

        if mode == 'model':
            flabel = ['_p_text_pre_tf', '_r_text_pre_tf']
            plabel = ['_p_thm_pre_tf', '_r_thm_pre_tf']
            flabel = ['_p_text_pre', '_r_text_pre']
            plabel = ['_p_thm_pre', '_r_thm_pre']

        omegac = 10.0
        Ts = 0.0001

        work = []
        for i in range(len(self.cfo)):
            data = self.cfo[i]
            for j in range(self.join):
                interfacenum = 'i' + str(j + 1)
                diff = []
                for k in range(len(flabel)):
                    thm_data = data[interfacenum + plabel[k]][self.start_num:self.end_num]
                    thm_data_lpf = CFO.calc_lpf_1st_order(self, thm_data, omegac, Ts)
                    thm_data_ = np.append(np.zeros(1), thm_data_lpf)
                    thm_diff = np.diff(thm_data_)

                    text_data = data[interfacenum + flabel[k]][self.start_num:self.end_num]
                    text_data_lpf = CFO.calc_lpf_1st_order(self, text_data, omegac, Ts)

                    diff.append(np.abs(thm_diff * text_data_lpf))
                rs = np.sqrt(diff[0] ** 2 + diff[1] ** 2)

                work.append(rs)

        work_np_ = np.concatenate([work[_] for _ in range(len(work))])
        work_np = work_np_.reshape([len(self.cfo), self.join, -1])
        return work_np

    def work_diff(self, graph=False):
        work_human = self.work_calc(mode='human')
        work_model = self.work_calc(mode='model')

        work_diff = work_human - work_model

        if graph == True:
            for j in range(len(self.cfo)):
                data = self.cfo[j]

                fig, (pwork, rwork) = plt.subplots(2, 1, figsize=(5, 8), dpi=150, sharex=True)

                # plt.xlim([10, 60])  # x軸の範囲
                # plt.xlim([0.28, 0.89])  # x軸の範囲
                plt.xlabel("Time (sec)")

                for i in range(self.join):
                    interfacenum = 'i' + str(i + 1)
                    pwork.plot(data['time'][self.start_num:self.end_num:10]-20.0, work_human[j][i][0][::10],
                               label='P' + str(i + 1) + 'Human')
                    pwork.plot(data['time'][self.start_num:self.end_num:10]-20.0, work_model[j][i][0][::10],
                               label='P' + str(i + 1) + 'Model')
                    # pwork.plot(data['time'][self.start_num:self.end_num:10]-20.0, work_diff[j][i][0][::10], label='P' + str(i + 1) + 'Diff')

                    rwork.plot(data['time'][self.start_num:self.end_num:10]-20.0, work_human[j][i][1][::10],
                               label='P' + str(i + 1) + 'Human')
                    rwork.plot(data['time'][self.start_num:self.end_num:10]-20.0, work_model[j][i][1][::10],
                               label='P' + str(i + 1) + 'Model')
                    # rwork.plot(data['time'][self.start_num:self.end_num:10]-20.0, work_diff[j][i][1][::10], label='P' + str(i + 1) + 'Diff')

                pwork.set_ylabel('Pitch work (J)')
                pwork.legend(ncol=2, columnspacing=1, loc='upper left')
                # pwork.set_yticks(np.arange(-10, 10, 0.5))
                # pwork.set_ylim([-1.5, 1.5])  # y軸の範囲

                rwork.set_ylabel('Roll work (J)')
                rwork.legend(ncol=2, columnspacing=1, loc='upper left')
                # rwork.set_yticks(np.arange(-10, 10, 0.5))
                # rwork.set_ylim([-1.5, 1.5])  # y軸の範囲

                plt.tight_layout()
                # plt.savefig("response.png")
            plt.show()

        return work_diff

    def work_diff_rs(self, graph=False):
        work_human = self.work_calc_rs(mode='human')
        work_model = self.work_calc_rs(mode='model')

        work_diff = work_human - work_model

        if graph == True:
            for j in range(len(self.cfo)):
                data = self.cfo[j]

                fig = plt.figure(figsize=(5, 4), dpi=150)

                # plt.xlim([10, 60])  # x軸の範囲
                # plt.xlim([0.28, 0.89])  # x軸の範囲
                plt.xlabel("Time (sec)")

                for i in range(self.join):
                    interfacenum = 'i' + str(i + 1)
                    plt.plot(data['time'][self.start_num:self.end_num:10]-20.0, work_human[j][i][::10],
                             label='P' + str(i + 1) + 'Human')
                    plt.plot(data['time'][self.start_num:self.end_num:10]-20.0, work_model[j][i][::10],
                             label='P' + str(i + 1) + 'Model')
                    plt.plot(data['time'][self.start_num:self.end_num:10]-20.0, work_diff[j][i][::10],
                             label='P' + str(i + 1) + 'Diff')

                plt.ylabel('Pitch work (J)')
                plt.legend(ncol=2, columnspacing=1, loc='upper left')
                # pwork.set_yticks(np.arange(-10, 10, 0.5))
                # pwork.set_ylim([-1.5, 1.5])  # y軸の範囲

                plt.tight_layout()
                # plt.savefig("response.png")
            plt.show()

        return work_diff

    def get_summation_force(self, graph=False, mode='abs', source='human'):
        summation = self.cfo[0]['i1_p_text'][self.start_num:self.end_num]
        types = ['_p_text', '_r_text']
        if source == 'model':
            types = ['_p_text_pre', '_r_text_pre']
        for type in types:
            for j in range(len(self.cfo)):
                data = self.cfo[j]
                summation_ = data['i1_p_text'][self.start_num:self.end_num]
                for i in range(self.join):
                    interfacenum = 'i' + str(i + 1)
                    dataname = interfacenum + type

                    summation_ = np.vstack((summation_, data[dataname][self.start_num:self.end_num]))
                summation_ = np.delete(summation_, 0, 0)
                # summation_ = np.abs(summation_)
                if mode == 'b_abs':
                    summation_ = np.abs(summation_)
                summation = np.vstack((summation, np.sum(summation_, axis=0)))

            # print(summation_)
        summation = np.delete(summation, 0, 0)
        # print(summation.shape)
        summation = summation.reshape([len(types), len(self.cfo), -1])
        # summation = summation / self.cfo[0]['join'][0]
        if mode == 'a_abs':
            summation = np.abs(summation)

        if graph == True:
            for i in range(len(self.cfo)):
                fig, (ptext, rtxt) = plt.subplots(2, 1, figsize=(5, 7), dpi=150, sharex=True)

                # plt.xlim([10, 60])  # x軸の範囲
                # plt.xlim([0.28, 0.89])  # x軸の範囲
                plt.xlabel("Time (sec)")

                data = self.cfo[i]

                for j in range(self.join):
                    interfacenum = 'i' + str(j + 1)
                    ptext.plot(data['time'][self.start_num:self.end_num:10]-20.0,
                               data[interfacenum + types[0]][self.start_num:self.end_num:10],
                               label='P' + str(j + 1))
                    rtxt.plot(data['time'][self.start_num:self.end_num:10]-20.0,
                              data[interfacenum + types[1]][self.start_num:self.end_num:10],
                              label='P' + str(j + 1))

                ptext.plot(data['time'][self.start_num:self.end_num:10]-20.0, summation[0][i][::10],
                           label='Summation_' + mode)
                rtxt.plot(data['time'][self.start_num:self.end_num:10]-20.0, summation[1][i][::10],
                          label='Summation_' + mode)

                ptext.set_ylabel('force (Nm)')
                rtxt.set_ylabel('force (Nm)')

                ptext.legend(ncol=10, columnspacing=1, loc='upper left')
                rtxt.legend(ncol=10, columnspacing=1, loc='upper left')

                ptext.set_yticks(np.arange(-10, 10, 0.5))
                rtxt.set_yticks(np.arange(-10, 10, 0.5))

                ptext.set_ylim([-5, 5.0])  # y軸の範囲
                rtxt.set_ylim([-5, 5.0])  # y軸の範囲

                plt.tight_layout()
                # os.makedirs('fig/summation_force', exist_ok=True)
                # plt.savefig('fig/summation_force/' + str(self.group_type) + '.png')
            plt.show()

        return summation[0], summation[1]

    # def get_summation_force_3sec(self):
    #     ptext_summation, rtext_summation = CFO.get_summation_force(self)
    #
    #     ptext_summation_3sec = ptext_summation.reshape([len(self.cfo), -1, self.num])
    #     ptext_summation_3sec = np.average(ptext_summation_3sec, axis=2)
    #
    #     rtext_summation_3sec = rtext_summation.reshape([len(self.cfo), -1, self.num])
    #     rtext_summation_3sec = np.average(rtext_summation_3sec, axis=2)
    #
    #     return ptext_summation_3sec, rtext_summation_3sec

    def get_plate_accel(self, graph=False):
        pitch_sum_force, roll_sum_force = CFO.get_summation_force(self, mode='no_abs')

        if graph == True:
            for i in range(len(self.cfo)):
                data = self.cfo[i]

                fig, (x, y) = plt.subplots(2, 1, figsize=(5, 5), dpi=150, sharex=True)

                x.scatter(pitch_sum_force[0][::10], data['pitch_ddot'][self.start_num:self.end_num:10], s=2)
                x.set_xlabel('sumation force')
                x.set_ylabel('pitch_ddot')
                # x.legend(ncol=2, columnspacing=1, loc='upper left')
                x.set_ylim([-20, 20.0])  # y軸の範囲
                x.set_xlim([-4, 4.0])  # y軸の範囲

                y.scatter(roll_sum_force[0][::10], data['roll_ddot'][self.start_num:self.end_num:10], s=2)
                y.set_xlabel('sumation force')
                y.set_ylabel('roll_ddot')
                # x.legend(ncol=2, columnspacing=1, loc='upper left')
                y.set_ylim([-20, 20.0])  # y軸の範囲
                y.set_xlim([-4, 4.0])  # y軸の範囲

                plt.tight_layout()
                # plt.savefig("First_time_target_movement.png")
            plt.show()

        # print(self.cfo[0]['pitch_ddot'][self.start_num:self.end_num:10])
        pitch_ddot = np.vstack((self.cfo[_]['pitch_ddot'][self.start_num:self.end_num] for _ in range(len(self.cfo))))
        roll_ddot = np.vstack((self.cfo[_]['roll_ddot'][self.start_num:self.end_num] for _ in range(len(self.cfo))))
        # print(pitch_ddot.shape)

        return pitch_ddot, roll_ddot

    def get_plate_velo(self):
        pitch_dot = np.vstack((self.cfo[_]['pitch_dot'][self.start_num:self.end_num] for _ in range(len(self.cfo))))
        roll_dot = np.vstack((self.cfo[_]['roll_dot'][self.start_num:self.end_num] for _ in range(len(self.cfo))))

        return pitch_dot, roll_dot

    def get_plate_dyanamics(self, graph=False):
        pitch_force, roll_force = CFO.get_summation_force(self, mode='no_abs')
        pitch_accel, roll_accel = CFO.get_plate_accel(self)
        pitch_velo, roll_velo = CFO.get_plate_velo(self)

        force = [pitch_force, roll_force]
        accel = [pitch_accel, roll_accel]
        velo = [pitch_velo, roll_velo]

        data = [
            [pitch_force, pitch_accel, pitch_velo],
            [roll_force, roll_accel, roll_velo]
        ]

        J_list = []
        D_list = []
        for i in range(len(data)):
            J_list.append([])
            D_list.append([])
            for j in range(len(data[i][0])):
                J, D = CFO.calc_plate_dynamics(self, j[0], j[1], j[2])
                J_list[i].append(J)
                D_list[i].append(D)

        pitch_J = np.array(J_list[0])
        pitch_D = np.array(D_list[0])
        roll_J = np.array(J_list[1])
        roll_D = np.array(D_list[1])

        return pitch_J, pitch_D, roll_J, roll_D

    def calc_plate_dynamics(self, force, accel, velo):
        parameter0 = [0.0, 0.0]
        # print(type[i] + "_" + porr[j] + " = ")

        # LSMで最小化（J+D）
        result = optimize.leastsq(fit_func_2nd, parameter0, args=(accel, velo, force))
        J = result[0][0]
        D = result[0][1]
        # print(result)

        # F[j][i].append(J[j][i][k] * datax + D[j][i][k] * datax_dot)

        return J, D

    def get_momentum(self):
        pitch_J, pitch_D, roll_J, roll_D = CFO.get_plate_dyanamics(self)
        pitch_velo, roll_velo = CFO.get_plate_velo(self)

        J = [pitch_J, roll_J]
        velo = [pitch_velo, roll_velo]

        p_list = []
        for i in range(len(J)):
            p_list.append([])
            for j in range(len(J[i])):
                p = CFO.calc_momentum(self, J[i][j], velo[i][j])
                p_list[i].append(p)

        pitch_momentum = np.array(p_list[0])
        roll_momentum = np.array(p_list[1])

        return pitch_momentum, roll_momentum

    def calc_momentum(self, J, velo):
        p = J * velo
        return p

    def get_ftr(self, graph=False, source='human'):
        pitch_force_plate, roll_force_plate = CFO.get_summation_force(self, mode='no_abs', source=source)
        pitch_force_all, roll_force_all = CFO.get_summation_force(self, mode='b_abs', source=source)

        force_plate = [pitch_force_plate, roll_force_plate]
        force_all = [pitch_force_all, roll_force_all]

        ftr = []
        for i in range(len(force_plate)):
            ftr.append([])
            ftr[i] = np.abs(force_plate[i]) / force_all[i]

        if graph == True:
            axis = ['pitch', 'roll']
            for i in range(len(self.cfo)):
                data = self.cfo[i]
                fig, ax = plt.subplots(2, 1, figsize=(5, 5), dpi=150, sharex=True)

                for j in range(len(axis)):
                    ax2 = ax[j].twinx()
                    ax[j].plot(data['time'][self.start_num:self.end_num:10]-20.0, force_all[j][i][::10], label='Total')
                    ax[j].plot(data['time'][self.start_num:self.end_num:10]-20.0, force_plate[j][i][::10], label='Plate')
                    ax2.plot(data['time'][self.start_num:self.end_num:10]-20.0, ftr[j][i][::10], label='Ratio', color='red')
                    ax[j].set_xlabel('Time (sec)')
                    ax[j].set_ylabel(axis[j] + ' force')
                    ax2.set_ylabel('FTR')
                    ax[j].legend(ncol=2, columnspacing=1, loc='upper left')
                    ax[j].set_ylim([-0, 4])  # y軸の範囲
                    ax2.set_ylim([0, 2])  # y軸の範囲
                plt.tight_layout()
                # plt.savefig("First_time_target_movement.png")
            plt.show()

        return ftr[0], ftr[1]

    def get_ftr_3sec(self, graph=False, source='human'):
        pitch_ftr, roll_ftr = CFO.get_ftr(self, source=source)
        pitch_ftr_3sec = CFO.calc_period_calculator(self, pitch_ftr)
        roll_ftr_3sec = CFO.calc_period_calculator(self, roll_ftr)

        if graph == True:
            axis = ['pitch', 'roll']
            for i in range(len(self.cfo)):
                fig = plt.figure(figsize=(5, 3), dpi=300)

                plt.xticks(np.arange(1, self.period + 1, 1))
                plt.plot(np.arange(self.period) + 1, pitch_ftr_3sec[i], label='pitch')
                plt.plot(np.arange(self.period) + 1, roll_ftr_3sec[i], label='roll')
                plt.ylabel('FTR')
                plt.ylim([0, 1.0])
                plt.xlabel('Period')
                plt.legend()
                plt.tight_layout()
            plt.show()

        return pitch_ftr_3sec, roll_ftr_3sec

    def get_ief(self, graph=False, source='human'):
        pitch_force_plate, roll_force_plate = CFO.get_summation_force(self, mode='no_abs', source=source)
        pitch_force_all, roll_force_all = CFO.get_summation_force(self, mode='b_abs', source=source)

        force_plate = [pitch_force_plate, roll_force_plate]
        force_all = [pitch_force_all, roll_force_all]

        ief = []
        for i in range(len(force_plate)):
            ief.append([])
            ief[i] = force_all[i] - np.abs(force_plate[i])

        if graph == True:
            axis = ['pitch', 'roll']
            for i in range(len(self.cfo)):
                data = self.cfo[i]
                fig, ax = plt.subplots(2, 1, figsize=(5, 5), dpi=150, sharex=True)

                for j in range(len(axis)):
                    ax2 = ax[j].twinx()
                    ax[j].plot(data['time'][self.start_num:self.end_num:10]-20.0, force_all[j][i][::10], label='Total')
                    ax[j].plot(data['time'][self.start_num:self.end_num:10]-20.0, force_plate[j][i][::10], label='Plate')
                    ax2.plot(data['time'][self.start_num:self.end_num:10]-20.0, ief[j][i][::10], label='Ratio', color='red')
                    ax[j].set_xlabel('Time (sec)')
                    ax[j].set_ylabel(axis[j] + ' force')
                    ax2.set_ylabel('Ineffective force')
                    ax[j].legend(ncol=2, columnspacing=1, loc='upper left')
                    ax[j].set_ylim([-0, 4])  # y軸の範囲
                    ax2.set_ylim([0, 2])  # y軸の範囲
                plt.tight_layout()
                # plt.savefig("First_time_target_movement.png")
            plt.show()

        return ief[0], ief[1]

    def get_ief_3sec(self, graph=False, source='human'):
        pitch_ief, roll_ief = CFO.get_ief(self, source=source)
        pitch_ief_3sec = CFO.calc_period_calculator(self, pitch_ief)
        roll_ief_3sec = CFO.calc_period_calculator(self, roll_ief)

        if graph == True:
            axis = ['pitch', 'roll']
            for i in range(len(self.cfo)):
                fig = plt.figure(figsize=(5, 3), dpi=300)

                plt.xticks(np.arange(1, self.period + 1, 1))
                plt.plot(np.arange(self.period) + 1, pitch_ief_3sec[i], label='pitch')
                plt.plot(np.arange(self.period) + 1, roll_ief_3sec[i], label='roll')
                plt.ylabel('Ineffective force')
                plt.ylim([0, 1.0])
                plt.xlabel('Period')
                plt.legend()
                plt.tight_layout()
            plt.show()

        return pitch_ief_3sec, roll_ief_3sec

    def get_ief_combine(self, graph=False, source='human'):
        pitch_force_plate, roll_force_plate = CFO.get_summation_force(self, mode='no_abs', source=source)
        pitch_force_all, roll_force_all = CFO.get_summation_force(self, mode='b_abs', source=source)

        force_plate = np.sqrt(pitch_force_plate ** 2 + roll_force_plate ** 2)
        force_all = np.sqrt(pitch_force_all ** 2 + roll_force_all ** 2)
        ief = force_all - force_plate

        if graph == True:
            for i in range(len(self.cfo)):
                data = self.cfo[i]
                plt.figure(figsize=(10, 5), dpi=150)

                plt.plot(data['time'][self.start_num:self.end_num:10]-20.0, force_all[i][::10], label='Total')
                plt.plot(data['time'][self.start_num:self.end_num:10]-20.0, force_plate[i][::10], label='Plate')
                plt.xlabel('Time (sec)')
                plt.ylabel('force')
                plt.legend(ncol=2, columnspacing=1, loc='upper left')
                plt.ylim([-0, 6])  # y軸の範囲

                ax2 = plt.twinx()
                ax2.plot(data['time'][self.start_num:self.end_num:10]-20.0, ief[i][::10], label='Ratio', color='red')
                ax2.set_ylabel('Ineffective force')
                ax2.set_ylim([0, 2])  # y軸の範囲

                plt.tight_layout()
                # plt.savefig("First_time_target_movement.png")
            plt.show()

        return ief

    def get_ief_combine_3sec(self, graph=False, source='human'):
        ief = CFO.get_ief_combine(self, source=source)
        ief_3sec = CFO.calc_period_calculator(self, ief)

        if graph == True:
            for i in range(len(self.cfo)):
                fig = plt.figure(figsize=(5, 3), dpi=300)

                plt.xticks(np.arange(1, self.period + 1, 1))
                plt.plot(np.arange(self.period) + 1, ief_3sec[i])
                plt.ylabel('Ineffective force')
                plt.ylim([0, 1.0])
                plt.xlabel('Period')
                plt.tight_layout()
            plt.show()

        return ief_3sec

    def get_ftr_combine(self, graph=False, source='human'):
        pitch_force_plate, roll_force_plate = CFO.get_summation_force(self, mode='no_abs', source=source)
        pitch_force_all, roll_force_all = CFO.get_summation_force(self, mode='b_abs', source=source)

        force_plate = np.sqrt(pitch_force_plate ** 2 + roll_force_plate ** 2)
        force_all = np.sqrt(pitch_force_all ** 2 + roll_force_all ** 2)
        ftr = force_plate / force_all

        if graph == True:
            for i in range(len(self.cfo)):
                data = self.cfo[i]
                plt.figure(figsize=(10, 5), dpi=150)

                plt.plot(data['time'][self.start_num:self.end_num:10]-20.0, force_all[i][::10], label='Total')
                plt.plot(data['time'][self.start_num:self.end_num:10]-20.0, force_plate[i][::10], label='Plate')
                plt.xlabel('Time (sec)')
                plt.ylabel('force')
                plt.legend(ncol=2, columnspacing=1, loc='upper left')
                plt.ylim([-0, 6])  # y軸の範囲

                ax2 = plt.twinx()
                ax2.plot(data['time'][self.start_num:self.end_num:10]-20.0, ftr[i][::10], label='Ratio', color='red')
                ax2.set_ylabel('FTR')
                ax2.set_ylim([0, 2])  # y軸の範囲

                plt.tight_layout()
                # plt.savefig("First_time_target_movement.png")
            plt.show()

        return ftr

    def get_ftr_combine_3sec(self, graph=False, source='human'):
        ftr = CFO.get_ftr_combine(self, source=source)
        ftr_3sec = CFO.calc_period_calculator(self, ftr)

        if graph == True:
            for i in range(len(self.cfo)):
                fig = plt.figure(figsize=(5, 3), dpi=300)

                plt.xticks(np.arange(1, self.period + 1, 1))
                plt.plot(np.arange(self.period) + 1, ftr_3sec[i])
                plt.ylabel('FTR')
                plt.ylim([0, 1.0])
                plt.xlabel('Period')
                plt.tight_layout()
            plt.show()

        return ftr_3sec

    def get_ftr_3sec_diff(self):
        ftr_3sec = CFO.get_ftr_3sec(self, source='human')
        ftr_3sec_model = CFO.get_ftr_3sec(self, source='model')
        ftr_3sec_diff = np.subtract(ftr_3sec, ftr_3sec_model)
        return ftr_3sec_diff

    def get_ftr_combine_3sec_diff(self):
        ftr_3sec = CFO.get_ftr_combine_3sec(self, source='human')
        ftr_3sec_model = CFO.get_ftr_combine_3sec(self, source='model')
        ftr_3sec_diff = ftr_3sec - ftr_3sec_model
        return ftr_3sec_diff

    def get_ief_3sec_diff(self):
        ief_3sec = CFO.get_ief_3sec(self, source='human')
        ief_3sec_model = CFO.get_ief_3sec(self, source='model')
        ief_3sec_diff = np.subtract(ief_3sec, ief_3sec_model)
        return ief_3sec_diff

    def get_ief_combine_3sec_diff(self):
        ief_3sec = CFO.get_ief_combine_3sec(self, source='human')
        ief_3sec_model = CFO.get_ief_combine_3sec(self, source='model')
        ief_3sec_diff = ief_3sec - ief_3sec_model
        return ief_3sec_diff

    def performance_ftr_diff(self):
        error_period, spend_period = CFO.period_performance_cooperation(self)
        ftr = CFO.get_ftr_combine_3sec_diff(self)

        performance = [
            error_period,
            spend_period,
        ]
        performance_label = [
            "RMSE",
            "Time"
        ]

        df_ = []
        for i in range(len(performance)):
            for j in range(len(self.cfo)):
                df_.append(pd.DataFrame({
                    'Performance': performance[i][j],
                    'Cooperative FTR': ftr[j],
                    'Performance_types': performance_label[i],
                    'Group': str(j + 1)
                }))

        df = pd.concat([i for i in df_], axis=0)
        df.reset_index(drop=True, inplace=True)

        # FacetGridを作成してグラフを設定
        g = sns.FacetGrid(df, col="Performance_types", hue="Group", height=4, aspect=1.2, sharey=False, sharex=False)
        g.map(plot_scatter, "Performance", "Cooperative FTR")

        ylims = [(-0.8, 0.2), (-0.8, 0.2)]

        if self.group_type == "dyad":
            xlims = [(-0.015, 0.015), (-0.1, 0.5)]
        if self.group_type == "triad":
            xlims = [(-0.015, 0.015), (-0.1, 0.5)]
        if self.group_type == "tetrad":
            xlims = [(-0.015, 0.015), (-0.1, 0.5)]

        # Iterate over the AxesSubplots in the FacetGrid object, and set the y limit individually
        for num, ax in enumerate(g.axes.flatten()):
            # col_var = ax.get_title().split(" ")[-1]  # get the column variable from the title
            ax.set_xlim(xlims[num])
            ax.set_ylim(ylims[num])

        os.makedirs("fig/FTR-Performance/", exist_ok=True)
        plt.savefig("fig/FTR-Performance/Cooperative FTR-Performance_" + self.group_type + ".png")

        plt.show()

    def performance_ief_diff(self):
        error_period, spend_period = CFO.period_performance_cooperation(self)
        ief = CFO.get_ief_combine_3sec_diff(self)

        performance = [
            error_period,
            spend_period,
        ]
        performance_label = [
            "RMSE",
            "Time"
        ]

        df_ = []
        for i in range(len(performance)):
            for j in range(len(self.cfo)):
                df_.append(pd.DataFrame({
                    'Performance': performance[i][j],
                    'Cooperative IEF': ief[j],
                    'Performance_types': performance_label[i],
                    'Group': str(j + 1)
                }))

        df = pd.concat([i for i in df_], axis=0)
        df.reset_index(drop=True, inplace=True)

        # FacetGridを作成してグラフを設定
        g = sns.FacetGrid(df, col="Performance_types", hue="Group", height=4, aspect=1.2, sharey=False, sharex=False)
        g.map(plot_scatter, "Performance", "Cooperative IEF")

        if self.group_type == "dyad":
            ylims = [(-0.5, 3.0), (-0.5, 3.0)]
            xlims = [(-0.015, 0.015), (-0.1, 0.5)]
        if self.group_type == "triad":
            ylims = [(0.0, 5.0), (0.0, 5.0)]
            xlims = [(-0.015, 0.015), (-0.1, 0.5)]
        if self.group_type == "tetrad":
            ylims = [(0.0, 5.0), (0.0, 5.0)]
            xlims = [(-0.015, 0.015), (-0.1, 0.5)]

        # Iterate over the AxesSubplots in the FacetGrid object, and set the y limit individually
        for num, ax in enumerate(g.axes.flatten()):
            # col_var = ax.get_title().split(" ")[-1]  # get the column variable from the title
            ax.set_xlim(xlims[num])
            ax.set_ylim(ylims[num])

        os.makedirs("fig/IEF-Performance/", exist_ok=True)
        plt.savefig("fig/IEF-Performance/Cooperative IEF-Performance_" + self.group_type + ".png")

        plt.show()

    def calc_lpf_1st_order(self, data, omegac, Ts):
        T = omegac * Ts

        data_now = 0.0
        data_old = 0.0
        data_lpf_now = 0.0
        data_lpf_old = 0.0

        data_lpf = np.zeros(len(data))
        for l in range(len(data)):
            data_now = data[l]
            data_lpf_now = (2 - T) / (2 + T) * data_lpf_old + T / (2 + T) * (data_now + data_old)
            data_lpf[l] = data_lpf_now
            data_lpf_old = data_lpf_now
            data_old = data_now

        return data_lpf

    def summationCFO_ftr(self, mode='no_abs'):
        pcfo_p, pcfo_r, fcfo_p, fcfo_r = CFO.summation_cfo(self, mode=mode)
        ratio_p, ratio_r = CFO.get_ftr(self)

        axis = ['pitch', 'roll']
        cfo = ['PCFO', 'FCFO']

        cfo_data = [
            [pcfo_p, pcfo_r],
            [fcfo_p, fcfo_r]
        ]

        ratio_data = [
            ratio_p, ratio_r
        ]

        df_ = []
        for i in range(len(axis)):
            for j in range(len(cfo)):
                for k in range(len(self.cfo)):
                    df_.append(pd.DataFrame({
                        'CFO': cfo_data[i][j][k][::100],
                        'FTR': ratio_data[j][k][::100],
                        'axis': axis[j],
                        'CFO_types': cfo[i],
                        'Group': k + 1
                    })
                    )

        df = pd.concat([i for i in df_], axis=0)
        df.reset_index(drop=True, inplace=True)
        # print(df)

        # FacetGridを作成してグラフを設定
        g = sns.FacetGrid(df, col="CFO_types", row="axis", hue="Group", height=4, aspect=1.2, sharey=False,
                          sharex=False)
        g.map(plot_scatter, "CFO", "FTR")

        ylims = [(0, 1), (0, 1), (0, 1), (0, 1)]

        if self.group_type == "dyad":
            xlims = [(0, 0.6), (0.0, 2.0),
                     (0, 0.6), (0.0, 2.0)]

            if mode == 'b_abs':
                xlims = [(0, 0.6), (0.0, 3.0),
                         (0, 0.6), (0.0, 3.0)]
            if mode == 'a_abs':
                xlims = [(0, 0.6), (0.0, 2.0),
                         (0, 0.6), (0.0, 2.0)]

        if self.group_type == "triad":
            xlims = [(0, 0.6), (0.0, 2.0),
                     (0, 0.6), (0.0, 2.0)]

            if mode == 'b_abs':
                xlims = [(0, 0.6), (0.0, 3.0),
                         (0, 0.6), (0.0, 3.0)]
            if mode == 'a_abs':
                xlims = [(0, 0.6), (0.0, 2.0),
                         (0, 0.6), (0.0, 2.0)]

        if self.group_type == "tetrad":
            xlims = [(0, 0.6), (0.0, 2.0),
                     (0, 0.6), (0.0, 2.0)]

            if mode == 'b_abs':
                xlims = [(0, 0.6), (0.0, 3.0),
                         (0, 0.6), (0.0, 3.0)]
            if mode == 'a_abs':
                xlims = [(0, 0.6), (0.0, 2.0),
                         (0, 0.6), (0.0, 2.0)]

        # Iterate over the AxesSubplots in the FacetGrid object, and set the y limit individually
        for num, ax in enumerate(g.axes.flatten()):
            # col_var = ax.get_title().split(" ")[-1]  # get the column variable from the title
            ax.set_xlim(xlims[num])
            ax.set_ylim(ylims[num])

        os.makedirs("fig/FTR-CFO/Time", exist_ok=True)
        plt.savefig("fig/FTR-CFO/Time/FTR-SummationCFO(" + mode + ")_" + self.group_type + ".png")

        plt.show()

    def summationCFO_ftr_3sec(self, mode='no_abs'):
        pcfo_p, pcfo_r, fcfo_p, fcfo_r = CFO.summation_cfo_3sec(self, mode=mode)
        ftr_p, ftr_r = CFO.get_ftr_3sec(self)

        axis = ['pitch', 'roll']
        cfo = ['PCFO', 'FCFO']

        cfo_data = [
            [pcfo_p, pcfo_r],
            [fcfo_p, fcfo_r]
        ]

        ftr_data = [
            ftr_p, ftr_r
        ]

        df_ = []
        for i in range(len(axis)):
            for j in range(len(cfo)):
                for k in range(len(self.cfo)):
                    df_.append(pd.DataFrame({
                        'CFO': cfo_data[i][j][k],
                        'FTR': ftr_data[j][k],
                        'axis': axis[j],
                        'CFO_types': cfo[i],
                        'Group': k + 1
                    })
                    )

        df = pd.concat([i for i in df_], axis=0)
        df.reset_index(drop=True, inplace=True)
        # print(df)

        # fig = plt.figure(figsize=(5, 3), dpi=300)

        # FacetGridを作成してグラフを設定
        g = sns.FacetGrid(df, col="CFO_types", row="axis", hue="Group", height=4, aspect=1.2, sharey=False,
                          sharex=False)
        g.map(plot_scatter, "CFO", "FTR")

        ylims = [(0, 1), (0, 1), (0, 1), (0, 1)]

        if self.group_type == "dyad":
            xlims = [(0, 0.04), (0.0, 0.1),
                     (0, 0.04), (0.0, 0.1)]

            if mode == 'b_abs':
                xlims = [(0, 0.25), (0.0, 2.0),
                         (0, 0.25), (0.0, 3.0)]
            if mode == 'a_abs':
                xlims = [(0, 0.25), (0.0, 0.5),
                         (0, 0.25), (0.0, 0.8)]

        if self.group_type == "triad":
            xlims = [(0, 0.04), (0.0, 0.2),
                     (0, 0.04), (0.0, 0.2)]

            if mode == 'b_abs':
                xlims = [(0, 0.2), (0.0, 1.5),
                         (0, 0.2), (0.0, 1.5)]
            if mode == 'a_abs':
                xlims = [(0, 0.25), (0.0, 0.5),
                         (0, 0.25), (0.0, 0.8)]

        if self.group_type == "tetrad":
            xlims = [(0, 0.03), (0.0, 0.1),
                     (0, 0.03), (0.0, 0.1)]

            if mode == 'b_abs':
                xlims = [(0, 0.15), (0.0, 1.5),
                         (0, 0.15), (0.0, 1.5)]
            if mode == 'a_abs':
                xlims = [(0, 0.15), (0.0, 0.5),
                         (0, 0.15), (0.0, 0.5)]

        # Iterate over the AxesSubplots in the FacetGrid object, and set the y limit individually
        for num, ax in enumerate(g.axes.flatten()):
            # col_var = ax.get_title().split(" ")[-1]  # get the column variable from the title
            ax.set_xlim(xlims[num])
            ax.set_ylim(ylims[num])

        os.makedirs("fig/FTR-CFO/Period", exist_ok=True)
        plt.savefig("fig/FTR-CFO/Period/FTR-SummationCFO(" + mode + ")_" + self.group_type + ".png")

        plt.show()

    def summationCFO_ief_3sec(self, mode='no_abs'):
        pcfo_p, pcfo_r, fcfo_p, fcfo_r = CFO.summation_cfo_3sec(self, mode=mode)
        ief_p, ief_r = CFO.get_ief_3sec(self)

        axis = ['pitch', 'roll']
        cfo = ['PCFO', 'FCFO']

        cfo_data = [
            [pcfo_p, pcfo_r],
            [fcfo_p, fcfo_r]
        ]

        ief_data = [
            ief_p, ief_r
        ]

        df_ = []
        for i in range(len(axis)):
            for j in range(len(cfo)):
                for k in range(len(self.cfo)):
                    df_.append(pd.DataFrame({
                        'CFO': cfo_data[i][j][k],
                        'IEF': ief_data[j][k],
                        'axis': axis[j],
                        'CFO_types': cfo[i],
                        'Group': k + 1
                    })
                    )

        df = pd.concat([i for i in df_], axis=0)
        df.reset_index(drop=True, inplace=True)
        # print(df)

        # fig = plt.figure(figsize=(5, 3), dpi=300)

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

        # FacetGridを作成してグラフを設定
        g = sns.FacetGrid(df, col="CFO_types", row="axis", hue="Group", height=8, aspect=1.2, sharey=False,
                          sharex=False)
        g.map(plot_scatter, "CFO", "IEF")

        if self.group_type == "dyad":
            ylims = [(0.0, 1.5), (0, 1.5),
                     (0.0, 4.0), (0, 4.0)]
            xlims = [(0, 0.04), (0.0, 0.1),
                     (0, 0.04), (0.0, 0.1)]

            if mode == 'b_abs':
                xlims = [(0, 0.25), (0.0, 2.0),
                         (0, 0.25), (0.0, 3.0)]
            if mode == 'a_abs':
                xlims = [(0, 0.25), (0.0, 0.5),
                         (0, 0.25), (0.0, 0.8)]

        if self.group_type == "triad":
            ylims = [(0.0, 3.0), (0, 3.0),
                     (0.0, 4.0), (0, 4.0)]
            xlims = [(0, 0.04), (0.0, 0.2),
                     (0, 0.04), (0.0, 0.2)]

            if mode == 'b_abs':
                xlims = [(0, 0.2), (0.0, 1.5),
                         (0, 0.2), (0.0, 1.5)]
            if mode == 'a_abs':
                xlims = [(0, 0.25), (0.0, 0.5),
                         (0, 0.25), (0.0, 0.8)]

        if self.group_type == "tetrad":
            ylims = [(0.0, 6.0), (0, 6.0),
                     (0.0, 6.0), (0, 6.0)]
            xlims = [(0, 0.03), (0.0, 0.1),
                     (0, 0.03), (0.0, 0.1)]

            if mode == 'b_abs':
                xlims = [(0, 0.15), (0.0, 1.5),
                         (0, 0.15), (0.0, 1.5)]
            if mode == 'a_abs':
                xlims = [(0, 0.15), (0.0, 0.5),
                         (0, 0.15), (0.0, 0.5)]

        # Iterate over the AxesSubplots in the FacetGrid object, and set the y limit individually
        for num, ax in enumerate(g.axes.flatten()):
            # col_var = ax.get_title().split(" ")[-1]  # get the column variable from the title
            ax.set_xlim(xlims[num])
            ax.set_ylim(ylims[num])

        os.makedirs("fig/IEF-CFO/Period", exist_ok=True)
        plt.savefig("fig/IEF-CFO/Period/IEF-SummationCFO(" + mode + ")_" + self.group_type + ".png")

        plt.show()

    def subtractionCFO_ftr(self):
        pcfo_p, pcfo_r, fcfo_p, fcfo_r = CFO.subtraction_cfo(self)
        ftr_p, ftr_r = CFO.get_ftr(self)

        axis = ['pitch', 'roll']
        cfo = ['PCFO', 'FCFO']

        cfo_data = [
            [pcfo_p, pcfo_r],
            [fcfo_p, fcfo_r]
        ]

        ftr_data = [
            ftr_p, ftr_r
        ]

        df_ = []
        for i in range(len(axis)):
            for j in range(len(cfo)):
                for k in range(len(self.cfo)):
                    df_.append(pd.DataFrame({
                        'CFO': cfo_data[i][j][k][::100],
                        'FTR': ftr_data[j][k][::100],
                        'axis': axis[j],
                        'CFO_types': cfo[i],
                        'Group': k + 1
                    })
                    )

        df = pd.concat([i for i in df_], axis=0)
        df.reset_index(drop=True, inplace=True)
        # print(df)

        # fig = plt.figure(figsize=(5, 3), dpi=300)

        # FacetGridを作成してグラフを設定
        g = sns.FacetGrid(df, col="CFO_types", row="axis", hue="Group", height=4, aspect=1.2, sharex=False)
        g.map(plot_scatter, "CFO", "FTR")

        ylims = [(0, 1), (0, 1), (0, 1), (0, 1)]

        if self.group_type == "dyad":
            xlims = [(0, 0.5), (0.0, 8.0),
                     (0, 0.5), (0.0, 8.0)]

        if self.group_type == "triad":
            xlims = [(0, 1.0), (0.0, 10.0),
                     (0, 1.0), (0.0, 10.0)]

        if self.group_type == "tetrad":
            xlims = [(0, 1.0), (0.0, 10.0),
                     (0, 1.0), (0.0, 10.0)]

        # Iterate over the AxesSubplots in the FacetGrid object, and set the y limit individually
        for num, ax in enumerate(g.axes.flatten()):
            # col_var = ax.get_title().split(" ")[-1]  # get the column variable from the title
            ax.set_xlim(xlims[num])
            ax.set_ylim(ylims[num])

        os.makedirs("fig/FTR-CFO/Time/", exist_ok=True)
        plt.savefig("fig/FTR-CFO/Time/FTR-SubtractionCFO_" + self.group_type + ".png")

        plt.show()

    def subtractionCFO_ftr_3sec(self):
        pcfo_p, pcfo_r, fcfo_p, fcfo_r = CFO.subtraction_cfo_3sec(self)
        ftr_p, ftr_r = CFO.get_ftr_3sec(self)

        axis = ['pitch', 'roll']
        cfo = ['PCFO', 'FCFO']

        cfo_data = [
            [pcfo_p, pcfo_r],
            [fcfo_p, fcfo_r]
        ]

        ftr_data = [
            ftr_p, ftr_r
        ]

        df_ = []
        for i in range(len(axis)):
            for j in range(len(cfo)):
                for k in range(len(self.cfo)):
                    df_.append(pd.DataFrame({
                        'CFO': cfo_data[i][j][k],
                        'FTR': ftr_data[j][k],
                        'axis': axis[j],
                        'CFO_types': cfo[i],
                        'Group': k + 1
                    })
                    )

        df = pd.concat([i for i in df_], axis=0)
        df.reset_index(drop=True, inplace=True)
        # print(df)

        # fig = plt.figure(figsize=(5, 3), dpi=300)

        # sns.lmplot(x="FTR", y="CFO", col="CFO_types", row="axis", hue="Group", data=df,
        #            x_ci=None, fit_reg=False, legend=False, scatter_kws={"s": 10, 'alpha': 0.5},
        #            )

        # FacetGridを作成してグラフを設定
        g = sns.FacetGrid(df, col="CFO_types", row="axis", hue="Group", height=4, aspect=1.2, sharey=False,
                          sharex=False)
        g.map(plot_scatter, "CFO", "FTR")

        ylims = [(0, 1), (0, 1), (0, 1), (0, 1)]

        if self.group_type == "dyad":
            xlims = [(0, 0.25), (0.0, 3.0), (0, 0.25), (0.0, 5.0)]

        if self.group_type == "triad":
            xlims = [(0, 0.25), (0.0, 5.0), (0, 0.25), (0.0, 5.0)]

        if self.group_type == "tetrad":
            xlims = [(0, 0.3), (0.0, 5.0), (0, 0.3), (0.0, 5.0)]

        # Iterate over the AxesSubplots in the FacetGrid object, and set the y limit individually
        for num, ax in enumerate(g.axes.flatten()):
            # col_var = ax.get_title().split(" ")[-1]  # get the column variable from the title
            ax.set_xlim(xlims[num])
            ax.set_ylim(ylims[num])

        os.makedirs("fig/FTR-CFO/Period/", exist_ok=True)
        plt.savefig("fig/FTR-CFO/Period/FTR-SubtractionCFO_" + self.group_type + ".png")

        plt.show()

    def subtractionCFO_ief_3sec(self):
        pcfo_p, pcfo_r, fcfo_p, fcfo_r = CFO.subtraction_cfo_3sec(self)
        ief_p, ief_r = CFO.get_ief_3sec(self)

        axis = ['pitch', 'roll']
        cfo = ['PCFO', 'FCFO']

        cfo_data = [
            [pcfo_p, pcfo_r],
            [fcfo_p, fcfo_r]
        ]

        ief_data = [
            ief_p, ief_r
        ]

        df_ = []
        for i in range(len(axis)):
            for j in range(len(cfo)):
                for k in range(len(self.cfo)):
                    df_.append(pd.DataFrame({
                        'CFO': cfo_data[i][j][k],
                        'IEF': ief_data[j][k],
                        'axis': axis[j],
                        'CFO_types': cfo[i],
                        'Group': k + 1
                    })
                    )

        df = pd.concat([i for i in df_], axis=0)
        df.reset_index(drop=True, inplace=True)
        # print(df)

        # fig = plt.figure(figsize=(5, 3), dpi=300)

        # sns.lmplot(x="IEF", y="CFO", col="CFO_types", row="axis", hue="Group", data=df,
        #            x_ci=None, fit_reg=False, legend=False, scatter_kws={"s": 10, 'alpha': 0.5},
        #            )

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

        # FacetGridを作成してグラフを設定
        g = sns.FacetGrid(df, col="CFO_types", row="axis", hue="Group", height=8, aspect=1.2, sharey=False,
                          sharex=False)
        g.map(plot_scatter, "CFO", "IEF")

        ylims = [(0, 3), (0, 3), (0, 3), (0, 3)]

        if self.group_type == "dyad":
            xlims = [(0, 0.25), (0.0, 3.0), (0, 0.25), (0.0, 5.0)]

        if self.group_type == "triad":
            xlims = [(0, 0.25), (0.0, 5.0), (0, 0.25), (0.0, 5.0)]

        if self.group_type == "tetrad":
            xlims = [(0, 0.3), (0.0, 5.0), (0, 0.3), (0.0, 5.0)]

        # Iterate over the AxesSubplots in the FacetGrid object, and set the y limit individually
        for num, ax in enumerate(g.axes.flatten()):
            # col_var = ax.get_title().split(" ")[-1]  # get the column variable from the title
            ax.set_xlim(xlims[num])
            ax.set_ylim(ylims[num])

        os.makedirs("fig/IEF-CFO/Period/", exist_ok=True)
        plt.savefig("fig/IEF-CFO/Period/IEF-SubtractionCFO_" + self.group_type + ".png")

        plt.show()

    def performance_ftr(self, mode='h-m'):
        error_period, spend_period = CFO.period_performance_cooperation(self)
        ftr = CFO.get_ftr_combine_3sec(self)
        if mode == 'h-h':
            error_period, spend_period = CFO.period_performance_human(self)

        performance = [
            error_period,
            spend_period,
        ]
        performance_label = [
            "RMSE",
            "Time"
        ]

        df_ = []
        for i in range(len(performance)):
            for j in range(len(self.cfo)):
                df_.append(pd.DataFrame({
                    'Performance': performance[i][j],
                    'FTR': ftr[j],
                    'Performance_types': performance_label[i],
                    'Group': str(j + 1)
                }))

        df = pd.concat([i for i in df_], axis=0)
        df.reset_index(drop=True, inplace=True)

        # FacetGridを作成してグラフを設定
        g = sns.FacetGrid(df, col="Performance_types", hue="Group", height=4, aspect=1.2, sharey=False, sharex=False)
        g.map(plot_scatter, "Performance", "FTR")

        ylims = [(0, 1), (0, 1)]

        if self.group_type == "dyad":
            xlims = [(-0.015, 0.015), (-0.1, 0.5)]
            if mode == 'h-h':
                xlims = [(0.0, 0.1), (0.0, 3.0)]

        if self.group_type == "triad":
            xlims = [(-0.015, 0.015), (-0.1, 0.5)]
            if mode == 'h-h':
                xlims = [(0.0, 0.1), (0.0, 3.0)]

        if self.group_type == "tetrad":
            xlims = [(-0.015, 0.015), (-0.1, 0.5)]
            if mode == 'h-h':
                xlims = [(0.0, 0.1), (0.0, 3.0)]

        # Iterate over the AxesSubplots in the FacetGrid object, and set the y limit individually
        for num, ax in enumerate(g.axes.flatten()):
            # col_var = ax.get_title().split(" ")[-1]  # get the column variable from the title
            ax.set_xlim(xlims[num])
            ax.set_ylim(ylims[num])

        if mode == 'h-h':
            os.makedirs("fig/FTR-Performance/h-h/", exist_ok=True)
            plt.savefig("fig/FTR-Performance/h-h/FTR-Performance_" + self.group_type + "_h-h.png")
        else:
            os.makedirs("fig/FTR-Performance/", exist_ok=True)
            plt.savefig("fig/FTR-Performance/FTR-Performance_" + self.group_type + ".png")

        plt.show()

    def performance_ief(self, mode='h-m'):
        error_period, spend_period = CFO.period_performance_cooperation(self)
        ief = CFO.get_ief_combine_3sec(self)
        if mode == 'h-h':
            error_period, spend_period = CFO.period_performance_human(self)

        performance = [
            error_period,
            spend_period,
        ]
        performance_label = [
            "RMSE",
            "Time"
        ]

        df_ = []
        for i in range(len(performance)):
            for j in range(len(self.cfo)):
                df_.append(pd.DataFrame({
                    'Performance': performance[i][j],
                    'IEF': ief[j],
                    'Performance_types': performance_label[i],
                    'Group': str(j + 1)
                }))

        df = pd.concat([i for i in df_], axis=0)
        df.reset_index(drop=True, inplace=True)

        sns.set(font='Times New Roman', font_scale=1.0)
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

        # FacetGridを作成してグラフを設定
        g = sns.FacetGrid(df, col="Performance_types", hue="Group", height=8, aspect=1.2, sharey=False, sharex=False)
        g.map(plot_scatter, "Performance", "IEF")

        if self.group_type == "dyad":
            xlims = [(-0.015, 0.015), (-0.1, 0.5)]
            ylims = [(0, 4), (0, 4)]
            if mode == 'h-h':
                xlims = [(0.0, 0.1), (0.0, 3.0)]

        if self.group_type == "triad":
            xlims = [(-0.015, 0.015), (-0.1, 0.5)]
            ylims = [(0, 6), (0, 6)]

            if mode == 'h-h':
                xlims = [(0.0, 0.1), (0.0, 3.0)]

        if self.group_type == "tetrad":
            xlims = [(-0.015, 0.015), (-0.1, 0.5)]
            ylims = [(0, 8), (0, 8)]

            if mode == 'h-h':
                xlims = [(0.0, 0.1), (0.0, 3.0)]

        # Iterate over the AxesSubplots in the FacetGrid object, and set the y limit individually
        for num, ax in enumerate(g.axes.flatten()):
            # col_var = ax.get_title().split(" ")[-1]  # get the column variable from the title
            ax.set_xlim(xlims[num])
            ax.set_ylim(ylims[num])

        if mode == 'h-h':
            os.makedirs("fig/IEF-Performance/h-h/", exist_ok=True)
            plt.savefig("fig/IEF-Performance/h-h/IEF-Performance_" + self.group_type + "_h-h.png")
        else:
            os.makedirs("fig/IEF-Performance/", exist_ok=True)
            plt.savefig("fig/IEF-Performance/IEF-Performance_" + self.group_type + ".png")

        plt.show()

    def get_force(self, source='human', dec=1, period='false'):
        types = ['_p_text', '_r_text']
        if source == 'model':
            types = ['_p_text_pre', '_r_text_pre']

        force = CFO.get_matrix(self, types, dec, period)

        return force  # [experiment][subject][types][force]

    def get_position(self, source='human', dec=1, period='false'):
        types = ['_p_thm', '_r_thm']
        if source == 'model':
            types = ['_p_thm_pre', '_r_thm_pre']

        position = CFO.get_matrix(self, types, dec, period)

        return position  # [experiment][subject][types][position]

    def get_pcfo(self, dec=1, period='false'):
        types = ['_p_pcfo', '_r_pcfo']
        pcfo = CFO.get_matrix(self, types, dec, period)

        return pcfo  # [experiment][subject][types][pcfo]

    def get_fcfo(self, dec=1, period='false'):
        types = ['_p_fcfo', '_r_fcfo']
        fcfo = CFO.get_matrix(self, types, dec, period)

        return fcfo  # [experiment][subject][types][fcfo]

    def get_matrix(self, types: list, dec=1, period='false'):
        matrix_ = []
        for i in range(len(self.cfo)):
            data = self.cfo[i]
            for j in range(self.join):
                for type in types:
                    interfacenum = 'i' + str(j + 1)
                    dataname = interfacenum + type
                    matrix_.append(data[dataname][self.start_num:self.end_num:dec])

        matrix = np.array([_ for _ in matrix_])
        if period == 'false':
            matrix = matrix.reshape([len(self.cfo), self.join, len(types), -1])
            return matrix  # [experiment][subject][types][data]
        else:
            matrix = matrix.reshape([len(self.cfo), self.join, len(types), self.period, -1])
            return matrix  # [experiment][subject][types][data]

    def calc_hilbert(self, mode='h-h'):
        if mode == 'h-h':
            force = self.get_force(source='human')
            # print(force.shape)
        env, freq_int, phase_inst = HT.hilbert(force, self.smp)
        print(env.shape)

        plt.figure(figsize=(10, 5))
        plt.plot(self.cfo[0]['time'], env[0][0][0])
        plt.plot(self.cfo[0]['time'], force[0][0][0])
        plt.show()

        # return hilbert

    def relative_phase(self, type: str, source: str = 'human', sigma: int = 'none', dec=1, graph=False):
        if type == 'position':
            data = self.get_position(source=source, period='true', dec=dec)
        elif type == 'force':
            data = self.get_force(source=source, period='true', dec=dec)
        elif type == 'pcfo':
            data = self.get_pcfo(period='true', dec=dec)
        elif type == 'fcfo':
            data = self.get_fcfo(period='true', dec=dec)

        # normal
        # env, freq_int, phase_inst = HT.hilbert(data, self.smp)
        # phase_inst = phase_inst.transpose(1, 0, 2, 3, 4)
        # rela_phase = HT.relative_phase(phase_inst[0], phase_inst[1], sigma=sigma)

        # parallel
        pe = ParallelExecutor.ParallelExecutor(HT.hilbert, max_workers=20, args_num=2, return_num=3, para_args_list=[0])
        env, freq_int, phase_inst = pe.parallel(data, self.smp)
        phase_inst = phase_inst.transpose(1, 0, 2, 3, 4)
        pe = ParallelExecutor.ParallelExecutor(HT.calc_relative_phase, max_workers=20, args_num=3, return_num=1, para_args_list=[0, 1])
        rela_phase = pe.parallel(phase_inst[0], phase_inst[1], sigma)


        # plt.plot(self.cfo[0]['time'][0:self.num], rela_phase[0][0][0])

        if graph == True:

            region = np.arange(0, 200, 20)
            freq, edge = hist.frequency(rela_phase, region)
            freq = freq.transpose(0, 3, 2, 1)
            freq = freq.reshape([len(freq), len(freq[0]), -1])
            freq = np.sum(freq, axis=2)

            freq_normal = freq / np.sum(freq, axis=1, keepdims=True) * 100
            # print(len(freq_normal))
            std_error = np.std(freq_normal, axis=0) / np.sqrt(len(freq_normal))
            # print(std_error.shape)

            freq_ave = np.sum(freq_normal, axis=0) / len(freq_normal)
            # std_error = np.arange(3, 30, 3)
            fig = plt.figure(figsize=(10, 10))
            plt.errorbar(edge, freq_ave, yerr=std_error, fmt='o', capsize=6, ecolor='black', markeredgecolor="black",
                         color='white')
            plt.plot(edge, freq_ave, color='black')
            for i in range(len(freq_normal)):
                plt.plot(edge, freq_normal[i], color='gray', alpha=0.5)
            plt.xlim(0, 180)
            plt.ylim(0, 100)
            plt.xlabel('Phase region ($^\circ$)')
            plt.ylabel('Relative phase occurrence')

            os.makedirs("fig/RelativePhase/dyad/error/", exist_ok=True)
            if type == 'position' or type == 'force':
                plt.savefig("fig/RelativePhase/dyad/error/RelativePhase_" + type + "_" + source + ".png")
            else:
                plt.savefig("fig/RelativePhase/dyad/error/RelativePhase_" + type + ".png")

            plt.show()

        return rela_phase

    def relative_phase_filter(self, type: str, source: str = 'human', sigma: int = 'none', dec=1, graph=False,
                              min_freq=0, max_freq=0, step=0):
        if type == 'position':
            data = self.get_position(source=source, period='true', dec=dec)
        elif type == 'force':
            data = self.get_force(source=source, period='true', dec=dec)
        elif type == 'pcfo':
            data = self.get_pcfo(period='true', dec=dec)
        elif type == 'fcfo':
            data = self.get_fcfo(period='true', dec=dec)

        filter_range = np.arange(min_freq, max_freq + step, step)
        rela_phase_ = []
        start = time.time()
        for i in range(len(filter_range) - 1):
            # data_filter = Filter.band_pass_filter(data, self.smp * dec, filter_range[i], filter_range[i + 1])
            # env, freq_int, phase_inst = HT.hilbert(data_filter, self.smp * dec)
            # phase_inst = phase_inst.transpose(1, 0, 2, 3, 4)
            # rela_phase_.append(HT.relative_phase(phase_inst[0], phase_inst[1], sigma=sigma))

            pe = ParallelExecutor.ParallelExecutor(Filter.band_pass_filter, max_workers=20, args_num=4, return_num=1, para_args_list=[0])
            data_filter_ = pe.parallel(data, self.smp * dec, filter_range[i], filter_range[i + 1])
            pe = ParallelExecutor.ParallelExecutor(HT.hilbert, max_workers=20, args_num=2, return_num=3, para_args_list=[0])
            env, freq_int, phase_inst = pe.parallel(data_filter_, self.smp * dec)
            phase_inst = phase_inst.transpose(1, 0, 2, 3, 4)
            pe = ParallelExecutor.ParallelExecutor(HT.calc_relative_phase, max_workers=20, args_num=3, return_num=1, para_args_list=[0, 1])
            rela_phase_.append(pe.parallel(phase_inst[0], phase_inst[1], sigma))
        print('time', time.time() - start)

        rela_phase = np.array([_ for _ in rela_phase_])

        if graph:
            region = np.arange(0, 200, 20)
            pe = ParallelExecutor.ParallelExecutor(hist.frequency, max_workers=20, args_num=2, return_num=2, para_args_list=[0])
            freq, edge = pe.parallel(rela_phase, region)
            edge = edge[0][0][0][0]
            # freq, edge = hist.frequency(rela_phase, region)
            freq = freq.transpose(0, 1, 4, 2, 3)
            freq = freq.reshape([len(freq), len(freq[0]), len(freq[0][0]), -1])
            freq = np.sum(freq, axis=3)

            freq_normal = freq / np.sum(freq, axis=2, keepdims=True) * 100
            # print(len(freq_normal))
            std_error = np.std(freq_normal, axis=1) / np.sqrt(len(freq_normal[0]))
            # print(std_error.shape)

            freq_ave = np.sum(freq_normal, axis=1) / len(freq_normal[0])
            # # std_error = np.arange(3, 30, 3)
            # nrows, ncols = 10, 10
            # if (len(filter_range) - 1) <= 5:
            #     nrows, ncols = 1, 5
            # elif (len(filter_range) - 1) <= 10:
            #     nrows, ncols = 2, 5
            # elif (len(filter_range) - 1) <= 15:
            #     nrows, ncols = 3, 5
            # elif (len(filter_range) - 1) <= 20:
            #     nrows, ncols = 4, 5
            #
            nrows, ncols = (len(filter_range) - 1) // 5, 5

            fig, axs = plt.subplots(nrows, ncols, figsize=(10, 1.5 + nrows * 1.6), dpi=100, sharex='all', sharey='all')


            for i in range(len(freq_ave)):
                if nrows == 1:
                    ax = axs[i]
                else:
                    ax = axs[i // ncols, i % ncols]
                ax.errorbar(edge, freq_ave[i], yerr=std_error[i], fmt='o', capsize=6, ecolor='black', markeredgecolor="black",
                             color='white')
                ax.plot(edge, freq_ave[i], color='black', lw=2)
                for j in range(len(freq_normal[i])):
                    ax.plot(edge, freq_normal[i][j], color='gray', alpha=0.5)
                ax.set_xlim(0, 180)
                ax.set_xticks(np.arange(0, 270, 90))
                ax.set_ylim(0, 100)
                ax.set_xlabel('Phase region ($^\circ$)')
                ax.set_ylabel('Relative phase occurrence')
                ax.set_title(str('{:.2f}'.format(filter_range[i])) + ' - ' + str('{:.2f}'.format(filter_range[i + 1])) + ' Hz')

            plt.tight_layout()
            os.makedirs("fig/RelativePhase/dyad/error/filter/", exist_ok=True)
            if type == 'position' or type == 'force':
                plt.savefig("fig/RelativePhase/dyad/error/filter/RelativePhase_" + type + "_" + source + ".png")
            else:
                plt.savefig("fig/RelativePhase/dyad/error/filter/RelativePhase_" + type + ".png")

            plt.show()

        return rela_phase

    def relative_phase_3sec(self, type: str, source: str = 'human', sigma: int = 'none', dec=1, graph=False):
        rela_phase = self.relative_phase(type=type, source=source, sigma=sigma, dec=dec)

        region = np.arange(0, 200, 20)
        freq, edge = hist.frequency(rela_phase, region)

        freq_normal = freq / np.sum(freq, axis=3, keepdims=True) * 100

        cmap = plt.get_cmap('winter')
        axis = ['Pitch', 'Roll']
        fig, ax = plt.subplots(len(freq_normal), len(axis), figsize=(10, 20), sharex='all', sharey='all')
        fig.supxlabel('Phase region ($^\circ$)')
        fig.supylabel('Relative phase occurrence')
        plt.xlim(0, 180)
        plt.ylim(0, 100)
        plt.xticks(np.arange(0, 200, 20))
        for i in range(len(freq_normal)):
            for j in range(len(freq_normal[0][0])):
                color = cmap(j / (self.period - 1))  # カラーマップの値を0から1に変化
                for k in range(len(freq_normal[0])):
                    ax[i, k].plot(edge, freq_normal[i][k][j], lw=0.5, color=color, alpha=0.4 - k * 0.005)
                    ax[i, k].set_title(axis[k])
                    ax[i, k].set_ylabel('Group ' + str(i + 1))
        plt.tight_layout()
        os.makedirs("fig/RelativePhase/dyad/each/", exist_ok=True)
        if type == 'position' or type == 'force':
            plt.savefig("fig/RelativePhase/dyad/each/RelativePhase_" + type + "_" + source + ".png")
        else:
            plt.savefig("fig/RelativePhase/dyad/each/RelativePhase_" + type + ".png")

        plt.show()

    def relative_phase_performance(self, type: str, sigma: int = 'none', dec=1, graph=False):
        rela_phase = self.relative_phase(type=type, sigma=sigma, dec=dec)
        error_period, spend_period = CFO.period_performance_cooperation(self)

        region = np.arange(0, 200, 20)
        freq, edge = hist.frequency(rela_phase, region)

        freq_normal = freq / np.sum(freq, axis=3, keepdims=True) * 100

        performance = [
            error_period,
            spend_period,
        ]
        performance_label = [
            "RMSE",
            "Time"
        ]
        axis_label = [
            'Pitch',
            'Roll'
        ]

        df_ = []
        for i in range(len(performance_label)):  # performance type
            for j in range(len(self.cfo)):  # group
                for k in range(len(axis_label)):  # roll and pitch
                    for l in range(len(freq_normal[0][0][0])):  # phase region
                        for m in range(self.period):  # region
                            df_.append(pd.DataFrame({
                                'Performance': performance[i][j][m],
                                'Relative phase occurrence': freq_normal[j][k][m][l],
                                'Performance types': performance_label[i],
                                'Group': str(j + 1),
                                'Phase region': str(region[l]) + '$^\circ$ - ' + str(region[l] + 20) + '$^\circ$',
                                'Period': str(m + 1),
                                'Axis': axis_label[k]
                            }, index=[0]))

        df = pd.concat([i for i in df_], axis=0)
        df.reset_index(drop=True, inplace=True)

        if graph:
            sns.set(font='Times New Roman', font_scale=0.8)
            sns.set_style('ticks')
            sns.set_context("talk",
                            rc={
                                "axes.linewidth": 0.5,
                                "legend.fancybox": False,
                                'pdf.fonttype': 42,
                                'xtick.direction': 'in',
                                'ytick.major.width': 1.0,
                                'xtick.major.width': 1.0,
                            })

            # FacetGridを作成してグラフを設定
            g = sns.FacetGrid(df, col="Phase region", row="Performance types", hue="Group", height=10, aspect=1,
                              sharey=False, sharex=False)
            g.map(plot_scatter, "Performance", "Relative phase occurrence")

            # プロットのサイズを調整
            # g.fig.set_size_inches(15, 8)

            if self.group_type == "dyad":
                xlims = [(-0.015, 0.015), (-0.1, 0.5)]
                ylims = [(0, 100), (0, 100)]

            if self.group_type == "triad":
                xlims = [(-0.015, 0.015), (-0.1, 0.5)]
                ylims = [(0, 6), (0, 6)]

            if self.group_type == "tetrad":
                xlims = [(-0.015, 0.015), (-0.1, 0.5)]
                ylims = [(0, 8), (0, 8)]

            # # Iterate over the AxesSubplots in the FacetGrid object, and set the y limit individually
            # for num,ax in enumerate(g.axes.flatten()):
            #     # col_var = ax.get_title().split(" ")[-1]  # get the column variable from the title
            #     ax.set_xlim(xlims[num])
            #     ax.set_ylim(ylims[num])

            os.makedirs("fig/RelativePhase-Performance/dyad/", exist_ok=True)
            plt.savefig("fig/RelativePhase-Performance/dyad/" + type + ".png")

            plt.show()

        return df

    def relative_phase_performance_filter(self, type: str, sigma: int = 'none', dec=1, graph=False,
                                          min_freq=0, max_freq=0, step=0):
        rela_phase = self.relative_phase_filter(type=type, sigma=sigma, dec=dec, graph=False,
                                                min_freq=min_freq, max_freq=max_freq, step=step)
        filter_range = np.arange(min_freq, max_freq + step, step)
        region = np.arange(0, 200, 20)
        pe = ParallelExecutor.ParallelExecutor(hist.frequency, max_workers=20, args_num=2, return_num=2, para_args_list=[0])
        freq, edge = pe.parallel(rela_phase, region)
        edge = edge[0][0][0][0]
        # freq, edge = hist.frequency(rela_phase, region)
        # freq = freq.transpose(0, 1, 4, 2, 3)
        # freq = freq.reshape([len(freq), len(freq[0]), len(freq[0][0]), -1])
        # freq = np.sum(freq, axis=3)

        freq_normal = freq / np.sum(freq, axis=4, keepdims=True) * 100
        # print(len(freq_normal))
        # std_error = np.std(freq_normal, axis=1) / np.sqrt(len(freq_normal[0]))
        # print(std_error.shape)

        # freq_ave = np.sum(freq_normal, axis=1) / len(freq_normal[0])

        error_period, spend_period = CFO.period_performance_cooperation(self)
        performance = [
            error_period,
            spend_period,
        ]
        performance_label = [
            "RMSE",
            "Time"
        ]
        axis_label = [
            'Pitch',
            'Roll'
        ]

        df_ = []
        for i in range(len(performance_label)):  # performance type
            for j in range(len(self.cfo)):  # group
                for k in range(len(axis_label)):  # roll and pitch
                    for l in range(len(region) - 1):  # phase region
                        for m in range(self.period):  # region
                            for n in range(len(filter_range) - 1):
                                df_.append(pd.DataFrame({
                                    'Performance': performance[i][j][m],
                                    'Relative phase occurrence': freq_normal[n][j][k][m][l],
                                    'Performance types': performance_label[i],
                                    'Group': str(j + 1),
                                    'Phase region': str(region[l]) + '$^\circ$ - ' + str(region[l] + 20) + '$^\circ$',
                                    'Period': str(m + 1),
                                    'Axis': axis_label[k],
                                    'Filter': str(filter_range[n]) + ' - ' + str(filter_range[n + 1]) + ' Hz'
                                }, index=[0]))

        df = pd.concat([i for i in df_], axis=0)
        df.reset_index(drop=True, inplace=True)

        if graph:

            sns.set(font='Times New Roman', font_scale=0.8)
            sns.set_style('ticks')
            sns.set_context("talk",
                            rc={
                                "axes.linewidth": 0.5,
                                "legend.fancybox": False,
                                'pdf.fonttype': 42,
                                'xtick.direction': 'in',
                                'ytick.major.width': 1.0,
                                'xtick.major.width': 1.0,
                            })

            # FacetGridを作成してグラフを設定
            for per_type in performance_label:
                g = sns.FacetGrid(df[df['Performance types'] == per_type], col="Phase region", row="Filter", hue="Group", height=10, aspect=1,
                                  sharey=False, sharex=False)
                g.map(plot_scatter, "Performance", "Relative phase occurrence")

                os.makedirs("fig/RelativePhase-Performance/dyad/filter/", exist_ok=True)
                g.savefig("fig/RelativePhase-Performance/dyad/filter/" + type + '_' + per_type +".png")

            # plt.show()

        return df

    def relative_phase_performance_reg_model(self, type: str, sigma: int = 'none', dec=1, graph=False):
        rela_phase = self.relative_phase(type=type, sigma=sigma, graph=False)
        region = np.arange(0, 200, 20)
        freq, edge = hist.frequency(rela_phase, region)
        freq_normal = freq / np.sum(freq, axis=3, keepdims=True) * 100
        print(freq_normal.shape)

        error_period, spend_period = CFO.period_performance_cooperation(self)

        axis = ['Pitch', 'Roll']
        all_label = []
        label = []
        for i in range(len(axis)):
            label.append([])
            for j in range(len(region) - 1):
                label[i].append(str(region[j]) + '-' + str(region[j + 1]) + '_' + axis[i])
                all_label.append(str(region[j]) + '-' + str(region[j + 1]) + '_' + axis[i])

        df_ = []
        for i in range(len(self.cfo)):
            for j in range(self.period):
                for k in range(len(axis)):
                    for l in range(len(region) - 1):
                        df_.append(pd.DataFrame({
                            'RMSE': error_period[i][j],
                            'Time': spend_period[i][j],
                            'rela': freq_normal[i][k][j][l],
                            'label': label[k][l],
                        }, index=[0]))

        df = pd.concat([i for i in df_], axis=0)
        df.reset_index(drop=True, inplace=True)

        df = df.pivot(index=['RMSE', 'Time'], columns='label', values='rela')
        df.reset_index(inplace=True)
        print(df)


        # 目的変数(Y)
        Y = np.array(df['RMSE'])

        # 全要素が1.0の列を説明変数の先頭に追加(おまじない)
        X = sm.add_constant(df[all_label])
        X = np.array(X)

        # モデルの設定(OLS:最小二乗法を指定)
        model = sm.OLS(Y, X)

        # 回帰分析の実行
        results = model.fit()

        # 結果の詳細を表示
        print(results.summary())

        Y = np.array(df['Time'])
        model = sm.OLS(Y, X)
        results = model.fit()
        print(results.summary())

    def relative_phase_performance_reg_model_filter(self, type: str, sigma: int = 'none', dec=1, graph=False,
                                                    min_freq=0, max_freq=0, step=0):
        rela_phase = self.relative_phase_filter(type=type, sigma=sigma, dec=dec, graph=False,
                                                min_freq=min_freq, max_freq=max_freq, step=step)
        filter_range = np.arange(min_freq, max_freq + step, step)
        region = np.arange(0, 200, 20)
        pe = ParallelExecutor.ParallelExecutor(hist.frequency, max_workers=20, args_num=2, return_num=2, para_args_list=[0])
        freq, edge = pe.parallel(rela_phase, region)
        edge = edge[0][0][0][0]
        # freq, edge = hist.frequency(rela_phase, region)
        # freq = freq.transpose(0, 1, 4, 2, 3)
        # freq = freq.reshape([len(freq), len(freq[0]), len(freq[0][0]), -1])
        # freq = np.sum(freq, axis=3)

        freq_normal = freq / np.sum(freq, axis=4, keepdims=True) * 100

        error_period, spend_period = CFO.period_performance_cooperation(self)

        axis = ['Pitch', 'Roll']
        all_label = []
        label = []
        for i in range(len(axis)):
            label.append([])
            for j in range(len(filter_range) - 1):
                label[i].append([])
                for k in range(len(region) - 1):
                    label_ = str(filter_range[j]) + '-' + str(filter_range[j + 1]) + '_' + str(region[k]) + '-' + str(region[k + 1]) + '_' + axis[i]
                    label[i][j].append(label_)
                    all_label.append(label_)

        df_ = []
        for i in range(len(self.cfo)):
            for j in range(self.period):
                for k in range(len(axis)):
                    for m in range(len(filter_range) - 1):
                        for l in range(len(region) - 1):
                            df_.append(pd.DataFrame({
                                'RMSE': error_period[i][j],
                                'Time': spend_period[i][j],
                                'rela': freq_normal[m][i][k][j][l],
                                'label': label[k][m][l],
                            }, index=[0]))

        df = pd.concat([i for i in df_], axis=0)
        df.reset_index(drop=True, inplace=True)

        df = df.pivot(index=['RMSE', 'Time'], columns='label', values='rela')
        df.reset_index(inplace=True)
        print(df)


        # 目的変数(Y)
        Y = np.array(df['RMSE'])

        # 全要素が1.0の列を説明変数の先頭に追加(おまじない)
        X = sm.add_constant(df[all_label])
        X = np.array(X)

        # モデルの設定(OLS:最小二乗法を指定)
        model = sm.OLS(Y, X)

        # 回帰分析の実行
        results = model.fit()

        # 結果の詳細を表示
        print(results.summary())

        Y = np.array(df['Time'])
        model = sm.OLS(Y, X)
        results = model.fit()
        print(results.summary())


    def show_coherence_stft(self):
        # ウィンドウ幅，STFTを施す数の設定
        t_wndw = 1000.0e-3  # 100 milisecond
        n_stft = 1000  # number of STFT
        freq_upper = 50.0e0  # 表示する周波数の上限

        time = self.cfo[0]['time'][self.start_num:self.end_num:100]
        fcfo = self.get_fcfo(dec=100)

        fcfo = fcfo.transpose(1, 0, 2, 3)

        tm_sp, freq_sp, sp = STFT.stft(fcfo, time, t_wndw=t_wndw, n_stft=n_stft, wndw='hanning')

        xsp = STFT.cross_spectrogram(sp[0], sp[1])

        coh, phs = STFT.coherence(xsp, sp[0], sp[1])

        fig, ax_ = plt.subplots(2, 10, figsize=(20, 10), sharex='none', sharey='row')

        cmap = mpl.cm.jet


        norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        for i in range(2):
            for j in range(10):
                ax = ax_[i, j]
                ax.set_ylim(0, freq_upper)
                # ax.set_ylabel('frequency\n(Hz)')
                starttime = self.starttime + 3.0 * (i * 10 + j)
                ax.set_xlim(starttime, starttime + 3)
                ax.set_xticks([starttime, starttime + 3])
                # ax.set_xlabel('')
                # ax.tick_params(labelbottom=False)
                # ax.set_ylim(0, freq_upper)
                # ax.set_ylabel('frequency\n(Hz)')

                ax.contourf(tm_sp[0][0][0], freq_sp[0][0][0], coh[0][0],
                            norm=norm, levels=10, cmap=cmap)

                # ax.text(0.99, 0.97, "coherence", color='white', ha='right', va='top',
                #             path_effects=[path_effects.Stroke(linewidth=2, foreground='black'),
                #                           path_effects.Normal()],
                #             transform=ax.transAxes)

        cb_coh = fig.add_axes([0.92, 0.10, 0.02, 0.70])
        mpl.colorbar.ColorbarBase(ax=cb_coh, cmap=cmap, norm=norm,
                                  # boundaries=np.linspace(0, 1, 11),
                                  orientation="vertical",
                                  label='coherence',
                                  drawedges='False',
                                  )

        plt.show()

    def calc_STFT(self):
        # ウィンドウ幅，STFTを施す数の設定
        t_wndw = 100.0e-3  # 100 milisecond
        n_stft = 256  # number of STFT
        freq_upper = 10.0e0  # 表示する周波数の上限

        sig1 = self.cfo[0]['i1_p_fcfo']
        sig2 = self.cfo[0]['i2_p_fcfo']
        time = self.cfo[0]['time']

        # STFTを実行
        tm_sp1, freq_sp1, sp1 = STFT.stft(sig1, time, t_wndw=t_wndw, n_stft=n_stft, wndw='hanning')
        tm_sp2, freq_sp2, sp2 = STFT.stft(sig2, time, t_wndw=t_wndw, n_stft=n_stft, wndw='hanning')
        # クロススペクトル
        xsp = STFT.cross_spectrogram(sp1, sp2)
        # コヒーレンスとフェイズ
        coh, phs = STFT.coherence(xsp, sp1, sp2)

        # 結果のプロット
        # 解析結果の可視化
        figsize = (210 / 25.4, 294 / 25.4)
        dpi = 200
        fig = plt.figure(figsize=figsize, dpi=dpi)

        # 図の設定 (全体)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['xtick.top'] = True
        plt.rcParams['xtick.major.size'] = 6
        plt.rcParams['xtick.minor.size'] = 3
        plt.rcParams['xtick.minor.visible'] = True
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['ytick.right'] = True
        plt.rcParams['ytick.major.size'] = 6
        plt.rcParams['ytick.minor.size'] = 3
        plt.rcParams['ytick.minor.visible'] = True
        plt.rcParams["font.size"] = 14
        plt.rcParams['font.family'] = 'Helvetica'

        # 窓関数幅をプロット上部に記載
        fig.text(0.10, 0.95, f't_wndw = {t_wndw} s')

        # プロット枠の設定
        ax01 = fig.add_axes([0.125, 0.79, 0.70, 0.08])
        ax02 = fig.add_axes([0.125, 0.59, 0.70, 0.08])

        ax_sp1 = fig.add_axes([0.125, 0.68, 0.70, 0.10])
        cb_sp1 = fig.add_axes([0.85, 0.68, 0.02, 0.10])
        ax_sp2 = fig.add_axes([0.125, 0.48, 0.70, 0.10])
        cb_sp2 = fig.add_axes([0.85, 0.48, 0.02, 0.10])

        ax_xsp = fig.add_axes([0.125, 0.33, 0.70, 0.10])
        cb_xsp = fig.add_axes([0.85, 0.33, 0.02, 0.10])
        ax_coh = fig.add_axes([0.125, 0.22, 0.70, 0.10])
        cb_coh = fig.add_axes([0.85, 0.22, 0.02, 0.10])
        ax_phs = fig.add_axes([0.125, 0.10, 0.70, 0.10])
        cb_phs = fig.add_axes([0.85, 0.10, 0.02, 0.10])

        # ---------------------------
        # テスト信号 sig1 のプロット
        ax01.set_xlim(self.starttime, self.endtime)
        ax01.set_xlabel('')
        ax01.tick_params(labelbottom=False)
        ax01.set_ylabel('x (sig1)')

        ax01.plot(time, sig1, c='black')

        # ---------------------------
        # テスト信号 sig2 のプロット
        ax02.set_xlim(self.starttime, self.endtime)
        ax02.set_xlabel('')
        ax02.tick_params(labelbottom=False)
        ax02.set_ylabel('y (sig2)')

        ax02.plot(time, sig2, c='black')

        # ---------------------------
        # テスト信号 sig1 のスペクトログラムのプロット
        ax_sp1.set_xlim(self.starttime, self.endtime)
        ax_sp1.set_xlabel('')
        ax_sp1.tick_params(labelbottom=False)
        ax_sp1.set_ylim(0, freq_upper)
        ax_sp1.set_ylabel('frequency\n(Hz)')

        norm = mpl.colors.Normalize(vmin=np.log10(np.abs(sp1[freq_sp1 < freq_upper, :]) ** 2).min(),
                                    vmax=np.log10(np.abs(sp1[freq_sp1 < freq_upper, :]) ** 2).max())
        cmap = mpl.cm.jet

        ax_sp1.contourf(tm_sp1, freq_sp1, np.log10(np.abs(sp1) ** 2),
                        norm=norm, levels=256, cmap=cmap)

        ax_sp1.text(0.99, 0.97, "spectrogram", color='white', ha='right', va='top',
                    path_effects=[path_effects.Stroke(linewidth=2, foreground='black'),
                                  path_effects.Normal()],
                    transform=ax_sp1.transAxes)

        mpl.colorbar.ColorbarBase(cb_sp1, cmap=cmap, norm=norm,
                                  orientation="vertical",
                                  label='$\log_{10}|X/N|^2$')

        # ---------------------------
        # テスト信号 sig2 のスペクトログラムのプロット
        ax_sp2.set_xlim(self.starttime, self.endtime)
        ax_sp2.set_xlabel('')
        ax_sp2.tick_params(labelbottom=True)
        ax_sp2.set_ylim(0, freq_upper)
        ax_sp2.set_ylabel('frequency\n(Hz)')

        norm = mpl.colors.Normalize(vmin=np.log10(np.abs(sp2[freq_sp2 < freq_upper, :]) ** 2).min(),
                                    vmax=np.log10(np.abs(sp2[freq_sp2 < freq_upper, :]) ** 2).max())
        ax_sp2.contourf(tm_sp2, freq_sp2, np.log10(np.abs(sp2) ** 2),
                        norm=norm, levels=256, cmap=cmap)

        ax_sp2.text(0.99, 0.97, "spectrogram", color='white', ha='right', va='top',
                    path_effects=[path_effects.Stroke(linewidth=2, foreground='black'),
                                  path_effects.Normal()],
                    transform=ax_sp2.transAxes)

        mpl.colorbar.ColorbarBase(cb_sp2, cmap=cmap, norm=norm,
                                  orientation="vertical",
                                  label='$\log_{10}|Y/N|^2$')

        # ---------------------------
        # テスト信号 sig1 と sig2 のクロススペクトルのプロット
        ax_xsp.set_xlim(self.starttime, self.endtime)
        ax_xsp.set_xlabel('')
        ax_xsp.tick_params(labelbottom=False)
        ax_xsp.set_ylim(0, freq_upper)
        ax_xsp.set_ylabel('frequency\n(Hz)')

        norm = mpl.colors.Normalize(vmin=np.log10(np.abs(xsp[freq_sp2 < freq_upper, :])).min(),
                                    vmax=np.log10(np.abs(xsp[freq_sp2 < freq_upper, :])).max())

        ax_xsp.contourf(tm_sp1, freq_sp1, np.log10(np.abs(xsp)),
                        norm=norm, levels=256, cmap=cmap)

        ax_xsp.text(0.99, 0.97, "cross-spectrum", color='white', ha='right', va='top',
                    path_effects=[path_effects.Stroke(linewidth=2, foreground='black'),
                                  path_effects.Normal()],
                    transform=ax_xsp.transAxes)

        mpl.colorbar.ColorbarBase(cb_xsp, cmap=cmap, norm=norm,
                                  orientation="vertical",
                                  label='$\log_{10}|XY^*/N^2|$')

        # ---------------------------
        # テスト信号 sig1 と sig2 のコヒーレンスのプロット
        ax_coh.set_xlim(self.starttime, self.endtime)
        ax_coh.set_xlabel('')
        ax_coh.tick_params(labelbottom=False)
        ax_coh.set_ylim(0, freq_upper)
        ax_coh.set_ylabel('frequency\n(Hz)')

        norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        ax_coh.contourf(tm_sp1, freq_sp1, coh,
                        norm=norm, levels=10, cmap=cmap)

        ax_coh.text(0.99, 0.97, "coherence", color='white', ha='right', va='top',
                    path_effects=[path_effects.Stroke(linewidth=2, foreground='black'),
                                  path_effects.Normal()],
                    transform=ax_coh.transAxes)

        mpl.colorbar.ColorbarBase(cb_coh, cmap=cmap, norm=norm,
                                  boundaries=np.linspace(0, 1, 11),
                                  orientation="vertical",
                                  label='coherence')

        # ---------------------------
        # テスト信号 sig1 と sig2 のフェイズのプロット
        ax_phs.set_xlim(self.starttime, self.endtime)
        ax_phs.set_xlabel('time (s)')
        ax_phs.tick_params(labelbottom=True)
        ax_phs.set_ylim(0, freq_upper)
        ax_phs.set_ylabel('frequency\n(Hz)')

        norm = mpl.colors.Normalize(vmin=-180.0, vmax=180.0)
        cmap = mpl.cm.hsv
        ax_phs.contourf(tm_sp1, freq_sp1, np.where(coh >= 0.75, phs, None),
                        norm=norm, levels=16, cmap=cmap)

        ax_phs.text(0.99, 0.97, "phase", color='white', ha='right', va='top',
                    path_effects=[path_effects.Stroke(linewidth=2, foreground='black'),
                                  path_effects.Normal()],
                    transform=ax_phs.transAxes)

        mpl.colorbar.ColorbarBase(cb_phs, cmap=cmap,
                                  norm=norm,
                                  boundaries=np.linspace(-180.0, 180.0, 17),
                                  orientation="vertical",
                                  label='phase (deg)')
        plt.show()

        # # plt.figure(figsize=(10, 5))
        # coh_ave = np.average(coh, axis=1)
        # f, Cxy = signal.coherence(x=sig1, y=sig2, fs=1.0/dt, nperseg=2000)
        #
        # plt.figure(figsize=(10, 5))
        # plt.plot(freq_sp1, coh_ave, label='STFT')
        # # print(freq_sp1)
        # plt.plot(f, Cxy, label='scipy')
        # plt.legend()
        # plt.show()
