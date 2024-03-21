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

class CFO_compare:
    def __init__(self, cfo_data_ind, cfo_data_shd, group_type, file_names):
        self.group_type = group_type
        self.file_names = file_names
        self.cfo_ind = cfo_data_ind
        self.cfo_shd = cfo_data_shd

        self.smp = 0.0001  # サンプリング時間
        self.duringtime = self.cfo_ind[0]['duringtime'][0]  # ターゲットの移動時間
        self.starttime = self.cfo_ind[0]['starttime'][0]  # タスク開始時間
        self.endtime = self.cfo_ind[0]['endtime'][0]  # タスク終了時間
        self.tasktime = self.endtime - self.starttime  # タスクの時間
        self.period = int(self.tasktime / self.duringtime)  # 回数
        self.num = int(self.duringtime / self.smp)  # 1ピリオドにおけるデータ数
        self.start_num = int((self.starttime - 20.0) / self.smp)
        self.end_num = int((self.endtime - 20.0) / self.smp)
        self.nn_read_flag = False
        self.join = self.cfo_ind[0]['join'][0]
        self.tasktype = ''.join(chr(char) for char in self.cfo_ind[0]['tasktype'])
        self.controltype = ''.join(chr(char) for char in self.cfo_ind[0]['controltype'])
        # print(self.join)

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

    def show_prediction(self):
        variable_label = ['thm', 'text']
        rp_label = ['_p', '_r']
        # variable_option_label = ['', '_pre', '_pre_solo']
        variable_option_label = ['', '_pre']
        lws = [2.0, 1.0]
        lts = ['-', '--']

        ylims = [[-1.5, 1.5], [-6.0, 6.0]]
        yticks = [np.arange(-10, 10, 0.5), np.arange(-10, 10, 2.0)]

        ylabels = [['Pitch angle (rad)', 'Roll angle (rad)',],
                   ['Pitch force (Nm)', 'Roll force (Nm)']]

        for i in range(len(self.cfo_ind)):
            data_ind = self.cfo_ind[i]
            data_shd = self.cfo_shd[i]
            fig, ax = plt.subplots(3, 2, figsize=(15, 10), dpi=200, sharex=True)

            plt.xticks(np.arange(self.starttime, self.endtime * 2, self.duringtime * 2))
            plt.xlim([self.starttime, self.endtime])  # x軸の範囲
            # plt.xlim([40.0, 60.0])  # x軸の範囲
            ax[0, 1].set_xlabel("Time (sec)")
            ax[1, 1].set_xlabel("Time (sec)")

            # thm
            for j, axis in enumerate(rp_label):
                ax[0, j].set_ylabel(ylabels[0][j])
                ax[0, j].set_yticks(yticks[0])
                ax[0, j].set_ylim(ylims[0][0], ylims[0][1])
                for k in range(self.join):
                    i_num = 'i' + str(k + 1)
                    ax[0, j].plot(data_ind['time'][self.start_num:self.end_num:10],
                                  data_ind[i_num + axis + '_thm'][self.start_num:self.end_num:10],
                                  '-', lw=2.0,
                                  label='P' + str(k + 1))

                thm_sum = np.zeros(len(data_ind['time'][self.start_num:self.end_num:10]))
                for k in range(self.join):
                    i_num = 'i' + str(k + 1)
                    ax[0, j].plot(data_ind['time'][self.start_num:self.end_num:10],
                                  data_ind[i_num + axis + '_thm_pre'][self.start_num:self.end_num:10],
                                  '--', lw=1.0,
                                  label='Ind Model' + str(k + 1))

                    thm_sum = np.sum([thm_sum, data_ind[i_num + axis + '_thm_pre'][self.start_num:self.end_num:10]], axis=0)

                # ax[0, j].plot(data_ind['time'][self.start_num:self.end_num:10],
                #               thm_sum,
                #               '--', lw=1.0,
                #               label='Ind Model total')
                ax[0, j].plot(data_shd['time'][self.start_num:self.end_num:10],
                              data_shd['i1' + axis + '_thm_pre'][self.start_num:self.end_num:10],
                              ':', lw=1.0,
                              label='Shd Model')
                ax[0, j].legend(ncol=self.join, columnspacing=1, loc='upper left')

            # text
            for j, axis in enumerate(rp_label):
                ax[1, j].set_ylabel(ylabels[1][j])
                ax[1, j].set_yticks(yticks[1])
                ax[1, j].set_ylim(ylims[1][0], ylims[1][1])
                text_sum = np.zeros(len(data_shd['time'][self.start_num:self.end_num:10]))
                for k in range(self.join):
                    i_num = 'i' + str(k + 1)
                    ax[1, j].plot(data_ind['time'][self.start_num:self.end_num:10],
                                  data_ind[i_num + axis + '_text'][self.start_num:self.end_num:10],
                                  '-', lw=1.0,
                                  label='P' + str(k + 1))

                    text_sum = np.sum([text_sum, data_ind[i_num + axis + '_text'][self.start_num:self.end_num:10]], axis=0)

                ax[1, j].plot(data_shd['time'][self.start_num:self.end_num:10],
                              text_sum,
                              '-', lw=2.0,
                              label='Coop total')


                for k in range(self.join):
                    i_num = 'i' + str(k + 1)
                    ax[1, j].plot(data_ind['time'][self.start_num:self.end_num:10],
                                  data_ind[i_num + axis + '_text_pre'][self.start_num:self.end_num:10],
                                  '--', lw=1.0,
                                  label='Ind Model' + str(k + 1))

                text_sum = np.zeros(len(data_shd['time'][self.start_num:self.end_num:10]))
                for k in range(self.join):
                    i_num = 'i' + str(k + 1)
                    ax[1, j].plot(data_shd['time'][self.start_num:self.end_num:10],
                                  data_shd[i_num + axis + '_text_pre'][self.start_num:self.end_num:10],
                                  ':', lw=0.5,
                                  label='Shd Model' + str(k + 1))

                    text_sum = np.sum([text_sum, data_shd[i_num + axis + '_text_pre'][self.start_num:self.end_num:10]], axis=0)

                ax[1, j].plot(data_shd['time'][self.start_num:self.end_num:10],
                              text_sum,
                              ':', lw=1.0,
                              label='Shd Model total')


                ax[1, j].legend(ncol=self.join, columnspacing=1, loc='upper left')

            # ball
            axis_label = ['x', 'y']
            for j, axis in enumerate(axis_label):
                ax[2, j].set_ylabel(axis + "-axis Position (m)")
                # ax[2, j].set_yticks(np.arange(-0.2, 0.2, 0.05))
                ax[2, j].set_ylim([-0.5, 0.5])  # y軸の範囲
                ax[2, j].plot(data_ind['time'][self.start_num:self.end_num:10],
                              data_ind['target' + axis][self.start_num:self.end_num:10],
                              '-', lw=2.0,
                              label='Target')
                ax[2, j].plot(data_ind['time'][self.start_num:self.end_num:10],
                              data_ind['ball' + axis][self.start_num:self.end_num:10],
                              '-', lw=2.0,
                              label='H-H')
                for k in range(self.join):
                    interfacenum = 'i' + str(k + 1)
                    ax[2, j].plot(data_ind['time'][self.start_num:self.end_num:10],
                                  data_ind[interfacenum + '_ball' + axis + '_pre'][self.start_num:self.end_num:10],
                                  '--', lw=1.0,
                                  label='Ind Model' + str(k + 1))
                ax[2, j].plot(data_shd['time'][self.start_num:self.end_num:10],
                              data_shd[interfacenum + '_ball' + axis + '_pre'][self.start_num:self.end_num:10],
                              ':', lw=1.0,
                              label='Shd Model')
                ax[2, j].legend(ncol=self.join, columnspacing=1, loc='upper left')

            plt.tight_layout()
            # os.makedirs('fig/comparison_ind_shd/' + self.group_type, exist_ok=True)
            # plt.savefig('fig/comparison_ind_shd/' + self.group_type + '/' + self.file_names[i][:-4] + '.png')

            # os.makedirs('fig/comparison_ind_shd/' + self.group_type + '/' + 'Expand', exist_ok=True)
            # plt.savefig('fig/comparison_ind_shd/' + self.group_type + '/Expand/' + self.file_names[i][:-4] + '_Expand.png')
        plt.show()