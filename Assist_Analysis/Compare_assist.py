import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
import CFO_analysis
from scipy.spatial.distance import correlation
import scipy.stats
from scipy.stats import beta
import statsmodels.api as sm
from numpy.random import Generator, PCG64, MT19937
from scipy import signal, optimize
from patsy import dmatrices
import itertools

import sys

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


def fit_func_mse(parameter, *args):
    x, y = args
    a = parameter[0]
    b = parameter[1]
    residual = y - (a * x + b)
    mse = np.mean(residual ** 2)
    return mse

def fit_func(parameter, *args):
    x, y = args
    a = parameter[0]
    b = parameter[1]
    residual = y - (a * x + b)
    return residual

def fit_func_2nd(parameter, *args):
    accel, velo, y = args
    a = parameter[0]
    b = parameter[1]
    residual = y - (a * accel + b * velo)
    return residual

def fit_func_2nd_mse(parameter, *args):
    accel, velo, y = args
    a = parameter[0]
    b = parameter[1]
    residual = y - (a * accel + b * velo)
    mse = np.mean(residual ** 2)
    return mse

class compare:
    def __init__(self, without_assist_npz, assist_npz, group_type, trajectory_type):
        self.woa: CFO_analysis.CFO = CFO_analysis.CFO(without_assist_npz, group_type=group_type, trajectory_type=trajectory_type)
        self.wa: CFO_analysis.CFO = CFO_analysis.CFO(assist_npz, group_type=group_type, trajectory_type=trajectory_type)

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
        plt.rcParams['font.size'] = 5  # フォントの大きさ
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

    def plot_performance_improve(self):
        performance = np.stack([
            self.woa.get_performance(),
            self.wa.get_performance()
        ])

        condition = ['w/o Assist', 'w/ Assist']

        print(performance.shape)
        performance = performance.reshape(len(condition), len(performance[0]), -1, self.wa.num)
        print(performance.shape)
        performance = np.average(performance, axis=3)
        print(performance.shape)
        init_performance = performance[:, :, 0]
        performance_norm = performance / init_performance[:, :, np.newaxis]
        performance_norm_reg = performance_norm.reshape(len(condition), -1)
        print(performance_norm.shape)

        performance_norm_mean = np.average(performance_norm, axis=1)
        print(performance_norm_mean.shape)

        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=200, sharex=True)
        period = np.arange(1, len(performance[0][0]) + 1)
        period_reg = np.tile(period, len(performance_norm[0]))
        states_palette = sns.color_palette("YlGnBu_r", n_colors=4)
        lines = []
        for i in range(len(performance_norm_mean)):
            sns.regplot(x=period_reg, y=performance_norm_reg[i], scatter=True, ax=ax, line_kws={'linewidth': 2.0},
                        order=5, color=states_palette[i], scatter_kws={'s': 10, 'alpha': 0.5})

            line = Line2D([0], [0], color=states_palette[i], lw=2, label=condition[i])
            lines.append(line)

        ax.set_ylabel('Normalized Performance')
        ax.set_xlabel('Period')
        # ax.set_ylim([0.5, 1.0])
        ax.set_xticks(np.arange(1, len(performance[0][0]) + 1, 1))
        ax.legend(handles=lines, loc='lower left', ncol=4, framealpha=0.0,)

        os.makedirs('fig/Compare/Performance', exist_ok=True)
        plt.savefig('fig/Compare/Performance/Performance_improvement.pdf')

        plt.show()
