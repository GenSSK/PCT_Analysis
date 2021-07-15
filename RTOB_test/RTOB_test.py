# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import control
import matplotlib as plt
import pandas as pd
from scipy import signal, optimize
import matplotlib.pyplot as plt


class Iden:
    def npz_load(self):
        self.data = np.load('50%_test_RTOB_if2.npz')

    def graph_sub(self):
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
        plt.rcParams['axes.linewidth'] = 1.0  # 軸の線幅edge linewidth。囲みの太さ

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

        fig, (top, bot) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        plt.xlim([10, 20])  # x軸の範囲
        # plt.xlim([0.28, 0.89])  # x軸の範囲
        plt.xlabel("Time[sec]")

        top.plot(self.data['time'], self.data['p_thmref'], label = 'Estimation')
        top.plot(self.data['time'], self.data['p_thm'], label = 'Actuality')
        top.plot(self.data['time'], self.data['r_thm'], label = 'Actuality')
        top.set_ylabel('Position [rad]')
        top.legend()
        top.set_yticks(np.arange(-10, 10, 0.5))
        top.set_ylim([-1, 1])  # y軸の範囲

        bot.plot(self.data['time'], self.data['r_tdish'], label='Estimation')
        bot.plot(self.data['time'], self.data['p_tdish'], label='Actuality')
        bot.set_ylabel('Reaction torque[Nm]')
        bot.legend()
        bot.set_yticks(np.arange(-10, 10, 1))
        bot.set_ylim([-3, 3])  # y軸の範囲

        plt.tight_layout()
        plt.savefig("50%_test_RTOB_if2.png")
        # plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ID = Iden()
    ID.npz_load()
    ID.graph_sub()