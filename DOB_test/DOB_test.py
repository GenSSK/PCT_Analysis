# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import control
import matplotlib as plt
import pandas as pd
from scipy import signal, optimize
import matplotlib.pyplot as plt


class DOB:
    def npz_load(self):
        # self.data = np.load('w_filter_w_dob_mass.npz')
        self.data = np.load('50%_test_w_DOB_if2.npz')

    def Arrangement(self):
        Ta = 0.0001  # データのサンプリング時間[sec]
        Ts = 0.0001  # 同定用のサンプリング時間[sec]
        Tstr = 10 # 同定を開始する時間[sec]
        Texp = 10  # 同定に必要な時間[sec]
        No = int(Texp / Ta)  # 同定に必要なデータの個数
        Decimation = int(Ts / Ta)  # 間引きの数
        str = int(Tstr / Ts)

        ExtractData_pitch = np.array([self.data['time'],
                                      self.data['p_iq'],
                                      self.data['p_amref'],
                                      self.data['p_am'],
                                      self.data['p_wmref'],
                                      self.data['p_wm'],
                                      self.data['p_thmref'],
                                      self.data['p_thm']
                                      ])  # 同定に必要なデータの抽出

        ExtractData_roll = np.array([self.data['time'],
                                     self.data['r_iq'],
                                     self.data['r_amref'],
                                     self.data['r_am'],
                                     self.data['r_wmref'],
                                     self.data['r_wm'],
                                     self.data['r_thmref'],
                                     self.data['r_thm']
                                     ])  # 同定に必要なデータの抽出

        DecimData_pitch = ExtractData_pitch[:, str:No + str:Decimation]  # 同定用データの作成
        DecimData_roll = ExtractData_roll[:, str:No + str:Decimation]  # 同定用データの作成

        self.TestData = np.stack(((DecimData_pitch, DecimData_roll)))

    def graph_sub(self, num):
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

        fig, (top, mid, bot) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        # plt.xlim([0, 2])  # x軸の範囲
        # plt.xlim([0.28, 0.89])  # x軸の範囲
        plt.xlabel("Time[sec]")

        top.plot(self.TestData[num][0], self.TestData[num][3], label = 'res')
        top.plot(self.TestData[num][0], self.TestData[num][2], label = 'ref')
        top.set_ylabel('Acceleration(rad/s^2)')
        top.legend()
        # top.set_yticks(np.arange(-2, 2, 0.5))
        top.set_ylim([-10, 10])  # y軸の範囲

        mid.plot(self.TestData[num][0], self.TestData[num][5], label = 'res')
        mid.plot(self.TestData[num][0], self.TestData[num][4], label = 'ref')
        mid.set_ylabel('Velocity(rad/s)')
        mid.legend()
        # mid.set_yticks(np.arange(-400, -100, 50))
        mid.set_ylim([-2.0, 2.0])  # y軸の範囲

        bot.plot(self.TestData[num][0], self.TestData[num][7], label = 'res')
        bot.plot(self.TestData[num][0], self.TestData[num][6], label = 'ref')
        bot.set_ylabel('Position(rad)')
        bot.legend()
        # bot.set_yticks(np.arange(-200, 200, 50))
        # bot.set_ylim([-150.0, 150.0])  # y軸の範囲

        plt.tight_layout()
        plt.savefig("50%_test_w_DOB_if2.png")
        # plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    DB = DOB()
    DB.npz_load()
    DB.Arrangement()
    DB.graph_sub(0)