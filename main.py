# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt


class read:
    def print_hi(self, name):
        # Use a breakpoint in the code line below to debug your script. aaaa
        print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

    def read_csv(self):
        self.data = pd.read_csv('DATA.csv')

    def npz_load(self):
        # self.npz = np.load('doutei.npz')
        # self.npz = np.load('doutei2.npz')
        # self.npz = np.load('doutei_n8.npz')
        # self.npz = np.load('doutei_n8_v2.npz')
        self.npz = np.load('doutei_n10.npz')
        # self.npz = np.load('pre_doutei.npz')
        # self.npz = np.load('ExpData.npz')
        # print(self.npz["time"])
        # print(self.npz["wm"])
        self.thm_dtre = signal.detrend(self.npz["thm"], type = 'constant')
        self.thm_dtre = signal.detrend(self.thm_dtre, type = 'linear')
        self.wm_dtre = signal.detrend(self.npz["wm"], type='constant')
        self.wm_dtre = signal.detrend(self.wm_dtre, type='linear')

    def graph(self):
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

        plt.figure(figsize=(20, 10), dpi=100)

        # plt.scatter(self.data['Time'], self.data['wm'])
        # plt.scatter(self.npz['time'], self.npz['wm'])
        # plt.plot(self.npz['time'], self.npz['wm'])
        plt.plot(self.npz['time'], self.npz['iq'])
        # plt.plot(self.data['Time'], self.data['wm'])
        # plt.xticks(np.arange(2.4, 2.6, 0.005))
        plt.xlim(2.5, 2.6)  # x軸の範囲
        plt.xlabel('Time[sec]')
        plt.ylabel('Velocity[rad/s]')
        plt.show()
        # plt.savefig("step_response2.png", format="png", dpi=300)

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

        fig, (top, mid, bot) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        plt.xlim([0, 1.530])  # x軸の範囲
        # plt.xlim([0.28, 0.89])  # x軸の範囲
        plt.xlabel("Time[sec]")

        top.plot(self.npz['time'], self.npz['iq'])
        top.set_ylabel('Current[%]')
        top.set_yticks(np.arange(-2, 2, 0.5))
        top.set_ylim([-1.5, 1.5])  # y軸の範囲

        mid.plot(self.npz['time'], self.npz['thm'], label = 'Raw')
        mid.plot(self.npz['time'], self.thm_dtre, label = 'Fix')
        mid.plot(self.npz['time'], self.npz['thm'] - self.thm_dtre, label = 'Bias & Trend')
        mid.set_ylabel('Position[rad]')
        mid.legend()
        # mid.set_yticks(np.arange(-400, -100, 50))
        # mid.set_ylim([-300.0, -200.0])  # y軸の範囲

        bot.plot(self.npz['time'], self.npz['wm'], label = 'Raw')
        bot.plot(self.npz['time'], self.wm_dtre, label = 'Fix')
        bot.plot(self.npz['time'], self.npz['wm'] - self.wm_dtre, label = 'Bias & Trend')
        bot.set_ylabel('Velocity[rad/s]')
        bot.legend()
        # bot.set_yticks(np.arange(-200, 200, 50))
        # bot.set_ylim([-150.0, 150.0])  # y軸の範囲

        plt.tight_layout()
        # plt.savefig("doutei_enl.png")
        plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rd = read()
    # rd.print_hi('PyCharm')
    # rd.read_csv()
    rd.npz_load()
    # rd.graph()
    rd.graph_sub()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
