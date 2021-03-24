
import numpy as np
import control
import matplotlib as plt
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt



class PreIden:
    def npz_load(self):
        self.npz = np.load('balanced_roll_step.npz')
        self.pitch = np.load('balanced_pitch_step.npz')
        # self.npz = np.load('new_roll_step.npz')

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

        plt.plot(self.npz['time'], self.npz['r_wm'], label = 'roll')
        plt.plot(self.pitch['time'], self.pitch['p_wm'], label = 'pitch')
        # plt.plot(self.npz['time'], self.npz['iq'])
        # plt.plot(self.data['Time'], self.data['wm'])
        # plt.xticks(np.arange(2.4, 2.6, 0.005))
        plt.xlim(0, 7)  # x軸の範囲
        plt.ylim(0, 100)  # x軸の範囲
        plt.xlabel('Time[sec]')
        plt.ylabel('Velocity[rad/s]')
        plt.legend()
        plt.show()
        # plt.savefig("balanced_step.png", format="png", dpi=300)

    def pf(self):
        a, b = np.polyfit(self.npz['time'][0:10000], self.npz['r_wm'][0:10000], 1)
        print(a, b)
        a, b = np.polyfit(self.pitch['time'][0:10000], self.pitch['p_wm'][0:10000], 1)
        print(a, b)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ID = PreIden()
    ID.npz_load()
    # ID.graph()
    ID.pf()