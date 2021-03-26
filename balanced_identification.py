# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import control
import matplotlib as plt
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt


class Iden:
    def npz_load(self):
        self.pitch = np.load('10percent_balanced_pitch_lpf.npz')
        self.roll = np.load('10percent_balanced_roll_lpf.npz')
        self.test = np.load('10percent_balanced_test_fix.npz')

    def Arrangement(self):
        Ta = 0.0001  # データのサンプリング時間[sec]
        Ts = 0.01  # 同定用のサンプリング時間[sec]
        Texp = 40  # 同定に必要な時間[sec]
        Tstr = 20 # 同定を開始する時間[sec]
        No = int(Texp / Ta)  # 同定に必要なデータの個数
        Decimation = int(Ts / Ta)  # 間引きの数
        str = int(Tstr / Ts)

        ExtractData_pitch = np.array([self.pitch['time'],
                                      self.pitch['p_iq'],
                                      self.pitch['p_am'],
                                      self.pitch['p_wm'],
                                      self.pitch['p_thm']
                                      ])  # 同定に必要なデータの抽出

        ExtractData_roll = np.array([self.roll['time'],
                                     self.roll['r_iq'],
                                     self.roll['r_am'],
                                     self.roll['r_wm'],
                                     self.roll['r_thm']
                                     ])  # 同定に必要なデータの抽出

        ExtractData_test = np.array([self.test['time'],
                                     self.test['p_iq'],
                                     self.test['p_am'],
                                     self.test['p_wm'],
                                     self.test['p_thm'],
                                     self.test['r_iq'],
                                     self.test['r_am'],
                                     self.test['r_wm'],
                                     self.test['r_thm']
                                     ])  # 同定に必要なデータの抽出

        DecimData_pitch = ExtractData_pitch[:, str:No:Decimation]  # 同定用データの作成
        DecimData_roll = ExtractData_roll[:, str:No:Decimation]  # 同定用データの作成

        self.IdenData = np.stack(((DecimData_pitch, DecimData_roll)))
        self.TestData = ExtractData_test[:, str:No:Decimation]  # 同定用データの作成

        self.IdenData_detr = np.zeros(((2, 5, len(self.IdenData[0][0]))))
        self.IdenData_detr[0][0] = self.IdenData[0][0]
        self.IdenData_detr[1][0] = self.IdenData[1][0]

        for i in range(2):
            for j in range(4):
                a = signal.detrend(self.IdenData[i][j + 1], type='constant')
                b = signal.detrend(a, type='linear')
                self.IdenData_detr[i][j + 1] = b

    def fit(self, num):
        y1 = self.IdenData_detr[num][2]
        u1 = self.IdenData_detr[num][1]

        # y1 = self.IdenData[num][2]
        # u1 = self.IdenData[num][1]

        a, b = np.polyfit(u1, y1, 1)
        print(a, b)

        x = np.arange(-1, 1, 0.1)
        y = a * x + b
        plt.plot(x, y)

        plt.scatter(u1, y1)
        # plt.xlim(-100, 100)
        plt.ylim(-100, 100)
        plt.xlabel('Current')
        plt.ylabel('Accel')
        plt.show()
        # plt.savefig("pitch_iden.png")


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

        fig, (top, mid1, mid2, bot) = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

        # plt.xlim([0, 2])  # x軸の範囲
        # plt.xlim([0.28, 0.89])  # x軸の範囲
        plt.xlabel("Time[sec]")

        top.plot(self.IdenData[num][0], self.IdenData[num][1], label = 'decimated')
        top.plot(self.IdenData_detr[num][0], self.IdenData_detr[num][1], label = 'detrend')
        top.set_ylabel('Current[%]')
        top.legend()
        top.set_yticks(np.arange(-2, 2, 0.5))
        top.set_ylim([-1.5, 1.5])  # y軸の範囲

        mid1.plot(self.IdenData[num][0], self.IdenData[num][2], label = 'decimated')
        mid1.plot(self.IdenData_detr[num][0], self.IdenData_detr[num][2], label = 'detrend')
        mid1.set_ylabel('acceleration[rad/s^2]')
        mid1.legend()
        # mid1.set_yticks(np.arange(-400, -100, 50))
        # mid1.set_ylim([-300.0, -200.0])  # y軸の範囲

        mid2.plot(self.IdenData[num][0], self.IdenData[num][3], label = 'decimated')
        mid2.plot(self.IdenData_detr[num][0], self.IdenData_detr[num][3], label = 'detrend')
        mid2.set_ylabel('Velocity[rad/s]')
        mid2.legend()
        # mid2.set_yticks(np.arange(-400, -100, 50))
        # mid2.set_ylim([-300.0, -200.0])  # y軸の範囲

        bot.plot(self.IdenData[num][0], self.IdenData[num][4], label = 'decimated')
        bot.plot(self.IdenData_detr[num][0], self.IdenData_detr[num][4], label = 'detrend')
        bot.set_ylabel('Position[rad]')
        bot.legend()
        # bot.set_yticks(np.arange(-200, 200, 50))
        # bot.set_ylim([-150.0, 150.0])  # y軸の範囲

        plt.tight_layout()
        # plt.savefig("detrend_ok.png")
        plt.show()

    def AutoCorrelation(self):

        n1 = 0
        n2 = 100000
        Decimation = 100
        time = self.pitch["time"][n1:n2:Decimation]
        data = self.roll["p_iq"][n1:n2:Decimation]
        plt.plot(time, data)
        plt.xlabel('Time [sec]')
        plt.ylabel('MSL')
        plt.show()

        N = int((n2 - n1) / Decimation / 2)

        r = np.zeros(N)

        for j in range(N):
            s = 0
            for i in range(N):
                s = s + data[i] * data[i + j]
            r[j] = s

        y = r[0]
        for i in range(N):
            r[i] = r[i] / y

        plt.plot(np.arange(N), r)
        plt.xlabel('Lag')
        plt.ylabel('Auto Correlation')
        # plt.savefig("AC.png")
        plt.show()

    def CsvOut(self):
        df1 = pd.DataFrame({
            'time': self.IdenData[0][0],
            'p_iq': self.IdenData[0][1],
            'p_am': self.IdenData[0][2],
            'p_wm': self.IdenData[0][3],
            'p_thm': self.IdenData[0][4],
            'r_iq': self.IdenData[1][1],
            'r_am': self.IdenData[1][2],
            'r_wm': self.IdenData[1][3],
            'r_thm': self.IdenData[1][4],
        })

        df2 = pd.DataFrame({
            'time': self.TestData[0],
            'p_iq': self.TestData[1],
            'p_am': self.TestData[2],
            'p_wm': self.TestData[3],
            'p_thm': self.TestData[4],
            'r_iq': self.TestData[5],
            'r_am': self.TestData[6],
            'r_wm': self.TestData[7],
            'r_thm': self.TestData[8]
        })

        df1.to_csv("10percent_balanced_data.csv", index=False)
        df1.to_csv("10percent_balanced_test.csv", index=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ID = Iden()
    ID.npz_load()
    ID.Arrangement()
    # ID.graph_sub(1)
    # ID.AutoCorrelation()
    # ID.CsvOut()
    ID.fit(1)
