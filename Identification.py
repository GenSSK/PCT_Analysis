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
        # self.npz = np.load('doutei.npz')
        # self.npz = np.load('doutei2.npz')
        # self.npz = np.load('doutei_n8.npz')
        # self.npz = np.load('doutei_n8_v2.npz')
        # self.npz = np.load('doutei_n10.npz')
        self.npz = np.load('new_rec_test.npz')
        # self.npz = np.load('pre_doutei.npz')
        # self.npz = np.load('ExpData.npz')
        # print(self.npz["time"])
        # print(self.npz["wm"])
        # self.thm_dtre = signal.detrend(self.npz["thm"], type='constant')
        # self.thm_dtre = signal.detrend(self.thm_dtre, type='linear')
        # self.wm_dtre = signal.detrend(self.npz["wm"], type='constant')
        # self.wm_dtre = signal.detrend(self.wm_dtre, type='linear')


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

        # plt.xlim([0, 1.530])  # x軸の範囲
        # plt.xlim([0.28, 0.89])  # x軸の範囲
        plt.xlabel("Time[sec]")

        # top.plot(self.npz['time'], self.npz['iq'])
        top.plot(self.IdenData[0], self.IdenData[1])
        top.set_ylabel('Current[%]')
        top.set_yticks(np.arange(-2, 2, 0.5))
        top.set_ylim([-1.5, 1.5])  # y軸の範囲

        # mid.plot(self.npz['time'], self.npz['thm'], label = 'Raw')
        # mid.plot(self.npz['time'], self.thm_dtre, label = 'Detrend')
        # mid.plot(self.npz['time'], self.npz['thm'] - self.thm_dtre, label = 'Bias & Trend')
        mid.plot(self.IdenData[0], self.IdenData[3], label='Raw')
        mid.plot(self.IdenData[0], self.IdenData[5], label='Detrend')
        mid.plot(self.IdenData[0], self.IdenData[3] - self.IdenData[5], label='Bias & Trend')
        mid.set_ylabel('Position[rad]')
        mid.legend()
        # mid.set_yticks(np.arange(-400, -100, 50))
        # mid.set_ylim([-300.0, -200.0])  # y軸の範囲

        # bot.plot(self.npz['time'], self.npz['wm'], label = 'Raw')
        # bot.plot(self.npz['time'], self.wm_dtre, label = 'Detrend')
        # bot.plot(self.npz['time'], self.npz['wm'] - self.wm_dtre, label = 'Bias & Trend')
        bot.plot(self.IdenData[0], self.IdenData[2], label='Raw')
        bot.plot(self.IdenData[0], self.IdenData[4], label='Detrend')
        bot.plot(self.IdenData[0], self.IdenData[2] - self.IdenData[4], label='Bias & Trend')
        bot.set_ylabel('Velocity[rad/s]')
        bot.legend()
        # bot.set_yticks(np.arange(-200, 200, 50))
        # bot.set_ylim([-150.0, 150.0])  # y軸の範囲

        plt.tight_layout()
        # plt.savefig("detrend_ok.png")
        plt.show()

    def Arrangement(self):
        Ta = 0.0001             # データのサンプリング時間
        Ts = 0.005               # 同定用のサンプリング時間
        Texp = 1.2              # 同定に必要な時間
        No = int(Texp/Ta)       # 同定に必要なデータの個数
        Decimation = int(Ts/Ta) # 間引きの数

        ExtractData = np.array([self.npz['time'],
                                self.npz['p_iq'],
                                self.npz['p_am'],
                                self.npz['p_wm'],
                                self.npz['p_thm'],
                                self.npz['r_iq'],
                                self.npz['r_am'],
                                self.npz['r_wm'],
                                self.npz['r_thm']
                                ])  # 同定に必要なデータの抽出

        self.IdenData = ExtractData[:, 0:No:Decimation]  # 同定用データの作成
        # self.IdenData = np.append(self.IdenData, np.array([self.IdenData[2],
        #                                                    self.IdenData[3]]), axis=0)
        # self.IdenData[4] = signal.detrend(self.IdenData[4], type='constant')
        # self.IdenData[4] = signal.detrend(self.IdenData[4], type='linear')
        # self.IdenData[5] = signal.detrend(self.IdenData[5], type='constant')
        # self.IdenData[5] = signal.detrend(self.IdenData[5], type='linear')

        self.TestData = ExtractData[:, No::Decimation]  # 同定用データの作成
        # self.TestData = np.append(self.TestData, np.array([self.TestData[2],
        #                                                    self.TestData[3]]), axis=0)
        # self.TestData[4] = signal.detrend(self.TestData[4], type='constant')
        # self.TestData[4] = signal.detrend(self.TestData[4], type='linear')
        # self.TestData[5] = signal.detrend(self.TestData[5], type='constant')
        # self.TestData[5] = signal.detrend(self.TestData[5], type='linear')


    def StateSpace(self):
        J = 1
        A = [[0, 1], [0, 0]] # 行列
        B = [[0], [1/J]] # 行ベクトル
        C = [1, 1] #　列ベクトル

        ss = control.ss(A, B, C, 0)
        print(ss)

    def AutoCorrelation(self):
        
        n1 = 50
        n2 = 30050
        Decimation = 50
        time = self.npz["time"][n1:n2:Decimation]
        data = self.npz["iq"][n1:n2:Decimation]
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
        plt.xlabel('Rag')
        plt.ylabel('Auto Correlation')
        plt.savefig("AC.png")
        # plt.show()

    def CorrelationAnalysis(self):
        var_u = 0
        N = len(self.IdenData[3])
        n = int(N / 2)
        g = np.zeros(N)
        for i in range(N):
            var_u = var_u + self.IdenData[1, i] ** 2
        var_u = var_u / N

        print(var_u)

        for j in range(n):
            s = 0
            for i in range(n):
                s = s + self.IdenData[1, i] * self.IdenData[3, i + j]
            g[j] = s / var_u

        print(g)

        # plt.plot(self.IdenData[0], self.IdenData[2])
        plt.xlim(0, 50)
        plt.stem(np.arange(N), g)
        # plt.xlabel('Time [sec]')
        # plt.ylabel('MSL')
        plt.show()

    # def TransferFunction(self):

    def CsvOut(self):
        df1 = pd.DataFrame({
            'time': self.IdenData[0],
            'p_iq': self.IdenData[1],
            'p_am': self.IdenData[2],
            'p_wm': self.IdenData[3],
            'p_thm': self.IdenData[4],
            'r_iq': self.IdenData[5],
            'r_am': self.IdenData[6],
            'r_wm': self.IdenData[7],
            'r_thm': self.IdenData[8]
        })

        df2 = pd.DataFrame({
            'time': self.IdenData[0],
            'p_iq': self.IdenData[1],
            'p_am': self.IdenData[2],
            'p_wm': self.IdenData[3],
            'p_thm': self.IdenData[4],
            'r_iq': self.IdenData[5],
            'r_am': self.IdenData[6],
            'r_wm': self.IdenData[7],
            'r_thm': self.IdenData[8]
        })

        df1.to_csv("new_data.csv", index=False)
        df2.to_csv("new_test.csv", index=False)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ID = Iden()
    ID.npz_load()
    ID.Arrangement()
    # ID.graph_sub()
    # ID.StateSpace()
    # ID.AutoCorrelation()
    # ID.CorrelationAnalysis()
    ID.CsvOut()