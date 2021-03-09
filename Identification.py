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
        self.npz = np.load('doutei_n10.npz')
        # self.npz = np.load('pre_doutei.npz')
        # self.npz = np.load('ExpData.npz')
        # print(self.npz["time"])
        # print(self.npz["wm"])
        self.thm_dtre = signal.detrend(self.npz["thm"], type='constant')
        self.thm_dtre = signal.detrend(self.thm_dtre, type='linear')
        self.wm_dtre = signal.detrend(self.npz["wm"], type='constant')
        self.wm_dtre = signal.detrend(self.wm_dtre, type='linear')

    def Arrangement(self):
        Ta = 0.0001             # データのサンプリング時間
        Ts = 0.005               # 同定用のサンプリング時間
        Texp = 1.2              # 同定に必要な時間
        No = int(Texp/Ta)       # 同定に必要なデータの個数
        Decimation = int(Ts/Ta) # 間引きの数

        ExtractData = np.array([self.npz['time'],
                                 self.npz['iq'],
                                 self.wm_dtre,
                                 self.thm_dtre])  # 同定に必要なデータの抽出
        self.IdenData = ExtractData[:, 0:No:Decimation]  # 同定用データの作成
        self.TestData = ExtractData[:, No::Decimation]  # 同定用データの作成


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
        # plt.xlabel('Time [sec]')
        # plt.ylabel('MSL')
        plt.show()

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
        df = pd.DataFrame({
            'time': self.IdenData[0],
            'iq': self.IdenData[1],
            'wm': self.IdenData[2],
            'thm': self.IdenData[3]
        })

        df.to_csv("out.csv")




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ID = Iden()
    ID.npz_load()
    ID.Arrangement()
    # ID.StateSpace()
    # ID.AutoCorrelation()
    # ID.CorrelationAnalysis()
    ID.CsvOut()