# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import control
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
        Ts = 0.05               # 同定用のサンプリング時間
        Texp = 1.2              # 同定に必要な時間
        No = int(Texp/Ta)       # 同定に必要なデータの個数
        Decimation = int(Ts/Ta) # 間引きの数

        ExtractData = np.array([self.npz['time'],
                                 self.npz['iq'],
                                 self.npz['wm'],
                                 self.npz['thm']])  # 同定に必要なデータの抽出
        self.IdenData = ExtractData[:, 0:No:Decimation]  # 同定用データの作成
        self.TestData = ExtractData[:, No::Decimation]  # 同定用データの作成


    def StateSpace(self):
        J = 1
        A = [[0, 1], [0, 0]] # 行列
        B = [[0], [1/J]] # 行ベクトル
        C = [1, 1] #　列ベクトル

        ss = control.ss(A, B, C, 0)
        print(ss)

    # def TransferFunction(self):




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ID = Iden()
    ID.npz_load()
    ID.Arrangement()
    ID.StateSpace()