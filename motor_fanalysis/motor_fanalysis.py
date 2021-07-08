# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import control
import matplotlib as plt
import pandas as pd
from scipy import signal, optimize
import matplotlib.pyplot as plt


class MFA:
    def npz_load(self):
        self.pitch = np.load('motor_analysis_pitch.npz')
        self.roll = np.load('motor_analysis_roll.npz')

    def Arrangement(self):
        Ta = 0.0001  # データのサンプリング時間[sec]
        Ts = 0.0001  # 同定用のサンプリング時間[sec]
        Tstr = 0 # 同定を開始する時間[sec]
        Texp = 5  # 同定に必要な時間[sec]
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

        DecimData_pitch = ExtractData_pitch[:, str:No+str:Decimation]  # 同定用データの作成
        DecimData_roll = ExtractData_roll[:, str:No+str:Decimation]  # 同定用データの作成

        self.IdenData = np.stack(((DecimData_pitch, DecimData_roll)))

        self.IdenData_detr = np.zeros(((2, 5, len(self.IdenData[0][0]))))
        self.IdenData_detr[0][0] = self.IdenData[0][0]
        self.IdenData_detr[1][0] = self.IdenData[1][0]

        for i in range(2):
            for j in range(4):
                a = signal.detrend(self.IdenData[i][j + 1], type='constant')
                b = signal.detrend(a, type='linear')
                self.IdenData_detr[i][j + 1] = b


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
        df1.to_csv("motor_fanalysis.csv", index=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    MA = MFA()
    MA.npz_load()
    MA.Arrangement()
    MA.CsvOut()