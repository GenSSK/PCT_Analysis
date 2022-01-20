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
        self.dir = "J:\マイドライブ\program\ARCS-PCT\data\doutei_PCT\\"
        # self.data = np.load(self.dir + '2021-12-09_doutei_p.npz')
        # self.data = np.load(self.dir + '2021-12-09_doutei_r.npz')
        # self.data = np.load(self.dir + '2021-12-09_test_p.npz')
        self.data = np.load(self.dir + '2021-12-09_test_r.npz')

    def CsvOut(self):
        Ta = 0.0001  # データのサンプリング時間[sec]
        Ts = 0.01  # 同定用のサンプリング時間[sec]
        Tstr = 0  # 同定を開始する時間[sec]
        Texp = 10  # 同定に必要な時間[sec]
        No = int(Texp / Ta)  # 同定に必要なデータの個数
        Dec = int(Ts / Ta)  # 間引きの数
        str = int(Tstr / Ta) # 開始データ番号

        df1 = pd.DataFrame({
            'time': self.data["time"][str:No+str:Dec],
            'tad': self.data["i1_p_tad"][str:No+str:Dec],
            'am': self.data["i1_p_am"][str:No+str:Dec],
            'wm': self.data["i1_p_wm"][str:No+str:Dec],
            'thm': self.data["i1_p_thm"][str:No+str:Dec],
        })

        df2 = pd.DataFrame({
            'time': self.data["time"][str:No+str:Dec],
            'tad': self.data["i1_r_tad"][str:No+str:Dec],
            'am': self.data["i1_r_am"][str:No+str:Dec],
            'wm': self.data["i1_r_wm"][str:No+str:Dec],
            'thm': self.data["i1_r_thm"][str:No+str:Dec],
        })

        # df1.to_csv(self.dir + "2021-12-09_doutei_dec_p.csv", index=False)
        # df2.to_csv(self.dir + "2021-12-09_doutei_dec_r.csv", index=False)
        # df1.to_csv(self.dir + "2021-12-09_test_dec_p.csv", index=False)
        df2.to_csv(self.dir + "2021-12-09_test_dec_r.csv", index=False)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ID = Iden()
    ID.npz_load()
    ID.CsvOut()