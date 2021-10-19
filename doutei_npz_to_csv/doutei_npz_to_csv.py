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
        self.dir = "J:\マイドライブ\program\ARCS-PCT\data\doutei_npz_to_csv\\"
        # self.data = np.load(self.dir + 'mls_n8_smp100_roll.npz')
        self.data = np.load(self.dir + 'cpr4000000_am500_sinwave.npz')

    def CsvOut(self):
        df1 = pd.DataFrame({
            'time': self.data["time"],
            'tad': self.data["i1_p_tad"],
            'am': self.data["i1_p_am"],
            'wm': self.data["i1_p_wm"],
            'thm': self.data["i1_p_thm"],
        })

        df2 = pd.DataFrame({
            'time': self.data["time"],
            'tad': self.data["i1_r_tad"],
            'am': self.data["i1_r_am"],
            'wm': self.data["i1_r_wm"],
            'thm': self.data["i1_r_thm"],
        })

        df2.to_csv(self.dir + "cpr4000000_am500_sinwave.csv", index=False)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ID = Iden()
    ID.npz_load()