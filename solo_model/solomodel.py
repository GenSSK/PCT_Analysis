import numpy
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd
import itertools

class SoloModel:
    def __init__(self, data):
        self.data = data

    def ttest(self, a, b):
        statistics, pvalue = stats.ttest_rel(a, b)
        return pvalue

    def ftest(self, a, b):
        statistics, pvalue = stats.bartlett(a, b)
        return pvalue

    def rms(self, data1, data2):
        val = np.sqrt(data1 ** 2 + data2 ** 2)
        return val

    def separator(self, dat):
        separeted = dat.reshape([self.period, self.num])
        return separeted

    def calc_data(self, p_position, r_position, p_force, r_force):
        p = self.rms(p_position, r_position)
        f = self.rms(p_force, r_force)
        return p, f

    def analyze(self):
        dec = 100
        df = pd.DataFrame({'time': [0.0],
                           'val': [0.0],
                           'type': ['0'],
                           'lorp':['0']})

        value = np.array([
            self.data['pre_thm_r'][::dec],
            self.data['pre_thm_p'][::dec],
            self.data['pre_text_r'][::dec],
            self.data['pre_text_p'][::dec],
            self.data['label_thm_r'][::dec],
            self.data['label_thm_p'][::dec],
            self.data['label_text_r'][::dec],
            self.data['label_text_p'][::dec]
        ])

        # 値だけのデータフレーム
        df_val = pd.DataFrame(
            columns=['val'],
            data = value.flatten()
        )

        # print(df_val)

        # 時間だけのデータフレーム
        pre_t = self.data['pre_time'][::dec].copy()
        for i in range(7):
            pre_t = np.hstack((pre_t, self.data['pre_time'][::dec]))

        df_time = pd.DataFrame(
            columns=['time'],
            data = pre_t
        )

        # print(df_time)

        types = ["thm_r", "thm_p", "text_r", "text_p"]
        type1 = []
        for j in range(2):
            for i in types:
                type1.append([i] * self.data['pre_time'][::dec].size)

        type1 = list(itertools.chain.from_iterable(type1))
        # print(len(type1))
        df_types = pd.DataFrame(
            columns=["type"],
            data = type1
        )

        lorps = ["label", "predict"]
        type2 = []
        for i in lorps:
            for k in range(4):
                type2.append([i] * self.data['pre_time'][::dec].size)

        type2 = list(itertools.chain.from_iterable(type2))
        # print(len(type2))
        df_lorp = pd.DataFrame(
            columns=["lorp"],
            data=type2
        )

        df = pd.concat([df_time, df_val, df_types, df_lorp], axis=1)

        # print(df)

        sns.relplot(data=df, row='type', x='time', y='val', hue='lorp', kind='line',height=2, aspect=3)

        # plt.tight_layout()
        # plt.legend()
        # plt.savefig('text_compare.pdf')
        plt.show()


    def check_loss(self):
        plt.plot(np.arange(self.data['train_loss'].size), self.data['train_loss'])
        plt.show()