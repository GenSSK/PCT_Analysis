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
        lorps = ["label", "predict"]
        df = pd.DataFrame({'time': [0.0],
                           'val': [0.0],
                           'type': ['0'],
                           'lorp':['0']})

        value = np.array([
            self.data['pre_thm_r'],
            self.data['pre_thm_p'],
            self.data['pre_text_r'],
            self.data['pre_text_p'],
            self.data['label_thm_r'],
            self.data['label_thm_p'],
            self.data['label_text_r'],
            self.data['label_text_p']
        ])

        # 値だけのデータフレーム
        df_val = pd.DataFrame(
            columns=['val'],
            data = value.flatten()
        )

        # 時間だけのデータフレーム
        pre_t = self.data['pre_time'].copy()
        for i in range(7):
            pre_t = np.hstack((pre_t, self.data['pre_time']))

        print(pre_t)

        # df_time = pd.DataFrame(
        #     columns=['time'],
        #     data =
        # )



        # types = ["thm", "text"]
        # type1 = []
        # for i in types:
        #     type1.append([i] * len(self.data[0]))
        # type1 = list(itertools.chain.from_iterable(type1))
        # # print(type1)
        # type = pd.DataFrame(
        #     columns=["Types"],
        #     data=type1
        # )
        #
        #
        #






    def diff_force(self):
        p_thm = {}
        r_thm = {}
        p_text = {}
        r_text = {}
        thm = {}
        text = {}
        thm_ave = np.zeros((len(self.data), len(self.data[0])))
        text_ave = np.zeros((len(self.data), len(self.data[0])))


        # plt.plot(self.data[0][3]['time'], self.rms(self.data[0][3]['i1_p_thm'], self.data[0][3]['i1_r_thm']))
        # plt.show()
        # print(self.data[0][0]['time'])
        # print(len(self.data[0][0]['time']))
        # print(len(self.data[0][0]['i1_p_thm'][self.start_num[0][0]:self.end_num[0][0]]))
        fig = []

        types = ["normal", "alone", "nothing"]
        text_df = pd.DataFrame({'Time': [0.0],
                                'text': [0.0],
                                'Types':['0']})

        # print(text_df)

        #実験の種類　３回
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):

                buf1   =     self.rms(self.data[i][j]['i1_p_thm'][self.start_num[i][j]:self.end_num[i][j]],
                                      self.data[i][j]['i1_r_thm'][self.start_num[i][j]:self.end_num[i][j]]) \
                           + self.rms(self.data[i][j]['i2_p_thm'][self.start_num[i][j]:self.end_num[i][j]],
                                      self.data[i][j]['i2_r_thm'][self.start_num[i][j]:self.end_num[i][j]]) \
                           + self.rms(self.data[i][j]['i3_p_thm'][self.start_num[i][j]:self.end_num[i][j]],
                                      self.data[i][j]['i3_r_thm'][self.start_num[i][j]:self.end_num[i][j]]) \
                           + self.rms(self.data[i][j]['i4_p_thm'][self.start_num[i][j]:self.end_num[i][j]],
                                      self.data[i][j]['i4_r_thm'][self.start_num[i][j]:self.end_num[i][j]])

                thm_ = buf1 / 4.0

                buf2    =     self.rms(self.data[i][j]['i1_p_text'][self.start_num[i][j]:self.end_num[i][j]],
                                       self.data[i][j]['i1_r_text'][self.start_num[i][j]:self.end_num[i][j]]) \
                            + self.rms(self.data[i][j]['i2_p_text'][self.start_num[i][j]:self.end_num[i][j]],
                                       self.data[i][j]['i2_r_text'][self.start_num[i][j]:self.end_num[i][j]]) \
                            + self.rms(self.data[i][j]['i3_p_text'][self.start_num[i][j]:self.end_num[i][j]],
                                       self.data[i][j]['i3_r_text'][self.start_num[i][j]:self.end_num[i][j]]) \
                            + self.rms(self.data[i][j]['i4_p_text'][self.start_num[i][j]:self.end_num[i][j]],
                                       self.data[i][j]['i4_r_text'][self.start_num[i][j]:self.end_num[i][j]])

                text_= buf2 / 4.0

                thm_ave[i][j] = np.sum(thm_) / len(thm_)
                text_ave[i][j] = np.sum(text_) / len(thm_)

                types_buf = []
                types_buf.append([types[i]] * self.num[i][j] * self.period[i][j])
                types_buf = list(itertools.chain.from_iterable(types_buf))

                # print(types_buf)
                time = self.data[i][j]['time'][:self.num[i][j]]
                for l in range(self.period[i][j] - 1):
                    time = np.append(time, self.data[i][j]['time'][:self.num[i][j]])

                # print(len(time))
                # print(len(text_))
                # print(len(types_buf))

                buf_df = pd.DataFrame({'Time': time[::100],
                                        'text': text_[::100],
                                        'Types': types_buf[::100]
                                       })

                # print(buf_df)

                text_df = pd.concat([text_df, buf_df], axis=0)

                # fig.append(plt.figure())
                # ax = fig[i * len(self.data) + j].add_subplot(111)
                # for k in range(self.period[i][j]):
                #     ax.plot(self.data[i][j]['time'][:self.num[i][j]], thm_[self.num[i][j] * k:self.num[i][j] * (k + 1)])

        text_df = text_df.drop(index=0)
        # print(text_df)

        plt.rcParams['pdf.fonttype'] = 42  # PDFにフォントを埋め込むためのパラメータ
        # sns.lmplot(x="Time", y="text", hue='Types', data=text_df, scatter=False)
        # sns.lmplot(x="Time", y="text", row='Types', data=text_df, scatter=True, order=15,  height=3, aspect= 5 / 3.438)
        # plt.ylim(0, 3)
        # plt.xlim(0, 3)

        sns.lmplot(x="Time", y="text", hue='Types', data=text_df, scatter=False, order=15, height=3, aspect= 5 / 3.438, legend=False)
        # plt.yticks(np.arange(-10, 10, 1))
        plt.ylim(0, 3)
        plt.xlim(0, 3)
        plt.legend()
        plt.savefig('text_compare_line.pdf')
        plt.show()

