import numpy
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd
import itertools

class ForceAnalysis:
    def __init__(self, normal_data, alone_data, nothing_data):
        # self.data = numpy.arange(3)
        self.data = {}
        # print(type(normal_data))
        self.data[0] = normal_data
        self.data[1] = alone_data
        self.data[2] = nothing_data
        self.smp = np.zeros((len(self.data), len(self.data[0])))
        self.time = np.zeros((len(self.data), len(self.data[0])))
        self.period = np.zeros((len(self.data), len(self.data[0])), dtype = int)
        self.num = np.zeros((len(self.data), len(self.data[0])), dtype = int)
        self.start_num = np.zeros((len(self.data), len(self.data[0])), dtype = int)
        self.end_num = np.zeros((len(self.data), len(self.data[0])), dtype = int)

        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                self.smp[i][j] = self.data[i][j]['time'][-1] / len(self.data[i][j]['time']) # サンプリング時間
                self.time[i][j] = self.data[i][j]['duringtime']  # ターゲットの移動時間
                self.period[i][j] = int((self.data[i][j]['tasktime'] - 9.0) / self.time[i][j])  # 回数
                self.num[i][j] = int(self.time[i][j] / self.smp[i][j])  # 1ピリオドにおけるデータ数
                self.start_num[i][j] = int((self.data[i][j]['starttime'] + 6.0) / self.smp[i][j])
                # self.end_num[i][j] = int((self.data[i][j]['endtime'] - 3.0) / self.smp[i][j])
                self.end_num[i][j] = self.start_num[i][j] + (self.num[i][j] * 10)

                # print(self.end_num[i][j])

        # self.smp = self.data[0][0]['time'][-1] / len(self.data[0][0]['time']) # サンプリング時間
        # self.time = self.data[0][0]['duringtime']  # ターゲットの移動時間
        # self.period = int((self.data[0][0]['tasktime'] - 9.0) / self.time)  # 回数
        # self.num = int(self.time / self.smp)  # 1ピリオドにおけるデータ数
        # self.start_num = int((self.data[0][0]['starttime'] + 6.0) / self.smp)
        # self.end_num = int((self.data[0][0]['endtime'] - 3.0) / self.smp)

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

    def compare_individual(self, subject_num):
        if (subject_num == 0):
            num = "i1"
        elif (subject_num == 1):
            num = "i2"
        elif (subject_num == 2):
            num = "i3"
        else:
            num = "i4"

        pos_rms = {}
        force_rms = {}

        pos_sep = {}
        force_sep = {}


        #実験の種類
        for i in range(len(self.data)):
            # #グループ数
            # for j in range(len(self.data[i])):
            #     pos_rms[i], force_rms[i] = self.calc_data(self.data[i][j][num + '_p_thm'][self.start_num:self.end_num],
            #                                               self.data[i][j][num + '_r_thm'][self.start_num:self.end_num],
            #                                               self.data[i][j][num + '_p_text'][self.start_num:self.end_num],
            #                                               self.data[i][j][num + '_r_text'][self.start_num:self.end_num])
            # pos_sep[i] = self.separator(pos_rms[i])
            # force_sep[i] = self.separator(force_rms[i])

            pos_sep[i] = self.separator(self.data[i][0][num + '_p_thm'][self.start_num:self.end_num])
            force_sep[i] = self.separator(self.data[i][0][num + '_p_text'][self.start_num:self.end_num])

        fig = []
        # for i in range(3):
        #     fig.append(plt.figure())
        #     ax = fig[i].add_subplot(111)
        #     for j in range(len(pos_sep[0]) - 10):
        #         ax.plot(np.arange(0.0, self.time, self.smp), force_sep[i][j])

        for i in range(len(pos_sep[0]) - 8):
            fig.append(plt.figure())
            ax = fig[i].add_subplot(111)
            for j in range(3):
                ax.plot(np.arange(0.0, self.time, self.smp), force_sep[j][i])

        # plt.savefig('sample_pos.png')
        plt.show()


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
                # p_thm[i] =self.data[i][0]['i1_p_thm'][self.start_num:self.end_num] \
                #         + self.data[i][0]['i2_p_thm'][self.start_num:self.end_num] \
                #         + self.data[i][0]['i3_p_thm'][self.start_num:self.end_num] \
                #         + self.data[i][0]['i4_p_thm'][self.start_num:self.end_num]
                # r_thm[i] =self.data[i][0]['i1_r_thm'][self.start_num:self.end_num] \
                #         + self.data[i][0]['i2_r_thm'][self.start_num:self.end_num] \
                #         + self.data[i][0]['i3_r_thm'][self.start_num:self.end_num] \
                #         + self.data[i][0]['i4_r_thm'][self.start_num:self.end_num]
                #
                # p_text[i]=self.data[i][0]['i1_p_text'][self.start_num:self.end_num] \
                #         + self.data[i][0]['i2_p_text'][self.start_num:self.end_num] \
                #         + self.data[i][0]['i3_p_text'][self.start_num:self.end_num] \
                #         + self.data[i][0]['i4_p_text'][self.start_num:self.end_num]
                #
                # r_text[i]= self.data[i][0]['i1_r_text'][self.start_num:self.end_num] \
                #          + self.data[i][0]['i2_r_text'][self.start_num:self.end_num] \
                #          + self.data[i][0]['i3_r_text'][self.start_num:self.end_num] \
                #          + self.data[i][0]['i4_r_text'][self.start_num:self.end_num]

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


        # print(thm_ave)
        # print(text_ave)

        # # f検定
        # f = {}
        # for i in range(len(self.data) - 1):
        #     f[i] = self.ftest(text_ave[0], text_ave[i + 1])
        # print("ftest from normal")
        # print(f)
        #
        # target = {(0, 0), (1, 2)}
        # for i, j in target:
        #     f[i] = self.ftest(text_ave[1], text_ave[j])
        # print("ftest from alone")
        # print(f)
        #
        # # t検定
        # t = {}
        # for i in range(len(self.data) - 1):
        #     t[i] = self.ttest(text_ave[0], text_ave[i + 1])
        # print("ttest from normal")
        # print(t)
        #
        # target = {(0, 0), (1, 2)}
        # for i, j in target:
        #     t[i] = self.ttest(text_ave[1], text_ave[j])
        # print("ftest from alone")
        # print(t)
        #
        # # 値だけのデータフレームnormal,alone, nothongの順番
        # value = pd.DataFrame(
        #     columns=["Value"],
        #     data=text_ave.flatten()
        # )
        #
        # # Typeのデータフレーム normal * 4, alone * 4, nothong * 4 の順
        # types = ["normal", "alone", "nothing"]
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
        # # 全てのデータフレームを横につなげる
        # df = pd.concat([value, type], axis=1)
        #
        # print(df)
        #
        # plt.rcParams['pdf.fonttype'] = 42  # PDFにフォントを埋め込むためのパラメータ
        # sns.set_theme(style="ticks", color_codes=False, font='Times New Roman', font_scale=1)
        # # sns.set_palette("muted", 8)
        # sns.catplot(x ="Types", y="Value", kind="bar", data=df, height= 3.438 * 1.1 / 1, aspect=1.1 / 1, legend=False)
        # plt.yticks(np.arange(-10, 10, 1))
        # plt.ylim(0, 2)
        #
        # # plt.tight_layout()
        # # plt.legend()
        # plt.savefig('text_compare.pdf')
        # plt.show()
