import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import each_analysis
from scipy.spatial.distance import correlation
import scipy.stats
from scipy.stats import beta
import statsmodels.api as sm
from numpy.random import Generator, PCG64, MT19937
from scipy import signal, optimize

import sys
# import os
# sys.path.append(os.path.abspath("../statistics"))
# import histogram

sys.path.append('../statistics')
import myhistogram


def fit_func_mse(parameter, *args):
    x, y = args
    a = parameter[0]
    b = parameter[1]
    residual = y - (a * x + b)
    mse = np.mean(residual ** 2)
    return mse

def fit_func(parameter, *args):
    x, y = args
    a = parameter[0]
    b = parameter[1]
    residual = y - (a * x + b)
    return residual

def fit_func_2nd(parameter, *args):
    accel, velo, y = args
    a = parameter[0]
    b = parameter[1]
    residual = y - (a * accel + b * velo)
    return residual

def fit_func_2nd_mse(parameter, *args):
    accel, velo, y = args
    a = parameter[0]
    b = parameter[1]
    residual = y - (a * accel + b * velo)
    mse = np.mean(residual ** 2)
    return mse

class combine:
    def __init__(self, PP_npz, AdPD_npz, AdAc_npz, Bi_npz):
        self.PP = each_analysis.each(PP_npz, 'PP')
        self.AdPD = each_analysis.each(AdPD_npz, 'AdPD')
        self.AdAc = each_analysis.each(AdAc_npz, 'AdAc')
        self.Bi = each_analysis.each(Bi_npz, 'Bi')

        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.default'] = 'regular'
        plt.rcParams['xtick.top'] = 'True'
        plt.rcParams['ytick.right'] = 'True'
        # plt.rcParams['axes.grid'] = 'True'
        plt.rcParams['xtick.direction'] = 'in'  # x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'  # y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['xtick.major.width'] = 1.0  # x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 1.0  # y軸主目盛り線の線幅
        plt.rcParams['font.size'] = 5  # フォントの大きさ
        plt.rcParams['axes.linewidth'] = 0.5  # 軸の線幅edge linewidth。囲みの太さ

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


    def performance_show(self):
        error_period_PP, spend_period_PP = self.PP.period_performance()
        error_period_AdPD, spend_period_AdPD = self.AdPD.period_performance()
        error_period_AdAc, spend_period_AdAc = self.AdAc.period_performance()
        error_period_Bi, spend_period_Bi = self.Bi.period_performance()

        error_periode = [error_period_PP, error_period_AdPD, error_period_AdAc, error_period_Bi]
        spend_periode = [spend_period_PP, spend_period_AdPD, spend_period_AdAc, spend_period_Bi]

        types = ['PP', 'Ad(PD)', 'Ad(Ac)', 'Bi']

        fig_error = plt.figure(figsize=(10, 7), dpi=150)
        fig_error.suptitle('Error Period')
        for i in range(4):
            ax = fig_error.add_subplot(4, 1, i + 1)
            ax.title.set_text(types[i])
            for error in error_periode[i]:
                ax.plot(error, label='Group' + str(i + 1))

        plt.tight_layout()
        plt.savefig('fig/performance_error.png')

        fig_spend = plt.figure(figsize=(10, 7), dpi=150)
        fig_spend.suptitle('Spent time Period')
        for i in range(4):
            ax = fig_spend.add_subplot(4, 1, i + 1)
            ax.title.set_text(types[i])
            for spend in spend_periode[i]:
                ax.plot(spend, label='Group' + str(i + 1))

        plt.tight_layout()
        plt.savefig('fig/performance_spend.png')
        plt.show()

    def performance_comparison(self):
        error_period_PP, spend_period_PP = self.PP.period_performance()
        error_period_AdPD, spend_period_AdPD = self.AdPD.period_performance()
        error_period_AdAc, spend_period_AdAc = self.AdAc.period_performance()
        error_period_Bi, spend_period_Bi = self.Bi.period_performance()

        types = ['PP', 'Ad(PD)', 'Ad(Ac)', 'Bi']


        sns.set(font='Times New Roman', font_scale=1.0)
        # sns.set_palette('cubehelix', 4)
        sns.set_palette('Set2', 4)
        # sns.set_palette('gist_stern_r', 4)
        # sns.set_palette('gist_earth', 10)
        sns.set_style('ticks')
        sns.set_context("paper",
                        # font_scale=1.5,
                        rc = {
                            "axes.linewidth": 0.5,
                            "legend.fancybox": False,
                            'pdf.fonttype': 42,
                            'xtick.direction': 'in',
                            'ytick.major.width': 1.0,
                            'xtick.major.width': 1.0,
                        })

        fig_per = plt.figure(figsize=(10, 4), dpi=150)


        ax = fig_per.add_subplot(1, 2, 1)


        ax.set_ylim(0.0, 0.1)

        ep = []
        ep_melt = []
        for i in range(len(error_period_PP)):
            ep.append(pd.DataFrame({
                types[0]: error_period_PP[i],
                types[1]: error_period_AdPD[i],
                types[2]: error_period_AdAc[i],
                types[3]: error_period_Bi[i]
            })
            )

            ep_melt.append(pd.melt(ep[i]))
            ep_melt[i]['Group'] = 'Group' + str(i + 1)

        df_ep = pd.concat([i for i in ep_melt], axis=0)

        sns.boxplot(x="variable", y="value", data=df_ep, ax=ax, sym="")
        # sns.stripplot(x='variable', y='value', data=df_ep, hue='Group', dodge=True,
        #               jitter=0.2, color='black', palette='Paired', ax=ax)

        ax.legend_ = None
        ax.set_ylabel('RMSE on each period (m)')



        # fig_spend = plt.figure(figsize=(10, 7), dpi=150)
        ax = fig_per.add_subplot(1, 2, 2)


        ax.set_ylim(0, 3.0)

        sp = []
        sp_melt = []
        for i in range(len(spend_period_PP)):
            sp.append(pd.DataFrame({
                types[0]: spend_period_PP[i],
                types[1]: spend_period_AdPD[i],
                types[2]: spend_period_AdAc[i],
                types[3]: spend_period_Bi[i]
            })
            )

            sp_melt.append(pd.melt(sp[i]))
            sp_melt[i]['Group'] = 'Group' + str(i + 1)

        df_sp = pd.concat([i for i in sp_melt], axis=0)

        sns.boxplot(x="variable", y="value", data=df_sp, ax=ax, sym="")
        # sns.stripplot(x='variable', y='value', data=df_sp, hue='Group', dodge=True,
        #               jitter=0.2, color='black', palette='Paired', ax=ax)

        ax.legend_ = None
        ax.set_ylabel('Spent time on each period (s)')
        # ax.set_ylim(-0.5, 0.5)

        plt.tight_layout()
        plt.savefig('fig/performance_comparison.png')
        # plt.savefig('fig/performance_comparison.pdf')
        plt.show()

    def performance_relation(self):
        error_period_PP, spend_period_PP = self.PP.period_performance()
        error_period_AdPD, spend_period_AdPD = self.AdPD.period_performance()
        error_period_AdAc, spend_period_AdAc = self.AdAc.period_performance()
        error_period_Bi, spend_period_Bi = self.Bi.period_performance()

        error_period_PP = np.array(error_period_PP)
        error_period_AdPD = np.array(error_period_AdPD)
        error_period_AdAc = np.array(error_period_AdAc)
        error_period_Bi = np.array(error_period_Bi)
        spend_period_PP = np.array(spend_period_PP)
        spend_period_AdPD = np.array(spend_period_AdPD)
        spend_period_AdAc = np.array(spend_period_AdAc)
        spend_period_Bi = np.array(spend_period_Bi)

        error_period_PP = error_period_PP.reshape(-1)
        error_period_AdPD = error_period_AdPD.reshape(-1)
        error_period_AdAc = error_period_AdAc.reshape(-1)
        error_period_Bi = error_period_Bi.reshape(-1)
        spend_period_PP = spend_period_PP.reshape(-1)
        spend_period_AdPD = spend_period_AdPD.reshape(-1)
        spend_period_AdAc = spend_period_AdAc.reshape(-1)
        spend_period_Bi = spend_period_Bi.reshape(-1)

        error_period = np.concatenate((error_period_PP, error_period_AdPD, error_period_AdAc, error_period_Bi))
        spend_period = np.concatenate((spend_period_PP, spend_period_AdPD, spend_period_AdAc, spend_period_Bi))

        print(1 - correlation(error_period, spend_period))

        fig = plt.figure(figsize=(5, 5), dpi=150)

        types = ['PP', 'Ad(PD)', 'Ad(Ac)', 'Bi']


        plt.scatter(error_period_PP, spend_period_PP, label=types[0], color='blue')
        plt.scatter(error_period_AdPD, spend_period_AdPD, label=types[1], color='orange')
        plt.scatter(error_period_AdAc, spend_period_AdAc, label=types[2], color='green')
        plt.scatter(error_period_Bi, spend_period_Bi, label=types[3], color='red')

        plt.xlabel('Error Period')
        plt.ylabel('Spend Period')
        plt.legend()
        plt.tight_layout()
        plt.savefig('fig/performance_relation.png')
        plt.show()

    def performance_each(self):
        error_period_PP, spend_period_PP = self.PP.period_performance()
        error_period_AdPD, spend_period_AdPD = self.AdPD.period_performance()
        error_period_AdAc, spend_period_AdAc = self.AdAc.period_performance()
        error_period_Bi, spend_period_Bi = self.Bi.period_performance()

        types = ['PP', 'Ad(PD)', 'Ad(Ac)', 'Bi']

        sns.set()
        # sns.set_style('whitegrid')
        sns.set_palette('Set3')

        fig_per = plt.figure(figsize=(10, 7), dpi=150)

        ax = fig_per.add_subplot(1, 2, 1)

        ax.set_ylim(0.0, 0.1)

        ep = []
        ep_melt = []
        for i in range(len(error_period_PP)):
            ep.append(pd.DataFrame({
                types[0]: error_period_PP[i],
                types[1]: error_period_AdPD[i],
                types[2]: error_period_AdAc[i],
                types[3]: error_period_Bi[i]
            })
            )

            ep_melt.append(pd.melt(ep[i]))
            ep_melt[i]['Group'] = 'Group' + str(i + 1)

        df_ep = pd.concat([i for i in ep_melt], axis=0)

        # sns.boxplot(x="variable", y="value", data=df_ep, ax=ax, sym="")
        sns.barplot(x='Group', y='value', data=df_ep, hue='variable', dodge=True,
                      color='black', palette='Paired', ax=ax)

        ax.legend_.axes.set_title('')
        ax.set_ylabel('Error Period')

        # fig_spend = plt.figure(figsize=(10, 7), dpi=150)
        ax = fig_per.add_subplot(1, 2, 2)

        ax.set_ylim(0, 3.0)

        sp = []
        sp_melt = []
        for i in range(len(spend_period_PP)):
            sp.append(pd.DataFrame({
                types[0]: spend_period_PP[i],
                types[1]: spend_period_AdPD[i],
                types[2]: spend_period_AdAc[i],
                types[3]: spend_period_Bi[i]
            })
            )

            sp_melt.append(pd.melt(sp[i]))
            sp_melt[i]['Group'] = 'Group' + str(i + 1)

        df_sp = pd.concat([i for i in sp_melt], axis=0)

        print(df_sp)

        # sns.boxplot(x="variable", y="value", data=df_sp, ax=ax, sym="")
        sns.barplot(x='Group', y='value', data=df_sp, hue='variable', dodge=True,
                    color='black', palette='Paired', ax=ax)

        # ax.legend_ = None
        ax.set_ylabel('Spend Period')
        ax.legend_.axes.set_title('')
        # ax.set_ylim(-0.5, 0.5)

        plt.tight_layout()
        # plt.savefig('fig/performance_comparison.png')
        plt.show()

        df_ep.rename(columns={'value': 'Error'}, inplace=True)
        df_ep.rename(columns={'variable': 'Types'}, inplace=True)
        df_sp.rename(columns={'value': 'Spent time'}, inplace=True)
        df_sp.rename(columns={'variable': 'Types'}, inplace=True)

        df_sp.drop(['Types', 'Group'], axis=1, inplace=True)
        df = pd.concat([df_ep, df_sp], axis=1)
        # print(df)
        return df

    def performance_each_ave(self, graph=1):
        error_period_PP, spend_period_PP = self.PP.period_performance()
        error_period_AdPD, spend_period_AdPD = self.AdPD.period_performance()
        error_period_AdAc, spend_period_AdAc = self.AdAc.period_performance()
        error_period_Bi, spend_period_Bi = self.Bi.period_performance()

        types = ['PP', 'Ad(PD)', 'Ad(Ac)', 'Bi']


        # error_ = [error_period_PP, error_period_AdPD, error_period_AdAc, error_period_Bi]
        # spend = np.array((np.array(spend_period_PP), np.array(spend_period_AdPD), np.array(spend_period_AdAc), np.array(spend_period_Bi)))
        error = np.array((error_period_PP, error_period_AdPD, error_period_AdAc, error_period_Bi))
        spend = np.array((spend_period_PP, spend_period_AdPD, spend_period_AdAc, spend_period_Bi))

        per = np.array((error, spend))
        per_ave = np.average(per, axis=3)

        # print(per_ave.shape)


        df_error_ = pd.DataFrame({
            types[0]: per_ave[0][0],
            types[1]: per_ave[0][1],
            types[2]: per_ave[0][2],
            types[3]: per_ave[0][3],
        })

        df_error = pd.melt(df_error_)
        df_error['Group'] = [_ for _ in range(1, len(error_period_PP) + 1)] * len(types)
        df_error.rename(columns={'value': 'Error'}, inplace=True)
        df_error.rename(columns={'variable': 'Types'}, inplace=True)

        df_spent_ = pd.DataFrame({
            types[0]: per_ave[1][0],
            types[1]: per_ave[1][1],
            types[2]: per_ave[1][2],
            types[3]: per_ave[1][3],
        })

        df_spent = pd.melt(df_spent_)
        df_spent['Group'] = [_ for _ in range(1, len(error_period_PP) + 1)] * len(types)
        df_spent.rename(columns={'value': 'Spent time'}, inplace=True)
        df_spent.rename(columns={'variable': 'Types'}, inplace=True)

        df = pd.concat([df_error, df_spent['Spent time']], axis=1)

        df_error_.to_csv('csv/performance_ave_error.csv', index=False)
        df_spent_.to_csv('csv/performance_ave_spent.csv', index=False)
        # print(df)

        # print(df_error_)

        if graph == 0:
            sns.set()
            # sns.set_style('whitegrid')
            sns.set_palette('Set3')

            fig_per = plt.figure(figsize=(10, 7), dpi=150)

            ax = fig_per.add_subplot(2, 1, 1)

            ax.set_ylim(0.03, 0.05)

            sns.boxplot(x="Types", y="Error", data=df, ax=ax, sym="")
            # sns.barplot(x='Types', y='Error', data=df, hue='Group', dodge=True,
            #             color='black', palette='Paired', ax=ax)
            sns.stripplot(x='Types', y='Error', data=df, hue='Group', dodge=True,
                          jitter=0.1, color='black', palette='Paired', ax=ax)

            # ax.legend_.axes.set_title('')
            ax.set_ylabel('Error Period')

            # fig_spend = plt.figure(figsize=(10, 7), dpi=150)
            ax = fig_per.add_subplot(2, 1, 2)

            ax.set_ylim(1.2, 2.2)


            sns.boxplot(x="Types", y="Spent time", data=df, ax=ax, sym="")
            # sns.barplot(x='Group', y='value', data=df_sp, hue='variable', dodge=True,
            #             color='black', palette='Paired', ax=ax)
            sns.stripplot(x='Types', y='Spent time', data=df, hue='Group', dodge=True,
                          jitter=0.1, color='black', palette='Paired', ax=ax)

            # ax.legend_ = None
            ax.set_ylabel('Spent Period')
            # ax.legend_.axes.set_title('')
            # ax.set_ylim(-0.5, 0.5)

            plt.tight_layout()
            plt.savefig('fig/performance_comparison_ave.png')
            plt.show()

        return df



    def bootstrap(self, data, R):
        lenth = len(data[0])
        data_all = data.reshape(-1)
        results = np.zeros(R)

        for i in range(R):
            sample = np.random.choice(data_all, lenth)
            results[i] = np.mean(sample)

        return results



    def summation_force_3sec(self, graph=1):
        PP_ptext, PP_rtext = self.PP.summation_force_3sec()
        AdPD_ptext, AdPD_rtext = self.AdPD.summation_force_3sec()
        AdAc_ptext, AdAc_rtext = self.AdAc.summation_force_3sec()
        Bi_ptext, Bi_rtext = self.Bi.summation_force_3sec()

        type = ['PP', 'Ad(PD)', 'Ad(Ac)', 'Bi']

        porr = ['pitch', 'roll']

        types = ['Avg. summation force of pitch-axis (Nm)',
                 'Avg. summation force of roll-axis (Nm)',
                 ]

        exp_num = [_ for _ in range(1, len(PP_ptext[0]) + 1)]


        summation_3sec_datas = [
            [PP_ptext, AdPD_ptext, AdAc_ptext, Bi_ptext],
            [PP_rtext, AdPD_rtext, AdAc_rtext, Bi_rtext],
        ]

        df = []

        for j in range(len(porr)):
            dfpp = []
            dfpp_melt = []
            for i in range(len(PP_ptext)):
                dfpp.append(pd.DataFrame({
                    type[0]: summation_3sec_datas[j][0][i],
                    type[1]: summation_3sec_datas[j][1][i],
                    type[2]: summation_3sec_datas[j][2][i],
                    type[3]: summation_3sec_datas[j][3][i],
                })
                )

                # print(dfpp[i])

                # dfpp_melt.append(pd.melt(dfpp[i])
                dfpp_melt.append(pd.melt(dfpp[i]))
                dfpp_melt[i]['Group'] = i + 1
                # print(dfpp_melt[i])`


            df_ = pd.concat([i for i in dfpp_melt], axis=0)
            df_.reset_index(drop=True, inplace=True)
            df_num = pd.DataFrame({'exp': exp_num * len(type) * len(PP_ptext)})

            df_porr = pd.DataFrame({'porr': [porr[j]] * len(type) * len(PP_ptext) * len(exp_num)})

            df.append(pd.concat([df_, df_num, df_porr], axis=1))
            df[j].rename(columns={'value': 'sum_force'}, inplace=True)
            df[j].rename(columns={'variable': 'types'}, inplace=True)

            # print(df[j])

        df_sum_force = pd.concat([df[0], df[1]], axis=0)
        df_sum_force.reset_index(drop=True, inplace=True)
        print(df_sum_force)

        ranges = [3.0, 5.0]


        if graph == 0:
            sns.set(font='Times New Roman', font_scale=1.0)
            # sns.set_palette('cubehelix', 4)
            sns.set_palette('Set2', 4)
            # sns.set_palette('gist_stern_r', 4)
            # sns.set_palette('gist_earth', 10)
            sns.set_style('ticks')
            sns.set_context("paper",
                            # font_scale=1.5,
                            rc={
                                "axes.linewidth": 0.5,
                                "legend.fancybox": False,
                                'pdf.fonttype': 42,
                                'xtick.direction': 'in',
                                'ytick.major.width': 1.0,
                                'xtick.major.width': 1.0,
                            })

            fig = plt.figure(figsize=(10, 4), dpi=150)

            plot = [
                fig.add_subplot(1, 2, 1),
                fig.add_subplot(1, 2, 2),
            ]

            for i in range(len(plot)):
                sns.boxplot(x="types", y="sum_force", data=df[i], ax=plot[i], sym="")
                # sns.stripplot(x='types', y='sub_pos', data=df[i], hue='Group', dodge=True,
                #               jitter=0.1, color='black', palette='Paired', ax=plot[i])

                # plot[j].legend_ = None
                plot[i].set_ylabel(types[i])
                # plot[i].axes.xaxis.set_visible(False)
                plot[i].set_ylim(0, ranges[i])

            plt.tight_layout()
            plt.savefig('fig/summation_ave_force.png')
            plt.show()

        return df_sum_force

    def subtraction_position_3sec(self, graph=1):
        PP_pthm, PP_rthm = self.PP.subtraction_position_3sec()
        AdPD_pthm, AdPD_rthm = self.AdPD.subtraction_position_3sec()
        AdAc_pthm, AdAc_rthm = self.AdAc.subtraction_position_3sec()
        Bi_pthm, Bi_rthm = self.Bi.subtraction_position_3sec()

        type = ['PP', 'Ad(PD)', 'Ad(ABC)', 'Bi']

        porr = ['pitch', 'roll']

        exp_num = [_ for _ in range(1, len(PP_pthm[0]) + 1)]



        subtraction_3sec_datas = [
            [PP_pthm, AdPD_pthm, AdAc_pthm, Bi_pthm],
            [PP_rthm, AdPD_rthm, AdAc_rthm, Bi_rthm],
        ]


        types = ['Avg. position difference of pitch-axis (rad)',
                 'Avg. position difference of roll-axis (rad)',
                 ]
        ranges = [0.1, 0.3]

        df = []

        for j in range(len(porr)):
            dfpp = []
            dfpp_melt = []
            for i in range(len(PP_pthm)):
                dfpp.append(pd.DataFrame({
                    type[0]: subtraction_3sec_datas[j][0][i],
                    type[1]: subtraction_3sec_datas[j][1][i],
                    type[2]: subtraction_3sec_datas[j][2][i],
                    type[3]: subtraction_3sec_datas[j][3][i],
                })
                )

                # print(dfpp[i])

                # dfpp_melt.append(pd.melt(dfpp[i])
                dfpp_melt.append(pd.melt(dfpp[i]))
                dfpp_melt[i]['Group'] = i + 1
                # print(dfpp_melt[i])`


            df_ = pd.concat([i for i in dfpp_melt], axis=0)
            df_.reset_index(drop=True, inplace=True)
            df_num = pd.DataFrame({'exp': exp_num * len(type) * len(PP_pthm)})

            df_porr = pd.DataFrame({'porr': [porr[j]] * len(type) * len(PP_pthm) * len(exp_num)})

            df.append(pd.concat([df_, df_num, df_porr], axis=1))
            df[j].rename(columns={'value': 'sub_pos'}, inplace=True)
            df[j].rename(columns={'variable': 'types'}, inplace=True)

            # print(df[j])

        df_sub_pos = pd.concat([df[0], df[1]], axis=0)
        df_sub_pos.reset_index(drop=True, inplace=True)
        print(df_sub_pos)

        if graph == 0:
            sns.set(font='Times New Roman', font_scale=1.0)
            # sns.set_palette('cubehelix', 4)
            sns.set_palette('Set2', 4)
            # sns.set_palette('gist_stern_r', 4)
            # sns.set_palette('gist_earth', 10)
            sns.set_style('ticks')
            sns.set_context("paper",
                            # font_scale=1.5,
                            rc = {
                                "axes.linewidth": 0.5,
                                "legend.fancybox": False,
                                'pdf.fonttype': 42,
                                'xtick.direction': 'in',
                                'ytick.major.width': 1.0,
                                'xtick.major.width': 1.0,
                            })

            fig = plt.figure(figsize=(10, 4), dpi=150)

            plot = [
                fig.add_subplot(1, 2, 1),
                fig.add_subplot(1, 2, 2),
            ]

            for i in range(len(plot)):
                sns.boxplot(x="types", y="sub_pos", data=df[i], ax=plot[i], sym="")
                # sns.stripplot(x='types', y='sub_pos', data=df[i], hue='Group', dodge=True,
                #               jitter=0.1, color='black', palette='Paired', ax=plot[i])

                # plot[j].legend_ = None
                plot[i].set_ylabel(types[i])
                # plot[i].axes.xaxis.set_visible(False)
                plot[i].set_ylim(0, ranges[i])

            plt.tight_layout()
            # plt.savefig('fig/subtraction_ave_position.png')
            plt.savefig('fig/subtraction_position_3sec.png')
            plt.show()

        return df_sub_pos

    def subtraction_position_ave(self, graph=1):
        PP_pthm, PP_rthm = self.PP.subtraction_position_ave()
        AdPD_pthm, AdPD_rthm = self.AdPD.subtraction_position_ave()
        AdAc_pthm, AdAc_rthm = self.AdAc.subtraction_position_ave()
        Bi_pthm, Bi_rthm = self.Bi.subtraction_position_ave()

        data = [
            [PP_pthm, AdPD_pthm, AdAc_pthm, Bi_pthm],
            [PP_rthm, AdPD_rthm, AdAc_rthm, Bi_rthm],
        ]

        porr = ['pitch', 'roll']

        type = ['PP', 'Ad(PD)', 'Ad(Ac)', 'Bi']

        df_ = []
        for i in range(len(type)):
            df_.append(pd.DataFrame({
                porr[0]: data[0][i],
                porr[1]: data[1][i],
                'Group': [_ for _ in range(1, len(data[0][0]) + 1)],
                'type': type[i],
            }))
        df = pd.concat([i for i in df_], axis=0)
        df.reset_index(drop=True, inplace=True)

        return df


    def estimation_inertia(self, graph=1):
        PP_ptext, PP_rtext = self.PP.summation_force(mode = 'noabs')
        AdPD_ptext, AdPD_rtext = self.AdPD.summation_force(mode = 'noabs')
        AdAc_ptext, AdAc_rtext = self.AdAc.summation_force(mode = 'noabs')
        Bi_ptext, Bi_rtext = self.Bi.summation_force(mode = 'noabs')

        PP_pddot, PP_rddot = self.PP.estimation_task_inertia()
        AdPD_pddot, AdPD_rddot = self.AdPD.estimation_task_inertia()
        AdAc_pddot, AdAc_rddot = self.AdAc.estimation_task_inertia()
        Bi_pddot, Bi_rddot = self.Bi.estimation_task_inertia()

        PP_pdot, PP_rdot = self.PP.get_plate_dot()
        AdPD_pdot, AdPD_rdot = self.AdPD.get_plate_dot()
        AdAc_pdot, AdAc_rdot = self.AdAc.get_plate_dot()
        Bi_pdot, Bi_rdot = self.Bi.get_plate_dot()

        type = ['PP', 'Ad(PD)', 'Ad(Ac)', 'Bi']

        porr = ['pitch', 'roll']

        axis = ['Summation force (Nm)',
                "Acceleration of plate (rad/s^2)",
                 ]


        summation_datas = [
            [PP_ptext, AdPD_ptext, AdAc_ptext, Bi_ptext],
            [PP_rtext, AdPD_rtext, AdAc_rtext, Bi_rtext],
        ]

        ddot_datas = [
            [PP_pddot, AdPD_pddot, AdAc_pddot, Bi_pddot],
            [PP_rddot, AdPD_rddot, AdAc_rddot, Bi_rddot],
        ]

        dot_datas = [
            [PP_pdot, AdPD_pdot, AdAc_pdot, Bi_pdot],
            [PP_rdot, AdPD_rdot, AdAc_rdot, Bi_rdot],
        ]

        df = []
        F = [] # 力の推定
        J = [] # 慣性モーメント
        D = [] # 減衰係数
        Slope = []
        Intercept = []

        dfpp = []
        for j in range(len(porr)):
            dfpp.append([])
            F.append([])
            J.append([])
            D.append([])
            Slope.append([])
            Intercept.append([])
            for i in range(len(type)):
                F[j].append([])
                J[j].append([])
                D[j].append([])
                Slope[j].append([])
                Intercept[j].append([])
                for k in range(len(PP_ptext)):
                    dec = 100
                    datax = ddot_datas[j][i][k][::dec]
                    datay = -summation_datas[j][i][k][::dec]

                    datax_dot = dot_datas[j][i][k][::dec]

                    # res = scipy.stats.linregress(datax, datay)
                    # print(type[i]+"_"+porr[j]+" = ", res)

                    parameter0 = [0.0, 0.0]
                    print(type[i] + "_" + porr[j] + " = ")

                    # MSEで最小化（Jのみ）
                    # result = optimize.minimize(fit_func_mse, parameter0, args=(datax, datay))
                    # print(result)
                    # # print(result.x)
                    # Slope[j][i].append(result.x[0])
                    # Intercept[j][i].append(result.x[1])

                    # LSMで最小化（Jのみ）
                    result = optimize.leastsq(fit_func, parameter0, args=(datax, datay))
                    print(result)
                    Slope[j][i].append(result[0][0])
                    Intercept[j][i].append(result[0][1])

                    F[j][i].append(Slope[j][i][k] * datax + Intercept[j][i][k])

                    # MSEで最小化（J+D）
                    # result = optimize.minimize(fit_func_2nd_mse, parameter0, args=(datax, datax_dot, datay))
                    # print(result)
                    # # print(result.x)
                    # J[j][i].append(result.x[0])
                    # D[j][i].append(result.x[1])

                    # LSMで最小化（J+D）
                    # result = optimize.leastsq(fit_func_2nd, parameter0, args=(datax, datax_dot, datay))
                    # J[j][i].append(result[0][0])
                    # D[j][i].append(result[0][1])
                    # print(result)

                    # F[j][i].append(J[j][i][k] * datax + D[j][i][k] * datax_dot)

                    # 決定係数の計算
                    sy2 = np.var(datay)  # データの分散
                    error = F[j][i][k] - datay  # 誤差
                    syx2 = np.mean(error ** 2)  # 誤差の二乗平均
                    sr2 = sy2 - syx2  # 回帰の分散
                    r2 = sr2 / sy2  # 決定係数
                    print("決定係数 = ", r2)

                    #
                    # #以下、確率分布にしたがって、データを一様に取得するやつ（問題あり）
                    # fit_res_x = beta.fit(datax)
                    # frozen_beta_x = beta.freeze(a=fit_res_x[0], b=fit_res_x[1], loc=fit_res_x[2], scale=fit_res_x[3])
                    # xx = np.linspace(np.min(datax), np.max(datax), len(datax))
                    # yhist, edges = combine.myhistogram_normalize(datax, 100)
                    #
                    # p_x = frozen_beta_x.pdf(xx)
                    # print(p_x)
                    # print(np.sum(p_x * (xx[1] - xx[0])))
                    #
                    # p_x_ = 1.0 - p_x
                    # # p_x_ = p_x_ / np.sum(p_x_ * (xx[1] - xx[0]))
                    # print(p_x_)
                    #
                    # # 乱数生成器にPCGを使う
                    # rg_pcg = Generator(PCG64())
                    # comparator = rg_pcg.random(len(p_x))
                    # # print(comparator)
                    #
                    # # 乱数生成器にメルセンヌ・ツイスタを使う
                    # # rg_mt = Generator(MT19937())
                    # # comparator = rg_mt.random(100)
                    #
                    # plt.figure(figsize=(5, 5), dpi=300)
                    #
                    #
                    # # yhist, edges = combine.myhistogram(comparator, 1000)
                    # # plt.bar(edges, yhist, label='random', color=(1, 0, 0, 0.5), width=0.001)
                    # # plt.show()
                    #
                    # # p_x = np.full(len(datax), 0.5)
                    #
                    #
                    # # while n_uniq < size:
                    # #     x = self.rand(size - n_uniq)
                    # #     if n_uniq > 0:
                    # #         p[flat_found[0:n_uniq]] = 0
                    # #     cdf = np.cumsum(p)
                    # #     cdf /= cdf[-1]
                    # #     new = cdf.searchsorted(x, side='right')
                    # #     _, unique_indices = np.unique(new, return_index=True)
                    # #     unique_indices.sort()
                    # #     new = new.take(unique_indices)
                    # #     flat_found[n_uniq:n_uniq + new.size] = new
                    # #     n_uniq += new.size
                    # # idx = found
                    #
                    #
                    #
                    #
                    #
                    #
                    #
                    #
                    #
                    #
                    #
                    #
                    #
                    # selection = np.where(comparator > p_x_, True, False)
                    # true_num = np.count_nonzero(selection == True)
                    # false_num = np.count_nonzero(selection == False)
                    # print(true_num)
                    # print(false_num)
                    #
                    # plt.figure(figsize=(5, 5), dpi=300)
                    # plt.bar(edges, yhist, label='histogram', color='orange')
                    # # plt.bar(xx, selection, label='selection', color='red')
                    # plt.plot(xx, p_x_, label='frozen pdf', color='blue')
                    # plt.legend()
                    # # plt.show()
                    #
                    # data = np.array([datax, datay])
                    # # print(data)
                    # data = data.T
                    # # print(data)
                    # data = data[np.argsort(data[:, 0])]
                    # # print(data)
                    # data = data.T
                    # # print(data)
                    #
                    # data_choice = data[:, selection]
                    # print(data_choice)
                    #
                    #
                    # fit_res_y = beta.fit(datay)
                    # frozen_beta_y = beta.freeze(a=fit_res_y[0], b=fit_res_y[1], loc=fit_res_y[2], scale=fit_res_y[3])
                    # xy = np.linspace(np.min(datay), np.max(datay), len(datay))
                    # # yhist, edges = combine.myhistogram_normalize(datay, 100)
                    #
                    # p_y = frozen_beta_y.pdf(xy)
                    # p_y = p_y / np.sum(p_y)
                    #
                    #
                    # # choice_num = 10000
                    # # datax = np.sort(datax)
                    # # datax_choice = np.random.choice(datax, choice_num, p=p_x)
                    # # datay_choice = np.random.choice(datay, choice_num, p=p_y)
                    # # datax_choice = np.sort(datax_choice)
                    # # datay_choice = np.sort(datay_choice)
                    # # res_choice = scipy.stats.linregress(datax_choice, datay_choice)
                    # # print('choice = ', res_choice)
                    #
                    # plt.figure(figsize=(5, 5), dpi=300)
                    # # yhist, edges = combine.myhistogram_normalize(datax, 1000)
                    # # plt.bar(edges, yhist, label='histogram_raw', color=(1, 0, 0, 0.5))
                    # plt.hist(datax, label='histogram_raw', color=(1, 0, 0, 0.5), bins=100, density=True)
                    # # yhist, edges = combine.myhistogram_normalize(datax_choice, 1000)
                    # # plt.bar(edges, yhist, label='histogram', color=(0, 1, 0, 0.5))
                    # plt.hist(data_choice[0], label='histogram', color=(0, 1, 0, 0.5), bins=100, density=True)
                    # plt.legend()
                    # plt.show()
                    #
                    #
                    # # print(res[0])
                    #
                    # # plt.scatter(datax, datay, s=0.5)
                    # # plt.scatter(datax_choice, datay_choice, s=0.5)
                    # #
                    # # # plt.plot(xx, xx * res[0], label='raw')
                    # # # plt.plot(xx, xx * res_choice[0], label='choice')
                    # #
                    # # plt.legend()
                    # # plt.show()

                    dfpp[j].append(pd.DataFrame({
                        'types': type[i],
                        axis[0]: datay,
                        axis[1]: datax,
                        'Estimation Force': F[j][i][k],
                        "Group" : k + 1,
                        "porr" : porr[j],
                    })
                    )



        df_pitch = pd.concat([i for i in dfpp[0]], axis=0)
        df_pitch.reset_index(drop=True, inplace=True)
        # print(df_pitch)

        df_roll = pd.concat([i for i in dfpp[1]], axis=0)
        df_roll.reset_index(drop=True, inplace=True)
        # print(df_roll)

        df = pd.concat([df_pitch, df_roll], axis=0)

        df_ = [df_pitch, df_roll]

        if graph == 0:
            sns.set(font='Times New Roman', font_scale=1.0)
            sns.set_style('ticks')
            sns.set_context("poster",
                            # font_scale=1.5,
                            rc = {
                                "axes.linewidth": 0.5,
                                "legend.fancybox": False,
                                'pdf.fonttype': 42,
                                'xtick.direction': 'in',
                                'ytick.major.width': 1.0,
                                'xtick.major.width': 1.0,
                            })

            sc_kws={
                'marker':'o',
                # 'color':'indianred',
                's':0.04,
                'alpha':0.4,
            }
            ln_kws={
                'linewidth':3,
                # 'color':'Black'
            }
            fc_kws={
                'sharex':False,
                'sharey':False,
                'legend_out':True
            }

            # 2つのグラフを並べる
            # g = sns.lmplot(data=df, x=axis[0], y=axis[1], col='porr', hue="types",
            #            fit_reg=True, scatter_kws=sc_kws, line_kws=ln_kws, ci=None,
            #            # palette=sns.color_palette('gist_stern_r', 4),
            #            # palette=sns.color_palette('gist_earth', 4),
            #            palette=sns.color_palette('Set2', 4),
            #            height=8, aspect=1, col_wrap=2,
            #            facet_kws=fc_kws,
            #            )
            #
            # g.set(xlim=(-8, 8), ylim=(50, 50), xticks=[-8, 0, 8], yticks=[-50, 0, 50])

            # 全部バラバラ
            xlim = [8, 8]
            ylim = [40, 40]
            for i in range(len(porr)):
                g = sns.lmplot(data=df_[i], x=axis[0], y=axis[1], col='types', hue="types",
                               fit_reg=True, scatter_kws=sc_kws, line_kws=ln_kws, ci=None,
                               # palette=sns.color_palette('gist_stern_r', 4),
                               # palette=sns.color_palette('gist_earth', 4),
                               palette=sns.color_palette('Set2', 4),
                               height=4, aspect=1.5, col_wrap=2,
                               facet_kws=fc_kws,
                               )

                g.set(xlim=(-xlim[i], xlim[i]), ylim=(-ylim[i], ylim[i]), xticks=[-xlim[i], 0, xlim[i]], yticks=[-ylim[i], 0, ylim[i]])
                plt.savefig('fig/estimation_inertia_changeDynamics_' + porr[i] + '.png')


            # 力の推定　全部バラバラ
            # xlim = [8, 8]
            # ylim = [8, 8]
            # for i in range(len(porr)):
            #     g = sns.lmplot(data=df_[i], x=axis[0], y='Estimation Force', col='types', hue="types",
            #                    fit_reg=False, scatter_kws=sc_kws, line_kws=ln_kws, ci=None,
            #                    # palette=sns.color_palette('gist_stern_r', 4),
            #                    # palette=sns.color_palette('gist_earth', 4),
            #                    palette=sns.color_palette('Set2', 4),
            #                    height=4, aspect=1.5, col_wrap=2,
            #                    facet_kws=fc_kws,
            #                    )
            #
            #     g.set(xlim=(-xlim[i], xlim[i]), ylim=(-ylim[i], ylim[i]), xticks=[-xlim[i], 0, xlim[i]], yticks=[-ylim[i], 0, ylim[i]])
                # plt.savefig('fig/estimation_force_changeDynamics_2nd_' + porr[i] + '.png')

            # # 散布図のみ
            # fig = plt.figure(figsize=(18, 8), dpi=100)
            #
            # plot = [
            #     fig.add_subplot(1, 2, 1),
            #     fig.add_subplot(1, 2, 2),
            # ]
            #
            # for i in range(len(porr)):
            #     g = sns.scatterplot(data=df_[i], x=axis[0], y=axis[1], hue="types", s=0.04, alpha=0.4, ax=plot[i])
            #     xlim = [-8, 8]
            #     ylim = [-50, 50]
            #     plot[i].set_xlim(xlim[0], xlim[1])
            #     plot[i].set_xticks([xlim[0], 0, xlim[1]])
            #     plot[i].set_ylim(ylim[0], ylim[1])
            #     plot[i].set_yticks([ylim[0], 0, ylim[1]])

            plt.tight_layout()
            # plt.savefig('fig/summation_ave_force.png')
            plt.show()

        return df


    # def fit_func(self, parameter, x, y):
    #     a = parameter[0]
    #     b = parameter[1]
    #     residual = y - (a * x + b)
    #     return residual
    #
    # def fit(self, x, y):
    #     parameter0 = [0., 0.]
    #     result = optimize.leastsq(combine.fit_func, parameter0, args=(x, y))
    #     print(result)
    #     a = result[0][0]
    #     b = result[0][1]
    #
    #     print('a = ', a, ' b = ', b)

    def myhistogram(data, bin):
        yhist, edges = np.histogram(data, bins=bin, density=True)
        return yhist, edges[:-1]

    def myhistogram_normalize(data, bin):
        yhist, edges = np.histogram(data, bins=bin, density=True)
        w = np.diff(edges)
        yhist = yhist * w
        return yhist, edges[:-1]