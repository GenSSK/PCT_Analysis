import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import CFO_analysis
from scipy.spatial.distance import correlation

import sys
# import os
# sys.path.append(os.path.abspath("../statistics"))
# import histogram

sys.path.append('../statistics')
import myhistogram

class combine:
    def __init__(self, dyad_npz, triad_npz, tetrad_npz):
        self.dyad_cfo = CFO_analysis.CFO(dyad_npz, 'dyad')
        self.triad_cfo = CFO_analysis.CFO(triad_npz, 'triad')
        self.tetrad_cfo = CFO_analysis.CFO(tetrad_npz, 'tetrad')


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

    def summation_cfo(self, graph=1, mode='noabs'):
        dyad_pp, dyad_rp, dyad_pf, dyad_rf = self.dyad_cfo.summation_cfo_3sec(mode)
        triad_pp, triad_rp, triad_pf, triad_rf = self.triad_cfo.summation_cfo_3sec(mode)
        tetrad_pp, tetrad_rp, tetrad_pf, tetrad_rf = self.tetrad_cfo.summation_cfo_3sec(mode)
        summation_3sec_datas = [
            [dyad_pp, triad_pp, tetrad_pp],
            [dyad_rp, triad_rp, tetrad_rp],
            [dyad_pf, triad_pf, tetrad_pf],
            [dyad_rf, triad_rf, tetrad_rf],
        ]

        if graph == 0:
            sns.set()
            # sns.set_style('whitegrid')
            sns.set_palette('Set3')

            types = ['Summation Pitch PCFO (Avg)',
                     'Summation Roll PCFO (Avg)',
                     'Summation Pitch FCFO (Avg)',
                     'Summation Roll FCFO (Avg)']
            ranges = [0.05, 0.05, 0.3, 0.3]

            if mode == 'b_abs':
                types = ['Before abs.\nSummation Pitch PCFO (Avg)',
                         'Before abs.\nSummation Roll PCFO (Avg)',
                         'Before abs.\nSummation Pitch FCFO (Avg)',
                         'Before abs.\nSummation Roll FCFO (Avg)']
                ranges = [0.2, 0.2, 2.0, 2.0]

            if mode == 'a_abs':
                types = ['After abs.\nSummation Pitch PCFO (Avg)',
                         'After abs.\nSummation Roll PCFO (Avg)',
                         'After abs.\nSummation Pitch FCFO (Avg)',
                         'After abs.\nSummation Roll FCFO (Avg)']
                ranges = [0.2, 0.2, 0.8, 0.8]



            fig = plt.figure(figsize=(10, 7), dpi=150)

            plot = [
                fig.add_subplot(2, 2, 1),
                fig.add_subplot(2, 2, 2),
                fig.add_subplot(2, 2, 3),
                fig.add_subplot(2, 2, 4),
            ]

            for j in range(len(plot)):
                dfpp = []
                dfpp_melt = []
                for i in range(len(dyad_pp)):
                    dfpp.append(pd.DataFrame({
                        'Dyad': summation_3sec_datas[j][0][i],
                        'Triad': summation_3sec_datas[j][1][i],
                        'Tetrad': summation_3sec_datas[j][2][i],
                    })
                    )

                    dfpp_melt.append(pd.melt(dfpp[i]))
                    dfpp_melt[i]['Group'] = 'Group' + str(i + 1)

                df = pd.concat([i for i in dfpp_melt], axis=0)

                sns.boxplot(x="variable", y="value", data=df, ax=plot[j], sym="")
                sns.stripplot(x='variable', y='value', data=df, hue='Group', dodge=True,
                              jitter=0.2, color='black', palette='Paired', ax=plot[j])

                plot[j].legend_ = None
                plot[j].set_ylabel(types[j])
                # plot[j].axes.xaxis.set_visible(False)
                plot[j].set_ylim(-ranges[j], ranges[j])
                if mode == 'b_abs' or mode == 'a_abs':
                    plot[j].set_ylim(0, ranges[j])

            plt.tight_layout()
            plt.savefig('fig/subtraction_3sec_compare' + str(mode) + '.png')
            plt.show()

        return summation_3sec_datas

    def subtraction_cfo(self, graph=1):
        dyad_pp, dyad_rp, dyad_pf, dyad_rf = self.dyad_cfo.subtraction_cfo_3sec()
        triad_pp, triad_rp, triad_pf, triad_rf = self.triad_cfo.subtraction_cfo_3sec()
        tetrad_pp, tetrad_rp, tetrad_pf, tetrad_rf = self.tetrad_cfo.subtraction_cfo_3sec()
        subtraction_3sec_datas = [
            [dyad_pp, triad_pp, tetrad_pp],
            [dyad_rp, triad_rp, tetrad_rp],
            [dyad_pf, triad_pf, tetrad_pf],
            [dyad_rf, triad_rf, tetrad_rf],
        ]

        if graph == 0:
            sns.set()
            # sns.set_style('whitegrid')
            sns.set_palette('Set3')

            types = ['Subtraction Pitch PCFO (Avg)', 'Subtraction Roll PCFO (Avg)', 'Subtraction Pitch FCFO (Avg)',
                     'Subtraction Roll FCFO (Avg)']
            ranges = [0.4, 0.4, 6.0, 6.0]



            fig = plt.figure(figsize=(10, 7), dpi=150)

            plot = [
                fig.add_subplot(2, 2, 1),
                fig.add_subplot(2, 2, 2),
                fig.add_subplot(2, 2, 3),
                fig.add_subplot(2, 2, 4),
            ]

            for j in range(len(plot)):
                dfpp = []
                dfpp_melt = []
                for i in range(len(dyad_pp)):
                    dfpp.append(pd.DataFrame({
                        'Dyad': subtraction_3sec_datas[j][0][i],
                        'Triad': subtraction_3sec_datas[j][1][i],
                        'Tetrad': subtraction_3sec_datas[j][2][i],
                    })
                    )

                    dfpp_melt.append(pd.melt(dfpp[i]))
                    dfpp_melt[i]['Group'] = 'Group' + str(i + 1)

                df = pd.concat([i for i in dfpp_melt], axis=0)

                sns.boxplot(x="variable", y="value", data=df, ax=plot[j], sym="")
                sns.stripplot(x='variable', y='value', data=df, hue='Group', dodge=True,
                              jitter=0.2, color='black', palette='Paired', ax=plot[j])

                plot[j].legend_ = None
                plot[j].set_ylabel(types[j])
                # plot[j].axes.xaxis.set_visible(False)
                plot[j].set_ylim(0.0, ranges[j])

            plt.tight_layout()
            plt.savefig('fig/subtraction_3sec_compare.png')
            plt.show()

        return subtraction_3sec_datas

    def performance_show(self):
        error_period_dyad, spend_period_dyad = self.dyad_cfo.period_performance_cooperation()
        error_period_triad, spend_period_triad = self.triad_cfo.period_performance_cooperation()
        error_period_tetrad, spend_period_tetrad = self.tetrad_cfo.period_performance_cooperation()

        error_periode = [error_period_dyad, error_period_triad, error_period_tetrad]
        spend_periode = [spend_period_dyad, spend_period_triad, spend_period_tetrad]

        types = ['dyad', 'triad', 'tetrad']

        fig_error = plt.figure(figsize=(10, 7), dpi=150)
        fig_error.suptitle('Error Period')
        for i in range(3):
            ax = fig_error.add_subplot(3, 1, i + 1)
            ax.title.set_text(types[i])
            for error in error_periode[i]:
                ax.plot(error, label='Group' + str(i + 1))

        plt.tight_layout()
        plt.savefig('fig/performance_error.png')

        fig_spend = plt.figure(figsize=(10, 7), dpi=150)
        fig_spend.suptitle('Spent time Period')
        for i in range(3):
            ax = fig_spend.add_subplot(3, 1, i + 1)
            ax.title.set_text(types[i])
            for spend in spend_periode[i]:
                ax.plot(spend, label='Group' + str(i + 1))

        plt.tight_layout()
        plt.savefig('fig/performance_spend.png')
        plt.show()

    def performance_comparison(self, mode='h-m'):
        if mode == 'h-m':
            error_period_dyad, spend_period_dyad = self.dyad_cfo.period_performance_cooperation()
            error_period_triad, spend_period_triad = self.triad_cfo.period_performance_cooperation()
            error_period_tetrad, spend_period_tetrad = self.tetrad_cfo.period_performance_cooperation()
        elif mode == 'h-h':
            error_period_dyad, spend_period_dyad = self.dyad_cfo.period_performance_human()
            error_period_triad, spend_period_triad = self.triad_cfo.period_performance_human()
            error_period_tetrad, spend_period_tetrad = self.tetrad_cfo.period_performance_human()
        elif mode == 'm-m':
            error_period_dyad, spend_period_dyad = self.dyad_cfo.period_performance_model()
            error_period_triad, spend_period_triad = self.triad_cfo.period_performance_model()
            error_period_tetrad, spend_period_tetrad = self.tetrad_cfo.period_performance_model()

        sns.set()
        # sns.set_style('whitegrid')
        sns.set_palette('Set3')

        fig_per = plt.figure(figsize=(10, 7), dpi=150)

        ax = fig_per.add_subplot(1, 2, 1)

        if mode == 'h-m':
            ax.set_ylim(-0.02, 0.02)
        elif mode == 'h-h':
            ax.set_ylim(0.0, 0.06)
        elif mode == 'm-m':
            ax.set_ylim(0.0, 0.06)

        ep = []
        ep_melt = []
        for i in range(len(error_period_dyad)):
            ep.append(pd.DataFrame({
                'Dyad': error_period_dyad[i],
                'Triad': error_period_triad[i],
                'Tetrad': error_period_tetrad[i],
            })
            )

            ep_melt.append(pd.melt(ep[i]))
            ep_melt[i]['Group'] = 'Group' + str(i + 1)

        df_ep = pd.concat([i for i in ep_melt], axis=0)

        sns.boxplot(x="variable", y="value", data=df_ep, ax=ax, sym="")
        sns.stripplot(x='variable', y='value', data=df_ep, hue='Group', dodge=True,
                      jitter=0.2, color='black', palette='Paired', ax=ax)

        ax.legend_ = None
        ax.set_ylabel('Error Period')



        # fig_spend = plt.figure(figsize=(10, 7), dpi=150)
        ax = fig_per.add_subplot(1, 2, 2)

        if mode == 'h-m':
            ax.set_ylim(-0.5, 0.5)
        elif mode == 'h-h':
            ax.set_ylim(1.0, 3.0)
        elif mode == 'm-m':
            ax.set_ylim(1.0, 3.0)

        sp = []
        sp_melt = []
        for i in range(len(spend_period_dyad)):
            sp.append(pd.DataFrame({
                'Dyad': spend_period_dyad[i],
                'Triad': spend_period_triad[i],
                'Tetrad': spend_period_tetrad[i],
            })
            )

            sp_melt.append(pd.melt(sp[i]))
            sp_melt[i]['Group'] = 'Group' + str(i + 1)

        df_sp = pd.concat([i for i in sp_melt], axis=0)

        sns.boxplot(x="variable", y="value", data=df_sp, ax=ax, sym="")
        sns.stripplot(x='variable', y='value', data=df_sp, hue='Group', dodge=True,
                      jitter=0.2, color='black', palette='Paired', ax=ax)

        ax.legend_ = None
        ax.set_ylabel('Spend Period')
        # ax.set_ylim(-0.5, 0.5)

        plt.tight_layout()
        plt.savefig('fig/performance_comparison' + str(mode) + '.png')
        plt.show()

    def performance_relation(self):
        error_period_dyad, spend_period_dyad = self.dyad_cfo.period_performance_cooperation()
        error_period_triad, spend_period_triad = self.triad_cfo.period_performance_cooperation()
        error_period_tetrad, spend_period_tetrad = self.tetrad_cfo.period_performance_cooperation()

        error_period_dyad = error_period_dyad.reshape(-1)
        error_period_triad = error_period_triad.reshape(-1)
        error_period_tetrad = error_period_tetrad.reshape(-1)
        spend_period_dyad = spend_period_dyad.reshape(-1)
        spend_period_triad = spend_period_triad.reshape(-1)
        spend_period_tetrad = spend_period_tetrad.reshape(-1)

        error_period = np.concatenate((error_period_dyad, error_period_triad, error_period_tetrad))
        spend_period = np.concatenate((spend_period_dyad, spend_period_triad, spend_period_tetrad))

        print(1 - correlation(error_period, spend_period))

        fig = plt.figure(figsize=(5, 5), dpi=150)

        plt.scatter(error_period_dyad, spend_period_dyad, label='Dyad', color='blue')
        plt.scatter(error_period_triad, spend_period_triad, label='Triad' , color='red')
        plt.scatter(error_period_tetrad, spend_period_tetrad, label='Tetrad', color='green')

        plt.xlabel('Error Period')
        plt.ylabel('Spend Period')
        plt.legend()
        plt.tight_layout()
        plt.savefig('fig/performance_relation.png')
        plt.show()

    def performance_hist(self):
        error_period_dyad, spend_period_dyad = self.dyad_cfo.period_performance_cooperation()
        error_period_triad, spend_period_triad = self.triad_cfo.period_performance_cooperation()
        error_period_tetrad, spend_period_tetrad = self.tetrad_cfo.period_performance_cooperation()

        error_period_dyad = error_period_dyad.reshape(-1)
        error_period_triad = error_period_triad.reshape(-1)
        error_period_tetrad = error_period_tetrad.reshape(-1)
        spend_period_dyad = spend_period_dyad.reshape(-1)
        spend_period_triad = spend_period_triad.reshape(-1)
        spend_period_tetrad = spend_period_tetrad.reshape(-1)



        error = pd.DataFrame({
            'Dyad': error_period_dyad,
            'Triad': error_period_triad,
            'Tetrad': error_period_tetrad,
        }
        )

        spend = pd.DataFrame({
            'Dyad': spend_period_dyad,
            'Triad': spend_period_triad,
            'Tetrad': spend_period_tetrad,
        }
        )

        performence = [error, spend]
        label = ['error', 'spend']

        xlim = [
            (-0.1, 0.1),
            (-1, 1),
        ]
        xtics = [
            (np.arange(-5, 5, 1)),
            (np.arange(-1, 1, 1)),
        ]

        mode = ['Dyad', 'Triad', 'Tetrad']
        kwargs = dict(hist_kws={'alpha': .6}, kde_kws={'linewidth': 2})

        for i in range(len(performence)):
            fig, ax = plt.subplots(3, 1, figsize=(10, 7), dpi=150)
            for j in range(3):
                sns.histplot(performence[i][mode[j]], kde=True, bins=10, label="Counts", ax=ax[j])

                ax[j].set_title(mode[i])
                ax[j].set_xlim(xlim[i])
            plt.tight_layout()
            plt.savefig('fig/performance_hist_' + str(label[i]) + '.png')
        plt.show()

    def performance_bootstrap(self):
        R = 100000
        error_period_dyad, spend_period_dyad = self.dyad_cfo.period_performance_cooperation()
        error_period_triad, spend_period_triad = self.triad_cfo.period_performance_cooperation()
        error_period_tetrad, spend_period_tetrad = self.tetrad_cfo.period_performance_cooperation()

        data = [
            error_period_dyad,
            error_period_triad,
            error_period_tetrad,
            spend_period_dyad,
            spend_period_triad,
            spend_period_tetrad,
        ]

        bs_data = []

        for i in range(len(data)):
            bs_data.append(combine.bootstrap(self, data[i], R))


        reshape_data = [data[_].reshape(-1) for _ in range(6)]
        bs_reshape_data = [bs_data[_].reshape(-1) for _ in range(6)]

        error = pd.DataFrame({
            'Dyad': reshape_data[0],
            'Triad': reshape_data[1],
            'Tetrad': reshape_data[2],
        }
        )

        spend = pd.DataFrame({
            'Dyad': reshape_data[3],
            'Triad': reshape_data[4],
            'Tetrad': reshape_data[5],
        }
        )
        performance = [error, spend]

        bs_error = pd.DataFrame({
            'Dyad': bs_reshape_data[0],
            'Triad': bs_reshape_data[1],
            'Tetrad': bs_reshape_data[2],
        }
        )

        bs_spend = pd.DataFrame({
            'Dyad': bs_reshape_data[3],
            'Triad': bs_reshape_data[4],
            'Tetrad': bs_reshape_data[5],
        }
        )
        bs_performance = [bs_error, bs_spend]
        label = ['Error', 'Spend']


        xlim = [
            (-0.1, 0.1),
            (-1, 1),
        ]
        xtics = [
            (np.arange(-5, 5, 1)),
            (np.arange(-1, 1, 1)),
        ]

        mode = ['Dyad', 'Triad', 'Tetrad']
        kwargs = dict(hist_kws={'alpha': .6}, kde_kws={'linewidth': 2})

        fig, ax_box = plt.subplots(1, 2, figsize=(10, 7), dpi=150)
        for i in range(2):
            fig, ax = plt.subplots(3, 1, figsize=(10, 7), dpi=150)
            for j in range(3):
                sns.histplot(performance[i][mode[j]], kde=True, bins=10, label="Counts", ax=ax[j], stat='probability')
                sns.histplot(bs_performance[i][mode[j]], kde=True, bins=10, label="Counts", ax=ax[j], stat='probability')

                ax[j].set_title(mode[j])
                ax[j].set_xlim(xlim[i])

            plt.title(label[i])
            plt.tight_layout()
            plt.savefig('fig/performance_bootstrap_hist_' + str(label[i]) + '.png')

            df = pd.melt(performance[0])
            bs_df = pd.melt(bs_performance[0])
            # sns.boxplot(x="variable", y="value", data=df, ax=ax_box[i], sym="")
            sns.boxplot(x="variable", y="value", data=bs_df, ax=ax_box[i], sym="")

            plt.tight_layout()
        plt.savefig('fig/performance_bootstrap.png')
        plt.show()



    def bootstrap(self, data, R):
        lenth = len(data[0])
        data_all = data.reshape(-1)
        results = np.zeros(R)

        for i in range(R):
            sample = np.random.choice(data_all, lenth)
            results[i] = np.mean(sample)

        return results



    def summation_ave_cfo(self, graph=1, mode='noabs'):
        dyad_p, dyad_f, dyad_pa, dyad_fa = self.dyad_cfo.summation_ave_cfo_3sec(mode)
        triad_p, triad_f, triad_pa, triad_fa = self.triad_cfo.summation_ave_cfo_3sec(mode)
        tetrad_p, tetrad_f, tetrad_pa, tetrad_fa = self.tetrad_cfo.summation_ave_cfo_3sec(mode)
        summation_3sec_datas = [
            [dyad_p, triad_p, tetrad_p],
            [dyad_f, triad_f, tetrad_f],
            [dyad_pa, triad_pa, tetrad_pa],
            [dyad_fa, triad_fa, tetrad_fa],
        ]

        if graph == 0:
            sns.set()
            # sns.set_style('whitegrid')
            sns.set_palette('Set3')

            types = ['Summation PCFO (Avg)',
                     'Summation FCFO (Avg)',
                     'Summation abs. PCFO (Avg)',
                     'Summation abs. FCFO (Avg)']
            ranges = [0.06, 0.5, 0.3, 4.0]

            if mode == 'b_abs':
                types = ['Before abs.\nSummation PCFO (Avg)',
                         'Before abs.\nSummation FCFO (Avg)',
                         'Before abs.\nSummation abs. PCFO (Avg)',
                         'Before abs.\nSummation abs. FCFO (Avg)']
                ranges = [0.25, 3.0, 0.25, 3.0]

            if mode == 'a_abs':
                types = ['After abs.\nSummation PCFO (Avg)',
                         'After abs.\nSummation FCFO (Avg)',
                         'After abs.\nSummation abs. PCFO (Avg)',
                         'After abs.\nSummation abs. FCFO (Avg)']
                ranges = [0.2, 1.0, 0.25, 3.0]



            fig = plt.figure(figsize=(10, 7), dpi=150)

            plot = [
                fig.add_subplot(2, 2, 1),
                fig.add_subplot(2, 2, 2),
                fig.add_subplot(2, 2, 3),
                fig.add_subplot(2, 2, 4),
            ]

            for j in range(len(plot)):
                dfpp = []
                dfpp_melt = []
                for i in range(len(dyad_p)):
                    dfpp.append(pd.DataFrame({
                        'Dyad': summation_3sec_datas[j][0][i],
                        'Triad': summation_3sec_datas[j][1][i],
                        'Tetrad': summation_3sec_datas[j][2][i],
                    })
                    )

                    dfpp_melt.append(pd.melt(dfpp[i]))
                    dfpp_melt[i]['Group'] = 'Group' + str(i + 1)

                df = pd.concat([i for i in dfpp_melt], axis=0)

                sns.boxplot(x="variable", y="value", data=df, ax=plot[j], sym="")
                sns.stripplot(x='variable', y='value', data=df, hue='Group', dodge=True,
                              jitter=0.2, color='black', palette='Paired', ax=plot[j])

                plot[j].legend_ = None
                plot[j].set_ylabel(types[j])
                # plot[j].axes.xaxis.set_visible(False)
                plot[j].set_ylim(-ranges[j], ranges[j])
                if mode == 'b_abs' or mode == 'a_abs':
                    plot[j].set_ylim(0, ranges[j])

            plt.tight_layout()
            plt.savefig('fig/summation_ave_cfo_' + str(mode) + '.png')
            # plt.show()

        return summation_3sec_datas

    def subtraction_ave_cfo(self, graph=1):
        dyad_p, dyad_f, dyad_pa, dyad_fa = self.dyad_cfo.subtraction_ave_cfo_3sec()
        triad_p, triad_f, triad_pa, triad_fa = self.triad_cfo.subtraction_ave_cfo_3sec()
        tetrad_p, tetrad_f, tetrad_pa, tetrad_fa = self.tetrad_cfo.subtraction_ave_cfo_3sec()
        subtraction_3sec_datas = [
            [dyad_p, triad_p, tetrad_p],
            [dyad_f, triad_f, tetrad_f],
            [dyad_pa, triad_pa, tetrad_pa],
            [dyad_fa, triad_fa, tetrad_fa],
        ]

        if graph == 0:
            sns.set()
            # sns.set_style('whitegrid')
            sns.set_palette('Set3')

            types = ['Subtraction PCFO (Avg)',
                     'Subtraction FCFO (Avg)',
                     'Subtraction abs. PCFO (Avg)',
                     'Subtraction abs. FCFO (Avg)'
                     ]
            ranges = [0.4, 8.0, 0.4, 6.0]



            fig = plt.figure(figsize=(10, 7), dpi=150)

            plot = [
                fig.add_subplot(2, 2, 1),
                fig.add_subplot(2, 2, 2),
                fig.add_subplot(2, 2, 3),
                fig.add_subplot(2, 2, 4),
            ]

            for j in range(len(plot)):
                dfpp = []
                dfpp_melt = []
                for i in range(len(dyad_p)):
                    dfpp.append(pd.DataFrame({
                        'Dyad': subtraction_3sec_datas[j][0][i],
                        'Triad': subtraction_3sec_datas[j][1][i],
                        'Tetrad': subtraction_3sec_datas[j][2][i],
                    })
                    )

                    dfpp_melt.append(pd.melt(dfpp[i]))
                    dfpp_melt[i]['Group'] = 'Group' + str(i + 1)

                df = pd.concat([i for i in dfpp_melt], axis=0)

                sns.boxplot(x="variable", y="value", data=df, ax=plot[j], sym="")
                sns.stripplot(x='variable', y='value', data=df, hue='Group', dodge=True,
                              jitter=0.2, color='black', palette='Paired', ax=plot[j])

                plot[j].legend_ = None
                plot[j].set_ylabel(types[j])
                # plot[j].axes.xaxis.set_visible(False)
                plot[j].set_ylim(0.0, ranges[j])

            plt.tight_layout()
            plt.savefig('fig/subtraction_ave_cfo.png')
            # plt.show()

        return subtraction_3sec_datas

    def summation_ave_cfo_bs(self, graph=1, mode='noabs'):
        dyad_p, dyad_f, dyad_pa, dyad_fa = self.dyad_cfo.summation_ave_cfo_3sec(mode)
        triad_p, triad_f, triad_pa, triad_fa = self.triad_cfo.summation_ave_cfo_3sec(mode)
        tetrad_p, tetrad_f, tetrad_pa, tetrad_fa = self.tetrad_cfo.summation_ave_cfo_3sec(mode)
        summation_3sec_datas = [
            [dyad_p, triad_p, tetrad_p],
            [dyad_f, triad_f, tetrad_f],
            [dyad_pa, triad_pa, tetrad_pa],
            [dyad_fa, triad_fa, tetrad_fa],
        ]

        R = 10000
        summation_3sec_datas_bs = []

        for i in range(len(summation_3sec_datas)):
            summation_3sec_datas_bs.append([])
            for j in range(len(summation_3sec_datas[i])):
                summation_3sec_datas_bs[i].append(combine.bootstrap(self, summation_3sec_datas[i][j], R))

        if graph == 0:
            sns.set()
            # sns.set_style('whitegrid')
            sns.set_palette('Set3')

            types = ['Summation PCFO (Avg)',
                     'Summation FCFO (Avg)',
                     'Summation abs. PCFO (Avg)',
                     'Summation abs. FCFO (Avg)']
            ranges = [0.06, 0.5, 0.3, 4.0]

            if mode == 'b_abs':
                types = ['Before abs.\nSummation PCFO (Avg)',
                         'Before abs.\nSummation FCFO (Avg)',
                         'Before abs.\nSummation abs. PCFO (Avg)',
                         'Before abs.\nSummation abs. FCFO (Avg)']
                ranges = [0.25, 3.0, 0.25, 3.0]

            if mode == 'a_abs':
                types = ['After abs.\nSummation PCFO (Avg)',
                         'After abs.\nSummation FCFO (Avg)',
                         'After abs.\nSummation abs. PCFO (Avg)',
                         'After abs.\nSummation abs. FCFO (Avg)']
                ranges = [0.2, 1.0, 0.25, 3.0]



            fig = plt.figure(figsize=(10, 7), dpi=150)

            plot = [
                fig.add_subplot(2, 2, 1),
                fig.add_subplot(2, 2, 2),
                fig.add_subplot(2, 2, 3),
                fig.add_subplot(2, 2, 4),
            ]

            for j in range(len(plot)):
                dfpp = pd.DataFrame({
                    'Dyad': summation_3sec_datas_bs[j][0],
                    'Triad': summation_3sec_datas_bs[j][1],
                    'Tetrad': summation_3sec_datas_bs[j][2],
                })

                df = pd.melt(dfpp)

                sns.boxplot(x="variable", y="value", data=df, ax=plot[j], sym="")

                plot[j].legend_ = None
                plot[j].set_ylabel(types[j])
                # plot[j].axes.xaxis.set_visible(False)
                plot[j].set_ylim(-ranges[j], ranges[j])
                if mode == 'b_abs' or mode == 'a_abs':
                    plot[j].set_ylim(0, ranges[j])

            plt.tight_layout()
            plt.savefig('fig/summation_ave_cfo_bs' + str(mode) + '.png')
            # plt.show()

        return summation_3sec_datas


    def subtraction_ave_cfo_bs(self, graph=1):
        dyad_p, dyad_f, dyad_pa, dyad_fa = self.dyad_cfo.subtraction_ave_cfo_3sec()
        triad_p, triad_f, triad_pa, triad_fa = self.triad_cfo.subtraction_ave_cfo_3sec()
        tetrad_p, tetrad_f, tetrad_pa, tetrad_fa = self.tetrad_cfo.subtraction_ave_cfo_3sec()
        subtraction_3sec_datas = [
            [dyad_p, triad_p, tetrad_p],
            [dyad_f, triad_f, tetrad_f],
            [dyad_pa, triad_pa, tetrad_pa],
            [dyad_fa, triad_fa, tetrad_fa],
        ]

        R = 10000
        subtraction_3sec_datas_bs = []

        for i in range(len(subtraction_3sec_datas)):
            subtraction_3sec_datas_bs.append([])
            for j in range(len(subtraction_3sec_datas[i])):
                subtraction_3sec_datas_bs[i].append(combine.bootstrap(self, subtraction_3sec_datas[i][j], R))

        if graph == 0:
            sns.set()
            # sns.set_style('whitegrid')
            sns.set_palette('Set3')

            types = ['Subtraction PCFO (Avg)',
                     'Subtraction FCFO (Avg)',
                     'Subtraction abs. PCFO (Avg)',
                     'Subtraction abs. FCFO (Avg)'
                     ]
            ranges = [0.4, 8.0, 0.4, 6.0]


            fig = plt.figure(figsize=(10, 7), dpi=150)

            plot = [
                fig.add_subplot(2, 2, 1),
                fig.add_subplot(2, 2, 2),
                fig.add_subplot(2, 2, 3),
                fig.add_subplot(2, 2, 4),
            ]

            for j in range(len(plot)):
                df = pd.DataFrame({
                    'Dyad': subtraction_3sec_datas_bs[j][0],
                     'Triad': subtraction_3sec_datas_bs[j][1],
                     'Tetrad': subtraction_3sec_datas_bs[j][2],
                })

                df_melt = pd.melt(df)

                sns.boxplot(x="variable", y="value", data=df_melt, ax=plot[j], sym="")

                plot[j].legend_ = None
                plot[j].set_ylabel(types[j])
                # plot[j].axes.xaxis.set_visible(False)
                plot[j].set_ylim(0.0, ranges[j])

            plt.tight_layout()
            plt.savefig('fig/subtraction_ave_cfo_bs.png')
            # plt.show()

        return subtraction_3sec_datas

    def performance_deviation(self):
        error_period_dyad, spend_period_dyad = self.dyad_cfo.period_performance_cooperation()
        error_period_triad, spend_period_triad = self.triad_cfo.period_performance_cooperation()
        error_period_tetrad, spend_period_tetrad = self.tetrad_cfo.period_performance_cooperation()

        data = [
            error_period_dyad.T,
            error_period_triad.T,
            error_period_tetrad.T,
            spend_period_dyad.T,
            spend_period_triad.T,
            spend_period_tetrad.T,
        ]

        # data_bs = []
        #
        # R = 10000
        # for i in range(len(data)):
        #     data_bs.append([])
        #     for j in range(len(data[i])):
        #         data_bs[i].append(combine.bootstrap(self, data[i][j], R))


        df_error_raw_ = []
        df_spend_raw_ = []
        for i in range(len(data[0])):
            df_error_ = pd.DataFrame({
                'Dyad': data[0][i],
                'Triad': data[1][i],
                'Tetrad': data[2][i],
            }
            )
            df_error_raw_.append(df_error_)

            df_spend_ = pd.DataFrame({
                'Dyad': data[3][i],
                'Triad': data[4][i],
                'Tetrad': data[5][i],
            }
            )
            df_spend_raw_.append(df_spend_)

        df_error_raw = pd.concat(df_error_raw_[i] for i in range(len(df_error_raw_)))
        df_error_raw_melt = pd.melt(df_error_raw)
        Period = [i for i in range(1, len(data[0]) + 1)] * len(data[0][0]) * 3
        df_error = pd.concat([df_error_raw_melt, pd.DataFrame({'Period': Period})], axis=1)

        df_error.rename(columns={'variable': 'Group size'}, inplace=True)
        df_error.rename(columns={'value': 'Value'}, inplace=True)

        df_spend_raw = pd.concat(df_spend_raw_[i] for i in range(len(df_spend_raw_)))
        df_spend_raw_melt = pd.melt(df_spend_raw)
        df_spend = pd.concat([df_spend_raw_melt, pd.DataFrame({'Period': Period})], axis=1)

        df_spend.rename(columns={'variable': 'Group size'}, inplace=True)
        df_spend.rename(columns={'value': 'Value'}, inplace=True)

        df = [df_error, df_spend]
        label = ['Error', 'Spend']
        ylim = [
            (-0.01, 0.01),
            (-0.1, 0.3)
        ]

        kwargs = dict(
            height=10,
            aspect=1.5,
            scatter=True,
            n_boot=1000,
            x_ci='sd',
        )

        for i in range(2):
            sns.set(font_scale=2)
            sns.set_context("poster")
            # sns.set_style("whitegrid", {'grid.linestyle': '--'})
            sns.set_style("white")
            p = sns.lmplot(data=df[i], x='Period', y='Value', hue='Group size', order=2, **kwargs)
            p.set(title=label[i])
            p.set(xlim=(0, 20))
            p.set(ylim=ylim[i])
        plt.show()




        #
        # std = []
        # mean = []
        # data_use = data
        # for i in range(len(data_use)):
        #     std.append([])
        #     mean.append([])
        #     for j in range(len(data_use[i])):
        #         std[i].append(np.std(data_use[i][j]))
        #         mean[i].append(np.mean(data_use[i][j]))
        #
        # period = [i+1 for i in range(len(std[0]))] * 3
        # df_period = pd.DataFrame(period, columns=['Period'])
        #
        # df_error_ = pd.DataFrame({
        #     'Dyad': mean[0],
        #     'Triad': mean[1],
        #     'Tetrad': mean[2],
        #     }
        # )
        #
        # df_melt_error = pd.concat([pd.melt(df_error_), df_period], axis=1)
        # df_melt_error.rename(columns={'variable': 'Group size'}, inplace=True)
        # df_melt_error.rename(columns={'value': 'Deviation'}, inplace=True)
        #
        # df_spend_melt_ = pd.DataFrame({
        #     'Dyad': mean[3],
        #     'Triad': mean[4],
        #     'Tetrad': mean[5],
        # }
        # )
        # df_melt_spend = pd.concat([pd.melt(df_spend_melt_), df_period], axis=1)
        # df_melt_spend.rename(columns={'variable': 'Group size'}, inplace=True)
        # df_melt_spend.rename(columns={'value': 'Deviation'}, inplace=True)
        #
        # df = [df_melt_error, df_melt_spend]
        # label = ['Error', 'Spend']
        # ylim = [
        #     (-0.01, 0.01),
        #     (-0.1, 0.3)
        # ]
        #
        # kwargs = dict(
        #     height=10,
        #     aspect=1.5,
        #     scatter=True,
        #     n_boot=1000,
        #     x_ci='sd',
        # )
        #
        # for i in range(2):
        #     sns.set(font_scale=2)
        #     sns.set_context("poster")
        #     # sns.set_style("whitegrid", {'grid.linestyle': '--'})
        #     sns.set_style("white")
        #     p = sns.lmplot(data=df[i], x='Period', y='Deviation', hue='Group size', order=2, **kwargs)
        #     p.set(title=label[i])
        #     p.set(xlim=(0, None))
        #     p.set(ylim=ylim[i])
        # plt.show()
