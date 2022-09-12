import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import CFO_analysis


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

    def summation_CFO_analysis(self):
        dyad_pp, dyad_rp, dyad_pf, dyad_rf = self.dyad_cfo.summation_cfo_3sec()
        triad_pp, triad_rp, triad_pf, triad_rf = self.triad_cfo.summation_cfo_3sec()
        tetrad_pp, tetrad_rp, tetrad_pf, tetrad_rf = self.tetrad_cfo.summation_cfo_3sec()
        summation_3sec_datas = [
            [dyad_pp, triad_pp, tetrad_pp],
            [dyad_rp, triad_rp, tetrad_rp],
            [dyad_pf, triad_pf, tetrad_pf],
            [dyad_rf, triad_rf, tetrad_rf],
        ]


        sns.set()
        # sns.set_style('whitegrid')
        sns.set_palette('Set3')

        types = ['Summation Pitch PCFO (Avg)', 'Summation Roll PCFO (Avg)', 'Summation Pitch FCFO (Avg)',
                 'Summation Roll FCFO (Avg)']
        ranges = [0.05, 0.05, 0.3, 0.3]



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

        plt.tight_layout()
        plt.show()
        # plt.savefig('fig/summation_3sec_noabs.png')

    def performance_show(self):
        error_period_dyad, spend_period_dyad = self.dyad_cfo.period_performance_cooperation()
        error_period_triad, spend_period_triad = self.triad_cfo.period_performance_cooperation()
        error_period_tetrad, spend_period_tetrad = self.tetrad_cfo.period_performance_cooperation()

        error_periode = [error_period_dyad, error_period_triad, error_period_tetrad]
        spend_periode = [spend_period_dyad, spend_period_triad, spend_period_tetrad]

        types = ['dyad', 'triad', 'tetrad']

        fig_error = plt.figure(figsize=(10, 7), dpi=300)
        fig_error.suptitle('Error Period')
        for i in range(3):
            ax = fig_error.add_subplot(3, 1, i + 1)
            ax.title.set_text(types[i])
            for error in error_periode[i]:
                ax.plot(error, label='Group' + str(i + 1))

        fig_spend = plt.figure(figsize=(10, 7), dpi=300)
        for i in range(3):
            ax = fig_spend.add_subplot(3, 1, i + 1)
            for spend in spend_periode[i]:
                ax.plot(spend, label='Group' + str(i + 1))

        plt.tight_layout()
        plt.show()

    def performance_comparison(self):
        error_period_dyad, spend_period_dyad = self.dyad_cfo.period_performance_cooperation()
        error_period_triad, spend_period_triad = self.triad_cfo.period_performance_cooperation()
        error_period_tetrad, spend_period_tetrad = self.tetrad_cfo.period_performance_cooperation()

        sns.set()
        # sns.set_style('whitegrid')
        sns.set_palette('Set3')

        fig_error = plt.figure(figsize=(10, 7), dpi=150)

        ax = fig_error.add_subplot(1, 1, 1)

        ep = []
        ep_melt = []
        for i in range(5):
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
        ax.set_ylim(-0.02, 0.02)


        fig_spend = plt.figure(figsize=(10, 7), dpi=150)

        ax = fig_spend.add_subplot(1, 1, 1)

        sp = []
        sp_melt = []
        for i in range(5):
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
        ax.set_ylim(-0.5, 0.5)

        plt.tight_layout()
        plt.show()

    def subtraction_cfo(self):
        dyad_sub_ppcfo, dyad_sub_rpcfo, dyad_sub_pfcfo, dyad_sub_rfcfo =  self.dyad_cfo.subtraction_cfo()
        triad_sub_ppcfo, triad_sub_rpcfo, triad_sub_pfcfo, triad_sub_rfcfo =  self.triad_cfo.subtraction_cfo()
        tetrad_sub_ppcfo, tetrad_sub_rpcfo, tetrad_sub_pfcfo, tetrad_sub_rfcfo =  self.tetrad_cfo.subtraction_cfo()

        sub_cfo = [
            [dyad_sub_ppcfo, dyad_sub_rpcfo, dyad_sub_pfcfo, dyad_sub_rfcfo],
            [triad_sub_ppcfo, triad_sub_rpcfo, triad_sub_pfcfo, triad_sub_rfcfo],
            [tetrad_sub_ppcfo, tetrad_sub_rpcfo, tetrad_sub_pfcfo, tetrad_sub_rfcfo]
        ]

