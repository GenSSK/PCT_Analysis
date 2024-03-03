# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize

import Npz
import seaborn as sns
import os


plt.switch_backend('Qt5Agg')

class Analysis:
    def __init__(self, read_data, mode):
        self.data = read_data
        self.mode = mode
        self.model_num = 0
        if self.mode == 'Follower':
            self.model_num = 1
        elif self.mode == 'Leader':
            self.model_num = 2
        elif self.mode == 'Altruistic':
            self.model_num = 3
        elif self.mode == 'Selfish':
            self.model_num = 4

        self.smp = 0.01  # サンプリング時間
        self.time = 3.0  # ターゲットの移動時間
        self.eliminationtime = 0.0  # 消去時間
        self.starttime = 12.0  # タスク開始時間
        self.endtime = 42.0  # タスク終了時間
        self.tasktime = self.endtime - self.starttime  # タスクの時間
        self.period = int((self.tasktime - self.eliminationtime) / self.time)  # 回数
        self.num = int(self.time / self.smp)  # 1ピリオドにおけるデータ数
        self.start_num = int((self.starttime) / self.smp)
        self.end_num = int((self.endtime) / self.smp)
        self.nn_read_flag = False

        self.dec = 10

        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.default'] = 'regular'
        plt.rcParams['xtick.top'] = 'True'
        plt.rcParams['ytick.right'] = 'True'
        # plt.rcParams['axes.grid'] = 'True'
        plt.rcParams['xtick.direction'] = 'in'  # x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'  # y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['xtick.major.width'] = 0.0  # x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 0.0  # y軸主目盛り線の線幅
        plt.rcParams['font.size'] = 8  # フォントの大きさ
        plt.rcParams['axes.linewidth'] = 0.0  # 軸の線幅edge linewidth。囲みの太さ

        plt.rcParams['lines.linewidth'] = 0.5  # 線の太さ

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

    def graph_sub(self):
        # type = ['thm_r', 'thm_p', 'text_r', 'text_p']
        gain_type = ['thm_gain', 'thm_gain', 'text_gain', 'text_gain']
        label = ['Roll angle (rad)', 'Pitch angle (rad)', 'Roll force (Nm)', 'Pitch force (Nm)']

        type = ['thm_p', 'text_p']
        label = ['Pitch angle (rad)', 'Pitch force (Nm)']

        fig, ax = plt.subplots(len(type), 1, figsize=(10, 6), dpi=150, sharex=True)
        plt.xticks(np.arange(self.starttime, self.endtime * 2, self.time * 2))
        plt.xlim([self.starttime, self.endtime])  # x軸の範囲
        plt.xlabel("Time (sec)")
        ytics = [
            np.arange(-10, 10, 0.5),
            np.arange(-10, 10, 0.5),
            np.arange(-8.0, 8.0, 2.0),
            np.arange(-8.0, 8.0, 2.0),
        ]
        ylim = [
            [-1.5, 1.5],
            [-1.5, 1.5],
            [-8.0, 8.0],
            [-8.0, 8.0],
        ]

        ytics = [
            np.arange(-10, 10, 0.5),
            np.arange(-8.0, 8.0, 2.0),
        ]
        ylim = [
            [-1.5, 1.5],
            [-8.0, 8.0],
        ]

        # カラーマップを選択（'Blues'から'Reds'に変化する）
        cmap = plt.get_cmap('winter')

        for i, sub in enumerate(ax):
            data = self.data[0]
            sub.plot(data['time'][self.start_num:self.end_num:self.dec] - 20.0, data['label_' + type[i]][self.start_num:self.end_num:self.dec], label='Label', lw=4, color='Black', alpha=0.4)
            for k in range(len(self.data)):
                data = self.data[k]
                delay = data['output_delay'][0] / self.smp
                delay = int(delay) * 2
                delay = 0

                color = cmap(k / (len(self.data) - 1))  # カラーマップの値を0から1に変化

                if self.mode == 'learn' or self.mode == 'predict':
                    sc = sub.plot(data['time'][self.start_num:self.end_num:self.dec] - 20.0, data['pre_' + type[i]][self.start_num - delay:self.end_num - delay:self.dec],
                             label=str(data['output_delay'][0]) + 's', lw=0.5, color=color, alpha=0.4 - k * 0.005)
                else:
                    sc = sub.plot(data['time'][self.start_num:self.end_num:self.dec] - 20.0, data['pre_' + type[i] + '_model_' + str(self.model_num)][self.start_num - delay:self.end_num - delay:self.dec],
                                  label=str(data['output_delay'][0]) + 's', lw=0.5, color=color, alpha=0.4 - k * 0.005)
                # sub.plot(data['time'][self.start_num:self.end_num:self.dec] - 20.0, data['label_' + type[i] + '_1'][self.start_num + interval:self.end_num + interval:self.dec] / data[gain_type[i]][0],
                #          label=str(data['output_delay'][0]) + 's', lw=0.5, color=color, alpha=0.7 - k * 0.01)
                # for k in range(data['predictionNum'][0] - 1):
                #     sub.plot(data['time'][::self.dec], data['label_' + type[i] + '_' + str(k+2)][::self.dec], label=str('{:.1f}'.format((k+1)*0.3)) + 's')
                #     sub.plot(data['time'][::self.dec], data['pre_' + type[i] + '_' + str(k+2)][::self.dec], label=str('{:.1f}'.format((k+1)*0.3)) + 's')

            sub.set_ylabel(label[i])
            # sub.legend(ncol=10, columnspacing=1, loc='upper left')
            sub.set_yticks(ytics[i])
            sub.set_ylim(ylim[i])


        # カラーバーを一つにまとめる
        norm = Normalize(vmin=0.0, vmax=3.0)
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax[0], location='top')
        cbar.set_label('Forward time')  # カラーバーにラベルを付ける

        plt.tight_layout()

        os.makedirs('fig', exist_ok=True)
        plt.savefig("fig/Time series_" + self.mode + ".pdf")
        plt.show()

    def graph_sub_single_integration(self, num):

        # type = ['thm_r', 'thm_p', 'text_r', 'text_p']
        gain_type = ['thm_gain', 'thm_gain', 'text_gain', 'text_gain']
        label = ['Roll angle (rad)', 'Pitch angle (rad)', 'Roll force (Nm)', 'Pitch force (Nm)']

        type = ['thm_p', 'text_p']
        label = ['Pitch angle (rad)', 'Pitch force (Nm)']

        fig, ax = plt.subplots(len(type), 1, figsize=(10, 6), dpi=150, sharex=True)
        plt.xticks(np.arange(self.starttime, self.endtime * 2, self.time * 2))
        plt.xlim([self.starttime, self.endtime])  # x軸の範囲
        plt.xlabel("Time (sec)")


        # ytics = [
        #     np.arange(-10, 10, 0.5),
        #     np.arange(-10, 10, 0.5),
        #     np.arange(-8.0, 8.0, 2.0),
        #     np.arange(-8.0, 8.0, 2.0),
        # ]
        # ylim = [
        #     [-1.5, 1.5],
        #     [-1.5, 1.5],
        #     [-8.0, 8.0],
        #     [-8.0, 8.0],
        # ]
        #
        # ylim_2nd = [
        #     [-1.0, 3.0],
        #     [-1.0, 3.0],
        #     [-3.0, 8.0],
        #     [-3.0, 8.0],
        # ]

        ytics = [
            np.arange(-20, 20, 2.0),
            np.arange(-12.0, 12.0, 4.0),
        ]
        ylim = [
            [-4.0, 2.0],
            [-8.0, 4.0],
        ]

        ylim_2nd = [
            [-2.0, 8.0],
            [-3.0, 15.0],
        ]

        model_mode = ['Follower', 'Leader', 'Altruistic', 'Selfish']

        delays = np.arange(0.0, 3.1, 0.1)

        for i, sub in enumerate(ax):
            data = self.data[num]
            sub.plot(data['time'][self.start_num:self.end_num:self.dec] - 20.0, data['label_' + type[i]][self.start_num:self.end_num:self.dec], label='Label (0.0s)', lw=4, color='Black', alpha=0.4)
            if data['output_delay'][0] != 0.0:
                sub.plot(data['time'][self.start_num:self.end_num:self.dec] - 20.0, data['label_' + type[i] + '_1'][self.start_num:self.end_num:self.dec], label='Label(' + str(data['output_delay'][0]) + 's)', lw=4, color='Blue', alpha=0.4)

            delay = data['output_delay'][0] / self.smp
            delay = int(delay) * 2
            delay = 0

            sc = sub.plot(data['time'][self.start_num:self.end_num:self.dec] - 20.0, data['pre_' + type[i]][self.start_num - delay:self.end_num - delay:self.dec],
                          label='Integration' , lw=2.0, color='red', alpha=1.0)

            ax2 = sub.twinx()
            for j in range(4):
                sub.plot(data['time'][self.start_num:self.end_num:self.dec] - 20.0, data['pre_' + type[i] + '_model_' + str(j+1)][self.start_num - delay:self.end_num - delay:self.dec],
                         label=model_mode[j], lw=1.0, alpha=0.4)

                ax2.plot(data['time'][self.start_num:self.end_num:self.dec] - 20.0, data['coefficient_' + type[i] + '_' + str(j+1)][self.start_num - delay:self.end_num - delay:self.dec],
                         label='coefficient_' + str(j+1), lw=1.0, alpha=0.4, linestyle='dashed')

            sub.set_ylabel(label[i])
            sub.legend(ncol=10, columnspacing=1, loc='upper left')
            sub.set_yticks(ytics[i])
            sub.set_ylim(ylim[i])
            ax2.set_ylim(ylim_2nd[i])

            ax2.set_ylabel('Coefficient')
            # sub.legend(ncol=10, columnspacing=1, loc='upper left')
            # sub.set_yticks(ytics[i])
            # ax2.set_ylim([0, 3.0])


        # # カラーバーを一つにまとめる
        # norm = Normalize(vmin=0.0, vmax=3.0)
        # cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax[0], location='top')
        # cbar.set_label('Forward time')  # カラーバーにラベルを付ける

        plt.tight_layout()

        os.makedirs('fig', exist_ok=True)
        plt.savefig("fig/Time series_Integration_single_" + '{:.1f}'.format(delays[num]) + ".pdf")
        plt.show()

    def graph_sub_integration(self):
        type = ['thm_r', 'thm_p', 'text_r', 'text_p']
        gain_type = ['thm_gain', 'thm_gain', 'text_gain', 'text_gain']
        label = ['Roll angle (rad)', 'Pitch angle (rad)', 'Roll force (Nm)', 'Pitch force (Nm)']

        type = ['thm_p', 'text_p']
        label = ['Pitch angle (rad)', 'Pitch force (Nm)']


        fig, ax = plt.subplots(len(type), 1, figsize=(10, 6), dpi=150, sharex=True)

        plt.xticks(np.arange(self.starttime, self.endtime * 2, self.time * 2))
        plt.xlim([self.starttime, self.endtime])  # x軸の範囲
        plt.xlabel("Time (sec)")

        # ytics = [
        #     np.arange(-10, 10, 0.5),
        #     np.arange(-10, 10, 0.5),
        #     np.arange(-8.0, 8.0, 2.0),
        #     np.arange(-8.0, 8.0, 2.0),
        # ]
        # ylim = [
        #     [-1.5, 1.5],
        #     [-1.5, 1.5],
        #     [-8.0, 8.0],
        #     [-8.0, 8.0],
        # ]

        ytics = [
            np.arange(-10, 10, 0.5),
            np.arange(-8.0, 8.0, 2.0),
        ]
        ylim = [
            [-1.5, 1.5],
            [-8.0, 8.0],
        ]

        # カラーマップを選択（'Blues'から'Reds'に変化する）
        cmap = plt.get_cmap('winter')

        for i, sub in enumerate(ax):
            data = self.data[0]
            sub.plot(data['time'][self.start_num:self.end_num:self.dec] - 20.0, data['label_' + type[i]][self.start_num:self.end_num:self.dec], label='Label', lw=4, color='Black', alpha=0.4)
            for k in range(len(self.data)):
                data = self.data[k]
                delay = data['output_delay'][0] / self.smp
                delay = int(delay)
                # print(delay)
                delay = 0

                color = cmap(k / (len(self.data) - 1))  # カラーマップの値を0から1に変化

                sc = sub.plot(data['time'][self.start_num:self.end_num:self.dec] - 20.0, data['pre_' + type[i]][self.start_num - delay:self.end_num - delay:self.dec],
                              label=str(data['output_delay'][0]) + 's', lw=0.5, color=color, alpha=0.4 - k * 0.005)
                # sub.plot(data['time'][self.start_num:self.end_num:self.dec] - 20.0, data['label_' + type[i] + '_1'][self.start_num + interval:self.end_num + interval:self.dec] / data[gain_type[i]][0],
                #          label=str(data['output_delay'][0]) + 's', lw=0.5, color=color, alpha=0.7 - k * 0.01)
                # for k in range(data['predictionNum'][0] - 1):
                #     sub.plot(data['time'][::self.dec], data['label_' + type[i] + '_' + str(k+2)][::self.dec], label=str('{:.1f}'.format((k+1)*0.3)) + 's')
                #     sub.plot(data['time'][::self.dec], data['pre_' + type[i] + '_' + str(k+2)][::self.dec], label=str('{:.1f}'.format((k+1)*0.3)) + 's')

            sub.set_ylabel(label[i])
            # sub.legend(ncol=10, columnspacing=1, loc='upper left')
            sub.set_yticks(ytics[i])
            sub.set_ylim(ylim[i])


        # カラーバーを一つにまとめる
        norm = Normalize(vmin=0.0, vmax=3.0)
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax[0], location='top')
        cbar.set_label('Forward time')  # カラーバーにラベルを付ける

        plt.tight_layout()

        os.makedirs('fig', exist_ok=True)
        plt.savefig("fig/Time series_" + self.mode + ".pdf")
        plt.show()

    def check_loss(self):
        data = self.data
        plt.figure(figsize=(5, 5), dpi=150)
        plt.plot(np.arange(data['epoch_num']) + 1, self.data['loss'])
        plt.show()

    def get_rmse(self, v1, v2):
        if v1.shape != v2.shape:
            exit(-1)
        if v1.shape[0] == 4:
            return np.sqrt(np.mean((v1 - v2) ** 2, axis=2))
        if v1.shape[0] == self.period * self.num:
            return np.sqrt(np.mean((v1 - v2) ** 2, axis=0))

        return np.sqrt(np.mean((v1 - v2) ** 2, axis=1))

    def get_stack(self):
        types = ['thm_r', 'thm_p', 'text_r', 'text_p']
        label_stack = []
        pre_stack = []
        for type in types:
            if self.mode == 'integration':
                label_stack.append(np.stack([self.data[_]['label_' + type + '_1'][self.start_num:self.end_num:self.dec] for _ in range(len(self.data))], axis=0))
                pre_stack.append(np.stack([self.data[_]['pre_' + type][self.start_num:self.end_num:self.dec] for _ in range(len(self.data))], axis=0))
            if self.mode == 'Follower':
                label_stack.append(np.stack([self.data[_]['label_' + type + '_1'][self.start_num:self.end_num:self.dec] for _ in range(len(self.data))], axis=0))
                pre_stack.append(np.stack([self.data[_]['pre_' + type + '_model_1'][self.start_num:self.end_num:self.dec] for _ in range(len(self.data))], axis=0))

            if self.mode == 'Leader':
                label_stack.append(np.stack([self.data[_]['label_' + type + '_1'][self.start_num:self.end_num:self.dec] for _ in range(len(self.data))], axis=0))
                pre_stack.append(np.stack([self.data[_]['pre_' + type + '_model_2'][self.start_num:self.end_num:self.dec] for _ in range(len(self.data))], axis=0))

            if self.mode == 'Altruistic':
                label_stack.append(np.stack([self.data[_]['label_' + type + '_1'][self.start_num:self.end_num:self.dec] for _ in range(len(self.data))], axis=0))
                pre_stack.append(np.stack([self.data[_]['pre_' + type + '_model_3'][self.start_num:self.end_num:self.dec] for _ in range(len(self.data))], axis=0))

            if self.mode == 'Selfish':
                label_stack.append(np.stack([self.data[_]['label_' + type + '_1'][self.start_num:self.end_num:self.dec] for _ in range(len(self.data))], axis=0))
                pre_stack.append(np.stack([self.data[_]['pre_' + type + '_model_4'][self.start_num:self.end_num:self.dec] for _ in range(len(self.data))], axis=0))

            if self.mode == 'predict' or self.mode == 'learn':
                label_stack.append(np.stack([self.data[_]['label_' + type + '_1'][self.start_num:self.end_num:self.dec] for _ in range(len(self.data))], axis=0))
                pre_stack.append(np.stack([self.data[_]['pre_' + type + '_1'][self.start_num:self.end_num:self.dec] for _ in range(len(self.data))], axis=0))

        # print(label_stack[0].shape)

        label_stack = np.stack([label_stack[_] for _ in range(len(types))], axis=0)
        pre_stack = np.stack([pre_stack[_] for _ in range(len(types))], axis=0)
        # print(label_stack.shape)

        return label_stack, pre_stack

    def error_show(self):
        label_stack, pre_stack = self.get_stack()
        error = (label_stack - pre_stack) ** 2

        print(error.shape)
        types = ['thm_r', 'thm_p', 'text_r', 'text_p']

        df_ = []
        df_melt = []
        for i in range(len(self.data)):
            df_.append(pd.DataFrame({
                types[0]: error[0][i],
                types[1]: error[1][i],
                types[2]: error[2][i],
                types[3]: error[3][i],
                'Forward': self.data[i]['output_delay'][0]
                                     }))

            # df_melt.append(pd.melt(df_[i]))
            # df_melt[i]['Forward'] = self.data[i]['output_delay'][0]

        df = pd.concat([_ for _ in df_], axis=0)
        # df.rename(columns={'variable': 'types', 'value': 'Error'}, inplace=True)
        # print(df)


        fig, ax = plt.subplots(4, 1, figsize=(5, 5), dpi=150, sharex=True)
        plt.xlim = [0, 3.0]

        if self.mode == 'predict':
            ylim = [
                [0, 0.5],
                [0, 0.5],
                [0, 10.0],
                [0, 10.0],
            ]
        if self.mode == 'learn':

            ylim = [
                [0, 0.002],
                [0, 0.002],
                [0, 0.5],
                [0, 0.5],
            ]

        if self.mode == 'integration':
            ylim = [
                [0, 0.2],
                [0, 0.2],
                [0, 2.0],
                [0, 2.0],
            ]

        for i, sub in enumerate(ax):
            sns.regplot(x='Forward', y=types[i], data=df, ax=sub,
                        scatter_kws={'s': .1, 'alpha': 0.01, 'color': 'red'}, line_kws={'lw': 1}, order=4)
            # sub.boxplot(error[i])
            # subsub.set_ylaberrorel(types[i * 2 + j])
            # subsub.set_xlabel('Time (sec)')
            # sub.set_xlim([0, 0.1])
            sub.set_ylim(ylim[i])

        plt.tight_layout()
        os.makedirs('fig', exist_ok=True)
        plt.savefig("fig/Error_" + self.mode + ".pdf")
        plt.show()

def plot_regplot(x, y, color, **kwargs):
    ax = plt.gca()
    ax.regplot(x, y, c=color, **kwargs)

def error_comparison(errors, labels):
    types = ['thm_r', 'thm_p', 'text_r', 'text_p']
    types = ['thm_p', 'text_p']

    delays = np.arange(0.0, 3.1, 0.1)

    df_ = []
    df_melt = []
    for i in range(len(errors[0][0])):
        for j, label in enumerate(labels):
            for k in range(len(types)):
                df_.append(pd.DataFrame({
                    'Error': errors[j][k * 2][i],
                    'Forward': delays[i],
                    'Model': label,
                    'Type': types[k],
                }))
            # df_.append(pd.DataFrame({
            #     types[0]: errors[j][0][i],
            #     types[1]: errors[j][1][i],
            #     types[2]: errors[j][2][i],
            #     types[3]: errors[j][3][i],
            #     'Forward': delays[i],
            #     'Model': label,
            # }))

        # df_melt.append(pd.melt(df_[i]))
        # df_melt[i]['Forward'] = self.data[i]['output_delay'][0]

    df = pd.concat([_ for _ in df_], axis=0)
    # df.rename(columns={'variable': 'types', 'value': 'Error'}, inplace=True)
    # print(df)


    # fig, ax = plt.subplots(4, 1, figsize=(5, 5), dpi=150, sharex=True)
    plt.xlim = [0, 3.0]

    ylim = [
        [0, 0.8],
        [0, 0.8],
        [0, 10.0],
        [0, 10.0],
    ]
    ylim = [
        [0, 0.5],
        [0, 5.0],
    ]

    # FacetGridを作成してグラフを設定
    # g = sns.FacetGrid(df, row='Type', hue="Model", height=4, aspect=1.2, sharey=False, sharex=False)
    # g.map(sns.regplot, "Forward", "Error")

    sns.set(font='Times New Roman', font_scale=2.0)
    sns.set_palette('Dark2', n_colors=6, desat=1.0)
    sns.set_style('ticks')
    sns.set_context("poster",
                    rc = {
                        "axes.linewidth": 0.0,
                        "legend.fancybox": False,
                        'pdf.fonttype': 42,
                        'xtick.direction': 'in',
                        'ytick.major.width': 1.0,
                        'xtick.major.width': 1.0,
                    })

    g = sns.lmplot(x='Forward', y='Error', data=df, row='Type', hue="Model",
                scatter=False, line_kws={'lw': 2.5}, order=4,
               height=6, aspect=2.5, sharey=False, sharex=True)
    # plt.legend('Model')

    for num,ax in enumerate(g.axes.flatten()):
        # col_var = ax.get_title().split(" ")[-1]  # get the column variable from the title
        ax.set_ylim(ylim[num])

    # sub.legend(ncol=10, columnspacing=1, loc='upper left')

    # for i, sub in enumerate(ax):
    #     for j, label in enumerate(labels):
    #         filtered_df = df.loc[df['Model'] == label]
    #         sns.regplot(x='Forward', y=types[i], data=filtered_df, ax=sub,
    #                     scatter_kws={'s': 0, 'alpha': 0.01, 'color': 'red'}, line_kws={'lw': 1}, order=4, label=label)
    #         sub.legend(ncol=10, columnspacing=1, loc='upper left')
    #     # sub.boxplot(error[i])
    #     # subsub.set_ylaberrorel(types[i * 2 + j])
    #     # subsub.set_xlabel('Time (sec)')
    #     sub.set_xlim([0, 3.0])
    #     sub.set_ylim(ylim[i])

    plt.tight_layout()
    os.makedirs('fig', exist_ok=True)
    plt.savefig("fig/Error_comparison.pdf")
    plt.show()




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    npz = Npz.NPZ()

    predict_data = [
        # '46c1c587678639dec35a09b1e5292cd7.npz',
        # '11362857831e94b1cc8660d08267d69a.npz',
        # 'deb5fabdaf398e79dc76dd27a4585340.npz',
        # 'c0da5c9ca8a57332e727608f80b3f06f.npz',
        # '86c0c1a13ae439db8866cbab73d8dd77.npz',
        # 'af25b4b6e4b5e8fe69a78b738a1777e2.npz',
        # '7b0f387c2e2418366458f4ba6bdc9347.npz',
        # '570f0fbc53084c9b17278a59f39ae7f8.npz',
        # 'b2ba23911dc989c90a493c7f31c977e4.npz',
        # '8d5ede416ef150533c09656200975a62.npz',
        # 'be247e5ea1b011027238fc07c50323be.npz',
        # '244b31a43f8b43f353fcfcc4544caec6.npz',
        # '83b996e4c5cec7f24629b02e44a9ce01.npz',
        # 'dbd3faad759a77100a3489861043dfd7.npz',
        # '7fdd9ec78c8667c0f5d5b2d4a19d205e.npz',
        # '15b9b29927c4654cbaa5583bc9628a17.npz',
        # '809f5b2858c51454de042c060b6253de.npz',
        # '9555eb10c210616153996a616e17f7f7.npz',
        # 'c651b1ef423bff0dd0b8588fdff92245.npz',
        # '9617a18a2622ad433bfbe75ecbf0b7b1.npz',
        # '43d1710ca2a6e894c5315397c11acbba.npz',
        # '01b01a28d4cac3298aa951d8574d6084.npz',
        # 'fc8e2c367c86fc6191c6e943d3f51472.npz',
        # '5226254875beb0cf8c3ea93f0bb8ab2a.npz',
        # '865c9db833c9cb401d9256324be6c0d1.npz',
        # 'f0a16fca7e42d0da12eac9e720878d4b.npz',
        # 'fdd5abd3c2b44beadd47e20709a0a1bd.npz',
        # '50c493339d268b7ca15d41dd6d56c219.npz',
        # 'ccf7046fbbdf91a930de232f10ffe33b.npz',
        # '5f9d874803eb8dc8a4c90b1991130a4f.npz',
        # '476a1979cf2932db865d46691e7e8410.npz',

        'de78a663568569942b7f19b517f81b8c.npz',
        'cc3a84b8a273d187326ad360e4282e9b.npz',
        '9a8488442b93c4f86c2de750c41b6919.npz',
        '7b5dd134f7c95ccf86ee4d8219257c24.npz',
        '28fc125a937eb06696d22843537dd86b.npz',
        'b568b8d1b4b9ff19be4ca0ba54082b4a.npz',
        '5e3da9e6586b3de21d62aec9f084539c.npz',
        'c3801b6f4c879d143c1a6f4287af6b07.npz',
        'acf56c476c13b0e1f0af54097fe0d011.npz',
        '1b0c31976ec1e46450c3f93bb327a098.npz',
        '68ee3895e86a66f9c41ac800ba558152.npz',
        '12ac9cfa108812ca37e1030756af42a1.npz',
        '2a962992378acdfb1af02c46d51888b4.npz',
        '1e4bb698151352a5a78401c2ce3d68c6.npz',
        '8bd2202820de53e247885914e0631812.npz',
        '4ef2f16082af27261e962699725fc1f7.npz',
        '7332380af50f560fe5a9a1a3b5cac869.npz',
        'e211b3fbb9ce8ac352c4fcd7c4ac80b5.npz',
        '17af0537993a36670648507273d2b575.npz',
        'af26e2cfb7d9a2460818f6104bab7b28.npz',
        '56d0f420e8f3de667ae63910f19fc54c.npz',
        'dda208c90d513a393b3fe33e3cdda571.npz',
        '6253736364d08a8b392ef011a967db2a.npz',
        '9d099691b503c28c92fbe9eb435d63c6.npz',
        '076edc78bbab935227abfbbf2b4326d2.npz',
        '94b0f5e87abbe6f79726877d77fd3726.npz',
        '5227d00a886a3e0fe73ec6a2412f25da.npz',
        '4c521adb8071b1e0f0ac3ebdf9c9ae00.npz',
        'c2acc2fcccdb65ddaf7e39d7105866cb.npz',
        'dc149bedb296f0de071bd8e15e307125.npz',
        'ffe1c64dc5ad37a7d919bce6fc6bbd58.npz',

    ]
    learn_data = [
        # 'cf98173673a1aa4a302a384f6b5ccfb7.npz',
        # '6a11015bf4185ea2294c77a46623f261.npz',
        # '4afa8943712b7ed12d4e00753a939257.npz',
        # 'c1f8783463bdd62a72bc87e5a4655c5b.npz',
        # 'b883dae1a1927617db45444606289ed8.npz',
        # 'b0ff82961c5d261240b6e142b23498c4.npz',
        # '086d25b8b805038b7327809520183f37.npz',
        # '3b85a637b12c98aade42ede9212ed54f.npz',
        # '612535211fd0ceb09067706a648dae56.npz',
        # 'a14def9f782210d0af011261ce30fe9a.npz',
        # 'e7b90e8973214a34110882527ae1c438.npz',
        # 'd0fb5f14c9fb439b63d6d04097ecd735.npz',
        # '1e2a452961f37ef283c5287172005223.npz',
        # 'f674124644eea75908af655a4a444236.npz',
        # '79a7b5494644123358ca7bba997068e7.npz',
        # 'ee7e6f1306e655dd8f5af6ca7deec77f.npz',
        # '612c45d5f5c7ebc57f4710ea1dc13367.npz',
        # '4c1f1a4a75277f91bff3b325bfaea24b.npz',
        # '606d0b76bdcbcca4756a7bade38f4a3e.npz',
        # 'cd8c2dfa6032e185778eda67aae9d539.npz',
        # '462fff161e789d70ada05b8014018c63.npz',
        # '97be301cf66a25c03faf040a66ccaa34.npz',
        # '8f1b7b5995f6b688ab1e6aaaa3729ddc.npz',
        # 'b7d6100ca2491a0d86594fee1e4f6f34.npz',
        # '1ddac49154ac33e3190fd210468b5d82.npz',
        # '15ad5dc3d58ca2a25af320bb6f09bc91.npz',
        # '356e8f1365dd258bf6f4f6f79f262826.npz',
        # '77b7db9c69a03aba31d526bcc60a3d57.npz',
        # 'b641380269687b808142a2073b1124fa.npz',
        # 'ed7042213b831f87a6b7bc85ce389510.npz',
        # 'cb2cd6be1bc50730cb2de6fcaf7891d3.npz',

        '2ec3b13118cf4312d73a205814994041.npz',
        '5ea9e60651e61973b64c6ba64d5346a6.npz',
        '1b0dbdd3c4f12d32eb878a03dee46729.npz',
        'f5c4bcc349a7a2e0c507daefc72aab95.npz',
        '62dba6028f548dbb982a3cacb61bcfa7.npz',
        '5c3e85839bf7bb80be7590ac93df48eb.npz',
        'e5d1c5cb9c28dbd1af99d58d1e286867.npz',
        '3183c5718f5f2bd1aeeb4822c12d24cd.npz',
        'a58ac0ae66653efaf248ea7720b476f3.npz',
        '51d1b29e6e9a7226d25b3953d468d1d1.npz',
        '384245f9e29754c30b03256289ae045d.npz',
        '94ea3e7537549e4196a067437d8433be.npz',
        '34f6094e3156b7aa588c05cda17b7584.npz',
        '101ab3f549f1afb5734d79478bf8958d.npz',
        '1b6be6d3030654932dbf9456d6867a23.npz',
        '63a3804f66a2ce2aba929bfcd95bd774.npz',
        '89cd961abe5dbc8bb5dfd7358138542b.npz',
        '8a140930a7cf8e881dfe732c403ad8b5.npz',
        '90ee44d42edecdd0a92a33cac7828928.npz',
        '31f0890ad5c61cb7801c2cd790b3b17b.npz',
        'cda5de8d6b501087a90802e7d0f3676c.npz',
        'fccabb4b34f9b7605c09762deedbcad7.npz',
        '9b04337d50cc8f937c3d4c1ba1c16ebf.npz',
        '9e2b10dc48b958c7164599cc05dd596a.npz',
        '3cb464bd3f6ef6d13ff423bcfb5eb154.npz',
        'accde277ae045b0ec91b1937f664f565.npz',
        'cf034c1caeab0398b377ee12f62203cf.npz',
        '54b9644ac86f0dc03f0e4cf2cca1d330.npz',
        '7142f5d7971e4bf56add25bdcc261051.npz',
    ]

    Integration_data = [
        'k.kobayashi_0.000000.npz',
        'k.kobayashi_0.100000.npz',
        'k.kobayashi_0.200000.npz',
        'k.kobayashi_0.300000.npz',
        'k.kobayashi_0.400000.npz',
        'k.kobayashi_0.500000.npz',
        'k.kobayashi_0.600000.npz',
        'k.kobayashi_0.700000.npz',
        'k.kobayashi_0.800000.npz',
        'k.kobayashi_0.900000.npz',
        'k.kobayashi_1.000000.npz',
        'k.kobayashi_1.100000.npz',
        'k.kobayashi_1.200000.npz',
        'k.kobayashi_1.300000.npz',
        'k.kobayashi_1.400000.npz',
        'k.kobayashi_1.500000.npz',
        'k.kobayashi_1.600000.npz',
        'k.kobayashi_1.700000.npz',
        'k.kobayashi_1.800000.npz',
        'k.kobayashi_1.900000.npz',
        'k.kobayashi_2.000000.npz',
        'k.kobayashi_2.100000.npz',
        'k.kobayashi_2.200000.npz',
        'k.kobayashi_2.300000.npz',
        'k.kobayashi_2.400000.npz',
        'k.kobayashi_2.500000.npz',
        'k.kobayashi_2.600000.npz',
        'k.kobayashi_2.700000.npz',
        'k.kobayashi_2.800000.npz',
        'k.kobayashi_2.900000.npz',
        'k.kobayashi_3.000000.npz',
    ]

    predict_npz = npz.select_load(predict_data)
    learn_npz = npz.select_load(learn_data)

    analysis_predict = Analysis(predict_npz, 'predict')
    analysis_learn = Analysis(learn_npz, 'learn')
    analysis_integration = Analysis(npz.select_load(Integration_data), 'integration')
    analysis_Follower = Analysis(npz.select_load(Integration_data), 'Follower')
    analysis_Leader = Analysis(npz.select_load(Integration_data), 'Leader')
    analysis_Altruistic = Analysis(npz.select_load(Integration_data), 'Altruistic')
    analysis_Selfish = Analysis(npz.select_load(Integration_data), 'Selfish')

    # analysis_learn.check_loss()

    # analysis_predict.graph_sub()
    # analysis_learn.graph_sub()
    # analysis_integration.graph_sub_single_integration(30)
    # analysis_integration.graph_sub_integration()
    # analysis_integration.graph_sub_single_gif(30)
    # analysis_Follower.graph_sub()
    # analysis_Leader.graph_sub()
    # analysis_Altruistic.graph_sub()
    analysis_Selfish.graph_sub()
    

    # analysis_learn.error_show()
    # analysis_predict.error_show()
    # analysis_integration.error_show()




    # stacks = [
    #     analysis_predict.get_stack(),
    #     analysis_Follower.get_stack(),
    #     analysis_Leader.get_stack(),
    #     analysis_Altruistic.get_stack(),
    #     analysis_Selfish.get_stack(),
    #     analysis_integration.get_stack(),
    # ]
    #
    # errors = []
    # for stack in stacks:
    #     errors.append((stack[0] - stack[1]) ** 2)
    # labels = [
    #     'Normal',
    #     'Follower',
    #     'Leader',
    #     'Altruistic',
    #     'Selfish',
    #     'Integration',
    # ]
    #
    # error_comparison(errors, labels)