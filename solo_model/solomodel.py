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
        self.dec = 100


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
        df = pd.DataFrame({'time': [0.0],
                           'val': [0.0],
                           'type': ['0'],
                           'lorp':['0']})

        value = np.array([
            self.data['label_pre_thm_r'][::self.dec],
            self.data['label_pre_thm_p'][::self.dec],
            self.data['label_pre_text_r'][::self.dec],
            self.data['label_pre_text_p'][::self.dec],
            self.data['pre_thm_r'][::self.dec],
            self.data['pre_thm_p'][::self.dec],
            self.data['pre_text_r'][::self.dec],
            self.data['pre_text_p'][::self.dec]

        ])

        # 値だけのデータフレーム
        df_val = pd.DataFrame(
            columns=['val'],
            data = value.flatten()
        )

        # print(df_val)

        # 時間だけのデータフレーム
        pre_t = self.data['pre_time'][::self.dec].copy()
        for i in range(7):
            pre_t = np.hstack((pre_t, self.data['pre_time'][::self.dec]))

        df_time = pd.DataFrame(
            columns=['time'],
            data = pre_t
        )

        # print(df_time)

        types = ["thm_r", "thm_p", "text_r", "text_p"]
        type1 = []
        for j in range(2):
            for i in types:
                type1.append([i] * self.data['pre_time'][::self.dec].size)

        type1 = list(itertools.chain.from_iterable(type1))
        # print(len(type1))
        df_types = pd.DataFrame(
            columns=["type"],
            data = type1
        )

        lorps = ["Label", "Prediction"]
        type2 = []
        for i in lorps:
            for k in range(4):
                type2.append([i] * self.data['pre_time'][::self.dec].size)

        type2 = list(itertools.chain.from_iterable(type2))
        # print(len(type2))
        df_lorp = pd.DataFrame(
            columns=["lorp"],
            data=type2
        )

        df = pd.concat([df_time, df_val, df_types, df_lorp], axis=1)

        # print(df)

        plt.rcParams['pdf.fonttype'] = 42  # PDFにフォントを埋め込むためのパラメータ
        sns.relplot(data=df, row='type', x='time', y='val', hue='lorp', kind='line',height=2, aspect=3)
        plt.ylim(-1.5, 1.5)
        # plt.xlim(95, 100)
        # plt.tight_layout()
        # plt.legend()
        # plt.savefig('text_compare.pdf')
        # plt.savefig("data.png")

        plt.show()

    def analyze_force(self):
        df = pd.DataFrame({'time': [0.0],
                           'val': [0.0],
                           'type': ['0'],
                           'lorp':['0']})

        value = np.array([
            self.data['label_pre_text_r'][::self.dec],
            self.data['label_pre_text_p'][::self.dec],
            self.data['pre_text_r'][::self.dec],
            self.data['pre_text_p'][::self.dec]

        ])

        # 値だけのデータフレーム
        df_val = pd.DataFrame(
            columns=['val'],
            data = value.flatten()
        )

        # print(df_val)

        # 時間だけのデータフレーム
        pre_t = self.data['pre_time'][::self.dec].copy()
        for i in range(3):
            pre_t = np.hstack((pre_t, self.data['pre_time'][::self.dec]))

        df_time = pd.DataFrame(
            columns=['time'],
            data = pre_t
        )

        # print(df_time)

        types = ["text_r", "text_p"]
        type1 = []
        for j in range(2):
            for i in types:
                type1.append([i] * self.data['pre_time'][::self.dec].size)

        type1 = list(itertools.chain.from_iterable(type1))
        # print(len(type1))
        df_types = pd.DataFrame(
            columns=["type"],
            data = type1
        )

        lorps = ["Label", "Prediction"]
        type2 = []
        for i in lorps:
            for k in range(2):
                type2.append([i] * self.data['pre_time'][::self.dec].size)

        type2 = list(itertools.chain.from_iterable(type2))
        # print(len(type2))
        df_lorp = pd.DataFrame(
            columns=["lorp"],
            data=type2
        )

        df = pd.concat([df_time, df_val, df_types, df_lorp], axis=1)

        # print(df)

        plt.rcParams['pdf.fonttype'] = 42  # PDFにフォントを埋め込むためのパラメータ
        sns.relplot(data=df, row='type', x='time', y='val', hue='lorp', kind='line',height=2, aspect=3)
        plt.ylim(-1.5, 1.5)
        # plt.xlim(95, 100)
        # plt.tight_layout()
        # plt.legend()
        # plt.savefig('text_compare.pdf')
        # plt.savefig("data.png")

        plt.show()


    def check_loss(self):
        plt.plot(np.arange(self.data['epoch_loss'].size) + 1, self.data['epoch_loss'])
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.tight_layout()

        # plt.savefig("loss.png")
        plt.show()

        plt.plot(np.arange(self.data['running_loss'].size) + 1, self.data['running_loss'])
        plt.show()

    def check_ball(self):
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.default'] = 'regular'
        plt.rcParams['xtick.top'] = 'True'
        plt.rcParams['ytick.right'] = 'True'
        # mpl.rcParams['axes.grid'] = 'True'
        plt.rcParams['xtick.direction'] = 'in'  # x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'  # y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['xtick.major.width'] = 0.5  # x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 0.5  # y軸主目盛り線の線幅
        plt.rcParams['font.size'] = 6  # フォントの大きさ
        plt.rcParams['axes.linewidth'] = 0.5  # 軸の線幅edge linewidth。囲みの太さ
        plt.rcParams['lines.linewidth'] = 0.5  # 軸の線幅edge linewidth。囲みの太さ

        plt.rcParams["legend.fancybox"] = False  # 丸角
        plt.rcParams["legend.framealpha"] = 1.0  # 透明度の指定、0で塗りつぶしなし
        # mpl.rcParams["legend.edgecolor"] = 'black'  # edgeの色を変更
        plt.rcParams["legend.handlelength"] = 2  # 凡例の線の長さを調節
        plt.rcParams["legend.labelspacing"] = 0.1  # 垂直方向（縦）の距離の各凡例の距離
        plt.rcParams["legend.handletextpad"] = .3  # 凡例の線と文字の距離の長さ
        # mpl.rcParams["legend.frameon"] = False
        plt.rcParams["legend.facecolor"] = 'white'

        plt.rcParams["legend.markerscale"] = 2  # 点がある場合のmarker scale
        plt.rcParams['axes.xmargin'] = '0'  # '.05'
        plt.rcParams['axes.ymargin'] = '0'
        plt.rcParams['savefig.facecolor'] = 'None'
        plt.rcParams['savefig.edgecolor'] = 'None'
        # mpl.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['pdf.fonttype'] = 42  # PDFにフォントを埋め込むためのパラメータ



        fig, (x, y) = plt.subplots(2, 1, figsize=(5, 5), dpi=150, sharex=True)

        plt.xlabel("Time[sec]")

        x.plot(self.data['pre_time'][::self.dec], self.data['pre_ball_x'][::self.dec], label='predicted')
        x.plot(self.data['pre_time'][::self.dec], self.data['label_pre_ball_x'][::self.dec], label='Actuality')
        x.set_ylabel('x-axis Position (m)')
        x.legend(ncol=2, columnspacing=1, loc='upper left')
        x.set_yticks(np.arange(-10, 10, 0.1))
        x.set_ylim([-0.2, 0.2])  # y軸の範囲

        y.plot(self.data['pre_time'][::self.dec], self.data['pre_ball_y'][::self.dec], label='predicted')
        y.plot(self.data['pre_time'][::self.dec], self.data['label_pre_ball_y'][::self.dec], label='Actuality')
        y.set_ylabel('y-axis Position (m)')
        y.legend(ncol=2, columnspacing=1, loc='upper left')
        y.set_yticks(np.arange(-10, 10, 0.1))
        y.set_ylim([-0.2, 0.2])  # y軸の範囲

        # plt.savefig("far.png")

        plt.show()

    def recalc_ball_movement(self):
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.default'] = 'regular'
        plt.rcParams['xtick.top'] = 'True'
        plt.rcParams['ytick.right'] = 'True'
        # mpl.rcParams['axes.grid'] = 'True'
        plt.rcParams['xtick.direction'] = 'in'  # x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'  # y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['xtick.major.width'] = 0.5  # x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 0.5  # y軸主目盛り線の線幅
        plt.rcParams['font.size'] = 6  # フォントの大きさ
        plt.rcParams['axes.linewidth'] = 0.5  # 軸の線幅edge linewidth。囲みの太さ
        plt.rcParams['lines.linewidth'] = 0.5  # 軸の線幅edge linewidth。囲みの太さ

        plt.rcParams["legend.fancybox"] = False  # 丸角
        plt.rcParams["legend.framealpha"] = 1.0  # 透明度の指定、0で塗りつぶしなし
        # mpl.rcParams["legend.edgecolor"] = 'black'  # edgeの色を変更
        plt.rcParams["legend.handlelength"] = 2  # 凡例の線の長さを調節
        plt.rcParams["legend.labelspacing"] = 0.1  # 垂直方向（縦）の距離の各凡例の距離
        plt.rcParams["legend.handletextpad"] = .3  # 凡例の線と文字の距離の長さ
        # mpl.rcParams["legend.frameon"] = False
        plt.rcParams["legend.facecolor"] = 'white'

        plt.rcParams["legend.markerscale"] = 2  # 点がある場合のmarker scale
        plt.rcParams['axes.xmargin'] = '0'  # '.05'
        plt.rcParams['axes.ymargin'] = '0'
        plt.rcParams['savefig.facecolor'] = 'None'
        plt.rcParams['savefig.edgecolor'] = 'None'
        # mpl.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['pdf.fonttype'] = 42  # PDFにフォントを埋め込むためのパラメータ



        fig, (x, y) = plt.subplots(2, 1, figsize=(5, 5), dpi=150, sharex=True)

        plt.xlabel("Time[sec]")

        x.plot(self.data['learn_time'][::self.dec], self.data['label_train_ball_x'][::self.dec], label='Actuality')
        x.plot(self.data['learn_time'][::self.dec], self.data['train_ball_x_recalc'][::self.dec], label='recalc')
        x.set_ylabel('x-axis Position (m)')
        x.legend(ncol=2, columnspacing=1, loc='upper left')
        x.set_yticks(np.arange(-10, 10, 0.1))
        x.set_ylim([-0.2, 0.2])  # y軸の範囲

        y.plot(self.data['learn_time'][::self.dec], self.data['label_train_ball_y'][::self.dec], label='Actuality')
        y.plot(self.data['learn_time'][::self.dec], self.data['train_ball_y_recalc'][::self.dec], label='recalc')
        y.set_ylabel('y-axis Position (m)')
        y.legend(ncol=2, columnspacing=1, loc='upper left')
        y.set_yticks(np.arange(-10, 10, 0.1))
        y.set_ylim([-0.2, 0.2])  # y軸の範囲

        # plt.savefig("far.png")

        plt.show()
