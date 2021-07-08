# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib as plt
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt


class MotorAnalysis:

    def npz_load(self):
        self.npz = np.load('50%_control_wo_dob.npz')
        # self.npz = np.load('50%_control_w_dob.npz')

    def CsvOut(self):
        df1 = pd.DataFrame({
            'time': self.npz['time'],
            'p_iq': self.npz['p_iq'],
            'p_am': self.npz['p_am'],
            'p_wm': self.npz['p_wm'],
            'p_thm': self.npz['p_thm'],
            'r_iq': self.npz['r_iq'],
            'r_am': self.npz['r_am'],
            'r_wm': self.npz['r_wm'],
            'r_thm': self.npz['r_thm'],
        })

        df1.to_csv("motor_analysis_StepWave_DOB_gd300.csv", index=False)

    def graph_sub(self):
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.default'] = 'regular'
        plt.rcParams['xtick.top'] = 'True'
        plt.rcParams['ytick.right'] = 'True'
        # plt.rcParams['axes.grid'] = 'True'
        plt.rcParams['xtick.direction'] = 'in'  # x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'  # y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['xtick.major.width'] = 1.0  # x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 1.0  # y軸主目盛り線の線幅
        plt.rcParams['font.size'] = 12  # フォントの大きさ
        plt.rcParams['axes.linewidth'] = 1.0  # 軸の線幅edge linewidth。囲みの太さ

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

        fig, (top, mid1, mid2, bot) = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

        plt.xlim([2, 25])  # x軸の範囲
        # plt.xlim([0.28, 0.89])  # x軸の範囲
        plt.xlabel("Time[sec]")

        # top.plot(self.npz['time'], self.npz['p_am'] * 0.1376386, label = 'NormTorque')
        # top.plot(self.npz['time'], self.npz['p_iq'])
        top.plot(self.npz['time'], self.npz['r_iq'])
        top.set_ylabel('Torque[Nm]')
        # top.legend()
        # top.set_yticks(np.arange(-2, 2, 0.5))
        top.set_ylim([-4, 4])  # y軸の範囲

        # mid1.plot(self.npz['time'], self.npz['p_am'])
        mid1.plot(self.npz['time'], self.npz['r_am'])
        mid1.set_ylabel('Acceleration[rad/s^2]')
        # mid1.legend()
        # mid1.set_yticks(np.arange(-400, -100, 50))
        mid1.set_ylim([-100.0, 100.0])  # y軸の範囲

        # mid2.plot(self.npz['time'], self.npz['p_wm'], label = 'Motor')
        # mid2.plot(self.npz['time'], self.npz['p_wmref'], label = 'Command')
        mid2.plot(self.npz['time'], self.npz['r_wm'], label='Motor')
        mid2.plot(self.npz['time'], self.npz['r_wmref'], label='Command')
        mid2.set_ylabel('Velocity[rad/s]')
        mid2.legend()
        # mid2.set_yticks(np.arange(-400, -100, 50))
        mid2.set_ylim([-3.0, 3.0])  # y軸の範囲

        # bot.plot(self.npz['time'], self.npz['p_thm'], label = 'Motor')
        # bot.plot(self.npz['time'], self.npz['p_thmref'], label = 'Command')
        bot.plot(self.npz['time'], self.npz['r_thm'], label='Motor')
        bot.plot(self.npz['time'], self.npz['r_thmref'], label='Command')
        bot.set_ylabel('Angle [rad]')
        bot.legend()
        # bot.set_yticks(np.arange(-200, 200, 50))
        bot.set_ylim([-1, 1])  # y軸の範囲

        # plt.tight_layout()
        plt.savefig("50%_control_wo_dob_roll.png")
        plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    MA = MotorAnalysis()
    MA.npz_load()
    # MA.CsvOut()
    MA.graph_sub()