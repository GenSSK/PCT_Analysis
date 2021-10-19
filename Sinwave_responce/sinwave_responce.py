# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
# plt.switch_backend('Qt5Agg')

class Iden:
    def npz_load(self):
        # self.data = np.load('cpr2000_dob200_sin.npz')
        # self.data = np.load('cpr2000_dob500_sin.npz')
        # self.data = np.load('cpr10000_dob500_sin_diff.npz')
        self.data = np.load('cpr4000000_dob1000_sin_diff.npz')
        # self.data = np.load('cpr4000000_dob1000_sin_smethod.npz')
        # self.data = np.load('normal_sin.npz')

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

        # print(self.data['i1_dob_gain'])

        fig, (thr, wm, am, tdish) = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

        plt.xlim([3, 4])  # x軸の範囲
        # plt.xlim([0.28, 0.89])  # x軸の範囲
        plt.xlabel("Time[sec]")

        thr.plot(self.data['time'], self.data['i1_p_thmref'], label = 'Target')
        thr.plot(self.data['time'], self.data['i1_p_thm'], label = 'Actual')
        thr.set_ylabel('Position [rad]')
        thr.legend()
        thr.set_yticks(np.arange(-10, 10, 0.5))
        thr.set_ylim([-1, 1])  # y軸の範囲

        wm.plot(self.data['time'], self.data['i1_p_wmref'], label='Target')
        wm.plot(self.data['time'], self.data['i1_p_wm'], label='Actual')
        wm.set_ylabel(r'Velocity [rad/s]')
        wm.legend()
        wm.set_yticks(np.arange(-10, 10, 1))
        wm.set_ylim([-5, 5])  # y軸の範囲

        am.plot(self.data['time'], self.data['i1_p_amref'], label='Target')
        am.plot(self.data['time'], self.data['i2_p_am'], label='Actual')
        am.set_ylabel(r'Acceleration [rad/s$^2$]')
        am.legend()
        am.set_yticks(np.arange(-1000, 1000, 10))
        am.set_ylim([-10, 10])  # y軸の範囲

        # text.plot(self.data['time'], self.data['i1_p_text'])
        # text.set_ylabel('Reaction torque[Nm]')
        # text.set_yticks(np.arange(-100, 100, 5))
        # text.set_ylim([-10, 10])  # y軸の範囲

        tdish.plot(self.data['time'], self.data['i1_p_tdish'])
        tdish.set_ylabel('Disturbance torque[Nm]')
        tdish.set_yticks(np.arange(-100, 100, 5))
        tdish.set_ylim([-10, 10])  # y軸の範囲

        # ccnt.plot(self.data['time'], self.data['i1_check_count'], label='Interface1')
        # ccnt.plot(self.data['time'], self.data['i2_check_count'], label='Interface2')
        # ccnt.set_ylabel('Check count')
        # ccnt.legend()
        # ccnt.set_yticks(np.arange(-10, 20, 2))
        # ccnt.set_ylim([0, 10])  # ycnt

        # cnt.stem(self.data['time'], self.data['i1_p_count'], label='Interface1', linefmt="b-", basefmt="None", markerfmt="b,")
        # cnt.stem(self.data['time'], self.data['i2_p_count'], label='Interface2', linefmt="r-", basefmt="None", markerfmt="r,")
        # cnt.set_ylabel('Deviation count')
        # cnt.legend()
        # cnt.set_yticks(np.arange(-1000, 1000, 1))
        # cnt.set_ylim([-2, 2])  # ycnt

        # wm_sm.plot(self.data['time'], self.data['i1_p_add_data1'], label='Interface1')
        # wm_sm.plot(self.data['time'], self.data['i2_p_add_data1'], label='Interface2')
        # # wm_sm.set_ylabel(r'Velocity [rad/s]')
        # wm_sm.set_ylabel(r'Ms')
        # wm_sm.legend()
        # wm_sm.set_yticks(np.arange(-10, 10, 5))
        # wm_sm.set_ylim([-10, 10])  # y軸の範囲

        plt.tight_layout()
        plt.savefig("cpr4000000_dob1000_sin_enlarge.png")
        # plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ID = Iden()
    ID.npz_load()
    ID.graph_sub()