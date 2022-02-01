import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import math

class PCT:
    def graph_sub(self, data):
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

        fig, (thm, wm, am, iad, text) = plt.subplots(5, 1, figsize=(10, 10), sharex=True)

        # plt.xlim([20, 20.01])  # x軸の範囲
        # plt.xlim([0.28, 0.89])  # x軸の範囲
        plt.xlabel("Time[sec]")

        thm.plot(data['time'], data['i2_r_thm'], label='Interface2')
        thm.plot(data['time'], data['i1_r_thm'], label='Interface1')

        # thm.plot(self.data['time'], self.data['i3_p_thm'], label='Interface3')
        # thm.plot(self.data['time'], self.data['i4_p_thm'], label='Interface4')
        thm.set_ylabel('Position [rad]')
        thm.legend()
        thm.set_yticks(np.arange(-10, 10, 0.5))
        thm.set_ylim([-1.5, 1.5])  # y軸の範囲

        wm.plot(data['time'], data['i2_r_wm'], label='Interface2')
        wm.plot(data['time'], data['i1_r_wm'], label='Interface1')
        # wm.plot(self.data['time'], self.data['i3_p_wm'], label='Interface3')
        # wm.plot(self.data['time'], self.data['i4_p_wm'], label='Interface4')
        wm.set_ylabel(r'Velocity [rad/s]')
        wm.legend()
        wm.set_yticks(np.arange(-10, 10, 1))
        wm.set_ylim([-5, 5])  # y軸の範囲

        am.plot(data['time'], data['i2_r_am'], label='Interface2')
        am.plot(data['time'], data['i1_r_am'], label='Interface1')

        am.set_ylabel(r'Acceleration [rad/s$^2$]')
        am.legend()
        am.set_yticks(np.arange(-1000, 1000, 250))
        am.set_ylim([-500, 500])  # y軸の範囲

        iad.plot(data['time'], data['i2_r_iad'], label='Interface2')
        iad.plot(data['time'], data['i1_r_iad'], label='Interface1')
        # iad.plot(self.data['time'], self.data['i3_p_iad'], label='Interface3')
        # iad.plot(self.data['time'], self.data['i4_p_iad'], label='Interface4')
        iad.set_ylabel('Current [A]')
        iad.legend()
        iad.set_yticks(np.arange(-100, 100, 1.6))
        iad.set_ylim([-4.8, 4.8])  # y軸の範囲

        text.plot(data['time'], data['i2_r_text'], label='Interface2')
        text.plot(data['time'], data['i1_r_text'], label='Interface1')
        # text.plot(self.data['time'], self.data['i3_p_text'], label='Interface3')
        # text.plot(self.data['time'], self.data['i4_p_text'], label='Interface4')
        text.set_ylabel('Reaction torque[Nm]')
        text.legend()
        text.set_yticks(np.arange(-6.0, 6.0, 2.0))
        text.set_ylim([-4.0, 4.0])  # y軸の範囲

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

        # wm_sm.plot(self.data['time'], self.data['i1_p_count_diff'], label='Interface1')
        # wm_sm.plot(self.data['time'], self.data['i2_p_count_diff'], label='Interface2')
        # # wm_sm.set_ylabel(r'Velocity [rad/s]')
        # wm_sm.set_ylabel(r'Ms')
        # wm_sm.legend()
        # wm_sm.set_yticks(np.arange(-10, 10, 1))
        # wm_sm.set_ylim([-10, 10])  # y軸の範囲

        plt.tight_layout()
        # plt.savefig("2022-01-22_nonFront_p.png")
        plt.show()

    def task_show(self, data):
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

        plt.plot(self.data['time'], self.data['ballx'], label='ballx')
        plt.plot(self.data['time'], self.data['bally'], label='bally')

        plt.plot(self.data['time'], self.data['targetx'], label='targetx')
        plt.plot(self.data['time'], self.data['targety'], label='targety')


        plt.ylabel(r'Position [m]')
        plt.legend()
        plt.yticks(np.arange(-4, 4, 0.1))
        plt.ylim([-0.4, 0.4])  # y軸の範囲
        plt.xlim([self.data['starttime'], self.data['endtime']])  # x軸の範囲
        plt.xlabel("Time[sec]")

        plt.tight_layout()
        # plt.savefig("First_time_target_movement.png")
        plt.show()