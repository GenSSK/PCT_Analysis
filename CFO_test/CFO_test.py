import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

class CFO:
    def __init__(self, data, cfoo):
        self.data = data
        self.cfoo = cfoo

        self.smp = 0.0001  # サンプリング時間
        self.time = self.data['duringtime']  # ターゲットの移動時間
        self.period = int((self.data['tasktime'] - 9.0) / self.time)  # 回数
        self.num = int(self.time / self.smp)  # 1ピリオドにおけるデータ数
        self.start_num = int(self.data['starttime'] / self.smp)
        self.end_num = int(self.data['endtime'] / self.smp)
        self.nn_read_flag = False
        self.join = self.cfoo['join'][0]

    def graph_sub(self):
        mpl.rcParams['font.family'] = 'Times New Roman'
        mpl.rcParams['mathtext.default'] = 'regular'
        mpl.rcParams['xtick.top'] = 'True'
        mpl.rcParams['ytick.right'] = 'True'
        # mpl.rcParams['axes.grid'] = 'True'
        mpl.rcParams['xtick.direction'] = 'in'  # x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        mpl.rcParams['ytick.direction'] = 'in'  # y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        mpl.rcParams['xtick.major.width'] = 0.5  # x軸主目盛り線の線幅
        mpl.rcParams['ytick.major.width'] = 0.5  # y軸主目盛り線の線幅
        mpl.rcParams['font.size'] = 6  # フォントの大きさ
        mpl.rcParams['axes.linewidth'] = 0.5  # 軸の線幅edge linewidth。囲みの太さ
        mpl.rcParams['lines.linewidth'] = 0.5  # 軸の線幅edge linewidth。囲みの太さ


        mpl.rcParams["legend.fancybox"] = False  # 丸角
        mpl.rcParams["legend.framealpha"] = 1.0  # 透明度の指定、0で塗りつぶしなし
        # mpl.rcParams["legend.edgecolor"] = 'black'  # edgeの色を変更
        mpl.rcParams["legend.handlelength"] = 2  # 凡例の線の長さを調節
        mpl.rcParams["legend.labelspacing"] = 0.1  # 垂直方向（縦）の距離の各凡例の距離
        mpl.rcParams["legend.handletextpad"] = .3  # 凡例の線と文字の距離の長さ
        # mpl.rcParams["legend.frameon"] = False
        mpl.rcParams["legend.facecolor"]  = 'white'



        mpl.rcParams["legend.markerscale"] = 2  # 点がある場合のmarker scale
        mpl.rcParams['axes.xmargin'] = '0'  # '.05'
        mpl.rcParams['axes.ymargin'] = '0'
        mpl.rcParams['savefig.facecolor'] = 'None'
        mpl.rcParams['savefig.edgecolor'] = 'None'
        # mpl.rcParams['savefig.bbox'] = 'tight'
        mpl.rcParams['pdf.fonttype'] = 42  # PDFにフォントを埋め込むためのパラメータ

        # print(self.data['i1_dob_gain'])

        fig, (thm, text) = plt.subplots(2, 1, figsize=(5, 5), dpi=150, sharex=True)

        # plt.xlim([10, 60])  # x軸の範囲
        # plt.xlim([0.28, 0.89])  # x軸の範囲
        plt.xlabel("Time[sec]")

        for i in range(self.join):
            interfacenum = 'i' + str(i + 1)
            thmname = interfacenum + '_p_thm'
            thm_prename = interfacenum + '_p_thm_pre'
            thm.plot(self.cfoo['time'][::10], self.cfoo[thmname][::10], label='P'+str(i+1)+'_act')
            thm.plot(self.cfoo['time'][::10], self.cfoo[thm_prename][::10], label='P'+str(i+1)+'_pre')


            textname = interfacenum + '_p_text'
            text_prename = interfacenum + '_p_text_pre'
            text.plot(self.cfoo['time'][::10], self.cfoo[textname][::10], label='P'+str(i+1)+'_act')
            text.plot(self.cfoo['time'][::10], self.cfoo[text_prename][::10], label='P'+str(i+1)+'_pre')

        thm.set_ylabel('Position [rad]')
        thm.legend(ncol=2, columnspacing=1, loc='upper left')
        # thm.set_yticks(np.arange(-10, 10, 0.5))
        # thm.set_ylim([-1.5, 1.5])  # y軸の範囲

        text.set_ylabel('Reaction torque[Nm]')
        text.legend(ncol=2, columnspacing=1, loc='upper left')
        # text.set_yticks(np.arange(-8.0, 8.0, 2.0))
        # text.set_ylim([-6.0, 6.0])  # y軸の範囲

        plt.tight_layout()
        # plt.savefig("response.png")
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
        # plt.rcParams['savefig.bbox'] = 'tight'self.cfoo
        plt.rcParams['pdf.fonttype'] = 42  # PDFにフォントを埋め込むためのパラメータself.

        fig, (x, y) = plt.subplots(2, 1, figsize=(5, 5), dpi=150, sharex=True)

        x.plot(self.cfoo['time'][::10], self.cfoo['targetx'][::10], label='Target')
        x.plot(self.cfoo['time'][::10], self.cfoo['ballx'][::10], label='Ball(H-H)')
        x.plot(self.cfoo['time'][::10], self.cfoo['ballx_pre'][::10], label='Ball(M-M)')
        # x.plot(data['pre_time'], data['pre_ball_x'], label='pre_ballx')
        x.set_ylabel('X-axis Position (m)')
        x.legend(ncol=2, columnspacing=1, loc='upper left')
        x.set_ylim([-0.2, 0.2])  # y軸の範囲


        y.plot(self.cfoo['time'][::10], self.cfoo['targety'][::10], label='Target')
        y.plot(self.cfoo['time'][::10], self.cfoo['bally'][::10], label='Ball(H-H)')
        y.plot(self.cfoo['time'][::10], self.cfoo['bally_pre'][::10], label='Ball(M-M)')
        # y.plot(data['pre_time'], data['pre_ball_x'], label='pre_bally')
        y.set_ylabel('Y-axis Position (m)')
        y.legend(ncol=2, columnspacing=1, loc='upper left')
        y.set_ylim([-0.2, 0.2])  # y軸の範囲


        # plt.ylabel(r'Position [m]')
        # plt.legend()
        # plt.yticks(np.arange(-4, 4, 0.1))
        # plt.ylim([-0.4, 0.4])  # y軸の範囲
        # plt.xlim([data['starttime'], data['endtime']])  # x軸の範囲
        plt.xlabel("Time[sec]")

        plt.tight_layout()
        # plt.savefig("First_time_target_movement.png")
        plt.show()

    def task_show_sub(self, data):
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

        plt.ylabel(r'Position [m]')
        plt.yticks(np.arange(-4, 4, 0.1))
        plt.ylim([-0.4, 0.4])  # y軸の範囲
        plt.xlim([data[0]['starttime'], data[0]['endtime']])  # x軸の範囲
        plt.xlabel("Time[sec]")

        for i in range(len(data)):
            # plt.plot(data['time'], data['ballx'], label='ballx_'+str(i))
            # plt.plot(data['time'], data['bally'], label='bally_'+str(i))

            plt.plot(data[i]['time'], data[i]['targetx'], label= 'targetx_'+str(i))
            plt.plot(data[i]['time'], data[i]['targety'], label= 'targety_'+str(i))
            plt.legend()




        plt.tight_layout()
        # plt.savefig("First_time_target_movement.png")
        plt.show()

    def cfo_sub(self):
        mpl.rcParams['font.family'] = 'Times New Roman'
        mpl.rcParams['mathtext.default'] = 'regular'
        mpl.rcParams['xtick.top'] = 'True'
        mpl.rcParams['ytick.right'] = 'True'
        # mpl.rcParams['axes.grid'] = 'True'
        mpl.rcParams['xtick.direction'] = 'in'  # x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        mpl.rcParams['ytick.direction'] = 'in'  # y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        mpl.rcParams['xtick.major.width'] = 0.5  # x軸主目盛り線の線幅
        mpl.rcParams['ytick.major.width'] = 0.5  # y軸主目盛り線の線幅
        mpl.rcParams['font.size'] = 6  # フォントの大きさ
        mpl.rcParams['axes.linewidth'] = 0.5  # 軸の線幅edge linewidth。囲みの太さ
        mpl.rcParams['lines.linewidth'] = 0.5  # 軸の線幅edge linewidth。囲みの太さ

        mpl.rcParams["legend.fancybox"] = False  # 丸角
        mpl.rcParams["legend.framealpha"] = 1.0  # 透明度の指定、0で塗りつぶしなし
        # mpl.rcParams["legend.edgecolor"] = 'black'  # edgeの色を変更
        mpl.rcParams["legend.handlelength"] = 2  # 凡例の線の長さを調節
        mpl.rcParams["legend.labelspacing"] = 0.1  # 垂直方向（縦）の距離の各凡例の距離
        mpl.rcParams["legend.handletextpad"] = .3  # 凡例の線と文字の距離の長さ
        # mpl.rcParams["legend.frameon"] = False
        mpl.rcParams["legend.facecolor"] = 'white'

        mpl.rcParams["legend.markerscale"] = 2  # 点がある場合のmarker scale
        mpl.rcParams['axes.xmargin'] = '0'  # '.05'
        mpl.rcParams['axes.ymargin'] = '0'
        mpl.rcParams['savefig.facecolor'] = 'None'
        mpl.rcParams['savefig.edgecolor'] = 'None'
        # mpl.rcParams['savefig.bbox'] = 'tight'
        mpl.rcParams['pdf.fonttype'] = 42  # PDFにフォントを埋め込むためのパラメータ

        # print(self.data['i1_dob_gain'])

        fig, (pcfo, fcfo) = plt.subplots(2, 1, figsize=(5, 5), dpi=150, sharex=True)

        # plt.xlim([10, 60])  # x軸の範囲
        # plt.xlim([0.28, 0.89])  # x軸の範囲
        plt.xlabel("Time[sec]")

        for i in range(self.join):
            interfacenum = 'i' + str(i + 1)
            pcfoname = interfacenum + '_p_pcfo'
            fcfoname = interfacenum + '_p_fcfo'

            pcfo.plot(self.cfoo['time'][::10], self.cfoo[pcfoname][::10], label='P'+str(i+1))
            fcfo.plot(self.cfoo['time'][::10], self.cfoo[fcfoname][::10], label='P'+str(i+1))


        pcfo.set_ylabel('PCFO [rad]')
        pcfo.legend(ncol=2, columnspacing=1, loc='upper left')
        pcfo.set_yticks(np.arange(-10, 10, 0.5))
        pcfo.set_ylim([-1.5, 1.5])  # y軸の範囲

        fcfo.set_ylabel('FCFO [Nm]')
        fcfo.legend(ncol=2, columnspacing=1, loc='upper left')
        fcfo.set_yticks(np.arange(-8.0, 8.0, 2.0))
        fcfo.set_ylim([-6.0, 6.0])  # y軸の範囲

        plt.tight_layout()
        # plt.savefig("response.png")
        plt.show()

    def ocfo(self):
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

        plt.ylabel(r'Position [m]')
        plt.yticks(np.arange(-4, 4, 0.1))
        plt.ylim([-0.4, 0.4])  # y軸の範囲
        # plt.xlim([self.data[0]['starttime'], self.data[0]['endtime']])  # x軸の範囲
        plt.xlabel("Time[sec]")

        plt.xlabel("Time[sec]")

        plt.plot(self.cfoo['time'][::10], self.cfoo['ecfo'][::10], label='ECFO')
        plt.plot(self.cfoo['time'][::10], self.cfoo['inecfo'][::10], label='InECFO')

        plt.ylabel('CFO')
        plt.legend(ncol=2, columnspacing=1, loc='upper left')
        plt.yticks(np.arange(-100, 100, 10.0))
        plt.ylim([-50, 50])  # y軸の範囲plt

        plt.tight_layout()
        plt.show()