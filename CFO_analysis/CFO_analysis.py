import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

import pandas as pd
import seaborn as sns

class CFO:
    def __init__(self, cfo_data, group_type):
        self.group_type = group_type

        self.cfo = cfo_data

        self.smp = 0.0001  # サンプリング時間
        self.time = 3.0  # ターゲットの移動時間
        self.eliminationtime = 0.0  # 消去時間
        self.starttime = 29.0  # タスク開始時間
        self.endtime = 77.0  # タスク終了時間
        self.tasktime = self.endtime - self.starttime  # タスクの時間
        self.period = int((self.tasktime - self.eliminationtime) / self.time)  # 回数
        self.num = int(self.time / self.smp)  # 1ピリオドにおけるデータ数
        self.start_num = int((self.starttime - 20.0) / self.smp)
        self.end_num = int((self.endtime - 20.0) / self.smp)
        self.nn_read_flag = False
        self.join = self.cfo[0]['join'][0]
        # print(self.join)

        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.default'] = 'regular'
        plt.rcParams['xtick.top'] = 'True'
        plt.rcParams['ytick.right'] = 'True'
        # plt.rcParams['axes.grid'] = 'True'
        plt.rcParams['xtick.direction'] = 'in'  # x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'  # y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['xtick.major.width'] = 1.0  # x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 1.0  # y軸主目盛り線の線幅
        plt.rcParams['font.size'] = 8  # フォントの大きさ
        plt.rcParams['axes.linewidth'] = 1.5  # 軸の線幅edge linewidth。囲みの太さ

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
        for j in range(len(self.cfo)):
            fig, (rthm, pthm, rtext, ptext) = plt.subplots(4, 1, figsize=(5, 5), dpi=150, sharex=True)

            plt.xticks(np.arange(self.starttime, self.endtime * 2, self.time * 2))
            plt.xlim([self.starttime, self.endtime])  # x軸の範囲
            plt.xlabel("Time (sec)")

            data = self.cfo[j]
            for i in range(self.join):
                interfacenum = 'i' + str(i + 1)

                thmname = interfacenum + '_r_thm'
                thm_prename = interfacenum + '_r_thm_pre'
                thm_prename_solo = interfacenum + '_r_thm_pre_solo'
                rthm.plot(data['time'][self.start_num:self.end_num:10], data[thmname][self.start_num:self.end_num:10], label='P' + str(i + 1) + '_act')
                rthm.plot(data['time'][self.start_num:self.end_num:10], data[thm_prename][self.start_num:self.end_num:10], label='P' + str(i + 1) + '_pre')
                rthm.plot(data['time'][self.start_num:self.end_num:10], data[thm_prename_solo][self.start_num:self.end_num:10], label='P' + str(i + 1) + '_pre_solo')

                thmname = interfacenum + '_p_thm'
                thm_prename = interfacenum + '_p_thm_pre'
                thm_prename_solo = interfacenum + '_p_thm_pre_solo'
                pthm.plot(data['time'][self.start_num:self.end_num:10], data[thmname][self.start_num:self.end_num:10], label='P'+str(i+1)+'_act')
                pthm.plot(data['time'][self.start_num:self.end_num:10], data[thm_prename][self.start_num:self.end_num:10], label='P'+str(i+1)+'_pre')
                pthm.plot(data['time'][self.start_num:self.end_num:10], data[thm_prename_solo][self.start_num:self.end_num:10], label='P'+str(i+1)+'_pre_solo')

                textname = interfacenum + '_r_text'
                text_prename = interfacenum + '_r_text_pre'
                text_prename_solo = interfacenum + '_r_text_pre_solo'
                rtext.plot(data['time'][self.start_num:self.end_num:10], data[textname][self.start_num:self.end_num:10], label='P' + str(i + 1) + '_act')
                rtext.plot(data['time'][self.start_num:self.end_num:10], data[text_prename][self.start_num:self.end_num:10], label='P' + str(i + 1) + '_pre')
                rtext.plot(data['time'][self.start_num:self.end_num:10], data[text_prename_solo][self.start_num:self.end_num:10], label='P' + str(i + 1) + '_pre_solo')


                textname = interfacenum + '_p_text'
                text_prename = interfacenum + '_p_text_pre'
                text_prename_solo = interfacenum + '_p_text_pre_solo'
                ptext.plot(data['time'][self.start_num:self.end_num:10], data[textname][self.start_num:self.end_num:10], label='P'+str(i+1)+'_act')
                ptext.plot(data['time'][self.start_num:self.end_num:10], data[text_prename][self.start_num:self.end_num:10], label='P'+str(i+1)+'_pre')
                ptext.plot(data['time'][self.start_num:self.end_num:10], data[text_prename_solo][self.start_num:self.end_num:10], label='P'+str(i+1)+'_pre_solo')

            rthm.set_ylabel('Roll angle (rad)')
            rthm.legend(ncol=2, columnspacing=1, loc='upper left')
            rthm.set_yticks(np.arange(-10, 10, 0.5))
            rthm.set_ylim([-1.5, 1.5])  # y軸の範囲

            pthm.set_ylabel('Pitch angle (rad)')
            pthm.legend(ncol=2, columnspacing=1, loc='upper left')
            pthm.set_yticks(np.arange(-10, 10, 0.5))
            pthm.set_ylim([-1.5, 1.5])  # y軸の範囲

            rtext.set_ylabel('Roll force (Nm)')
            rtext.legend(ncol=2, columnspacing=1, loc='upper left')
            rtext.set_yticks(np.arange(-8.0, 8.0, 2.0))
            rtext.set_ylim([-6.0, 6.0])  # y軸の範囲

            ptext.set_ylabel('Pitch force (Nm)')
            ptext.legend(ncol=2, columnspacing=1, loc='upper left')
            ptext.set_yticks(np.arange(-8.0, 8.0, 2.0))
            ptext.set_ylim([-6.0, 6.0])  # y軸の範囲

            plt.tight_layout()
            # plt.savefig("response.png")
        plt.show()

    def task_show(self):
        for i in range(len(self.cfo)):
            data = self.cfo[i]

            fig, (x, y) = plt.subplots(2, 1, figsize=(5, 5), dpi=150, sharex=True)

            x.plot(data['time'][self.start_num:self.end_num:10], data['targetx'][self.start_num:self.end_num:10], label='Target')
            x.plot(data['time'][self.start_num:self.end_num:10], data['targetx_act'][self.start_num:self.end_num:10], label='Target_act')
            x.plot(data['time'][self.start_num:self.end_num:10], data['ballx'][self.start_num:self.end_num:10], label='Ball(H-H)')
            x.plot(data['time'][self.start_num:self.end_num:10], data['ballx_pre'][self.start_num:self.end_num:10], label='Ball(M-M)')
            # x.plot(data['pre_time'], data['pre_ball_x'], label='pre_ballx')
            x.set_ylabel('X-axis Position (m)')
            x.legend(ncol=2, columnspacing=1, loc='upper left')
            x.set_ylim([-0.2, 0.2])  # y軸の範囲


            y.plot(data['time'][self.start_num:self.end_num:10], data['targety'][self.start_num:self.end_num:10], label='Target')
            y.plot(data['time'][self.start_num:self.end_num:10], data['targety_act'][self.start_num:self.end_num:10], label='Target_act')
            y.plot(data['time'][self.start_num:self.end_num:10], data['bally'][self.start_num:self.end_num:10], label='Ball(H-H)')
            y.plot(data['time'][self.start_num:self.end_num:10], data['bally_pre'][self.start_num:self.end_num:10], label='Ball(M-M)')
            # y.plot(data['pre_time'], data['pre_ball_x'], label='pre_bally')
            y.set_ylabel('Y-axis Position (m)')
            y.legend(ncol=2, columnspacing=1, loc='upper left')
            y.set_ylim([-0.2, 0.2])  # y軸の範囲


            # plt.ylabel(r'Position (m)')
            # plt.legend()
            # plt.yticks(np.arange(-4, 4, 0.1))
            # plt.ylim([-0.4, 0.4])  # y軸の範囲
            # plt.xlim([data['starttime'], data['endtime']])  # x軸の範囲

            plt.xticks(np.arange(self.starttime, self.endtime * 2, self.time * 2))
            plt.xlim([self.starttime, self.endtime])  # x軸の範囲
            plt.xlabel("Time (sec)")

            plt.tight_layout()
            # plt.savefig("First_time_target_movement.png")
        plt.show()

    def task_show_solo(self):
        for i in range(len(self.cfo)):
            data = self.cfo[i]

            fig, (x, y) = plt.subplots(2, 1, figsize=(5, 5), dpi=150, sharex=True)

            x.plot(data['time'][self.start_num:self.end_num:10], data['targetx'][self.start_num:self.end_num:10], label='Target')
            # x.plot(data['time'][self.start_num:self.end_num:10], data['targetx_act'][self.start_num:self.end_num:10], label='Target_act')
            x.plot(data['time'][self.start_num:self.end_num:10], data['ballx'][self.start_num:self.end_num:10], label='Ball(H-H)')
            x.plot(data['time'][self.start_num:self.end_num:10], data['ballx_pre'][self.start_num:self.end_num:10], label='Ball(M-M)')
            for j in range(self.join):
                x.plot(data['time'][self.start_num:self.end_num:10], data['i'+str(j+1)+'_ballx_solo'][self.start_num:self.end_num:10], label='Ball(solo'+str(j + 1)+')')


            # x.plot(data['pre_time'], data['pre_ball_x'], label='pre_ballx')
            x.set_ylabel('X-axis Position (m)')
            x.legend(ncol=2, columnspacing=1, loc='upper left')
            x.set_ylim([-0.2, 0.2])  # y軸の範囲


            y.plot(data['time'][self.start_num:self.end_num:10], data['targety'][self.start_num:self.end_num:10], label='Target')
            y.plot(data['time'][self.start_num:self.end_num:10], data['targety_act'][self.start_num:self.end_num:10], label='Target_act')
            y.plot(data['time'][self.start_num:self.end_num:10], data['bally'][self.start_num:self.end_num:10], label='Ball(H-H)')
            y.plot(data['time'][self.start_num:self.end_num:10], data['bally_pre'][self.start_num:self.end_num:10], label='Ball(M-M)')
            for j in range(self.join):
                y.plot(data['time'][self.start_num:self.end_num:10], data['i'+str(j+1)+'_bally_solo'][self.start_num:self.end_num:10], label='Ball(solo'+str(j + 1)+')')
            # y.plot(data['pre_time'], data['pre_ball_x'], label='pre_bally')
            y.set_ylabel('Y-axis Position (m)')
            y.legend(ncol=2, columnspacing=1, loc='upper left')
            y.set_ylim([-0.2, 0.2])  # y軸の範囲


            # plt.ylabel(r'Position (m)')
            # plt.legend()
            # plt.yticks(np.arange(-4, 4, 0.1))
            # plt.ylim([-0.4, 0.4])  # y軸の範囲
            # plt.xlim([data['starttime'], data['endtime']])  # x軸の範囲

            plt.xticks(np.arange(self.starttime, self.endtime * 2, self.time * 2))
            plt.xlim([self.starttime, self.endtime])  # x軸の範囲
            plt.xlabel("Time (sec)")

            plt.tight_layout()
            # plt.savefig("First_time_target_movement.png")
        plt.show()

    def task_show_sub(self, data):

        plt.ylabel(r'Position (m)')
        plt.yticks(np.arange(-4, 4, 0.1))
        plt.ylim([-0.4, 0.4])  # y軸の範囲
        plt.xlim([data[0]['starttime'], data[0]['endtime']])  # x軸の範囲
        plt.xlabel("Time (sec)")

        for i in range(len(data)):
            # plt.plot(data['time'], data['ballx'], label='ballx_'+str(i))
            # plt.plot(data['time'], data['bally'], label='bally_'+str(i))

            plt.plot(data[i]['time'][self.start_num:self.end_num:10], data[i]['targetx'][self.start_num:self.end_num:10], label= 'targetx_'+str(i))
            plt.plot(data[i]['time'][self.start_num:self.end_num:10], data[i]['targety'][self.start_num:self.end_num:10], label= 'targety_'+str(i))
            plt.legend()

        plt.tight_layout()
        # plt.savefig("First_time_target_movement.png")
        plt.show()

    def cfo_sub(self):

        for j in range(len(self.cfo)):
            data = self.cfo[j]

            fig, (ppcfo, rpcfo, pfcfo, rfcfo) = plt.subplots(4, 1, figsize=(5, 8), dpi=150, sharex=True)

            # plt.xlim([10, 60])  # x軸の範囲
            # plt.xlim([0.28, 0.89])  # x軸の範囲
            plt.xlabel("Time (sec)")

            for i in range(self.join):
                interfacenum = 'i' + str(i + 1)
                pcfoname = interfacenum + '_p_pcfo'
                fcfoname = interfacenum + '_p_fcfo'

                ppcfo.plot(data['time'][self.start_num:self.end_num:10], data[pcfoname][self.start_num:self.end_num:10], label='P'+str(i+1))
                pfcfo.plot(data['time'][self.start_num:self.end_num:10], data[fcfoname][self.start_num:self.end_num:10], label='P'+str(i+1))

                pcfoname = interfacenum + '_r_pcfo'
                fcfoname = interfacenum + '_r_fcfo'

                rpcfo.plot(data['time'][self.start_num:self.end_num:10], data[pcfoname][self.start_num:self.end_num:10], label='P' + str(i + 1))
                rfcfo.plot(data['time'][self.start_num:self.end_num:10], data[fcfoname][self.start_num:self.end_num:10], label='P' + str(i + 1))


            ppcfo.set_ylabel('Pitch PCFO (rad)')
            ppcfo.legend(ncol=2, columnspacing=1, loc='upper left')
            ppcfo.set_yticks(np.arange(-10, 10, 0.5))
            ppcfo.set_ylim([-1.5, 1.5])  # y軸の範囲

            rpcfo.set_ylabel('Roll PCFO (rad)')
            rpcfo.legend(ncol=2, columnspacing=1, loc='upper left')
            rpcfo.set_yticks(np.arange(-10, 10, 0.5))
            rpcfo.set_ylim([-1.5, 1.5])  # y軸の範囲

            pfcfo.set_ylabel('Pitch FCFO (Nm)')
            pfcfo.legend(ncol=2, columnspacing=1, loc='upper left')
            pfcfo.set_yticks(np.arange(-8.0, 8.0, 2.0))
            pfcfo.set_ylim([-6.0, 6.0])  # y軸の範囲

            rfcfo.set_ylabel('Roll FCFO (Nm)')
            rfcfo.legend(ncol=2, columnspacing=1, loc='upper left')
            rfcfo.set_yticks(np.arange(-8.0, 8.0, 2.0))
            rfcfo.set_ylim([-6.0, 6.0])  # y軸の範囲r

            plt.tight_layout()
            # plt.savefig("response.png")
        plt.show()

    def ocfo(self):

        plt.ylabel(r'Position (m)')
        plt.yticks(np.arange(-4, 4, 0.1))
        plt.ylim([-0.4, 0.4])  # y軸の範囲
        # plt.xlim([self.data[0]['starttime'], self.data[0]['endtime']])  # x軸の範囲
        plt.xlabel("Time (sec)")

        plt.plot(self.cfoo['time'][::10], self.cfoo['ecfo'][::10], label='ECFO')
        plt.plot(self.cfoo['time'][::10], self.cfoo['inecfo'][::10], label='InECFO')

        plt.ylabel('CFO')
        plt.legend(ncol=2, columnspacing=1, loc='upper left')
        plt.yticks(np.arange(-100, 100, 10.0))
        plt.ylim([-50, 50])  # y軸の範囲plt

        plt.tight_layout()
        plt.show()

    def summation_cfo_3sec(self, mode='noabs'):
        ppcfo_summation, rpcfo_summation, pfcfo_summation, rfcfo_summation = CFO.summation_cfo(self, graph=1, mode=mode)

        ppcfo_summation_3sec = ppcfo_summation.reshape([len(self.cfo), -1, self.num])
        ppcfo_summation_3sec = np.average(ppcfo_summation_3sec, axis=2)

        rpcfo_summation_3sec = rpcfo_summation.reshape([len(self.cfo), -1, self.num])
        rpcfo_summation_3sec = np.average(rpcfo_summation_3sec, axis=2)

        pfcfo_summation_3sec = pfcfo_summation.reshape([len(self.cfo), -1, self.num])
        pfcfo_summation_3sec = np.average(pfcfo_summation_3sec, axis=2)

        rfcfo_summation_3sec = rfcfo_summation.reshape([len(self.cfo), -1, self.num])
        rfcfo_summation_3sec = np.average(rfcfo_summation_3sec, axis=2)

        return ppcfo_summation_3sec, rpcfo_summation_3sec, pfcfo_summation_3sec, rfcfo_summation_3sec

    def summation_cfo(self, graph=1, mode='noabs'):
        summation = self.cfo[0]['i1_p_pcfo'][self.start_num:self.end_num]
        types = ['_p_pcfo', '_r_pcfo', '_p_fcfo', '_r_fcfo']
        for type in types:
            for j in range(len(self.cfo)):
                data = self.cfo[j]
                summation_ = data['i1_p_pcfo'][self.start_num:self.end_num]
                for i in range(self.join):
                    interfacenum = 'i' + str(i + 1)
                    pcfoname = interfacenum + type

                    summation_ = np.vstack((summation_, data[pcfoname][self.start_num:self.end_num]))
                summation_ = np.delete(summation_, 0, 0)
                if mode == 'b_abs':
                    summation_ = np.abs(summation_)
                summation = np.vstack((summation, np.sum(summation_, axis=0)))

            # print(summation_)
        summation = np.delete(summation, 0, 0)
        # print(summation.shape)
        summation = summation.reshape([4, len(self.cfo), -1])
        summation = summation / self.cfo[0]['join'][0]
        if mode == 'a_abs':
            summation = np.abs(summation)

        if graph == 0:
            fig, (ppcfo, rpcfo, pfcfo, rfcfo) = plt.subplots(4, 1, figsize=(5, 7), dpi=150, sharex=True)

            # plt.xlim([10, 60])  # x軸の範囲
            # plt.xlim([0.28, 0.89])  # x軸の範囲
            plt.xlabel("Time (sec)")

            for i in range(len(self.cfo)):
                data = self.cfo[i]

                ppcfo.plot(data['time'][self.start_num:self.end_num:10], summation[0][i][::10],
                           label='Group' + str(i + 1))
                rpcfo.plot(data['time'][self.start_num:self.end_num:10], summation[1][i][::10],
                           label='Group' + str(i + 1))
                pfcfo.plot(data['time'][self.start_num:self.end_num:10], summation[2][i][::10],
                           label='Group' + str(i + 1))
                rfcfo.plot(data['time'][self.start_num:self.end_num:10], summation[3][i][::10],
                           label='Group' + str(i + 1))

            ppcfo.set_ylabel('Summation\nPitch PCFO (rad)')
            rpcfo.set_ylabel('Summation\nRoll PCFO (rad)')
            pfcfo.set_ylabel('Summation\nPitch FCFO (Nm)')
            rfcfo.set_ylabel('Summation\nRoll FCFO (Nm)')

            ppcfo.legend(ncol=10, columnspacing=1, loc='upper left')
            rpcfo.legend(ncol=10, columnspacing=1, loc='upper left')
            pfcfo.legend(ncol=10, columnspacing=1, loc='upper left')
            rfcfo.legend(ncol=10, columnspacing=1, loc='upper left')

            ppcfo.set_yticks(np.arange(-10, 10, 0.5))
            rpcfo.set_yticks(np.arange(-10, 10, 0.5))
            pfcfo.set_yticks(np.arange(-8.0, 8.0, 1.0))
            rfcfo.set_yticks(np.arange(-8.0, 8.0, 1.0))

            ppcfo.set_ylim([0, 1.0])  # y軸の範囲
            rpcfo.set_ylim([0, 1.0])  # y軸の範囲
            pfcfo.set_ylim([0, 4.0])  # y軸の範囲
            rfcfo.set_ylim([0, 4.0])  # y軸の範囲

            plt.tight_layout()
            # plt.savefig(savename)
            plt.show()


        return summation[0], summation[1], summation[2], summation[3]


    def performance_calc(self, data, ballx, bally):
        error = np.sqrt(
            (data['targetx'][self.start_num:self.end_num] - ballx[self.start_num:self.end_num]) ** 2
            + (data['targety'][self.start_num:self.end_num] - bally[self.start_num:self.end_num]) ** 2)

        target_size = 0.03
        spent = np.where(error < target_size, 1, 0)
        # spent = numpy.where(error < self.data['targetsize'], 1, 0)


        return error, spent

    def period_performance_human(self, graph=1):
        error_period = []
        spent_period = []
        for i in range(len(self.cfo)):
            data = self.cfo[i]

            error, spent = CFO.performance_calc(self, data, data['ballx'], data['bally'])
            error_reshape = error.reshape([self.period, self.num]) #[回数][データ]にわける
            # error_period = np.sum(error_reshape, axis=1) # 回数ごとに足す
            error_period.append(np.sum(error_reshape, axis=1) / self.num)


            spent_reshape = spent.reshape([self.period, self.num])
            spent_period_ = np.sum(spent_reshape, axis=1)
            # spent_period = spent_period_ * self.smp
            spent_period.append(spent_period_ * self.smp)

            if graph == 0:

                fig, (error_fig, spend_fig) = plt.subplots(2, 1, figsize=(5, 3), dpi=150, sharex=True)
                plt.xticks(np.arange(1, self.period + 1, 1))
                error_fig.plot(np.arange(self.period) + 1, error_period[i])
                error_fig.scatter(np.arange(self.period) + 1, error_period[i], s=5, marker='x')
                error_fig.set_ylabel('Error (m)')
                error_fig.set_ylim([0, 0.06])

                spend_fig.plot(np.arange(self.period) + 1, spent_period[i])
                spend_fig.scatter(np.arange(self.period) + 1, spent_period[i], s=5, marker='x')
                spend_fig.set_ylabel('Spend (sec)')
                spend_fig.set_ylim([0, 3.0])

                spend_fig.set_xlabel('Period')

        if graph == 0:
            plt.tight_layout()
            plt.show()
            plt.show()


        return error_period, spent_period

    def period_performance_model(self, graph=1):
        error_period = []
        spent_period = []
        for i in range(len(self.cfo)):
            data = self.cfo[i]

            error, spent = CFO.performance_calc(self, data, data['ballx_pre'], data['bally_pre'])
            error_reshape = error.reshape([self.period, self.num]) #[回数][データ]にわける
            # error_period = np.sum(error_reshape, axis=1) # 回数ごとに足す
            error_period.append(np.sum(error_reshape, axis=1) / self.num)

            spent_reshape = spent.reshape([self.period, self.num])
            spent_period_ = np.sum(spent_reshape, axis=1)
            # spent_period = spent_period_ * self.smp
            spent_period.append(spent_period_ * self.smp)

            if graph == 0:
                fig, (error_fig, spend_fig) = plt.subplots(2, 1, figsize=(5, 3), dpi=150, sharex=True)
                plt.xticks(np.arange(1, self.period + 1, 1))
                error_fig.plot(np.arange(self.period) + 1, error_period[i])
                error_fig.scatter(np.arange(self.period) + 1, error_period[i], s=5, marker='x')
                error_fig.set_ylabel('Error (m)')
                error_fig.set_ylim([0, 0.06])

                spend_fig.plot(np.arange(self.period) + 1, spent_period[i])
                spend_fig.scatter(np.arange(self.period) + 1, spent_period[i], s=5, marker='x')
                spend_fig.set_ylabel('Spend (sec)')
                spend_fig.set_ylim([0, 3.0])

                spend_fig.set_xlabel('Period')

        if graph == 0:
            plt.tight_layout()
            plt.show()
            plt.show()

        return error_period, spent_period

    def period_performance_human_model(self, graph=1):
        error_period_human, spent_period_human = CFO.period_performance_human(self)
        error_period_model, spent_period_model = CFO.period_performance_model(self)

        for i in range(len(self.cfo)):

            fig, (error_fig, spend_fig) = plt.subplots(2, 1, figsize=(5, 3), dpi=150, sharex=True)
            plt.xticks(np.arange(1, self.period + 1, 1))
            error_fig.plot(np.arange(self.period) + 1, error_period_human[i], label='H-H')
            error_fig.scatter(np.arange(self.period) + 1, error_period_human[i], s=5, marker='x')
            error_fig.plot(np.arange(self.period) + 1, error_period_model[i], label='M-M')
            error_fig.scatter(np.arange(self.period) + 1, error_period_model[i], s=5, marker='+')
            error_fig.set_ylabel('Error (m)')
            error_fig.set_ylim([0, 0.06])
            error_fig.legend()

            spend_fig.plot(np.arange(self.period) + 1, spent_period_human[i], label='H-H')
            spend_fig.scatter(np.arange(self.period) + 1, spent_period_human[i], s=5, marker='x')
            spend_fig.plot(np.arange(self.period) + 1, spent_period_model[i], label='M-M')
            spend_fig.scatter(np.arange(self.period) + 1, spent_period_model[i], s=5, marker='+')
            spend_fig.set_ylabel('Spent time (sec)')
            spend_fig.set_ylim([0, 3.0])
            spend_fig.legend()

            spend_fig.set_xlabel('Period')

        plt.tight_layout()
        plt.show()


    def period_performance_cooperation(self):
        error_period_human, spend_period_human = CFO.period_performance_human(self)
        error_period_model, spend_period_model = CFO.period_performance_model(self)
        error_period = np.subtract(error_period_human, error_period_model)
        spend_period = np.subtract(spend_period_human, spend_period_model)

        return error_period, spend_period


    def performance_calc_each_axis(self, data, ballx, bally):
        errorx = np.abs(data['targetx'][self.start_num:self.end_num] - ballx[self.start_num:self.end_num])
        errory = np.abs(data['targety'][self.start_num:self.end_num] - bally[self.start_num:self.end_num])

        target_size = 0.03
        spentx = np.where(errorx < target_size, 1, 0)
        spenty = np.where(errory < target_size, 1, 0)
        # spent = numpy.where(error < self.data['targetsize'], 1, 0)


        return errorx, errory, spentx, spenty


    def period_performance_human_each_axis(self, graph=1):
        errorx_period = []
        errory_period = []
        spentx_period = []
        spenty_period = []
        for i in range(len(self.cfo)):
            data = self.cfo[i]

            errorx, errory, spentx, spenty = CFO.performance_calc_each_axis(self, data, data['ballx'], data['bally'])
            errorx_reshape = errorx.reshape([self.period, self.num]) #[回数][データ]にわける
            errory_reshape = errory.reshape([self.period, self.num]) #[回数][データ]にわける
            # error_period = np.sum(error_reshape, axis=1) # 回数ごとに足す
            errorx_period.append(np.sum(errorx_reshape, axis=1) / self.num)
            errory_period.append(np.sum(errory_reshape, axis=1) / self.num)


            spentx_reshape = spentx.reshape([self.period, self.num])
            spenty_reshape = spenty.reshape([self.period, self.num])
            spentx_period_ = np.sum(spentx_reshape, axis=1)
            spenty_period_ = np.sum(spenty_reshape, axis=1)
            # spent_period = spent_period_ * self.smp
            spentx_period.append(spentx_period_ * self.smp)
            spenty_period.append(spenty_period_ * self.smp)

            if graph == 0:

                fig, (errorx_fig, errory_fig, spendx_fig, spendy_fig) = plt.subplots(4, 1, figsize=(5, 3), dpi=150, sharex=True)
                plt.xticks(np.arange(1, self.period + 1, 1))
                errorx_fig.plot(np.arange(self.period) + 1, errorx_period[i])
                errorx_fig.scatter(np.arange(self.period) + 1, errorx_period[i], s=5, marker='x')
                errorx_fig.set_ylabel('Error (m)')
                errorx_fig.set_ylim([0, 0.06])

                plt.xticks(np.arange(1, self.period + 1, 1))
                errory_fig.plot(np.arange(self.period) + 1, errory_period[i])
                errory_fig.scatter(np.arange(self.period) + 1, errory_period[i], s=5, marker='x')
                errory_fig.set_ylabel('Error (m)')
                errory_fig.set_ylim([0, 0.06])

                spendx_fig.plot(np.arange(self.period) + 1, spentx_period[i])
                spendx_fig.scatter(np.arange(self.period) + 1, spentx_period[i], s=5, marker='x')
                spendx_fig.set_ylabel('Spend (sec)')
                spendx_fig.set_ylim([0, 3.0])

                spendy_fig.plot(np.arange(self.period) + 1, spenty_period[i])
                spendy_fig.scatter(np.arange(self.period) + 1, spenty_period[i], s=5, marker='x')
                spendy_fig.set_ylabel('Spend (sec)')
                spendy_fig.set_ylim([0, 3.0])

                spendy_fig.set_xlabel('Period')
        if graph == 0:
            plt.tight_layout()
            plt.show()
            plt.show()


        return errorx_period, errory_period, spentx_period, spenty_period

    def period_performance_model_each_axis(self, graph=1):
        errorx_period = []
        errory_period = []
        spentx_period = []
        spenty_period = []
        for i in range(len(self.cfo)):
            data = self.cfo[i]

            errorx, errory, spentx, spenty = CFO.performance_calc_each_axis(self, data, data['ballx_pre'], data['bally_pre'])
            errorx_reshape = errorx.reshape([self.period, self.num])  # [回数][データ]にわける
            errory_reshape = errory.reshape([self.period, self.num])  # [回数][データ]にわける
            # error_period = np.sum(error_reshape, axis=1) # 回数ごとに足す
            errorx_period.append(np.sum(errorx_reshape, axis=1) / self.num)
            errory_period.append(np.sum(errory_reshape, axis=1) / self.num)

            spentx_reshape = spentx.reshape([self.period, self.num])
            spenty_reshape = spenty.reshape([self.period, self.num])
            spentx_period_ = np.sum(spentx_reshape, axis=1)
            spenty_period_ = np.sum(spenty_reshape, axis=1)
            # spent_period = spent_period_ * self.smp
            spentx_period.append(spentx_period_ * self.smp)
            spenty_period.append(spenty_period_ * self.smp)

            if graph == 0:
                fig, (errorx_fig, errory_fig, spendx_fig, spendy_fig) = plt.subplots(4, 1, figsize=(5, 3), dpi=150,
                                                                                     sharex=True)
                plt.xticks(np.arange(1, self.period + 1, 1))
                errorx_fig.plot(np.arange(self.period) + 1, errorx_period[i])
                errorx_fig.scatter(np.arange(self.period) + 1, errorx_period[i], s=5, marker='x')
                errorx_fig.set_ylabel('Error (m)')
                errorx_fig.set_ylim([0, 0.06])

                plt.xticks(np.arange(1, self.period + 1, 1))
                errory_fig.plot(np.arange(self.period) + 1, errory_period[i])
                errory_fig.scatter(np.arange(self.period) + 1, errory_period[i], s=5, marker='x')
                errory_fig.set_ylabel('Error (m)')
                errory_fig.set_ylim([0, 0.06])

                spendx_fig.plot(np.arange(self.period) + 1, spentx_period[i])
                spendx_fig.scatter(np.arange(self.period) + 1, spentx_period[i], s=5, marker='x')
                spendx_fig.set_ylabel('Spend (sec)')
                spendx_fig.set_ylim([0, 3.0])

                spendy_fig.plot(np.arange(self.period) + 1, spenty_period[i])
                spendy_fig.scatter(np.arange(self.period) + 1, spenty_period[i], s=5, marker='x')
                spendy_fig.set_ylabel('Spend (sec)')
                spendy_fig.set_ylim([0, 3.0])

                spendy_fig.set_xlabel('Period')
        if graph == 0:
            plt.tight_layout()
            plt.show()
            plt.show()

        return errorx_period, errory_period, spentx_period, spenty_period

    def period_performance_human_model_each_axis(self, graph=1):
        errorx_period_human, errory_period_human, spentx_period_human, spenty_period_human = CFO.period_performance_human_each_axis(self)
        errorx_period_model, errory_period_model, spentx_period_model, spenty_period_model = CFO.period_performance_model_each_axis(self)

        for i in range(len(self.cfo)):

            fig, (errorx_fig, errory_fig, spendx_fig, spendy_fig) = plt.subplots(4, 1, figsize=(5, 3), dpi=150, sharex=True)
            plt.xticks(np.arange(1, self.period + 1, 1))
            errorx_fig.plot(np.arange(self.period) + 1, errorx_period_human[i], label='H-H')
            errorx_fig.scatter(np.arange(self.period) + 1, errorx_period_human[i], s=5, marker='x')
            errorx_fig.plot(np.arange(self.period) + 1, errorx_period_model[i], label='M-M')
            errorx_fig.scatter(np.arange(self.period) + 1, errorx_period_model[i], s=5, marker='+')
            errorx_fig.set_ylabel('Error (m)')
            errorx_fig.set_ylim([0, 0.06])
            errorx_fig.legend()

            errory_fig.plot(np.arange(self.period) + 1, errory_period_human[i], label='H-H')
            errory_fig.scatter(np.arange(self.period) + 1, errory_period_human[i], s=5, marker='x')
            errory_fig.plot(np.arange(self.period) + 1, errory_period_model[i], label='M-M')
            errory_fig.scatter(np.arange(self.period) + 1, errory_period_model[i], s=5, marker='+')
            errory_fig.set_ylabel('Error (m)')
            errory_fig.set_ylim([0, 0.06])
            errory_fig.legend()

            spendx_fig.plot(np.arange(self.period) + 1, spentx_period_human[i], label='H-H')
            spendx_fig.scatter(np.arange(self.period) + 1, spentx_period_human[i], s=5, marker='x')
            spendx_fig.plot(np.arange(self.period) + 1, spentx_period_model[i], label='M-M')
            spendx_fig.scatter(np.arange(self.period) + 1, spentx_period_model[i], s=5, marker='+')
            spendx_fig.set_ylabel('Spent time (sec)')
            spendx_fig.set_ylim([0, 3.0])
            spendx_fig.legend()

            spendy_fig.plot(np.arange(self.period) + 1, spenty_period_human[i], label='H-H')
            spendy_fig.scatter(np.arange(self.period) + 1, spenty_period_human[i], s=5, marker='x')
            spendy_fig.plot(np.arange(self.period) + 1, spenty_period_model[i], label='M-M')
            spendy_fig.scatter(np.arange(self.period) + 1, spenty_period_model[i], s=5, marker='+')
            spendy_fig.set_ylabel('Spent time (sec)')
            spendy_fig.set_ylim([0, 3.0])
            spendy_fig.legend()

            spendy_fig.set_xlabel('Period')

        plt.tight_layout()
        plt.show()

    def period_performance_cooperation_each_axis(self):
        errorx_period_human, errory_period_human, spendx_period_human, spendy_period_human = CFO.period_performance_human_each_axis(self)
        errorx_period_model, errory_period_model, spendx_period_model, spendy_period_model = CFO.period_performance_model_each_axis(self)
        errorx_period = np.subtract(errorx_period_human, errorx_period_model)
        errory_period = np.subtract(errory_period_human, errory_period_model)
        spendx_period = np.subtract(spendx_period_human, spendx_period_model)
        spendy_period = np.subtract(spendy_period_human, spendy_period_model)

        return errorx_period, errory_period, spendx_period, spendy_period


    def each_ocfo_performance(self, mode='normal'):
        error_period, spend_period = CFO.period_performance_cooperation(self)
        ecfo = CFO.period_ecfo(self, mode)
        inecfo = CFO.period_inecfo(self, mode)

        df = []
        for i in range(len(self.cfo)):
            df.append(pd.DataFrame({
                'error': error_period[i],
                'spend': spend_period[i],
                'ECFO': ecfo[i],
                'Ineffective CFO': inecfo[i],
            }))

            df[i]['Group'] = 'Group' + str(i + 1)

        df_all = pd.concat([i for i in df], axis=0)
        # print(df_all)

        marker_size = 5

        fig = plt.figure(figsize=(4, 4), dpi=200)

        xdata = ['ECFO', 'Ineffective CFO']
        ydata = ['error', 'spend']

        xlim = [[-1.5, 1.5],    #ECOF
                [ 0.0, 4.0]]    #InECFO
        ylim = [[-0.015, 0.015],    #error
                [-0.3, 0.4]]    #spend


        for i in range(2):
            for j in range(2):
                ax = fig.add_subplot(2, 2, i * 2 + j + 1)
                ax.set_xlim(xlim[i][0], xlim[i][1])
                ax.set_ylim(ylim[j][0], ylim[j][1])
                g = sns.scatterplot(data=df_all, x=xdata[i], y=ydata[j], hue='Group', s=marker_size)
                for lh in g.legend_.legendHandles:
                    lh.set_alpha(1)
                    lh._sizes = [10]
        plt.tight_layout()
        plt.show()

    def ocfo_performance(self):
        error_period, spend_period = CFO.period_performance_cooperation(self)
        ecfo = CFO.period_ecfo(self)
        inecfo = CFO.period_inecfo(self)

        df = []
        for i in range(len(self.cfo)):
            df.append(pd.DataFrame({
                'error': error_period[i],
                'spend': spend_period[i],
                'Effective CFO': ecfo[i],
                'Ineffective CFO': inecfo[i],
            }))

            df[i]['Group'] = 'Group' + str(i + 1)

        df_all = pd.concat([i for i in df], axis=0)
        # print(df_all)

        marker_size = 5

        fig = plt.figure(figsize=(6, 3), dpi=300)

        performance = ['error', 'spend']
        label = ['Error (m)', 'Spent time (sec)']

        for i in range(2):
            ax = fig.add_subplot(1, 2, i + 1)
            ax.set_xlim(0.0, 3.0)
            ax.set_ylim(-3.0, 3.0)
            points = ax.scatter(df_all['Ineffective CFO'], df_all['Effective CFO'], c=df_all[performance[i]], s=marker_size, cmap="Blues")
            plt.colorbar(points, ax=ax, label=label[i])
            ax.set_xlabel('Ineffective CFO')
            ax.set_ylabel('Effective CFO')

        plt.tight_layout()
        plt.show()

    def sum_sub_performance(self):
        error_period, spend_period = CFO.period_performance_cooperation(self)
        ppcof_summation_3sec, rpcof_summation_3sec, pfcof_summation_3sec, rfcof_summation_3sec = CFO.summation_cfo_3sec(self)
        ppcof_babs_summation_3sec, rpcof_babs_summation_3sec, pfcof_babs_summation_3sec, rfcof_babs_summation_3sec = CFO.summation_cfo_3sec(self, 'b_abs')
        ppcof_aabs_summation_3sec, rpcof_aabs_summation_3sec, pfcof_aabs_summation_3sec, rfcof_aabs_summation_3sec = CFO.summation_cfo_3sec(self, 'a_abs')
        ppcof_subtraction_3sec, rpcof_subtraction_3sec, pfcof_subtraction_3sec, rfcof_subtraction_3sec = CFO.subtraction_cfo_3sec(self)


        df = []
        for i in range(len(self.cfo)):
            df.append(pd.DataFrame({
                'error': error_period[i],
                'spend': spend_period[i],
                'Pitch summation PCFO': ppcof_summation_3sec[i],
                'Roll summation PCFO': rpcof_summation_3sec[i],
                'Pitch summation FCFO': pfcof_summation_3sec[i],
                'Roll summation FCFO': rfcof_summation_3sec[i],
                'Before abs. Pitch summation PCFO': ppcof_babs_summation_3sec[i],
                'Before abs. Roll summation PCFO': rpcof_babs_summation_3sec[i],
                'Before abs. Pitch summation FCFO': pfcof_babs_summation_3sec[i],
                'Before abs. Roll summation FCFO': rfcof_babs_summation_3sec[i],
                'After abs. Pitch summation PCFO': ppcof_aabs_summation_3sec[i],
                'After abs. Roll summation PCFO': rpcof_aabs_summation_3sec[i],
                'After abs. Pitch summation FCFO': pfcof_aabs_summation_3sec[i],
                'After abs. Roll summation FCFO': rfcof_aabs_summation_3sec[i],
                'Pitch subtraction PCFO': ppcof_subtraction_3sec[i],
                'Roll subtraction PCFO': rpcof_subtraction_3sec[i],
                'Pitch subtraction FCFO': pfcof_subtraction_3sec[i],
                'Roll subtraction FCFO': rfcof_subtraction_3sec[i],
            }))

            df[i]['Group'] = 'Group' + str(i + 1)

        df_all = pd.concat([i for i in df], axis=0)
        # print(df_all)

        marker_size = 5

        performance = ['error', 'spend']
        cmap = ['Blues_r', 'Blues']
        ylabel = [
            ['Pitch summation PCFO', 'Roll summation PCFO', 'Pitch summation FCFO', 'Roll summation FCFO'],
            ['Before abs. Pitch summation PCFO', 'Before abs. Roll summation PCFO', 'Before abs. Pitch summation FCFO', 'Before abs. Roll summation FCFO'],
            ['After abs. Pitch summation PCFO', 'After abs. Roll summation PCFO', 'After abs. Pitch summation FCFO', 'After abs. Roll summation FCFO'],
        ]
        xlabel = ['Pitch subtraction PCFO', 'Roll subtraction PCFO', 'Pitch subtraction FCFO','Roll subtraction FCFO']

        xlim = [0.2, 0.2, 4.0, 3.0]

        ylim = [
            [
                [-0.04, 0.04],
                [-0.04, 0.04],
                [-0.2, 0.2],
                [-0.2, 0.2],
            ],
            [
                [0.0, 0.15],
                [0.0, 0.15],
                [0.0, 2.0],
                [0.0, 1.25],
            ],
            [
                [0.0, 0.12],
                [0.0, 0.12],
                [0.0, 0.7],
                [0.0, 0.6],
            ]
        ]


        for i in range(3):
            fig = plt.figure(figsize=(8, 4), dpi=300)
            for j in range(2):
                for k in range(4):
                    ax = fig.add_subplot(2, 4, 4 * j + k + 1)
                    ax.set_xlim(0, xlim[k])
                    ax.set_ylim(ylim[i][k][0], ylim[i][k][1])
                    points = ax.scatter(df_all[xlabel[k]], df_all[ylabel[i][k]], c=df_all[performance[j]], s=marker_size, cmap=cmap[j])
                    plt.colorbar(points, ax=ax, label=performance[j])
                    ax.set_xlabel(xlabel[k])
                    ax.set_ylabel(ylabel[i][k])

            plt.tight_layout()
        plt.show()


    def period_ecfo(self, mode='normal'):
        ecfo = []
        for i in range(len(self.cfo)):
            data = self.cfo[i]
            if mode == 'error':
                error, spent = CFO.performance_calc(self, data, data['ballx'], data['bally'])
                ecfo.append(CFO.period_calculation_consider_error(self, data['ecfo'][self.start_num:self.end_num], spent))
            else:
                ecfo.append(CFO.period_calculation(self, data['ecfo'][self.start_num:self.end_num]))

        return ecfo

    def period_inecfo(self, mode='normal'):
        inecfo = []
        for i in range(len(self.cfo)):
            data = self.cfo[i]
            if mode == 'error':
                error, spent = CFO.performance_calc(self, data, data['ballx'], data['bally'])
                inecfo.append(CFO.period_calculation_consider_error(self, np.abs(data['inecfo'][self.start_num:self.end_num]), spent))
            else:
                inecfo.append(CFO.period_calculation(self, np.abs(data['inecfo'][self.start_num:self.end_num])))
        return inecfo

    def period_calculation(self, data):
        data_reshape = data.reshape([self.period, self.num])  # [回数][データ]にわける
        data_period = np.sum(data_reshape, axis=1) / self.num
        return data_period

    def period_calculation_consider_error(self, data, spend):
        spent_inv = np.where(spend == 0, 1, 0)
        data_consider = data * spent_inv
        data_reshape = data_consider.reshape([self.period, self.num])  # [回数][データ]にわける
        count = np.zeros(len(data_reshape))
        for i in range(len(data_reshape)):
            count[i] = self.num - np.count_nonzero(data_reshape[i])

        data_period = np.sum(data_reshape, axis=1) / count

        return data_period

    def subtraction_cfo(self, graph=1):
        subtraction = self.cfo[0]['i1_p_pcfo'][self.start_num:self.end_num]
        types = ['p_pcfo', 'r_pcfo', 'p_fcfo', 'r_fcfo']
        for type in types:
            subtraction_ = self.cfo[0]['i1_p_pcfo'][self.start_num:self.end_num]
            for i in range(len(self.cfo)):
                data = self.cfo[i]
                if self.group_type == 'dyad':
                    sub_cfo1 = np.abs(np.subtract(data['i1_' + type][self.start_num:self.end_num], data['i2_' + type][self.start_num:self.end_num]))
                    sub_cfo2 = np.abs(np.subtract(data['i2_' + type][self.start_num:self.end_num], data['i1_' + type][self.start_num:self.end_num]))
                    sub_cfo_ave = np.add(np.abs(sub_cfo1), np.abs(sub_cfo2)) / 2
                    subtraction_ = np.vstack((subtraction_, sub_cfo_ave))

                elif self.group_type == 'triad':
                    sub_cfo1 = np.subtract(np.subtract(2 * data['i1_' + type][self.start_num:self.end_num], data['i2_' + type][self.start_num:self.end_num]), data['i3_' + type][self.start_num:self.end_num])
                    sub_cfo2 = np.subtract(np.subtract(2 * data['i2_' + type][self.start_num:self.end_num], data['i1_' + type][self.start_num:self.end_num]), data['i3_' + type][self.start_num:self.end_num])
                    sub_cfo3 = np.subtract(np.subtract(2 * data['i3_' + type][self.start_num:self.end_num], data['i1_' + type][self.start_num:self.end_num]), data['i2_' + type][self.start_num:self.end_num])
                    sub_cfo_ave = np.add(np.add(np.abs(sub_cfo1), np.abs(sub_cfo2)), np.abs(sub_cfo3)) / 3
                    subtraction_ = np.vstack((subtraction_, sub_cfo_ave))

                elif self.group_type == 'tetrad':
                    sub_cfo1 = np.subtract(np.subtract(np.subtract(3 * data['i1_' + type][self.start_num:self.end_num], data['i2_' + type][self.start_num:self.end_num]), data['i3_' + type][self.start_num:self.end_num]), data['i4_' + type][self.start_num:self.end_num])
                    sub_cfo2 = np.subtract(np.subtract(np.subtract(3 * data['i2_' + type][self.start_num:self.end_num], data['i1_' + type][self.start_num:self.end_num]), data['i3_' + type][self.start_num:self.end_num]), data['i4_' + type][self.start_num:self.end_num])
                    sub_cfo3 = np.subtract(np.subtract(np.subtract(3 * data['i3_' + type][self.start_num:self.end_num], data['i1_' + type][self.start_num:self.end_num]), data['i2_' + type][self.start_num:self.end_num]), data['i4_' + type][self.start_num:self.end_num])
                    sub_cfo4 = np.subtract(np.subtract(np.subtract(3 * data['i4_' + type][self.start_num:self.end_num], data['i1_' + type][self.start_num:self.end_num]), data['i2_' + type][self.start_num:self.end_num]), data['i3_' + type][self.start_num:self.end_num])
                    sub_cfo_ave = np.add(np.add(np.add(np.abs(sub_cfo1), np.abs(sub_cfo2)), np.abs(sub_cfo3)), np.abs(sub_cfo4)) / 4
                    subtraction_ = np.vstack((subtraction_, sub_cfo_ave))

            subtraction_ = np.delete(subtraction_, 0, 0)
            subtraction = np.vstack((subtraction, subtraction_))
        subtraction = np.delete(subtraction, 0, 0)
        subtraction = subtraction.reshape([4, len(self.cfo), -1])

        if graph == 0:
            fig, (ppcfo, rpcfo, pfcfo, rfcfo) = plt.subplots(4, 1, figsize=(5, 7), dpi=150, sharex=True)

            # plt.xlim([10, 60])  # x軸の範囲
            # plt.xlim([0.28, 0.89])  # x軸の範囲
            plt.xlabel("Time (sec)")

            for i in range(len(self.cfo)):
                data = self.cfo[i]

                ppcfo.plot(data['time'][self.start_num:self.end_num:10], subtraction[0][i][::10], label='Group' + str(i + 1))
                rpcfo.plot(data['time'][self.start_num:self.end_num:10], subtraction[1][i][::10], label='Group' + str(i + 1))
                pfcfo.plot(data['time'][self.start_num:self.end_num:10], subtraction[2][i][::10], label='Group' + str(i + 1))
                rfcfo.plot(data['time'][self.start_num:self.end_num:10], subtraction[3][i][::10], label='Group' + str(i + 1))

            ppcfo.set_ylabel('Subtraction\nPitch PCFO (rad)')
            rpcfo.set_ylabel('Subtraction\nRoll PCFO (rad)')
            pfcfo.set_ylabel('Subtraction\nPitch FCFO (Nm)')
            rfcfo.set_ylabel('Subtraction\nRoll FCFO (Nm)')

            ppcfo.legend(ncol=10, columnspacing=1, loc='upper left')
            rpcfo.legend(ncol=10, columnspacing=1, loc='upper left')
            pfcfo.legend(ncol=10, columnspacing=1, loc='upper left')
            rfcfo.legend(ncol=10, columnspacing=1, loc='upper left')

            ppcfo.set_yticks(np.arange(-10, 10, 0.5))
            rpcfo.set_yticks(np.arange(-10, 10, 0.5))
            pfcfo.set_yticks(np.arange(-8.0, 8.0, 1.0))
            rfcfo.set_yticks(np.arange(-8.0, 8.0, 1.0))

            ppcfo.set_ylim([0, 1.0])  # y軸の範囲
            rpcfo.set_ylim([0, 1.0])  # y軸の範囲
            pfcfo.set_ylim([0, 10.0])  # y軸の範囲
            rfcfo.set_ylim([0, 10.0])  # y軸の範囲

            plt.tight_layout()
            # plt.savefig(savename)
            plt.show()


        return subtraction[0], subtraction[1], subtraction[2], subtraction[3]

    def subtraction_cfo_3sec(self):
        ppcfo_subtraction, rpcfo_subtraction, pfcfo_subtraction, rfcfo_subtraction = CFO.subtraction_cfo(self)

        ppcfo_subtraction_3sec = ppcfo_subtraction.reshape([len(self.cfo), -1, self.num])
        ppcfo_subtraction_3sec = np.average(ppcfo_subtraction_3sec, axis=2)

        rpcfo_subtraction_3sec = rpcfo_subtraction.reshape([len(self.cfo), -1, self.num])
        rpcfo_subtraction_3sec = np.average(rpcfo_subtraction_3sec, axis=2)

        pfcfo_subtraction_3sec = pfcfo_subtraction.reshape([len(self.cfo), -1, self.num])
        pfcfo_subtraction_3sec = np.average(pfcfo_subtraction_3sec, axis=2)

        rfcfo_subtraction_3sec = rfcfo_subtraction.reshape([len(self.cfo), -1, self.num])
        rfcfo_subtraction_3sec = np.average(rfcfo_subtraction_3sec, axis=2)

        return ppcfo_subtraction_3sec, rpcfo_subtraction_3sec, pfcfo_subtraction_3sec, rfcfo_subtraction_3sec

    def subtraction_performance(self):
        error_period, spend_period = CFO.period_performance_cooperation(self)
        ppcof_subtraction_3sec, rpcof_subtraction_3sec, pfcof_subtraction_3sec, rfcof_subtraction_3sec = CFO.subtraction_cfo_3sec(self)

        df = []
        for i in range(len(self.cfo)):
            df.append(pd.DataFrame({
                'error': error_period[i],
                'spend': spend_period[i],
                'Pitch subtraction PCFO': ppcof_subtraction_3sec[i],
                'Roll subtraction PCFO': rpcof_subtraction_3sec[i],
                'Pitch subtraction FCFO': pfcof_subtraction_3sec[i],
                'Roll subtraction FCFO': rfcof_subtraction_3sec[i],
            }))

            df[i]['Group'] = 'Group' + str(i + 1)

        df_all = pd.concat([i for i in df], axis=0)
        # print(df_all)

        marker_size = 5

        performance = ['error', 'spend']
        cmap = ['Blues_r', 'Blues']
        xlabel = ['Pitch subtraction PCFO', 'Roll subtraction PCFO', 'Pitch subtraction FCFO', 'Roll subtraction FCFO']

        if self.group_type == 'dyad':
            xlim = [0.2, 0.2, 3.0, 3.0]
        elif self.group_type == 'triad':
            xlim = [0.3, 0.3, 4.0, 5.0]
        else:
            xlim = [0.4, 0.4, 5.0, 5.0]

        fig = plt.figure(figsize=(8, 4), dpi=300)
        for i in range(2):
            for j in range(4):
                ax = fig.add_subplot(2, 4, 4 * i + j + 1)
                ax.set_xlim(0, xlim[j])
                # ax.set_ylim(ylim[i][k][0], ylim[i][k][1])
                g = sns.scatterplot(data=df_all, x=xlabel[j], y=performance[i], hue='Group', s=marker_size)
                for lh in g.legend_.legendHandles:
                    lh.set_alpha(1)
                    lh._sizes = [10]

                ax.set_xlabel(xlabel[j])
                ax.set_ylabel(performance[i])

        plt.tight_layout()
        plt.savefig('fig/subtraction_performance_' + str(self.group_type) + '.png')
        plt.show()

    def summation_performance(self, mode='noabs'):
        error_period, spend_period = CFO.period_performance_cooperation(self)
        ppcof_summation_3sec, rpcof_summation_3sec, pfcof_summation_3sec, rfcof_summation_3sec = CFO.summation_cfo_3sec(self, mode)
        if mode == 'noabs':
            xlabel = ['Pitch summation PCFO', 'Roll summation PCFO', 'Pitch summation FCFO', 'Roll summation FCFO']
            xlim = [
                [-0.04, 0.04],
                [-0.04, 0.04],
                [-0.2, 0.2],
                [-0.2, 0.2],
            ]
        elif mode == 'b_abs':
            xlabel = ['Before abs. Pitch summation PCFO', 'Before abs. Roll summation PCFO', 'Before abs. Pitch summation FCFO',
             'Before abs. Roll summation FCFO']
            xlim = [
                [0.0, 0.15],
                [0.0, 0.15],
                [0.0, 2.0],
                [0.0, 1.25],
            ]
        elif mode == 'a_abs':
            xlabel = ['After abs. Pitch summation PCFO', 'After abs. Roll summation PCFO', 'After abs. Pitch summation FCFO',
             'After abs. Roll summation FCFO']
            xlim = [
                [0.0, 0.12],
                [0.0, 0.12],
                [0.0, 0.7],
                [0.0, 0.6],
            ]

        performance = ['error', 'spend']

        df = []
        for i in range(len(self.cfo)):
            df.append(pd.DataFrame({
                performance[0]: error_period[i],
                performance[1]: spend_period[i],
                xlabel[0]: ppcof_summation_3sec[i],
                xlabel[1]: rpcof_summation_3sec[i],
                xlabel[2]: pfcof_summation_3sec[i],
                xlabel[3]: rfcof_summation_3sec[i],
            }))

            df[i]['Group'] = 'Group' + str(i + 1)

        df_all = pd.concat([i for i in df], axis=0)
        # print(df_all)

        marker_size = 5

        cmap = ['Blues_r', 'Blues']

        fig = plt.figure(figsize=(8, 4), dpi=300)
        for i in range(2):
            for j in range(4):
                ax = fig.add_subplot(2, 4, 4 * i + j + 1)
                ax.set_xlim(xlim[j][0], xlim[j][1])
                # ax.set_ylim(ylim[i][k][0], ylim[i][k][1])
                g = sns.scatterplot(data=df_all, x=xlabel[j], y=performance[i], hue='Group', s=marker_size)
                for lh in g.legend_.legendHandles:
                    lh.set_alpha(1)
                    lh._sizes = [10]

                ax.set_xlabel(xlabel[j])
                ax.set_ylabel(performance[i])

        plt.tight_layout()
        plt.savefig('fig/summation_performance_' + str(self.group_type) + '.png')
        plt.show()



    def subtraction_performance_each_axis(self):
        errorx_period, errory_period, spendx_period, spendy_period = CFO.period_performance_cooperation_each_axis(self)
        ppcof_subtraction_3sec, rpcof_subtraction_3sec, pfcof_subtraction_3sec, rfcof_subtraction_3sec = CFO.subtraction_cfo_3sec(self)

        df = []
        for i in range(len(self.cfo)):
            df.append(pd.DataFrame({
                'errorx': errorx_period[i],
                'errory': errory_period[i],
                'spendx': spendx_period[i],
                'spendy': spendy_period[i],
                'Pitch subtraction PCFO': ppcof_subtraction_3sec[i],
                'Roll subtraction PCFO': rpcof_subtraction_3sec[i],
                'Pitch subtraction FCFO': pfcof_subtraction_3sec[i],
                'Roll subtraction FCFO': rfcof_subtraction_3sec[i],
            }))

            df[i]['Group'] = 'Group' + str(i + 1)

        df_all = pd.concat([i for i in df], axis=0)
        # print(df_all)

        marker_size = 5

        performance = [['errorx', 'errory'], ['spendx', 'spendy']]
        cmap = ['Blues_r', 'Blues']
        xlabel = ['Pitch subtraction PCFO', 'Roll subtraction PCFO', 'Pitch subtraction FCFO', 'Roll subtraction FCFO']

        if self.group_type == 'dyad':
            xlim = [0.2, 0.2, 3.0, 3.0]
        elif self.group_type == 'triad':
            xlim = [0.3, 0.3, 4.0, 5.0]
        else:
            xlim = [0.4, 0.4, 5.0, 5.0]

        fig = plt.figure(figsize=(8, 4), dpi=300)
        for i in range(2):
            for j in range(4):
                ax = fig.add_subplot(2, 4, 4 * i + j + 1)
                ax.set_xlim(0, xlim[j])
                # ax.set_ylim(ylim[i][k][0], ylim[i][k][1])
                g = sns.scatterplot(data=df_all, x=xlabel[j], y=performance[i][j%2], hue='Group', s=marker_size)
                for lh in g.legend_.legendHandles:
                    lh.set_alpha(1)
                    lh._sizes = [10]

                ax.set_xlabel(xlabel[j])
                ax.set_ylabel(performance[i][j%2])

        plt.tight_layout()
        plt.savefig('fig/subtraction_performance_each_axis_' + str(self.group_type) + '.png')
        plt.show()

    def summation_performance_each_axis(self, mode='noabs'):
        errorx_period, errory_period, spendx_period, spendy_period = CFO.period_performance_cooperation_each_axis(self)
        ppcof_summation_3sec, rpcof_summation_3sec, pfcof_summation_3sec, rfcof_summation_3sec = CFO.summation_cfo_3sec(self, mode)
        if mode == 'noabs':
            xlabel = ['Pitch summation PCFO', 'Roll summation PCFO', 'Pitch summation FCFO', 'Roll summation FCFO']
            xlim = [
                [-0.04, 0.04],
                [-0.04, 0.04],
                [-0.2, 0.2],
                [-0.2, 0.2],
            ]
        elif mode == 'b_abs':
            xlabel = ['Before abs. Pitch summation PCFO', 'Before abs. Roll summation PCFO', 'Before abs. Pitch summation FCFO',
             'Before abs. Roll summation FCFO']
            xlim = [
                [0.0, 0.15],
                [0.0, 0.15],
                [0.0, 2.0],
                [0.0, 1.25],
            ]
        elif mode == 'a_abs':
            xlabel = ['After abs. Pitch summation PCFO', 'After abs. Roll summation PCFO', 'After abs. Pitch summation FCFO',
             'After abs. Roll summation FCFO']
            xlim = [
                [0.0, 0.12],
                [0.0, 0.12],
                [0.0, 0.7],
                [0.0, 0.6],
            ]

        performance = [['errorx', 'errory'], ['spendx', 'spendy']]

        df = []
        for i in range(len(self.cfo)):
            df.append(pd.DataFrame({
                'errorx': errorx_period[i],
                'errory': errory_period[i],
                'spendx': spendx_period[i],
                'spendy': spendy_period[i],
                xlabel[0]: ppcof_summation_3sec[i],
                xlabel[1]: rpcof_summation_3sec[i],
                xlabel[2]: pfcof_summation_3sec[i],
                xlabel[3]: rfcof_summation_3sec[i],
            }))

            df[i]['Group'] = 'Group' + str(i + 1)

        df_all = pd.concat([i for i in df], axis=0)
        # print(df_all)

        marker_size = 5

        cmap = ['Blues_r', 'Blues']

        fig = plt.figure(figsize=(8, 4), dpi=300)
        for i in range(2):
            for j in range(4):
                ax = fig.add_subplot(2, 4, 4 * i + j + 1)
                ax.set_xlim(xlim[j][0], xlim[j][1])
                # ax.set_ylim(ylim[i][k][0], ylim[i][k][1])
                g = sns.scatterplot(data=df_all, x=xlabel[j], y=performance[i][j%2], hue='Group', s=marker_size)
                for lh in g.legend_.legendHandles:
                    lh.set_alpha(1)
                    lh._sizes = [10]

                ax.set_xlabel(xlabel[j])
                ax.set_ylabel(performance[i][j%2])

        plt.tight_layout()
        plt.savefig('fig/summation_performance_each_axis_' + str(self.group_type) + '.png')
        plt.show()

    def CFO_relation_axis(self):
        df_list = []
        for i in range(len(self.cfo)):
            data = self.cfo[i]
            for j in range(self.join):
                interface = 'i'+str(j+1)
                df_each = pd.DataFrame({
                    'Pitch PCFO': data[interface+'_p_pcfo'][self.start_num:self.end_num:1000],
                    'Roll PCFO': data[interface+'_r_pcfo'][self.start_num:self.end_num:1000],
                    'Pitch FCFO': data[interface+'_p_fcfo'][self.start_num:self.end_num:1000],
                    'Roll FCFO': data[interface+'_r_fcfo'][self.start_num:self.end_num:1000],
                })
                df_list.append(df_each)
        df = pd.concat([i for i in df_list], axis=0)
        df = df.reset_index()

        # plt.figure(figsize=(8, 4), dpi=300)

        kwargs = dict(
            ratio=10,
            height=10,
            kind='hist',
            # kind="kde",
            # kind="scatter",
            )

        kwargs_PCFO = dict(
            xlim=(-0.25, 0.25),
            ylim=(-0.25, 0.25),
            # gridsize=500,
        )

        sns.set(font_scale=1)
        sns.jointplot(data=df, x='Pitch PCFO', y='Roll PCFO', **kwargs, **kwargs_PCFO)

        kwargs_FCFO = dict(
            xlim=(-2.5, 2.5),
            ylim=(-2.5, 2.5),
            # gridsize=300,
        )

        sns.jointplot(data=df, x='Pitch FCFO', y='Roll FCFO', **kwargs, **kwargs_FCFO)

        kwargs_PCFO_FCFO = dict(
            xlim=(-0.25, 0.25),
            ylim=(-2.5, 2.5),
            # gridsize=500,
        )

        sns.set(font_scale=1)
        sns.jointplot(data=df, x='Pitch PCFO', y='Pitch FCFO', **kwargs, **kwargs_PCFO_FCFO)

        sns.jointplot(data=df, x='Roll PCFO', y='Roll FCFO', **kwargs, **kwargs_PCFO_FCFO)

        plt.tight_layout()
        plt.show()

    def CFO_relation_axis_3sec(self):

        df_list = []
        for i in range(len(self.cfo)):
            data = self.cfo[i]
            for j in range(self.join):
                interface = 'i' + str(j + 1)
                period_ppcfo = CFO.period_calculation(self, data[interface + '_p_pcfo'][self.start_num:self.end_num])
                period_rpcfo = CFO.period_calculation(self, data[interface + '_r_pcfo'][self.start_num:self.end_num])
                period_pfcfo = CFO.period_calculation(self, data[interface + '_p_fcfo'][self.start_num:self.end_num])
                period_rfcfo = CFO.period_calculation(self, data[interface + '_r_fcfo'][self.start_num:self.end_num])


                df_each = pd.DataFrame({
                    'Pitch PCFO': period_ppcfo,
                    'Roll PCFO': period_rpcfo,
                    'Pitch FCFO': period_pfcfo,
                    'Roll FCFO': period_rfcfo,
                })
                df_list.append(df_each)
        df = pd.concat([i for i in df_list], axis=0)
        df = df.reset_index()

        # plt.figure(figsize=(8, 4), dpi=300)

        kwargs = dict(
            ratio=10,
            height=10,
            kind='hist',
            # kind="kde",
            # kind="scatter",
            # kind='hex'
        )

        kwargs_PCFO = dict(
            xlim=(-0.25, 0.25),
            ylim=(-0.25, 0.25),
            # gridsize=20,
        )

        sns.set(font_scale=1)
        sns.jointplot(data=df, x='Pitch PCFO', y='Roll PCFO', **kwargs, **kwargs_PCFO)

        kwargs_FCFO = dict(
            xlim=(-2.5, 2.5),
            ylim=(-2.5, 2.5),
            # gridsize=15,
        )

        sns.jointplot(data=df, x='Pitch FCFO', y='Roll FCFO', **kwargs, **kwargs_FCFO)

        kwargs_PCFO_FCFO = dict(
            xlim=(-0.25, 0.25),
            ylim=(-2.5, 2.5),
            # gridsize=20,
        )

        sns.set(font_scale=1)
        sns.jointplot(data=df, x='Pitch PCFO', y='Pitch FCFO', **kwargs, **kwargs_PCFO_FCFO)

        sns.jointplot(data=df, x='Roll PCFO', y='Roll FCFO', **kwargs, **kwargs_PCFO_FCFO)

        plt.tight_layout()
        plt.show()

    def summation_ave_performance(self, mode='noabs'):
        error_period, spend_period = CFO.period_performance_cooperation(self)
        pcfo_summation_3sec, fcfo_summation_3sec, pcfo_ave_summation_3sec, fcfo_ave_summation_3sec = CFO.summation_cfo_3sec(self, mode)
        if mode == 'noabs':
            xlabel = ['Summation PCFO',
                      'Summation FCFO',
                      'Summation abs. PCFO',
                      'Summation abs. FCFO',
                      ]
            xlim = [
                [-0.04, 0.04],
                [-0.04, 0.04],
                [-0.2, 0.2],
                [-0.2, 0.2],
            ]
        elif mode == 'b_abs':
            xlabel = ['Before abs. summation PCFO',
                      'Before abs. summation FCFO',
                      'Before abs. summation abs. PCFO',
                      'Before abs. summation abs. FCFO',
                      ]
            xlim = [
                [0.0, 0.15],
                [0.0, 0.15],
                [0.0, 2.0],
                [0.0, 1.5],
            ]
        elif mode == 'a_abs':
            xlabel = ['After abs. summation PCFO',
                      'After abs. summation FCFO',
                      'After abs. summation abs. PCFO',
                      'After abs. summation abs. FCFO',
                      ]
            xlim = [
                [0.0, 0.12],
                [0.0, 0.12],
                [0.0, 0.7],
                [0.0, 0.6],
            ]

        performance = ['error', 'spend']

        df = []
        for i in range(len(self.cfo)):
            df.append(pd.DataFrame({
                performance[0]: error_period[i],
                performance[1]: spend_period[i],
                xlabel[0]: pcfo_summation_3sec[i],
                xlabel[1]: fcfo_summation_3sec[i],
                xlabel[2]: pcfo_ave_summation_3sec[i],
                xlabel[3]: fcfo_ave_summation_3sec[i],
            }))

            df[i]['Group'] = 'Group' + str(i + 1)

        df_all = pd.concat([i for i in df], axis=0)
        # print(df_all)

        marker_size = 5

        cmap = ['Blues_r', 'Blues']

        fig = plt.figure(figsize=(8, 4), dpi=300)
        for i in range(2):
            for j in range(4):
                ax = fig.add_subplot(2, 4, 4 * i + j + 1)
                ax.set_xlim(xlim[j][0], xlim[j][1])
                # ax.set_ylim(ylim[i][k][0], ylim[i][k][1])
                g = sns.scatterplot(data=df_all, x=xlabel[j], y=performance[i], hue='Group', s=marker_size)
                for lh in g.legend_.legendHandles:
                    lh.set_alpha(1)
                    lh._sizes = [10]

                ax.set_xlabel(xlabel[j])
                ax.set_ylabel(performance[i])

        plt.tight_layout()
        plt.show()

    def subtraction_ave_performance(self):
        error_period, spend_period = CFO.period_performance_cooperation(self)
        ppcof_subtraction_3sec, rpcof_subtraction_3sec, pfcof_subtraction_3sec, rfcof_subtraction_3sec = CFO.subtraction_ave_cfo_3sec(self)

        xlabel = ['Subtraction PCFO', 'Subtraction PCFO', 'Subtraction abs. FCFO', 'Subtraction abs. FCFO']

        df = []
        for i in range(len(self.cfo)):
            df.append(pd.DataFrame({
                'error': error_period[i],
                'spend': spend_period[i],
                xlabel[0]: ppcof_subtraction_3sec[i],
                xlabel[1]: rpcof_subtraction_3sec[i],
                xlabel[2]: pfcof_subtraction_3sec[i],
                xlabel[3]: rfcof_subtraction_3sec[i],
            }))

            df[i]['Group'] = 'Group' + str(i + 1)

        df_all = pd.concat([i for i in df], axis=0)
        # print(df_all)

        marker_size = 5

        performance = ['error', 'spend']
        cmap = ['Blues_r', 'Blues']

        xlim = [0.4, 0.4, 6.0, 6.0]

        fig = plt.figure(figsize=(8, 4), dpi=300)
        for i in range(2):
            for j in range(4):
                ax = fig.add_subplot(2, 4, 4 * i + j + 1)
                ax.set_xlim(0, xlim[j])
                # ax.set_ylim(ylim[i][k][0], ylim[i][k][1])
                g = sns.scatterplot(data=df_all, x=xlabel[j], y=performance[i], hue='Group', s=marker_size)
                for lh in g.legend_.legendHandles:
                    lh.set_alpha(1)
                    lh._sizes = [10]

                ax.set_xlabel(xlabel[j])
                ax.set_ylabel(performance[i])

        plt.tight_layout()
        plt.show()


    def summation_ave_cfo(self, graph=1, mode='noabs'):
        summation = self.cfo[0]['i1_pcfo_sum'][self.start_num:self.end_num]
        types = ['_pcfo_sum', '_fcfo_sum', '_pcfo_sum_abs', '_fcfo_sum_abs']
        for type in types:
            for j in range(len(self.cfo)):
                data = self.cfo[j]
                summation_ = data['i1_pcfo_sum'][self.start_num:self.end_num]
                for i in range(self.join):
                    interfacenum = 'i' + str(i + 1)
                    pcfoname = interfacenum + type

                    summation_ = np.vstack((summation_, data[pcfoname][self.start_num:self.end_num]))
                summation_ = np.delete(summation_, 0, 0)
                if mode == 'b_abs':
                    summation_ = np.abs(summation_)
                summation = np.vstack((summation, np.sum(summation_, axis=0)))

            # print(summation_)
        summation = np.delete(summation, 0, 0)
        # print(summation.shape)
        summation = summation.reshape([4, len(self.cfo), -1])
        summation = summation / self.cfo[0]['join'][0]
        if mode == 'a_abs':
            summation = np.abs(summation)

        if graph == 0:
            fig, (pcfo, fcfo, pcfoave, fcfoave) = plt.subplots(4, 1, figsize=(5, 7), dpi=150, sharex=True)

            # plt.xlim([10, 60])  # x軸の範囲
            # plt.xlim([0.28, 0.89])  # x軸の範囲
            plt.xlabel("Time (sec)")

            for i in range(len(self.cfo)):
                data = self.cfo[i]

                pcfo.plot(data['time'][self.start_num:self.end_num:10], summation[0][i][::10],
                           label='Group' + str(i + 1))
                fcfo.plot(data['time'][self.start_num:self.end_num:10], summation[1][i][::10],
                           label='Group' + str(i + 1))
                pcfoave.plot(data['time'][self.start_num:self.end_num:10], summation[2][i][::10],
                           label='Group' + str(i + 1))
                fcfoave.plot(data['time'][self.start_num:self.end_num:10], summation[3][i][::10],
                           label='Group' + str(i + 1))

            pcfo.set_ylabel('Summation\nPCFO (rad)')
            pcfo.set_ylabel('Summation\nFCFO (Nm)')
            fcfoave.set_ylabel('Summation\nabs. PCFO (rad)')
            fcfoave.set_ylabel('Summation\nabs. FCFO (Nm)')

            pcfo.legend(ncol=10, columnspacing=1, loc='upper left')
            fcfo.legend(ncol=10, columnspacing=1, loc='upper left')
            pcfoave.legend(ncol=10, columnspacing=1, loc='upper left')
            fcfoave.legend(ncol=10, columnspacing=1, loc='upper left')

            pcfo.set_yticks(np.arange(-10, 10, 0.5))
            fcfo.set_yticks(np.arange(-10, 10, 0.5))
            pcfoave.set_yticks(np.arange(-8.0, 8.0, 1.0))
            fcfoave.set_yticks(np.arange(-8.0, 8.0, 1.0))

            pcfo.set_ylim([0, 1.0])  # y軸の範囲
            fcfo.set_ylim([0, 1.0])  # y軸の範囲
            pcfoave.set_ylim([0, 4.0])  # y軸の範囲
            fcfoave.set_ylim([0, 4.0])  # y軸の範囲

            plt.tight_layout()
            # plt.savefig(savename)
            plt.show()


        return summation[0], summation[1], summation[2], summation[3]

    def subtraction_ave_cfo(self, graph=1):
        subtraction = self.cfo[0]['i1_pcfo_sum'][self.start_num:self.end_num]
        types = ['pcfo_sum', 'fcfo_sum', 'pcfo_sum_abs', 'fcfo_sum_abs']
        for type in types:
            subtraction_ = self.cfo[0]['i1_pcfo_sum'][self.start_num:self.end_num]
            for i in range(len(self.cfo)):
                data = self.cfo[i]
                if self.group_type == 'dyad':
                    sub_cfo1 = np.abs(np.subtract(data['i1_' + type][self.start_num:self.end_num], data['i2_' + type][self.start_num:self.end_num]))
                    sub_cfo2 = np.abs(np.subtract(data['i2_' + type][self.start_num:self.end_num], data['i1_' + type][self.start_num:self.end_num]))
                    sub_cfo_ave = np.add(np.abs(sub_cfo1), np.abs(sub_cfo2)) / 2
                    subtraction_ = np.vstack((subtraction_, sub_cfo_ave))

                elif self.group_type == 'triad':
                    sub_cfo1 = np.subtract(np.subtract(2 * data['i1_' + type][self.start_num:self.end_num], data['i2_' + type][self.start_num:self.end_num]), data['i3_' + type][self.start_num:self.end_num])
                    sub_cfo2 = np.subtract(np.subtract(2 * data['i2_' + type][self.start_num:self.end_num], data['i1_' + type][self.start_num:self.end_num]), data['i3_' + type][self.start_num:self.end_num])
                    sub_cfo3 = np.subtract(np.subtract(2 * data['i3_' + type][self.start_num:self.end_num], data['i1_' + type][self.start_num:self.end_num]), data['i2_' + type][self.start_num:self.end_num])
                    sub_cfo_ave = np.add(np.add(np.abs(sub_cfo1), np.abs(sub_cfo2)), np.abs(sub_cfo3)) / 3
                    subtraction_ = np.vstack((subtraction_, sub_cfo_ave))

                elif self.group_type == 'tetrad':
                    sub_cfo1 = np.subtract(np.subtract(np.subtract(3 * data['i1_' + type][self.start_num:self.end_num], data['i2_' + type][self.start_num:self.end_num]), data['i3_' + type][self.start_num:self.end_num]), data['i4_' + type][self.start_num:self.end_num])
                    sub_cfo2 = np.subtract(np.subtract(np.subtract(3 * data['i2_' + type][self.start_num:self.end_num], data['i1_' + type][self.start_num:self.end_num]), data['i3_' + type][self.start_num:self.end_num]), data['i4_' + type][self.start_num:self.end_num])
                    sub_cfo3 = np.subtract(np.subtract(np.subtract(3 * data['i3_' + type][self.start_num:self.end_num], data['i1_' + type][self.start_num:self.end_num]), data['i2_' + type][self.start_num:self.end_num]), data['i4_' + type][self.start_num:self.end_num])
                    sub_cfo4 = np.subtract(np.subtract(np.subtract(3 * data['i4_' + type][self.start_num:self.end_num], data['i1_' + type][self.start_num:self.end_num]), data['i2_' + type][self.start_num:self.end_num]), data['i3_' + type][self.start_num:self.end_num])
                    sub_cfo_ave = np.add(np.add(np.add(np.abs(sub_cfo1), np.abs(sub_cfo2)), np.abs(sub_cfo3)), np.abs(sub_cfo4)) / 4
                    subtraction_ = np.vstack((subtraction_, sub_cfo_ave))

            subtraction_ = np.delete(subtraction_, 0, 0)
            subtraction = np.vstack((subtraction, subtraction_))
        subtraction = np.delete(subtraction, 0, 0)
        subtraction = subtraction.reshape([4, len(self.cfo), -1])

        if graph == 0:
            fig, (pcfo, fcfo, pcfoave, fcfoave) = plt.subplots(4, 1, figsize=(5, 7), dpi=150, sharex=True)

            # plt.xlim([10, 60])  # x軸の範囲
            # plt.xlim([0.28, 0.89])  # x軸の範囲
            plt.xlabel("Time (sec)")

            for i in range(len(self.cfo)):
                data = self.cfo[i]

                pcfo.plot(data['time'][self.start_num:self.end_num:10], subtraction[0][i][::10], label='Group' + str(i + 1))
                fcfo.plot(data['time'][self.start_num:self.end_num:10], subtraction[1][i][::10], label='Group' + str(i + 1))
                pcfoave.plot(data['time'][self.start_num:self.end_num:10], subtraction[2][i][::10], label='Group' + str(i + 1))
                fcfoave.plot(data['time'][self.start_num:self.end_num:10], subtraction[3][i][::10], label='Group' + str(i + 1))

            pcfo.set_ylabel('Subtraction\nPCFO (rad)')
            fcfo.set_ylabel('Subtraction\nFCFO (Nm)')
            pcfoave.set_ylabel('Subtraction\nabs. PCFO (rad)')
            fcfoave.set_ylabel('Subtraction\nabs. FCFO (Nm)')

            pcfo.legend(ncol=10, columnspacing=1, loc='upper left')
            fcfo.legend(ncol=10, columnspacing=1, loc='upper left')
            pcfoave.legend(ncol=10, columnspacing=1, loc='upper left')
            fcfoave.legend(ncol=10, columnspacing=1, loc='upper left')

            pcfo.set_yticks(np.arange(-10, 10, 0.5))
            fcfo.set_yticks(np.arange(-10, 10, 0.5))
            pcfoave.set_yticks(np.arange(-8.0, 8.0, 1.0))
            fcfoave.set_yticks(np.arange(-8.0, 8.0, 1.0))

            pcfo.set_ylim([0, 1.0])  # y軸の範囲
            fcfo.set_ylim([0, 1.0])  # y軸の範囲
            pcfoave.set_ylim([0, 10.0])  # y軸の範囲
            fcfoave.set_ylim([0, 10.0])  # y軸の範囲

            plt.tight_layout()
            # plt.savefig(savename)
            plt.show()

        return subtraction[0], subtraction[1], subtraction[2], subtraction[3]


    def summation_ave_cfo_3sec(self, mode='noabs'):
        pcfo_summation, fcfo_summation, pcfo_abs_summation, fcfo_abs_summation = CFO.summation_ave_cfo(self, graph=1, mode=mode)

        pcfo_summation_3sec = pcfo_summation.reshape([len(self.cfo), -1, self.num])
        pcfo_summation_3sec = np.average(pcfo_summation_3sec, axis=2)

        fcfo_summation_3sec = fcfo_summation.reshape([len(self.cfo), -1, self.num])
        fcfo_summation_3sec = np.average(fcfo_summation_3sec, axis=2)

        pcfo_ave_summation_3sec = pcfo_abs_summation.reshape([len(self.cfo), -1, self.num])
        pcfo_ave_summation_3sec = np.average(pcfo_ave_summation_3sec, axis=2)

        fcfo_ave_summation_3sec = fcfo_abs_summation.reshape([len(self.cfo), -1, self.num])
        fcfo_ave_summation_3sec = np.average(fcfo_ave_summation_3sec, axis=2)

        return pcfo_summation_3sec, fcfo_summation_3sec, pcfo_ave_summation_3sec, fcfo_ave_summation_3sec


    def subtraction_ave_cfo_3sec(self):
        pcfo_subtraction, fcfo_subtraction, pcfo_ave_subtraction, fcfo_ave_subtraction = CFO.subtraction_ave_cfo(self)

        pcfo_subtraction_3sec = pcfo_subtraction.reshape([len(self.cfo), -1, self.num])
        pcfo_subtraction_3sec = np.average(pcfo_subtraction_3sec, axis=2)

        fcfo_subtraction_3sec = fcfo_subtraction.reshape([len(self.cfo), -1, self.num])
        fcfo_subtraction_3sec = np.average(fcfo_subtraction_3sec, axis=2)

        pcfo_ave_subtraction_3sec = pcfo_ave_subtraction.reshape([len(self.cfo), -1, self.num])
        pcfo_ave_subtraction_3sec = np.average(pcfo_ave_subtraction_3sec, axis=2)

        fcfo_ave_subtraction_3sec = fcfo_ave_subtraction.reshape([len(self.cfo), -1, self.num])
        fcfo_ave_subtraction_3sec = np.average(fcfo_ave_subtraction_3sec, axis=2)

        return pcfo_subtraction_3sec, fcfo_subtraction_3sec, pcfo_ave_subtraction_3sec, fcfo_ave_subtraction_3sec
