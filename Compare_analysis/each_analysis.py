import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import os

import pandas as pd
import seaborn as sns

class each:
    def __init__(self, read_data, group_type):
        self.group_type = group_type

        self.data = read_data

        self.smp = 0.0001  # サンプリング時間
        self.time = 3.0  # ターゲットの移動時間
        self.eliminationtime = 0.0  # 消去時間
        self.starttime = 20.0  # タスク開始時間
        self.endtime = 80.0  # タスク終了時間
        self.tasktime = self.endtime - self.starttime  # タスクの時間
        self.period = int((self.tasktime - self.eliminationtime) / self.time)  # 回数
        self.num = int(self.time / self.smp)  # 1ピリオドにおけるデータ数
        self.start_num = int((self.starttime) / self.smp)
        self.end_num = int((self.endtime) / self.smp)
        self.nn_read_flag = False
        self.join = self.data[0]['join'][0]

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
        for j in range(len(self.data)):
            fig, (rthm, pthm, rtext, ptext) = plt.subplots(4, 1, figsize=(5, 5), dpi=150, sharex=True)

            plt.xticks(np.arange(self.starttime, self.endtime * 2, self.time * 2))
            plt.xlim([self.starttime, self.endtime])  # x軸の範囲
            plt.xlabel("Time (sec)")

            data = self.data[j]
            for i in range(self.join):
                interfacenum = 'i' + str(i + 1)

                thmname = interfacenum + '_r_thm'
                thm_prename = interfacenum + '_r_thm_pre'
                thm_prename_solo = interfacenum + '_r_thm_pre_solo'
                rthm.plot(data['time'][self.start_num:self.end_num:10], data[thmname][self.start_num:self.end_num:10], label='P' + str(i + 1) + '_act')
                # rthm.plot(data['time'][self.start_num:self.end_num:10], data[thm_prename][self.start_num:self.end_num:10], label='P' + str(i + 1) + '_pre')
                # rthm.plot(data['time'][self.start_num:self.end_num:10], data[thm_prename_solo][self.start_num:self.end_num:10], label='P' + str(i + 1) + '_pre_solo')

                thmname = interfacenum + '_p_thm'
                thm_prename = interfacenum + '_p_thm_pre'
                thm_prename_solo = interfacenum + '_p_thm_pre_solo'
                pthm.plot(data['time'][self.start_num:self.end_num:10], data[thmname][self.start_num:self.end_num:10], label='P'+str(i+1)+'_act')
                # pthm.plot(data['time'][self.start_num:self.end_num:10], data[thm_prename][self.start_num:self.end_num:10], label='P'+str(i+1)+'_pre')
                # pthm.plot(data['time'][self.start_num:self.end_num:10], data[thm_prename_solo][self.start_num:self.end_num:10], label='P'+str(i+1)+'_pre_solo')

                textname = interfacenum + '_r_text'
                text_prename = interfacenum + '_r_text_pre'
                text_prename_solo = interfacenum + '_r_text_pre_solo'
                rtext.plot(data['time'][self.start_num:self.end_num:10], data[textname][self.start_num:self.end_num:10], label='P' + str(i + 1) + '_act')
                # rtext.plot(data['time'][self.start_num:self.end_num:10], data[text_prename][self.start_num:self.end_num:10], label='P' + str(i + 1) + '_pre')
                # rtext.plot(data['time'][self.start_num:self.end_num:10], data[text_prename_solo][self.start_num:self.end_num:10], label='P' + str(i + 1) + '_pre_solo')


                textname = interfacenum + '_p_text'
                text_prename = interfacenum + '_p_text_pre'
                text_prename_solo = interfacenum + '_p_text_pre_solo'
                ptext.plot(data['time'][self.start_num:self.end_num:10], data[textname][self.start_num:self.end_num:10], label='P'+str(i+1)+'_act')
                # ptext.plot(data['time'][self.start_num:self.end_num:10], data[text_prename][self.start_num:self.end_num:10], label='P'+str(i+1)+'_pre')
                # ptext.plot(data['time'][self.start_num:self.end_num:10], data[text_prename_solo][self.start_num:self.end_num:10], label='P'+str(i+1)+'_pre_solo')

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
        for i in range(len(self.data)):
            data = self.data[i]

            fig, (x, y) = plt.subplots(2, 1, figsize=(5, 5), dpi=150, sharex=True)

            x.plot(data['time'][self.start_num:self.end_num:10], data['targetx'][self.start_num:self.end_num:10], label='Target')
            # x.plot(data['time'][self.start_num:self.end_num:10], data['targetx_act'][self.start_num:self.end_num:10], label='Target_act')
            x.plot(data['time'][self.start_num:self.end_num:10], data['ballx'][self.start_num:self.end_num:10], label='Ball(H-H)')
            # x.plot(data['time'][self.start_num:self.end_num:10], data['ballx_pre'][self.start_num:self.end_num:10], label='Ball(M-M)')
            # x.plot(data['pre_time'], data['pre_ball_x'], label='pre_ballx')
            x.set_ylabel('X-axis Position (m)')
            x.legend(ncol=2, columnspacing=1, loc='upper left')
            x.set_ylim([-0.2, 0.2])  # y軸の範囲


            y.plot(data['time'][self.start_num:self.end_num:10], data['targety'][self.start_num:self.end_num:10], label='Target')
            # y.plot(data['time'][self.start_num:self.end_num:10], data['targety_act'][self.start_num:self.end_num:10], label='Target_act')
            y.plot(data['time'][self.start_num:self.end_num:10], data['bally'][self.start_num:self.end_num:10], label='Ball(H-H)')
            # y.plot(data['time'][self.start_num:self.end_num:10], data['bally_pre'][self.start_num:self.end_num:10], label='Ball(M-M)')
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
        for i in range(len(self.data)):
            data = self.data[i]

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


    def performance_calc(self, data, ballx, bally):
        error = np.sqrt(
            (data['targetx'][self.start_num:self.end_num] - ballx[self.start_num:self.end_num]) ** 2
            + (data['targety'][self.start_num:self.end_num] - bally[self.start_num:self.end_num]) ** 2)

        target_size = 0.03
        spent = np.where(error < target_size, 1, 0)
        # spent = numpy.where(error < self.data['targetsize'], 1, 0)

        return error, spent

    def period_performance(self, graph=1):
        error_period = []
        spent_period = []
        for i in range(len(self.data)):
            data = self.data[i]

            error, spent = each.performance_calc(self, data, data['ballx'], data['bally'])
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

        error_period_array = np.vstack((error_period[_] for _ in range(len(error_period))))
        spent_period_array = np.vstack((spent_period[_] for _ in range(len(spent_period))))
        # spent_period_array = np.vstack((i for i in spent_period))

        return error_period_array, spent_period_array

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


    def subtraction_position(self, graph=1):
        subtraction = self.data[0]['i1_p_thm'][self.start_num:self.end_num]
        types = ['p_thm', 'r_thm']
        for type in types:
            subtraction_ = self.data[0]['i1_p_thm'][self.start_num:self.end_num]
            for i in range(len(self.data)):
                data = self.data[i]

                sub = np.abs(np.subtract(data['i1_' + type][self.start_num:self.end_num], data['i2_' + type][self.start_num:self.end_num]))

                subtraction_ = np.vstack((subtraction_, sub))

            subtraction_ = np.delete(subtraction_, 0, 0)
            subtraction = np.vstack((subtraction, subtraction_))
        subtraction = np.delete(subtraction, 0, 0)
        subtraction = subtraction.reshape([len(types), len(self.data), -1])

        if graph == 0:
            fig, (psub, rsub) = plt.subplots(2, 1, figsize=(5, 7), dpi=150, sharex=True)

            # plt.xlim([10, 60])  # x軸の範囲
            # plt.xlim([0.28, 0.89])  # x軸の範囲
            plt.xlabel("Time (sec)")

            for i in range(len(self.data)):
                data = self.data[i]

                psub.plot(data['time'][self.start_num:self.end_num:10], subtraction[0][i][::10], label='Group' + str(i + 1))
                rsub.plot(data['time'][self.start_num:self.end_num:10], subtraction[1][i][::10], label='Group' + str(i + 1))

            psub.set_ylabel('Subtraction\nPosition (rad)')
            rsub.set_ylabel('Subtraction\nPosition (rad)')

            psub.legend(ncol=10, columnspacing=1, loc='upper left')
            rsub.legend(ncol=10, columnspacing=1, loc='upper left')

            psub.set_yticks(np.arange(-10, 10, 0.5))
            rsub.set_yticks(np.arange(-10, 10, 0.5))

            psub.set_ylim([0, 1.0])  # y軸の範囲
            rsub.set_ylim([0, 1.0])  # y軸の範囲

            plt.tight_layout()
            os.makedirs('fig/subtraction_position', exist_ok=True)
            plt.savefig('fig/subtraction_position/' + str(self.group_type) + '.png')
            plt.show()


        return subtraction[0], subtraction[1]


    def subtraction_position_3sec(self):
        pthm_subtraction, rthm_subtraction= each.subtraction_position(self)

        pthm_subtraction_3sec = pthm_subtraction.reshape([len(self.data), -1, self.num])
        pthm_subtraction_3sec = np.average(pthm_subtraction_3sec, axis=2)

        rthm_subtraction_3sec = rthm_subtraction.reshape([len(self.data), -1, self.num])
        rthm_subtraction_3sec = np.average(rthm_subtraction_3sec, axis=2)

        return pthm_subtraction_3sec, rthm_subtraction_3sec

    def subtraction_position_ave(self):
        # pthm_subtraction, rthm_subtraction = each.subtraction_position(self)
        #
        # pthm_subtraction_ave = np.average(pthm_subtraction, axis=1)
        #
        # rthm_subtraction_ave = np.average(rthm_subtraction, axis=1)
        #
        # return pthm_subtraction_ave, rthm_subtraction_ave

        pthm_subtraction_3sec, rthm_subtraction_3sec = each.subtraction_position_3sec(self)

        pthm_subtraction_3sec_ave = np.average(pthm_subtraction_3sec, axis=1)

        rthm_subtraction_3sec_ave = np.average(rthm_subtraction_3sec, axis=1)

        return pthm_subtraction_3sec_ave, rthm_subtraction_3sec_ave



    def summation_force(self, graph=1, mode='abs'):
        summation = self.data[0]['i1_p_text'][self.start_num:self.end_num]
        types = ['_p_text', '_r_text']
        for type in types:
            for j in range(len(self.data)):
                data = self.data[j]
                summation_ = data['i1_p_text'][self.start_num:self.end_num]
                for i in range(self.join):
                    interfacenum = 'i' + str(i + 1)
                    dataname = interfacenum + type

                    summation_ = np.vstack((summation_, data[dataname][self.start_num:self.end_num]))
                summation_ = np.delete(summation_, 0, 0)
                # summation_ = np.abs(summation_)
                summation = np.vstack((summation, np.sum(summation_, axis=0)))

            # print(summation_)
        summation = np.delete(summation, 0, 0)
        # print(summation.shape)
        summation = summation.reshape([len(types), len(self.data), -1])
        # summation = summation / self.cfo[0]['join'][0]
        if mode == 'abs':
            summation = np.abs(summation)

        if graph == 0:
            fig, (ptext, rtxt) = plt.subplots(2, 1, figsize=(5, 7), dpi=150, sharex=True)

            # plt.xlim([10, 60])  # x軸の範囲
            # plt.xlim([0.28, 0.89])  # x軸の範囲
            plt.xlabel("Time (sec)")

            for i in range(len(self.data)):
                data = self.data[i]

                ptext.plot(data['time'][self.start_num:self.end_num:10], summation[0][i][::10],
                           label='Group' + str(i + 1))
                rtxt.plot(data['time'][self.start_num:self.end_num:10], summation[1][i][::10],
                           label='Group' + str(i + 1))

            ptext.set_ylabel('Summation\nforce (Nm)')
            rtxt.set_ylabel('Summation\nforce (Nm)')

            ptext.legend(ncol=10, columnspacing=1, loc='upper left')
            rtxt.legend(ncol=10, columnspacing=1, loc='upper left')

            ptext.set_yticks(np.arange(-10, 10, 0.5))
            rtxt.set_yticks(np.arange(-10, 10, 0.5))

            ptext.set_ylim([0, 5.0])  # y軸の範囲
            rtxt.set_ylim([0, 5.0])  # y軸の範囲

            plt.tight_layout()
            os.makedirs('fig/summation_force', exist_ok=True)
            plt.savefig('fig/summation_force/' + str(self.group_type) + '.png')
            plt.show()


        return summation[0], summation[1]


    def summation_force_3sec(self):
        ptext_summation, rtext_summation= each.summation_force(self)

        ptext_summation_3sec = ptext_summation.reshape([len(self.data), -1, self.num])
        ptext_summation_3sec = np.average(ptext_summation_3sec, axis=2)

        rtext_summation_3sec = rtext_summation.reshape([len(self.data), -1, self.num])
        rtext_summation_3sec = np.average(rtext_summation_3sec, axis=2)

        return ptext_summation_3sec, rtext_summation_3sec

    def estimation_task_inertia(self, graph=1):
        pitch_sum_force, roll_sum_force = each.summation_force(self, mode='no_abs')

        if graph == 0:
            for i in range(len(self.data)):
                data = self.data[i]

                fig, (x, y) = plt.subplots(2, 1, figsize=(5, 5), dpi=150, sharex=True)

                x.scatter(pitch_sum_force[0][::10], data['pitch_ddot'][self.start_num:self.end_num:10], s=2)
                x.set_xlabel('sumation force')
                x.set_ylabel('pitch_ddot')
                # x.legend(ncol=2, columnspacing=1, loc='upper left')
                x.set_ylim([-20, 20.0])  # y軸の範囲
                x.set_xlim([-4, 4.0])  # y軸の範囲

                y.scatter(roll_sum_force[0][::10], data['roll_ddot'][self.start_num:self.end_num:10], s=2)
                y.set_xlabel('sumation force')
                y.set_ylabel('roll_ddot')
                # x.legend(ncol=2, columnspacing=1, loc='upper left')
                y.set_ylim([-20, 20.0])  # y軸の範囲
                y.set_xlim([-4, 4.0])  # y軸の範囲

                plt.tight_layout()
                # plt.savefig("First_time_target_movement.png")
            plt.show()

        # print(self.data[0]['pitch_ddot'][self.start_num:self.end_num:10])
        pitch_ddot = np.vstack((self.data[_]['pitch_ddot'][self.start_num:self.end_num] for _ in range(len(self.data))))
        roll_ddot = np.vstack((self.data[_]['roll_ddot'][self.start_num:self.end_num] for _ in range(len(self.data))))
        # print(pitch_ddot.shape)

        return pitch_ddot, roll_ddot

    def get_plate_dot (self):
        pitch_dot = np.vstack((self.data[_]['pitch_dot'][self.start_num:self.end_num] for _ in range(len(self.data))))
        roll_dot = np.vstack((self.data[_]['roll_dot'][self.start_num:self.end_num] for _ in range(len(self.data))))

        return pitch_dot, roll_dot


    def variance_calculation(self, data):
        variance = np.var(data, axis=0)
        return variance

    def work_calc(self, mode='human'):
        flabel = ['_p_text_tf', '_r_text_tf']
        plabel = ['_p_thm_tf', '_r_thm_tf']
        flabel = ['_p_text', '_r_text']
        plabel = ['_p_thm', '_r_thm']

        if mode == 'model':
            flabel = ['_p_text_pre_tf', '_r_text_pre_tf']
            plabel = ['_p_thm_pre_tf', '_r_thm_pre_tf']
            flabel = ['_p_text_pre', '_r_text_pre']
            plabel = ['_p_thm_pre', '_r_thm_pre']

        omegac =10.0
        Ts = 0.0001

        work = []
        for i in range(len(self.data)):
            data = self.data[i]
            for j in range(self.join):
                interfacenum = 'i' + str(j + 1)
                for k in range(len(flabel)):
                    thm_data = data[interfacenum + plabel[k]][self.start_num:self.end_num]
                    thm_data_lpf = CFO.lpf_1st_order(self, thm_data, omegac, Ts)
                    thm_data_ = np.append(np.zeros(1), thm_data_lpf)
                    thm_diff = np.diff(thm_data_)

                    text_data = data[interfacenum + flabel[k]][self.start_num:self.end_num]
                    text_data_lpf = CFO.lpf_1st_order(self, text_data, omegac, Ts)

                    diff = thm_diff * text_data_lpf

                    work.append(diff)

        work_np_ = np.concatenate([work[_] for _ in range(len(work))])
        work_np = work_np_.reshape([len(self.data), self.join, len(flabel), -1])
        return work_np

    def lpf_1st_order(self, data, omegac, Ts):
        T = omegac * Ts

        data_now = 0.0
        data_old = 0.0
        data_lpf_now = 0.0
        data_lpf_old = 0.0

        data_lpf = np.zeros(len(data))
        for l in range(len(data)):
            data_now = data[l]
            data_lpf_now = (2-T)/(2+T)*data_lpf_old + T/(2+T)*(data_now + data_old)
            data_lpf[l] = data_lpf_now
            data_lpf_old = data_lpf_now
            data_old = data_now

        return data_lpf