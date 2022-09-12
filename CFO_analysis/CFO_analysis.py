import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

import pandas as pd
import seaborn as sns


class CFO:
    def __init__(self, cfo_data, group_type):
        self.rpcof_summation = None
        self.ppcof_summation = None
        self.pfcof_summation = None
        self.rfcof_summation = None
        self.group_type = group_type

        self.cfo = cfo_data

        self.smp = 0.0001  # サンプリング時間
        self.time = 3.0  # ターゲットの移動時間
        self.eliminationtime = 0.0  # 消去時間
        self.starttime = 20.0  # タスク開始時間
        self.endtime = 80.0  # タスク終了時間
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
            fig, (thm, text) = plt.subplots(2, 1, figsize=(5, 5), dpi=150, sharex=True)

            # plt.xlim([10, 60])  # x軸の範囲
            # plt.xlim([0.28, 0.89])  # x軸の範囲
            plt.xlabel("Time[sec]")

            data = self.cfo[j]
            for i in range(self.join):
                interfacenum = 'i' + str(i + 1)
                thmname = interfacenum + '_p_thm'
                thm_prename = interfacenum + '_p_thm_pre'
                thm.plot(data['time'][self.start_num:self.end_num:10], data[thmname][self.start_num:self.end_num:10], label='P'+str(i+1)+'_act')
                thm.plot(data['time'][self.start_num:self.end_num:10], data[thm_prename][self.start_num:self.end_num:10], label='P'+str(i+1)+'_pre')


                textname = interfacenum + '_p_text'
                text_prename = interfacenum + '_p_text_pre'
                text.plot(data['time'][self.start_num:self.end_num:10], data[textname][self.start_num:self.end_num:10], label='P'+str(i+1)+'_act')
                text.plot(data['time'][self.start_num:self.end_num:10], data[text_prename][self.start_num:self.end_num:10], label='P'+str(i+1)+'_pre')

            thm.set_ylabel('Position [rad]')
            thm.legend(ncol=2, columnspacing=1, loc='upper left')
            thm.set_yticks(np.arange(-10, 10, 0.5))
            thm.set_ylim([-1.5, 1.5])  # y軸の範囲

            text.set_ylabel('Reaction torque[Nm]')
            text.legend(ncol=2, columnspacing=1, loc='upper left')
            text.set_yticks(np.arange(-8.0, 8.0, 2.0))
            text.set_ylim([-6.0, 6.0])  # y軸の範囲

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

        plt.ylabel(r'Position [m]')
        plt.yticks(np.arange(-4, 4, 0.1))
        plt.ylim([-0.4, 0.4])  # y軸の範囲
        plt.xlim([data[0]['starttime'], data[0]['endtime']])  # x軸の範囲
        plt.xlabel("Time[sec]")

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
            plt.xlabel("Time[sec]")

            for i in range(self.join):
                interfacenum = 'i' + str(i + 1)
                pcfoname = interfacenum + '_p_pcfo'
                fcfoname = interfacenum + '_p_fcfo'

                ppcfo.plot(data['time'][self.start_num:self.end_num:10], data[pcfoname][::10], label='P'+str(i+1))
                pfcfo.plot(data['time'][self.start_num:self.end_num:10], data[fcfoname][::10], label='P'+str(i+1))

                pcfoname = interfacenum + '_r_pcfo'
                fcfoname = interfacenum + '_r_fcfo'

                rpcfo.plot(data['time'][self.start_num:self.end_num:10], data[pcfoname][::10], label='P' + str(i + 1))
                rfcfo.plot(data['time'][self.start_num:self.end_num:10], data[fcfoname][::10], label='P' + str(i + 1))


            ppcfo.set_ylabel('Pitch PCFO [rad]')
            ppcfo.legend(ncol=2, columnspacing=1, loc='upper left')
            ppcfo.set_yticks(np.arange(-10, 10, 0.5))
            ppcfo.set_ylim([-1.5, 1.5])  # y軸の範囲

            rpcfo.set_ylabel('Roll PCFO [rad]')
            rpcfo.legend(ncol=2, columnspacing=1, loc='upper left')
            rpcfo.set_yticks(np.arange(-10, 10, 0.5))
            rpcfo.set_ylim([-1.5, 1.5])  # y軸の範囲

            pfcfo.set_ylabel('Pitch FCFO [Nm]')
            pfcfo.legend(ncol=2, columnspacing=1, loc='upper left')
            pfcfo.set_yticks(np.arange(-8.0, 8.0, 2.0))
            pfcfo.set_ylim([-6.0, 6.0])  # y軸の範囲

            rfcfo.set_ylabel('Roll FCFO [Nm]')
            rfcfo.legend(ncol=2, columnspacing=1, loc='upper left')
            rfcfo.set_yticks(np.arange(-8.0, 8.0, 2.0))
            rfcfo.set_ylim([-6.0, 6.0])  # y軸の範囲r

            plt.tight_layout()
            # plt.savefig("response.png")
        plt.show()

    def ocfo(self):

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

    def summation_cfo_graph(self, savename):
        CFO.summation(self)
        fig, (ppcfo, rpcfo, pfcfo, rfcfo) = plt.subplots(4, 1, figsize=(5, 7), dpi=150, sharex=True)

        # plt.xlim([10, 60])  # x軸の範囲
        # plt.xlim([0.28, 0.89])  # x軸の範囲
        plt.xlabel("Time[sec]")

        for i in range(len(self.cfo)):
            data = self.cfo[i]

            ppcfo.plot(data['time'][self.start_num:self.end_num:10], self.ppcof_summation[i][::10], label='Group'+str(i + 1))
            rpcfo.plot(data['time'][self.start_num:self.end_num:10], self.rpcof_summation[i][::10], label='Group'+str(i + 1))
            pfcfo.plot(data['time'][self.start_num:self.end_num:10], self.pfcof_summation[i][::10], label='Group' + str(i + 1))
            rfcfo.plot(data['time'][self.start_num:self.end_num:10], self.rfcof_summation[i][::10], label='Group' + str(i + 1))


        ppcfo.set_ylabel('Summation\nPitch PCFO [rad]')
        rpcfo.set_ylabel('Summation\nRoll PCFO [rad]')
        pfcfo.set_ylabel('Summation\nPitch FCFO [Nm]')
        rfcfo.set_ylabel('Summation\nRoll FCFO [Nm]')

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
        plt.savefig(savename)
        # plt.show()

    def summation_cfo_3sec(self):
        CFO.summation(self)
        ppcof_summation_3sec = self.ppcof_summation.reshape([len(self.cfo), -1, self.num])
        ppcof_summation_3sec = np.average(ppcof_summation_3sec, axis=2)

        rpcof_summation_3sec = self.rpcof_summation.reshape([len(self.cfo), -1, self.num])
        rpcof_summation_3sec = np.average(rpcof_summation_3sec, axis=2)

        pfcof_summation_3sec = self.pfcof_summation.reshape([len(self.cfo), -1, self.num])
        pfcof_summation_3sec = np.average(pfcof_summation_3sec, axis=2)

        rfcof_summation_3sec = self.rfcof_summation.reshape([len(self.cfo), -1, self.num])
        rfcof_summation_3sec = np.average(rfcof_summation_3sec, axis=2)

        return ppcof_summation_3sec, rpcof_summation_3sec, pfcof_summation_3sec, rfcof_summation_3sec

    def summation(self):
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
                # summation_ = np.abs(summation_)
                summation = np.vstack((summation, np.sum(summation_, axis=0)))

            # print(summation_)
        summation = np.delete(summation, 0, 0)
        # print(summation.shape)
        summation = summation.reshape([4, len(self.cfo), -1])
        summation = summation / self.cfo[0]['join'][0]
        self.ppcof_summation = summation[0]
        self.rpcof_summation = summation[1]
        self.pfcof_summation = summation[2]
        self.rfcof_summation = summation[3]


    def performance_calc(self, data, ballx, bally):
        error = np.sqrt(
            (data['targetx'][self.start_num:self.end_num] - ballx[self.start_num:self.end_num]) ** 2
            + (data['targety'][self.start_num:self.end_num] - bally[self.start_num:self.end_num]) ** 2)

        # plt.plot(self.data['time'][self.start_num:self.end_num], error)
        # plt.show()

        target_size = 0.03
        spent = np.where(error < target_size, 1, 0)
        # spent = numpy.where(error < self.data['targetsize'], 1, 0)

        # plt.plot(self.data['time'][self.start_num:self.end_num], spent)
        # plt.show()

        return error, spent

    def period_performance_human(self):
        error_period = []
        spent_period = []
        for i in range(len(self.cfo)):
            data = self.cfo[i]

            error, spent = CFO.performance_calc(self, data, data['ballx'], data['bally'])
            error_reshape = error.reshape([self.period, self.num]) #[回数][データ]にわける
            # error_period = np.sum(error_reshape, axis=1) # 回数ごとに足す
            error_period.append(np.sum(error_reshape, axis=1) / self.num)

            # plt.plot(np.arange(self.period) + 1, error_period[i])
            # plt.show()

            spent_reshape = spent.reshape([self.period, self.num])
            spent_period_ = np.sum(spent_reshape, axis=1)
            # spent_period = spent_period_ * self.smp
            spent_period.append(spent_period_ * self.smp)

            # plt.plot(np.arange(self.period) + 1, spent_period[i])
            # plt.xticks(np.arange(1, self.period + 1, 1))
            # plt.show()

        return error_period, spent_period

    def period_performance_model(self):
        error_period = []
        spent_period = []
        for i in range(len(self.cfo)):
            data = self.cfo[i]

            error, spent = CFO.performance_calc(self, data, data['ballx_pre'], data['bally_pre'])
            error_reshape = error.reshape([self.period, self.num]) #[回数][データ]にわける
            # error_period = np.sum(error_reshape, axis=1) # 回数ごとに足す
            error_period.append(np.sum(error_reshape, axis=1) / self.num)

            # plt.plot(np.arange(self.period) + 1, error_period[i])
            # plt.show()

            spent_reshape = spent.reshape([self.period, self.num])
            spent_period_ = np.sum(spent_reshape, axis=1)
            # spent_period = spent_period_ * self.smp
            spent_period.append(spent_period_ * self.smp)

            # plt.plot(np.arange(self.period) + 1, spent_period[i])
            # plt.show()

        return error_period, spent_period

    def period_performance_cooperation(self):
        error_period_human, spend_period_human = CFO.period_performance_human(self)
        error_period_model, spend_period_model = CFO.period_performance_model(self)
        error_period = np.subtract(error_period_human, error_period_model)
        spend_period = np.subtract(spend_period_human, spend_period_model)

        return error_period, spend_period


    def ocfo_performance_relation(self):
        error_period, spend_period = CFO.period_performance_cooperation(self)
        ecfo = CFO.period_ecfo(self)
        inecfo = CFO.period_inecfo(self)

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

        # ax1 = fig.add_subplot(2, 2, 1)
        # ax1.set_ylim(-0.3, 0.5)
        # ax1.set_xlim(-1.5, 1.5)
        # g = sns.scatterplot(data=df_all, x='ecfo', y='spend', hue='Group', s=marker_size)
        # for lh in g.legend_.legendHandles:
        #     lh.set_alpha(1)
        #     lh._sizes = [10]
        # fig.add_subplot(2, 2, 2)
        # sns.scatterplot(data=df_all, x='ecfo', y='spend', hue='Group', s=marker_size)
        #
        # fig.add_subplot(2, 2, 3)
        # sns.scatterplot(data=df_all, x='inecfo', y='spend', hue='Group', s=marker_size)
        #
        # fig.add_subplot(2, 2, 4)
        # sns.scatterplot(data=df_all, x='inecfo', y='spend', hue='Group', s=marker_size)

        plt.tight_layout()
        plt.show()

    def period_ecfo(self):
        ecfo = []
        for i in range(len(self.cfo)):
            data = self.cfo[i]
            ecfo.append(CFO.period_calculation(self, data['ecfo'][self.start_num:self.end_num]))

        return ecfo

    def period_inecfo(self):
        inecfo = []
        for i in range(len(self.cfo)):
            data = self.cfo[i]
            inecfo.append(CFO.period_calculation(self, np.abs(data['inecfo'][self.start_num:self.end_num])))

        return inecfo

    def period_calculation(self, data):
        data_reshape = data.reshape([self.period, self.num])  # [回数][データ]にわける
        data_period = np.sum(data_reshape, axis=1) / self.num

        return data_period

    def subtraction_cfo(self, graph=1):
        sub_cfo = []
        types = ['p_pcfo', 'r_pcfo', 'p_fcfo', 'r_fcfo']
        for type in types:
            sub_cfo_types = []
            for i in range(len(self.cfo)):
                data = self.cfo[i]
                if self.group_type == 'dyad':
                    sub_cfo_types.append(np.abs(np.subtract(data['i1_' + type][self.start_num:self.end_num], data['i2_' + type][self.start_num:self.end_num])))

                elif self.group_type == 'triad':
                    sub_cfo1 = np.subtract(np.subtract(2 * data['i1_' + type][self.start_num:self.end_num], data['i2_' + type][self.start_num:self.end_num]), data['i3_' + type][self.start_num:self.end_num])
                    sub_cfo2 = np.subtract(np.subtract(2 * data['i2_' + type][self.start_num:self.end_num], data['i1_' + type][self.start_num:self.end_num]), data['i3_' + type][self.start_num:self.end_num])
                    sub_cfo3 = np.subtract(np.subtract(2 * data['i3_' + type][self.start_num:self.end_num], data['i1_' + type][self.start_num:self.end_num]), data['i2_' + type][self.start_num:self.end_num])
                    sub_cfo_ave = np.add(np.add(np.abs(sub_cfo1), np.abs(sub_cfo2)), np.abs(sub_cfo3)) / 3
                    sub_cfo_types.append(sub_cfo_ave)

                elif self.group_type == 'tetrad':
                    sub_cfo1 = np.subtract(np.subtract(np.subtract(3 * data['i1_' + type][self.start_num:self.end_num], data['i2_' + type][self.start_num:self.end_num]), data['i3_' + type][self.start_num:self.end_num]), data['i4_' + type][self.start_num:self.end_num])
                    sub_cfo2 = np.subtract(np.subtract(np.subtract(3 * data['i2_' + type][self.start_num:self.end_num], data['i1_' + type][self.start_num:self.end_num]), data['i3_' + type][self.start_num:self.end_num]), data['i4_' + type][self.start_num:self.end_num])
                    sub_cfo3 = np.subtract(np.subtract(np.subtract(3 * data['i3_' + type][self.start_num:self.end_num], data['i1_' + type][self.start_num:self.end_num]), data['i2_' + type][self.start_num:self.end_num]), data['i4_' + type][self.start_num:self.end_num])
                    sub_cfo4 = np.subtract(np.subtract(np.subtract(3 * data['i4_' + type][self.start_num:self.end_num], data['i1_' + type][self.start_num:self.end_num]), data['i2_' + type][self.start_num:self.end_num]), data['i3_' + type][self.start_num:self.end_num])
                    sub_cfo_ave = np.add(np.add(np.add(np.abs(sub_cfo1), np.abs(sub_cfo2)), np.abs(sub_cfo3)), np.abs(sub_cfo4)) / 4
                    sub_cfo_types.append(sub_cfo_ave)
            sub_cfo.append(sub_cfo_types)

        if graph == 0:
            fig, (ppcfo, rpcfo, pfcfo, rfcfo) = plt.subplots(4, 1, figsize=(5, 7), dpi=150, sharex=True)

            # plt.xlim([10, 60])  # x軸の範囲
            # plt.xlim([0.28, 0.89])  # x軸の範囲
            plt.xlabel("Time[sec]")

            for i in range(len(self.cfo)):
                data = self.cfo[i]

                ppcfo.plot(data['time'][self.start_num:self.end_num:10], sub_cfo[0][i][::10], label='Group' + str(i + 1))
                rpcfo.plot(data['time'][self.start_num:self.end_num:10], sub_cfo[1][i][::10], label='Group' + str(i + 1))
                pfcfo.plot(data['time'][self.start_num:self.end_num:10], sub_cfo[2][i][::10], label='Group' + str(i + 1))
                rfcfo.plot(data['time'][self.start_num:self.end_num:10], sub_cfo[3][i][::10], label='Group' + str(i + 1))

            ppcfo.set_ylabel('Subtraction\nPitch PCFO [rad]')
            rpcfo.set_ylabel('Subtraction\nRoll PCFO [rad]')
            pfcfo.set_ylabel('Subtraction\nPitch FCFO [Nm]')
            rfcfo.set_ylabel('Subtraction\nRoll FCFO [Nm]')

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


        return sub_cfo[0], sub_cfo[1], sub_cfo[2], sub_cfo[3]