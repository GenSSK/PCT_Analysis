import numpy
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt


class Calculation:
    def __init__(self, data):
        self.data = data

        self.smp = 0.0001  # サンプリング時間
        self.time = self.data['duringtime']  # ターゲットの移動時間
        self.period = int(self.data['tasktime'] / self.time)  # 回数
        self.num = int(self.time / self.smp)  # 1ピリオドにおけるデータ数
        self.start_num = int(self.data['starttime'] / self.smp)
        self.end_num = int(self.data['endtime'] / self.smp)
        self.nn_read_flag = False

    def performance_calc(self):
        error = np.sqrt(
            (self.data['targetx'][self.start_num:self.end_num] - self.data['ballx'][self.start_num:self.end_num]) ** 2
            + (self.data['targety'][self.start_num:self.end_num] - self.data['bally'][self.start_num:self.end_num]) ** 2)

        # plt.plot(self.data['time'][self.start_num:self.end_num], error)
        # plt.show()

        spent = numpy.where(error < self.data['targetsize'], 1, 0)
        # spent = numpy.where(error < self.data['targetsize'], 1, 0)

        # plt.plot(self.data['time'][self.start_num:self.end_num], spent)
        # plt.show()

        return error, spent

    def period_performance(self):
        error, spent = Calculation.performance_calc(self)
        error_reshape = error.reshape([self.period, self.num]) #[回数][データ]にわける
        error_period = np.sum(error_reshape, axis=1) # 回数ごとに足す

        plt.plot(np.arange(self.period) + 1, error_period)
        plt.show()

        spent_reshape = spent.reshape([self.period, self.num])
        spent_period = np.sum(spent_reshape, axis=1)
        spent_period = spent_period * self.smp

        plt.plot(np.arange(self.period) + 1, spent_period)
        plt.show()

        return error_period, spent_period

    def nn_data_set(self, nn_data):
        corr = np.correlate(self.data['targetx'], nn_data['targetx'])
        if corr < 0.9:
            print("Target does not much")
            exit(-1)
        if self.data['join'] != nn_data['join']:
            print("joining people does not much")
            exit(-1)

        # for i in range(self.data['join']):
        #     name1 = 'j' + str(i) + '_'
        #     if self.data


        self.nn = nn_data



        self.nn_read_flag = True

    def cof_calculation(self):
        if not self.nn_read_flag:
            print("Didn't read nn file")
            exit(-1)