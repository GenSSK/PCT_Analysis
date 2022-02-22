import numpy
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt


class ForceAnalysis:
    def __init__(self, normal_data, alone_data, nothing_data):
        # self.data = numpy.arange(3)
        self.data = {}
        # print(type(normal_data))
        self.data[0] = normal_data
        self.data[1] = alone_data
        self.data[2] = nothing_data

        self.smp = 0.0001  # サンプリング時間
        self.time = self.data[0][0]['duringtime'][0]  # ターゲットの移動時間
        self.period = int(self.data[0][0]['tasktime'] / self.time)  # 回数
        self.num = int(self.time / self.smp)  # 1ピリオドにおけるデータ数
        self.start_num = int(self.data[0][0]['starttime'] / self.smp)
        self.end_num = int(self.data[0][0]['endtime'] / self.smp)

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

    def compare_individual(self, subject_num):
        if (subject_num == 0):
            num = "i1"
        elif (subject_num == 1):
            num = "i2"
        elif (subject_num == 2):
            num = "i3"
        else:
            num = "i4"

        pos_rms = {}
        force_rms = {}

        pos_sep = {}
        force_sep = {}


        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                pos_rms[i], force_rms[i] = self.calc_data(self.data[i][j][num + '_p_thm'][self.start_num:self.end_num],
                                                          self.data[i][j][num + '_r_thm'][self.start_num:self.end_num],
                                                          self.data[i][j][num + '_p_text'][self.start_num:self.end_num],
                                                          self.data[i][j][num + '_r_text'][self.start_num:self.end_num])
            pos_sep[i] = self.separator(pos_rms[i])
            force_sep[i] = self.separator(force_rms[i])

        for j in range(len(pos_sep[0])):
            plt.plot(np.arange(0.0, self.time, self.smp), pos_sep[2][j])


        plt.savefig('sample_pos.png')
        # plt.show()
