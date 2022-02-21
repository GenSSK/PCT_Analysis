import numpy
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt


class ForceAnalysis:
    def __init__(self, normal_data, alone_data, nothing_data):
        self.data = numpy.arange(3)
        self.data[0] = normal_data
        self.data[1] = alone_data
        self.data[2] = nothing_data

    def rms(self, data1, data2):
        val = np.sqrt(data1 ** 2 + data2 ** 2)
        return val

    def separator(self, dat, time_smp, separation_time):
        separeted = dat.reshape((separation_time / time_smp, len(dat) / separation_time))
        return separeted

    def calc_data(self, p_position, r_position, p_force, r_force):
        p = self.rms(p_position, r_position)
        f = self.rms(p_force, r_force)
        return p, f

    def compare_individual(self, subject_num):

        smp = 0.0001
        tasktime = 3.0

        if (subject_num == 0):
            num = "i1"
        elif (subject_num == 1):
            num = "i2"
        elif (subject_num == 2):
            num = "i3"
        else:
            num = "i4"

        pos_rms = numpy.arange(3)
        force_rms = numpy.arange(3)

        pos_sep = numpy.arange(3)
        force_sep = numpy.arange(3)

        for i in range(len(self.data)):
            pos_rms[i], force_rms[i] = self.calc_data(self.data[i][num + '_p_thm'],
                                                      self.data[i][num + '_r_thm'],
                                                      self.data[i][num + '_p_text'],
                                                      self.data[i][num + '_r_text'])
            pos_sep = self.separator(pos_rms[i], smp, tasktime)
            force_sep = self.separator(pos_rms[i], smp, tasktime)

        for j in range(len(pos_sep[0])):
            plt.plot(np.arrange(tasktime / smp), pos_sep[0][j])

        plt.show()
