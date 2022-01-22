import numpy as np


class LoadNPZ:
    def load(self):
        self.dir = "G:\\マイドライブ\\program\\ARCS-PCT\\data\\PCT\\"  # win
        # self.dir = "/Volumes/GoogleDrive/My Drive/program/ARCS-PCT/data/Sinwave_responce/"  # mac
        # self.data = np.load(self.dir + '2021-10-27_BilateralTest.npz')
        # self.data = np.load(self.dir + '2022-01-20_sixth_test.npz')
        # self.data = np.load(self.dir + '2022-01-20_FourPeople_test.npz')
        self.data = np.load(self.dir + '2022-01-22_Eleventh_test.npz')  # pitchのみ　非対面
        # self.data = np.load(self.dir + '2022-01-22_Twelfth_test.npz')  #pitchのみ　対面
