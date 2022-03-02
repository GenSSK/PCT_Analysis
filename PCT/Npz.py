import numpy as np
import platform
import glob

class NPZ:
    def __init__(self):
        os = platform.system()
        if os == "Windows":
            self.dir = "G:\\マイドライブ\\program\\ARCS-PCT\\data\\PCT\\Robomech\\"  # win
        elif os == "Darwin":
            self.dir = "/Volumes/GoogleDrive/My Drive/program/ARCS-PCT/data/PCT/Robomech/npz/"  # mac

    def single_load(self, filename):

        # self.data = np.load(self.dir + '2021-10-27_BilateralTest.npz')
        # self.data = np.load(self.dir + '2022-01-20_sixth_test.npz')
        # data = np.load(self.dir + '2022-01-20_FourPeople_test.npz')
        # data = np.load(self.dir + '2022-01-22_Eleventh_test.npz')  # pitchのみ　非対面
        # self.data = np.load(self.dir + '2022-01-22_Twelfth_test.npz')  #pitchのみ　対面

        data = np.load(self.dir + filename, mmap_mode='r')

        return data

    def all_load(self):
        filename = glob.glob(self.dir + '*.npz')  # パスのnpzを全部読み込んじゃう

        if len(filename) == 0:
            print("Cannot detect the file")

        numpy_vars = {}
        for i in range(len(filename)):
            numpy_vars[i] = np.load(filename[i], mmap_mode='r')

        return numpy_vars

    def type_load(self, type):
        filename = glob.glob(self.dir + '*_' + type + '.npz')  # パスのnpzを全部読み込んじゃう

        if len(filename) == 0:
            print("Cannot detect the file")

        numpy_vars = {}
        for i in range(len(filename)):
            numpy_vars[i] = np.load(filename[i], mmap_mode='r')

        return numpy_vars