import numpy as np
import platform
import glob

class NPZ:
    def __init__(self):
        os = platform.system()
        print(os)
        if os == "Windows":
            self.dir = "D:\\Program\\ez_libtorch\\data\\"  # win
            # self.dir = "G:\\マイドライブ\\program\\ARCS-PCT\\data\\solo_model\\"  # win
        elif os == "Darwin":
            self.dir = "/Volumes/GoogleDrive/My Drive/program/ARCS-PCT/data/solo_model/"  # mac
        else:
            # self.dir = "/home/genki/tmp/ez_libtorch/data/"  # linux
            # self.dir = "/home/genki/tmp/tmp.DCaNlQozen/data/"  # linux
            self.dir = "/home/genki/data/cfo/"  # linux


    def single_load(self, filename):

        # self.data = np.load(self.dir + '2021-10-27_BilateralTest.npz')
        # self.data = np.load(self.dir + '2022-01-20_sixth_test.npz')
        # data = np.load(self.dir + '2022-01-20_FourPeople_test.npz')
        # data = np.load(self.dir + '2022-01-22_Eleventh_test.npz')  # pitchのみ　非対面
        # self.data = np.load(self.dir + '2022-01-22_Twelfth_test.npz')  #pitchのみ　対面

        data = np.load(self.dir + filename, mmap_mode='r')
        print("read = " + self.dir + filename)
        return data

    def all_load(self):
        filename = glob.glob(self.dir + '*.npz')  # パスのnpzを全部読み込んじゃう

        if len(filename) == 0:
            print("Cannot detect the file")

        numpy_vars = {}
        for i in range(len(filename)):
            numpy_vars[i] = np.load(filename[i], mmap_mode='r')

        return numpy_vars