import pandas as pd
import glob
import platform

class readCSV:
    def __init__(self):
        os = platform.system()
        print(os)
        if os == "Windows":
            self.dir = "D:\\Program\\ez_libtorch\\data\\"  # win
            # self.dir = "G:\\マイドライブ\\program\\ARCS-PCT\\data\\solo_model\\"  # win
        elif os == "Darwin":
            self.dir = "/Users/genki/Google Drive/My Drive/program/ARCS-PCT/data/linear2dof/Resonance/"  # mac
        else:
            # self.dir = "/home/genki/tmp/ez_libtorch/data/"  # linux
            # self.dir = "/home/genki/tmp/tmp.DCaNlQozen/data/"  # linux
            self.dir = "/home/genki/data/cfo/"  # linux
    def csv_read(self, filename):
        # CSV読み込むよ！
        data = pd.read_csv(self.dir + filename, header=None)
        return data