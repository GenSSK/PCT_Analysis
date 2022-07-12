# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import pandas as pd
import Npz
import PCT
import CFO_test
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


plt.switch_backend('Qt5Agg')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    npz = Npz.NPZ()
    # pct = PCT.PCT()
    # cfoo = npz.single_load('/cfo/2022-05-02_ko.kobayashi_s.watanabe_1_CFO.npz')
    # data = npz.single_load('/npz/cooperation/2022-05-02_ko.kobayashi_s.watanabe_1.npz')

    cfoo = npz.single_load('/cfo/2022-07-01_y.inoue_k.tozuka_b.poitorimol_y.baba_1234_CFO.npz')
    data = npz.single_load('/npz/cooperation/2022-07-01_y.inoue_k.tozuka_b.poitorimol_y.baba_1234.npz')

    plt.plot(cfoo['ballx_pre'])
    plt.plot(cfoo['ballx'])
    plt.show()

    cfo = CFO_test.CFO(data, cfoo)
    cfo.graph_sub()

    plt.show()

    cfo.cfo_sub()

    # cfo.task_show(data)

    plt.show()

