# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import pandas as pd
import Npz
import PCT
import CFO_analysis
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


plt.switch_backend('Qt5Agg')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    npz = Npz.NPZ()
    # pct = PCT.PCT()

    dyad_npz_filename = [
        'dyad/2022-07-28_y.inoue_b.poitrimol_1234_CFO.npz',
        'dyad/2022-07-29_i.tsunokuni_k.tozuka_2134_CFO.npz',
        'dyad/2022-07-29_i.tsunokuni_y.inoue_1234_CFO.npz',
        'dyad/2022-07-31_m.sugaya_n.ito_1234_CFO.npz',
        'dyad/2022-08-26_y.kobayashi_r.yanase_1234_CFO.npz',
    ]

    triad_npz_filename = [
        'triad/2022-07-31_h.igarashi_ko.kobayashi_t.kassai_1234_CFO.npz',
        'triad/2022-07-31_k.tozuka_y.inoue_m.nakamura_1234_CFO.npz',
        'triad/2022-07-31_m.sugaya_s.sanuka_m.nakamura_1234_CFO.npz',
        'triad/2022-08-23_s.watanabe_t.kassai_h.nishimura_1234_CFO.npz',
        'triad/2022-08-26_k.kobayashi_r.yanase_ko.kobayashi_1234_CFO.npz',
    ]

    tetrad_npz_filename = [
        'tetrad/2022-07-01_y.inoue_k.tozuka_b.poitrimol_y.baba_1234_trans_CFO.npz',
        'tetrad/2022-07-30_k.ohya_r.tanaka_y.baba_m.nakamura_1234_CFO.npz',
        'tetrad/2022-07-30_r.tanaka_h.nishimura_k.tozuka_b.poitrimol_1234_CFO.npz',
        'tetrad/2022-07-30_s.watanabe_h.nishimura_y.baba_y.yoshida_1234_CFO.npz',
        'tetrad/2022-07-30_s.watanabe_ko.kobayashi_y.baba_k.tozuka_1234_CFO.npz',
        'tetrad/2022-07-30_y.inoue_y.yoshida_n.ito_s.sanuka_1234_CFO.npz',
        'tetrad/2022-08-26_k.kobayashi_r.yanase_ko.kobayashi_h.nishimura_1234_CFO.npz',
    ]

    dyad_npz = npz.select_load(dyad_npz_filename)
    triad_npz = npz.select_load(triad_npz_filename)
    tetrad_npz = npz.select_load(tetrad_npz_filename)

    dyad_cfo = CFO_analysis.CFO(dyad_npz)
    triad_cfo = CFO_analysis.CFO(triad_npz)
    tetrad_cfo = CFO_analysis.CFO(tetrad_npz)

    ##予測確認
    # dyad_cfo.graph_sub()
    # triad_cfo.graph_sub()
    # tetrad_cfo.graph_sub()

    ##タスク確認
    # dyad_cfo.task_show()
    # triad_cfo.task_show()
    # tetrad_cfo.task_show()

    ##CFO確認
    # dyad_cfo.cfo_sub()
    # triad_cfo.cfo_sub()
    # tetrad_cfo.cfo_sub()

    dyad_cfo.summation()

    # plt.plot(data['targetx'])
    # plt.plot(cfoo['targetx'])
    # # plt.show()
    #
    # plt.plot(data['targety'])
    # plt.plot(cfoo['targety'])
    # plt.show()

    # cfo = CFO_test.CFO(data, cfoo)
    # cfo.graph_sub()
    #
    # plt.show()
    #
    # cfo.cfo_sub()

    # cfo.task_show(data)

    # plt.show()

    # cfo.ocfo()
    # plt.show()


