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
import seaborn as sns
import Combine_analysis


plt.switch_backend('Qt5Agg')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    npz = Npz.NPZ()

    dyad_npz_filename = [
        'dyad/2022-07-28_y.inoue_b.poitrimol_1234_CFO.npz',
        'dyad/2022-07-29_i.tsunokuni_k.tozuka_2134_CFO.npz',
        'dyad/2022-07-29_i.tsunokuni_y.inoue_1234_CFO.npz',
        'dyad/2022-07-31_m.sugaya_n.ito_1234_CFO.npz',
        'dyad/2022-08-23_r.yanase_ko.kobayashi_1324_CFO.npz',
        'dyad/2022-08-26_y.kobayashi_r.yanase_1234_CFO.npz',
        'dyad/2022-09-05_s.watanabe_ko.kobayashi_1234_CFO.npz',
        'dyad/2022-09-07_k.kobayashi_r.yanase_1234_CFO.npz',
        'dyad/2022-09-07_k.tozuka_ko.kobayashi_1234_CFO.npz',
        'dyad/2022-09-07_y.inoue_k.kobayashi_1234_CFO.npz',
    ]

    triad_npz_filename = [
        'triad/2022-07-31_h.igarashi_ko.kobayashi_t.kassai_1234_CFO.npz',
        'triad/2022-07-31_k.tozuka_y.inoue_m.nakamura_1234_CFO.npz',
        'triad/2022-07-31_m.sugaya_s.sanuka_m.nakamura_1234_CFO.npz',
        'triad/2022-08-23_s.watanabe_t.kassai_h.nishimura_1234_CFO.npz',
        'triad/2022-08-26_k.kobayashi_r.yanase_ko.kobayashi_1234_CFO.npz',
        'triad/2022-09-05_b.poitrimol_y.inoue_k.tozuka_1234_CFO.npz',
        'triad/2022-09-07_b.poitrimol_k.tozuka_y.kobayashi_1234_CFO.npz',
        'triad/2022-09-07_b.poitrimol_y.inoue_r.yanase_1234_CFO.npz',
        'triad/2022-09-07_ko.kobayashi_k.kobayashi_y.kobayashi_1234_CFO.npz',
        'triad/2022-09-07_r.yanase_k.kobayashi_s.kamioka_1234_CFO.npz',
    ]

    tetrad_npz_filename = [
        'tetrad/2022-07-01_y.inoue_k.tozuka_b.poitrimol_y.baba_1234_trans_CFO.npz',
        'tetrad/2022-07-30_k.ohya_r.tanaka_y.baba_m.nakamura_1234_CFO.npz',
        'tetrad/2022-07-30_r.tanaka_h.nishimura_k.tozuka_b.poitrimol_1234_CFO.npz',
        'tetrad/2022-07-30_s.watanabe_h.nishimura_y.baba_y.yoshida_1234_CFO.npz',
        'tetrad/2022-07-30_s.watanabe_ko.kobayashi_y.baba_k.tozuka_1234_CFO.npz',
        'tetrad/2022-07-30_y.inoue_y.yoshida_n.ito_s.sanuka_1234_CFO.npz',
        'tetrad/2022-08-26_k.kobayashi_r.yanase_ko.kobayashi_h.nishimura_1234_CFO.npz',
        'tetrad/2022-09-07_b.poitrimol_y.kobayashi_s.kamioka_y.inoue_1234_CFO.npz',
        'tetrad/2022-09-07_k.kobayashi_y.kobayashi_s.kamioka_r.yanase_1234_CFO.npz',
        'tetrad/2022-09-07_r.yanase_ko.kobayashi_b.poitrimol_y.kobayashi_1234_CFO.npz',
    ]

    dyad_npz = npz.select_load(dyad_npz_filename)
    triad_npz = npz.select_load(triad_npz_filename)
    tetrad_npz = npz.select_load(tetrad_npz_filename)

    com = Combine_analysis.combine(dyad_npz, triad_npz, tetrad_npz)

    ##予測確認r
    # com.dyad_cfo.graph_sub()
    # com.triad_cfo.graph_sub()
    # com.tetrad_cfo.graph_sub()

    ##タスク確認
    # com.dyad_cfo.task_show()
    # com.triad_cfo.task_show()
    # com.tetrad_cfo.task_show()

    ##CFO確認
    com.dyad_cfo.cfo_sub()
    # com.triad_cfo.cfo_sub()
    # com.tetrad_cfo.cfo_sub(.npz)


    ##和のCFOを確認
    # com.dyad_cfo.summation_cfo_graph('fig/dyad_summation_cfo.png')
    # com.triad_cfo.summation_cfo_graph('fig/triad_summation_cfo.png')
    # com.tetrad_cfo.summation_cfo_graph('fig/tetrad_summation_cfo.png')
    # plt.show()


    ##和のCFOを比較
    # com.summation_CFO_analysis()

    ##パフォーマンスの確認
    # com.performance_show()
    # com.dyad_cfo.period_performance_human()
    # com.performance_comparison()

    ##ECFOとパフォーマンスの確認
    # com.dyad_cfo.ocfo_performance_relation()
    # com.triad_cfo.ocfo_performance_relation()
    com.tetrad_cfo.ocfo_performance_relation()