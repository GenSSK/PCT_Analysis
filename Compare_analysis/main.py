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

    PP_npz_filename = [
        '2023-01-24_b.poitrimol_k.tozuka_PP.npz',
        '2023-01-24_h.nishimura_y.inoue_PP.npz',
        '2023-01-24_i.kato_ko.kobayashi_PP.npz',
        '2023-01-24_m.nakamura_r.tanaka_PP.npz',
        '2023-01-24_t.onogawa_n.ito_PP.npz',
        '2023-01-24_t.oriuchi_s.watanabe_PP.npz',
        '2023-01-24_y.yamada_s.tsuchiya_PP.npz',
        '2023-01-24_y.yoshida_k.kobayashi_PP.npz',
    ]

    AdPD_npz_filename = [
        '2023-01-24_b.poitrimol_k.tozuka_Admittance(PD).npz',
        '2023-01-24_h.nishimura_y.inoue_Admittance(PD).npz',
        '2023-01-24_i.kato_ko.kobayashi_Admittance(PD).npz',
        '2023-01-24_m.nakamura_r.tanaka_Admittance(PD).npz',
        '2023-01-24_t.onogawa_n.ito_Admittance(PD).npz',
        '2023-01-24_t.oriuchi_s.watanabe_Admittance(PD).npz',
        '2023-01-24_y.yamada_s.tsuchiya_Admittance(PD).npz',
        '2023-01-24_y.yoshida_k.kobayashi_Admittance(PD).npz',
    ]

    AdAc_npz_filename = [
        '2023-01-24_b.poitrimol_k.tozuka_Admittance(Accel).npz',
        '2023-01-24_h.nishimura_y.inoue_Admittance(Accel).npz',
        '2023-01-24_i.kato_ko.kobayashi_Admittance(Accel).npz',
        '2023-01-24_m.nakamura_r.tanaka_Admittance(Accel).npz',
        '2023-01-24_t.onogawa_n.ito_Admittance(Accel).npz',
        '2023-01-24_t.oriuchi_s.watanabe_Admittance(Accel).npz',
        '2023-01-24_y.yamada_s.tsuchiya_Admittance(Accel).npz',
        '2023-01-24_y.yoshida_k.kobayashi_Admittance(Accel).npz',
    ]

    Bi_npz_filename = [
        '2023-01-24_b.poitrimol_k.tozuka_Bilateral.npz',
        '2023-01-24_h.nishimura_y.inoue_Bilateral.npz',
        '2023-01-24_i.kato_ko.kobayashi_Bilateral.npz',
        '2023-01-24_m.nakamura_r.tanaka_Bilateral.npz',
        '2023-01-24_t.onogawa_n.ito_Bilateral.npz',
        '2023-01-24_t.oriuchi_s.watanabe_Bilateral.npz',
        '2023-01-24_y.yamada_s.tsuchiya_Bilateral.npz',
        '2023-01-24_y.yoshida_k.kobayashi_Bilateral.npz',
    ]

    PP_npz = npz.select_load(PP_npz_filename)
    AdPD_npz = npz.select_load(AdPD_npz_filename)
    AdAc_npz = npz.select_load(AdAc_npz_filename)
    Bi_npz = npz.select_load(Bi_npz_filename)

    com = Combine_analysis.combine(PP_npz, AdPD_npz, AdAc_npz, Bi_npz)

    ##グラフ確認
    # com.PP.graph_sub()
    # com.AdPD.graph_sub()
    # com.AdAc.graph_sub()
    # com.Bi.graph_sub()


    ##タスク確認
    # com.PP.task_show()
    # com.AdPD.task_show()
    # com.AdAc.task_show()
    # com.Bi.task_show()

    ##パフォーマンスの確認
    # com.performance_show() #パフォーマンスの時系列重ね
    # com.PP.period_performance(0)
    # com.AdPD.period_performance(0)
    # com.AdAc.period_performance(0)
    # com.Bi.period_performance(0)

    # com.performance_comparison() #パフォーマンス比較

    ##パフォーマンス同士の相関
    # com.performance_relation()

    ##パフォーマンスの比較
    # com.performance_hist()
    # com.performance_bootstrap()
    # performance = com.performance_each()
    # performance.to_csv('csv/performance.csv')

    ##位置の差分
    com.PP.subtraction_position(0)
    com.AdPD.subtraction_position(0)
    com.AdAc.subtraction_position(0)
    com.Bi.subtraction_position(0)


    # com.performance_deviation()

    # com.dyad_cfo.fcfo_valiance()
    # com.tetrad_cfo.fcfo_valiance(0)

    # com.variance_analysis('noabs')
    # com.variance_analysis('b_abs')
    # com.variance_analysis('a_abs')

    # com.dyad_cfo.tf_graph_sub()
    # com.triad_cfo.tf_graph_sub()
    # com.tetrad_cfo.tf_graph_sub()

    # com.dyad_cfo.tf_cfo_sub()
    # com.triad_cfo.tf_cfo_sub()
    # com.tetrad_cfo.tf_cfo_sub()

    # com.dyad_cfo.work_calc()
    # com.dyad_cfo.work_diff(0)
    # com.triad_cfo.work_diff(0)
    # com.tetrad_cfo.work_diff(0)