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
        # 'dyad/2022-07-28_y.inoue_b.poitrimol_1234_CFO.npz',
        # 'dyad/2022-07-29_i.tsunokuni_k.tozuka_2134_CFO.npz',
        # 'dyad/2022-07-29_i.tsunokuni_y.inoue_1234_CFO.npz',
        # 'dyad/2022-07-31_m.sugaya_n.ito_1234_CFO.npz',
        # 'dyad/2022-08-23_r.yanase_ko.kobayashi_1324_CFO.npz',
        # 'dyad/2022-08-26_y.kobayashi_r.yanase_1234_CFO.npz',
        # 'dyad/2022-09-05_s.watanabe_ko.kobayashi_1234_CFO.npz',
        # 'dyad/2022-09-07_k.kobayashi_r.yanase_1234_CFO.npz',
        # 'dyad/2022-09-07_k.tozuka_ko.kobayashi_1324_CFO.npz',
        # 'dyad/2022-09-07_y.inoue_k.kobayashi_1234_CFO.npz',

        'dyad/2022-07-29_i.tsunokuni_k.tozuka_2134_CFO.npz',
        'dyad/2022-07-29_i.tsunokuni_y.inoue_1324_CFO.npz',
        'dyad/2022-07-31_m.sugaya_n.ito_1324_CFO.npz',
        'dyad/2022-08-23_r.yanase_ko.kobayashi_1324_CFO.npz',
        'dyad/2022-08-26_y.kobayashi_r.yanase_1324_CFO.npz',
        'dyad/2022-09-05_s.watanabe_ko.kobayashi_1324_CFO.npz',
        'dyad/2022-09-07_k.kobayashi_r.yanase_1324_CFO.npz',
        'dyad/2022-09-07_k.tozuka_ko.kobayashi_1324_CFO.npz',
        'dyad/2022-09-07_y.inoue_k.kobayashi_1324_CFO.npz',
    ]

    triad_npz_filename = [
        # 'triad/2022-07-31_h.igarashi_ko.kobayashi_t.kassai_1234_CFO.npz', #使えない
        'triad/2022-07-31_k.tozuka_y.inoue_m.nakamura_1234_CFO.npz',
        'triad/2022-07-31_m.sugaya_s.sanuka_m.nakamura_1234_CFO.npz',
        'triad/2022-08-23_s.watanabe_t.kassai_h.nishimura_2134_CFO.npz',
        'triad/2022-08-26_k.kobayashi_r.yanase_ko.kobayashi_1234_CFO.npz',
        'triad/2022-09-05_b.poitrimol_y.inoue_k.tozuka_1234_CFO.npz',
        'triad/2022-09-07_b.poitrimol_k.tozuka_y.kobayashi_1234_CFO.npz',
        'triad/2022-09-07_b.poitrimol_y.inoue_r.yanase_1234_CFO.npz',
        'triad/2022-09-07_ko.kobayashi_k.kobayashi_y.kobayashi_1234_CFO.npz',
        'triad/2022-09-07_r.yanase_k.kobayashi_s.kamioka_1234_CFO.npz',
    ]

    tetrad_npz_filename = [
        ## 'tetrad/2022-07-01_y.inoue_k.tozuka_b.poitrimol_y.baba_1234_trans_CFO.npz', #使えない
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

    ##予測確認
    # com.dyad_cfo.graph_sub()
    # com.triad_cfo.graph_sub()
    # com.tetrad_cfo.graph_sub()

    ##タスク確認
    # com.dyad_cfo.task_show()
    # com.triad_cfo.task_show()
    # com.tetrad_cfo.task_show()

    ##タスク確認（単独あり）
    # com.dyad_cfo.task_show_solo()
    # com.triad_cfo.task_show_solo()
    # com.tetrad_cfo.task_show_solo()

    ##CFO確認
    # com.dyad_cfo.cfo_sub()
    # com.triad_cfo.cfo_sub()
    # com.tetrad_cfo.cfo_sub()

    ##和のCFOを確認
    # com.dyad_cfo.summation_cfo(0) #和のCFOの時系列重ね
    # com.triad_cfo.summation_cfo(0) #和のCFOの時系列重ね
    # com.tetrad_cfo.summation_cfo(0) #和のCFOの時系列重ね
    # com.summation_cfo(0) #和のCFOの人数間比較
    # com.summation_cfo(0, 'b_abs') #和のCFOの人数間比較、前絶対値
    # com.summation_cfo(0, 'a_abs') #和のCFOの人数間比較、後絶対値

    ##パフォーマンスの確認
    # com.performance_show() #パフォーマンスの時系列重ね
    # com.dyad_cfo.period_performance_human(0) #H-Hのパフォーマンス、グループごと
    # com.triad_cfo.period_performance_human(0) #H-Hのパフォーマンス、グループごと
    # com.tetrad_cfo.period_performance_human(0) #H-Hのパフォーマンス、グループごと
    # com.dyad_cfo.period_performance_model(0) #M-Mのパフォーマンス、グループごと
    # com.triad_cfo.period_performance_model(0) #M-Mのパフォーマンス、グループごと
    # com.tetrad_cfo.period_performance_model(0) #M-Mのパフォーマンス、グループごと
    # com.dyad_cfo.period_performance_human_model() #H-HとM-Mのパフォーマンス比較、グループごと
    # com.triad_cfo.period_performance_human_model()
    # com.tetrad_cfo.period_performance_human_model()
    # com.performance_comparison() #人数間のパフォーマンス比較
    # com.performance_comparison('h-h')
    # com.performance_comparison('m-m')


    ##OCFOとパフォーマンスの確認
    # com.dyad_cfo.each_ocfo_performance('error')
    # com.triad_cfo.each_ocfo_performance('error')
    # com.tetrad_cfo.each_ocfo_performance('error')
    # com.dyad_cfo.ocfo_performance()
    # com.triad_cfo.ocfo_performance()
    # com.tetrad_cfo.ocfo_performance()

    ##差のCFOの確認
    # com.dyad_cfo.subtraction_cfo(0)
    # com.triad_cfo.subtraction_cfo(0)
    # com.tetrad_cfo.subtraction_cfo(0)
    # com.subtraction_cfo(0)

    ##差のCFOとパフォーマンスの確認
    # com.dyad_cfo.subtraction_performance()
    # com.triad_cfo.subtraction_performance()
    # com.tetrad_cfo.subtraction_performance()

    ##和のCFOとパフォーマンスの確認
    # com.dyad_cfo.summation_performance()
    # com.triad_cfo.summation_performance()
    # com.tetrad_cfo.summation_performance()
    # com.dyad_cfo.summation_performance('b_abs')
    # com.triad_cfo.summation_performance('b_abs')
    # com.tetrad_cfo.summation_performance('b_abs')
    # com.dyad_cfo.summation_performance('a_abs')
    # com.triad_cfo.summation_performance('a_abs')
    # com.tetrad_cfo.summation_performance('a_abs')

    ##和のCFOと差のCFOとパフォーマンスの関係
    # com.dyad_cfo.sum_sub_performance()

    ##パフォーマンス同士の相関
    # com.performance_relation()

    ##パフォーマンスの相関
    # com.performance_hist()
    # com.performance_bootstrap()

    ##CFOの相関
    # com.dyad_cfo.CFO_relation_axis()
    # com.triad_cfo.CFO_relation_axis()
    # com.tetrad_cfo.CFO_relation_axis()
    # com.dyad_cfo.CFO_relation_axis_3sec()
    # com.triad_cfo.CFO_relation_axis_3sec()
    # com.tetrad_cfo.CFO_relation_axis_3sec()

    ##sum_cfoとパフォーマンスの相関
    # com.dyad_cfo.summation_ave_performance()
    # com.triad_cfo.summation_ave_performance()
    # com.tetrad_cfo.summation_ave_performance()
    # com.dyad_cfo.summation_ave_performance('b_abs')
    # com.triad_cfo.summation_ave_performance('b_abs')
    # com.tetrad_cfo.summation_ave_performance('b_abs')
    # com.dyad_cfo.summation_ave_performance('a_abs')
    # com.triad_cfo.summation_ave_performance('a_abs')
    # com.tetrad_cfo.summation_ave_performance('a_abs')
    # com.dyad_cfo.subtraction_ave_performance()
    # com.triad_cfo.subtraction_ave_performance()
    # com.tetrad_cfo.subtraction_ave_performance()

    ##sum_cfoの比較
    # com.summation_ave_cfo(graph=0)
    # com.summation_ave_cfo(graph=0, mode='b_abs')
    # com.summation_ave_cfo(graph=0, mode='a_abs')
    # com.summation_ave_cfo_bs(graph=0)
    # com.summation_ave_cfo_bs(graph=0, mode='b_abs')
    # com.summation_ave_cfo_bs(graph=0, mode='a_abs')
    # com.subtraction_ave_cfo(graph=0)
    # com.subtraction_ave_cfo_bs(graph=0)

    ##each axis
    # com.dyad_cfo.summation_performance_each_axis()
    # com.triad_cfo.summation_performance_each_axis()
    # com.tetrad_cfo.summation_performance_each_axis()
    # com.dyad_cfo.summation_performance_each_axis('b_abs')
    # com.triad_cfo.summation_performance_each_axis('b_abs')
    # com.tetrad_cfo.summation_performance_each_axis('b_abs')
    # com.dyad_cfo.summation_performance_each_axis('a_abs')
    # com.triad_cfo.summation_performance_each_axis('a_abs')
    # com.tetrad_cfo.summation_performance_each_axis('a_abs')

    # com.dyad_cfo.subtraction_performance_each_axis()
    # com.triad_cfo.subtraction_performance_each_axis()
    # com.tetrad_cfo.subtraction_performance_each_axis()

    # com.performance_deviation()

    # com.dyad_cfo.fcfo_valiance()
    # com.tetrad_cfo.fcfo_valiance(0)

    com.variance_analysis()