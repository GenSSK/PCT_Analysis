# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt
import numpy as np

import Npz
import Combine_analysis

plt.switch_backend('Qt5Agg')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    npz = Npz.NPZ()

    dyad_npz_filename = [
        # # 'dyad/2022-07-29_i.tsunokuni_k.tozuka_2134_CFO.npz',
        # # 'dyad/2022-07-29_i.tsunokuni_y.inoue_1324_CFO.npz',
        # # 'dyad/2022-07-31_m.sugaya_n.ito_1324_CFO.npz',
        # # 'dyad/2022-08-23_r.yanase_ko.kobayashi_1324_CFO.npz',
        # # 'dyad/2022-08-26_y.kobayashi_r.yanase_1324_CFO.npz',
        # # 'dyad/2022-09-05_s.watanabe_ko.kobayashi_1324_CFO.npz',
        # # 'dyad/2022-09-07_k.kobayashi_r.yanase_1324_CFO.npz',
        # # 'dyad/2022-09-07_k.tozuka_ko.kobayashi_1324_CFO.npz',
        # # 'dyad/2022-09-07_y.inoue_k.kobayashi_1324_CFO.npz',
        #
        'dyad/2022-07-29_i.tsunokuni_k.tozuka_3124_CFO.npz',
        'dyad/2022-07-29_i.tsunokuni_y.inoue_1423_CFO.npz',
        # 'dyad/2022-07-31_m.sugaya_n.ito_1234_CFO.npz',
        # 'dyad/2022-08-23_r.yanase_ko.kobayashi_1324_CFO.npz',
        'dyad/2022-08-26_y.kobayashi_r.yanase_1423_CFO.npz',
        # 'dyad/2022-09-05_s.watanabe_ko.kobayashi_1234_CFO.npz',
        'dyad/2022-09-07_k.kobayashi_r.yanase_1324_CFO.npz',
        'dyad/2022-09-07_k.tozuka_ko.kobayashi_1423_CFO.npz',
        # 'dyad/2022-09-07_y.inoue_k.kobayashi_1234_CFO.npz',
    ]

    triad_npz_filename = [
        # # # 'triad/2022-07-31_h.igarashi_ko.kobayashi_t.kassai_1234_CFO.npz', #使えない
        # # 'triad/2022-07-31_k.tozuka_y.inoue_m.nakamura_1234_CFO.npz',
        # # 'triad/2022-07-31_m.sugaya_s.sanuka_m.nakamura_1234_CFO.npz',
        # # 'triad/2022-08-23_s.watanabe_t.kassai_h.nishimura_2134_CFO.npz',
        # # 'triad/2022-08-26_k.kobayashi_r.yanase_ko.kobayashi_1234_CFO.npz',
        # # 'triad/2022-09-05_b.poitrimol_y.inoue_k.tozuka_1234_CFO.npz',
        # # 'triad/2022-09-07_b.poitrimol_k.tozuka_y.kobayashi_1234_CFO.npz',
        # # 'triad/2022-09-07_b.poitrimol_y.inoue_r.yanase_1234_CFO.npz',
        # # 'triad/2022-09-07_ko.kobayashi_k.kobayashi_y.kobayashi_1234_CFO.npz',
        # # 'triad/2022-09-07_r.yanase_k.kobayashi_s.kamioka_1234_CFO.npz',
        #
        'triad/2022-07-31_k.tozuka_y.inoue_m.nakamura_1234_CFO.npz',
        'triad/2022-07-31_m.sugaya_s.sanuka_m.nakamura_2134_CFO.npz',
        'triad/2022-08-23_s.watanabe_t.kassai_h.nishimura_3214_CFO.npz',
        'triad/2022-08-26_k.kobayashi_r.yanase_ko.kobayashi_1234_CFO.npz',
        # 'triad/2022-09-05_b.poitrimol_y.inoue_k.tozuka_2314_CFO.npz',
        # 'triad/2022-09-07_b.poitrimol_k.tozuka_y.kobayashi_3421_CFO.npz',
        # 'triad/2022-09-07_b.poitrimol_y.inoue_r.yanase_4132_CFO.npz',
        # 'triad/2022-09-07_ko.kobayashi_k.kobayashi_y.kobayashi_3412_CFO.npz',
        'triad/2022-09-07_r.yanase_k.kobayashi_s.kamioka_1234_CFO.npz',
    ]

    tetrad_npz_filename = [
        # ## 'tetrad/2022-07-01_y.inoue_k.tozuka_b.poitrimol_y.baba_1234_trans_CFO.npz', #使えない
        # 'tetrad/2022-07-30_k.ohya_r.tanaka_y.baba_m.nakamura_1234_CFO.npz',
        # 'tetrad/2022-07-30_r.tanaka_h.nishimura_k.tozuka_b.poitrimol_1234_CFO.npz',
        # 'tetrad/2022-07-30_s.watanabe_h.nishimura_y.baba_y.yoshida_1234_CFO.npz',
        # 'tetrad/2022-07-30_s.watanabe_ko.kobayashi_y.baba_k.tozuka_1234_CFO.npz',
        # 'tetrad/2022-07-30_y.inoue_y.yoshida_n.ito_s.sanuka_1234_CFO.npz',
        # 'tetrad/2022-08-26_k.kobayashi_r.yanase_ko.kobayashi_h.nishimura_1234_CFO.npz',
        # 'tetrad/2022-09-07_b.poitrimol_y.kobayashi_s.kamioka_y.inoue_1234_CFO.npz',
        # 'tetrad/2022-09-07_k.kobayashi_y.kobayashi_s.kamioka_r.yanase_1234_CFO.npz',
        # 'tetrad/2022-09-07_r.yanase_ko.kobayashi_b.poitrimol_y.kobayashi_1234_CFO.npz',

        # 'tetrad/2022-07-30_k.ohya_r.tanaka_y.baba_m.nakamura_1234_CFO.npz',
        'tetrad/2022-07-30_r.tanaka_h.nishimura_k.tozuka_b.poitrimol_4123_CFO.npz',
        # 'tetrad/2022-07-30_s.watanabe_h.nishimura_y.baba_y.yoshida_3421_CFO.npz',
        'tetrad/2022-07-30_s.watanabe_ko.kobayashi_y.baba_k.tozuka_1324_CFO.npz',
        # 'tetrad/2022-07-30_y.inoue_y.yoshida_n.ito_s.sanuka_1234_CFO.npz',
        'tetrad/2022-08-26_k.kobayashi_r.yanase_ko.kobayashi_h.nishimura_3421_CFO.npz',
        'tetrad/2022-09-07_b.poitrimol_y.kobayashi_s.kamioka_y.inoue_1234_CFO.npz',
        'tetrad/2022-09-07_k.kobayashi_y.kobayashi_s.kamioka_r.yanase_2413_CFO.npz',
        # 'tetrad/2022-09-07_r.yanase_ko.kobayashi_b.poitrimol_y.kobayashi_2341_CFO.npz',

        # 'tetrad/2022-10-07_t.test_t.test_t.test_t.test_1234_CFO.npz',
    ]

    # dyad_npz_filename = [
    #     'dyad/shared/2022-07-29_i.tsunokuni_k.tozuka_3124_CFO.npz',
    #     'dyad/shared/2022-07-29_i.tsunokuni_y.inoue_1423_CFO.npz',
    #     'dyad/shared/2022-08-26_y.kobayashi_r.yanase_1423_CFO.npz',
    #     'dyad/shared/2022-09-07_k.kobayashi_r.yanase_1324_CFO.npz',
    #     'dyad/shared/2022-09-07_k.tozuka_ko.kobayashi_1423_CFO.npz',
    # ]
    #
    # triad_npz_filename = [
    #     'triad/shared/2022-07-31_k.tozuka_y.inoue_m.nakamura_1234_CFO.npz',
    #     'triad/shared/2022-07-31_m.sugaya_s.sanuka_m.nakamura_2134_CFO.npz',
    #     'triad/shared/2022-08-23_s.watanabe_t.kassai_h.nishimura_3214_CFO.npz',
    #     'triad/shared/2022-08-26_k.kobayashi_r.yanase_ko.kobayashi_1234_CFO.npz',
    #     'triad/shared/2022-09-07_r.yanase_k.kobayashi_s.kamioka_1234_CFO.npz',
    # ]
    #
    # tetrad_npz_filename = [
    #     'tetrad/shared/2022-07-30_r.tanaka_h.nishimura_k.tozuka_b.poitrimol_4123_CFO.npz',
    #     'tetrad/shared/2022-07-30_s.watanabe_ko.kobayashi_y.baba_k.tozuka_1324_CFO.npz',
    #     'tetrad/shared/2022-08-26_k.kobayashi_r.yanase_ko.kobayashi_h.nishimura_3421_CFO.npz',
    #     'tetrad/shared/2022-09-07_b.poitrimol_y.kobayashi_s.kamioka_y.inoue_1234_CFO.npz',
    #     'tetrad/shared/2022-09-07_k.kobayashi_y.kobayashi_s.kamioka_r.yanase_2413_CFO.npz',
    # ]

    dyad_npz = npz.select_load(dyad_npz_filename)
    triad_npz = npz.select_load(triad_npz_filename)
    tetrad_npz = npz.select_load(tetrad_npz_filename)

    com = Combine_analysis.combine(dyad_npz, triad_npz, tetrad_npz)

    ##予測確認
    # com.dyad_cfo.show_prediction()
    # com.triad_cfo.show_prediction()
    # com.tetrad_cfo.show_prediction()

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

    # 時系列データすべて確認
    # com.dyad_cfo.plot_time_series('p')
    # com.triad_cfo.plot_time_series('p')

    ##CFOの和を確認
    # com.dyad_cfo.summation_cfo(graph=True, mode='no_abs') #和のCFOの時系列重ね
    # com.triad_cfo.summation_cfo(graph=True, mode='no_abs') #和のCFOの時系列重ね
    # com.tetrad_cfo.summation_cfo(graph=True, mode='no_abs') #和のCFOの時系列重ね
    # com.dyad_cfo.summation_cfo(graph=True, mode='b_abs') #和のCFOの時系列重ね
    # com.triad_cfo.summation_cfo(graph=True, mode='b_abs') #和のCFOの時系列重ね
    # com.tetrad_cfo.summation_cfo(graph=True, mode='b_abs') #和のCFOの時系列重ね
    # com.dyad_cfo.summation_cfo(graph=True, mode='a_abs') #和のCFOの時系列重ね
    # com.triad_cfo.summation_cfo(graph=True, mode='a_abs') #和のCFOの時系列重ね
    # com.tetrad_cfo.summation_cfo(graph=True, mode='a_abs') #和のCFOの時系列重ね
    # com.summation_cfo(graph=True, mode='no_abs') #和のCFOの人数間比較
    # com.summation_cfo(graph=True, mode='b_abs') #和のCFOの人数間比較、前絶対値
    # com.summation_cfo(graph=True, mode='a_abs') #和のCFOの人数間比較、後絶対値

    ##CFOの差の確認
    # com.dyad_cfo.subtraction_cfo(graph=True)
    # com.triad_cfo.subtraction_cfo(graph=True)
    # com.tetrad_cfo.subtraction_cfo(graph=True)
    # com.subtraction_cfo(graph=True)

    ##パフォーマンスの確認
    # com.performance_show() #パフォーマンスの時系列重ね
    # com.dyad_cfo.period_performance(graph=True, mode='H-H') #H-Hのパフォーマンス、グループごと
    # com.triad_cfo.period_performance(graph=True, mode='H-H') #H-Hのパフォーマンス、グループごと
    # com.tetrad_cfo.period_performance(graph=True, mode='H-H') #H-Hのパフォーマンス、グループごと
    # com.dyad_cfo.period_performance(graph=True, mode='M-M') #M-Mのパフォーマンス、グループごと
    # com.triad_cfo.period_performance(graph=True, mode='M-M') #M-Mのパフォーマンス、グループごと
    # com.tetrad_cfo.period_performance(graph=True, mode='M-M') #M-Mのパフォーマンス、グループごと
    # com.dyad_cfo.period_performance_cooperation(graph=True) #H-HとM-Mのパフォーマンス比較、グループごと
    # com.triad_cfo.period_performance_cooperation(graph=True)
    # com.tetrad_cfo.period_performance_cooperation(graph=True)
    # com.performance_comparison() #人数間のパフォーマンス比較
    # com.performance_comparison('h-h')
    # com.performance_comparison('m-m')

    com.dyad_cfo.period_performance_cooperation_and_solo(graph=True)
    # com.triad_cfo.period_performance_cooperation_and_solo(graph=True)
    # com.tetrad_cfo.period_performance_cooperation_and_solo(graph=True)


    ##OCFOとパフォーマンスの確認
    # com.dyad_cfo.each_ocfo_performance('error') #TODO not yet
    # com.triad_cfo.each_ocfo_performance('error') #TODO not yet
    # com.tetrad_cfo.each_ocfo_performance('error') #TODO not yet
    # com.dyad_cfo.ocfo_performance() #TODO not yet
    # com.triad_cfo.ocfo_performance() #TODO not yet
    # com.tetrad_cfo.ocfo_performance() #TODO not yet

    ##CFOの差とパフォーマンスの確認
    # com.dyad_cfo.subtraction_performance()
    # com.triad_cfo.subtraction_performance()
    # com.tetrad_cfo.subtraction_performance()

    #CFOの和とパフォーマンスの確認
    # com.dyad_cfo.summation_performance()
    # com.triad_cfo.summation_performance()
    # com.tetrad_cfo.summation_performance()
    # com.dyad_cfo.summation_performance('b_abs')
    # com.triad_cfo.summation_performance('b_abs')
    # com.tetrad_cfo.summation_performance('b_abs')
    # com.dyad_cfo.summation_performance('a_abs')
    # com.triad_cfo.summation_performance('a_abs')
    # com.tetrad_cfo.summation_performance('a_abs')

    ##CFOの和とCFOの差とパフォーマンスの関係
    # com.dyad_cfo.sum_sub_performance()

    ##CFOの和とCFOの差の関係
    # com.dyad_cfo.sum_sub()
    # com.triad_cfo.sum_sub()
    # com.tetrad_cfo.sum_sub()

    ##パフォーマンス同士の相関
    # com.performance_relation()

    ##パフォーマンスの比較
    # com.performance_hist()
    # com.performance_bootstrap() #TODO not yet

    ##CFOのpitchとrollの関係
    # com.dyad_cfo.CFO_relation_axis() #TODO not yet
    # com.triad_cfo.CFO_relation_axis() #TODO not yet
    # com.tetrad_cfo.CFO_relation_axis() #TODO not yet
    # com.dyad_cfo.CFO_relation_axis_3sec() #TODO not yet
    # com.triad_cfo.CFO_relation_axis_3sec() #TODO not yet
    # com.tetrad_cfo.CFO_relation_axis_3sec() #TODO not yet

    ##sum_cfoとパフォーマンスの相関
    # com.dyad_cfo.summation_ave_performance() #TODO わからん（多分消していい、いまデフォで人数平均してる）
    # com.triad_cfo.summation_ave_performance()
    # com.tetrad_cfo.summation_ave_performance()
    # com.dyad_cfo.summation_ave_performance('b_abs')
    # com.triad_cfo.summation_ave_performance('b_abs')
    # com.tetrad_cfo.summation_ave_performance('b_abs')
    # com.dyad_cfo.summation_ave_performance('a_abs')
    # com.triad_cfo.summation_ave_performance('a_abs')
    # com.tetrad_cfo.summation_ave_performance('a_abs')
    # com.dyad_cfo.subtraction_ave_performance() #TODO わからん
    # com.triad_cfo.subtraction_ave_performance()
    # com.tetrad_cfo.subtraction_ave_performance()

    ##sum_cfoの比較
    # com.summation_ave_cfo(graph=True) #TODO わからん
    # com.summation_ave_cfo(graph=True, mode='b_abs')
    # com.summation_ave_cfo(graph=True, mode='a_abs')
    # com.summation_ave_cfo_bs(graph=True)
    # com.summation_ave_cfo_bs(graph=True, mode='b_abs')
    # com.summation_ave_cfo_bs(graph=True, mode='a_abs')
    # com.subtraction_ave_cfo(graph=True) #TODO わからん
    # com.subtraction_ave_cfo_bs(graph=True)

    ##CFOの和とパフォーマンスの相関（each axis）
    # com.dyad_cfo.summation_performance_each_axis() #TODO not yet
    # com.triad_cfo.summation_performance_each_axis() #TODO not yet
    # com.tetrad_cfo.summation_performance_each_axis() #TODO not yet
    # com.dyad_cfo.summation_performance_each_axis('b_abs') #TODO not yet
    # com.triad_cfo.summation_performance_each_axis('b_abs') #TODO not yet
    # com.tetrad_cfo.summation_performance_each_axis('b_abs') #TODO not yet
    # com.dyad_cfo.summation_performance_each_axis('a_abs') #TODO not yet
    # com.triad_cfo.summation_performance_each_axis('a_abs') #TODO not yet
    # com.tetrad_cfo.summation_performance_each_axis('a_abs') #TODO not yet

    ##CFOの差とパフォーマンスの相関（each axis）
    # com.dyad_cfo.subtraction_performance_each_axis() #TODO not yet
    # com.triad_cfo.subtraction_performance_each_axis() #TODO not yet
    # com.tetrad_cfo.subtraction_performance_each_axis() #TODO not yet

    ##パフォーマンス（each axis）
    # com.dyad_cfo.period_performance_each_axis(mode='H-H', graph=True)
    # com.triad_cfo.period_performance_each_axis(mode='H-H', graph=True)
    # com.tetrad_cfo.period_performance_each_axis(mode='H-H', graph=True)
    # com.dyad_cfo.period_performance_each_axis(mode='M-M', graph=True)
    # com.triad_cfo.period_performance_each_axis(mode='M-M', graph=True)
    # com.tetrad_cfo.period_performance_each_axis(mode='M-M', graph=True)
    # com.dyad_cfo.period_performance_cooperation_each_axis(graph=True)
    # com.triad_cfo.period_performance_cooperation_each_axis(graph=True)
    # com.tetrad_cfo.period_performance_cooperation_each_axis(graph=True)
    #TODO Compparisonがない

    ##CFOの和の確認（Combine）
    # com.dyad_cfo.summation_cfo_combine(graph=True, mode='no_abs') #TODO not yet
    # com.triad_cfo.summation_cfo_combine(graph=True, mode='no_abs') #TODO not yet
    # com.tetrad_cfo.summation_cfo_combine(graph=True, mode='no_abs') #TODO not yet
    # com.dyad_cfo.summation_cfo_combine(graph=True, mode='b_abs') #TODO not yet
    # com.triad_cfo.summation_cfo_combine(graph=True, mode='b_abs') #TODO not yet
    # com.tetrad_cfo.summation_cfo_combine(graph=True, mode='b_abs') #TODO not yet
    # com.dyad_cfo.summation_cfo_combine(graph=True, mode='a_abs') #TODO not yet
    # com.triad_cfo.summation_cfo_combine(graph=True, mode='a_abs') #TODO not yet
    # com.tetrad_cfo.summation_cfo_combine(graph=True, mode='a_abs') #TODO not yet
    # com.summation_cfo_combine(graph=True, mode='no_abs')
    # com.summation_cfo_combine(graph=True, mode='b_abs')
    # com.summation_cfo_combine(graph=True, mode='a_abs')

    ##CFOの差の確認（Combine）
    # com.dyad_cfo.subtraction_cfo_combine(graph=True) #TODO not yet
    # com.triad_cfo.subtraction_cfo_combine(graph=True) #TODO not yet
    # com.tetrad_cfo.subtraction_cfo_combine(graph=True) #TODO not yet
    # com.subtraction_cfo_combine(graph=True)

    ##CFOの和とパフォーマンスの相関（Combine）
    # com.dyad_cfo.subtraction_performance_combine()
    # com.triad_cfo.subtraction_performance_combine()
    # com.tetrad_cfo.subtraction_performance_combine()
    # com.dyad_cfo.summation_performance_combine(mode='no_abs')
    # com.triad_cfo.summation_performance_combine(mode='no_abs')
    # com.tetrad_cfo.summation_performance_combine(mode='no_abs')
    # com.dyad_cfo.summation_performance_combine(mode='b_abs')
    # com.triad_cfo.summation_performance_combine(mode='b_abs')
    # com.tetrad_cfo.summation_performance_combine(mode='b_abs')
    # com.dyad_cfo.summation_performance_combine(mode='a_abs')
    # com.triad_cfo.summation_performance_combine(mode='a_abs')
    # com.tetrad_cfo.summation_performance_combine(mode='a_abs')

    ##CFOの和とパフォーマンスの相関（Combine）
    # com.summation_cfo_combine(graph=True, mode='no_abs')
    # com.summation_cfo_combine(graph=True, mode='b_abs')
    # com.summation_cfo_combine(graph=True, mode='a_abs')

    ##CFOの差とパフォーマンスの相関（Combine）
    # com.subtraction_cfo_combine(graph=True)


    ##パフォーマンスの分散
    # com.performance_deviation() #TODO not yet

    ##CFOの分散
    # com.dyad_cfo.fcfo_valiance()  #グループごとの分散　 #TODO not yet
    # com.tetrad_cfo.fcfo_valiance(graph=True)  #グループごとの分散　 #TODO not yet

    # com.variance_analysis('no_abs')
    # com.variance_analysis('b_abs')
    # com.variance_analysis('a_abs')

    # com.dyad_cfo.tf_graph_sub()
    # com.triad_cfo.tf_graph_sub()
    # com.tetrad_cfo.tf_graph_sub()

    # com.dyad_cfo.tf_cfo_sub()
    # com.triad_cfo.tf_cfo_sub()
    # com.tetrad_cfo.tf_cfo_sub()

    # com.dyad_cfo.work_calc()
    # com.dyad_cfo.work_diff(graph=True)
    # com.triad_cfo.work_diff(graph=True)
    # com.tetrad_cfo.work_diff(graph=True)

    # com.dyad_cfo.work_calc_rs()
    # com.dyad_cfo.work_diff_rs(graph=True)

    # com.dyad_cfo.estimation_plate_accel(graph=True)


    # com.dyad_cfo.get_summation_force(mode='b_abs',graph=True)
    # com.dyad_cfo.get_summation_force(mode='b_abs',graph=True, source='model')

    ##FTR(Force Transfer Ratio)の時間経過
    # com.dyad_cfo.get_ftr(graph=True)
    # com.triad_cfo.get_ftr(graph=True)
    # com.tetrad_cfo.get_ftr(graph=True)
    ##FTR(Force Transfer Ratio)のピリオド経過
    # com.dyad_cfo.get_ftr_3sec(graph=True)
    # com.triad_cfo.get_ftr_3sec(graph=True)
    # com.tetrad_cfo.get_ftr_3sec(graph=True)
    # com.dyad_cfo.get_ftr_3sec(graph=True, source='model')
    # com.triad_cfo.get_ftr_3sec(graph=True, source='model')
    # com.tetrad_cfo.get_ftr_3sec(graph=True, source='model')

    ##FTR(Force Transfer Ratio)の比較
    # com.ftr_3sec(source='human')
    # com.ftr_3sec(source='model')
    ##pitchとrollを合わせたFTR(Force Transfer Ratio)の比較
    # com.ftr_3sec_combine(source='human')
    # com.ftr_3sec_combine(source='model')
    # com.ftr_3sec_diff()
    # com.ftr_3sec_combine_diff()

    ##pitchとrollを合わせたFTR(Force Transfer Ratio)の時間経過
    # com.dyad_cfo.get_ftr_combine(graph=True)
    # com.triad_cfo.get_ftr_combine(graph=True)
    # com.tetrad_cfo.get_ftr_combine(graph=True)
    ##pitchとrollを合わせたFTR(Force Transfer Ratio)のピリオド経過
    # com.dyad_cfo.get_ftr_combine_3sec(graph=True)
    # com.triad_cfo.get_ftr_combine_3sec(graph=True)
    # com.tetrad_cfo.get_ftr_combine_3sec(graph=True)

    ## FTR(Force Transfer Ratio)とCFOの和
    # com.dyad_cfo.summationCFO_ftr('no_abs')
    # com.dyad_cfo.summationCFO_ftr('b_abs')
    # com.dyad_cfo.summationCFO_ftr('a_abs')
    # com.triad_cfo.summationCFO_ftr('no_abs')
    # com.triad_cfo.summationCFO_ftr('b_abs')
    # com.triad_cfo.summationCFO_ftr('a_abs')
    # com.tetrad_cfo.summationCFO_ftr('no_abs')
    # com.tetrad_cfo.summationCFO_ftr('b_abs')
    # com.tetrad_cfo.summationCFO_ftr('a_abs')

    ## FTR(Force Transfer Ratio)とCFOの和（ピリオド）
    # com.dyad_cfo.summationCFO_ftr_3sec('no_abs')
    # com.dyad_cfo.summationCFO_ftr_3sec('b_abs')
    # com.dyad_cfo.summationCFO_ftr_3sec('a_abs')
    # com.triad_cfo.summationCFO_ftr_3sec('no_abs')
    # com.triad_cfo.summationCFO_ftr_3sec('b_abs')
    # com.triad_cfo.summationCFO_ftr_3sec('a_abs')
    # com.tetrad_cfo.summationCFO_ftr_3sec('no_abs')
    # com.tetrad_cfo.summationCFO_ftr_3sec('b_abs')
    # com.tetrad_cfo.summationCFO_ftr_3sec('a_abs')

    ## FTR(Force Transfer Ratio)とCFOの差
    # com.dyad_cfo.subtractionCFO_ftr()
    # com.triad_cfo.subtractionCFO_ftr()
    # com.tetrad_cfo.subtractionCFO_ftr()

    ## FTR(Force Transfer Ratio)とCFOの差（ピリオド）
    # com.dyad_cfo.subtractionCFO_ftr_3sec()
    # com.triad_cfo.subtractionCFO_ftr_3sec()
    # com.tetrad_cfo.subtractionCFO_ftr_3sec()

    ## FTRとパフォーマンス
    # com.dyad_cfo.performance_ftr()
    # com.triad_cfo.performance_ftr()
    # com.tetrad_cfo.performance_ftr()
    # com.dyad_cfo.performance_ftr('h-h')
    # com.triad_cfo.performance_ftr('h-h')
    # com.tetrad_cfo.performance_ftr('h-h`')

    # com.dyad_cfo.performance_ftr_diff()
    # com.triad_cfo.performance_ftr_diff()
    # com.tetrad_cfo.performance_ftr_diff()

    ## FTRとパフォーマンスの比較
    # com.performance_ftr()
    # com.performance_ftr('h-h')

    # IEFとパフォーマンス
    # com.dyad_cfo.performance_ief()
    # com.triad_cfo.performance_ief()
    # com.tetrad_cfo.performance_ief()
    # com.dyad_cfo.performance_ief('h-h')
    # com.triad_cfo.performance_ief('h-h')
    # com.tetrad_cfo.performance_ief('h-h')

    # IEFの3秒間のデータ比較
    # com.ief_3sec()
    # com.ief_3sec(source='model')
    # com.ief_3sec_combine()
    # com.ief_3sec_combine(source='model')
    # com.ief_3sec_diff()
    # com.ief_3sec_combine_diff()

    # IEFとCFOの和の関係
    # com.dyad_cfo.summationCFO_ief_3sec('no_abs')
    # com.dyad_cfo.summationCFO_ief_3sec('b_abs')
    # com.dyad_cfo.summationCFO_ief_3sec('a_abs')
    # com.triad_cfo.summationCFO_ief_3sec('no_abs')
    # com.triad_cfo.summationCFO_ief_3sec('b_abs')
    # com.triad_cfo.summationCFO_ief_3sec('a_abs')
    # com.tetrad_cfo.summationCFO_ief_3sec('no_abs')
    # com.tetrad_cfo.summationCFO_ief_3sec('b_abs')
    # com.tetrad_cfo.summationCFO_ief_3sec('a_abs')

    # IEFとCFOの差の関係
    # com.dyad_cfo.subtractionCFO_ief_3sec()

    # IEFとパフォーマンスの関係
    # com.dyad_cfo.performance_ief_diff()
    # com.triad_cfo.performance_ief_diff()
    # com.tetrad_cfo.performance_ief_diff()

    # com.dyad_cfo.calc_hilbert('h-h')
    # com.dyad_cfo.calc_STFT()
    # com.dyad_cfo.show_coherence_stft()

    # Relative Phase
    # com.dyad_cfo.relative_phase(type='position', source='human', sigma=100, graph=True)
    # com.dyad_cfo.relative_phase(type='position', source='model', sigma=100, graph=True)
    # com.dyad_cfo.relative_phase(type='force', source='human', sigma=100, graph=True)
    # com.dyad_cfo.relative_phase(type='force', source='model', sigma=100, graph=True)
    # com.dyad_cfo.relative_phase(type='pcfo', sigma=100, graph=True)
    # com.dyad_cfo.relative_phase(type='fcfo', sigma=100, graph=True)
    # com.dyad_cfo.relative_phase_3sec(type='position', source='human', sigma=100, graph=True)
    # com.dyad_cfo.relative_phase_3sec(type='position', source='model', sigma=100, graph=True)
    # com.dyad_cfo.relative_phase_3sec(type='force', source='human', sigma=100, graph=True)
    # com.dyad_cfo.relative_phase_3sec(type='force', source='model', sigma=100, graph=True)
    # com.dyad_cfo.relative_phase_3sec(type='pcfo', source='human', sigma=100, graph=True)
    # com.dyad_cfo.relative_phase_3sec(type='fcfo', source='human', sigma=100, graph=True)

    # com.dyad_cfo.relative_phase_performance(type='pcfo', sigma=10, graph=True, dec=1)
    # com.dyad_cfo.relative_phase_performance(type='fcfo', sigma=10, graph=True, dec=10)

    # com.dyad_cfo.relative_phase_performance_filter(type='fcfo', sigma=10, graph=True, dec=1)

    # com.dyad_cfo.relative_phase_filter(type='pcfo', source='human', sigma=100, graph=True, min_freq=0, max_freq=50, step=1)
    # com.dyad_cfo.relative_phase_filter(type='fcfo', source='human', sigma=100, graph=True, min_freq=0, max_freq=10, step=0.2)


    # com.dyad_cfo.relative_phase_performance_filter(type='position', sigma=100, graph=True, dec=1, min_freq=0, max_freq=50, step=1)
    # com.dyad_cfo.relative_phase_performance_filter(type='force', sigma=100, graph=True, dec=1, min_freq=0, max_freq=10, step=0.5)
    # com.dyad_cfo.relative_phase_performance_filter(type='pcfo', sigma=100, graph=True, dec=1, min_freq=0, max_freq=50, step=1)
    # com.dyad_cfo.relative_phase_performance_filter(type='fcfo', sigma=100, graph=True, dec=1, min_freq=0, max_freq=10, step=0.5)

    # com.dyad_cfo.relative_phase_performance_reg_model(type='position', sigma=100, graph=True)
    # com.dyad_cfo.relative_phase_performance_reg_model(type='force', sigma=100, graph=True)
    # com.dyad_cfo.relative_phase_performance_reg_model(type='pcfo', sigma=100, graph=True)
    # com.dyad_cfo.relative_phase_performance_reg_model(type='fcfo', sigma=100, graph=True)

    # com.dyad_cfo.relative_phase_performance_reg_model_filter(type='position', sigma=100, graph=True, dec=1, min_freq=0, max_freq=50, step=25)
    # com.dyad_cfo.relative_phase_performance_reg_model_filter(type='force', sigma=100, graph=True, dec=1, min_freq=0, max_freq=10, step=5)
    # com.dyad_cfo.relative_phase_performance_reg_model_filter(type='pcfo', sigma=100, graph=True, dec=1, min_freq=0, max_freq=50, step=25)
    # com.dyad_cfo.relative_phase_performance_reg_model_filter(type='fcfo', sigma=100, graph=True, dec=1, min_freq=0, max_freq=10, step=5)

