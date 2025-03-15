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


# plt.switch_backend('Qt5Agg')

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
        '2024-07-22_PP_Dyad_All_a.noto_mi.nakamura_Discrete_Random_1.npz',
        '2024-07-25_PP_Dyad_All_r.zyuhou_h.kitaiwa_Discrete_Random_1.npz',
    ]

    AdPD_npz_filename = [
        # '2023-01-24_b.poitrimol_k.tozuka_Admittance(PD).npz',
        # '2023-01-24_h.nishimura_y.inoue_Admittance(PD).npz',
        # '2023-01-24_i.kato_ko.kobayashi_Admittance(PD).npz',
        # '2023-01-24_m.nakamura_r.tanaka_Admittance(PD).npz',
        # '2023-01-24_t.onogawa_n.ito_Admittance(PD).npz',
        # '2023-01-24_t.oriuchi_s.watanabe_Admittance(PD).npz',
        # '2023-01-24_y.yamada_s.tsuchiya_Admittance(PD).npz',
        # '2023-01-24_y.yoshida_k.kobayashi_Admittance(PD).npz',
        # '2023-02-13_y.inoue_g.sasaki_Admittance(PD).npz',
        '2024-06-07_Adpos_Dyad_All_k.kobayashi_b.poitrimol_Discrete_Random_1.npz',
        '2024-06-25_Adpos_Dyad_All_g.otsuka_h.kitaiwa_Discrete_Random_1.npz',
        '2024-06-25_Adpos_Dyad_All_n.ito_y.takahashi_Discrete_Random_1.npz',
        '2024-06-25_Adpos_Dyad_All_t.onogawa_s.tsuchiya_Discrete_Random_1.npz',
        '2024-06-27_Adpos_Dyad_All_mi.nakamura_a.noto_Discrete_Random_1.npz',
        '2024-06-27_Adpos_Dyad_All_r.tanaka_r.morishita_Discrete_Random_1.npz',
        '2024-06-27_Adpos_Dyad_All_r.yanase_i.kato_Discrete_Random_1.npz',
        '2024-06-27_Adpos_Dyad_All_y.yamada_r.zyuhou_Discrete_Random_1.npz',
        '2024-06-28_Adpos_Dyad_All_k.tozuka_r.hiratsuka_Discrete_Random_1.npz',
        '2024-08-04_Adpos_Dyad_All_s.tahara_m.nakamura_Discrete_Random_1.npz',
    ]

    # AdAc_npz_filename = [
    #     '2023-01-24_b.poitrimol_k.tozuka_Admittance(Accel).npz',
    #     '2023-01-24_h.nishimura_y.inoue_Admittance(Accel).npz',
    #     '2023-01-24_i.kato_ko.kobayashi_Admittance(Accel).npz',
    #     # '2023-01-24_m.nakamura_r.tanaka_Admittance(Accel).npz',
    #     '2023-01-24_t.onogawa_n.ito_Admittance(Accel).npz',
    #     # '2023-01-24_t.oriuchi_s.watanabe_Admittance(Accel).npz',
    #     '2023-01-24_y.yamada_s.tsuchiya_Admittance(Accel).npz',
    #     '2023-01-24_y.yoshida_k.kobayashi_Admittance(Accel).npz',
    # ]

    Bi_npz_filename = [
        '2023-01-24_b.poitrimol_k.tozuka_Bilateral.npz',
        '2023-01-24_h.nishimura_y.inoue_Bilateral.npz',
        '2023-01-24_i.kato_ko.kobayashi_Bilateral.npz',
        '2023-01-24_m.nakamura_r.tanaka_Bilateral.npz',
        '2023-01-24_t.onogawa_n.ito_Bilateral.npz',
        '2023-01-24_t.oriuchi_s.watanabe_Bilateral.npz',
        '2023-01-24_y.yamada_s.tsuchiya_Bilateral.npz',
        '2023-01-24_y.yoshida_k.kobayashi_Bilateral.npz',
        '2024-07-19_Bi_Dyad_All_r.zyuhou_h.kitaiwa_Discrete_Random_1.npz',
        '2024-07-22_Bi_Dyad_All_r.yanase_m.kashiwagi_Discrete_Random_1.npz',
    ]

    PP_npz = npz.select_load(PP_npz_filename)
    AdPD_npz = npz.select_load(AdPD_npz_filename)
    # AdAc_npz = npz.select_load(AdAc_npz_filename)
    Bi_npz = npz.select_load(Bi_npz_filename)

    com: Combine_analysis.combine = Combine_analysis.combine(PP_npz, AdPD_npz, Bi_npz)


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
    # com.performance_comparison()

    # ave = com.performance_each_ave(0)
    # ave.to_csv('csv/performance_ave.csv', index=False)

    ##位置の差分
    # com.PP.subtraction_position(0)
    # com.AdPD.subtraction_position(0)
    # com.AdAc.subtraction_position(0)
    # com.Bi.subtraction_position(0)
    # com.subtraction_position_3sec(0)
    # df = com.subtraction_position_ave()
    # df.to_csv('csv/subtraction_position_ave.csv', index=False)

    ##力の和
    # com.PP.summation_force(0)
    # com.AdPD.summation_force(0)
    # com.AdAc.summation_force(0)
    # com.Bi.summation_force(0)
    # com.summation_force_3sec(0)

    ##位置の差分と力の和のCSV出力
    # df_sum_force = com.summation_force_3sec()
    # df_sub_position = com.subtraction_position_3sec()
    # df_calc = pd.concat([df_sub_position, df_sum_force['sum_force']], axis=1)
    # print(df_calc)
    # df_calc.to_csv('csv/summation_subtraction.csv', index=False)

    ##力の和と加速度のグラフの確認
    # com.PP.estimation_plate_accel()
    # com.AdPD.estimation_plate_accel()
    # com.AdAc.estimation_plate_accel()
    # com.Bi.estimation_plate_accel()
    # com.estimation_inertia(graph=0)

    # com.plot_ine_improve()
    # com.plot_performance_improve()
    # com.plot_maf_improve()

    # df_ef = com.plot_compare_effort()
    # df_per = com.plot_compare_performance()
    # df_mad = com.maf()
    # df_t2t = com.time_to_target()
    # df_list = [df_ef, df_per, df_mad, df_t2t]
    # df = pd.concat([_ for _ in df_list], axis=1)
    # df.to_csv('csv/compare.csv', index=False)


    # com.performance_survey_diff()

    # df_survey = com.prediction_performance_by_survey_diff()
    # df_survey.to_csv('csv/performance_survey_diff.csv', index=False)
    com.prediction_performance_by_survey_diff_aic()

    # df_compare_period = com.get_df_period()
    # df_compare_period.to_csv('csv/compare_period.csv', index=False)

    # com.plot_period()