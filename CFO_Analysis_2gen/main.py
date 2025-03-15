# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import sys, os
import matplotlib.pyplot as plt
import numpy as np

import Npz
import Combine_analysis
import CFO_analysis_compare
import Trajectory_analysis

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from mypackage import StringUtils as su


# plt.switch_backend('Qt5Agg')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    npz = Npz.NPZ()

    dyad_npz_filename = [
        # '2024-03-02_AdABC_Dyad_h.nakamura_k.tozuka_Circle_1_CFO.npz',
        # '2024-03-02_AdABC_Dyad_h.nakamura_k.tozuka_Discrete_Random_1_CFO.npz',
        # '2024-03-02_AdABC_Dyad_h.nakamura_k.tozuka_Lemniscate_1_CFO.npz',
        # '2024-03-02_AdABC_Dyad_h.nakamura_k.tozuka_Random_1_CFO.npz',
        # '2024-03-02_AdABC_Dyad_h.nakamura_k.tozuka_RoseCurve_1_CFO.npz',
        #
        # '2024-03-02_AdABC_Dyad_s.tsuchiya_h.nakamura_Circle_1_CFO.npz',
        # '2024-03-02_AdABC_Dyad_s.tsuchiya_h.nakamura_Discrete_Random_1_CFO.npz',
        # '2024-03-02_AdABC_Dyad_s.tsuchiya_h.nakamura_Lemniscate_1_CFO.npz',
        # '2024-03-02_AdABC_Dyad_s.tsuchiya_h.nakamura_Random_1_CFO.npz',
        # '2024-03-02_AdABC_Dyad_s.tsuchiya_h.nakamura_RoseCurve_1_CFO.npz',

        '2024-02-09_AdABC_Dyad_k.tozuka_k.kobayashi_Circle_1_CFO.npz',
        '2024-02-09_AdABC_Dyad_k.tozuka_k.kobayashi_Discrete_Random_1_CFO.npz',
        '2024-02-09_AdABC_Dyad_k.tozuka_k.kobayashi_Lemniscate_1_CFO.npz',
        '2024-02-09_AdABC_Dyad_k.tozuka_k.kobayashi_Random_1_CFO.npz',
        '2024-02-09_AdABC_Dyad_k.tozuka_k.kobayashi_RoseCurve_1_CFO.npz',

        # '2024-04-09_AdABC_Dyad_All_s.tsuchiya_h.kitaiwa_RoseCurve_1_CFO.npz',
        # '2024-04-10_AdABC_Dyad_All_i.kato_n.ito_RoseCurve_1_CFO.npz',
        # '2024-04-10_AdABC_Dyad_All_l.nicolas_h.nakamura_RoseCurve_1_CFO.npz',
        # '2024-04-10_AdABC_Dyad_All_r.yanase_g.otsuka_RoseCurve_1_CFO.npz',

    ]

    triad_npz_filename = [
        '2024-02-29_AdABC_Triad_t.onogawa_r.yanase_g.otsuka_Circle_1_CFO.npz',
        '2024-02-29_AdABC_Triad_t.onogawa_r.yanase_g.otsuka_Discrete_Random_1_CFO.npz',
        '2024-02-29_AdABC_Triad_t.onogawa_r.yanase_g.otsuka_Lemniscate_1_CFO.npz',
        '2024-02-29_AdABC_Triad_t.onogawa_r.yanase_g.otsuka_Random_1_CFO.npz',
        '2024-02-29_AdABC_Triad_t.onogawa_r.yanase_g.otsuka_RoseCurve_1_CFO.npz',

        '2024-03-02_AdABC_Triad_k.tozuka_i.kato_s.tsuchiya_Circle_1_CFO.npz',
        '2024-03-02_AdABC_Triad_k.tozuka_i.kato_s.tsuchiya_Discrete_Random_1_CFO.npz',
        '2024-03-02_AdABC_Triad_k.tozuka_i.kato_s.tsuchiya_Lemniscate_1_CFO.npz',
        '2024-03-02_AdABC_Triad_k.tozuka_i.kato_s.tsuchiya_Random_1_CFO.npz',
        '2024-03-02_AdABC_Triad_k.tozuka_i.kato_s.tsuchiya_RoseCurve_1_CFO.npz',
    ]

    tetrad_npz_filename = [
        '2024-03-01_AdABC_Tetrad_t.onogawa_r.yanase_g.otsuka_n.ito_Circle_1_CFO.npz',
        '2024-03-01_AdABC_Tetrad_t.onogawa_r.yanase_g.otsuka_n.ito_Discrete_Random_1_CFO.npz',
        '2024-03-02_AdABC_Tetrad_t.onogawa_r.yanase_g.otsuka_n.ito_Lemniscate_1_CFO.npz',
        '2024-03-02_AdABC_Tetrad_t.onogawa_r.yanase_g.otsuka_n.ito_Random_1_CFO.npz',
        '2024-03-02_AdABC_Tetrad_t.onogawa_r.yanase_g.otsuka_n.ito_RoseCurve_1_CFO.npz',

        '2024-03-13_AdABC_Tetrad_k.tozuka_k.kobayashi_r.tanaka_n.ito_Circle_1_CFO.npz',
        '2024-03-13_AdABC_Tetrad_k.tozuka_k.kobayashi_r.tanaka_n.ito_Discrete_Random_1_CFO.npz',
        '2024-03-13_AdABC_Tetrad_k.tozuka_k.kobayashi_r.tanaka_n.ito_Lemniscate_1_CFO.npz',
        '2024-03-13_AdABC_Tetrad_k.tozuka_k.kobayashi_r.tanaka_n.ito_Random_1_CFO.npz',
        '2024-03-13_AdABC_Tetrad_k.tozuka_k.kobayashi_r.tanaka_n.ito_RoseCurve_1_CFO.npz',
    ]


    dyad_ind_npz = npz.select_load("/nfs/ssk-storage/data/cfo/dyad/", dyad_npz_filename)
    triad_ind_npz = npz.select_load("/nfs/ssk-storage/data/cfo/triad/", triad_npz_filename)
    tetrad_ind_npz = npz.select_load("/nfs/ssk-storage/data/cfo/tetrad/", tetrad_npz_filename)

    dyad_shd_npz = npz.select_load("/nfs/ssk-storage/data/cfo/dyad/shared/", dyad_npz_filename)
    triad_shd_npz = npz.select_load("/nfs/ssk-storage/data/cfo/triad/shared/", triad_npz_filename)
    tetrad_shd_npz = npz.select_load("/nfs/ssk-storage/data/cfo/tetrad/shared/", tetrad_npz_filename)

    # com = Combine_analysis.combine(dyad_ind_npz, triad_ind_npz, tetrad_ind_npz)

    comp_dyad = CFO_analysis_compare.CFO_compare(dyad_ind_npz, dyad_shd_npz, 'dyad', dyad_npz_filename)
    comp_triad = CFO_analysis_compare.CFO_compare(triad_ind_npz, triad_shd_npz, 'triad', triad_npz_filename)
    comp_tetrad = CFO_analysis_compare.CFO_compare(tetrad_ind_npz, tetrad_shd_npz, 'tetrad', tetrad_npz_filename)

    # trj = Trajectory_analysis.TrajectoryAnalysis(dyad_ind_npz, triad_ind_npz, tetrad_ind_npz, dyad_npz_filename, triad_npz_filename, tetrad_npz_filename)
    trj = Trajectory_analysis.TrajectoryAnalysis(dyad_shd_npz, triad_shd_npz, tetrad_shd_npz, dyad_npz_filename, triad_npz_filename, tetrad_npz_filename)


    ## indとshdの比較
    # comp_dyad.show_prediction()
    # comp_triad.show_prediction()
    # comp_tetrad.show_prediction()

    ##予測確認
    # trj.com_circle.dyad_cfo.show_prediction()
    # trj.com_lemniscate.dyad_cfo.show_prediction()
    # trj.com_rose_curve.dyad_cfo.show_prediction()
    # trj.com_random.dyad_cfo.show_prediction()
    # trj.com_discrete_random.dyad_cfo.show_prediction()
    # trj.com_circle.triad_cfo.show_prediction()
    # trj.com_lemniscate.triad_cfo.show_prediction()
    # trj.com_rose_curve.triad_cfo.show_prediction()
    # trj.com_random.triad_cfo.show_prediction()
    # trj.com_discrete_random.triad_cfo.show_prediction()
    # trj.com_circle.tetrad_cfo.show_prediction()
    # trj.com_lemniscate.tetrad_cfo.show_prediction()
    # trj.com_rose_curve.tetrad_cfo.show_prediction()
    # trj.com_random.tetrad_cfo.show_prediction()
    # trj.com_discrete_random.tetrad_cfo.show_prediction()

    ##タスク確認
    # com.dyad_cfo.task_show()
    # com.triad_cfo.task_show()
    # com.tetrad_cfo.task_show()

    ##タスク確認（単独あり）
    # com.dyad_cfo.task_show_solo()
    # com.triad_cfo.task_show_solo()
    # com.tetrad_cfo.task_show_solo()

    ##CFO確認
    # trj.com_circle.dyad_cfo.cfo_sub()
    # trj.com_lemniscate.dyad_cfo.cfo_sub()
    # trj.com_rose_curve.dyad_cfo.cfo_sub()
    # trj.com_random.dyad_cfo.cfo_sub()
    # trj.com_discrete_random.dyad_cfo.cfo_sub()
    # trj.com_circle.triad_cfo.cfo_sub()
    # trj.com_lemniscate.triad_cfo.cfo_sub()
    # trj.com_rose_curve.triad_cfo.cfo_sub()
    # trj.com_random.triad_cfo.cfo_sub()
    # trj.com_discrete_random.triad_cfo.cfo_sub()
    # trj.com_circle.tetrad_cfo.cfo_sub()
    # trj.com_lemniscate.tetrad_cfo.cfo_sub()
    # trj.com_rose_curve.tetrad_cfo.cfo_sub()
    # trj.com_random.tetrad_cfo.cfo_sub()
    # trj.com_discrete_random.tetrad_cfo.cfo_sub()

    ##CFOのすべてを確認
    # trj.com_rose_curve.dyad_cfo.get_cfo()
    # trj.com_rose_curve.dyad_cfo.cfo_all_sub()
    # trj.com_rose_curve.dyad_cfo.cfo_group()
    # trj.com_rose_curve.dyad_cfo.cfo_combine_all_sub(cutoff=2.0)

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
    # trj.com_rose_curve.summation_cfo_combine(graph=True, mode='no_abs') #和のCFOの時系列重ね
    # trj.com_rose_curve.summation_cfo_combine(graph=True, mode='b_abs') #和のCFOの時系列重ね
    # trj.com_rose_curve.summation_cfo_combine(graph=True, mode='a_abs') #和のCFOの時系列重ね

    # trj.com_rose_curve.dyad_cfo.summation_ave_cfo(graph=True, mode='a_abs')

    ##CFOの差の確認
    # com.dyad_cfo.subtraction_cfo(graph=True)
    # com.triad_cfo.subtraction_cfo(graph=True)
    # com.tetrad_cfo.subtraction_cfo(graph=True)
    # com.subtraction_cfo(graph=True)
    # trj.com_rose_curve.subtraction_cfo_combine(graph=True)

    ##パフォーマンスの確認
    # trj.com_circle.performance_show()
    # trj.com_lemniscate.performance_show()
    # trj.com_rose_curve.performance_show()
    # trj.com_random.performance_show()
    # trj.com_discrete_random.performance_show()

    # trj.com_circle.dyad_cfo.time_series_performance(graph=True, sigma=1000, mode='H-H')
    # trj.com_lemniscate.dyad_cfo.time_series_performance(graph=True, sigma=1000, mode='H-H')
    # trj.com_rose_curve.dyad_cfo.time_series_performance(graph=True, sigma=1000, mode='H-H')
    # trj.com_random.dyad_cfo.time_series_performance(graph=True, sigma=1000, mode='H-H')
    # trj.com_discrete_random.dyad_cfo.time_series_performance(graph=True, sigma=1000, mode='H-H')
    # trj.com_circle.dyad_cfo.time_series_performance(graph=True, sigma=1000, mode='M-M')
    # trj.com_lemniscate.dyad_cfo.time_series_performance(graph=True, sigma=1000, mode='M-M')
    # trj.com_rose_curve.dyad_cfo.time_series_performance(graph=True, sigma=1000, mode='M-M')
    # trj.com_random.dyad_cfo.time_series_performance(graph=True, sigma=1000, mode='M-M')
    # trj.com_discrete_random.dyad_cfo.time_series_performance(graph=True, sigma=1000, mode='M-M')
    # trj.com_circle.dyad_cfo.time_series_performance_cooperation(graph=True, sigma=1000)
    # trj.com_lemniscate.dyad_cfo.time_series_performance_cooperation(graph=True, sigma=1000)
    # trj.com_rose_curve.dyad_cfo.time_series_performance_cooperation(graph=True, sigma=1000)
    # trj.com_random.dyad_cfo.time_series_performance_cooperation(graph=True, sigma=1000)
    # trj.com_discrete_random.dyad_cfo.time_series_performance_cooperation(graph=True, sigma=1000)

    # trj.com_circle.time_series_performance_show(sigma=1000)
    # trj.com_lemniscate.time_series_performance_show(sigma=1000)
    # trj.com_rose_curve.time_series_performance_show(sigma=1000)
    # trj.com_random.time_series_performance_show(sigma=1000)
    # trj.com_discrete_random.time_series_performance_show(sigma=1000)

    # trj.com_circle.time_series_performance_summation_ave_cfo(graph=True, mode='no_abs', sigma=1, dec=100)
    # trj.com_lemniscate.time_series_performance_summation_ave_cfo(graph=True, mode='no_abs', sigma=1, dec=100)
    # trj.com_rose_curve.time_series_performance_summation_ave_cfo(graph=True, mode='no_abs', sigma=1, dec=100)
    # trj.com_random.time_series_performance_summation_ave_cfo(graph=True, mode='no_abs', sigma=1, dec=100)
    # trj.com_discrete_random.time_series_performance_summation_ave_cfo(graph=True, mode='no_abs', sigma=1, dec=100)
    # trj.com_circle.time_series_performance_summation_ave_cfo(graph=True, mode='b_abs', sigma=1, dec=100)
    # trj.com_lemniscate.time_series_performance_summation_ave_cfo(graph=True, mode='b_abs', sigma=1, dec=100)
    # trj.com_rose_curve.time_series_performance_summation_ave_cfo(graph=True, mode='b_abs', sigma=1, dec=100)
    # trj.com_random.time_series_performance_summation_ave_cfo(graph=True, mode='b_abs', sigma=1, dec=100)
    # trj.com_discrete_random.time_series_performance_summation_ave_cfo(graph=True, mode='b_abs', sigma=1, dec=100)
    # trj.com_circle.time_series_performance_summation_ave_cfo(graph=True, mode='a_abs', sigma=1, dec=100)
    # trj.com_lemniscate.time_series_performance_summation_ave_cfo(graph=True, mode='a_abs', sigma=1, dec=100)
    # trj.com_rose_curve.time_series_performance_summation_ave_cfo(graph=True, mode='a_abs', sigma=1, dec=100)
    # trj.com_random.time_series_performance_summation_ave_cfo(graph=True, mode='a_abs', sigma=1, dec=100)
    # trj.com_discrete_random.time_series_performance_summation_ave_cfo(graph=True, mode='a_abs', sigma=1, dec=100)
    #
    # trj.com_circle.time_series_performance_subtraction_ave_cfo(graph=True, sigma=1, dec=100)
    # trj.com_lemniscate.time_series_performance_subtraction_ave_cfo(graph=True, sigma=1, dec=100)
    # trj.com_rose_curve.time_series_performance_subtraction_ave_cfo(graph=True, sigma=1, dec=100)
    # trj.com_random.time_series_performance_subtraction_ave_cfo(graph=True, sigma=1, dec=100)
    # trj.com_discrete_random.time_series_performance_subtraction_ave_cfo(graph=True, sigma=1, dec=100)

    # trj.com_rose_curve.time_series_performance_summation_ave_cfo_axis(graph=True, mode='no_abs', sigma=1, dec=100)
    # trj.com_rose_curve.time_series_performance_summation_ave_cfo_axis(graph=True, mode='b_abs', sigma=1, dec=100)
    # trj.com_rose_curve.time_series_performance_summation_ave_cfo_axis(graph=True, mode='a_abs', sigma=1, dec=100)

    # trj.com_rose_curve.time_series_performance_subtraction_ave_cfo_axis(graph=True, sigma=1, dec=100)

    # trj.com_circle.dyad_cfo.period_performance_New(graph=True, mode='M-M')
    # trj.comparison_performance_human_model()

    # trj.com_rose_curve.robomech2024()
    # trj.com_rose_curve.robomech2024_axis()

    # trj.com_rose_curve.lrm_size()
    # trj.com_rose_curve.lrm_check(mode='size')



    # trj.com_circle.random_forest_all()
    # trj.com_circle.random_forest_size()
    # trj.com_circle.random_forest_each()
    # trj.com_lemniscate.random_forest_all()
    # trj.com_lemniscate.random_forest_size()
    # trj.com_lemniscate.random_forest_each()
    # trj.com_rose_curve.random_forest_all()
    # trj.com_rose_curve.random_forest_size()
    # trj.com_rose_curve.random_forest_each()
    # trj.com_random.random_forest_all()
    # trj.com_random.random_forest_size()
    # trj.com_random.random_forest_each()
    # trj.com_discrete_random.random_forest_all()
    # trj.com_discrete_random.random_forest_size()
    # trj.com_discrete_random.random_forest_each()

    # trj.com_circle.random_forest_check(mode='all')
    # trj.com_lemniscate.random_forest_check(mode='all')
    # trj.com_rose_curve.random_forest_check(mode='all')
    # trj.com_rose_curve.random_forest_check(mode='size')
    # trj.com_random.random_forest_check(mode='all')
    # trj.com_discrete_random.random_forest_check(mode='all')

    # trj.com_circle.simulation_all()
    # trj.com_lemniscate.simulation_all()
    # trj.com_rose_curve.simulation_all()
    # trj.com_random.simulation_all()
    # trj.com_discrete_random.simulation_all()
    # trj.com_rose_curve.simulation_size(size='Dyad')
    # trj.com_rose_curve.simulation_size(size='Triad')
    # trj.com_rose_curve.simulation_size(size='Tetrad')

    # trj.com_rose_curve.simulation_verification_all()

    # trj.com_rose_curve.fnn_size('Dyad')
    # trj.com_rose_curve.fnn_check('Dyad')

    # trj.com_rose_curve.cross_correlation(size='Dyad')

    # plt.rcParams['font.size'] = 6  # フォントの大きさ
    # delay_array = np.linspace(0.0, 0.2, 3)
    # # delay_array = np.linspace(0.0, 0.1, 11)
    # # delay_array = [0.0]
    # size = 'Dyad'
    # model_names = [
    #     # 'LRM',
    #     # 'KernelRidge',
    #     # 'RBF-LRM',
    #     # 'RBF-Ridge',
    #     # 'RBF-Lasso',
    #     # 'RBF-ElasticNet',
    #     # 'Lasso',
    #     # 'RandomForest',
    #     # 'NN',
    #     # 'Polar-LRM',
    #     # 'Polar-KernelRidge',
    #     'Polar-NN',
    # ]
    # test = False
    #
    # for model_name in model_names:
    #     print(f"{model_name=}")
    #     for delay in delay_array:
    #         print(f"{delay=}")
    #         # trj.com_rose_curve.performance_learn_model_size(size=size, model_name=model_name, delay=delay, test=test)
    #         trj.com_rose_curve.performance_learn_model_each(size=size, model_name=model_name, delay=delay, test=test)
    #
    #     # use_model = 'Dyad'
    #     use_model = 'each'
    #     obj = trj.com_rose_curve
    #     figg, ax = plt.subplots(len(obj.dyad_cfo.cfo), len(delay_array), figsize=(len(delay_array) * 10, len(obj.dyad_cfo.cfo) * 5), dpi=100, sharex=True)
    #
    #     # axを常に2次元配列として扱う
    #     if len(obj.dyad_cfo.cfo) == 1 or len(delay_array) == 1:
    #         ax = ax.reshape(len(obj.dyad_cfo.cfo), len(delay_array))
    #
    #     for i, delay in enumerate(delay_array):
    #         print(f"{delay=}")
    #         time, pre, ovs = obj.performance_check_model_any(size=size, use_model=use_model, model_name=model_name, delay=delay, test=test)
    #
    #         for j in range(len(obj.dyad_cfo.cfo)):
    #             ax[j, i].plot(time, pre[j], label='Predicted')
    #             ax[j, i].plot(time, ovs[j], label='Observed')
    #             ax[j, i].set_title('Group ' + str(j+1) + ' delay=' + f"{delay:.1f}")
    #             ax[j, i].set_xlabel('time')
    #             ax[j, i].set_ylabel('CP')
    #             ax[j, i].legend()
    #
    #     if test:
    #         figg.savefig('fig/' + model_name + '/' + obj.trajectory_dir + 'Prediction/Prediction_CooperativeRMSE_TEST_' + size + '_' + use_model + '.png')
    #     else:
    #         figg.savefig('fig/' + model_name + '/' + obj.trajectory_dir + 'Prediction/Prediction_CooperativeRMSE_' + size + '_' + use_model + '.png')

    # trj.com_circle.dyad_cfo.simulate_ball_angle()
    # trj.com_circle.dyad_cfo.relation_pcfo_and_rmse()
    # trj.com_discrete_random.dyad_cfo.relation_pcfo_and_rmse()

    # trj.com_rose_curve.dyad_cfo.relation_orthogonal_fcfo_and_rmse()
    # trj.com_lemniscate.dyad_cfo.check_relation_plate_fcfo_and_crmse()
    # trj.com_rose_curve.dyad_cfo.check_relation_plate_fcfo_and_crmse()
    # trj.com_rose_curve.dyad_cfo.relation_fcfo_and_rmse_limit()
    # trj.com_circle.dyad_cfo.relation_mod_plate_fcfo_rmse()
    # trj.com_lemniscate.dyad_cfo.relation_mod_plate_fcfo_rmse()
    # trj.com_rose_curve.dyad_cfo.relation_mod_plate_fcfo_rmse()
    # trj.com_random.dyad_cfo.relation_mod_plate_fcfo_rmse()
    # trj.com_discrete_random.dyad_cfo.relation_mod_plate_fcfo_rmse()
    # trj.com_rose_curve.dyad_cfo.relation_plate_force_and_rmse()
    # trj.com_rose_curve.dyad_cfo.relation_sum_force_and_plate_position()
    # trj.com_rose_curve.dyad_cfo.relation_force_cfo_and_rmse()
    # trj.com_rose_curve.dyad_cfo.relation_pcfo_and_rmse()
    # trj.com_rose_curve.dyad_cfo.relation_summation_pcfo_and_rmse()
    # trj.com_rose_curve.dyad_cfo.relation_summation_fcfo_and_pcfo()
    # trj.com_rose_curve.dyad_cfo.relation_summation_fcfo_and_rmse()
    # trj.com_rose_curve.dyad_cfo.relation_cfo_force_and_position()
    # trj.com_rose_curve.dyad_cfo.relation_force_and_position()

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

    # com.dyad_cfo.`period_performance_cooperation_and_solo`(graph=True)
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
    # trj.com_rose_curve.

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
    # trj.com_rose_curve.dyad_cfo.summation_performance_combine('a_abs')

    # CFOの偏差とパフォーマンスの確認
    # trj.com_rose_curve.dyad_cfo.deviation_performance_combine()

    ##CFOの和とCFOの差とパフォーマンスの関係
    # com.dyad_cfo.sum_sub_performance()

    ##CFOの和とCFOの差の関係
    # trj.com_rose_curve.dyad_cfo.sum_sub()
    # trj.com_rose_curve.triad_cfo.sum_sub()
    # trj.com_rose_curve.tetrad_cfo.sum_sub()
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
    # com.tetrad_cfo.CFO_relation_axis() # TODO not yet
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
    # trj.com_rose_curve.summation_cfo_combine(graph=True, mode='b_abs')
    # trj.com_rose_curve.summation_cfo_combine(graph=True, mode='a_abs')
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
    # trj.com_rose_curve.subtraction_cfo_combine(graph=True)
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

