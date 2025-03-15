# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt
import numpy as np

import Npz
import Combine_analysis
import CFO_analysis
import Compare_assist

# plt.switch_backend('Qt5Agg')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    npz = Npz.NPZ()

    dyad_wa_npz_filename = [
        # '2024-08-21_AdABC_Dyad_t.onogawa_r.tanaka_Random_WithAssist_1.npz',
        # '2024-08-21_AdABC_Dyad_t.onogawa_r.tanaka_RoseCurve_WithAssist_1.npz',
        # '2024-08-22_AdABC_Dyad_k.kobayashi_b.poitrimol_RoseCurve_WithAssist_1.npz',
        # '2024-08-22_AdABC_Dyad_b.poitrimol_k.kobayashi_RoseCurve_WithAssist_3.npz',
        # '2024-08-22_AdABC_Dyad_b.poitrimol_k.kobayashi_RoseCurve_WithAssist_4.npz',
        # '2024-08-22_AdABC_Dyad_t.onogawa_k.kobayashi_RoseCurve_WithAssist_1.npz',
        # '2024-08-22_AdABC_Dyad_t.onogawa_k.kobayashi_RoseCurve_WithAssist_2.npz',
        '2024-08-22_AdABC_Dyad_t.onogawa_k.kobayashi_RoseCurve_WithAssist_3.npz',
        # '2024-08-22_AdABC_Dyad_t.onogawa_k.kobayashi_RoseCurve_WithAssist_4.npz',
        '2024-08-22_AdABC_Dyad_b.poitrimol_l.nicolas_RoseCurve_WithAssist_1.npz',
        # '2024-08-22_AdABC_Dyad_b.poitrimol_l.nicolas_RoseCurve_WithAssist_2.npz',
    ]

    dyad_woa_npz_filename = [
        # '2024-08-21_AdABC_Dyad_t.onogawa_r.tanaka_Random_NoAssist_1.npz',
        # '2024-08-21_AdABC_Dyad_t.onogawa_r.tanaka_RoseCurve_NoAssist_1.npz',
        # '2024-08-22_AdABC_Dyad_t.onogawa_k.kobayashi_RoseCurve_WithAssist_1.npz',
        # '2024-08-22_AdABC_Dyad_t.onogawa_k.kobayashi_RoseCurve_WithAssist_2.npz',
        # '2024-08-21_AdABC_Dyad_t.onogawa_r.tanaka_Random_WithAssist_1.npz',
        '2024-08-21_AdABC_Dyad_t.onogawa_r.tanaka_RoseCurve_WithAssist_1.npz',
        '2024-08-22_AdABC_Dyad_k.kobayashi_b.poitrimol_RoseCurve_WithAssist_1.npz',
    ]

# /nfs/ssk-storage/data/cooperation/proto/assist/dyad/2024-08-22_AdABC_Dyad_t.onogawa_k.kobayashi_RoseCurve_WithAssist_1.bin
# /nfs/ssk-storage/data/cooperation/proto/assist/dyad/2024-08-22_AdABC_Dyad_t.onogawa_k.kobayashi_RoseCurve_WithAssist_2.bin
# /nfs/ssk-storage/data/cooperation/proto/assist/dyad/2024-08-22_AdABC_Dyad_t.onogawa_k.kobayashi_RoseCurve_WithAssist_3.bin
# /nfs/ssk-storage/data/cooperation/proto/assist/dyad/2024-08-22_AdABC_Dyad_t.onogawa_k.kobayashi_RoseCurve_WithAssist_4.bin
# /nfs/ssk-storage/data/cooperation/proto/assist/dyad/2024-08-22_AdABC_Dyad_b.poitrimol_l.nicolas_RoseCurve_WithAssist_1.bin
# /nfs/ssk-storage/data/cooperation/proto/assist/dyad/2024-08-22_AdABC_Dyad_b.poitrimol_l.nicolas_RoseCurve_WithAssist_2.bin
# /nfs/ssk-storage/data/cooperation/proto/assist/dyad/2024-08-22_AdABC_Dyad_k.kobayashi_b.poitrimol_RoseCurve_WithAssist_1.bin
# /nfs/ssk-storage/data/cooperation/proto/assist/dyad/2024-08-22_AdABC_Dyad_b.poitrimol_k.kobayashi_RoseCurve_WithAssist_3.bin
# /nfs/ssk-storage/data/cooperation/proto/assist/dyad/2024-08-22_AdABC_Dyad_b.poitrimol_k.kobayashi_RoseCurve_WithAssist_4.bin
# /nfs/ssk-storage/data/cooperation/proto/assist/dyad/2024-08-21_AdABC_Dyad_t.onogawa_r.tanaka_Random_NoAssist_1.bin
# /nfs/ssk-storage/data/cooperation/proto/assist/dyad/2024-08-21_AdABC_Dyad_t.onogawa_r.tanaka_RoseCurve_WithAssist_1.bin
# /nfs/ssk-storage/data/cooperation/proto/assist/dyad/2024-08-21_AdABC_Dyad_t.onogawa_r.tanaka_RoseCurve_NoAssist_1.bin
# /nfs/ssk-storage/data/cooperation/proto/assist/dyad/2024-08-21_AdABC_Dyad_t.onogawa_r.tanaka_Random_WithAssist_1.bin

    # triad_npz_filename = [
    #     '2024-02-29_AdABC_Triad_t.onogawa_r.yanase_g.otsuka_Circle_1_CFO.npz',
    #     '2024-02-29_AdABC_Triad_t.onogawa_r.yanase_g.otsuka_Discrete_Random_1_CFO.npz',
    #     '2024-02-29_AdABC_Triad_t.onogawa_r.yanase_g.otsuka_Lemniscate_1_CFO.npz',
    #     '2024-02-29_AdABC_Triad_t.onogawa_r.yanase_g.otsuka_Random_1_CFO.npz',
    #     '2024-02-29_AdABC_Triad_t.onogawa_r.yanase_g.otsuka_RoseCurve_1_CFO.npz',
    #
    #     '2024-03-02_AdABC_Triad_k.tozuka_i.kato_s.tsuchiya_Circle_1_CFO.npz',
    #     '2024-03-02_AdABC_Triad_k.tozuka_i.kato_s.tsuchiya_Discrete_Random_1_CFO.npz',
    #     '2024-03-02_AdABC_Triad_k.tozuka_i.kato_s.tsuchiya_Lemniscate_1_CFO.npz',
    #     '2024-03-02_AdABC_Triad_k.tozuka_i.kato_s.tsuchiya_Random_1_CFO.npz',
    #     '2024-03-02_AdABC_Triad_k.tozuka_i.kato_s.tsuchiya_RoseCurve_1_CFO.npz',
    # ]
    #
    # tetrad_npz_filename = [
    #     '2024-03-01_AdABC_Tetrad_t.onogawa_r.yanase_g.otsuka_n.ito_Circle_1_CFO.npz',
    #     '2024-03-01_AdABC_Tetrad_t.onogawa_r.yanase_g.otsuka_n.ito_Discrete_Random_1_CFO.npz',
    #     '2024-03-02_AdABC_Tetrad_t.onogawa_r.yanase_g.otsuka_n.ito_Lemniscate_1_CFO.npz',
    #     '2024-03-02_AdABC_Tetrad_t.onogawa_r.yanase_g.otsuka_n.ito_Random_1_CFO.npz',
    #     '2024-03-02_AdABC_Tetrad_t.onogawa_r.yanase_g.otsuka_n.ito_RoseCurve_1_CFO.npz',
    #
    #     '2024-03-13_AdABC_Tetrad_k.tozuka_k.kobayashi_r.tanaka_n.ito_Circle_1_CFO.npz',
    #     '2024-03-13_AdABC_Tetrad_k.tozuka_k.kobayashi_r.tanaka_n.ito_Discrete_Random_1_CFO.npz',
    #     '2024-03-13_AdABC_Tetrad_k.tozuka_k.kobayashi_r.tanaka_n.ito_Lemniscate_1_CFO.npz',
    #     '2024-03-13_AdABC_Tetrad_k.tozuka_k.kobayashi_r.tanaka_n.ito_Random_1_CFO.npz',
    #     '2024-03-13_AdABC_Tetrad_k.tozuka_k.kobayashi_r.tanaka_n.ito_RoseCurve_1_CFO.npz',
    # ]


    dyad_wa_npz = npz.select_load("/nfs/ssk-storage/data/cooperation/npz/dyad/assist/", dyad_wa_npz_filename)
    dyad_woa_npz = npz.select_load("/nfs/ssk-storage/data/cooperation/npz/dyad/assist/", dyad_woa_npz_filename)
    # triad_ind_npz = npz.select_load("/nfs/ssk-storage/data/cfo/triad/", triad_npz_filename)
    # tetrad_ind_npz = npz.select_load("/nfs/ssk-storage/data/cfo/tetrad/", tetrad_npz_filename)

    # dyad_cfo: CFO_analysis.CFO = CFO_analysis.CFO(dyad_npz, 'dyad', 'RoseCurve')

    com = Compare_assist.compare(dyad_woa_npz, dyad_wa_npz, 'dyad', 'RoseCurve')

    # dyad_cfo.plot_performance()
    # dyad_cfo.plot_performance_improve()
    # dyad_cfo.summation_cfo(graph=True, mode='b_abs')
    # dyad_cfo.period_performance(graph=True, mode='H-H')
    # dyad_cfo.plot_assistance_tau()
    # dyad_cfo.plot_cfosum()

    com.plot_performance_improve()