# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt
import numpy as np

import Npz
import analysis_group as ag
# plt.switch_backend('Qt5Agg')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    npz = Npz.NPZ()

    dyad_ALL = [
        "2024-04-09_AdABC_Dyad_All_b.poitrimol_r.hiratsuka_RoseCurve_1.npz",
        "2024-04-09_AdABC_Dyad_All_s.tsuchiya_h.kitaiwa_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_All_t.onogawa_s.tahara_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_All_r.yanase_g.otsuka_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_All_c.barbaut_k.kobayashi_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_All_m.lopez_k.tozuka_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_All_r.tanaka_m.nakamura_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_All_i.kato_n.ito_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_All_l.nicolas_h.nakamura_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_All_a.noto_y.yamada_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_All_ken.hayashi_r.morishita_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_All_r.kato_y.takahashi_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_All_k.ohya_s.imamura_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_All_d.nakajima_m.kashiwagi_RoseCurve_1.npz",
    ]

    dyad_ALONE = [
        "2024-04-10_AdABC_Dyad_Alone_b.poitrimol_r.hiratsuka_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Alone_s.tsuchiya_h.kitaiwa_RoseCurve_1.npz",
        "2024-04-09_AdABC_Dyad_Alone_t.onogawa_s.tahara_RoseCurve_1.npz",
        "2024-04-09_AdABC_Dyad_Alone_r.yanase_g.otsuka_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Alone_c.barbaut_k.kobayashi_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Alone_m.lopez_k.tozuka_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Alone_r.tanaka_m.nakamura_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Alone_i.kato_n.ito_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Alone_l.nicolas_h.nakamura_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Alone_a.noto_y.yamada_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Alone_ken.hayashi_r.morishita_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Alone_r.kato_y.takahashi_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Alone_k.ohya_s.imamura_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Alone_d.nakajima_m.kashiwagi_RoseCurve_1.npz",
    ]

    dyad_OTHER = [
        "2024-04-10_AdABC_Dyad_Other_b.poitrimol_r.hiratsuka_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Other_s.tsuchiya_h.kitaiwa_RoseCurve_1.npz",
        "2024-04-09_AdABC_Dyad_Other_t.onogawa_s.tahara_RoseCurve_1.npz",
        "2024-04-09_AdABC_Dyad_Other_r.yanase_g.otsuka_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Other_c.barbaut_k.kobayashi_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Other_m.lopez_k.tozuka_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Other_r.tanaka_m.nakamura_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Other_i.kato_n.ito_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Other_l.nicolas_h.nakamura_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Other_a.noto_y.yamada_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Other_ken.hayashi_r.morishita_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Other_r.kato_y.takahashi_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Other_k.ohya_s.imamura_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Other_d.nakajima_m.kashiwagi_RoseCurve_1.npz",
    ]

    dyad_NONE = [
        "2024-04-10_AdABC_Dyad_None_b.poitrimol_r.hiratsuka_RoseCurve_1.npz",
        "2024-04-09_AdABC_Dyad_None_s.tsuchiya_h.kitaiwa_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_None_t.onogawa_s.tahara_RoseCurve_1.npz",
        "2024-04-09_AdABC_Dyad_None_r.yanase_g.otsuka_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_None_c.barbaut_k.kobayashi_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_None_m.lopez_k.tozuka_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_None_r.tanaka_m.nakamura_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_None_i.kato_n.ito_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_None_l.nicolas_h.nakamura_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_None_a.noto_y.yamada_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_None_ken.hayashi_r.morishita_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_None_r.kato_y.takahashi_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_None_k.ohya_s.imamura_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_None_d.nakajima_m.kashiwagi_RoseCurve_1.npz",
    ]

    dyad_1 = [
        "2024-04-09_AdABC_Dyad_All_b.poitrimol_r.hiratsuka_RoseCurve_1.npz",
        "2024-04-09_AdABC_Dyad_All_s.tsuchiya_h.kitaiwa_RoseCurve_1.npz",
        "2024-04-09_AdABC_Dyad_Other_t.onogawa_s.tahara_RoseCurve_1.npz",
        "2024-04-09_AdABC_Dyad_Other_r.yanase_g.otsuka_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_All_c.barbaut_k.kobayashi_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_All_m.lopez_k.tozuka_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Alone_r.tanaka_m.nakamura_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Alone_i.kato_n.ito_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Other_l.nicolas_h.nakamura_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Other_a.noto_y.yamada_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Alone_ken.hayashi_r.morishita_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_None_r.kato_y.takahashi_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_None_k.ohya_s.imamura_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_None_d.nakajima_m.kashiwagi_RoseCurve_1.npz",
    ]

    dyad_2 = [
        "2024-04-10_AdABC_Dyad_Other_b.poitrimol_r.hiratsuka_RoseCurve_1.npz",
        "2024-04-09_AdABC_Dyad_None_s.tsuchiya_h.kitaiwa_RoseCurve_1.npz",
        "2024-04-09_AdABC_Dyad_Alone_t.onogawa_s.tahara_RoseCurve_1.npz",
        "2024-04-09_AdABC_Dyad_Alone_r.yanase_g.otsuka_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Other_c.barbaut_k.kobayashi_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Alone_m.lopez_k.tozuka_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Other_r.tanaka_m.nakamura_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_All_i.kato_n.ito_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_None_l.nicolas_h.nakamura_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Alone_a.noto_y.yamada_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_All_ken.hayashi_r.morishita_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_All_r.kato_y.takahashi_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_All_k.ohya_s.imamura_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Other_d.nakajima_m.kashiwagi_RoseCurve_1.npz",
    ]

    dyad_3 = [
        "2024-04-10_AdABC_Dyad_Alone_b.poitrimol_r.hiratsuka_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Alone_s.tsuchiya_h.kitaiwa_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_All_t.onogawa_s.tahara_RoseCurve_1.npz",
        "2024-04-09_AdABC_Dyad_None_r.yanase_g.otsuka_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_None_c.barbaut_k.kobayashi_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Other_m.lopez_k.tozuka_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_None_r.tanaka_m.nakamura_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Other_i.kato_n.ito_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_All_l.nicolas_h.nakamura_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_None_l.nicolas_h.nakamura_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Other_ken.hayashi_r.morishita_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Alone_r.kato_y.takahashi_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Other_k.ohya_s.imamura_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_All_d.nakajima_m.kashiwagi_RoseCurve_1.npz",
    ]

    dyad_4 = [
        "2024-04-10_AdABC_Dyad_None_b.poitrimol_r.hiratsuka_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Other_s.tsuchiya_h.kitaiwa_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_None_t.onogawa_s.tahara_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_All_r.yanase_g.otsuka_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Alone_c.barbaut_k.kobayashi_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_None_m.lopez_k.tozuka_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_All_r.tanaka_m.nakamura_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_None_i.kato_n.ito_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Alone_l.nicolas_h.nakamura_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_All_a.noto_y.yamada_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Other_ken.hayashi_r.morishita_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Other_r.kato_y.takahashi_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Alone_k.ohya_s.imamura_RoseCurve_1.npz",
        "2024-04-10_AdABC_Dyad_Alone_d.nakajima_m.kashiwagi_RoseCurve_1.npz",
    ]

    dyad_all_npz = npz.select_load("/nfs/ssk-storage/data/cooperation/npz/dyad/avatar/", dyad_ALL)
    dyad_alone_npz = npz.select_load("/nfs/ssk-storage/data/cooperation/npz/dyad/avatar/", dyad_ALONE)
    dyad_other_npz = npz.select_load("/nfs/ssk-storage/data/cooperation/npz/dyad/avatar/", dyad_OTHER)
    dyad_none_npz = npz.select_load("/nfs/ssk-storage/data/cooperation/npz/dyad/avatar/", dyad_NONE)

    dyad_1_npz = npz.select_load("/nfs/ssk-storage/data/cooperation/npz/dyad/avatar/", dyad_1)
    dyad_2_npz = npz.select_load("/nfs/ssk-storage/data/cooperation/npz/dyad/avatar/", dyad_2)
    dyad_3_npz = npz.select_load("/nfs/ssk-storage/data/cooperation/npz/dyad/avatar/", dyad_3)
    dyad_4_npz = npz.select_load("/nfs/ssk-storage/data/cooperation/npz/dyad/avatar/", dyad_4)

    dyad = ag.group(dyad_all_npz, dyad_alone_npz, dyad_other_npz, dyad_none_npz, 'dyad', 'RoseCurve')

    dyad.csv_effort_and_force()

    # dyad.plot_force()
    # dyad.plot_compare_effort()

    # dyad.plot_compare_performance()

    # dyad.plot_ts_force()
    # dyad.plot_ts_force_rms()
    # dyad.plot_ts_force_diff()

    # dyad.plot_performance_improve()
    # dyad.plot_ine_improve()

    # dyad.plot_force_diff()