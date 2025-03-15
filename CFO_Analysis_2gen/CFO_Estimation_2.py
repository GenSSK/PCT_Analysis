import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
import time

def process(filename):
    subprocess.call(["/home/genki/clion/CFO_Estimator/cmake-build-release/CFO_Estimator",
                     filename[0],
                     filename[1],
                     filename[2],
                     ])

    subprocess.call(["/home/genki/clion/CFO_Estimator/cmake-build-release/CFO_Estimator_shared",
                     filename[0],
                     filename[1],
                     filename[2]
                     ])


if __name__ == '__main__':

    files = [
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'dyad', '2024-02-09_AdABC_Dyad_k.tozuka_k.kobayashi_Circle_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'dyad', '2024-02-09_AdABC_Dyad_k.tozuka_k.kobayashi_Discrete_Random_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'dyad', '2024-02-09_AdABC_Dyad_k.tozuka_k.kobayashi_Lemniscate_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'dyad', '2024-02-09_AdABC_Dyad_k.tozuka_k.kobayashi_Random_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'dyad', '2024-02-09_AdABC_Dyad_k.tozuka_k.kobayashi_RoseCurve_1'],

        ['/nfs/ssk-storage/data/cooperation/proto/', 'dyad', '2024-03-02_AdABC_Dyad_h.nakamura_k.tozuka_Circle_1'],
        ['/nfs/ssk-storage/data/cooperation/proto/', 'dyad', '2024-03-02_AdABC_Dyad_h.nakamura_k.tozuka_Discrete_Random_1'],
        ['/nfs/ssk-storage/data/cooperation/proto/', 'dyad', '2024-03-02_AdABC_Dyad_h.nakamura_k.tozuka_Lemniscate_1'],
        ['/nfs/ssk-storage/data/cooperation/proto/', 'dyad', '2024-03-02_AdABC_Dyad_h.nakamura_k.tozuka_Random_1'],
        ['/nfs/ssk-storage/data/cooperation/proto/', 'dyad', '2024-03-02_AdABC_Dyad_h.nakamura_k.tozuka_RoseCurve_1'],
        ['/nfs/ssk-storage/data/cooperation/proto/', 'dyad', '2024-03-02_AdABC_Dyad_s.tsuchiya_h.nakamura_Circle_1'],
        ['/nfs/ssk-storage/data/cooperation/proto/', 'dyad', '2024-03-02_AdABC_Dyad_s.tsuchiya_h.nakamura_Discrete_Random_1'],
        ['/nfs/ssk-storage/data/cooperation/proto/', 'dyad', '2024-03-02_AdABC_Dyad_s.tsuchiya_h.nakamura_Lemniscate_1'],
        ['/nfs/ssk-storage/data/cooperation/proto/', 'dyad', '2024-03-02_AdABC_Dyad_s.tsuchiya_h.nakamura_Random_1'],
        #
        # # ['/nfs/ssk-storage/data/cooperation/proto/avatar/', 'dyad', '2024-04-09_AdABC_Dyad_All_b.poitrimol_r.hiratsuka_RoseCurve_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/avatar/', 'dyad', '2024-04-09_AdABC_Dyad_All_s.tsuchiya_h.kitaiwa_RoseCurve_1'],
        # # ['/nfs/ssk-storage/data/cooperation/proto/avatar/', 'dyad', '2024-04-10_AdABC_Dyad_All_a.noto_y.yamada_RoseCurve_1'],
        # # ['/nfs/ssk-storage/data/cooperation/proto/avatar/', 'dyad', '2024-04-10_AdABC_Dyad_All_c.barbaut_k.kobayashi_RoseCurve_1'],
        # # ['/nfs/ssk-storage/data/cooperation/proto/avatar/', 'dyad', '2024-04-10_AdABC_Dyad_All_d.nakajima_m.kashiwagi_RoseCurve_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/avatar/', 'dyad', '2024-04-10_AdABC_Dyad_All_i.kato_n.ito_RoseCurve_1'],
        # # ['/nfs/ssk-storage/data/cooperation/proto/avatar/', 'dyad', '2024-04-10_AdABC_Dyad_All_ken.hayashi_r.morishita_RoseCurve_1'],
        # # ['/nfs/ssk-storage/data/cooperation/proto/avatar/', 'dyad', '2024-04-10_AdABC_Dyad_All_k.ohya_s.imamura_RoseCurve_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/avatar/', 'dyad', '2024-04-10_AdABC_Dyad_All_l.nicolas_h.nakamura_RoseCurve_1'],
        # # ['/nfs/ssk-storage/data/cooperation/proto/avatar/', 'dyad', '2024-04-10_AdABC_Dyad_All_m.lopez_k.tozuka_RoseCurve_1'],
        # # ['/nfs/ssk-storage/data/cooperation/proto/avatar/', 'dyad', '2024-04-10_AdABC_Dyad_All_r.kato_y.takahashi_RoseCurve_1'],
        # # ['/nfs/ssk-storage/data/cooperation/proto/avatar/', 'dyad', '2024-04-10_AdABC_Dyad_All_r.tanaka_m.nakamura_RoseCurve_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/avatar/', 'dyad', '2024-04-10_AdABC_Dyad_All_r.yanase_g.otsuka_RoseCurve_1'],
        # # ['/nfs/ssk-storage/data/cooperation/proto/avatar/', 'dyad', '2024-04-10_AdABC_Dyad_All_t.onogawa_s.tahara_RoseCurve_1'],



        # ['/nfs/ssk-storage/data/cooperation/proto/', 'dyad', '2024-03-02_AdABC_Dyad_h.nakamura_k.tozuka_Circle_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'dyad', '2024-03-02_AdABC_Dyad_h.nakamura_k.tozuka_Discrete_Random_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'dyad', '2024-03-02_AdABC_Dyad_h.nakamura_k.tozuka_Lemniscate_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'dyad', '2024-03-02_AdABC_Dyad_h.nakamura_k.tozuka_Random_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'dyad', '2024-03-02_AdABC_Dyad_h.nakamura_k.tozuka_RoseCurve_1'],
        #
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'dyad', '2024-03-02_AdABC_Dyad_s.tsuchiya_h.nakamura_Circle_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'dyad', '2024-03-02_AdABC_Dyad_s.tsuchiya_h.nakamura_Discrete_Random_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'dyad', '2024-03-02_AdABC_Dyad_s.tsuchiya_h.nakamura_Lemniscate_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'dyad', '2024-03-02_AdABC_Dyad_s.tsuchiya_h.nakamura_Random_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'dyad', '2024-03-02_AdABC_Dyad_s.tsuchiya_h.nakamura_RoseCurve_1'],
        #
        #
        #
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'triad', '2024-02-29_AdABC_Triad_t.onogawa_r.yanase_g.otsuka_Circle_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'triad', '2024-02-29_AdABC_Triad_t.onogawa_r.yanase_g.otsuka_Discrete_Random_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'triad', '2024-02-29_AdABC_Triad_t.onogawa_r.yanase_g.otsuka_Lemniscate_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'triad', '2024-02-29_AdABC_Triad_t.onogawa_r.yanase_g.otsuka_Random_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'triad', '2024-02-29_AdABC_Triad_t.onogawa_r.yanase_g.otsuka_RoseCurve_1'],
        #
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'triad', '2024-03-02_AdABC_Triad_k.tozuka_i.kato_s.tsuchiya_Circle_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'triad', '2024-03-02_AdABC_Triad_k.tozuka_i.kato_s.tsuchiya_Discrete_Random_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'triad', '2024-03-02_AdABC_Triad_k.tozuka_i.kato_s.tsuchiya_Lemniscate_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'triad', '2024-03-02_AdABC_Triad_k.tozuka_i.kato_s.tsuchiya_Random_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'triad', '2024-03-02_AdABC_Triad_k.tozuka_i.kato_s.tsuchiya_RoseCurve_1'],
        #
        #
        #
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'tetrad', '2024-03-01_AdABC_Tetrad_t.onogawa_r.yanase_g.otsuka_n.ito_Circle_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'tetrad', '2024-03-01_AdABC_Tetrad_t.onogawa_r.yanase_g.otsuka_n.ito_Discrete_Random_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'tetrad', '2024-03-02_AdABC_Tetrad_t.onogawa_r.yanase_g.otsuka_n.ito_Lemniscate_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'tetrad', '2024-03-02_AdABC_Tetrad_t.onogawa_r.yanase_g.otsuka_n.ito_Random_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'tetrad', '2024-03-02_AdABC_Tetrad_t.onogawa_r.yanase_g.otsuka_n.ito_RoseCurve_1'],
        #
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'tetrad', '2024-03-13_AdABC_Tetrad_k.tozuka_k.kobayashi_r.tanaka_n.ito_Circle_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'tetrad', '2024-03-13_AdABC_Tetrad_k.tozuka_k.kobayashi_r.tanaka_n.ito_Discrete_Random_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'tetrad', '2024-03-13_AdABC_Tetrad_k.tozuka_k.kobayashi_r.tanaka_n.ito_Lemniscate_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'tetrad', '2024-03-13_AdABC_Tetrad_k.tozuka_k.kobayashi_r.tanaka_n.ito_Random_1'],
        # ['/nfs/ssk-storage/data/cooperation/proto/', 'tetrad', '2024-03-13_AdABC_Tetrad_k.tozuka_k.kobayashi_r.tanaka_n.ito_RoseCurve_1'],

    ]


    tpe = ThreadPoolExecutor(max_workers=10)

    for file in files:
        tpe.submit(process, file)

    tpe.shutdown()
    print('done')