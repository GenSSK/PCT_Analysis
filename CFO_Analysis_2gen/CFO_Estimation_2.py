import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
import time

def process(filename):
    subprocess.call(["/home/genki/clion/CFO_Estimator/cmake-build-release/CFO_Estimator",
                     filename[0],
                     filename[1]
                     ])

    subprocess.call(["/home/genki/clion/CFO_Estimator/cmake-build-release/CFO_Estimator_shared",
                     filename[0],
                     filename[1]
                     ])


if __name__ == '__main__':

    files = [

        ['dyad', '2024-03-02_AdABC_Dyad_h.nakamura_k.tozuka_Circle_1'],
        ['dyad', '2024-03-02_AdABC_Dyad_h.nakamura_k.tozuka_Discrete_Random_1'],
        ['dyad', '2024-03-02_AdABC_Dyad_h.nakamura_k.tozuka_Lemniscate_1'],
        ['dyad', '2024-03-02_AdABC_Dyad_h.nakamura_k.tozuka_Random_1'],
        ['dyad', '2024-03-02_AdABC_Dyad_h.nakamura_k.tozuka_RoseCurve_1'],

        ['dyad', '2024-03-02_AdABC_Dyad_s.tsuchiya_h.nakamura_Circle_1'],
        ['dyad', '2024-03-02_AdABC_Dyad_s.tsuchiya_h.nakamura_Discrete_Random_1'],
        ['dyad', '2024-03-02_AdABC_Dyad_s.tsuchiya_h.nakamura_Lemniscate_1'],
        ['dyad', '2024-03-02_AdABC_Dyad_s.tsuchiya_h.nakamura_Random_1'],
        ['dyad', '2024-03-02_AdABC_Dyad_s.tsuchiya_h.nakamura_RoseCurve_1'],



        ['triad', '2024-02-29_AdABC_Triad_t.onogawa_r.yanase_g.otsuka_Circle_1'],
        ['triad', '2024-02-29_AdABC_Triad_t.onogawa_r.yanase_g.otsuka_Discrete_Random_1'],
        ['triad', '2024-02-29_AdABC_Triad_t.onogawa_r.yanase_g.otsuka_Lemniscate_1'],
        ['triad', '2024-02-29_AdABC_Triad_t.onogawa_r.yanase_g.otsuka_Random_1'],
        ['triad', '2024-02-29_AdABC_Triad_t.onogawa_r.yanase_g.otsuka_RoseCurve_1'],

        ['triad', '2024-03-02_AdABC_Triad_k.tozuka_i.kato_s.tsuchiya_Circle_1'],
        ['triad', '2024-03-02_AdABC_Triad_k.tozuka_i.kato_s.tsuchiya_Discrete_Random_1'],
        ['triad', '2024-03-02_AdABC_Triad_k.tozuka_i.kato_s.tsuchiya_Lemniscate_1'],
        ['triad', '2024-03-02_AdABC_Triad_k.tozuka_i.kato_s.tsuchiya_Random_1'],
        ['triad', '2024-03-02_AdABC_Triad_k.tozuka_i.kato_s.tsuchiya_RoseCurve_1'],



        ['tetrad', '2024-03-01_AdABC_Tetrad_t.onogawa_r.yanase_g.otsuka_n.ito_Circle_1'],
        ['tetrad', '2024-03-01_AdABC_Tetrad_t.onogawa_r.yanase_g.otsuka_n.ito_Discrete_Random_1'],
        ['tetrad', '2024-03-02_AdABC_Tetrad_t.onogawa_r.yanase_g.otsuka_n.ito_Lemniscate_1'],
        ['tetrad', '2024-03-02_AdABC_Tetrad_t.onogawa_r.yanase_g.otsuka_n.ito_Random_1'],
        ['tetrad', '2024-03-02_AdABC_Tetrad_t.onogawa_r.yanase_g.otsuka_n.ito_RoseCurve_1'],

        ['tetrad', '2024-03-13_AdABC_Tetrad_k.tozuka_k.kobayashi_r.tanaka_n.ito_Circle_1'],
        ['tetrad', '2024-03-13_AdABC_Tetrad_k.tozuka_k.kobayashi_r.tanaka_n.ito_Discrete_Random_1'],
        ['tetrad', '2024-03-13_AdABC_Tetrad_k.tozuka_k.kobayashi_r.tanaka_n.ito_Lemniscate_1'],
        ['tetrad', '2024-03-13_AdABC_Tetrad_k.tozuka_k.kobayashi_r.tanaka_n.ito_Random_1'],
        ['tetrad', '2024-03-13_AdABC_Tetrad_k.tozuka_k.kobayashi_r.tanaka_n.ito_RoseCurve_1'],

    ]


    tpe = ThreadPoolExecutor(max_workers=10)

    for file in files:
        tpe.submit(process, file)

    tpe.shutdown()
    print('done')