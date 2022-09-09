import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
import time

def process(filename):
    subprocess.call(["/home/genki/clion/CFO_Estimator/cmake-build-release/CFO_Estimator",
                     filename,
                     ])


if __name__ == '__main__':

    dyad = [
        # 'dyad/2022-07-28_y.inoue_b.poitrimol_1234',
        # 'dyad/2022-07-28_y.inoue_b.poitrimol_1324',
        # 'dyad/2022-07-28_y.inoue_b.poitrimol_1432',
        # 'dyad/2022-07-29_i.tsunokuni_k.tozuka_2134',
        # 'dyad/2022-07-29_i.tsunokuni_k.tozuka_3124',
        # 'dyad/2022-07-29_i.tsunokuni_k.tozuka_4123',
        # 'dyad/2022-07-29_i.tsunokuni_y.inoue_1234',
        # 'dyad/2022-07-29_i.tsunokuni_y.inoue_1324',
        # 'dyad/2022-07-29_i.tsunokuni_y.inoue_1423',
        # 'dyad/2022-07-31_m.sugaya_n.ito_1234',
        # 'dyad/2022-07-31_m.sugaya_n.ito_1324',
        # 'dyad/2022-07-31_m.sugaya_n.ito_1423',
        # 'dyad/2022-08-23_r.yanase_ko.kobayashi_1234',
        'dyad/2022-08-23_r.yanase_ko.kobayashi_1324',
        # 'dyad/2022-08-23_r.yanase_ko.kobayashi_1423',
        # 'dyad/2022-08-26_y.kobayashi_r.yanase_1234',
        # 'dyad/2022-08-26_y.kobayashi_r.yanase_1324',
        # 'dyad/2022-08-26_y.kobayashi_r.yanase_1423',
        # 'dyad/2022-09-05_s.watanabe_ko.kobayashi_1234',
        # 'dyad/2022-09-05_s.watanabe_ko.kobayashi_1324',
        # 'dyad/2022-09-05_s.watanabe_ko.kobayashi_1423',
        # 'dyad/2022-09-07_k.kobayashi_r.yanase_1234',
        # 'dyad/2022-09-07_k.kobayashi_r.yanase_1324',
        # 'dyad/2022-09-07_k.kobayashi_r.yanase_1423',
        # 'dyad/2022-09-07_k.tozuka_ko.kobayashi_1234',
        # 'dyad/2022-09-07_k.tozuka_ko.kobayashi_1324',
        # 'dyad/2022-09-07_k.tozuka_ko.kobayashi_1423',
        # 'dyad/2022-09-07_y.inoue_k.kobayashi_1234',
        # 'dyad/2022-09-07_y.inoue_k.kobayashi_1324',
        # 'dyad/2022-09-07_y.inoue_k.kobayashi_1423',
    ]

    triad = [
        # 'triad/2022-07-31_h.igarashi_ko.kobayashi_t.kassai_1234',
        # 'triad/2022-07-31_k.tozuka_y.inoue_m.nakamura_1234',
        # 'triad/2022-07-31_k.tozuka_y.inoue_m.nakamura_2134',
        # 'triad/2022-07-31_k.tozuka_y.inoue_m.nakamura_3412',
        # 'triad/2022-07-31_m.sugaya_s.sanuka_m.nakamura_1234',
        # 'triad/2022-07-31_m.sugaya_s.sanuka_m.nakamura_2134',
        # 'triad/2022-07-31_m.sugaya_s.sanuka_m.nakamura_4213',
        # 'triad/2022-08-23_s.watanabe_t.kassai_h.nishimura_1234',
        # 'triad/2022-08-23_s.watanabe_t.kassai_h.nishimura_2134',
        # 'triad/2022-08-23_s.watanabe_t.kassai_h.nishimura_3214',
        # 'triad/2022-08-26_k.kobayashi_r.yanase_ko.kobayashi_1234',
        # 'triad/2022-08-26_k.kobayashi_r.yanase_ko.kobayashi_2314',
        # 'triad/2022-08-26_k.kobayashi_r.yanase_ko.kobayashi_3412',
        # 'triad/2022-09-05_b.poitrimol_y.inoue_k.tozuka_1234',
        # 'triad/2022-09-05_b.poitrimol_y.inoue_k.tozuka_2314',
        # 'triad/2022-09-07_b.poitrimol_k.tozuka_y.kobayashi_1234',
        # 'triad/2022-09-07_b.poitrimol_k.tozuka_y.kobayashi_2431',
        # 'triad/2022-09-07_b.poitrimol_k.tozuka_y.kobayashi_3421',
        # 'triad/2022-09-07_b.poitrimol_y.inoue_r.yanase_1234',
        # 'triad/2022-09-07_b.poitrimol_y.inoue_r.yanase_3241',
        # 'triad/2022-09-07_b.poitrimol_y.inoue_r.yanase_4132',
        # 'triad/2022-09-07_ko.kobayashi_k.kobayashi_y.kobayashi_1234',
        # 'triad/2022-09-07_ko.kobayashi_k.kobayashi_y.kobayashi_3412',
        # 'triad/2022-09-07_ko.kobayashi_k.kobayashi_y.kobayashi_4132',
        # 'triad/2022-09-07_r.yanase_k.kobayashi_s.kamioka_1234',
        # 'triad/2022-09-07_r.yanase_k.kobayashi_s.kamioka_2431',
        # 'triad/2022-09-07_r.yanase_k.kobayashi_s.kamioka_4132',
    ]

    tetrad = [
        # 'tetrad/2022-07-01_y.inoue_k.tozuka_b.poitrimol_y.baba_1234_trans',
        # 'tetrad/2022-07-30_k.ohya_r.tanaka_y.baba_m.nakamura_1234',
        # 'tetrad/2022-07-30_k.ohya_r.tanaka_y.baba_m.nakamura_2143',
        # 'tetrad/2022-07-30_k.ohya_r.tanaka_y.baba_m.nakamura_4312',
        # 'tetrad/2022-07-30_r.tanaka_h.nishimura_k.tozuka_b.poitrimol_1234',
        # 'tetrad/2022-07-30_r.tanaka_h.nishimura_k.tozuka_b.poitrimol_2413',
        # 'tetrad/2022-07-30_r.tanaka_h.nishimura_k.tozuka_b.poitrimol_4123',
        # 'tetrad/2022-07-30_s.watanabe_h.nishimura_y.baba_y.yoshida_1234',
        # 'tetrad/2022-07-30_s.watanabe_h.nishimura_y.baba_y.yoshida_3421',
        # 'tetrad/2022-07-30_s.watanabe_h.nishimura_y.baba_y.yoshida_4312',
        # 'tetrad/2022-07-30_s.watanabe_ko.kobayashi_y.baba_k.tozuka_1234',
        # 'tetrad/2022-07-30_s.watanabe_ko.kobayashi_y.baba_k.tozuka_1324',
        # 'tetrad/2022-07-30_s.watanabe_ko.kobayashi_y.baba_k.tozuka_3241',
        # 'tetrad/2022-07-30_y.inoue_y.yoshida_n.ito_s.sanuka_1234',
        # 'tetrad/2022-07-30_y.inoue_y.yoshida_n.ito_s.sanuka_2413_2',
        # 'tetrad/2022-07-30_y.inoue_y.yoshida_n.ito_s.sanuka_2413',
        # 'tetrad/2022-07-30_y.inoue_y.yoshida_n.ito_s.sanuka_3241',
        # 'tetrad/2022-08-26_k.kobayashi_r.yanase_ko.kobayashi_h.nishimura_1234',
        # 'tetrad/2022-08-26_k.kobayashi_r.yanase_ko.kobayashi_h.nishimura_2314',
        # 'tetrad/2022-08-26_k.kobayashi_r.yanase_ko.kobayashi_h.nishimura_3421',
        # 'tetrad/2022-09-07_b.poitrimol_y.kobayashi_s.kamioka_y.inoue_1234',
        # 'tetrad/2022-09-07_b.poitrimol_y.kobayashi_s.kamioka_y.inoue_2413',
        # 'tetrad/2022-09-07_b.poitrimol_y.kobayashi_s.kamioka_y.inoue_4132',
        # 'tetrad/2022-09-07_k.kobayashi_y.kobayashi_s.kamioka_r.yanase_1234',
        # 'tetrad/2022-09-07_k.kobayashi_y.kobayashi_s.kamioka_r.yanase_2413',
        # 'tetrad/2022-09-07_k.kobayashi_y.kobayashi_s.kamioka_r.yanase_3142',
        # 'tetrad/2022-09-07_r.yanase_ko.kobayashi_b.poitrimol_y.kobayashi_1234',
        # 'tetrad/2022-09-07_r.yanase_ko.kobayashi_b.poitrimol_y.kobayashi_2341',
        # 'tetrad/2022-09-07_r.yanase_ko.kobayashi_b.poitrimol_y.kobayashi_3124',
    ]

    tpe = ThreadPoolExecutor(max_workers=24)

    files = []
    files.extend(dyad)
    files.extend(triad)
    files.extend(tetrad)

    for file in files:
        tpe.submit(process, file)

    tpe.shutdown()
    print('done')