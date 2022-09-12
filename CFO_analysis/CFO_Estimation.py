import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
import time

def process(filename):
    subprocess.call(["/home/genki/clion/CFO_Estimator/cmake-build-release/CFO_Estimator",
                     filename[0],
                     filename[1]
                     ])


if __name__ == '__main__':

    dyad = [
        # ['dyad/2022-07-28_y.inoue_b.poitrimol_1234', 'y.inoue_b.poitrimol'],
        # ['dyad/2022-07-28_y.inoue_b.poitrimol_1324', 'y.inoue_b.poitrimol'],
        # ['dyad/2022-07-28_y.inoue_b.poitrimol_1432', 'y.inoue_b.poitrimol'],
        # ['dyad/2022-07-29_i.tsunokuni_k.tozuka_2134', 'i.tsunokuni_k.tozuka'],
        # ['dyad/2022-07-29_i.tsunokuni_k.tozuka_3124', 'i.tsunokuni_k.tozuka'],
        # ['dyad/2022-07-29_i.tsunokuni_k.tozuka_4123', 'i.tsunokuni_k.tozuka'],
        # ['dyad/2022-07-29_i.tsunokuni_y.inoue_1234', 'i.tsunokuni_y.inoue'],
        # ['dyad/2022-07-29_i.tsunokuni_y.inoue_1324', 'i.tsunokuni_y.inoue'],
        # ['dyad/2022-07-29_i.tsunokuni_y.inoue_1423', 'i.tsunokuni_y.inoue'],
        # ['dyad/2022-07-31_m.sugaya_n.ito_1234', 'm.sugaya_n.ito'],
        # ['dyad/2022-07-31_m.sugaya_n.ito_1324', 'm.sugaya_n.ito'],
        # ['dyad/2022-07-31_m.sugaya_n.ito_1423', 'm.sugaya_n.ito'],
        # ['dyad/2022-08-23_r.yanase_ko.kobayashi_1234', 'r.yanase_ko.kobayashi'],
        # ['dyad/2022-08-23_r.yanase_ko.kobayashi_1324', 'r.yanase_ko.kobayashi'],
        # ['dyad/2022-08-23_r.yanase_ko.kobayashi_1423', 'r.yanase_ko.kobayashi'],
        # ['dyad/2022-08-26_y.kobayashi_r.yanase_1234', 'y.kobayashi_r.yanase'],
        # ['dyad/2022-08-26_y.kobayashi_r.yanase_1324', 'y.kobayashi_r.yanase'],
        # ['dyad/2022-08-26_y.kobayashi_r.yanase_1423', 'y.kobayashi_r.yanase'],
        # ['dyad/2022-09-05_s.watanabe_ko.kobayashi_1234', 's.watanabe_ko.kobayashi'],
        # ['dyad/2022-09-05_s.watanabe_ko.kobayashi_1324', 's.watanabe_ko.kobayashi'],
        # ['dyad/2022-09-05_s.watanabe_ko.kobayashi_1423', 's.watanabe_ko.kobayashi'],
        # ['dyad/2022-09-07_k.kobayashi_r.yanase_1234', 'k.kobayashi_r.yanase'],
        # ['dyad/2022-09-07_k.kobayashi_r.yanase_1324', 'k.kobayashi_r.yanase'],
        # ['dyad/2022-09-07_k.kobayashi_r.yanase_1423', 'k.kobayashi_r.yanase'],
        # ['dyad/2022-09-07_k.tozuka_ko.kobayashi_1234', 'k.tozuka_ko.kobayashi'], #使えない
        # ['dyad/2022-09-07_k.tozuka_ko.kobayashi_1324', 'k.tozuka_ko.kobayashi'],
        # ['dyad/2022-09-07_k.tozuka_ko.kobayashi_1423', 'k.tozuka_ko.kobayashi'],
        # ['dyad/2022-09-07_y.inoue_k.kobayashi_1234', 'y.inoue_k.kobayashi'],
        # ['dyad/2022-09-07_y.inoue_k.kobayashi_1324', 'y.inoue_k.kobayashi'],
        # ['dyad/2022-09-07_y.inoue_k.kobayashi_1423', 'y.inoue_k.kobayashi'],
    ]

    triad = [
        # ['triad/2022-07-31_h.igarashi_ko.kobayashi_t.kassai_1234', 'h.igarashi_ko.kobayashi_t.kassai'], #使えない
        # ['triad/2022-07-31_k.tozuka_y.inoue_m.nakamura_1234', 'k.tozuka_y.inoue_m.nakamura'],
        # ['triad/2022-07-31_k.tozuka_y.inoue_m.nakamura_2134', 'k.tozuka_y.inoue_m.nakamura'],
        # ['triad/2022-07-31_k.tozuka_y.inoue_m.nakamura_3412', 'k.tozuka_y.inoue_m.nakamura'],
        # ['triad/2022-07-31_m.sugaya_s.sanuka_m.nakamura_1234', 'm.sugaya_s.sanuka_m.nakamura'],
        # ['triad/2022-07-31_m.sugaya_s.sanuka_m.nakamura_2134', 'm.sugaya_s.sanuka_m.nakamura'],
        # ['triad/2022-07-31_m.sugaya_s.sanuka_m.nakamura_4213', 'm.sugaya_s.sanuka_m.nakamura'],
        # ['triad/2022-08-23_s.watanabe_t.kassai_h.nishimura_1234', 's.watanabe_t.kassai_h.nishimura'], #使えない
        # ['triad/2022-08-23_s.watanabe_t.kassai_h.nishimura_2134', 's.watanabe_t.kassai_h.nishimura'],
        # ['triad/2022-08-23_s.watanabe_t.kassai_h.nishimura_3214', 's.watanabe_t.kassai_h.nishimura'],
        # ['triad/2022-08-26_k.kobayashi_r.yanase_ko.kobayashi_1234', 'k.kobayashi_r.yanase_ko.kobayashi'],
        # ['triad/2022-08-26_k.kobayashi_r.yanase_ko.kobayashi_2314', 'k.kobayashi_r.yanase_ko.kobayashi'],
        # ['triad/2022-08-26_k.kobayashi_r.yanase_ko.kobayashi_3412', 'k.kobayashi_r.yanase_ko.kobayashi'],
        # ['triad/2022-09-05_b.poitrimol_y.inoue_k.tozuka_1234', 'b.poitrimol_y.inoue_k.tozuka'],
        # ['triad/2022-09-05_b.poitrimol_y.inoue_k.tozuka_2314', 'b.poitrimol_y.inoue_k.tozuka'],
        # ['triad/2022-09-07_b.poitrimol_k.tozuka_y.kobayashi_1234', 'b.poitrimol_k.tozuka_y.kobayashi'],
        # ['triad/2022-09-07_b.poitrimol_k.tozuka_y.kobayashi_2431', 'b.poitrimol_k.tozuka_y.kobayashi'],
        # ['triad/2022-09-07_b.poitrimol_k.tozuka_y.kobayashi_3421', 'b.poitrimol_k.tozuka_y.kobayashi'],
        # ['triad/2022-09-07_b.poitrimol_y.inoue_r.yanase_1234', 'b.poitrimol_y.inoue_r.yanase'],
        # ['triad/2022-09-07_b.poitrimol_y.inoue_r.yanase_3241', 'b.poitrimol_y.inoue_r.yanase'],
        # ['triad/2022-09-07_b.poitrimol_y.inoue_r.yanase_4132', 'b.poitrimol_y.inoue_r.yanase'],
        # ['triad/2022-09-07_ko.kobayashi_k.kobayashi_y.kobayashi_1234', 'ko.kobayashi_k.kobayashi_y.kobayashi'],
        # ['triad/2022-09-07_ko.kobayashi_k.kobayashi_y.kobayashi_3412', 'ko.kobayashi_k.kobayashi_y.kobayashi'],
        # ['triad/2022-09-07_ko.kobayashi_k.kobayashi_y.kobayashi_4132', 'ko.kobayashi_k.kobayashi_y.kobayashi'],
        # ['triad/2022-09-07_r.yanase_k.kobayashi_s.kamioka_1234', 'r.yanase_k.kobayashi_s.kamioka'],
        # ['triad/2022-09-07_r.yanase_k.kobayashi_s.kamioka_2431', 'r.yanase_k.kobayashi_s.kamioka'],
        # ['triad/2022-09-07_r.yanase_k.kobayashi_s.kamioka_4132', 'r.yanase_k.kobayashi_s.kamioka'],
    ]

    tetrad = [
        # ['tetrad/2022-07-01_y.inoue_k.tozuka_b.poitrimol_y.baba_1234_trans', 'y.inoue_k.tozuka_b.poitrimol_y.baba_1234_'], #使えない
        # ['tetrad/2022-07-30_k.ohya_r.tanaka_y.baba_m.nakamura_1234', 'k.ohya_r.tanaka_y.baba_m.nakamura'],
        # ['tetrad/2022-07-30_k.ohya_r.tanaka_y.baba_m.nakamura_2143', 'k.ohya_r.tanaka_y.baba_m.nakamura'],
        # ['tetrad/2022-07-30_k.ohya_r.tanaka_y.baba_m.nakamura_4312', 'k.ohya_r.tanaka_y.baba_m.nakamura'],
        # ['tetrad/2022-07-30_r.tanaka_h.nishimura_k.tozuka_b.poitrimol_1234', 'r.tanaka_h.nishimura_k.tozuka_b.poitrimol'],
        # ['tetrad/2022-07-30_r.tanaka_h.nishimura_k.tozuka_b.poitrimol_2413', 'r.tanaka_h.nishimura_k.tozuka_b.poitrimol'],
        # ['tetrad/2022-07-30_r.tanaka_h.nishimura_k.tozuka_b.poitrimol_4123', 'r.tanaka_h.nishimura_k.tozuka_b.poitrimol'],
        # ['tetrad/2022-07-30_s.watanabe_h.nishimura_y.baba_y.yoshida_1234', 's.watanabe_h.nishimura_y.baba_y.yoshida'],
        # ['tetrad/2022-07-30_s.watanabe_h.nishimura_y.baba_y.yoshida_3421', 's.watanabe_h.nishimura_y.baba_y.yoshida'],
        # ['tetrad/2022-07-30_s.watanabe_h.nishimura_y.baba_y.yoshida_4312', 's.watanabe_h.nishimura_y.baba_y.yoshida'],
        # ['tetrad/2022-07-30_s.watanabe_ko.kobayashi_y.baba_k.tozuka_1234', 's.watanabe_ko.kobayashi_y.baba_k.tozuka'],
        # ['tetrad/2022-07-30_s.watanabe_ko.kobayashi_y.baba_k.tozuka_1324', 's.watanabe_ko.kobayashi_y.baba_k.tozuka'],
        # ['tetrad/2022-07-30_s.watanabe_ko.kobayashi_y.baba_k.tozuka_3241', 's.watanabe_ko.kobayashi_y.baba_k.tozuka'],
        ['tetrad/2022-07-30_y.inoue_y.yoshida_n.ito_s.sanuka_1234', 'y.inoue_y.yoshida_n.ito_s.sanuka'],
        ['tetrad/2022-07-30_y.inoue_y.yoshida_n.ito_s.sanuka_2413_2', 'y.inoue_y.yoshida_n.ito_s.sanuka'],
        ['tetrad/2022-07-30_y.inoue_y.yoshida_n.ito_s.sanuka_2413', 'y.inoue_y.yoshida_n.ito_s.sanuka'],
        ['tetrad/2022-07-30_y.inoue_y.yoshida_n.ito_s.sanuka_3241', 'y.inoue_y.yoshida_n.ito_s.sanuka'],
        # ['tetrad/2022-08-26_k.kobayashi_r.yanase_ko.kobayashi_h.nishimura_1234', 'k.kobayashi_r.yanase_ko.kobayashi_h.nishimura'],
        # ['tetrad/2022-08-26_k.kobayashi_r.yanase_ko.kobayashi_h.nishimura_2314', 'k.kobayashi_r.yanase_ko.kobayashi_h.nishimura'],
        # ['tetrad/2022-08-26_k.kobayashi_r.yanase_ko.kobayashi_h.nishimura_3421', 'k.kobayashi_r.yanase_ko.kobayashi_h.nishimura'],
        # ['tetrad/2022-09-07_b.poitrimol_y.kobayashi_s.kamioka_y.inoue_1234', 'b.poitrimol_y.kobayashi_s.kamioka_y.inoue'],
        # ['tetrad/2022-09-07_b.poitrimol_y.kobayashi_s.kamioka_y.inoue_2413', 'b.poitrimol_y.kobayashi_s.kamioka_y.inoue'],
        # ['tetrad/2022-09-07_b.poitrimol_y.kobayashi_s.kamioka_y.inoue_4132', 'b.poitrimol_y.kobayashi_s.kamioka_y.inoue'],
        # ['tetrad/2022-09-07_k.kobayashi_y.kobayashi_s.kamioka_r.yanase_1234', 'k.kobayashi_y.kobayashi_s.kamioka_r.yanase'],
        # ['tetrad/2022-09-07_k.kobayashi_y.kobayashi_s.kamioka_r.yanase_2413', 'k.kobayashi_y.kobayashi_s.kamioka_r.yanase'],
        # ['tetrad/2022-09-07_k.kobayashi_y.kobayashi_s.kamioka_r.yanase_3142', 'k.kobayashi_y.kobayashi_s.kamioka_r.yanase'],
        # ['tetrad/2022-09-07_r.yanase_ko.kobayashi_b.poitrimol_y.kobayashi_1234', 'r.yanase_ko.kobayashi_b.poitrimol_y.kobayashi'],
        # ['tetrad/2022-09-07_r.yanase_ko.kobayashi_b.poitrimol_y.kobayashi_2341', 'r.yanase_ko.kobayashi_b.poitrimol_y.kobayashi'],
        # ['tetrad/2022-09-07_r.yanase_ko.kobayashi_b.poitrimol_y.kobayashi_3124', 'r.yanase_ko.kobayashi_b.poitrimol_y.kobayashi'],
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