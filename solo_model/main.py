# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import pandas as pd
import Npz
import PCT
import solomodel
import histogram

plt.switch_backend('Qt5Agg')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    npz = Npz.NPZ()
    pct = PCT.PCT()
    hist = histogram.histogram()
    # pct.task_show(data)

    tes = pd.read_table('/home/genki/data/log/pandf.log', header=None)
    print(tes[0][0])


    # f = open('/home/genki/data/log/pandf.log', 'r')
    # tes = f.read()
    # print(tes)
    exit()

    data = npz.single_load('g.sasaki_ln5_id30_h130_h250_h327_od4_lr0.000100_bn3000_en10_dt0.010000_it0.010000_dtt0.010000_itt0.010000_dte0.010000_ite0.010000_tg0.500000_ttg0.100000_teg1.000000.npz')
    data = npz.single_load('g.sasaki_ln5_id30_h130_h250_h327_od4_lr0.000100_bn3000_en10_dt0.010000_it0.010000_dtt0.010000_itt0.010000_dte0.010000_ite0.010000_tg0.500000_ttg0.100000_teg1.000000_ACTUAL.npz')
    sm = solomodel.SoloModel(data)
    sm.check_loss()
    # sm.recalc_ball_movement()
    # sm.check_ball()
    sm.analyze()

    # data_force = npz.single_load('g.sasaki_ln5_id120_h1180_h2240_h3100_od2_lr0.000001_bn3000_en350_dt0.001000_it0.010000_dtt0.001000_itt0.010000_dte0.001000_ite0.010000_tg0.050000_ttg0.250000_teg1.000000_force.npz')
    # # data_force = npz.single_load('g.sasaki_ln5_id30_h130_h250_h325_od2_lr0.000010_bn3000_en300_dt0.010000_it0.010000_dtt0.010000_itt0.010000_dte0.010000_ite0.010000_tg0.500000_ttg0.100000_teg1.000000_force_ACTUAL.npz')
    # sm_force = solomodel.SoloModel(data_force)
    # sm_force.check_loss()
    # sm_force.analyze_force()

    # data_position = npz.single_load('g.sasaki_ln5_id30_h130_h250_h325_od2_lr0.000010_bn3000_en300_dt0.010000_it0.010000_dtt0.010000_itt0.010000_dte0.010000_ite0.010000_tg0.500000_ttg0.100000_teg1.000000_force.npz')
    # sm_position = solomodel.SoloModel(data_position)
    # sm_position.check_loss()
    # sm_position.check_ball()
    # sm_position.analyze_position()

    plt.show()


    # hist.graph_out(data)

