# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import Npz
import PCT
import solomodel
import histogram

plt.switch_backend('Qt5Agg')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    npz = Npz.NPZ()
    data = npz.single_load('g.sasaki_ln5_id30_h130_h250_h327_od4_lr0_bn3000_en1000.npz')
    # data = npz.single_load('2022-04-08_5lay_mini_epo1000_1_ACTUAL.npz')
    data_force = npz.single_load('g.sasaki_ln5_id30_h130_h250_h325_od2_lr0.000100_bn3000_en500_force.npz')
    # data_force = npz.single_load('g.sasaki_ln5_id30_h130_h250_h325_od2_lr0_bn3000_en500_force_ACTUAL.npz')
    pct = PCT.PCT()
    sm = solomodel.SoloModel(data)
    hist = histogram.histogram()


    # pct.task_show(data)

    # sm.check_loss()
    # sm.recalc_ball_movement()
    # sm.check_ball()
    # sm.analyze()

    sm_force = solomodel.SoloModel(data_force)
    sm_force.check_loss()
    sm_force.analyze_force()


    # hist.graph_out(data)

