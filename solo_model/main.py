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
    data = npz.single_load('2022-04-07_5lay_epo600_1.npz')
    data = npz.single_load('2022-04-07_5lay_epo30_1_ACTUAL.npz')
    data_force = npz.single_load('2022-04-08_force_5lay_epo100_1.npz')
    data_force = npz.single_load('2022-04-08_force_5lay_epo3_1_ACTUAL.npz')
    pct = PCT.PCT()
    sm = solomodel.SoloModel(data)
    sm_force = solomodel.SoloModel(data_force)
    hist = histogram.histogram()


    # pct.task_show(data)

    sm.check_loss()
    # sm.recalc_ball_movement()
    # sm.check_ball()
    sm.analyze()

    # sm_force.check_loss()
    # sm_force.analyze_force()


    # hist.graph_out(data)

