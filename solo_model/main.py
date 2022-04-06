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
    data = npz.single_load('test.npz')
    pct = PCT.PCT()
    sm = solomodel.SoloModel(data)
    hist = histogram.histogram()


    # pct.task_show(data)

    # sm.check_loss()
    sm.check_ball()
    sm.analyze()

    hist.graph_out(data)

