# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import math
import Calc
import Npz
import PCT
import Analyze

plt.switch_backend('Qt5Agg')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    npz = Npz.NPZ()
    data = npz.single_load('2022-03-26_g.sasaki_solo.npz')

    # calc = Calc.Calculation(normal[0])
    # calc.period_performance()

    pct = PCT.PCT()

    pct.graph_sub(data)
    # pct.task_show(data)
    # pct.task_show(data2)
    # pct.task_show(data3)
    # pct.task_show_sub(data)

    # anz = Analyze.Analyze()