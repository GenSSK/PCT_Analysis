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

plt.switch_backend('Qt5Agg')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    npz = Npz.NPZ()
    data = npz.single_load('2022-01-20_FourPeople_test.npz')
    # data = npz.all_load()

    calc = Calc.Calculation(data)
    calc.period_performance()

    # pct = PCT.PCT()

    # pct.npz_load()
    # pct.graph_sub()
    # PCT.task_show()
    # PCT.performance_calc()
    # PCT.performance()