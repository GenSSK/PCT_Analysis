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
    # data = npz.single_load('2022-01-22_Eleventh_test.npz')
    # data = npz.single_load('2022-01-22_Twelfth_test.npz')
    # data = npz.single_load('2022-01-25_nofront_test.npz')
    # data = npz.single_load('2022-02-01_nonfront_test.npz')
    # data = npz.single_load('2022-02-10_macromicro_ratio_normalize.npz')
    # data = npz.single_load('2022-02-10_macromicro.npz')
    # data = npz.single_load('2022-02-15_amref=text.npz')
    # data = npz.single_load('2022-02-01_front_test.npz')
    # data = npz.single_load('2022-01-20_FourPeople_test.npz')
    data1 = npz.single_load('2022-02-17_test.npz')
    # data2 = npz.single_load('2022-02-17_test2.npz')
    # data3 = npz.single_load('2022-02-17_test3.npz')
    # data = npz.all_load()

    # calc = Calc.Calculation(data)
    # calc.period_performance()

    pct = PCT.PCT()

    # pct.npz_load()
    pct.graph_sub(data1)
    # pct.task_show(data1)
    # pct.task_show(data2)
    # pct.task_show(data3)
    # pct.task_show_sub(data)
    # PCT.performance_calc()
    # PCT.performance()

    # anz = Analyze.Analyze()
    # anz.fit(data['i1_p_iq'][60000:130000:100], data['i1_p_am'][60000:130000:100])