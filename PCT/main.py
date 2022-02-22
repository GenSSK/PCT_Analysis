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
import ForceAnalysisRobomech

plt.switch_backend('Qt5Agg')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    npz = Npz.NPZ()
    # data = npz.single_load('2022-02-21_watanabe-inoue-kobayashi-tozuka_normal.npz')
    # data = npz.single_load('2022-02-21_watanabe-inoue-kobayashi-tozuka_alone.npz')
    # data = npz.all_load()
    normal = npz.type_load('normal')
    alone = npz.type_load('alone')
    nothing = npz.type_load('nothing')

    far = ForceAnalysisRobomech.ForceAnalysis(normal, alone, nothing)
    far.compare_individual(1)


    # calc = Calc.Calculation(data)
    # calc.period_performance()

    pct = PCT.PCT()

    # pct.graph_sub(normal[0])
    # pct.task_show(data)
    # pct.task_show(data2)
    # pct.task_show(data3)
    # pct.task_show_sub(data)

    # anz = Analyze.Analyze()
    # anz.fit(data['i1_p_iq'][60000:130000:100], data['i1_p_am'][60000:130000:100])