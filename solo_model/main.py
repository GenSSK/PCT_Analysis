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


    # OperationModel.cpp
    log_read = pd.read_table('/home/genki/data/log/pandf.log', header=None)
    size = log_read[0].size - 1
    filename = log_read[0][size]
    print(filename)
    #g.sasaki_ln5_id30_h145_h255_h320_od4_lr0.000100_bn30000_en50_dec1_dt0.010000_it0.100000_dtt0.010000_itt0.100000_dte0.010000_ite0.100000_tg0.500000_ttg0.100000_teg1.000000.npz うまくいってるかも
    data = npz.single_load(log_read[0][size])
    sm = solomodel.SoloModel(data)
    sm.check_loss()
    sm.recalc_ball_movement()
    sm.check_ball()
    sm.analyze()
    sm.check_input()

    # OperationModel_predict.cpp
    # log_read = pd.read_table('/home/genki/data/log/predict.log', header=None)
    # size = log_read[0].size - 1
    # filename = log_read[0][size]
    # print(filename)
    # data = npz.single_load(log_read[0][size])
    # sm_pre = solomodel.SoloModel(data)
    # sm_pre.check_ball()
    # sm_pre.analyze()
    # sm_pre.check_input()




    # filename = log_read[0][size]
    # print(filename)
    # data = npz.single_load(log_read[0][size])
    # sm = solomodel.SoloModel(data)
    # sm.check_loss()
    # # sm.recalc_ball_movement()
    # # sm.check_ball()
    # sm.analyze()
    # sm.check_input()

    # log_read = pd.read_table('/home/genki/data/log/f.log', header=None)
    # size = log_read[0].size - 1
    # data_force = npz.single_load(log_read[0][size])
    # # sm_force = solomodel.SoloModel(data_force)
    # # sm_force.check_loss()
    # # sm_force.analyze_force()

    # log_read = pd.read_table('/home/genki/data/log/p.log', header=None)
    # size = log_read[0].size - 1
    # data_position = npz.single_load(log_read[0][size])
    # # sm_position = solomodel.SoloModel(data_position)
    # # sm_position.check_loss()
    # # sm_position.check_ball()
    # # sm_position.analyze_position()

    plt.show()


    # hist.graph_out(data)

