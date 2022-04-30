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
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


plt.switch_backend('Qt5Agg')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    npz = Npz.NPZ()
    pct = PCT.PCT()
    hist = histogram.histogram()
    # pct.task_show(data)


    # OperationModel.cpp
    # log_read = pd.read_table('/home/genki/data/log/pandf.log', header=None)
    # size = log_read[0].size - 1
    # filename = log_read[0][size]
    # print(filename)
    # data = npz.single_load(log_read[0][size])
    # data = npz.single_load('y.inoue_ln3_id30_h10_h255_h30_od4_lr0.000100_bn30000_en100_dec1_dt0.010000_it0.100000_dtt0.010000_itt0.100000_dte0.010000_ite0.100000_tg0.500000_ttg0.100000_teg1.000000.npz') #OperationModel.cpp OK
    # data = npz.single_load('y.inoue_ln3_id30_h10_h255_h30_od4_lr0.000100_bn30000_en100_dec1_dt0.010000_it0.100000_dtt0.010000_itt0.100000_dte0.010000_ite0.100000_tg0.500000_ttg0.100000_teg1.000000_optuna.npz') #OperationModel_optuna.cpp
    data = npz.single_load('y.inoue_ln3_id30_h10_h255_h30_od4_lr0.000100_bn30000_en100_dec1_dt0.010000_it0.100000_dtt0.010000_itt0.100000_dte0.010000_ite0.100000_tg0.500000_ttg0.100000_teg1.000000_predict.npz') #main.cpp
    # print('y.inoue_ln3_id30_h10_h255_h30_od4_lr0.000100_bn30000_en100_dec1_dt0.010000_it0.100000_dtt0.010000_itt0.100000_dte0.010000_ite0.100000_tg0.500000_ttg0.100000_teg1.000000.npz')
    sm = solomodel.SoloModel(data)
    # sm.check_loss()
    # sm.recalc_ball_movement()
    sm.check_ball()
    # sm.analyze()
    sm.analyze_subplot()
    # sm.check_input()

    distance_r, path_r = fastdtw(data['label_pre_text_r'][::100], data['pre_text_r'][::100], dist=euclidean)

    print(distance_r)

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

