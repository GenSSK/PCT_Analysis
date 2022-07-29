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
    # data = npz.single_load('y.inoue_ln3_id30_h10_h255_h30_od4_lr0.000100_bn30000_en100_dec1_dt0.010000_it0.100000_dtt0.010000_itt0.100000_dte0.010000_ite0.100000_tg0.500000_ttg0.100000_teg1.000000_predict.npz') #main.cpp
    # data = npz.single_load('y.inoue_ln3_id150_h10_h2220_h30_od4_lr0.000100_bn30000_en70_dec1_dt0.027349_it0.189596_dtt0.027349_itt0.189596_dte0.027349_ite0.189596_tg0.042071_ttg0.028865_teg1.570370_predict.npz') #main.cpp
    # data = npz.single_load('k.tozuka_ln3_id150_h10_h2115_h30_od4_lr0.000100_bn30000_en45_dec1_dt0.033422_it0.131264_dtt0.033422_itt0.131264_dte0.033422_ite0.131264_tg0.019921_ttg0.015404_teg1.839330_predict.npz') #main.cpp
    # data = npz.single_load('b.poitrimol_ln3_id120_h10_h2165_h30_od4_lr0.000100_bn30000_en55_dec1_dt0.015045_it0.153912_dtt0.015045_itt0.153912_dte0.015045_ite0.153912_tg0.163840_ttg0.024175_teg1.562840_predict.npz') #main.cpp
    data = npz.single_load('y.baba_ln3_id150_h10_h2195_h30_od4_lr0.000100_bn30000_en60_dec1_dt0.043150_it0.149401_dtt0.043150_itt0.149401_dte0.043150_ite0.149401_tg0.046506_ttg0.027517_teg1.920420_predict.npz') #main.cpp
    sm = solomodel.SoloModel(data)
    # sm.check_loss()
    # sm.recalc_ball_movement()
    sm.check_ball()
    # sm.analyze()
    sm.analyze_subplot()
    # sm.check_input()

    # distance_r, path_r = fastdtw(data['label_pre_text_r'][::100], data['pre_text_r'][::100], dist=euclidean)
    #
    # print(distance_r)

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

