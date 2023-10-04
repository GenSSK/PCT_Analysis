import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation, ArtistAnimation
from matplotlib.colors import Normalize

import Npz
import PCT
import CFO_analysis
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import seaborn as sns
import os

plt.switch_backend('Qt5Agg')

Integration_data = [
    'k.kobayashi_0.000000.npz',
    'k.kobayashi_0.100000.npz',
    'k.kobayashi_0.200000.npz',
    'k.kobayashi_0.300000.npz',
    'k.kobayashi_0.400000.npz',
    'k.kobayashi_0.500000.npz',
    'k.kobayashi_0.600000.npz',
    'k.kobayashi_0.700000.npz',
    'k.kobayashi_0.800000.npz',
    'k.kobayashi_0.900000.npz',
    'k.kobayashi_1.000000.npz',
    'k.kobayashi_1.100000.npz',
    'k.kobayashi_1.200000.npz',
    'k.kobayashi_1.300000.npz',
    'k.kobayashi_1.400000.npz',
    'k.kobayashi_1.500000.npz',
    'k.kobayashi_1.600000.npz',
    'k.kobayashi_1.700000.npz',
    'k.kobayashi_1.800000.npz',
    'k.kobayashi_1.900000.npz',
    'k.kobayashi_2.000000.npz',
    'k.kobayashi_2.100000.npz',
    'k.kobayashi_2.200000.npz',
    'k.kobayashi_2.300000.npz',
    'k.kobayashi_2.400000.npz',
    'k.kobayashi_2.500000.npz',
    'k.kobayashi_2.600000.npz',
    'k.kobayashi_2.700000.npz',
    'k.kobayashi_2.800000.npz',
    'k.kobayashi_2.900000.npz',
    'k.kobayashi_3.000000.npz',
]

npz = Npz.NPZ()
read_data = npz.select_load(Integration_data)

smp = 0.01  # サンプリング時間
time = 3.0  # ターゲットの移動時間
eliminationtime = 0.0  # 消去時間
starttime = 12.0  # タスク開始時間
endtime = 42.0  # タスク終了時間
tasktime = endtime - starttime  # タスクの時間
period = int((tasktime - eliminationtime) / time)  # 回数
num = int(time / smp)  # 1ピリオドにおけるデータ数
start_num = int((starttime) / smp)
end_num = int((endtime) / smp)
nn_read_flag = False

dec = 10

plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['xtick.top'] = 'True'
plt.rcParams['ytick.right'] = 'True'
# plt.rcParams['axes.grid'] = 'True'
plt.rcParams['xtick.direction'] = 'out'  # x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction'] = 'out'  # y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['xtick.major.width'] = 0.0  # x軸主目盛り線の線幅
plt.rcParams['ytick.major.width'] = 0.0  # y軸主目盛り線の線幅
plt.rcParams['font.size'] = 14  # フォントの大きさ
plt.rcParams['axes.linewidth'] = 0.0  # 軸の線幅edge linewidth。囲みの太さ

plt.rcParams['lines.linewidth'] = 0.5  # 線の太さ

plt.rcParams["legend.fancybox"] = True  # 丸角
plt.rcParams["legend.framealpha"] = 0  # 透明度の指定、0で塗りつぶしなし
plt.rcParams["legend.edgecolor"] = 'black'  # edgeの色を変更
plt.rcParams["legend.handlelength"] = 2  # 凡例の線の長さを調節
plt.rcParams["legend.labelspacing"] = 0.1  # 垂直方向（縦）の距離の各凡例の距離
plt.rcParams["legend.handletextpad"] = .4  # 凡例の線と文字の距離の長さ

plt.rcParams["legend.markerscale"] = 2  # 点がある場合のmarker scale
plt.rcParams['axes.xmargin'] = '0'  # '.05'
plt.rcParams['axes.ymargin'] = '0'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.edgecolor'] = 'white'
# plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['pdf.fonttype'] = 42  # PDFにフォントを埋め込むためのパラメータ
# plt.rcParams["axes.facecolor"] = 'white'

# type = ['thm_r', 'thm_p', 'text_r', 'text_p']
gain_type = ['thm_gain', 'thm_gain', 'text_gain', 'text_gain']
label = ['Roll angle (rad)', 'Pitch angle (rad)', 'Roll force (Nm)', 'Pitch force (Nm)']

type = ['thm_p', 'text_p']
label = ['Pitch angle (rad)', 'Pitch force (Nm)']

fig, ax = plt.subplots(len(type), 1, figsize=(10, 6), dpi=300, sharex=True, facecolor="white")

# ytics = [
#     np.arange(-10, 10, 0.5),
#     np.arange(-10, 10, 0.5),
#     np.arange(-8.0, 8.0, 2.0),
#     np.arange(-8.0, 8.0, 2.0),
# ]
# ylim = [
#     [-1.5, 1.5],
#     [-1.5, 1.5],
#     [-8.0, 8.0],
#     [-8.0, 8.0],
# ]
#
# ylim_2nd = [
#     [-1.0, 3.0],
#     [-1.0, 3.0],
#     [-3.0, 8.0],
#     [-3.0, 8.0],
# ]

ytics = [
    np.arange(-20, 20, 2.0),
    np.arange(-12.0, 12.0, 4.0),
]
ylim = [
    [-2.0, 2.0],
    [-4.0, 4.0],
]

ylim_2nd = [
    [-2.0, 8.0],
    [-3.0, 15.0],
]

model_mode = ['Follower', 'Leader', 'Altruistic', 'Selfish']

delays = np.arange(0.0, 3.1, 0.1)

delay_set = 10
data = read_data[delay_set]

print(end_num - start_num)
delay = data['output_delay'][0]
delay_num = int(delay / smp)


# delay_num = int(delay_num) * 2
# delay = 0

def graph_initial():
    plt.xticks(np.arange(starttime, endtime * 2, time * 2))
    plt.xlim([starttime, endtime])  # x軸の範囲
    plt.xlabel("Time (sec)")

    update(0)
    for i, sub in enumerate(ax):
        print("graph")
        sub.set_ylabel(label[i])
        sub.legend(ncol=10, columnspacing=1, loc='upper left')
        sub.set_yticks(ytics[i])
        sub.set_ylim(ylim[i])


def update(num):
    for i, sub in enumerate(ax):
        # if data['output_delay'][0] != 0.0:
        #     sub.plot(data['time'][start_num:start_num + num * dec] - 20.0, data['label_' + type[i] + '_1'][start_num:start_num + num * dec], label='Label(' + str(data['output_delay'][0]) + 's)', lw=4, color='Blue', alpha=0.4)
        period1 = (start_num - delay_num) + (num - 1) * dec
        period2 = (start_num - delay_num) + num * dec + (dec - 1)
        sc = sub.plot(data['time'][period1:period2] - 20.0 + delay, data['pre_' + type[i]][period1:period2],
                      label='Prediction Horizon(' + str(data['output_delay'][0]) + 's)', lw=2, color='skyblue', alpha=1.0)

        sub.plot(data['time'][period1:period2] - 20.0, data['label_' + type[i]][period1:period2],
                 label='Label', lw=1, color='red', alpha=1.0)

        # ax2 = sub.twinx()
        # for j in range(4):
        #     sub.plot(data['time'][start_num:end_num:dec] - 20.0, data['pre_' + type[i] + '_model_' + str(j+1)][start_num - delay:end_num - delay:dec],
        #              label=model_mode[j], lw=1.0, alpha=0.4)
        #
        #     ax2.plot(data['time'][start_num:end_num:dec] - 20.0, data['coefficient_' + type[i] + '_' + str(j+1)][start_num - delay:end_num - delay:dec],
        #              label='coefficient_' + str(j+1), lw=1.0, alpha=0.4, linestyle='dashed')

        sub.plot()

    # ax2.set_ylim(ylim_2nd[i])
    # ax2.set_ylabel('Coefficient')
    # sub.legend(ncol=10, columnspacing=1, loc='upper left')
    # sub.set_yticks(ytics[i])
    # ax2.set_ylim([0, 3.0])

    print(data['time'][(start_num - delay_num) + num * dec] - 20.0)
    # print(num * dec)
    # return ax[0], ax[1]

    # plt.show()


graph_initial()
anim = FuncAnimation(fig, update, frames=int((end_num - start_num + delay_num) / dec))

os.makedirs('fig/gif', exist_ok=True)
anim.save("fig/gif/Time series_Integration_single_" + '{:.1f}'.format(delays[delay_set]) + ".gif", writer='pillow', fps=int(1 / (dec * smp)))
# plt.show()
plt.close()
