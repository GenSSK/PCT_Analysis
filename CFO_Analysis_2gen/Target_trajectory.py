import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator

import scipy
import Npz

plt.switch_backend('Qt5Agg')


def generate_cmap(colors, cmap_name = 'custom_cmap'):
    values = range(len(colors))
    vmax = np.ceil(np.max(values))
    color_list = []
    for vi, ci in zip(values, colors):
        color_list.append( ( vi/ vmax, ci) )

    return mcolors.LinearSegmentedColormap.from_list(cmap_name, color_list, 256)


# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.family']= 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['xtick.top'] = 'True'
plt.rcParams['ytick.right'] = 'True'
# plt.rcParams['axes.grid'] = 'True'
plt.rcParams['xtick.direction'] = 'in'  # x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction'] = 'in'  # y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['xtick.major.width'] = 1.0  # x軸主目盛り線の線幅
plt.rcParams['ytick.major.width'] = 1.0  # y軸主目盛り線の線幅
plt.rcParams['font.size'] = 10  # フォントの大きさ
plt.rcParams['axes.linewidth'] = 1.0  # 軸の線幅edge linewidth。囲みの太さ

plt.rcParams['lines.linewidth'] = 0.5  # 線の太さ

plt.rcParams["legend.fancybox"] = False  # 丸角
plt.rcParams["legend.framealpha"] = 0  # 透明度の指定、0で塗りつぶしなし
plt.rcParams["legend.edgecolor"] = 'black'  # edgeの色を変更
plt.rcParams["legend.handlelength"] = 2  # 凡例の線の長さを調節
plt.rcParams["legend.labelspacing"] = 0.1  # 垂直方向（縦）の距離の各凡例の距離
plt.rcParams["legend.handletextpad"] = .4  # 凡例の線と文字の距離の長さ

plt.rcParams["legend.markerscale"] = 2  # 点がある場合のmarker scale
plt.rcParams['axes.xmargin'] = '0'  # '.05'
plt.rcParams['axes.ymargin'] = '0'
plt.rcParams['savefig.facecolor'] = 'None'
plt.rcParams['savefig.edgecolor'] = 'None'
# plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['pdf.fonttype'] = 42  # PDFにフォントを埋め込むためのパラメータ


file_names = [
    '2024-03-02_AdABC_Dyad_h.nakamura_k.tozuka_Circle_1_CFO.npz',
    '2024-03-02_AdABC_Dyad_h.nakamura_k.tozuka_Lemniscate_1_CFO.npz',
    '2024-03-02_AdABC_Dyad_h.nakamura_k.tozuka_RoseCurve_1_CFO.npz',
    '2024-03-02_AdABC_Dyad_h.nakamura_k.tozuka_Random_1_CFO.npz',
    '2024-03-02_AdABC_Dyad_h.nakamura_k.tozuka_Discrete_Random_1_CFO.npz',
]

Type = [
    'Circle',
    'Lemniscate',
    'Rose Curve',
    'Random',
    'Discrete Random',
]

fig, axs = plt.subplots(3, 2, figsize=(5.5, 8.5), dpi=200)

npz = Npz.NPZ()

load_data = npz.select_load(dir="/nfs/ssk-storage/data/cfo/dyad/shared/", filenames=file_names)
# load_data = npz.select_load(dir="/Users/genki/Downloads/exp/cfo/dyad/shared/", filenames=file_names)

ax_num = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [2, 0],
]

cmap = plt.get_cmap('viridis')
# cmthermal = generate_cmap(['#1c3f75', '#068fb9','#f1e235', '#d64e8b', '#730e22'], 'cmthermal')

for i in range(len(load_data)):
    data = load_data[i]
    ax = axs[ax_num[i][0], ax_num[i][1]]
    scatter = ax.scatter(data['targetx'][::100], data['targety'][::100], s=0.1, cmap=cmap, c=data['target_dot_magnitude'][::100])
    ax.set_title(Type[i])
    ax.set_xlabel('X-Axis (m)')
    ax.set_ylabel('Y-Axis (m)')
    ax.set_xticks(np.arange(-0.5, 0.5, 0.1))
    ax.set_yticks(np.arange(-0.5, 0.5, 0.1))
    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)

# カラーバーを作成し、figの右側に配置
cbar = fig.colorbar(scatter, ax=axs[2,1], orientation='horizontal')
cbar.set_label('Velocity (m/s)')
# カラーバーの目盛りの間隔を設定
cbar.locator = MaxNLocator(nbins=6)  # 例として5つの目盛りを設定
cbar.update_ticks()

os.makedirs('fig/Trajectory/', exist_ok=True)
plt.subplots_adjust(wspace=0.4, hspace=0.4)  # 横方向の余白を調整
plt.savefig('fig/Trajectory/Target_trajectory.png')
plt.savefig('fig/Trajectory/Target_trajectory.pdf')
plt.show()