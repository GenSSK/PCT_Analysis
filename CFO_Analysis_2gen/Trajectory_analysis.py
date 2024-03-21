import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import CFO_Analysis_2gen.CFO_analysis as CFO_analysis
from scipy.spatial.distance import correlation
from statannotations.Annotator import Annotator
from statsmodels.formula.api import ols
import statsmodels.api as sm

from mypackage.mystatistics import myHilbertTransform as HT
from mypackage.mystatistics import mySTFT as STFT
from mypackage.mystatistics import myhistogram as hist
from mypackage.mystatistics import myFilter as Filter
from mypackage.mystatistics import statistics as mystat
from mypackage import ParallelExecutor

import Combine_analysis

class TrajectoryAnalysis:
    def __init__(self, dyad_npz, triad_npz, tetrad_npz,
                 dyad_file_name, triad_file_name, tetrad_file_name,):
        trajectory = [
            'Circle',
            'Lemniscate',
            'RoseCurve',
            'Random',
            'Discrete_Random',
        ]
        # extract data of each trajectory
        npz = [dyad_npz, triad_npz, tetrad_npz]
        file_names = [dyad_file_name, triad_file_name, tetrad_file_name]
        extract_npz = []
        for i in range(len(npz)):
            extract_npz.append([])
            for j in range(len(trajectory)):
                extract_npz[i].append([])
                numpy_vars = {}
                file_name = []
                key = 0
                for k in range(len(npz[i])):
                    if file_names[i][k].find(trajectory[j]) != -1:
                        if trajectory[j] == 'Random' and file_names[i][k].find('Discrete') != -1:
                            pass
                        else:
                            numpy_vars[key] = npz[i][k]
                            file_name.append(file_names[i][k])
                            key += 1
                extract_npz[i][j] = numpy_vars
                # extract_npz[i][j] = file_name
        self.com_circle: Combine_analysis.combine = Combine_analysis.combine(extract_npz[0][0], extract_npz[1][0], extract_npz[2][0], 'Circle')
        self.com_lemniscate: Combine_analysis.combine = Combine_analysis.combine(extract_npz[0][1], extract_npz[1][1], extract_npz[2][1], 'Lemniscate')
        self.com_rose_curve: Combine_analysis.combine = Combine_analysis.combine(extract_npz[0][2], extract_npz[1][2], extract_npz[2][2], 'RoseCurve')
        self.com_random: Combine_analysis.combine = Combine_analysis.combine(extract_npz[0][3], extract_npz[1][3], extract_npz[2][3], 'Random')
        self.com_discrete_random: Combine_analysis.combine = Combine_analysis.combine(extract_npz[0][4], extract_npz[1][4], extract_npz[2][4], 'Discrete_Random')

    def comparison_performance_human_model(self, sigma: int = 'none'):
        error_ts_dyad_circle_hh, error_dot_ts_dyad_circle_hh = self.com_circle.dyad_cfo.period_performance_New(mode='H-H', sigma=sigma)
        error_ts_triad_circle_hh, error_dot_ts_triad_circle_hh = self.com_circle.triad_cfo.period_performance_New(mode='H-H', sigma=sigma)
        error_ts_tetrad_circle_hh, error_dot_ts_tetrad_circle_hh = self.com_circle.tetrad_cfo.period_performance_New(mode='H-H', sigma=sigma)

        error_ts_dyad_lemniscate_hh, error_dot_ts_dyad_lemniscate_hh = self.com_lemniscate.dyad_cfo.period_performance_New(mode='H-H', sigma=sigma)
        error_ts_triad_lemniscate_hh, error_dot_ts_triad_lemniscate_hh = self.com_lemniscate.triad_cfo.period_performance_New(mode='H-H', sigma=sigma)
        error_ts_tetrad_lemniscate_hh, error_dot_ts_tetrad_lemniscate_hh = self.com_lemniscate.tetrad_cfo.period_performance_New(mode='H-H', sigma=sigma)

        error_ts_dyad_rosecurve_hh, error_dot_ts_dyad_rosecurve_hh = self.com_rose_curve.dyad_cfo.period_performance_New(mode='H-H', sigma=sigma)
        error_ts_triad_rosecurve_hh, error_dot_ts_triad_rosecurve_hh = self.com_rose_curve.triad_cfo.period_performance_New(mode='H-H', sigma=sigma)
        error_ts_tetrad_rosecurve_hh, error_dot_ts_tetrad_rosecurve_hh = self.com_rose_curve.tetrad_cfo.period_performance_New(mode='H-H', sigma=sigma)

        error_ts_dyad_random_hh, error_dot_ts_dyad_random_hh = self.com_random.dyad_cfo.period_performance_New(mode='H-H', sigma=sigma)
        error_ts_triad_random_hh, error_dot_ts_triad_random_hh = self.com_random.triad_cfo.period_performance_New(mode='H-H', sigma=sigma)
        error_ts_tetrad_random_hh, error_dot_ts_tetrad_random_hh = self.com_random.tetrad_cfo.period_performance_New(mode='H-H', sigma=sigma)

        error_ts_dyad_discrete_random_hh, error_dot_ts_dyad__discrete_random_hh = self.com_discrete_random.dyad_cfo.period_performance_New(mode='H-H', sigma=sigma)
        error_ts_triad_discrete_random_hh, error_dot_ts_triad__discrete_random_hh = self.com_discrete_random.triad_cfo.period_performance_New(mode='H-H', sigma=sigma)
        error_ts_tetrad_discrete_random_hh, error_dot_ts_tetrad__discrete_random_hh = self.com_discrete_random.tetrad_cfo.period_performance_New(mode='H-H', sigma=sigma)

        error_ts_dyad_dyad_circle_mm, error_dot_ts_dyad_circle_mm = self.com_circle.dyad_cfo.period_performance_New(mode='M-M', sigma=sigma)
        error_ts_triad_triad_circle_mm, error_dot_ts_triad_circle_mm = self.com_circle.triad_cfo.period_performance_New(mode='M-M', sigma=sigma)
        error_ts_tetrad_tetrad_circle_mm, error_dot_ts_tetrad_circle_mm = self.com_circle.tetrad_cfo.period_performance_New(mode='M-M', sigma=sigma)

        error_ts_dyad_lemniscate_mm, error_dot_ts_dyad_lemniscate_mm = self.com_lemniscate.dyad_cfo.period_performance_New(mode='M-M', sigma=sigma)
        error_ts_triad_lemniscate_mm, error_dot_ts_triad_lemniscate_mm = self.com_lemniscate.triad_cfo.period_performance_New(mode='M-M', sigma=sigma)
        error_ts_tetrad_lemniscate_mm, error_dot_ts_tetrad_lemniscate_mm = self.com_lemniscate.tetrad_cfo.period_performance_New(mode='M-M', sigma=sigma)

        error_ts_dyad_rosecurve_mm, error_dot_ts_dyad_rosecurve_mm = self.com_rose_curve.dyad_cfo.period_performance_New(mode='M-M', sigma=sigma)
        error_ts_triad_rosecurve_mm, error_dot_ts_triad_rosecurve_mm = self.com_rose_curve.triad_cfo.period_performance_New(mode='M-M', sigma=sigma)
        error_ts_tetrad_rosecurve_mm, error_dot_ts_tetrad_rosecurve_mm = self.com_rose_curve.tetrad_cfo.period_performance_New(mode='M-M', sigma=sigma)

        error_ts_dyad_random_mm, error_dot_ts_dyad_random_mm = self.com_random.dyad_cfo.period_performance_New(mode='M-M', sigma=sigma)
        error_ts_triad_random_mm, error_dot_ts_triad_random_mm = self.com_random.triad_cfo.period_performance_New(mode='M-M', sigma=sigma)
        error_ts_tetrad_random_mm, error_dot_ts_tetrad_random_mm = self.com_random.tetrad_cfo.period_performance_New(mode='M-M', sigma=sigma)

        error_ts_dyad_discrete_random_mm, error_dot_ts_dyad_discrete_random_mm = self.com_discrete_random.dyad_cfo.period_performance_New(mode='M-M', sigma=sigma)
        error_ts_triad_discrete_random_mm, error_dot_ts_triad_discrete_random_mm = self.com_discrete_random.triad_cfo.period_performance_New(mode='M-M', sigma=sigma)
        error_ts_tetrad_discrete_random_mm, error_dot_ts_tetrad_discrete_random_mm = self.com_discrete_random.tetrad_cfo.period_performance_New(mode='M-M', sigma=sigma)


        trajectory_types = [
            'Circle',
            'Lemniscate',
            'Rose Curve',
            'Random',
            'Discrete Random',
        ]

        error_ts_hh = [
            [error_ts_dyad_circle_hh, error_ts_triad_circle_hh, error_ts_tetrad_circle_hh],
            [error_ts_dyad_lemniscate_hh, error_ts_triad_lemniscate_hh, error_ts_tetrad_lemniscate_hh],
            [error_ts_dyad_rosecurve_hh, error_ts_triad_rosecurve_hh, error_ts_tetrad_rosecurve_hh],
            [error_ts_dyad_random_hh, error_ts_triad_random_hh, error_ts_tetrad_random_hh],
            [error_ts_dyad_discrete_random_hh, error_ts_triad_discrete_random_hh, error_ts_tetrad_discrete_random_hh],
        ]

        error_dot_ts_hh = [
            [error_dot_ts_dyad_circle_hh, error_dot_ts_triad_circle_hh, error_dot_ts_tetrad_circle_hh],
            [error_dot_ts_dyad_lemniscate_hh, error_dot_ts_triad_lemniscate_hh, error_dot_ts_tetrad_lemniscate_hh],
            [error_dot_ts_dyad_rosecurve_hh, error_dot_ts_triad_rosecurve_hh, error_dot_ts_tetrad_rosecurve_hh],
            [error_dot_ts_dyad_random_hh, error_dot_ts_triad_random_hh, error_dot_ts_tetrad_random_hh],
            [error_dot_ts_dyad__discrete_random_hh, error_dot_ts_triad__discrete_random_hh, error_dot_ts_tetrad__discrete_random_hh],
        ]

        error_ts_mm = [
            [error_ts_dyad_dyad_circle_mm, error_ts_triad_triad_circle_mm, error_ts_tetrad_tetrad_circle_mm],
            [error_ts_dyad_lemniscate_mm, error_ts_triad_lemniscate_mm, error_ts_tetrad_lemniscate_mm],
            [error_ts_dyad_rosecurve_mm, error_ts_triad_rosecurve_mm, error_ts_tetrad_rosecurve_mm],
            [error_ts_dyad_random_mm, error_ts_triad_random_mm, error_ts_tetrad_random_mm],
            [error_ts_dyad_discrete_random_mm, error_ts_triad_discrete_random_mm, error_ts_tetrad_discrete_random_mm],
        ]

        error_dot_ts_mm = [
            [error_dot_ts_dyad_circle_mm, error_dot_ts_triad_circle_mm, error_dot_ts_tetrad_circle_mm],
            [error_dot_ts_dyad_lemniscate_mm, error_dot_ts_triad_lemniscate_mm, error_dot_ts_tetrad_lemniscate_mm],
            [error_dot_ts_dyad_rosecurve_mm, error_dot_ts_triad_rosecurve_mm, error_dot_ts_tetrad_rosecurve_mm],
            [error_dot_ts_dyad_random_mm, error_dot_ts_triad_random_mm, error_dot_ts_tetrad_random_mm],
            [error_dot_ts_dyad_discrete_random_mm, error_dot_ts_triad_discrete_random_mm, error_dot_ts_tetrad_discrete_random_mm],
        ]

        error_ts = [
            error_ts_hh,
            error_ts_mm
        ]

        error_dot_ts = [
            error_dot_ts_hh,
            error_dot_ts_mm
        ]


        Group_size = ['Dyad', 'Triad', 'Tetrad']

        Types = ['Human-Human', 'Model-Model']
        labels = ['Error (m)', 'Error speed (m/s$^2$)']
        labels_for_fig = ['Error', 'Error speed']

        df_error_ = []
        for i in range(len(trajectory_types)):
            for j in range(len(Group_size)):
                for k in range(len(error_ts_hh[i][j])):
                    for l in range(len(Types)):
                        df_error_.append(pd.DataFrame({
                            labels[0]: error_ts[l][i][j][k],
                            'Types': Types[l],
                            'trajectory': trajectory_types[i],
                            'Group size': Group_size[j],
                            'Group': 'Group' + str(k + 1)
                        }))

        df_error = pd.concat([i for i in df_error_], axis=0)
        df_error.reset_index(drop=True, inplace=True)

        df_error_dot_ = []
        for i in range(len(trajectory_types)):
            for j in range(len(Group_size)):
                for k in range(len(error_dot_ts_hh[i][j])):
                    for l in range(len(Types)):
                        df_error_dot_.append(pd.DataFrame({
                            labels[1]: error_dot_ts[l][i][j][k],
                            'Types': Types[l],
                            'trajectory': trajectory_types[i],
                            'Group size': Group_size[j],
                            'Group': 'Group' + str(k + 1)
                        }))
        df_error_dot = pd.concat([i for i in df_error_dot_], axis=0)
        df_error_dot.reset_index(drop=True, inplace=True)

        base_dir = 'fig/Performance/Comparison/'

        dfs = [df_error, df_error_dot]

        pairs = [
            ((trajectory_types[0], Types[0]), (trajectory_types[0], Types[1])),
            ((trajectory_types[1], Types[0]), (trajectory_types[1], Types[1])),
            ((trajectory_types[2], Types[0]), (trajectory_types[2], Types[1])),
            ((trajectory_types[3], Types[0]), (trajectory_types[3], Types[1])),
            ((trajectory_types[4], Types[0]), (trajectory_types[4], Types[1])),
        ]


        ylim = [[-0.0, 0.1], [-0.0, 0.2]]
        yticks = [np.arange(-10, 10, 0.05), np.arange(-10, 10, 0.05)]
        for i, gs in enumerate(Group_size):
            fig, ax = plt.subplots(1, 2, figsize=(15, 5), dpi=150, sharex=True)
            # ax[0].text(-0.1, -0.1, 'trajectory', ha='center', va='center', transform=ax[0].transAxes, fontsize=15)
            for j, df in enumerate(dfs):
                sns.boxplot(data=df[df['Group size'] == gs],
                            x='trajectory', y=labels[j], hue='Types', ax=ax[j])
                ax[j].set_ylim(ylim[j][0], ylim[j][1])

                mystat.t_test(ax=ax[j], pairs=pairs, data=df[df['Group size'] == gs],
                              x='trajectory', y=labels[j], hue='Types',  test='t-test_ind')


            os.makedirs(base_dir + 'test1', exist_ok=True)
            plt.savefig(base_dir + 'test1/' + 'Performance_comparison_' + gs + '.png')


        pairs = [
            ((Group_size[0], Types[0]), (Group_size[0], Types[1])),
            ((Group_size[1], Types[0]), (Group_size[1], Types[1])),
            ((Group_size[2], Types[0]), (Group_size[2], Types[1])),
        ]

        ylim = [[[-0.0, 0.04], [-0.0, 0.04], [-0.0, 0.05], [-0.0, 0.15], [0.04, 0.1]],
                [[-0.0, 0.05], [-0.0, 0.08], [-0.0, 0.06], [-0.0, 0.15], [0.08, 0.14]]]
        yticks = [np.arange(-10, 10, 0.05), np.arange(-10, 10, 0.05)]
        for i, df in enumerate(dfs):
            fig, ax = plt.subplots(5, 1, figsize=(5, 15), dpi=100, sharex=True)
            # ax[0].text(-0.1, -0.1, 'trajectory', ha='center', va='center', transform=ax[0].transAxes, fontsize=15)
            for j, tj in enumerate(trajectory_types):
                sns.boxplot(data=df[df['trajectory'] == tj],
                            x='Group size', y=labels[i], hue='Types', ax=ax[j])

                mystat.t_test(ax=ax[j], pairs=pairs, data=df[df['trajectory'] == tj],
                              x='Group size', y=labels[i], hue='Types',  test='t-test_ind')


                ax[j].set_ylim(ylim[i][j][0], ylim[i][j][1])
                ax[j].set_title(tj)

            os.makedirs(base_dir + 'test2', exist_ok=True)
            plt.savefig(base_dir + 'test2/' + 'Performance_comparison_' + labels_for_fig[i] + '.png')
            plt.savefig(base_dir + 'test2/' + 'Performance_comparison_' + labels_for_fig[i] + '.pdf')

        # plt.show()