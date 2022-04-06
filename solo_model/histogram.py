#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

class histogram:
    def graph(self, data):
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.default'] = 'regular'
        plt.rcParams['xtick.top'] = 'True'
        plt.rcParams['ytick.right'] = 'True'
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        # plt.rcParams['axes.grid'] = 'True'
        plt.rcParams['axes.xmargin'] = '0'  # '.05'
        plt.rcParams['axes.ymargin'] = '0'
        plt.rcParams['savefig.facecolor'] = 'None'
        plt.rcParams['savefig.edgecolor'] = 'None'
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['pdf.fonttype'] = 42  # PDFにフォントを埋め込むためのパラメータ
        plt.xlim([-0.4, 0.4])  # x軸の範囲
        plt.ylim([0, 0.1])  # y軸の範囲

        # ey = npz["y"].reshape(len(npz["y"]), 1) - npz["LeftRaw"]


        yhist = np.histogram(data, bins=100, density=False)
        plt.plot(yhist[1][:-1], yhist[0] / np.sum(yhist[0]))
        print(yhist[0] / np.sum(yhist[0]))
        # plt.savefig('test.pdf')
        plt.show()

    def graph_out(self, data):
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['mathtext.default'] = 'regular'
        plt.rcParams['xtick.top'] = 'True'
        plt.rcParams['ytick.right'] = 'True'
        # plt.rcParams['axes.grid'] = 'True'
        plt.rcParams['xtick.direction'] = 'in'  # x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'  # y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['xtick.major.width'] = 1.0  # x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 1.0  # y軸主目盛り線の線幅
        plt.rcParams['font.size'] = 12  # フォントの大きさ
        plt.rcParams['axes.linewidth'] = 1.0  # 軸の線幅edge linewidth。囲みの太さ

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

        bin = 10000

        fig, sub = plt.subplots(2, 2, figsize=(10, 8), sharex=False)

        error = data['label_thm_r'] - data['pre_thm_r']
        yhist = np.histogram(error, bins=bin, density=False)
        sub[0, 0].plot(yhist[1][:-1], yhist[0] / np.sum(yhist[0]))
        sub[0, 0].set_ylabel('Normalized frequency')
        sub[0, 0].set_yticks(np.arange(0, 2, 0.04))
        sub[0, 0].set_ylim([0, 0.08])  # y軸の範囲
        sub[0, 0].set_xticks(np.arange(-2, 2, 0.2))
        sub[0, 0].set_xlim([-0.6, 0.6])  # x軸の範囲
        sub[0, 0].set_xlabel('roll thm')

        error = data['label_thm_p'] - data['pre_thm_p']
        yhist = np.histogram(error, bins=bin, density=False)
        sub[0, 1].plot(yhist[1][:-1], yhist[0] / np.sum(yhist[0]))
        sub[0, 1].set_ylabel('Normalized frequency')
        sub[0, 1].set_yticks(np.arange(0, 2, 0.04))
        sub[0, 1].set_ylim([0, 0.08])  # y軸の範囲
        sub[0, 1].set_xticks(np.arange(-2, 2, 0.2))
        sub[0, 1].set_xlim([-0.6, 0.6])  # x軸の範囲
        sub[0, 1].set_xlabel('pitch thm')

        error = data['label_text_r'] - data['pre_text_r']
        yhist = np.histogram(error, bins=bin, density=False)
        sub[1, 0].plot(yhist[1][:-1], yhist[0] / np.sum(yhist[0]))
        sub[1, 0].set_ylabel('Normalized frequency')
        sub[1, 0].set_yticks(np.arange(0, 2, 0.04))
        sub[1, 0].set_ylim([0, 0.08])  # y軸の範囲
        sub[1, 0].set_xticks(np.arange(-2, 2, 0.2))
        sub[1, 0].set_xlim([-0.6, 0.6])  # x軸の範囲
        sub[1, 0].set_xlabel('roll text')

        error = data['label_text_p'] - data['pre_text_p']
        yhist = np.histogram(error, bins=bin, density=False)
        sub[1, 1].plot(yhist[1][:-1], yhist[0] / np.sum(yhist[0]))
        sub[1, 1].set_ylabel('Normalized frequency')
        sub[1, 1].set_yticks(np.arange(0, 2, 0.04))
        sub[1, 1].set_ylim([0, 0.08])  # y軸の範囲
        sub[1, 1].set_xticks(np.arange(-2, 2, 0.2))
        sub[1, 1].set_xlim([-0.6, 0.6])  # x軸の範囲
        sub[1, 1].set_xlabel('pitch text')

        plt.tight_layout()
        # plt.savefig('hist_cfo.pdf')
        plt.show()