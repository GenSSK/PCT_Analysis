#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def histogram(data, bins: int):
    data_flatten = data.reshape(-1, data.shape[-1])
    yhist_ = []
    for d in data_flatten:
        yhist, edges = calc_histogram(d, bins=bins)
        yhist_.append(yhist)
    yhist_array = np.array([_ for _ in yhist_])
    yhist = yhist_array.reshape(data.shape[:-1] + (bins,))
    #
    # if data.ndim == 1:
    #     yhist, edges = calc_histogram(data, bins=bins)
    # if data.ndim == 2:
    #     yhist = np.zeros((data.shape[0], bins))
    #     edges = np.zeros((data.shape[0], bins))
    #     for i in range(data.shape[0]):
    #         yhist[i, :], edges[i, :] = calc_histogram(data[i, :], bins=bins)
    # if data.ndim == 3:
    #     yhist = np.zeros((data.shape[0], data.shape[1], bins))
    #     edges = np.zeros((data.shape[0], data.shape[1], bins))
    #     for i in range(data.shape[0]):
    #         for j in range(data.shape[1]):
    #             yhist[i, j, :], edges[i, j, :] = calc_histogram(data[i, j, :], bins=bins)
    # if data.ndim == 4:
    #     yhist = np.zeros((data.shape[0], data.shape[1], data.shape[2], bins))
    #     edges = np.zeros((data.shape[0], data.shape[1], data.shape[2], bins))
    #     for i in range(data.shape[0]):
    #         for j in range(data.shape[1]):
    #             for k in range(data.shape[2]):
    #                 yhist[i, j, k, :], edges[i, j, k, :] = calc_histogram(data[i, j, k, :], bins=bins)
    return yhist, edges


def calc_histogram(data, bins: int):
    yhist, edges = np.histogram(data, bins=bins, density=True)
    return yhist, edges[:-1]


def get_histogram_normalize(data, bin: int):
    yhist, edges = np.histogram(data, bins=bin, density=True)
    w = np.diff(edges)
    yhist = yhist * w
    return yhist, edges[:-1]


def frequency(data, area):
    data_flatten = data.reshape(-1, data.shape[-1])
    freq_ = []
    for d in data_flatten:
        freq, edge = calc_frequency(d, area)
        freq_.append(freq)
    freq_array = np.array([_ for _ in freq_])
    freq = freq_array.reshape(data.shape[:-1] + (len(area) - 1,))

    return freq, edge


def calc_frequency(data, area):
    freq = np.zeros(len(area) - 1)
    edge = np.zeros(len(area) - 1)
    for i in range(len(area) - 1):
        count = np.where((data >= area[i]) & (data < area[i + 1]), 1, 0)
        freq[i] = np.sum(count)
        edge[i] = (area[i] + area[i + 1]) / 2

    return freq, edge
