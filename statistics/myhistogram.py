#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def myhistogram(data, bin):
    yhist, edges = np.histogram(data, bins=bin, density=True)
    return yhist, edges[:-1]

def myhistogram_normalize(data, bin):
    yhist, edges = np.histogram(data, bins=bin, density=True)
    w = np.diff(edges)
    yhist = yhist * w
    return yhist, edges[:-1]