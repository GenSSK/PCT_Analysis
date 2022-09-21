#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def histogram(data, bin):
    yhist, edges = np.histogram(data, bins=bin, density=True)
    return yhist, edges[:-1]

def histogram_normalize(data, bin):
    yhist, edges = np.histogram(data, bins=bin, density=True)
    w = np.diff(edges)
    yhist = yhist * w
    return yhist, edges[:-1]