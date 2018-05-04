#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
import timeit
from datetime import datetime
import time

import numpy as np

def minmax_standardization(x, x_min, x_max):
    '''
    this function realizes data minmax standardization.
    x_nor = (x-x_min)/(x_max - x_min)
    '''
    for i in xrange(x.shape[0]):
        for j in xrange(x.shape[1]):
            if x[i, j] < x_min[j]:
                x[i, j] = x_min[j]
            if x[i, j] > x_max[j]:
                x[i, j] = x_max[j]
    x_nor = (x - x_min) / (x_max - x_min)
    return x_nor

def fea_standardization(x):
    '''
    this function realizes data feature standardization.
    The data is converted to a mean of 0 and variance data 1.
    '''
    x -= np.mean(x, axis=0)
    for i in range(x.shape[1]):
        if np.std(x[:, i]) != 0: x[:, i] /= np.std(x[:, i])
    return x

def fea_standardization2(x, x_mean, x_std):
    '''
    this function realizes data feature standardization.
    The data is converted to a mean of 0 and variance data 1.
    '''
    x -= x_mean
    x /= x_std
    return x

def fea_standardization_inverse(x, x_mean, x_std):
    x *= x_std
    x += x_mean
    return x

def get_usv(x):
    x = fea_standardization(x)
    cov = np.dot(x.T, x) / x.shape[0]
    x_u, x_s, x_v = np.linalg.svd(cov)
    return x_u, x_s

def zca_whitening(x, x_u, x_s, epsilon, x_mean, x_std):
    '''
    this function is aimed to reduce the relevance of data and noises.
    '''
    x -= np.mean(x, axis=0)
    xrot = np.dot(x, x_u)
    xpcawhite = xrot / np.sqrt(x_s + epsilon)
    xzcawhite = np.dot(xpcawhite, x_u.T)
    xzcawhite = fea_standardization_inverse(xzcawhite, x_mean, x_std)
    return xzcawhite

def ts_ms(ts):
    fault_timestamp = str(ts)
    fault_timestamp_1 = datetime.strptime(fault_timestamp,'%Y_%m_%d_%H:%M:%S:%f')
    fault_timestamp_2 = fault_timestamp_1.strftime('%Y-%m-%d %H:%M:%S:%f')
    millisecond =  int(time.mktime(fault_timestamp_1.timetuple()))
    return fault_timestamp_2, millisecond
