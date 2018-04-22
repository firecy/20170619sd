#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
import timeit
from datetime import datetime
import time

import numpy
from numpy import *

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
    x -= numpy.mean(x, axis=0)
    x /= numpy.std(x, axis=0)
    return x

def fea_standardization2(x, x_mean, x_std):
    '''
    this function realizes data feature standardization.
    The data is converted to a mean of 0 and variance data 1.
    '''
    x -= x_mean
    x /= x_std
    return x

def get_usv(x):
    x = fea_standardization(x)
    cov = numpy.dot(x.T, x) / x.shape[0]
    u, s, v = numpy.linalg.svd(cov)
    return u, s

def zca_whitening(x, epsilon):
    '''
    this function is aimed to reduce the relevance of data and noises.
    '''
    cov = numpy.dot(x.T, x) / x.shape[0]
    U, S, V = numpy.linalg.svd(cov)
    xrot = numpy.dot(x, U)
    xpcawhite = xrot / numpy.sqrt(S + epsilon)
    xzcawhite = numpy.dot(xpcawhite, U.T)
    return xzcawhite

def zca_whitening2(x, u, s, epsilon):
    '''
    this function is aimed to reduce the relevance of data and noises.
    '''
    x -= numpy.mean(x, axis=0)
    xrot = numpy.dot(x, u)
    xpcawhite = xrot / numpy.sqrt(s + epsilon)
    xzcawhite = numpy.dot(xpcawhite, u.T)
    return xzcawhite

def ts_ms(ts):
    fault_timestamp = str(ts)
    fault_timestamp_1 = datetime.strptime(fault_timestamp,'%Y_%m_%d_%H:%M:%S:%f')
    fault_timestamp_2 = fault_timestamp_1.strftime('%Y-%m-%d %H:%M:%S:%f')
    millisecond =  int(time.mktime(fault_timestamp_1.timetuple()))
    return fault_timestamp_2, millisecond
