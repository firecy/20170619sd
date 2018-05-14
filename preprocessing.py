#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
import timeit
from datetime import datetime

import numpy as np
from functions import *
import gc

def get_limits():
    WeatherLimits = np.array([[0., -216., 0., 0., -216.],
                                 [38.4, 216., 38.4, 38.4, 216.]]) #天气变量上下限
    PitchSystemLimits = np.zeros((2, 27)) #变桨系统变量上下限
    PitchSystemLimits[:, 0: 10] = np.array([[-15.1, 0., -1.9, -1.9, -12., -51., 0., -15.1, -51, -15.1],
                                               [106.1, 552., 20.9, 20.9, 12., 81., 552., 106.1, 81., 106.1]])
    PitchSystemLimits[:, 10: 20] = np.array([[-12., 0., -15.1, -15.1, -51., -25.2, -51., -75., -1.9, 0.],
                                                [12., 91., 106.1, 106.1, 81., 116.2, 81., 225., 20.9, 552.]])
    PitchSystemLimits[:, 20: 27] = np.array([[-51., -51., -15.1, -75., -75., -51., -12.],
                                                [81., 81., 106.1, 225., 225., 81., 12.]])
    EgineRoomLimits = np.array([[-0.6, -0.6, -0.1],
                                   [0.6, 0.6, 0.6]]) #机舱振动变量上下限
    ControlVariableLimits = np.array([[-63., -63., -63., -63., -63.],
                                         [93., 93., 93., 93., 93.]]) #控制因素变量上下限
    GearBoxLimits = np.array([[-23., -10., -10, -23., -10.],
                                 [133., 110., 110., 133., 110.]]) #齿轮箱变量上下限
    GeneratorLimits = np.array([[-23., -250., -14.5, -14.5, -23., -14.5, -14.5, -12.],
                                   [133., 2750., 159.5, 159.5, 133., 159.5, 159.5, 132.]])
                                   #发电机变量上下限
    YawSystemLimits = np.array([[-25., 0., -900., -216.],
                                   [275, 120., 900., 216.]]) #偏航系统变量上下限
    PowerLimits = np.array([[-250., -250., -250., 225., 225., 225., -1., 0., -1.2, 0., -1250., -305.],
                               [2750., 2750., 2750., 525., 525., 525., 1., 100., 1.2, 1., 1750., 2755.]])
                                #电网变量上下限
    ConverterLimits = np.array([[-12.5, 0.],
                                   [77.5, 100.]]) #变流器变量上下限
    FanBrakeLimits = np.array([[0., 0.],
                                  [70., 9.]]) #风机制动变量上下限
    limits = [WeatherLimits, PitchSystemLimits, EgineRoomLimits, ControlVariableLimits,
                      GearBoxLimits, GeneratorLimits, YawSystemLimits, PowerLimits,
                      ConverterLimits, FanBrakeLimits]
    limits2 = SetData(limits)
    return limits2

def get_mean_std_usv(xset, limits_array):
    min_array = limits_array[0, :]
    max_array = limits_array[1, :]
    xset = np.vstack((xset))
    xnor = minmax_standardization(xset, min_array, max_array)
    x_mean = np.mean(xnor, axis=0)
    x_std = np.std(xnor, axis=0)
    x_u, x_s = get_usv(xnor)
    del xset, xnor, min_array, max_array, limits_array
    gc.collect()
    return x_mean, x_std, x_u, x_s

def data_preprocessing(data, limits_array, x_mean, x_std, x_u, x_s, epsilon):
    min_array = limits_array[0, :]
    max_array = limits_array[1, :]
    datanor_ins = minmax_standardization(data, min_array, max_array)
    datafea_ins = fea_standardization2(datanor_ins, x_mean, x_std)
    datazca_ins = zca_whitening(datafea_ins, x_u, x_s, epsilon, x_mean, x_std)
    datazca_ins = fea_standardization_inverse(datazca_ins, x_mean, x_std)
    del min_array, max_array, limits_array, datanor_ins, datafea_ins
    gc.collect()
    return datazca_ins

def load_trainset(xset, limits_array, x_mean, x_std, x_u, x_s, epsilon):
    xset_pre = []
    for i in range(len(xset)):
        x_pre = data_preprocessing(data=xset[i], limits_array=limits_array,
                                      x_mean=x_mean, x_std=x_std,
                                      x_u=x_u, x_s=x_s, epsilon=epsilon)
        xset_pre.append(x_pre)
    gc.collect()
    return xset_pre

def load_trainset2(dataset, limits_array, epsilon):
    dataset_pre = []
    data_pre = data_preprocessing(dataset, limits_array, epsilon)
    dataset_pre.append(data_pre)
    return dataset_pre

def trainset_trans(trainset, labels):
    x = trainset
    y = labels
    x0 = x[0]
    y0 = y[0] * np.ones(x0.shape[0], dtype=int)
    for i in range(len(x)):
        if  (i != 0):
            x1 = x[i]
            y1 = y[i] * np.ones(x1.shape[0], dtype=int)
            x0 = np.vstack((x0, x1))
            y0 = np.hstack((y0, y1))
    del x, y, x1, y1
    gc.collect()
    return x0, y0

if __name__ == '__main__':
    xset = np.load('data_all.npy')
    print len(xset[1])
    #limits_array = np.loadtxt('data_limits.csv', delimiter=',')
    #x_mean, x_std, x_u, x_s = get_mean_std_usv(xset, limits_array)
    #np.save('x_mean_lgt', x_mean)
    #np.save('x_std_lgt', x_std)
    #np.save('x_u_lgt', x_u)
    #np.save('x_s_lgt', x_s)
    #print 'finish'
