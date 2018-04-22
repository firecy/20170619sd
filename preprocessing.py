#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
import timeit
from datetime import datetime

import numpy as np
from functions import *

def data_preprocessing(data, limits_array, x_mean, x_std, u, s, epsilon):
    min_array = limits_array[0, :]
    max_array = limits_array[1, :]
    datanor_ins = minmax_standardization(data, min_array, max_array)
    datafea_ins = fea_standardization2(datanor_ins, x_mean, x_std)
    datazca_ins = zca_whitening2(datafea_ins, u, s, epsilon)
    return datazca_ins

def data_preprocessing2(data, limits_array, epsilon):
    min_array = limits_array[0, :]
    max_array = limits_array[1, :]
    datanor_ins = minmax_standardization(data, min_array, max_array)
    windnor_ins, spnor_ins, varnor_ins = data_segment(datanor_ins)
    #windfea_ins = fea_standardization(windnor_ins)
    #spfea_ins = fea_standardization(spnor_ins)
    #varfea_ins = fea_standardization(varnor_ins)
    #varzca_ins = zca_whitening(varnor_ins, epsilon)
    #print windfea_ins
    #print spnor_ins
    #print varnor_ins
    #varzca_ins = zca_whitening(varfea_ins, epsilon)
    datanew_ins = np.hstack((windnor_ins, spnor_ins, varnor_ins))
    return datanew_ins

def load_trainset(dataset, limits_array, x_mean, x_std, u, s, epsilon):
    dataset_pre = []
    for i in range(len(dataset)):
        data_pre = data_preprocessing(dataset[i], limits_array, x_mean,
                                      x_std, u, s, epsilon)
        #data_pre = data_preprocessing2(dataset[i], limits_array, epsilon)
        dataset_pre.append(data_pre)
    return dataset_pre

def load_trainset2(dataset, limits_array, epsilon):
    dataset_pre = []
    data_pre = data_preprocessing(dataset, limits_array, epsilon)
    dataset_pre.append(data_pre)
    return dataset_pre

def trainset_trans(trainset, labels):
    x = trainset
    y = labels
    x0 = x[0]
    y0 = y[0] * np.ones(x0.shape[0], dtype=int32)
    for i in range(len(x)):
        if  (i != 0):
            x1 = x[i]
            y1 = y[i] * np.ones(x1.shape[0], dtype=int32)
            x0 = np.vstack((x0, x1))
            y0 = np.hstack((y0, y1))
    return x0, y0

def data_segment(dataset):
    weather = dataset[:, : 5]
    wind2 = np.vstack((0.4*weather[:,0]+0.4*weather[:,2]+0.2*weather[:,3],
                            0.5*weather[:,1]+0.5*weather[:,4]))
    wind2 = transpose(wind2)
    setpoint = np.vstack((dataset[:, 16], dataset[:, 54], dataset[:, 63],
                            dataset[:, 71], dataset[:, 72]))
    setpoint = transpose(setpoint)
    variables = np.hstack((dataset[:, 5:16], dataset[:, 17:54], dataset[:, 55:63],
                            dataset[:, 64:71]))
    return wind2, setpoint, variables
