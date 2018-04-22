#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys

import timeit
from datetime import datetime
import time

import numpy as np

from sae import *
from a_fault_diagnosis import diagnosis
from test import load_diagdata

import pickle

def PFLofFC(scada_data, fault_list, Model_dict, limits_array,
            x_mean, x_std, x_u, x_s, epsilon):
    #故障列表诊断
    pfl_list = diagnosis(fault_list)
    #buffer诊断
    sae_list = []
    for i in range(pfl_list):
        fault_code = pfl_list[i]
        if Model_dict.has_key(fault_code):
            model_path = Model_dict[fault_code][0]
            weights_path = Model_dict[fault_code][1]
            model = load_model(model_path, weights_path)
            result = location_model(model=model, dataset=scada_data,
                                    limits_array=limits_array,
                                    x_mean=x_mean, x_std=x_std,
                                    x_u=x_u, x_s=x_s, epsilon=epsilon)
            sae_list.append(result)
    return pfl_list, sae_list

def mod_freqfaul():
    code_list = ['20011', '30071', '30156', '50035', '60300', '70022',
                 '80072', '110101', '30011456']
    Model_dict = {}
    Model_dict['20011'] = ['model/20011_architechture.json', 'model/20011_weights.h5']
    Model_dict['30071'] = ['model/30071_architechture.json', 'model/30071_weights.h5']
    Model_dict['30156'] = ['model/30156_architechture.json', 'model/30156_weights.h5']
    Model_dict['50035'] = ['model/50035_architechture.json', 'model/50035_weights.h5']
    Model_dict['60300'] = ['model/60300_architechture.json', 'model/60300_weights.h5']
    Model_dict['70022'] = ['model/70022_architechture.json', 'model/70022_weights.h5']
    Model_dict['80072'] = ['model/80072_architechture.json', 'model/80072_weights.h5']
    Model_dict['110101'] = ['model/110101_architechture.json', 'model/110101_weights.h5']
    Model_dict['30011456'] = ['model/30011456_architechture.json', 'model/30011456_weights.h5']
    output = open('data/Model_dict.pkl', 'wb')
    pickle.dump(Model_dict, output)
    output.close()

def main():
    mod_freqfaul()
    '''
    pkl_file = open('data/Model_dict.pkl', 'rb')
    Model_dict = pickle.load(pkl_file)
    limits = np.loadtxt('data/data_limits.csv', delimiter=',')
    means = np.loadtxt('data/dataxnor_mean.csv', delimiter=',')
    stds = np.loadtxt('data/dataxnor_std.csv', delimiter=',')
    u = np.loadtxt('data/dataxnor_u.csv', delimiter=',')
    s = np.loadtxt('data/dataxnor_s.csv', delimiter=',')
    epsilon = 0.01
    fault_list = [910000, 301601, 30205, 30160, 910008, 800070, 800001, 30071]
    scada_data = np.load()
    results = PFLofFC(scada_data=scada_data, fault_list=fault_list,
                      Model_dict=Model_dict, limits_array=limits, x_mean=means,
                      x_std=stds, x_u=u, x_s=s, epsilon=epsilon)
    result_exp = results[0]
    result_data = results[1]
    print 'According to the expert experience, the principal fault is: '
    print result_exp
    if result_data == []:
        print 'there is no frequect fault'
    else:
        print 'Based on data analysis, there is a frequect fault: '
        print result_data
    '''
if __name__ == '__main__':
    main()
