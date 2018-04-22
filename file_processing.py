#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import shutil
import numpy as np
import pandas as pd

def mkdataset(path, path2):
    f = path
    with open(f, 'r') as f2:
        print 'finished 1'
        f2.readline()
        for line in f2.readlines():
            line = line.strip()
            line = line.split(',')
            oldfile = line[0]
            olddir = oldfile.split('/')[1]
            print oldfile
            label1 = line[1]
            label2 = line[3]
            if (label2 == '') and (label1 != ''):
                label = 'singlefault'
                newdir = os.path.join(path2, label, label1, olddir)
            elif (label2 != '') and (label1 != ''):
                label = 'multifault'
                newdir = os.path.join(path2, label, label1+'-'+label2, olddir)
            else:
                label = 'unknowfault'
                newdir = os.path.join(path2, label, olddir)
            mkdirs(newdir)
            shutil.copy(oldfile, newdir)

def plresult(path):
    path = os.path.expanduser(path)
    count = 0
    for (dirname, subdir, subfile) in os.walk(path):
        i = len(subfile)
        count += i
    return count

def mkdirs(newdir):
    path = newdir
    path.strip()
    #path = path.rstrip('//')
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

def calcu_labelnum(path):
    path = os.path.expanduser(path)
    fault_codes_list = dict()
    for (dirname, subdir, subfile) in os.walk(path):
        for f1 in subdir:
            subpath = os.path.join(dirname, f1)
            i = 0
            for (dirname2, subdir2, subfile2) in os.walk(subpath):
                for f2 in subfile2:
                    i += 1
            fault_codes_list[f1] = i
    fault_code_nums = sorted(fault_codes_list.items(), key=lambda e:e[0])
    for k in fault_code_nums:
       print k[0] + '\t' + str(k[1])
    #for key, value in fault_codes_list.items():
        #print key + '\t' + str(value)

def jymatch(path_j, path_y):
    result_j = pd.read_csv(path_j)
    result_y = pd.read_csv(path_y)
    coincide = set(result_j.iloc[:, 0]) & set(result_y.iloc[:, 0])
    print len(coincide)
    result_j = result_j.set_index('Unnamed: 0')
    result_y = result_y.set_index('Unnamed: 0')
    for i in coincide:
        print i
        result_y.loc[i, 'JModel_fault1'] = result_j.loc[i, 'fault1']
        result_y.loc[i, 'JModel_cof1'] = result_j.loc[i, 'cof1']
        result_y.loc[i, 'JModel_fault2'] = result_j.loc[i, 'fault2']
        result_y.loc[i, 'JModel_cof2'] = result_j.loc[i, 'cof2']
    path2 = '30G/coincide_result.csv'
    print result_y
    result_y.to_csv(path_y)


if __name__ == '__main__':

    #label_file = '30G/result200.csv'
    #data_file = 'new'
    #mkdataset(label_file, data_file)

    #path = '../data/data_new2_/train'
    #calcu_labelnum(path)

    #code_list = ['50035-60300', '50035-80072', '80072-30011456']
    #for i in code_list:
    #    path = os.path.join('new/multifault/', i)
    #    print i, plresult(path)

    path_y = '30G/diagnosis_result3.csv'
    path_j = '30G/result200.csv'
    jymatch(path_j, path_y)
