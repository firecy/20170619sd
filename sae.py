#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys

import timeit
from datetime import datetime
import time

import numpy as np

#mport matplotlib.pyplot as plt
#import matplotlib.colors as colors
#import matplotlib.cm as cmx

from functions import *
from preprocessing import *

from sklearn.utils import shuffle
from sklearn.metrics import recall_score, f1_score

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import np_utils

import pickle

def get_output_of_layer(model, x, k):
    get_layer_output = K.function([model.layers[0].input],
                                  [model.layers[k].output])
    layer_output = get_layer_output([x])[0]
    return layer_output

def label_pre(y_pred):
    y_pred = y_pred
    labels_list, labels_counts = numpy.unique(y_pred, return_counts=True)
    label_pre = labels_list[numpy.where(labels_counts==numpy.max(labels_counts))]
    if len(label_pre) == 1:
        label_pre = int(label_pre)
    else:
        for i in range(len(label_pre)):
            label_pre[i] = int(label_pre[i])
        label_pre = label_pre[1]
    return label_pre

def label_proba(proba, y_pred):
    y_proba = np.mean(proba[:, y_pred])
    return y_proba

def cal_acc(y, y_pred):
    y = y
    y_pred = y_pred
    k = 0
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            k += 1
    print k, len(y)
    k = float(k)
    acc = k / float(len(y))
    return acc

def train_greedy_layer(dataset, hidden_unit, dense_activation, epoch, batch_size, lr):
    '''
    this function is aimed to get the weights of each layers.
    Input: dataset, hidden_unit, epoch, batch_size, lr
           dataset: the input data to autoencoder with (n x d) size, n
                    is the number of samples, d is the dimension(the
                    number of features) of input data.
           hidden_unit: the number of hidden units
           epoch: the number of iterations
           lr: learning rate of optimizer
    Output: weights, h
            weights: the weights of hidden layer
            h: new features extracted through the autoencoder
    '''
    # load data
    x = dataset
    input_num = x.shape[1]
    print input_num
    hidden_num = hidden_unit
    nb_epoch = epoch
    batch_size = batch_size
    optimizer = Adam(lr=lr)
    # construct autoencoder of each layer and initialize parameters
    '''
    input_dim: input layer, input_num is the number of features
    encoded: encoder layer, hidden_num is the number of encoder layer units,
             activation has many types with 'relu', 'sigmoid' and so on.
    decoded: decoder layer, input_num is the number of decoder layer units.
    autoencoder: the networks including input layer, encoder, and decoder.
    loss: loss function of the model
    '''
    input_dim = Input(shape=(input_num,))
    encoded = Dense(hidden_num, activation='relu',
                activity_regularizer=regularizers.l1(10e-15))(input_dim)
    decoded = Dense(input_num, activation=dense_activation)(encoded)
    autoencoder = Model(input=input_dim, output=decoded)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    autoencoder.compile(optimizer=optimizer, loss='mse')
    # train the model and get the weights of each layer
    history = autoencoder.fit(x, x,
                              nb_epoch=nb_epoch,
                              batch_size=batch_size,
                              verbose=1,
                              shuffle=True,
                              validation_split=0.3,
                              callbacks=[early_stopping])
    h = get_output_of_layer(autoencoder, x, 1)
    x_reconstruct = get_output_of_layer(autoencoder, x, 2)
    weights = autoencoder.layers[1].get_weights()
    return weights, h

def train_SAE(dataset, hidden_layer_list, pre_epoch, fine_epoch,
              batch_size, lr1, lr2):
    # load data
    x_train,  y_train = dataset[0]
    x_valid, y_valid = dataset[1]
    input_num = x_train.shape[1]
    nb_classes = len(set(y_train))
    yr_train = np_utils.to_categorical(y_train, nb_classes)
    yr_valid = np_utils.to_categorical(y_valid, nb_classes)
    x_train, y_train, yr_train = shuffle(x_train, y_train, yr_train)
    x_valid, y_valid, yr_valid = shuffle(x_valid, y_valid, yr_valid)
    pre_epoch = pre_epoch
    nb_epoch = fine_epoch
    batch_size = batch_size
    lr1 = lr1
    lr2 = lr2
    start_time = timeit.default_timer()
    # construct sae model an initialize parameters
    k = len(hidden_layer_list)
    input_dim = Input(shape=(input_num,))
    hidden_layer = Dense(hidden_layer_list[0], activation='relu',
                activity_regularizer=regularizers.l1(10e-15),
                name='1')(input_dim)
    for i in range(k-1):
        layer_name = str(i+2)
        hidden_layer = Dense(hidden_layer_list[i+1], activation='relu',
                        activity_regularizer=regularizers.l1(10e-15),
                        name=layer_name)(hidden_layer)
    classfier = Dense(nb_classes, activation='softmax')(hidden_layer)
    SAE = Model(input_dim, output=classfier)
    #optimizer = RMSprop(lr=lr2)
    optimizer = Adam(lr=lr2)
    SAE.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    # greedy layer training
    x = x_train
    for i in range(k):
        if i == 0:
            dense_activation = 'linear'
        else:
            dense_activation = 'relu'
        parameters, h = train_greedy_layer(dataset=x,
                            hidden_unit=hidden_layer_list[i],
                            dense_activation = dense_activation,
                            epoch=pre_epoch[i],
                            batch_size=batch_size,
                            lr=lr1[i])
        SAE.layers[i+1].set_weights(parameters)
        x = h
    # finetune_sae
    x2 = np.vstack((x_valid, x_train))
    y2 = np.vstack((yr_valid, yr_train))
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    history = SAE.fit(x2, y2,
                      nb_epoch=nb_epoch,
                      batch_size=batch_size,
                      verbose=1,
                      callbacks=[early_stopping],
                      #validation_data=(x_valid, yr_valid))
                      validation_split=0.3)
    end_time = timeit.default_timer()
    train_time = end_time - start_time
    valid_accuracy = history.history['val_acc'][-1]
    return SAE, valid_accuracy, train_time

def train_model(trainset, validset, limits_array, x_mean, x_std, u, s, epsilon=0.1,
                hidden_layer_list=[50, 20],pre_epoch=[50, 20], fine_epoch=100,
                batch_size=1380, lr1=[0.001, 0.001], lr2=0.001):
    x_train = load_trainset(trainset[0], limits_array, x_mean, x_std, u, s, epsilon)
    y_train = trainset[1]
    data_train = trainset_trans(x_train, y_train)
    x_valid = load_trainset(validset[0], limits_array, x_mean, x_std, u, s, epsilon)
    y_valid = validset[1]
    data_valid = trainset_trans(x_valid, y_valid)
    dataset = [data_train, data_valid]
    results = train_SAE(dataset=dataset,
                        hidden_layer_list=hidden_layer_list,
                        pre_epoch=pre_epoch,
                        fine_epoch=fine_epoch,
                        batch_size=batch_size,
                        lr1=lr1,
                        lr2=lr2)
    model = results[0]
    valid_accuracy = results[1]
    train_time = results[2]
    return model, valid_accuracy, train_time

def save_model(model, model_path, weights_path):
    model = model
    json_string = model.to_json()
    open(model_path, 'w').write(json_string)
    model.save_weights(weights_path)

def load_model(model_path, weights_path):
    model = model_from_json(open(model_path).read())
    model.load_weights(weights_path)
    return model

def test_model(model, testset, limits_array, x_mean, x_std, u, s, epsilon=0.01):
    x_test = load_trainset(testset[0], limits_array, x_mean, x_std, u, s, epsilon)
    y_test = testset[1]
    y_test_pred = np.zeros(len(y_test), dtype=int32)
    start_time = timeit.default_timer()
    for i in range(len(x_test)):
        yr_pred_proba = model.predict(x_test[i])
        yr_pred = np.argmax(yr_pred_proba, axis=1)
        y_test_pred[i] = label_pre(yr_pred)
    end_time = timeit.default_timer()
    test_time = end_time - start_time
    accuracy = cal_acc(y_test, y_test_pred)
    y_test2 = []
    y_test_pred2 = []
    for i in range(len(y_test)):
        y_test2.append(y_test[i])
        y_test_pred2.append(y_test_pred[i])
        print y_test[i], y_test_pred[i]
    #recall = recall_score(y_test2, y_test_pred2) # metrics for binmodel
    #return accuracy, recall, test_time
    F1 = f1_score(y_test2, y_test_pred2, average='macro')
    return accuracy, F1, test_time


def location_model(model, dataset, limits_array, x_mean, x_std, x_u, x_s, sfault_num=2, epsilon=0.01):
    x = load_trainset(dataset, limits_array, x_mean, x_std, x_u, x_s, epsilon)
    #y_pred = np.zeros(fault_num, dtype=int32)
    result_list = []
    start_time = timeit.default_timer()
    for i in range(len(x)):
        yr_pred_proba = model.predict(x[i])
        yr_pred = np.argmax(yr_pred_proba, axis=1)
        y_pred = label_pre(yr_pred)
        y_proba = label_proba(yr_pred_proba, y_pred)
        end_time = timeit.default_timer()
        location_time = end_time - start_time
        result = [y_pred, y_proba, location_time]
        result_list.append(result)
        #print dataset[1][i],'\t', y_pred,'\t', y_proba, '\t', location_time
    return result_list

def main():
    trainset = np.load('data/multidata_train.npy')
    validset = np.load('data/multidata_valid.npy')
    testset = np.load('data/multidata_test.npy')
    #locaset = np.load('data/test60300.npy')
    limits_array = np.loadtxt('data/data_limits.csv', delimiter=',')
    x_mean = np.load('data/dataxnor_mean4.npy')
    x_std = np.load('data/dataxnor_std4.npy')
    x_u = np.load('data/dataxnor_u4.npy')
    x_s = np.load('data/dataxnor_s4.npy')
    '''
    model, val_acc, train_time = train_model(trainset, validset, limits_array,
                                             x_mean, x_std, x_u, x_s,
                                             epsilon=0.01,
                                             hidden_layer_list=[200, 100, 50, 20],
                                             pre_epoch=[100, 100, 100, 100],
                                             fine_epoch=100,
                                             lr1=[0.001, 0.001, 0.001, 0.001], lr2=0.0003)
    '''

    #print val_acc, train_time
    model_path = 'data/multi_architechture_7.json'
    weights_path = 'data/multi_weights_7.h5'
    #save_model(model, model_path, weights_path)
    model_path1 = 'data/multi_architechture.json'
    weights_path1 = 'data/multi_weights.h5'
    model = load_model(model_path, weights_path)
    save_model(model, model_path1, weights_path1)

    test_results = test_model(model, testset, limits_array, x_mean, x_std, x_u, x_s)
    print test_results

    #location_results = location_model(model, locaset[0], limits_array, x_mean, x_std, x_u, x_s)
    #print location_results

if __name__ == '__main__':
    main()
