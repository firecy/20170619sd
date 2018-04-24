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

from keras.models import Sequtial
from keras import regularizers
from keras.models import model_from_json
from keras import backend as K
from keras.optimizers import Adam
from keras.utils import np_utils

import pickle

def create_SAEmodel1(input_num, hidden_layer_list, lr, dec_act):
    #create model
    model = Sequential()
    k = len(hidden_layer_list)
    #encoder layer
    model.add(Dense(hidden_layer_list[0], input_dim=input_num, init='uniform',
                    activation='relu', activity_regularizer=regularizers.l2(10e-10)))
    for i in range(k-1):
        model.add(Dense(hidden_layer_list[i+1], activation='relu', init='uniform',
                        activity_regularizer=regularizers.l2(10e-10)))
    #decoder layers
    if k == 1:
        model.add(Dense(input_num, activation=dec_act, init='uniform'))
    else:
        model.add(Dense(hidden_layer_list[-2], activation='relu', init='uniform'))
        for i in range(k-2):
            model.add(Dense(hidden_layer_list[k-i-3], activation='relu', init='uniform'))
        model.add(Dense(input_num, activation=dec_act, init='uniform'))
    # compile model
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='mse', metrics=['mae'])
    return model

def train_SAEmodel1(dataset, hidden_layer_list, lr, dec_act, batch_size, epochs):
    x = dataset
    #create model
    model = KerasRegressor(build_fn=create_SAEmodel1, input_num=x.shape[1], verbose=0)
    # define the grid search parameters
    hidden_layer_list = hidden_layer_list
    lr2 = lr2
    batch_size = batch_size
    epochs = epochs
    dec_act = dec_act
    param_grid = dict(hidden_layer_list=hidden_layer_list, lr=lr,
                      dec_act=dec_act, batch_size=batch_size, nb_epoch=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid,
                        scoring='r2', n_jobs=-1)
    grid_result = grid.fit(x, x)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    del x, hidden_layer_list, lr, batch_size, epochs, param_grid
    gc.collect()
    best_model = grid_result.best_estimator_.model
    k = best_model.layers
    for i in range(k/2.):
        best_model.pop()
    return best_model, grid_result.best_params_

def create_PFLmodel1(saemodel, output_num, lr, loss):
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model = saemodel
    #classfier layer
    model.add(Dense(output_num, init='uniform', activation='softmax'))
    model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
    return model

def train_PFLmodel1(dataset, saemodel, lr, batch_size, epochs, loss):
    #create_model
    x_train, y_train = dataset
    output_num = len(set(y_train))
    yr_train = np_utils.to_categorical(y_train, output_num)
    x_train, y_train, yr_train = shuffle(x_train, y_train, yr_train)
    model = KerasClassifier(build_fn=create_PFLmodel1, loss=loss,
                            output_num=output_num, saemodel=saemodel, verbose=0)
    #define the grid search parameters
    lr = lr
    batch_size = batch_size
    epochs = epochs
    param_grid = dict(lr=lr, batch_size=batch_size,
                      nb_epoch=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,
                        scoring='accuracy', cv=5)
    grid_result = grid.fit(x_train, yr_train)
    del dataset, saemodel, lr, batch_size, epochs, param_grid
    gc.collect()
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    return grid_result.best_estimator_.modle, grid_result.best_params_

def train_model1(trainset, limits_array, x_mean, x_std, x_u, x_s, epsilon=0.1,
                hidden_layer_list=[[228, 228], [228, 76], [228, 76, 38]],
                pre_epoch=[200, 500, 700],
                fine_epoch=[200, 500, 1000],
                batch_size=[1380, 5520, 8820],
                lr1=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
                lr2=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
                dec_act=['linear', 'sigmoid'],
                loss='binary_crossentropy'):
    x_train = load_trainset(trainset[0], limits_array, x_mean, x_std, x_u, x_s, epsilon)
    y_train = trainset[1]
    dataset_train = trainset_trans(x_train, y_train)
    overparams_lists = []
    start_time = timeit.default_timer()
    SAE_encoder, se_overparams = train_SAEmodel1(dataset=dataset_train[0],
                        hidden_layer_list=hidden_layer_list,
                        epochs=pre_epoch,
                        batch_size=batch_size,
                        lr=lr1,
                        dec_act=dec_act)
    overparams_lists.append(se_overparams)
    PFL_model, pfl_overparams = train_PFLmodel1(dataset=dataset_train,
                                saemodel=SAE_encoder,
                                lr=lr2,
                                batch_size=batch_size,
                                epochs=fine_epoch,
                                loss=loss)
    overparams_lists.append(pfl_overparams)
    end_time = timeit.default_timer()
    train_time = (end_time - start_time) / 60.
    return PFL_model, overparams_lis, train_time

def create_greedylayer(input_num, hidden_unit, lr, dec_act):
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model = Sequential()
    #encoder layer
    model.add(Dense(hidden_unit, input_dim=input_num, init='uniform',
                    activation='relu', activity_regularizer=regularizers.l2(10e-10)))
    #decoder layer
    model.add(Dense(input_num, activation=dec_act, init='uniform'))
    model.compile(optimizer=adam, loss='mse')
    return model

def train_greedylayer(trainset, dec_act, hidden_unit, lr, batch_size, epochs):
    x = trainset
    #create_model
    model = KerasRegressor(build_fn=create_greedylayer, input_num=x.shape[1],
                           dec_act=dec_act, verbose=0)
    #define the grid search parameters
    lr = lr
    batch_size = batch_size
    epochs = epochs
    hidden_unit = hidden_unit
    param_grid = dict(lr=lr, batch_size=batch_size,
                      nb_epoch=epoch, hidden_unit= hidden_unit)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,
                        scoring='r2', cv=5)
    grid_result = grid.fit(x, x)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    autoencoder = grid_result.best_estimator_.model
    weights = autoencoder.layers[0].get_weights()
    del grid, grid_result
    gc.collect()
    return weights, grid_result.best_params_

def train_model2(trainset, limits_array, x_mean, x_std, x_u, x_s, epsilon=0.1,
                 hidden_unit=[20, 40, 160, 320], hidden_num=3,
                 pre_epoch=[200, 500, 800],
                 fine_epoch=[300, 500, 1000],
                 batch_size=[1380, 2760, 5520, 11040],
                 lr1=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
                 lr2=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
                 loss='binary_crossentropy'):
    # load data
    x_train = load_trainset(trainset[0], limits_array, x_mean, x_std, x_u, x_s, epsilon)
    y_train = trainset[1]
    x_train,  y_train = trainset_trans(x_train, y_train)
    input_num = x_train.shape[1]
    nb_classes = len(set(y_train))
    yr_train = np_utils.to_categorical(y_train, nb_classes)
    x_train, y_train, yr_train = shuffle(x_train, y_train, yr_train)
    h_train = x_train
    pre_epoch = pre_epoch
    fine_epoch = fine_epoch
    batch_size = batch_size
    lr1 = lr1
    lr2 = lr2
    loss = loss
    # construct saepfl model
    SAEPFL_model = Sequential()
    k = hidden_num
    overparams_lists = []
    start_time = timeit.default_timer()
    # greedy layer training
    for i in range(k):
        if i == 0:
            dec_act = 'linear'
        else:
            dec_act = 'relu'
        weights, overparams = train_greedylayer(trainset=h_train,
                            hidden_unit=hidden_unit,
                            dec_act = dec_act,
                            epochs=pre_epoch,
                            batch_size=batch_size,
                            lr=lr1)
        if i == 0:
            SAEPFL_model.add(Dense(overparams['hidden_unit'],
                            input_dim=input_num, init='uniform',
                            activation='relu',
                            activity_regularizer=regularizers.l2(10e-10)))
        else:
            SAE_encoder.add(Dense(overparams['hidden_unit'],
                            init='uniform',
                            activation='relu',
                            activity_regularizer=regularizers.l2(10e-10)))
        SAE_encoder.layers[i].set_weights(weights)
        SAE_encoder.layers[i].trainable = False
        overparams_lists.append(overparams)
        h_train = get_layer_output(SAE_encoder, h_train, -1)
    # finetune_saepfl
    PFL_model, overparams = train_PFLmodel1(dataset=x_train,
                                saemodel=SAE_encoder,
                                lr=lr2,
                                batch_size=batch_size,
                                epochs=fine_epoch,
                                loss=loss)
    overparams_lists.append(overparams)
    end_time = timeit.default_timer()
    train_time = (end_time - start_time) / 60.
    return SAEPFL_model, overparams_lists, train_time

def create_clflayer(input_num, output_num, lr, loss):
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model = Sequential()
    #classfier layer
    model.add(Dense(output_num, input_dim=input_num, init='uniform',
                    activation='softmax'))
    model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
    return model

def train_clflayer(x, y, lr, batch_size, epoch, loss):
    #create_model
    input_num = x.shape[1]
    output_num = len(set(y))
    model = KerasClassifier(build_fn=create_clflayer, input_num=x.shape[1],
                           input_num=input_num, output_num=output_num,
                           loss=loss, verbose=0)
    #define the grid search parameters
    lr = lr
    batch_size = batch_size
    epoch = epoch
    param_grid = dict(lr=lr, batch_size=batch_size,
                      nb_epoch=epoch)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,
                        scoring='accuracy', cv=5)
    grid_result = grid.fit(x, y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    pflclassifer = grid_result.best_estimator_.model
    weights = pflclassifer.layers[0].get_weights()
    overparams = grid_result.best_params_
    return weights, overparams

def train_model3(trainset, limits_array, x_mean, x_std, x_u, x_s, epsilon=0.1,
                hidden_unit=[20, 40, 160, 320],
                pre_epoch=[200, 500, 800],
                fine_epoch=[300, 500, 1000],
                batch_size=[1380, 2760, 5520, 11040],
                lr1=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
                lr2=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
                model1_params=None,
                model1_hids=None,
                loss='binary_crossentropy'
                type=0,
                overparams_lists):
    # load data
    x_train = load_trainset(trainset[0], limits_array, x_mean, x_std, x_u, x_s, epsilon)
    y_train = trainset[1]
    x_train, y_train = trainset_trans(x_train, y_train)
    if model1_hids == None:
        input_num = x_train.shape[1]
        dec_act = 'linear'
    else:
        input_num = model1_params[0][0].shape[0]
        dec_act = 'relu'
    nb_classes = len(set(y_train))
    yr_train = np_utils.to_categorical(y_train, nb_classes)
    x_train, y_train, yr_train = shuffle(x_train, y_train, yr_train)
    pre_epoch = pre_epoch
    fine_epoch = fine_epoch
    batch_size = batch_size
    hidden_unit = hidden_unit
    lr1 = lr1
    lr2 = lr2
    loss=loss
    model1_params = model1_params
    model1_hids = model1_hids
    start_time = timeit.default_timer()
    PFL_model = Sequtial()
    # greedy layer training
    if type == 0:
        print "greedy layer training"
        parameters, overparams = train_greedylayer(trainset=x_train,
                                                hidden_unit=hidden_unit,
                                                epochs=pre_epoch,
                                                batch_size=batch_size,
                                                lr=lr1,
                                                dec_act=dec_act)
        overparams_lists.append(overparams)
        if model1_hids == None:
            PFL_model.add(Dense(overparams['hidden_unit'], input_dim=input_num,
                            init='uniform', activation='relu',
                            activity_regularizer=regularizers.l2(10e-10)))
        else:
            k = len(model1_hids)
            PFL_model.add(Dense(model1_hids[0], input_dim=input_num,
                            init='uniform', activation='relu',
                            activity_regularizer=regularizers.l2(10e-10)))
            PFL_model.layers[0].get_weights(model1_params[0])
            for i in range(k-1):
                PFL_model.add(Dense(model1_hids[i+1], init='uniform',
                activation='relu',
                activity_regularizer=regularizers.l2(10e-10)))
                PFL_model.layers[i+1].get_weights(model1_params[i+1])
            PFL_model.add(Dense(overparams['hidden_unit'],
                            init='uniform', activation='relu',
                            activity_regularizer=regularizers.l2(10e-10)))
            PFL_model.layers[-1].get_weights(parameters)
    if type == 1:
        print "finetune classfier training"
        parameters, overparams = train_clflayer(x=x_train, y=yr_train,
                                                lr=lr2,
                                                batch_size=batch_size,
                                                epoch=fine_epoch,
                                                loss=loss)
        overparams_lists.append(overparams)
        k = len(model1_hids)
        PFL_model.add(Dense(model1_hids[0], input_dim=input_num,
                        init='uniform', activation='relu',
                        activity_regularizer=regularizers.l2(10e-10)))
        PFL_model.layers[0].get_weights(model1_params[0])
        for i in range(k-1):
            PFL_model.add(Dense(model1_hids[i+1], init='uniform',
            activation='relu',
            activity_regularizer=regularizers.l2(10e-10)))
            PFL_model.layers[i+1].get_weights(model1_params[i+1])
        PFL_model.add(Dense(overparams['hidden_unit'], init='uniform',
                            activation='softmax',)
        PFL_model.layers[-1].get_weights(parameters)
    print "finish traning"
    end_time = timeit.default_timer()
    train_time = (end_time-start_time)/60.
    return PFL_model, overparams_lists, train_time

def train_model4(trainset_old, trainset_new, limits_array, x_mean, x_std, x_u,
    x_s, epsilon=0.1, old_model, overparams_list, fine_epoch, batch_size, lr, k):
    # load data
    start_time = timeit.default_timer()
    x_train_old = load_trainset(trainset_old[0], limits_array, x_mean, x_std, x_u, x_s, epsilon)
    y_train_old = trainset[1]
    x_train_old, y_train_old = trainset_trans(x_train_old, y_train_old)
    x_train_new = load_trainset(trainset_new[0], limits_array, x_mean, x_std, x_u, x_s, epsilon)
    y_train_new = trainset[1]
    x_train_new, y_train_new = trainset_trans(x_train_new, y_train_new)
    x_train = np.vstack((x_train_old, x_train_new))
    y_train = np.hstack((y_train_old, y_train_new))
    nb_classes = len(set(y_train))
    if nb_classes < 3: loss = 'binary_crossentropy'
    else: loss = 'categorical_crossentropy'
    yr_train = np_utils.to_categorical(y_train, nb_classes)
    x_train, y_train, yr_train = shuffle(x_train, y_train, yr_train)
    fine_epoch = fine_epoch
    batch_size = batch_size
    lr = lr
    new_model = old_model
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    new_model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    history0 = new_model.fit(x_train, y_train,
                  nb_epoch=fine_epoch,
                  batch_size=batch_size,
                  verbose=0,
                  callbacks=[early_stopping],
                  validation_split=0.3)
    for i in range(k-1):
        history = new_model.fit(x_train, y_train,
                      nb_epoch=fine_epoch,
                      batch_size=batch_size,
                      verbose=0,
                      callbacks=[early_stopping],
                      validation_split=0.3)
        if history0.history['metrics'][-1] < history.history['metrics'][-1]:
            PFL_model = new_model
    end_time = timeit.default_timer()
    train_time = (end_time - start_time) / 60.
    return PFL_model, train_time




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
