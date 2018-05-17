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

from keras.models import Sequential, Model
from keras import regularizers
from keras.models import model_from_json
from keras import backend as K
from keras.optimizers import Adam, RMSprop
from keras.utils import np_utils
from keras.layers import Dense, LSTM, Masking, Input
from sklearn.grid_search import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from keras.callbacks import EarlyStopping

import pickle

def get_output_of_layer(model, x, k):
    get_layer_output = K.function([model.layers[0].input],
                                  [model.layers[k].output])
    layer_output = get_layer_output([x])[0]
    return layer_output

def label_pre(y_pred):
    y_pred = y_pred
    labels_list, labels_counts = np.unique(y_pred, return_counts=True)
    label_pre = labels_list[np.where(labels_counts==np.max(labels_counts))]
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

# 自己手动调特征提取编码器的超参数，分类器不考虑时序性
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
    autoencoder.compile(optimizer=optimizer, loss='msle', metrics=['mae'])
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

def train_SAEPFL(dataset, hidden_layer_list, sae_epoch, clf_epoch,
              sae_batch_size, clf_batch_size, sae_lr, clf_lr):
    # load data
    x_train,  y_train = dataset
    input_num = x_train.shape[1]
    yr_train = np_utils.to_categorical(y_train, 2)
    x_train, y_train, yr_train = shuffle(x_train, y_train, yr_train)
    overparams = dict()
    overparams['hidden_layer_list'] = hidden_layer_list
    overparams['sae_epoch'] = sae_epoch
    overparams['clf_epoch'] = clf_epoch
    overparams['sae_batch_size'] = sae_batch_size
    overparams['clf_batch_size'] = clf_batch_size
    overparams['sae_lr'] = sae_lr
    overparams['clf_lr'] = clf_lr
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
    classfier = Dense(2, activation='softmax')(hidden_layer)
    SAEPFL = Model(input_dim, output=classfier)
    optimizer = Adam(lr=clf_lr)
    SAEPFL.compile(optimizer=optimizer,
                loss='binary_crossentropy',
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
                            epoch=sae_epoch[i],
                            batch_size=sae_batch_size,
                            lr=sae_lr[i])
        SAEPFL.layers[i+1].set_weights(parameters)
        x = h
    # finetune_sae
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    history = SAEPFL.fit(x_train, yr_train,
                      nb_epoch=clf_epoch,
                      batch_size=clf_batch_size,
                      verbose=1,
                      callbacks=[early_stopping],
                      validation_split=0.3)
    valid_accuracy =  history.history['val_acc'][-1]
    return SAEPFL, valid_accuracy, overparams

def train_model1(trainset, limits_array, x_u, x_s, x_mean, epsilon=0.1,
                hidden_layer_list=[50, 20],sae_epoch=[50, 20], clf_epoch=100,
                sae_batch_size=1380, clf_batch_size=1380, sae_lr=[0.001, 0.001], clf_lr=0.001):
    start_time = timeit.default_timer()
    x_train = load_trainset(trainset[0], limits_array, x_u, x_s, x_mean, epsilon)
    y_train = trainset[1]
    data_train = trainset_trans(x_train, y_train)
    results = train_SAEPFL(dataset=data_train,
                        hidden_layer_list=hidden_layer_list,
                        sae_epoch=sae_epoch,
                        clf_epoch=clf_epoch,
                        sae_batch_size=sae_batch_size,
                        clf_batch_size=clf_batch_size,
                        sae_lr=sae_lr,
                        clf_lr=clf_lr)
    end_time = timeit.default_timer()
    train_time = end_time - start_time
    model = results[0]
    valid_accuracy = results[1]
    overparams = results[2]
    return model, valid_accuracy, overparams, train_time


# 自动寻优特征提取编码器的超参数，分类器不考虑时序性
def create_SAE2(input_num, hidden_layer_list, sae_lr):
    #create model
    model = Sequential()
    k = len(hidden_layer_list)
    #encoder layer
    model.add(Dense(hidden_layer_list[0], input_dim=input_num,
                    activation='relu', activity_regularizer=regularizers.l2(10e-10)))
    for i in range(k-1):
        model.add(Dense(hidden_layer_list[i+1], activation='relu',
                        activity_regularizer=regularizers.l2(10e-10)))
    #decoder layers
    if k == 1:
        model.add(Dense(input_num, activation='linear'))
    else:
        model.add(Dense(hidden_layer_list[-2], activation='relu'))
        for i in range(k-2):
            model.add(Dense(hidden_layer_list[k-i-3], activation='relu'))
        model.add(Dense(input_num, activation='linear'))
    # compile model
    adam = Adam(lr=sae_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='msle', metrics=['mae'])
    return model

def train_SAE2(dataset, hidden_layer_list, lr, batch_size, epochs):
    x = dataset
    #create model
    model = KerasRegressor(build_fn=create_SAE2, input_num=x.shape[1], verbose=0)
    # define the grid search parameters
    hidden_layer_list = hidden_layer_list
    sae_lr = lr
    sae_batch_size = batch_size
    sae_epoch = epochs
    param_grid = dict(hidden_layer_list=hidden_layer_list, sae_lr=sae_lr,
                      batch_size=sae_batch_size, nb_epoch=sae_epoch)
    grid = GridSearchCV(estimator=model, param_grid=param_grid,
                        scoring='r2', n_jobs=-1)
    grid_result = grid.fit(x, x)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    del x, hidden_layer_list, lr, batch_size, epochs, param_grid
    gc.collect()
    best_model = grid_result.best_estimator_.model
    k = len(best_model.layers)
    for i in range(k/2):
        best_model.pop()
    print len(best_model.layers)
    return best_model, grid_result.best_params_

def train_model2(trainset, limits_array, x_u, x_s, x_mean, epsilon=0.1,
                hidden_layer_list=[[228, 228], [228, 76], [228, 76, 38]],
                sae_epoch=[200, 500, 700],
                clf_epoch=500,
                sae_batch_size=[1380, 5520, 8820],
                clf_batch_size=1380,
                sae_lr=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
                clf_lr=0.003):
    start_time = timeit.default_timer()
    x_train = load_trainset(trainset[0], limits_array, x_u, x_s, x_mean, epsilon)
    dataset_train = trainset_trans(x_train, trainset[1])
    y_train = np_utils.to_categorical(dataset_train[1], 2)
    PFL_model, overparams = train_SAE2(dataset=dataset_train[0],
                        hidden_layer_list=hidden_layer_list,
                        epochs=sae_epoch,
                        batch_size=sae_batch_size,
                        lr=sae_lr)
    overparams['sae_batch_size'] = overparams.pop('batch_size')
    overparams['sae_epoch'] = overparams.pop('nb_epoch')
    print 'finish greedy traning'
    adam = Adam(lr=clf_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    PFL_model.add(Dense(2, init='uniform', activation='softmax'))
    PFL_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    history = PFL_model.fit(x=dataset_train[0], y=y_train, batch_size=clf_batch_size, epochs=clf_epoch,
                verbose=1, callbacks=[early_stopping], validation_split=0.2,
                shuffle=True)
    overparams['clf_lr'] = clf_lr
    overparams['clf_epoch'] = clf_epoch
    overparams['clf_batch_size'] = clf_batch_size
    valid_accuracy = history.history['val_acc'][-1]
    end_time = timeit.default_timer()
    train_time = (end_time - start_time) / 60.
    print overparams
    return PFL_model, valid_accuracy, overparams, train_time


# 自动寻优特征提取编码器的超参数，分类器考虑时序性
def train_PFLmodel(dataset, batch_size, epoch, lr):
    x_train2, y_train2 = dataset
    model = Sequential()
    model.add(Masking(mask_value=-1, input_shape=(1380, x_train2[0].shape[1])))
    model.add(LSTM(x_train2[0].shape[1], return_sequences=False, activation='softsign',
                    kernel_initializer='orthogonal'))
    model.add(Dense(2, init='uniform', activation='softmax'))
    rmsprop = RMSprop(lr=lr)
    model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(x=x_train2, y=y_train2, batch_size=batch_size, epochs=epoch,
                verbose=1, callbacks=[early_stopping], validation_split=0.2,
                shuffle=True)
    valid_accuracy = history.history['val_acc'][-1]
    return model, valid_accuracy

def missdata_implement(x, encoder):
    print encoder.layers[-1].output_shape[1]
    x2 = np.zeros((len(x), 1380, int(encoder.layers[-1].output_shape[1])))
    for i in range(len(x)):
        if x[i].shape[0] == 1380:
            x2[i] = get_output_of_layer(encoder, x[i], -1)
        else:
            x1 = get_output_of_layer(encoder, x[i], -1)
            x3 = -1 * np.ones((1380-x1.shape[0], x1.shape[1]))
            x2[i] = np.vstack((x1, x3))
    return x2

def train_model3(trainset, limits_array, x_u, x_s, x_mean, epsilon=0.1,
                hidden_layer_list=[[292, 292, 146]],
                sae_batch_size=[1380, 2760],
                sae_epoch=[200, 200, 200],
                sae_lr=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
                clf_epoch=1000,
                clf_lr=0.003,
                clf_batch_size=10):
    x_train = load_trainset(trainset[0], limits_array, x_u, x_s, x_mean, epsilon)
    y_train = trainset[1]
    xencd_train = trainset_trans(x_train, y_train)[0]
    start_time = timeit.default_timer()
    print 'start encoder traning'
    Encoder_model, overparams = train_SAE2(dataset=xencd_train,
                        hidden_layer_list=hidden_layer_list,
                        epochs=sae_epoch,
                        batch_size=sae_batch_size,
                        lr=sae_lr)
    overparams['sae_batch_size'] = overparams.pop('batch_size')
    overparams['sae_epoch'] = overparams.pop('nb_epoch')
    print 'finish greedy traning'
    print 'start classifier trainig'
    x_train2 = missdata_implement(x_train, Encoder_model)
    y_train2 = np_utils.to_categorical(y_train, 2)
    PFL_model, valid_accuracy = train_PFLmodel(dataset=(x_train2, y_train2),
                                batch_size=clf_batch_size,
                                epoch=clf_epoch,
                                lr=clf_lr)
    overparams['clf_lr'] = clf_lr
    overparams['clf_epoch'] = clf_epoch
    overparams['clf_batch_size'] = clf_batch_size
    end_time = timeit.default_timer()
    train_time = (end_time - start_time) / 60.
    return Encoder_model, PFL_model, valid_accuracy, overparams, train_time


# 自己手动调特征提取编码器的超参数，分类器考虑时序性
def train_SAE4(dataset, hidden_layer_list, epoch, batch_size, lr):
    # load data
    x_train = dataset
    input_num = x_train.shape[1]
    overparams = dict(hidden_layer_list=hidden_layer_list, sae_epoch=epoch,
                    sae_lr=lr, sae_batch_size=batch_size)
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
    SAE = Model(input_dim, output=hidden_layer)
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
                            epoch=epoch[i],
                            batch_size=batch_size,
                            lr=lr[i])
        SAE.layers[i+1].set_weights(parameters)
        x = h
    return SAE, overparams

def train_model4(trainset, limits_array, x_u, x_s, x_mean, epsilon=1e+24,
                hidden_layer_list=[292, 292, 146],
                sae_batch_size=1380,
                sae_epoch=[200, 200, 200],
                sae_lr=[0.001, 0.003, 0.01],
                clf_epoch=1000,
                clf_lr=0.003,
                clf_batch_size=10):
    x_train = load_trainset(trainset[0], limits_array, x_u, x_s, x_mean, epsilon)
    y_train = trainset[1]
    xencd_train = trainset_trans(x_train, y_train)[0]
    start_time = timeit.default_timer()
    print 'start encoder traning'
    Encoder_model, overparams = train_SAE4(dataset=xencd_train,
                        hidden_layer_list=hidden_layer_list,
                        epoch=sae_epoch,
                        batch_size=sae_batch_size,
                        lr=sae_lr)
    print 'finish greedy traning'
    print 'start classifier trainig'
    x_train2 = missdata_implement(x_train, Encoder_model)
    y_train2 = np_utils.to_categorical(y_train, 2)
    PFL_model, valid_accuracy= train_PFLmodel(dataset=(x_train2, y_train2),
                                batch_size=clf_batch_size,
                                epoch=clf_epoch,
                                lr=clf_lr)
    overparams['clf_lr'] = clf_lr
    overparams['clf_epoch'] = clf_epoch
    overparams['clf_batch_size'] = clf_batch_size
    end_time = timeit.default_timer()
    train_time = (end_time - start_time) / 60.
    return Encoder_model, PFL_model, valid_accuracy, overparams, train_time


def save_model(model, model_path, weights_path):
    model = model
    json_string = model.to_json()
    open(model_path, 'w').write(json_string)
    model.save_weights(weights_path)

def load_model(model_path, weights_path):
    model = model_from_json(open(model_path).read())
    model.load_weights(weights_path)
    return model

def save_params(params, params_path):
    output = open(params_path, 'wb')
    pickle.dump(params, output, -1)
    output.close()

def load_params(params_path):
    pkl_file = open(params_path, 'rb')
    params = pickle.load(pkl_file)
    pkl_file.close()
    return params

def test_model(model, testset, limits_array, x_u, x_s, x_mean, epsilon=0.1):
    x_test = load_trainset(testset[0], limits_array, x_u, x_s, x_mean, epsilon)
    y_test = testset[1]
    y_test_pred = np.zeros(len(y_test), dtype=int)
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

def test_model2(encoder, pfl, testset, limits_array, x_u, x_s, x_mean, epsilon=0.1):
    x_test = load_trainset(testset[0], limits_array, x_u, x_s, x_mean, epsilon)
    y_test = testset[1]
    x_test = missdata_implement(x_test, encoder)
    y_test_pred = np.zeros(len(y_test), dtype=int)
    start_time = timeit.default_timer()
    y_test_pred = np.argmax(pfl.predict(x_test), axis=1)
    end_time = timeit.default_timer()
    test_time = end_time - start_time
    accuracy = cal_acc(y_test, y_test_pred)
    y_test2 = []
    y_test_pred2 = []
    for i in range(len(y_test)):
        y_test2.append(y_test[i])
        y_test_pred2.append(y_test_pred[i])
        print y_test[i], y_test_pred[i]
    F1 = f1_score(y_test2, y_test_pred2, average='macro')
    return accuracy, F1, test_time

def location_model(model, dataset, limits_array, x_u, x_s, x_mean, epsilon=0.01):
    x = load_trainset(dataset, limits_array, x_u, x_s, x_mean, epsilon)
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

def location_model2(encoder, pfl, dataset, limits_array, x_u, x_s, x_mean, epsilon=0.01):
    x = load_trainset(dataset, limits_array, x_u, x_s, x_mean, epsilon)
    x = missdata_implement(x, encoder)
    result_list = []
    start_time = timeit.default_timer()
    yr_pred_proba = pfl.predict(x)
    y_pred = np.argmax(yr_pred_proba, axis=1)
    y_proba = np.max(yr_pred_proba, axis=1)
    end_time = timeit.default_timer()
    location_time = end_time - start_time
    result = [y_pred, y_proba, location_time]
    result_list.append(result)
    return result_list

def main():
    trainset = np.load('test_30071.npy')
    testset = np.load('test_30071.npy')
    #locaset = np.load('data/test60300.npy')
    limits_array = np.loadtxt('data_limits.csv', delimiter=',')
    x_u = np.loadtxt('x_u_lgt.csv', delimiter=',')
    x_s = np.loadtxt('x_s_lgt.csv', delimiter=',')
    x_mean = np.loadtxt('x_mean_lgt.csv', delimiter=',')

    hidden_layer_list = [304, 152, 76]
    sae_epoch = [10, 10, 10]
    clf_epoch = 1
    sae_batch_size = 1380
    clf_batch_size = 1380
    sae_lr = [0.01, 0.01, 0.01]
    clf_lr = 0.003
    epsilon = 0.1
    '''
    model, train_accuracy, params, train_time = train_model1(trainset, limits_array,
                                            x_u, x_s, x_mean, epsilon,
                                    hidden_layer_list=hidden_layer_list,
                                    sae_batch_size=sae_batch_size,
                                    clf_batch_size=clf_batch_size,
                                    sae_epoch=sae_epoch,
                                    clf_epoch=clf_epoch,
                                    sae_lr=sae_lr,
                                    clf_lr=clf_lr)


    model, train_accuracy, params, train_time = train_model2(trainset, limits_array,
                                            x_u, x_s, x_mean, epsilon,
                                    hidden_layer_list=hidden_layer_list,
                                    sae_batch_size=sae_batch_size,
                                    clf_batch_size=clf_batch_size,
                                    sae_epoch=sae_epoch,
                                    clf_epoch=clf_epoch,
                                    sae_lr=sae_lr,
                                    clf_lr=clf_lr)


    test_results = test_model(model, testset, limits_array, x_u, x_s, x_mean, epsilon)
    print test_results


    encoder, pfl, train_accuracy, overparams, train_time = train_model3(trainset,
                    limits_array, x_u, x_s, x_mean, epsilon,
                    hidden_layer_list=hidden_layer_list,
                    sae_batch_size=sae_batch_size,
                    sae_epoch=sae_epoch,
                    sae_lr=sae_lr,
                    clf_epoch=clf_epoch,
                    clf_lr=clf_lr,
                    clf_batch_size=clf_batch_size)
    '''
    encoder, pfl, train_accuracy, overparams, train_time = train_model4(trainset,
                    limits_array, x_u, x_s, x_mean, epsilon,
                    hidden_layer_list=hidden_layer_list,
                    sae_batch_size=sae_batch_size,
                    sae_epoch=sae_epoch,
                    sae_lr=sae_lr,
                    clf_epoch=clf_epoch,
                    clf_lr=clf_lr,
                    clf_batch_size=clf_batch_size)

    test_results = test_model2(encoder, pfl, testset, limits_array, x_u, x_s, x_mean, epsilon)
    print test_results

    #model_path = 'lgt30071_architechture.json'
    #weights_path = 'lgt30071_weights.h5'
    #save_model(model, model_path, weights_path)
    
    #model = load_model(model_path, weights_path)
    #location_results = location_model(model, locaset[0], limits_array, x_mean, x_std, x_u, x_s)
    #print location_results

if __name__ == '__main__':
    main()
