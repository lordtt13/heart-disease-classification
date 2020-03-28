#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 22:03:22 2020

@author: tanmay
"""
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D


def get_model(features, nclass, timestep = 1, act = "softmax"):
    '''
    Returns tf-keras model, stacks of Conv1D-Conv1D-MaxPool1D-Dropout, with a Dense Top.

    Parameters
    ----------
    features : int
        Total number of features in your data
    timestep : int
        Look back value in time, default = 1
    nclass : int
        Total number of unique labels in your data
    act : string
        Final activation, either "sigmoid" or "softmax" (default)

    Returns
    -------
    model : tf.keras.model
        Neural Network

    '''
    
    inp_ = Input(shape=(features, timestep))
    
    x = Convolution1D(16, kernel_size = 5, activation = "relu", padding="valid")(inp_)
    x = Convolution1D(16, kernel_size = 5, activation = "relu", padding="valid")(x)
    x = MaxPool1D(pool_size = 2)(x)
    x = Dropout(rate = 0.1)(x)
    
    x = Convolution1D(32, kernel_size = 3, activation = "relu", padding="valid")(x)
    x = Convolution1D(32, kernel_size = 3, activation = "relu", padding="valid")(x)
    x = MaxPool1D(pool_size = 2)(x)
    x = Dropout(rate = 0.1)(x)
    
    x = Convolution1D(32, kernel_size = 3, activation = "relu", padding="valid")(x)
    x = Convolution1D(32, kernel_size = 3, activation = "relu", padding="valid")(x)
    x = MaxPool1D(pool_size = 2)(x)
    x = Dropout(rate = 0.1)(x)
    
    x = Convolution1D(256, kernel_size = 3, activation = "relu", padding="valid")(x)
    x = Convolution1D(256, kernel_size = 3, activation = "relu", padding="valid")(x)
    
    x = GlobalMaxPool1D()(x)
    x = Dropout(rate = 0.25)(x)
    x = Dense(128, activation = "relu")(x)
    x = Dense(32, activation = "relu")(x)
    out = Dense(nclass, activation = act)(x)

    model = tf.keras.models.Model(inp_, out)

    return model
