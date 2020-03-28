#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 22:03:22 2020

@author: tanmay
"""
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D


def get_model(features, timestep, nclass):
    
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
    out = Dense(nclass, activation = "softmax")(x)

    model = tf.keras.models.Model(inp_, out)
    opt = tf.keras.optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss = tf.keras.losses.sparse_categorical_crossentropy, metrics=['acc'])

    return model
