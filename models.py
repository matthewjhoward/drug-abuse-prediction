import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

kl = keras.layers


def baseNN(feature_length, n_classes, regress):
    inputs = kl.Input(shape=(feature_length,))
    
    x = kl.Dense(128, activation='relu')(inputs)
    x = kl.Dense(128, activation='relu')(x)
    x = kl.Dense(256, activation='relu')(x)
    x = kl.Dense(512, activation='relu')(x)
    x = kl.Dense(512, activation='relu')(x)
    x = kl.Dense(256, activation='relu')(x)

    x = kl.Dropout(0.25)(x)
    if regress:
        act = 'linear'
    else: 
        act = 'sigmoid'
    outputs = kl.Dense(n_classes, activation=act)(x)
    model = keras.Model(inputs = inputs, outputs=outputs)
    return model

def seqNN(feature_length, num_years, n_classes, regress):
    inputs = kl.Input(shape=(num_years,feature_length,))

    x = kl.TimeDistributed(kl.Dense(128, activation='relu', 
    kernel_initializer='he_normal'))(inputs)
    x = kl.TimeDistributed(kl.Dense(128, activation='relu'))(x)
    x = kl.TimeDistributed(kl.Dense(256, activation='relu'))(x)
    x = kl.TimeDistributed(kl.Dense(512, activation='relu'))(x)
    x = kl.TimeDistributed(kl.Dense(512, activation='relu'))(x)
    x = kl.TimeDistributed(kl.Dense(256, activation='relu'))(x)


    x = kl.LSTM(125, return_sequences=True)(x)
    x = kl.LSTM(125, return_sequences=True)(x)

    x = kl.Flatten()(x)
    x = kl.Dropout(0.25)(x)

    outputs = kl.Dense(n_classes, activation='sigmoid')(x)
    model = keras.Model(inputs = inputs, outputs=outputs)
    return model



