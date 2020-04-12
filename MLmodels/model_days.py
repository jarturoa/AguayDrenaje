# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 19:53:48 2020

@author: ArturoA
"""

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

import tensorflow as tf

import seaborn as sns

root_path = 'C:/Users/Arturo A/AguayDrenaje/preQData.csv'
# load data
def parse(x):
    return datetime.strptime(x, '%Y %m %d')

dataset = read_csv(root_path,  parse_dates = [['year', 'month', 'day']], index_col=0, date_parser=parse)

# drop the old index column named "No"
dataset.drop('No', axis=1, inplace=True)

# set date as index
dataset.index.name = 'date'

# check for nulls
#display('-'*100)
#display(dataset.isnull().any())


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

values = dataset.values

# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)

# drop columns we don't want to predict
reframed.drop(reframed.columns[[35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67]], axis=1, inplace=True)

# split into train and test sets
values = reframed.values
n_train_days = 365*5
train = values[:n_train_days, :]
test = values[n_train_days:, :]

# split into input and outputs
X_train, y_train = train[:, :-1], train[:, -1]
X_test, y_test = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))


# design network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(70, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(tf.keras.layers.Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
model.fit(X_train, y_train, epochs=50, batch_size=True, validation_data=(X_test, y_test), verbose=False, shuffle=False)

# make a prediction
y_pred = model.predict(X_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))

print(y_pred)