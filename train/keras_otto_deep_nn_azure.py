#!/usr/bin/python

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

np.random.seed(1337) # for reproducibility

def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler
    
def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def add_converted_val(target_arr, add_arr):
    for elem in add_arr:
        target_arr.append(float(elem))
    
ftr_x = open('./azure_train_.csv', 'r')
te_mat = []
for trxline in ftr_x:
    splited = trxline.split(",")
    col = []
    col.append(float(splited[1]))
    add_converted_val(col, splited[4:79])
    te_mat.append(col)

print("Loading data...")
te_np_arr = np.array(te_mat)
X = te_np_arr[:,1:76]
labels = te_np_arr[:,0]
X, scaler = preprocess_data(X)
y, encoder = preprocess_labels(labels)

fts = open('./azure_test_.csv', 'r')
enroll_ids = []
ts_mat = []
for tsline in fts:
    splited = tsline.split(",")
    enroll_ids.append(splited[0])
    col = []
    add_converted_val(col, splited[3:78])
    ts_mat.append(col)

X_test = np.array(ts_mat)
X_test, _ = preprocess_data(X_test, scaler)

nb_classes = y.shape[1]
print(nb_classes, 'classes')

dims = X.shape[1]
print(dims, 'dims')

print("Building model...")

neuro_num = 512

model = Sequential()
model.add(Dense(dims, neuro_num, init='glorot_uniform'))
model.add(PReLU((neuro_num,)))
model.add(BatchNormalization((neuro_num,)))
model.add(Dropout(0.5))

model.add(Dense(neuro_num, neuro_num, init='glorot_uniform'))
model.add(PReLU((neuro_num,)))
model.add(BatchNormalization((neuro_num,)))
model.add(Dropout(0.5))

model.add(Dense(neuro_num, neuro_num, init='glorot_uniform'))
model.add(PReLU((neuro_num,)))
model.add(BatchNormalization((neuro_num,)))
model.add(Dropout(0.5))

model.add(Dense(neuro_num, nb_classes, init='glorot_uniform'))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer="adam")

print("Training model...")

model.fit(X, y, nb_epoch=200, batch_size=128, validation_split=0.15)

print("Generating submission...")

proba = model.predict_proba(X_test)

frslt = open('../test/keras_otto_azure_3.csv', 'w')
for idx in xrange(len(enroll_ids)):
    frslt.write(enroll_ids[idx] + "," + str(proba[idx][1]) + "\n")
