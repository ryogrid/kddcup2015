#!/usr/bin/python

from __future__ import absolute_import

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

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

X_list = []
labels_list = []
# setup training data
for i in xrange(20000):
    # [1,2] => [0]
    X_list.append([1,2])
    labels_list.append(0)
    # [2,1] => [1]
    X_list.append([2,1])
    labels_list.append(1)

print("Loading data...")
X = np.array(X_list)
labels = np.array(labels_list)
X, scaler = preprocess_data(X)
Y, encoder = preprocess_labels(labels)

np.random.seed(1337) # for reproducibility

# input for predection
X_test_list = [[1,2],[2,1]]
X_test = np.array(X_test_list)
X_test, _ = preprocess_data(X_test_list, scaler)

nb_classes = Y.shape[1]
print(nb_classes, 'classes')

dims = X.shape[1]
print(dims, 'dims')

print("Building model...")

neuro_num = 16

# setup deep NN
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
model.fit(X, Y, nb_epoch=20, batch_size=128, validation_split=0.15)

print("Prediction...")
proba = model.predict_proba(X_test)

# predicted result
print("probability of [label=0 label=1]")
print("  input: [1,2] => " + str(proba[0]))
print("  input: [2,1] => " + str(proba[1]))
