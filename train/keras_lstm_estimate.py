from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
#from keras.datasets import imdb

'''
    Train a LSTM on the IMDB sentiment classification task.

    The dataset is actually too small for LSTM to be of any advantage 
    compared to simpler, much faster methods such as TF-IDF+LogReg.

    Notes: 

    - RNNs are tricky. Choice of batch size is important, 
    choice of loss and optimizer is critical, etc. 
    Most configurations won't converge.

    - LSTM loss decrease during training can be quite different 
    from what you see with CNNs/MLPs/etc. It's more or less a sigmoid
    instead of an inverse exponential.

    GPU command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_lstm.py

    250s/epoch on GPU (GT 650M), vs. 400s/epoch on CPU (2.4Ghz Core i7).
'''

max_features=20000
maxlen = 100 # cut texts after this number of words (among top max_features most common words)
#batch_size = 16
batch_size = 1

"""
print("Loading data...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features, test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')
"""
ftr_x = open('./torch_input.csv', 'r')
X_train = []
for trxline in ftr_x:
    splited = trxline.split(",")
    X_train.insert(len(X_train), [int(splited[1]), int(splited[2]), int(splited[3]), int(splited[4]), int(splited[5]), \
                                int(splited[6]), int(splited[7]), int(splited[8])])

ftr_y = open('./truth_train.csv', 'r')
Y_train = []
for tryline in ftr_y:
    splited = tryline.split(",")
    Y_train.insert(len(Y_train), [int(splited[1])])

#print(X_train)
#print(Y_train)
#exit()

"""
X_train = [[1,4,7],[11,13,19]]
Y_train = [[1],[2]]
X_test = [[1,4,7],[11,13,19]]
Y_test = [[1],[2]]
"""
"""
print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
Y_train = sequence.pad_sequences(Y_train, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
"""
print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 8))
model.add(LSTM(8, 64)) # try using a GRU instead, for fun
model.add(Dropout(0.5))
model.add(Dense(64, 1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

print("Train...")
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=5, validation_split=0.1, show_accuracy=True)
score = model.evaluate(X_train, Y_train, batch_size=batch_size)
print('Test score:', score)

fts = open('../test/torch_input_test.csv', 'r')
enroll_ids = []
X_test = []
for tsline in fts:
    splited = tsline.split(",")
    enroll_ids.insert(len(enroll_ids), splited[0])
    X_test.insert(len(X_test), [int(splited[1]), int(splited[2]), int(splited[3]), int(splited[4]), int(splited[5]), \
                                int(splited[6]), int(splited[7]), int(splited[8])])
classes = model.predict_classes(X_test, batch_size=batch_size)

print(classes)
frslt = open('../test/resut_simple_nn_0621_3.csv', 'w')
for idx in xrange(len(enroll_ids)):
    frslt.write(enroll_ids[idx] + "," + str(classes[idx]))

#acc = np_utils.accuracy(classes, Y_test)
#print('Test accuracy:', acc)



