#!/usr/bin/python
import numpy as np
import scipy.sparse
import xgboost as xgb
### simple example
# load file from text file, also binary buffer generated by xgboostdtrain = xgb.DMatrix('../data/agaricus.txt.train')

def my_int(str1):
    if "NA" in str1:
        return 0
    else:
        return int(str1)

def my_float(str1):
    if "NA" in str1:
        return 0
    else:
        return float(str1)    

def add_converted_val(target_arr, add_arr):
    for elem in add_arr:
        target_arr.append(my_float(elem))
    
ftr_x = open('./azure_train_.csv', 'r')
te_mat = []
for trxline in ftr_x:
    splited = trxline.split(",")
    col = []
    col.append(my_float(splited[1]))
    add_converted_val(col, splited[4:79])
    te_mat.append(col)

te_np_arr = np.array(te_mat)
dtrain = xgb.DMatrix(te_np_arr[:,1:76], label=te_np_arr[:,0])
# specify parameters via map, definition are same as c++ version
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }

# specify validations set to watch performance
watchlist  = [(dtrain,'train')]
num_round = 10
bst = xgb.train(param, dtrain, num_round, watchlist)

fts = open('./azure_test_.csv', 'r')
enroll_ids = []
ts_mat = []
for tsline in fts:
    splited = tsline.split(",")
    enroll_ids.append(splited[0])
    col = []
    add_converted_val(col, splited[3:78])
    ts_mat.append(col)

ts_np_arr = np.array(ts_mat)
dtest = xgb.DMatrix(ts_np_arr)

# this is prediction
preds = bst.predict(dtest)

frslt = open('../test/xgb_azure_1_10.csv', 'w')
for idx in xrange(len(enroll_ids)):
    frslt.write(enroll_ids[idx] + "," + str(preds[idx]) + "\n")
