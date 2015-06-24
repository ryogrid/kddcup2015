#!/usr/bin/python

import re
from sklearn.ensemble import RandomForestClassifier


ftr_x = open('./xgb_input_add_users.csv', 'r')
tr_input_arr = []
for trxline in ftr_x:
    splited = trxline.split(",")
    tr_input_arr.append([int(splited[1]), int(splited[2]), int(splited[3]), int(splited[4]), int(splited[5]), \
                                int(splited[6]), int(splited[7]), int(splited[8]), int(splited[9]), int(splited[10])])

tr_label_arr = []
ftr_y = open('./truth_train.csv', 'r')
for tryline in ftr_y:
    splited = tryline.split(",")
    tr_label_arr.append(int(splited[1]))

fts = open('../test/xgb_input_add_users.csv', 'r')
enroll_ids = []
ts_input_arr = []
for tsline in fts:
    splited = tsline.split(",")
    enroll_ids.insert(len(enroll_ids), splited[0])
    ts_input_arr.append([int(splited[1]), int(splited[2]), int(splited[3]), int(splited[4]), int(splited[5]), \
                                int(splited[6]), int(splited[7]), int(splited[8]), int(splited[9]), int(splited[10])])

model = RandomForestClassifier()
model.fit(tr_input_arr, tr_label_arr)
preds = model.predict(ts_input_arr)
    
frslt = open('../test/rf_result_1.csv', 'w')
for idx in xrange(len(enroll_ids)):
    frslt.write(enroll_ids[idx] + "," + str(preds[idx]) + "\n")
