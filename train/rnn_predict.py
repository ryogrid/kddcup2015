import sys
import json
import numpy as np
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConnection
from pybrain.structure import RecurrentNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

def construct_network(input_len, hidden_nodes, is_elman=True):
    n = RecurrentNetwork()
    n.addInputModule(LinearLayer(input_len, name="i"))
    n.addModule(BiasUnit("b"))
    n.addModule(SigmoidLayer(hidden_nodes, name="h"))
    n.addOutputModule(LinearLayer(1, name="o"))

    n.addConnection(FullConnection(n["i"], n["h"]))
    n.addConnection(FullConnection(n["b"], n["h"]))
    n.addConnection(FullConnection(n["b"], n["o"]))
    n.addConnection(FullConnection(n["h"], n["o"]))

    if is_elman:
        # Elman (hidden->hidden)
        n.addRecurrentConnection(FullConnection(n["h"], n["h"]))
    else:
        # Jordan (out->hidden)
        n.addRecurrentConnection(FullConnection(n["o"], n["h"]))

    n.sortModules()
    n.reset()

    return n





"""
main
"""
hidden_nodes = 250
events_len = 500
is_elman = True


parameters = {}

# build rnn
rnn_net = construct_network(events_len, hidden_nodes, is_elman)

training_ds = []

ftr_x = open('./rnn_train.csv', 'r')

t_ds = SupervisedDataSet(events_len, 1)
for trxline in ftr_x:
    events_list = []
    splited = trxline.split(",")
    truth_val_list = (int(splited[0]),)
    rvsd = splited[2:]
    rvsd.reverse()
    for event_str in rvsd[:events_len]:
        events_list.append(int(event_str))

    while len(events_list) < events_len:
        events_list.append(0)
        
    t_ds.addSample(events_list, truth_val_list)

trainer = BackpropTrainer(rnn_net, **parameters)
trainer.setData(t_ds)
trainer.train()

del t_ds  # release memory

# predict
rnn_net.reset()
frslt = open('../test/rnn_result.csv', 'w')

fts = open('../test/rnn_test.csv', 'r')
for tsline in fts:
    splited = tsline.split(",")
    enroll_id_str = str(int(splited[0]))
    
    rvsd = splited[1:]
    rvsd.reverse()

    events_list = []    
    for event_str in rvsd[:events_len]:
        events_list.append(int(event_str))

    while len(events_list) < events_len:
        events_list.append(0)
        
    result = rnn_net.activate(events_list)
    frslt.write(enroll_id_str + "," + str(result[0]) + "\n")
