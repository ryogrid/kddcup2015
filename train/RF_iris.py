#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from sklearn.ensemble import RandomForestClassifier

training_label = []
training_data = []
predict_label = []
predict_data = []
num = 0

file = open( './iris.scale' , 'r' )
for line in file :
    line = line.rstrip()
    node = []
    label = re.search( r'^(.*?)\s', line ).group(1)
    for i in range(1,5) :
        try :
            pattern = r'%s' % ( str( i ) + ':(.*?\s)' )
            match = re.search( pattern, line ).group(1)
            if match is None :
                node.append(0)
            else:
                node.append(match)
        except AttributeError :
            node.append(0)
        continue
    if num % 2 == 0 :
        training_data.append( node )
        training_label.append( label )
    else :
        predict_data.append( node )
        predict_label.append( label )
    num = num + 1
        
predict_data = training_data
model = RandomForestClassifier()
model.fit(training_data, training_label)
output = model.predict(predict_data)

for i in range( 0,len( output ) ) :
    str = "ok" if( int( predict_label[i] ) == int( output[i] ) ) else "miss"
    print "predict %s id = %d" % ( str, i )
