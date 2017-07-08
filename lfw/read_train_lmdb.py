import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2

lmdb_env = lmdb.open('/home/tomoya/caffe/examples/handson/lfw_64x64_mc_train_lmdb')
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe_pb2.Datum()

data_num = 0

for key, value in lmdb_cursor:
    datum.ParseFromString(value)

    label = datum.label
    data = caffe.io.datum_to_array(datum)
    print str(key) +':'+str(label)
    data_num += 1

print "Train data num = " + str(data_num)
