import os

import caffe
import lmdb

from caffe.proto import caffe_pb2
import numpy
import PIL.Image
import random

IMAGE_SIZE = 64

def make_datum(image, label):
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_SIZE,
        height=IMAGE_SIZE,
        label=label,
        data=numpy.rollaxis(numpy.asarray(image), 2).tostring())

train_lmdb = '/home/tomoya/caffe/examples/coil-100/coil-100_miniclass_train_lmdb'
test_lmdb  = '/home/tomoya/caffe/examples/coil-100/coil-100_miniclass_test_lmdb'

os.system('rm -rf  '+train_lmdb)
os.system('rm -rf  '+test_lmdb)

#== preparation for store to lmdb ============================================

print 'filepaths'

label_file_map = {}
filepaths = []
filepaths_train = []
filepaths_test = []

#train------------------------------------------------------------------------
for dirpath, _, filenames in os.walk('./miniclass/block/train_block'):
	for filename in filenames:
	        if filename.endswith(('.png', '.jpg')):
	            file = os.path.join(dirpath, filename)
	            filepaths_train.append(file)
	            label_file_map[file] = 0
	            print file
for dirpath, _, filenames in os.walk('./miniclass/can/train_can'):
	for filename in filenames:
	        if filename.endswith(('.png', '.jpg')):
	            file = os.path.join(dirpath, filename)
	            filepaths_train.append(file)
	            label_file_map[file] = 1
	            print file
for dirpath, _, filenames in os.walk('./miniclass/cup/train_cup'):
	for filename in filenames:
	        if filename.endswith(('.png', '.jpg')):
	            file = os.path.join(dirpath, filename)
	            filepaths_train.append(file)
	            label_file_map[file] = 2
	            print file
for dirpath, _, filenames in os.walk('./miniclass/toycar/train_toycar'):
	for filename in filenames:
	        if filename.endswith(('.png', '.jpg')):
	            file = os.path.join(dirpath, filename)
	            filepaths_train.append(file)
	            label_file_map[file] = 3
	            print file


#test---------------------------------------------------------------------------		
for dirpath, _, filenames in os.walk('./miniclass/block/test_block'):
	for filename in filenames:
	        if filename.endswith(('.png', '.jpg')):
	            file = os.path.join(dirpath, filename)
	            filepaths_test.append(file)
	            label_file_map[file] = 0
	            print file
for dirpath, _, filenames in os.walk('./miniclass/can/test_can'):
	for filename in filenames:
	        if filename.endswith(('.png', '.jpg')):
	            file = os.path.join(dirpath, filename)
	            filepaths_test.append(file)
	            label_file_map[file] = 1
	            print file
for dirpath, _, filenames in os.walk('./miniclass/cup/test_cup'):
	for filename in filenames:
	        if filename.endswith(('.png', '.jpg')):
	            file = os.path.join(dirpath, filename)
	            filepaths_test.append(file)
	            label_file_map[file] = 2
	            print file
for dirpath, _, filenames in os.walk('./miniclass/toycar/test_toycar'):
	for filename in filenames:
	        if filename.endswith(('.png', '.jpg')):
	            file = os.path.join(dirpath, filename)
	            filepaths_test.append(file)
	            label_file_map[file] = 3
	            print file
	

random.shuffle(filepaths_train)
random.shuffle(filepaths_test)
#random.shuffle(filepaths)




#=== store to lmdb ============================================================

#---train-----------------------------------------------------------------------

print 'train'

in_db = lmdb.open(train_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, file in enumerate(filepaths_train):
        image = PIL.Image.open(file)
        datum = make_datum(image, label_file_map[file])
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(in_idx) + ':' + file
in_db.close()


#---test-------------------------------------------------------------------------


print 'test'

in_db = lmdb.open(test_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, file in enumerate(filepaths_test):
        image = PIL.Image.open(file)
        datum = make_datum(image, label_file_map[file])
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(in_idx) + ':' + file
in_db.close()
