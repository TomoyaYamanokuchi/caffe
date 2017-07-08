import os

import caffe
import lmdb

from caffe.proto import caffe_pb2
import numpy
import PIL.Image
import random

IMAGE_SIZE = 64


print 'filepaths'

label_file_map = {}
filepaths = []
filepaths_train = []
filepaths_test = []
i = 0

for dirpath, _, filenames in os.walk('./man_64x64'):
    for idx, filename in enumerate(filenames):
      if i > 71:
	i=0
	break
      if i < 64:
	 if filename.endswith(('.png', '.jpg')):
            if idx % 3 != 0 :
                continue

            file = os.path.join(dirpath, filename)
            filepaths_train.append(file)
            label_file_map[file] = 0
            print file + "	" + str(idx) + "	" + str(i)
	    i += 1
      else:
            if filename.endswith(('.png', '.jpg')):
                if idx % 3 != 0 :
                    continue

                file = os.path.join(dirpath, filename)
                filepaths_test.append(file)
                label_file_map[file] = 0
                print file + "	" + str(idx) + "	" + str(i)
	        i += 1
for dirpath, _, filenames in os.walk('./woman_64x64'):
    for idx, filename in enumerate(filenames):
      if i > 71:
	i=0
	break
      if i < 64:
	 if filename.endswith(('.png', '.jpg')):
            file = os.path.join(dirpath, filename)
            filepaths_train.append(file)
            label_file_map[file] = 0
            print file + "	" + str(idx) + "	" + str(i)
	    i += 1
      else:
            if filename.endswith(('.png', '.jpg')):
                file = os.path.join(dirpath, filename)
                filepaths_test.append(file)
                label_file_map[file] = 0
                print file + "	" + str(idx) + "	" + str(i)
	        i += 1


random.shuffle(filepaths)

print 'train'

in_db = lmdb.open(train_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, file in enumerate(filepaths):
        if in_idx %  9 == 0 :
            continue

        image = PIL.Image.open(file)
        datum = make_datum(image, label_file_map[file])
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(in_idx) + ':' + file
in_db.close()

print 'test'

in_db = lmdb.open(test_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, file in enumerate(filepaths):
        if in_idx % 9 != 0 :
            continue

        image = PIL.Image.open(file)
        datum = make_datum(image, label_file_map[file])
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(in_idx) + ':' + file
in_db.close()
