import sys
import os
import caffe
from caffe.proto import caffe_pb2
import numpy

cifar_map = {
        0: "man",
        1: "woman"
}

os.system('convert ' + sys.argv[1] + ' -equalize test.jpg')

mean_blob = caffe_pb2.BlobProto()
with open('caffe/examples/handson/mean.binaryproto') as f:
        mean_blob.ParseFromString(f.read())

mean_array = numpy.asarray(mean_blob.data, dtype=numpy.float32).reshape(
        (mean_blob.channels, mean_blob.height, mean_blob.width)
)

classifier = caffe.Classifier(
        'caffe/examples/handson/handson_quick.prototxt',
        'caffe/examples/handson/handson_quick_iter_4000.caffemodel.h5',
        mean = mean_array,
        raw_scale = 255)

image = caffe.io.load_image('test.jpg')
predictions = classifier.predict([image], oversample= False)
answer = numpy.argmax(predictions)
print(predictions)
print(str(answer) + ":" + cifar_map[answer])
