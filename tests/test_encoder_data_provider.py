'''
TESTS : OpenEXR utilities functions
'''

import argparse
import os
import sys
sys.path.append(os.path.abspath(
    '/mnt/p4/avila/moennen_wkspce/tflow_autoencoder'))
import exrChnRecords

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import exrChnRecords 

from tensorflow.contrib.data import Dataset, Iterator

def randomCrop(sample,img_depths,img_size):
   return tf.random_crop(sample,[img_size[0],img_size[1],np.sum(img_depths)])

img_depths = [ 3, 3, 3 ]
img_size = [ 128, 128 ]
img_batch  = 4


#exrChnRecords.createDataRecords('data/fi_hollywood_100.lst', 'data/lst.rec')
exrChnRecords.createFilenameRecords('data/fi_hollywood_100.lst', 'data/lst2.rec')

# create TensorFlow Dataset objects
#tr_data = exrChnRecords.createDatasetFromData(['data/lst.rec'],len(img_depths))
tr_data = exrChnRecords.createDatasetFromFilename(['data/lst2.rec'],len(img_depths))
tr_data = tr_data.map(lambda s: randomCrop(s,img_depths,img_size) )
tr_data = tr_data.batch(img_batch)


# create TensorFlow Iterator object
print tr_data.output_types
print tr_data.output_shapes
iterator = Iterator.from_structure(tr_data.output_types,
                                   tr_data.output_shapes)
next_element = iterator.get_next()

# create two initialization ops to switch between the datasets
training_init_op = iterator.make_initializer(tr_data)

with tf.Session() as sess:

        # initialize the iterator on the training data
    sess.run(training_init_op)

    # get each element of the training dataset until the end is reached
    while True:
        try:
            elem = sess.run(next_element)
            print elem.shape
            exrChnRecords.show(elem,img_depths)
            
        except tf.errors.OutOfRangeError:
            print("End of training dataset.")
            break
