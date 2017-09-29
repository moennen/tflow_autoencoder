'''
Data Provider

- convert a list of several images into a TensorFlow records with optionnal modifications
- provide a accessor on the TensorFlow records with optionnal modifications

'''

import tensorflow as tf
import numpy as np
import OpenEXR, Imath, pprint, uuid, os, subprocess
from matplotlib import pyplot as plt

#-------------------------------------------------------------------------
# Print the data header of the given openEXR file


def printExrInfo(filename):
    data = OpenEXR.InputFile(filename)
    pprint.PrettyPrinter(indent=3).pprint(data.header())
    data.close()

#-------------------------------------------------------------------------
# Write a new openEXR image from an input one by selecting a set of channels
# channels : string specifying the channels to use


def writeExrChannels(input_filename, output_filename, channels):
    input_exr = OpenEXR.InputFile(input_filename)
    input_header = input_exr.header()
    input_channels = input_header['channels'].keys()
    # print input_channels
    output_channels = dict(zip(channels, input_exr.channels(channels)))
    output_header = input_header
    output_header['channels'] = {
        k: input_header['channels'][k] for k in channels}
    output_exr = OpenEXR.OutputFile(output_filename, output_header)
    output_exr.writePixels(output_channels)
    output_exr.close()
    input_exr.close()

#-------------------------------------------------------------------------
# Open an openEXR file and return a numpy array corresponding to the selected
# channels data


def readExrData(input_filename, input_channels):
    try: 
       input_exr = OpenEXR.InputFile(input_filename)
    except:
        # try to temporarly convert the input file into a valid exr if needed
        tmp_filename = '/tmp/' + str(uuid.uuid4()) + '.exr'
        convert_cmd = 'convert ' + input_filename + " " + tmp_filename
        subprocess.call(convert_cmd, shell=True)
        input_exr = OpenEXR.InputFile(tmp_filename)
        os.remove(tmp_filename)

    input_header = input_exr.header()
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    dw = input_header['dataWindow']
    size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    input_data = [np.reshape(np.fromstring(input_exr.channel(
        c, pt), dtype=np.float32), size) for c in input_channels]
    return np.stack(input_data, axis=2)

#-------------------------------------------------------------------------
# Create an records storing a list of [ exr_filename, channels_name]
#


def createFilenameRecords(img_chn_lst, records_name):
    """Encode a list of image name/channel name n-uplets into tfrecords."""

    # open the writer
    writer = tf.python_io.TFRecordWriter(records_name)

    # go through the list of image n-uplets
    with open(img_chn_lst, 'r') as img_chn_file:

        # go through the n-uplet
        for img_chn_line in img_chn_file:
            # initialize the feature
            feature = {}
            img_chn = img_chn_line.split()
            for idx in range(0, len(img_chn) - 1, 2):

                key = '{:02d}'.format(idx / 2)

                name_key = 'name_' + key
                chn_key = 'chn_' + key

                feature[name_key] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[img_chn[idx]]))
                feature[chn_key] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[img_chn[idx + 1]]))

            # write the current n-uplet
            example = tf.train.Example(
                features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    # close the writer
    writer.close()

#-------------------------------------------------------------------------
# Read and return the list of [exr_filename,channel_name] for a given
# filename record
#


def readSingleFilenameRecord(record, n_images):

    # construct the dictionnary model to be read
    feature = {}

    for img_idx in range(n_images):

        key = '{:02d}'.format(img_idx)

        name_key = 'name_' + key
        chn_key = 'chn_' + key

        feature[name_key] = tf.FixedLenFeature([], tf.string)
        feature[chn_key] = tf.FixedLenFeature([], tf.string)

    # parse the record
    features = tf.parse_single_example(record, features=feature)

    output_images = []
    for img_idx in range(n_images):

        key = '{:02d}'.format(img_idx)

        name_key = 'name_' + key
        chn_key = 'chn_' + key

        output_images.append([features[name_key], features[chn_key]])

    return output_images

#-------------------------------------------------------------------------
# Create and return a tensorflow dataset returning the exr tensors from
# a filename record


def createDatasetFromFilename(records_name, n_images):

    def extractSample(sample):
        return readSingleFilenameRecord(sample, n_images)

    def readSampleData(fn):
        return np.concatenate([readExrData(fn[i, 0], fn[i, 1]) for i in range(n_images)], 2)

    data = tf.contrib.data.TFRecordDataset(records_name)
    data = data.map(extractSample)
    data = data.map(lambda fn: tf.py_func(readSampleData, [fn], tf.float32))

    return data

#-------------------------------------------------------------------------
# Create an records storing a list of exr_data


def createDataRecords(img_chn_lst, records_name):
    """Encode a list of image/openEXR data n-uplets into tfrecords."""

    # open the writer
    writer = tf.python_io.TFRecordWriter(records_name)

    # go through the list of image n-uplets
    with open(img_chn_lst, 'r') as img_chn_file:

        # go through the n-uplet
        for img_chn_line in img_chn_file:
            # initialize the feature
            feature = {}
            img_chn = img_chn_line.split()
            for idx in range(0, len(img_chn) - 1, 2):
                img_name = img_chn[idx]
                img_chan = img_chn[idx + 1]

                img_data = readExrData(img_name, img_chan)

                key = '{:02d}'.format(idx / 2)

                data_key = 'data_' + key
                width_key = 'w_' + key
                height_key = 'h_' + key
                depth_key = 'd_' + key

                feature[data_key] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[img_data.tostring()]))
                feature[width_key] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[img_data.shape[0]]))
                feature[height_key] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[img_data.shape[1]]))
                feature[depth_key] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[img_data.shape[2]]))

            # write the current n-uplet
            example = tf.train.Example(
                features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    # close the writer
    writer.close()

#-------------------------------------------------------------------------
# Read and return the list of exr_data for a given data record
#


def readSingleDataRecord(record, n_images):

    # construct the dictionnary model to be read
    feature = {}

    for img_idx in range(n_images):

        key = '{:02d}'.format(img_idx)

        data_key = 'data_' + key
        width_key = 'w_' + key
        height_key = 'h_' + key
        depth_key = 'd_' + key

        feature[data_key] = tf.FixedLenFeature([], tf.string)
        feature[width_key] = tf.FixedLenFeature([], tf.int64)
        feature[height_key] = tf.FixedLenFeature([], tf.int64)
        feature[depth_key] = tf.FixedLenFeature([], tf.int64)

    # parse the record
    features = tf.parse_single_example(record, features=feature)

    output_images = []
    for img_idx in range(n_images):

        key = '{:02d}'.format(img_idx)

        data_key = 'data_' + key
        width_key = 'w_' + key
        height_key = 'h_' + key
        depth_key = 'd_' + key

        img = tf.decode_raw(features[data_key], tf.float32)

        img = tf.reshape(img, [tf.cast(features[width_key], tf.int32),
                               tf.cast(features[height_key], tf.int32),
                               tf.cast(features[depth_key], tf.int32)])

        output_images.append(img)

    return output_images

#-------------------------------------------------------------------------
# Create and return a tensorflow dataset returning the exr tensors from
# a exr data record


def createDatasetFromData(records_name, n_images):

    def readSampleData(record):
        return tf.concat(readSingleDataRecord(record, n_images),2)

    data = tf.contrib.data.TFRecordDataset(records_name)
    data = data.map(readSampleData)

    return data


#-------------------------------------------------------------------------
# Display a batch
#

def show(batch, img_depths):

    n_imgs = len(img_depths)

    if np.sum(img_depths) != batch.shape[3]:
       raise ValueError()

    batch_im = np.zeros(
        (batch.shape[0] * batch.shape[1], n_imgs * batch.shape[2], 3))

    for b in range(batch.shape[0]):

        n_offset = 0
        for n in range(n_imgs):

            im_d = img_depths[n]
            im = batch[b, :, :, n_offset:n_offset + im_d]

            if im_d > 3:
                gray = np.mean(im, axis=2)
                im = np.stack([gray, gray, gray], 2)

            batch_im[b * batch.shape[1]:(b + 1) * batch.shape[1],
                     n * batch.shape[2]:(n + 1) * batch.shape[2], 0:im_d] = im

            n_offset += im_d

    plt.imshow(batch_im)
    plt.show()