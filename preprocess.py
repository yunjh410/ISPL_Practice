import tensorflow as tf
import numpy as np
import glob
# use following commands when 'Segmentation fault' error occurs
import matplotlib
# import sys

# matplotlib.use('Qt4Agg')

from matplotlib import pyplot as plt
from PIL import Image

plt.ion()


def _bytes_feature(value):
    """ Returns a bytes_list from a string/byte"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """ Returns a float_list from a float/double """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """ Returns a int64_list from a bool/enum/int/uint """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _image_as_bytes(imagefile):
    image = np.array(Image.open(imagefile))
    image_raw = image.tostring()
    return image_raw


def make_example(img, lab):
    """ TODO: Return serialized Example from img, lab """

    feature = {'encoded': _bytes_feature(img),
               'label': _float_feature(lab),
               }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def write_tfrecord(imagedir, datadir):
    """ TODO: write a tfrecord file containing img-lab pairs
        imagedir: directory of input images
        datadir: directory of output a tfrecord file (or multiple tfrecord files) """
    val_size = 500
    filenames = glob.glob(imagedir + '/*/*.png')
    writer = tf.python_io.TFRecordWriter(datadir)
    temp = datadir.split('/')[-1]
    prev_cls = -1
    count = 1
    index = len(filenames)
    if temp == 'test.tfrecord':
        for i in range(index):
            filename = filenames[i]
            img_data = _image_as_bytes(filename)
            lab = int(filename.split('\\')[-2])
            example = make_example(img_data, lab)
            writer.write(example)
        writer.close()

    elif temp == 'train.tfrecord':
        for i in range(index):
            filename = filenames[i]
            img_data = _image_as_bytes(filename)
            lab = int(filename.split('\\')[-2])
            example = make_example(img_data, lab)

            if prev_cls != lab:
                count += 1
                if count == (val_size - 1):
                    count = 0
                    prev_cls += 1
            else:
                writer.write(example)
        writer.close()

    elif temp == 'val.tfrecord':
        for i in range(index):
            filename = filenames[i]
            img_data = _image_as_bytes(filename)
            lab = int(filename.split('\\')[-2])
            example = make_example(img_data, lab)

            if prev_cls != lab:
                writer.write(example)
                count += 1
                if count % val_size == 0:
                    prev_cls += 1
        writer.close()

def read_tfrecord(folder, batch=100, epoch=1):
    """ TODO: read tfrecord files in folder, Return shuffled mini-batch img,lab pairs
    img: float, 0.0~1.0 normalized
    lab: dim 10 one-hot vectors
    folder: directory where tfrecord files are stored in
    epoch: maximum epochs to train, default: 1 """

    filenames = glob.glob(folder)
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=epoch)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    key_to_feature = {'encoded': tf.FixedLenFeature([], tf.string, default_value=''),
                      'label': tf.FixedLenFeature([], tf.float32, default_value=0)
                      }
    features = tf.parse_single_example(serialized_example, features=key_to_feature)

    img = tf.decode_raw(features['encoded'], tf.uint8)
    img = tf.reshape(img, [28, 28, 1])
    img = tf.cast(img, tf.float32) / 255.0 - 0.5

    cls = tf.cast(features['label'], dtype=tf.int32)
    lab = tf.one_hot(cls, 10, dtype=tf.float32)

    min_after_dequeue = 50000
    img, lab = tf.train.shuffle_batch([img, lab], batch_size=100,
                                      capacity=min_after_dequeue + 3 * batch, num_threads=1,
                                      min_after_dequeue=min_after_dequeue)

    return img, lab
