import matplotlib.pyplot as plt
import numpy as np
from pandas import *
import pandas as pd
from sklearn.datasets import  load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import tensorflow as tf

import tensorflow as tf

def decode_image(filename, channels):
    value = tf.io.read_file(filename)
    decoded_image = tf.image.decode_png(value, channels=channels)
    return decoded_image

def get_dataset(image_paths, channels):
    filename_tensor = tf.constant(image_paths)
    print(filename_tensor)
    dataset = tf.data.Dataset.from_tensor_slices(filename_tensor)

    def _map_fn(filename):
        decode_images = decode_image(filename, channels=channels)
        return decode_images

    map_dataset = dataset.map(_map_fn) # we use the map method: allow to apply the function _map_fn to all the
    # elements of dataset
    return map_dataset

def get_image_data(image_paths, channels):
    dataset = get_dataset(image_paths, channels)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    next_image = iterator.get_next()

    return next_image


print(get_image_data('dataset/image/IMAGE_0000.jpg', 0))

# Loading the CSV file
io_pairs = pandas.read_csv('dataset/label.csv')
labels = io_pairs.drop('file_name',axis=1)

print(labels)