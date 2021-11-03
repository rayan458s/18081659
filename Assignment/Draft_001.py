import matplotlib.pyplot as plt
import numpy as np
from pandas import *
import pandas as pd
from sklearn.datasets import  load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import tensorflow as tf
values = tf.io.read_file('dataset/image/IMAGE_0000.jpg')


# Loading the CSV file
io_pairs = pandas.read_csv('dataset/label.csv')
labels = io_pairs.drop('file_name',axis=1)

print(labels)