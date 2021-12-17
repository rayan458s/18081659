# Date: 17/12/2021
# Institution: University College London
# Dept: EEE
# Class: ELEC0134
# SN: 18081659
# Description: CNN Testing. See plots file for Results

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
import time
from sklearn.model_selection import ShuffleSplit
from Assignment.Functions import data_processing as dt

########################################## DATA PROCESSING ######################################################

# 1. Get the input data (matrix of image vectors) and output data (vector of labels)
image_size = (32,32)
print('\nLoading Initial Dataset.')
images_array, images_vectors, labels = dt.process_multiclass_data(one_hot=True, gray_2D=False, resize=image_size)      #process all the images into pixel vectors and get the labels
X, Y = images_array, labels

# 2. Split train an test dataset
print('\nSplitting Dataset.\n')
#Split train an test dataset
X_train,X_valid,Y_train,Y_valid=train_test_split(X,Y,test_size=0.2,random_state=3)

# # #Plot the features importances
# dt.plot_features_importances(X_train, Y_train, dim_reduction, n_features)


########################################## CNN ######################################################

# 3. Define the baseline CNN architecture
def baseline_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(4, activation='sigmoid'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

# 4. Train the CNN model and get accuracy and learning curve
CNN = baseline_model()	# define model
print('\nTraining CNN.\n')
start_time = time.time()        #start the counter
history = CNN.fit(X_train, Y_train, epochs=50, batch_size=64, validation_data=(X_valid, Y_valid), verbose=0)	# fit CNN model
elapsed_time = time.time() - start_time         #get the elapsed time since the counter was started
print(f"\nElapsed time to classify the data with CNN: {elapsed_time/60:.2f} minutes")
_, acc = CNN.evaluate(X_valid, Y_valid, verbose=0)	# evaluate model
print('> %.3f' % (acc * 100.0))
dt.plot_loss_accuracy(history)	#plot learning curves


