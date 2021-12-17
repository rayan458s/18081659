# Date: 17/12/2021
# Institution: University College London
# Dept: EEE
# Class: ELEC0134
# SN: 18081659
# Description: MLP Testing. See plots file for Results

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import sys
import time
import numpy as np

from Assignment.Functions import data_processing as dt

########################################## DATA PROCESSING ######################################################

# 1. Get the input data (vectors of pixel intensities) and output data (vector of labels)
images_array, images_vectors, labels = dt.process_multiclass_data(resize=(32,32), gray_2D=True)      #process all the images into pixel vectors and get the labels

# 2. Transform inputs into feature vectors
dim_reduction = 'PCA'       #define the DR method
n_features = 10     #define the number of features to select/project from the pixels vectors
filepath = '/Users/rayan/PycharmProjects/AMLS/Assignment/Task2/Features/'+dim_reduction+'-'+str(n_features)+'.dat'  #get corresponding filepath
images_features = dt.get_features(filepath)

X, Y = images_vectors, labels

# 3. Split train an test dataset
X_train,X_valid,Y_train,Y_valid=train_test_split(X,Y,test_size=0.2,random_state=3)
print('\ntrain set: {}  | test set: {}\n'.format(round(((len(Y_train)*1.0)/len(X)),3),round((len(Y_valid)*1.0)/len(Y),3)))

# # #Plot the features importances
# dt.plot_features_importances(X_train, Y_train, dim_reduction, n_features)

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=1024, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

########################################## MLP CLASSIFIER ######################################################

MLP = baseline_model()
start_time = time.time()        #start the counter
history = MLP.fit(X_train, Y_train, epochs=150, batch_size=10, verbose=0) # fit the keras model on the dataset
elapsed_time = time.time() - start_time         #get the elapsed time since the counter was started
print(f"\nElapsed time to classify the data with CNN: {elapsed_time/60:.2f} minutes")
# evaluate the keras model
_, accuracy = MLP.evaluate(X_valid, Y_valid)
print('Accuracy: %.3f' % (accuracy*100))
dt.plot_loss_accuracy(history, False)	#plot learning curves
