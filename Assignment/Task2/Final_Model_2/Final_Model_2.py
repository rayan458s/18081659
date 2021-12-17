from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
import time
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import ShuffleSplit

from Assignment.Functions import data_processing as dt

# 1. Get the input data (matrix of image vectors and output data (vector of labels)
image_size = (32,32)
print('\nLoading Initial Dataset.')
images_array, images_vectors, labels = dt.process_multiclass_data(one_hot=True, gray_2D=False, resize=image_size)      #process all the images into pixel vectors and get the labels
X, Y = images_array, labels

# 2. Define the baseline CNN architecture
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

CNN = KerasClassifier(build_fn=baseline_model, batch_size=64, epochs=100, verbose=0)   # define model

# # 3. Cross Validation
# start_time = time.time()         #start the counter
# cv_scores = cross_val_score(CNN, X, Y, cv=5)       #run 5 iterations of k-fold cross validation
# elapsed_time = time.time() - start_time         #get the elapsed time since the counter was started
# print(f"\nElapsed time to cross validate (5 Splits) the data with CNN: {elapsed_time:.2f} seconds")
# print("%0.2f accuracy with a standard deviation of %0.2f" % (cv_scores.mean(), cv_scores.std()))

# 4. Split train an validation dataset
X_train,X_valid,Y_train,Y_valid=train_test_split(X,Y,test_size=0.2,random_state=3)     #split the initial dataset into 80% training, 20% validation
print('\ntrain set: {}  | test set: {}\n'.format(round(((len(Y_train)*1.0)/len(X)),3),round((len(Y_valid)*1.0)/len(Y),3)))

# 6. Parameters Tuning
# batch_size = [10, 20, 40, 60, 80]
# epochs = [10, 50, 100]
# optimizer = ['SGD', 'Adam']
# learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
# activation = ['softmax', 'relu', 'sigmoid']
# neurons = [1, 5, 10, 15, 20, 25, 30]
# parameters = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer, learn_rate=learn_rate, activation=activation) #define the values to change in all parameters
#
# grid_clf = GridSearchCV(estimator=CNN, param_grid=parameters, refit='accuracy')  #define the grid search classifier instance
# grid_clf.fit(X_train, Y_train)           #fit the data with all the parameters configurations
#
# tuning_scores = pd.DataFrame(grid_clf.cv_results_)       #store all the scrores in a dataframe
# tuning_scores = tuning_scores[['rank_test_accuracy', 'param_batch_size', 'param_epochs', 'mean_test_accuracy']]  # leave only the accuracy scores
# tuning_scores['mean_test_accuracy'] = tuning_scores['mean_test_accuracy'].round(4)*100
# tuning_scores = tuning_scores.sort_values(by=['rank_test_accuracy'])    #rank the configurations by accuracy scores
# tuning_scores.to_csv('Tuning_Results_by_Accuracy.csv')  #save the configuration in a csv file

# 6. Train  CNN with best hyperparameters
batch_size = 64
epochs = 100
# optimizer = 'SGD'
# learn_rate = 0.01
# activation = 'relu'
# neurons = 60
CNN = baseline_model()	# define model
print('\nTraining CNN with best parameters.\n')
start_time = time.time()        #start the counter
history = CNN.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, Y_valid), verbose=0)	# fit CNN model
elapsed_time = time.time() - start_time         #get the elapsed time since the counter was started
print(f"\nElapsed time to classify the data with CNN: {elapsed_time/60:.2f} minutes")

# 7. Get Accuracy and plot learning curve for validation
_, acc = CNN.evaluate(X_valid, Y_valid, verbose=0)	# evaluate model
print(f'\nCNN Accuracy Score on Validation data after tuning: {round(acc,3)*100}%')
print('> %.3f' % (acc * 100.0))
dt.plot_loss_accuracy(history)	#plot learning curves

# 9. Get Prediction and Performance scores on Final Test dataset
images_array, images_vectors_test, labels_test = dt.process_multiclass_test_data(one_hot=True, gray_2D=False, resize=image_size)     #process test data
X_test, Y_test = images_array, labels_test
_, acc = CNN.evaluate(X_test, Y_test, verbose=0)     # predictions on test set
print(f'\nCNN Accuracy Score on Test data after tuning: {round(acc,3)*100}%')

