
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


from Assignment.Functions import data_processing_T2 as dt
from Assignment.Functions import Classifiers as classf

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(8, input_dim=5, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(4, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


########################################## DATA PROCESSING ######################################################

images_vectors, labels = dt.process_data()      #process all the images into pixel vectors and get the labels
n_features = 5      #define the number of features to select/project from the pixels vectors
dim_reduction = "ANOVA"
if dim_reduction == "ANOVA":
	images_features = dt.get_ANOVA_features(n_features)     #select the desired number of feartures using ANOVA
elif dim_reduction == "PCA":
	images_features = dt.get_PCA_features(n_features)     #project the desired number of feartures using PCA
else:
	print('\nNot a valid dimensionality reduction technique\n')

#Split train an test dataset
X_train,X_test,Y_train,Y_test=train_test_split(images_features,labels,test_size=0.2,random_state=3)
print('\ntrain set: {}  | test set: {}\n'.format(round(((len(Y_train)*1.0)/len(images_features)),3),round((len(Y_test)*1.0)/len(labels),3)))

# # #Plot the features importances
# dt.plot_features_importances(X_train, Y_train, dim_reduction, n_features)


########################################## MLP CLASSIFIER ######################################################

model = baseline_model()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X_train, Y_train, epochs=150, batch_size=10, verbose=0)
# evaluate the keras model
_, accuracy = model.evaluate(X_test, Y_test)
print('Accuracy: %.2f' % (accuracy*100))