import imageio
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import csv
import cv2
import pandas as pd


IMG_WIDTH=200
IMG_HEIGHT=200
img_folder=r'/Users/rayan/PycharmProjects/AMLS/Assignment/dataset/image_small'
label_file = r'/Users/rayan/PycharmProjects/AMLS/Assignment/dataset/label_small.csv'


def plot_4_images():
    plt.figure(figsize=(20,20))
    for i in range(4):
        file = random.choice(os.listdir(img_folder))
        image_path= os.path.join(img_folder, file)
        img = np.array(imageio.imread(image_path))
        ax=plt.subplot(2,2,i+1)
        ax.title.set_text(file)
        plt.imshow(img)
    plt.show()


def load_images(img_folder):
    images_array=[]
    class_name=[]
    for file in os.listdir(img_folder):     #for all the files in dataset/image
        #print('Loading {}'.format(file))
        image_path = os.path.join(img_folder, file)      #join the path to the image filename
        image = np.array(imageio.imread(image_path))             #open and convert to numpy array
        #image= np.resize(image,(IMG_HEIGHT,IMG_WIDTH,3))        #rescale
        #image = image.astype('float32')                         #converto to float
        #image /= 255
        images_array.append(image)                    #final list with all the image arrays
        class_name.append(file)                             #image names
    return images_array , class_name


def load_labels(label_file_path):
    open_file = open(label_file_path)
    read_file = csv.reader(open_file, delimiter=',')
    labels = []
    for row in read_file:
        if row[1] == 'no_tumor':
           labels.append(0)
        else:
            labels.append(1)
    labels.pop(0)
    labels = np.array(labels)
    return labels

def image_to_feature_vector(images_array, size=(512, 512)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
    image_vectors = []
    for i in range(len(images_array)):
        image = images_array[i]
        image_vector = cv2.resize(image, size).flatten()
        image_vectors.append(image_vector)
    image_vectors = np.array(image_vectors)
    return image_vectors

def Decision_Tree_Classifier(X_train, Y_train, X_test, tree_params):
    clf = tree.DecisionTreeClassifier( **tree_params )      #Define the decision three classifier
    clf.fit(X_train,Y_train)
    Y_pred_DT =  clf.predict(X_test)
    return Y_pred_DT, clf

def Random_Forest_Classifier(X_train, Y_train, X_test):
    clf = RandomForestClassifier(n_estimators=100)   #Define the random forest classifier
    clf.fit(X_train,Y_train)        #Train the model using the training sets
    Y_pred_RF = clf.predict(X_test)      # prediction on test set
    return Y_pred_RF, clf

def visualise_tree(tree_to_print):
    plt.figure()
    tree.plot_tree(tree_to_print,
               filled = True,
              rounded=True);
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=800)
    plt.show()

#Get images (inputs) array
images_array, class_name = load_images(img_folder)
images_array = np.array(images_array)

print("\nDataset shape: {}".format(images_array.shape))
a,b,c,d = images_array.shape
print("\nNumber of Images: {}".format(a))
print("\nImage Size: {}x{}x{}".format(b,c,d))

#Get labels (outputs) array
labels = load_labels(label_file)
#print(labels)
print("\nNumber of Labels: {}".format(len(labels)))

#Array to Feature Vectors
image_vectors = image_to_feature_vector(images_array)
print("\nNumber of Images: {}".format(len(image_vectors)))
print("\nFeature Vector Size: {}".format(len(image_vectors[0])))

#Split train an test dataset
X_train,X_test,Y_train,Y_test=train_test_split(image_vectors,labels,test_size=0.2,random_state=3)
print('\ntrain set: {}  | test set: {}'.format(round(((len(Y_train)*1.0)/len(image_vectors)),3),round((len(Y_test)*1.0)/len(labels),3)))


########################################## DT CLASSIFIER ######################################################

#Fit DT model
tree_params={'criterion':'entropy'}
Y_pred_DT, clf_DT = Decision_Tree_Classifier(X_train, Y_train, X_test, tree_params)
DT_accuracy = round(accuracy_score(Y_test,Y_pred_DT),32)*100
print('Decision Tree Accuracy Score on Test data: {}%'.format(DT_accuracy))

#Decision Three visualisation
visualise_tree(clf_DT)
# Add  decision boundary +

# Hyperparameter Tuning
# tree_params = {'criterion': 'entropy', 'min_samples_split':50}
# Y_pred_DT = Decision_Tree_Classifier(X_train, Y_train, X_test, tree_params)
# print('Decision Tree Accuracy Score on Test data: {}%'.format(DT_accuracy))

# Add new three visualisation + new decision boundary +

########################################## RF CLASSIFIER ######################################################

Y_pred_RF, clf_RF = Random_Forest_Classifier(X_train, Y_train, X_test)
RF_accuracy = round(accuracy_score(Y_test, Y_pred_RF),2)*100
print("Random Forest Accuracy Score on Test data: {}%".format(RF_accuracy))

#Random Forest  visualisation
visualise_tree(clf_RF)

# Add decision boundary + remove unimportant features + retrain and re-visualise


