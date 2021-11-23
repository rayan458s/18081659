import imageio
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.metrics import ConfusionMatrixDisplay
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

def image_array_to_vector(images_array, size=(512, 512)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
    image_vectors = []
    for i in range(len(images_array)):
        image = images_array[i]
        image_vector = cv2.resize(image, size).flatten()
        image_vectors.append(image_vector)
    image_vectors = np.array(image_vectors)
    return image_vectors


def SVM_Classifier(X_train,Y_train, X_test, kernel):
    svm_clf = SVC(kernel=kernel)
    svm_clf.fit(X_train,Y_train)
    y_pred = svm_clf.predict(X_test)
    return y_pred, svm_clf


########################################## DATA PROCESSING ######################################################

#Get images (inputs) array
images_array, class_name = load_images(img_folder)
images_array = np.array(images_array)

print("\nDataset shape: {}".format(images_array.shape))
a,b,c,d = images_array.shape
print("\nImage Size: {}x{}x{}".format(b,c,d))
print("\nNumber of Images: {}".format(a))

#Get labels (outputs) array
labels = load_labels(label_file)
#print(labels)
print("\nNumber of Labels: {}".format(len(labels)))

#Array to  Vectors
images_vectors = image_array_to_vector(images_array)
print("\nVector Size: {}".format(len(images_vectors[0])))



#Split train an test dataset
X_train,X_test,Y_train,Y_test=train_test_split(images_vectors,labels,test_size=0.2,random_state=3)
print('\ntrain set: {}  | test set: {}'.format(round(((len(Y_train)*1.0)/len(images_vectors)),3),round((len(Y_test)*1.0)/len(labels),3)))

########################################## SVM CLASSIFIER ######################################################

# #1. Test SVM model accuracy for different Kernel
# kernels = ["linear", "rbf"]
# accuracies_df = pd.DataFrame(list(range(1,3)), columns=["kernels"])
# accuracies = []
# for kernel in kernels:
#     Y_pred_SVM, svm_clf = SVM_Classifier(X_train, Y_train, X_test,kernel)
#     accuracies.append(round(accuracy_score(Y_test,Y_pred_SVM),2)*100)
#     print('\nAccuracy for kernel = {} computed'.format(kernel))
# accuracies_df['accuracies']=accuracies

# #2. Plot accuracy vs kernel
# fig, ax = plt.subplots()
# ax.scatter(accuracies_df['kernels'], accuracies_df['accuracies'])
# ax.set(title = 'Accuracy against type of Kernel',
#             ylabel='Accuracy (%)',xlabel='Kernel', ylim=[60, 100])
# plt.title('Accuracy against type of Kernel', weight = 'bold')
# plt.show()

# 3. Fit SVM model for linear Kernel and get accuracy score
Y_pred, svm_clf = SVM_Classifier(X_train, Y_train, X_test,"rbf")
print('\nSVM Accuracy Score on Test data: {}\n'.format(round(metrics.accuracy_score(Y_test,Y_pred),3)*100))

# 4. Plot non-normalized confusion matrix
titles_options = [
    ("SVM Confusion matrix, without normalization", None),
    #("SVM Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        svm_clf,
        X_test,
        Y_test,
        display_labels=["No Tumor", "Tumor"],
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)
    #print(title)
    #print(disp.confusion_matrix)
plt.show()