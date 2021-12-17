# Date: 17/12/2021
# Institution: University College London
# Dept: EEE
# Class: ELEC0134
# SN: 18081659


Task 1: Build a classifier to identify whether there is a tumor in the MRI images.

Task 2: Build a classifier to identify the type of tumor in each MRI image (meningioma tumor, glioma tumor, pituitary tumor or no tumor).



In this Assignment Folder, you will find 5 files: 

1. dataset:

This file contains the 2 datasets provided for this assignment: the initial dataset (image and labels.csv) and the test dataset (image_test and label_test.csv)

2. Functions:

This file contains 3 pythons programs:
- Classifiers.py contains methods to fit all the non-deep learning classifiers we have tested and get the predictions on validation/test data
- data_processing.py contains all the methods to process the data and some plotting functions (features importance, random forest visualisation, etc). Every model test import this package to call it methods
- extract_features.py is a program used to extract the PCA and ANOVA features and having them stored inn Features files of both Task 1 and 2 Folders

3. Task1:
- Bagging1: KNN with Bagging Testing. See plots file for results
- Boosting1: Decision Tree with Boosting Testing. See plots file for results.
- Decision-Tree1: Decision Tree Testing. See plots file for Results
- Features: Saved ANOVA (5 AND 10) and PCA (5 and 10) features for binary classification
- Final_Model_1: Final Task 1 model implementation. See plots file for Test Results
- KNN1: KNN Testing. See plots file for Results
- Random_Forest_1: Random Forest Testing. See plots file for Results
- SVM1: SVM Testing. See plots file for Results

4. Task2:
- CNN: CNN Testing. See plots file for Results
- Features: Saved ANOVA (5 AND 10) and PCA (5 and 10) features for multi-class classification
- Final_Model_2: Final Task 2 model implementation. See plots file for Test Results
- MLP: MLP Testing. See plots file for Results
- Random_Forest2: Random Forest Testing. See plots file for Results


#To Test the Final Models 
1. Go to Functions->data_processing.py and correct the path to your personal files in lines 30, 31, 233, 246, 306, 329.
2. Go to Task1->Final_Model_1.py and correct the path to your personal files in line 32.
3. Go to Task2->Final_Model_2.py and correct the path to your personal files in line 32.
4. You can now run the files Final_Model_1.py, Final_Model_2.py and see the models results on the SciView and Command window

# Necessary Packages:
-	Keras-Preprocessing
-	dlib
-	imageio
-	importlib-metadata
-	imutils
-	ipykernel
-	ipython
-	ipython-genutils
-	keras
-	matplotlib
-	numpy
-	opencv-python
-	packaging
-	pandas
-	pip
-	plotly
-	scikit-image
-	scikit-learn
-	scipy
-	setuptools
-	sklearn
-	tensorboard
-	tensorboard-data-server
-	tensorboard-plugin-wit
-	tensorflow
-	tensorflow-estimator
