import imageio
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def KNN_Classifier(X_train, Y_train, X_test,k):
    KNN_clf = KNeighborsClassifier(n_neighbors=k)     #Create KNN object with a K coefficient
    KNN_clf.fit(X_train, Y_train)       # Fit KNN model
    Y_pred_KNN = KNN_clf.predict(X_test)
    return Y_pred_KNN, KNN_clf


def Decision_Tree_Classifier(X_train, Y_train, X_test, tree_params):
    dt_clf = tree.DecisionTreeClassifier( **tree_params )      #Define the decision three classifier
    dt_clf.fit(X_train,Y_train)
    Y_pred_DT =  dt_clf.predict(X_test)
    return Y_pred_DT, dt_clf


def Random_Forest_Classifier(X_train, Y_train, X_test):
    rf_clf = RandomForestClassifier(n_estimators=100)   #Define the random forest classifier
    rf_clf.fit(X_train,Y_train)        #Train the model using the training sets
    Y_pred_RF = rf_clf.predict(X_test)      # prediction on test set
    return Y_pred_RF, rf_clf


def visualise_tree(tree_to_print):
    plt.figure()
    tree.plot_tree(tree_to_print,
               filled = True,
              rounded=True);
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=800)
    plt.show()


def Bagging_Classifier(X_train, y_train, X_test,k):
    bag_clf = BaggingClassifier(n_estimators=k,max_samples=0.5, max_features=4,random_state=1)  #Create KNN object with a K coefficient
    bag_clf.fit(X_train, y_train)      #Fit KNN model
    Y_pred_BAG = bag_clf.predict(X_test)
    return Y_pred_BAG, bag_clf


def Boosting_Classifier(X_train, Y_train, X_test, estimator, k):
    boost_clf = AdaBoostClassifier(base_estimator=estimator, n_estimators=k)       # AdaBoost takes Decision Tree as its base-estimator model by default.
    boost_clf.fit(X_train,Y_train,sample_weight=None)  # Fit KNN model
    Y_pred_BOOST = boost_clf.predict(X_test)
    return Y_pred_BOOST, boost_clf


def Logistic_Classifier(X_train, Y_train, X_test, Y_test):
    logistic_clf = LogisticRegression(solver='lbfgs')     # Build Logistic Regression Model
    logistic_clf.fit(X_train, Y_train)            # Train the model using the training sets
    Y_pred= logistic_clf.predict(X_test)
    return Y_pred, logistic_clf


def SVM_Classifier(X_train,Y_train, X_test, kernel):
    svm_clf = SVC(kernel=kernel)
    svm_clf.fit(X_train,Y_train)
    y_pred = svm_clf.predict(X_test)
    return y_pred, svm_clf