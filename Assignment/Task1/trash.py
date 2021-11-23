

def PCAPredict(X, k):
    '''
    Inputs
        X: dataset;
        k: number of Components.

    Return
        SValue: The singular values corresponding to each of the selected components.
        Variance: The amount of variance explained by each of the selected components.
                It will provide you with the amount of information or variance each principal component holds after projecting the data to a lower dimensional subspace.
        Vcomp: The estimated number of components.
    '''
    pca = PCA(n_components=k)   # the built-in function for PCA, where n_clusters is the number of clusters.
    pca.fit(X)      # fit the algorithm with dataset

    Variance = pca.explained_variance_ratio_
    SValue = pca.singular_values_
    Vcomp = pca.components_
    return SValue, Variance, Vcomp

def get_data():
    X, y = l2.extract_features_labels()
    Y = np.array([y, -(y - 1)]).T

    return tr_X, tr_Y, te_X, te_Y


#Dimension Reduction using PCA (feature extraction)
k1 = 10
SingularValue, Variance, Vcomponent = PCAPredict(images_vectors,k1)
#print("\nSingular Value: {}".format(SingularValue))
#print("\nVariance: {}".format(Variance))
#print("\nVcomponent: {}".format(Vcomponent))

images_features = []
single_image_feature = []
for image_vector in images_vectors:
    for component in Vcomponent:
        single_image_feature.append(abs(np.dot(image_vector,component)))
    images_features.append(single_image_feature)
    single_image_feature = []

print("\nNumber of features: {}".format(k1))
print("\nShape: {}".format(np.array(images_features).shape))


#Feature Selection
best_images_features = SelectPercentile(chi2, percentile=10).fit_transform(images_vectors, labels)
print(best_images_features.shape)