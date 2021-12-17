# Date: 17/12/2021
# Institution: University College London
# Dept: EEE
# Class: ELEC0134
# SN: 18081659
# Description: Extract the ANOVA/PCA features for Task 1/2 and save then in their respective files

from Assignment.Functions import data_processing as dt

Task = 2
n_features = 5
dim_reduction = 'ANOVA'

if dim_reduction == 'ANOVA':
    ANOVA_features = dt.process_ANOVA_features(Task, n_features)
elif dim_reduction == 'PCA':
    PCA_features = dt.process_PCA_features(Task, n_features)
else:
    print('Invalid dimensionality reduction method')


# ANOVA_features = dt.process_ANOVA_features(Task,5)
# ANOVA_features = dt.process_ANOVA_features(Task,10)
# PCA_features = dt.process_PCA_features(Task,5)
# PCA_features = dt.process_PCA_features(Task,10)

