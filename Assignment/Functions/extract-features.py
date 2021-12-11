
from Assignment.Functions import data_processing_T1 as dt1
from Assignment.Functions import data_processing_T2 as dt2
import matplotlib.pyplot as plt

Task = 2
n_features = 5
dim_reduction = 'ANOVA'


# if Task == 1:
#     if dim_reduction == 'ANOVA':
#         ANOVA_features = dt1.process_ANOVA_features(n_features)
#     elif dim_reduction == 'PCA':
#         PCA_features = dt1.process_PCA_features(n_features)
#     else:
#         print('Invalid dimensionality reduction method')
# elif Task == 2:
#     if dim_reduction == 'ANOVA':
#         ANOVA_features = dt2.process_ANOVA_features(n_features)
#     elif dim_reduction == 'PCA':
#         PCA_features = dt2.process_PCA_features(n_features)
#     else:
#         print('Invalid dimensionality reduction method')
# else:
#     print('Task Invalid')

# ANOVA_features = dt2.process_ANOVA_features(2)
# PCA_features = dt2.process_PCA_features(2)
ANOVA_features = dt2.process_ANOVA_features(5)
# PCA_features = dt2.process_PCA_features(5)
# ANOVA_features = dt2.process_ANOVA_features(10)
# PCA_features = dt2.process_PCA_features(10)