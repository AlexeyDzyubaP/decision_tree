
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import math
import pandas as pd
from functools import reduce


for idata in range(21):
    for icit in range(len(kernFuncs)):
        for isplit in range(len(kernFuncs)):

filename1 = '0(%d)_train.csv' % (idata+1)  #'01_train.csv'
filename2 = '0(%d)_test.csv' % (idata+1)  #'01_test.csv'

#filename1 = '02_train.csv'
#filename2 = '02_test.csv'

df_train = pd.read_csv(filename1, na_values="?" )
df_test = pd.read_csv(filename2, na_values="?" )
print(df_train.head())
print(df_train.shape)
print(df_test.shape[1])

########train

X_train = df_train.values[:, :(df_train.shape[1] - 1)]
Y = df_test.values[:, (df_train.shape[1] - 1)]
print(X_train, '====', Y)
Nlbl = len(np.unique(Y)) # Number of classes

print(X_train.ptp(0))
X_train_norm = (X_train - X_train.min(0)) / X_train.ptp(0)  # X.ptp - peak to peak range
print(X_train_norm)

#########test

X_test = df_train.values[:, :(df_test.shape[1] - 1)]
Y_test = df_test.values[:, (df_test.shape[1] - 1)]
#print(X_test, '====', Y_test)
Nlbl = len(np.unique(Y_test)) # Number of classes

print(X_test.ptp(0))
X_test_norm = (X_test - X_test.min(0)) / X_test.ptp(0)  # X.ptp - peak to peak range
print(X_test_norm)



tree = DecisionTreeClassifier(criterion="gini", splitter="best")
tree.fit(X_train_norm, Y)

print('Model Accuracy:', tree.score(X_test_norm, Y_test))
print('Depth:', tree.get_depth())