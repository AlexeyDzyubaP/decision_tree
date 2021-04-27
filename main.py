
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from functools import reduce


criterion = ["gini","entropy"]
splitter = ["best","random"]
max_depth = 0
min_depth = 100
ovrl_best_score = 0

for idata in range(21):
    max_loc_depth = 0
    min_loc_depth = 100
    best_score = 0
    for icrit in range(2):
        for isplit in range(2):

            if idata+1 <= 9:
                filename1 = '0%d_train.csv' % (idata+1)  #'01_train.csv'
                filename2 = '0%d_test.csv' % (idata+1)  #'01_test.csv'
            else:
                filename1 = '%d_train.csv' % (idata + 1)  # '01_train.csv'
                filename2 = '%d_test.csv' % (idata + 1)  # '01_test.csv'

            #filename1 = '02_train.csv'
            #filename2 = '02_test.csv'

            df_train = pd.read_csv(filename1, na_values="?" )
            df_test = pd.read_csv(filename2, na_values="?" )
            #print(df_train.head())
            #print(df_train.shape)
            #print(df_test.shape[1])

            ########train

            X_train = df_train.values[:, :(df_train.shape[1] - 1)]

            Y = df_train.values[:, (df_train.shape[1] - 1)]
            #print(X_train, '====', Y)
            Nlbl = len(np.unique(Y)) # Number of classes

            #print(X_train.ptp(0))
            X_train_norm = (X_train - X_train.min(0)) / X_train.ptp(0)  # X.ptp - peak to peak range
            Y_norm = (Y - Y.min(0)) / Y.ptp(0)
            #print(X_train.ptp(0))
            #print(X_train.ptp(0))
            #print(X_train_norm)

            #########test

            X_test = df_test.values[:, :(df_test.shape[1] - 1)]
            Y_test = df_test.values[:, (df_test.shape[1] - 1)]
            #print(X_test, '====', Y_test)
            Nlbl = len(np.unique(Y_test)) # Number of classes

            #print(X_test.ptp(0))
            X_test_norm = (X_test - X_test.min(0)) / X_test.ptp(0)  # X.ptp - peak to peak range
            #print(X_test_norm)



            tree = DecisionTreeClassifier(criterion=criterion[icrit], splitter=splitter[isplit])
            #print(cross_val_score(tree, X_train_norm, Y))
            #forest = RandomForestClassifier()
            #forest.fit(X_train_norm,Y)
            tree.fit(X_train_norm, Y)

            if tree.get_depth() > max_loc_depth:
                max_loc_depth = tree.get_depth()
                max_loc_depth_param = [idata+1, criterion[icrit], splitter[isplit], tree.get_depth()]

            if tree.score(X_train_norm, Y) > best_score:
                best_score = tree.score(X_train_norm, Y)
                best_score_param = [idata+1, criterion[icrit], splitter[isplit], tree.get_depth()]

            if tree.get_depth() < min_loc_depth:
                min_loc_depth = tree.get_depth()
                min_loc_depth_param = [idata+1, criterion[icrit], splitter[isplit], tree.get_depth()]



            print('data file, train acc, test acc, depth = ', idata+1, tree.score(X_train_norm, Y), tree.score(X_test_norm, Y_test), tree.get_depth(), criterion[icrit], splitter[isplit])
            #print('data file, train acc, test acc, depth = ', idata + 1, tree.score(X_train, Y), tree.score(X_test, Y_test), tree.get_depth(), criterion[icrit], splitter[isplit])


            #print('Model Accuracy:', tree.score(X_test_norm, Y_test))
            #print('Depth:', tree.get_depth())

    if max_loc_depth > max_depth:
        max_depth = max_loc_depth
        max_depth_param =max_loc_depth_param

    if best_score > ovrl_best_score:
        ovrl_best_score = best_score
        ovrl_best_score_param =best_score_param

    if min_loc_depth < min_depth:
        min_depth = min_loc_depth
        min_depth_param = min_loc_depth_param



print('Max depth = ', max_depth_param)
print('Min depth = ', min_depth_param)
print('Best score = ', ovrl_best_score_param, ovrl_best_score)



tree_train_score = np.zeros(35)
tree_test_score = np.zeros(35)
for idepth in range(35):

    filename1 = '12_train.csv'
    filename2 = '12_test.csv'

    df_train = pd.read_csv(filename1, na_values="?")
    df_test = pd.read_csv(filename2, na_values="?")


    X_train = df_train.values[:, :(df_train.shape[1] - 1)]
    Y = df_test.values[:, (df_train.shape[1] - 1)]
    # print(X_train, '====', Y)
    Nlbl = len(np.unique(Y))  # Number of classes

    X_train_norm = (X_train - X_train.min(0)) / X_train.ptp(0)  # X.ptp - peak to peak range


    X_test = df_train.values[:, :(df_test.shape[1] - 1)]
    Y_test = df_test.values[:, (df_test.shape[1] - 1)]

    Nlbl = len(np.unique(Y_test))  # Number of classes

    X_test_norm = (X_test - X_test.min(0)) / X_test.ptp(0)  # X.ptp - peak to peak range

    tree = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth = idepth+1)
    tree.fit(X_train_norm, Y)
    tree_train_score[idepth] = tree.score(X_train_norm, Y)
    tree_test_score[idepth] = tree.score(X_test_norm, Y_test)

plt.plot(tree_train_score[1:35])
plt.xlabel("Depth")
plt.ylabel("Score")
plt.show()

plt.plot(tree_test_score[1:35])
plt.xlabel("Depth")
plt.ylabel("Score")
plt.show()

for idepth in range(10):

    filename1 = '08_train.csv'
    filename2 = '08_test.csv'

    df_train = pd.read_csv(filename1, na_values="?")
    df_test = pd.read_csv(filename2, na_values="?")


    X_train = df_train.values[:, :(df_train.shape[1] - 1)]
    Y = df_test.values[:, (df_train.shape[1] - 1)]
    # print(X_train, '====', Y)
    Nlbl = len(np.unique(Y))  # Number of classes

    X_train_norm = (X_train - X_train.min(0)) / X_train.ptp(0)  # X.ptp - peak to peak range


    X_test = df_train.values[:, :(df_test.shape[1] - 1)]
    Y_test = df_test.values[:, (df_test.shape[1] - 1)]

    Nlbl = len(np.unique(Y_test))  # Number of classes

    X_test_norm = (X_test - X_test.min(0)) / X_test.ptp(0)  # X.ptp - peak to peak range

    tree = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth = idepth+1)
    tree.fit(X_train_norm, Y)
    #plot_tree(tree,filled=True)
    tree_train_score[idepth] = tree.score(X_train_norm, Y)
    tree_test_score[idepth] = tree.score(X_test_norm, Y_test)

plt.plot(tree_train_score[1:10])
plt.xlabel("Depth")
plt.ylabel("Score")
plt.show()

plt.plot(tree_test_score[1:10])
plt.xlabel("Depth")
plt.ylabel("Score")
plt.show()

