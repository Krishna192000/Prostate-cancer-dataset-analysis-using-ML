import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesClassifier

def FS_RF():
    print("\n\n------------Starting run of clinic_3_FS_RF------------\n")
    print("This file will read the filtered 25k fetures and apply \nFeature Selection with Random forest and output top600 features.\n")
    data = pd.read_csv('Clinic_score/clinic_Data_2_Low_Variance.csv')
    print("Read: ", data.shape)

    #Input X and Target Y (here gleason_score)

    X = np.array(data.drop(['Unnamed: 0', 'PATH_T_STAGE', 'clinTscore'],1))
    Y = np.array(data['clinTscore'])

    minmax_scale = preprocessing.MinMaxScaler().fit(X)
    normalX = minmax_scale.transform(X)

    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=300, random_state=0)

    forest.fit(normalX, Y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

    #[::-1] is for reverse or ascending order
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(600):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    #Drop all the columns apart from the top 20 columns based on importance ranking
    index = indices[601:-1]
    data.drop(data.columns[[index]], axis=1, inplace=True)
    data.to_csv('Clinic_score/clinic_Data_3_FS_RF_top600.csv', sep=',')

    print("\n------------Ending run of clinic_3_FS_RF------------")
