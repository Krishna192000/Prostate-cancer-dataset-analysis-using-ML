import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing

def Low_Variance():
    print("\n\n------------Starting run of gleason_2_Low_Variance------------\n")
    print(
        "This file will filter data and drop Nan and zero fetures \nand bring down data from 60k features to 20-25k features.\n")

    data = pd.read_csv('Gleason_score/gleason_Data_1_Predict_GleasonScore.csv')

    data = data.drop(columns=data.columns[((data == 0).mean() > 0.2)], axis=1)
    data = data.dropna(axis='columns')

    print(data.shape)

    #Input X and Target Y (here pathT score)
    #X = data.iloc[:, 2:60490]
    X = np.array(data.drop(['Unnamed: 0', 'gleason_score', 'Transcript'],1))
    Y = np.array(data['gleason_score'])

    X = preprocessing.scale(X)

    selector = VarianceThreshold()
    new_features = selector.fit_transform(X)

    print(X.shape)
    print(new_features.shape)
    #print(len(X.T)) #no. of columns

    feature_indices = selector.get_support(indices=True)
    remove = []
    for i in range(len(X.T)):
        if i not in feature_indices:
            remove.append(i)
            print(i)

    #Drop feature columns with low variance
    data = data.drop(data.columns[remove],axis=1)

    new_X = data.drop(['Unnamed: 0', 'Transcript' ],1)
    new_X.to_csv('Gleason_score/gleason_Data_2_Low_Variance.csv', sep=',')
    print(new_X.shape)

    '''
    new_X = preprocessing.scale(new_X)
    print(new_X)
    data.to_csv('/Users/krishna/Desktop/newFeatures1.csv', sep=',')
    '''
    print("\n------------Ending run of gleason_2_Low_Variance------------")
