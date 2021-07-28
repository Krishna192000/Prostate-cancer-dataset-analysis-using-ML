import pandas as pd
import pickle
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA,KernelPCA
from sklearn import svm
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

def Low_Variance():
    print("\n\n------------Starting run of clinic_2_Low_Variance------------\n")
    print("This file will filter data and drop Nan and zero fetures \nand bring down data from 60k features to 20-25k features.\n")
    data = pd.read_csv('Clinic_score/clinic_Data_1_Predict_Tscore.csv')
    print("Read: ", data.shape)

    #Dropping columns(features) that have more that 20% zero's
    data = data.drop(columns=data.columns[((data == 0).mean() > 0.2)], axis=1)
    print('New Length: ' , data.shape)

    row = data[(data['clinTscore'] == '[Not Available]')].index
    data.drop(row, inplace=True)
    #data = data.dropna(axis='columns')
    print('New Length: ' , data.shape)

    #Input X and Target Y (here gleason_score)
    #X = data.iloc[:, 2:60490]
    X = np.array(data.drop(['Unnamed: 0', 'clinTscore', 'Transcript'], 1))
    Y = np.array(data['clinTscore'])

    X = preprocessing.scale(X)

    selector = VarianceThreshold()
    new_features = selector.fit_transform(X)

    print('New Length: ' , X.shape)
    print('New Length: ' , new_features.shape)

    feature_indices = selector.get_support(indices=True)
    remove = []
    for i in range(len(X.T)):
        if i not in feature_indices:
            remove.append(i)
            #print(i)

    #print("length: ", len(remove))
    #Drop feature columns with low variance
    data = data.drop(data.columns[remove],axis=1)

    new_X = data.drop(['Unnamed: 0', 'Transcript' ],1)
    new_X.to_csv('Clinic_score/clinic_Data_2_Low_Variance.csv', sep=',')
    print('New Length: ' , new_X.shape)
    print("Done filtering data in Low Variance. Please find the file clinic_Data_1_Low_Variance.csv")
    '''
    new_X = preprocessing.scale(new_X)
    print(new_X)
    data.to_csv('/Users/krishna/Desktop/newFeatures1.csv', sep=',')
    '''
    print("\n------------Ending run of clinic_2_Low_Variance------------")
