import pandas as pd
import numpy as np
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold, RepeatedStratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import warnings
import os
from sklearn import svm
import pymrmr
warnings.filterwarnings("ignore")
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.feature_selection import chi2


def matrix_metrix(y_test, y_pred):
    CM = confusion_matrix(y_test, y_pred)

    TP = CM[0][0]
    FN = CM[1][0]
    TN = CM[1][1]
    FP = CM[0][1]

    PPV = round(TP / (TP + FP), 4)
    NPV= round(TN / (TN + FN), 4)
    Sensitivity = round(TP / (TP + FN), 4)
    Specificity= round(TN / (TN + FP), 4)
    print(pd.DataFrame({
        'Metric': ['PPV', 'NPV', 'Sensitivity', 'Specificity'], 'Value': [PPV, NPV, Sensitivity, Specificity]}))

def Final_Score():
    print("\n\n------------Starting run of gleason_4_Final_Score------------\n")
    print("This file will take top 600 feaatures and using MRMR filter it down to top 20 \nand then uses LDA classifier to pick top 3 and find the accuracy.\n")

    data = pd.read_csv('Gleason_score/gleason_Data_3_FS_RF_top600.csv')

    # Combined two classes 9 and 10
    data["GLEASON_SCORE"].replace({10: 9}, inplace=True)
    data["gleason_score"].replace({10: 9}, inplace=True)

    X_new = data.drop(['gleason_score', 'GLEASON_SCORE', 'CLIN_T_STAGE', 'PATH_T_STAGE'], 1)
    # Drop target gleason_score and other unwanted columns
    # X = data.drop(['Unnamed: 0', 'GLEASON_SCORE', 'CLIN_T_STAGE', 'PATH_T_STAGE', 'gleason_score'], 1)
    #X = X_new[['ENSG00000019582.13' ,'ENSG00000212766.8' ,'ENSG00000116151.12' ,'ENSG00000189164.13' ,'ENSG00000117262.17' ,'GLEASON_PATTERN_PRIMARY','ENSG00000135314.11' ,'ENSG00000229237.2' ,'ENSG00000144152.11','ENSG00000089723.8','ENSG00000251192.6','ENSG00000223658.6','ENSG00000272717.1','ENSG00000100027.13','ENSG00000200550.1','ENSG00000128203.6','ENSG00000164366.3','ENSG00000132622.9','GLEASON_PATTERN_SECONDARY','ENSG00000113272.12']]
    print(X_new.shape)
    selectedColumns = pymrmr.mRMR(X_new, 'MIQ', 20)
    X = X_new[selectedColumns]
    Y = data['GLEASON_SCORE']

    lda = LinearDiscriminantAnalysis(n_components=3, solver='svd', shrinkage=None)
    X_features = lda.fit_transform(X, Y)

    # 10-fold cross-validation (x_test and y_test is holdout for final test)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_features, Y, test_size=0.2, random_state=42)


    clf = RandomForestClassifier(n_estimators=100,n_jobs=-1, random_state=42)
    #clf = svm.SVC(kernel='rbf')
    #clf = KNeighborsClassifier(n_neighbors=7)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # print(confusion_matrix(solution.iloc[0]), y_test)
    scores = cross_val_score(clf, X_features, Y, scoring='accuracy', cv=cv)
    print()
    print(cm)
    Accuracy = round((sklearn.metrics.accuracy_score(y_test, y_pred)), 4)
    print('Accuracy: %.4f' % Accuracy)
    matrix_metrix(y_test, y_pred)
    print('Average Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)) + '\n')

    '''
    # plot to generate predict_gs.png. Please see image section for image.
    colors = ['m', 'r', 'c', 'b', 'k', 'm', 'yellow', 'orchid', 'fuchsia', 'lightcoral', 'g']
    markers = ['o', 'x', '*', '^', '1', 'p', 'D', '8', 's', 'P', 'o']

    le = preprocessing.LabelEncoder()
    le.fit(Y)
    label = le.transform(Y)

    for i in range(len(X_features)):
        plt.scatter(X_features[i][0], X_features[i][1], c=colors[label[i]], marker=markers[label[i]])
    plt.show()
    '''
    print("\n------------Ending run of clinic_4_Final_Score------------")
