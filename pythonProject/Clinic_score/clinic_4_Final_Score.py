import pandas as pd
import numpy as np
import pymrmr
import sklearn
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn import svm
import warnings
warnings.filterwarnings("ignore")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt


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
    print("\n\n------------Starting run of clinic_4_Final_Score------------\n")
    print("This file will take top 600 feaatures and using LDA classifier, pick top 6 and find the accuracy.\n")
    data = pd.read_csv('Clinic_score/clinic_Data_3_FS_RF_top600.csv')
    print("Read: ", data.shape)

    # Drop target gleason_score and other unwanted columns
    X_new = data.drop(['Unnamed: 0', 'clinTscore'], 1)
    Y = data['clinTscore']
    print("New Length: ", len(X_new))

    minmax_scale = MinMaxScaler().fit(X_new)
    normalX = minmax_scale.transform(X_new)

    X_new = SelectKBest(chi2, k=420).fit_transform(normalX, Y)

    Xnew = SelectKBest(chi2, k=20).fit_transform(normalX, Y)

    lda = LinearDiscriminantAnalysis(n_components=6, solver='svd', shrinkage=None)
    normalX = lda.fit_transform(X_new, Y)

    # 10-fold cross-validation (x_test and y_test is holdout for final test)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(normalX, Y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    #clf = svm.SVC(kernel='rbf')
    #clf = KNeighborsClassifier(n_neighbors=7)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    scores = cross_val_score(clf, normalX, Y, scoring='accuracy', cv=cv)
    print()
    print(cm)
    Accuracy = round((sklearn.metrics.accuracy_score(y_test, y_pred)), 4)
    print(': Accuracy: ')
    print(Accuracy)
    matrix_metrix(y_test, y_pred)
    print('Average Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)) + '\n')

    '''
    # plot to generate predict_gs.png. Please see image section for image.
    colors = ['m', 'r', 'c', 'b', 'k', 'm', 'yellow', 'orchid', 'fuchsia', 'lightcoral', 'g']
    markers = ['o', 'x', '*', '^', '1', 'p', 'D', '8', 's', 'P', 'o']

    le = preprocessing.LabelEncoder()
    le.fit(Y)
    label = le.transform(Y)

    for i in range(len(normalX)):
        plt.scatter(normalX[i][0], normalX[i][1], c=colors[label[i]], marker=markers[label[i]])
    plt.show()
    '''
    print("\n------------Ending run of clinic_4_Final_Score------------")