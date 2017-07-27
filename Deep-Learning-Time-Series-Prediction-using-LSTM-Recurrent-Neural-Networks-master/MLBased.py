# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 12:31:55 2017

@author: kaust
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
#import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn import svm

data = pd.read_csv("F:\\FinanceRepo\\Deep-Learning-Time-Series-Prediction-using-LSTM-Recurrent-Neural-Networks-master\\NSE_NIFTY.csv")
#data = data.drop(['DATE'],axis=1)
data.head()

NumpyData = data.as_matrix()
look_back = 4

def convertSeriesToMatrix(vectorSeries, look_back_window):
    matrix=[]
    for i in range(len(vectorSeries)-look_back_window+1):
        A= vectorSeries[i:i+look_back_window]
        A = A.reshape(1,(look_back_window*A.shape[1]))
        matrix.append(A)
    return matrix

Matrix = convertSeriesToMatrix(NumpyData,look_back)
Matrix = np.asarray(Matrix)
Matrix = Matrix.reshape(Matrix.shape[0],Matrix.shape[2])

Pandas_Matrix = pd.DataFrame(Matrix)
for i in range(0,(7*look_back-7),7):
    Pandas_Matrix = Pandas_Matrix.drop([i,i+1,i+2,i+3,i+4,i+6],axis=1)

Pandas_Matrix = Pandas_Matrix.drop([(7*look_back-7)],axis = 1)
Enter = []
row_iterator = Pandas_Matrix.iterrows()
Start = True
Gain = []
for index, row in row_iterator:
   if(Start != True):
        Gain_il = (row[(7*look_back-2)] - LastRow[(7*look_back-2)])/LastRow[(7*look_back-2)]*100
        Gain.append(Gain_il)
        if (Gain_il>0 ):
            Enter.append(1)
        else:
            Enter.append(0)
        LastRow = row
   else:
        LastRow = row
        Start = False
Enter.append(0)
Gain.append(0)
Pandas_Matrix['Enter'] = Enter

y = Pandas_Matrix['Enter'].values
Pandas_Matrix = Pandas_Matrix.drop(['Enter'],axis=1)
X = Pandas_Matrix.values

Xtrain, Xtest = X[:int(len(X) * 0.60)], X[int(len(X) * 0.60):] 
ytrain, ytest = y[:int(len(y) * 0.60)], y[int(len(y) * 0.60):] 
print(Xtrain.shape)
print(Xtest.shape)

gadaboost = GradientBoostingClassifier()
gadaboost.fit(Xtrain, ytrain)
y_val_l = gadaboost.predict_proba(Xtest)
print("Validation accuracy: ", sum(pd.DataFrame(y_val_l).idxmax(axis=1).values
                                   == ytest)/len(ytest))

target = open('C:\\Users\\kaust\\Downloads\\WinPython-64bit-3.5.2.3Qt5\\myScripts\\out.txt', 'w')
y_pre = pd.DataFrame(y_val_l).idxmax(axis=1).values

Total_Trades = 0
Total_Matched_Trades = 0
Total_False_Positives = 0
Total_False_Negatives = 0

for i in range(Xtest.shape[0]):
    line = str(ytest[i])+','+str(y_pre[i])
    target.write(line)
    target.write('\n')
    Total_Trades = Total_Trades + 1
    if(ytest[i] == y_pre[i] ):
        Total_Matched_Trades = Total_Matched_Trades + 1
    if(ytest[i] == 0 and y_pre[i] == 1):
        Total_False_Positives = Total_False_Positives + 1
    if(ytest[i] == 1 and y_pre[i] == 0):
        Total_False_Negatives = Total_False_Negatives + 1
target.close()

print("Total Trades = "+str(Total_Trades))
print("Total Matched Trades = "+str(Total_Matched_Trades))
print("Total False Positive Trades = "+str(Total_False_Positives))
print("Total False Negative Trades = "+str(Total_False_Negatives))

import numpy as np
indices = np.argsort(gadaboost.feature_importances_)[::-1]

# Print the feature ranking
print('Feature ranking:')

for f in range(Pandas_Matrix.shape[1]):
    print('%d. feature %d %s (%f)' % (f+1 , indices[f], Pandas_Matrix.columns[indices[f]],
                                      gadaboost.feature_importances_[indices[f]]))