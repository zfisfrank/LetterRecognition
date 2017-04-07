#/usr/bin/python3
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import svm
import string
#from joblib import Parallel, delayed

# targetNums = list(range(1,27)) * 1000
# letter2NumMap = dict(zip(string.ascii_lowercase,targetNums))
# num2LetterMap = dict(zip(targetNums,string.ascii_lowercase))

fullData = pd.read_csv('train.csv')

dataId = fullData['Id']
target = fullData['Prediction']
le = preprocessing.LabelEncoder()
le.fit(target)
target = le.transform(target)

data = fullData.drop(['Id', 'Prediction', 'NextId', 'Position'],axis = 1)

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV

parameters = {'kernel':['rbf'], 'C':[1], 'gamma': [0.1, 0.01]}
svm_clf = svm.SVC()
clf = GridSearchCV(svm_clf, parameters)

clf = svm.SVC(gamma = 0.06,C = 10)

clf.fit(data,target)

""" this part to predict the actual test data, then map 1 to 26 back to 'a' to 'z'"""
testValue = pd.read_csv('test.csv')
testValueIndex = testValue['Id']
testValue = testValue.drop(['Id', 'Prediction', 'NextId', 'Position'],axis = 1)

Predictions = pd.Series(clf.predict(testValue))
Predictions = le.inverse_transform(Predictions)
Predictions = pd.DataFrame([testValueIndex,Predictions])
Predictions = Predictions.T
#Predictions = Predictions.set_index('Id')
oldTarget = fullData[['Id','Prediction']]
Predictions.columns = oldTarget.columns
finalTable = pd.concat([Predictions,oldTarget]).set_index('Id').sort()
finalTable.to_csv('answerSVM.csv')
