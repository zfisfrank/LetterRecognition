#/usr/bin/python3
import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn import preprocessing


fullData = pd.read_csv('train.csv')#三个dataframe合成一个

dataId = fullData['Id']
target = fullData['Prediction']
le = preprocessing.LabelEncoder()
le.fit(target)
target = le.transform(target)

data = fullData.drop(['Id', 'Prediction', 'NextId', 'Position'],axis = 1)

position = pd.get_dummies(fullData['Position'])
data = pd.concat([position,data],axis = 1)

clf = svm.SVC(gamma = 0.06, C = 5)



clf.fit(data,target)

""" this part to predict the actual test data, then map 1 to 26 back to 'a' to 'z'"""
testValue = pd.read_csv('test.csv')
testValuePosition = pd.get_dummies(testValue['Position'])
testValueIndex = testValue['Id']
testValue = testValue.drop(['Id', 'Prediction', 'NextId', 'Position'],axis = 1)
testValue = pd.concat([testValuePosition,testValue],axis = 1)
Predictions = pd.Series(clf.predict(testValue))
Predictions = le.inverse_transform(Predictions)
Predictions = pd.DataFrame([testValueIndex,Predictions])
Predictions = Predictions.T

oldTarget = fullData[['Id','Prediction']]
Predictions.columns = oldTarget.columns
finalTable = pd.concat([Predictions,oldTarget]).set_index('Id').sort_index()
finalTable.to_csv('answer.csv')
