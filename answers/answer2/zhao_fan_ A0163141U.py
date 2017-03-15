import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split
import string


targetNums = list(range(1,27)) * 1000
letter2NumMap = dict(zip(string.ascii_lowercase,targetNums))
num2LetterMap = dict(zip(targetNums,string.ascii_lowercase))
fullData = pd.read_csv('train.csv')
dataId = fullData['Id']
target = fullData['Prediction']
target = target.map(letter2NumMap)
data = fullData.drop(['Id', 'Prediction', 'NextId', 'Position'],axis = 1)

""" training part """
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(300), random_state=10)
clf.fit(data,target)

""" this part to predict the actual test data, then map 1 to 26 back to 'a' to 'z'"""
testValue = pd.read_csv('test.csv')
testValueIndex = testValue['Id']
testValue = testValue.drop(['Id', 'Prediction', 'NextId', 'Position'],axis = 1)

Predictions = pd.Series(clf.predict(testValue))
Predictions = Predictions.map(num2LetterMap)
Predictions = pd.DataFrame([testValueIndex,Predictions])
Predictions = Predictions.T
#Predictions = Predictions.set_index('Id')
oldTarget = fullData[['Id','Prediction']]
Predictions.columns = oldTarget.columns
finalTable = pd.concat([Predictions,oldTarget]).set_index('Id').sort()
finalTable.to_csv('answer.csv')
# pd.merge([Predictions, oldTarget]).to_csv('answer.csv')
# pd.merge(Predictions, oldTarget).to_csv('answer.csv')
