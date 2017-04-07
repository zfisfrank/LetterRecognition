import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split
from sklearn import svm
import string


targetNums = list(range(1,27)) * 1000
letter2NumMap = dict(zip(string.ascii_lowercase,targetNums))
num2LetterMap = dict(zip(targetNums,string.ascii_lowercase))
fullData = pd.read_csv('train.csv')
dataId = fullData['Id']
target = fullData['Prediction']
target = target.map(letter2NumMap)
data = fullData.drop(['Id', 'Prediction', 'NextId', 'Position'],axis = 1)


""" filter out some of the no matching data """
trainData, testData, trainTarget, testTarget = train_test_split(data,target,test_size= .9)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(300), random_state=10)
clf.fit(trainData,trainTarget)
Predictions = clf.predict(testData)

print(sum(Predictions == testTarget)/len(Predictions))
#only select correct predicted samples
trainData2 = testData[Predictions == testTarget]
trainTarget2 = testTarget[Predictions == testTarget]
""" this part to predict the actual test data, then map 1 to 26 back to 'a' to 'z'"""

clf.fit(trainData2,trainTarget2)

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
finalTable.to_csv('answer2.csv')
