import numpy as np
import pandas as pd
#from sklearn import datasets, svm, metrics
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn import neighbors

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
#data = fullData.drop(['Id', 'Prediction'],axis = 1)

"""this part to test current algorithms' accuracies"""
accuracies = []
layerSize = 300
for i in range(3):
    trainData, testData, trainTarget, testTarget = train_test_split(data,target,test_size= .5)
    clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(layerSize), random_state=26
    ,learning_rate = 'adaptive')
    #clf = tree.DecisionTreeClassifier()
    #clf = neighbors.KNeighborsClassifier(26, weights='distance')
    clf.fit(trainData,trainTarget)
    #trainData, testData, trainTarget, testTarget = train_test_split(data,target,test_size= .5)
    Predictions = clf.predict(testData)
    accuracies = (sum(Predictions == testTarget)/len(testTarget))
    print(accuracies)
# print(test_target)
# print(clf.predict(test_data))


""" this part to predict the actual test data, then map 1 to 26 back to 'a' to 'z'"""
trainData, testData, trainTarget, testTarget = train_test_split(data,target,test_size= .5)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(50, 20), random_state=10)
clf.fit(data,target)

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
#target = pd.DataFrame([dataId,target.map(num2LetterMap)]).T.set_index('Id')

finalTable = pd.concat([p,t]).set_index('Id').sort()
pd.merge([Predictions, target]).to_csv('answer.csv')
