#/usr/bin/python3
import numpy as np
import pandas as pd
#from sklearn import datasets, svm, metrics
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn import neighbors
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPRegressor
from sklearn import svm
import string
#from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
# targetNums = list(range(1,27)) * 1000
# letter2NumMap = dict(zip(string.ascii_lowercase,targetNums))
# num2LetterMap = dict(zip(targetNums,string.ascii_lowercase))

fullData = pd.read_csv('train.csv')

dataId = fullData['Id']
target = fullData['Prediction']
le = preprocessing.LabelEncoder()
le.fit(target)
target = le.transform(target)
# target = target.map(letter2NumMap)
data = fullData.drop(['Id', 'Prediction', 'NextId', 'Position'],axis = 1)
#data = fullData.drop(['Id', 'Prediction'],axis = 1)

"""this part to test current algorithms' accuracies"""
accuracies = []
#layerSize = [1]*1000
# for layerSize in range(100,1000,100):
# # for acti in ['identity','logistic','tanh']:
#     trainData, testData, trainTarget, testTarget = train_test_split(data,target,test_size= .5)
#     #clf = MLPClassifier(solver='lbfgs', activation = acti,alpha=1e-5,hidden_layer_sizes=(5000), learning_rate = 'invscaling',random_state=1)
#     clf = MLPClassifier(solver='lbfgs', activation = 'relu',alpha=1e-5,hidden_layer_sizes=layerSize, learning_rate = 'invscaling',random_state=1)
#     #clf = Perceptron(n_jobs = -1)
#     #clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100000), random_state=1)
#     #clf = svm.SVC(degree = 30,max_iter = -1,kernel = 'rbf')
#     #clf = tree.DecisionTreeClassifier()
#     #clf = neighbors.KNeighborsClassifier(26, weights='distance')
#     clf.fit(trainData,trainTarget)
#     #trainData, testData, trainTarget, testTarget = train_test_split(data,target,test_size= .5)
#     Predictions = clf.predict(testData)
#     #Predictions = round(Predictions)
#     accuracies.append(sum(Predictions == testTarget)/len(testTarget))
#     acc = pd.Series(accuracies)
#     acc.to_csv('results.txt')
#     print(layerSize)
#     print(accuracies)

# for layers in range(1,5):
# for acti in ['identity','logistic','tanh']:
trainData, testData, trainTarget, testTarget = train_test_split(data,target,test_size= .5)
#clf = MLPClassifier(solver='lbfgs', activation = acti,alpha=1e-5,hidden_layer_sizes=(5000), learning_rate = 'invscaling',random_state=1)

# clf = MLPClassifier(solver='lbfgs', activation = 'relu',alpha=1e-10,hidden_layer_sizes=900, learning_rate = 'invscaling',random_state=1)
# clf = MLPClassifier(solver='sgd', activation = 'relu',learning_rate_init = 0.0001,alpha=1e-5,hidden_layer_sizes=400, learning_rate = 'adaptive',random_state=1,)
# clf = MLPClassifier(solver='sgd', activation = 'relu',alpha=1e-2,hidden_layer_sizes=2000, learning_rate = 'adaptive',random_state=1)
# clf = svm.SVC(decision_function_shape='ovo')
#clf = Perceptron(n_jobs = -1)
#clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100000), random_state=1)
#clf = svm.SVC(degree = 30,max_iter = -1,kernel = 'rbf')
#clf = tree.DecisionTreeClassifier()
#clf = neighbors.KNeighborsClassifier(26, weights='distance')

# from sklearn.model_selection import ParameterGrid
# param_grid = [{'kernel': ['rbf'], 'C': [0.1,1], 'gamma': [0.1]}]
# pg =ParameterGrid(param_grid)
# list(pg)
# from sklearn.model_selection import GridSearchCV
# parameters = {'kernel':['rbf'], 'C':[1], 'gamma': [0.1, 0.01]}
# svm_clf = svm.SVC()
# clf = GridSearchCV(svm_clf, parameters)
# sorted(clf.cv_results_.keys())

clf = svm.SVC(gamma = 0.06,C = 5)

# clf = RandomForestClassifier()
clf.fit(trainData,trainTarget)
#trainData, testData, trainTarget, testTarget = train_test_split(data,target,test_size= .5)
Predictions = clf.predict(testData)
#Predictions = round(Predictions)
accuracies.append(sum(Predictions == testTarget)/len(testTarget))
acc = pd.Series(accuracies)
acc.to_csv('results2.txt')
#print(layers)
print(accuracies)
# print(test_target)
# print(clf.predict(test_data))


# """ this part to predict the actual test data, then map 1 to 26 back to 'a' to 'z'"""
# trainData, testData, trainTarget, testTarget = train_test_split(data,target,test_size= .5)
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(50, 20), random_state=10)
# clf.fit(data,target)
#
# testValue = pd.read_csv('test.csv')
# testValueIndex = testValue['Id']
# testValue = testValue.drop(['Id', 'Prediction', 'NextId', 'Position'],axis = 1)
#
# Predictions = pd.Series(clf.predict(testValue))
# Predictions = Predictions.map(num2LetterMap)
# Predictions = pd.DataFrame([testValueIndex,Predictions])
# Predictions = Predictions.T
# #Predictions = Predictions.set_index('Id')
# oldTarget = fullData[['Id','Prediction']]
# Predictions.columns = oldTarget.columns
# #target = pd.DataFrame([dataId,target.map(num2LetterMap)]).T.set_index('Id')
#
# finalTable = pd.concat([p,t]).set_index('Id').sort()
# pd.merge([Predictions, target]).to_csv('answer.csv')
