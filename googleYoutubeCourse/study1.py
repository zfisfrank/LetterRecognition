"""Collect training data >>> Train classifier >>> Make Predictions"""

from sklearn import tree
""" collect training data"""
# features = [[140,smooth],[130,smooth],[150,bumpy],[170,bumpy]]

features = [[140,1],[130,1],[150,0],[170,0]]
#labels = ["apple","apple","orange","orange"]
labels = [0,0,1,1]
""" train classfier"""

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)

print(clf.predict([[150,0]]))
