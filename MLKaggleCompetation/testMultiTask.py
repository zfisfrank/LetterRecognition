#/usr/bin/python3

from joblib import Parallel, delayed
def f(x):
    print('Running f(%s)' % x)
    return x


a = Parallel(n_jobs=16)(delayed(f)(i) for i in range(10))
Parallel(n_jobs=10, backend="threading")(delayed(f(i)) for i in range(10))

# (accumulator, n_iter)
parameters = {'kernel':['rbf'], 'C':[1], 'gamma': [0.1, 0.01]}
svm_clf = svm.SVC()
clf = GridSearchCV(svm_clf, parameters)
