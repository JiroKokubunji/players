# -*- encoding: utf-8 -*-

import importlib
from multiprocessing import Pool
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""
model inform schema
id
category 'sklearn'
module name
class name
initialize parameter
grid search parameter

transaction predict schema "classification"
id
project_id
model_id
accuracy
precision
recall
f-value
confusion matrix
feature importance 

transaction predict schema "regression"
id
project_id
model_id
mae
mse
rmse
rsquare
feature importance 
"""

models = [[1, 'sklearn.svm', 'SVC']
          , [2, 'sklearn.ensemble', 'RandomForestClassifier']
            ]


def fit_and_predict(model, data, target):
    module = importlib.import_module(model[1])
    instance = getattr(module, model[2])()
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33)
    instance.fit(X_train, y_train)
    y_pred = instance.predict(X_test)
    print("{0}:accuracy {1}".format(model[2], accuracy_score(y_test, y_pred)))


def wrap_fit_ant_predict(args):
    return fit_and_predict(*args)

def main():
    iris = load_iris()
    data = iris.data
    target = iris.target

    pool = Pool(processes=4)
    pool.map(wrap_fit_ant_predict, [(models[0], data, target), (models[1], data, target)])
    pool.close()

    print('process ended')


if __name__ == '__main__':
    main()

