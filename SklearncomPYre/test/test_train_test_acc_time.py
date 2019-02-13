
# import packages
import os
import numpy as np
import pandas as pd
import pytest
import sys

# sklearn packages
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
sys.path.insert(0, os.path.abspath("../../SklearncomPYre"))

import SklearncomPYre

# sample inputs
dictionary = {'knn': KNeighborsClassifier(), 'LogRegression':LogisticRegression() , 'RForest': RandomForestClassifier()}

iris = load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

# test train_test_acc_time(dictionary, X_train, y_train, X_test, y_test)

# input type test for the Dictionary and the Input Arrays (X_train, y_train, X_test, y_test)

def test_dict_type():
    with pytest.raises(TypeError):
        SklearncomPYre.train_test_acc_time('birinder', X_train, y_train, X_test, y_test),
        SklearncomPYre.train_test_acc_time(55, X_train, y_train, X_test, y_test)  ,
        SklearncomPYre.train_test_acc_time([1,2,3], X_train, y_train, X_test, y_test) ,
        SklearncomPYre.train_test_acc_time((4,44,5), X_train, y_train, X_test, y_test)

def test_X_array_type():
    with pytest.raises(TypeError):
        SklearncomPYre.train_test_acc_time({}, 123, y_train, X_test, y_test),
        SklearncomPYre.train_test_acc_time({}, "Vancouver", y_train, X_test, y_test),
        SklearncomPYre.train_test_acc_time({}, X_train, [5,6,7], X_test, y_test),
        SklearncomPYre.train_test_acc_time({}, X_train,y_train, (7,8,9), y_test)

def test_inputs_given():
    with pytest.raises(TypeError):
        SklearncomPYre.train_test_acc_time({},  y_train, X_test, y_test)
