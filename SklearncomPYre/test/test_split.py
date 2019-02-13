import os
import sys

sys.path.insert(0, os.path.abspath("../../SklearncomPYre"))

import pytest
from sklearn.model_selection import train_test_split
import SklearncomPYre
import numpy as np
import pandas as pd

#sample inputs
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

#testing that X and y are dataframes
def test_type():
    with pytest.raises(TypeError):
        SklearncomPYre.split(list(X),y,0.6,0.2,0.2)
        SklearncomPYre.split(X, 2, 0.6,0.2,0.2)

# test all inputs are given
def test_inputs_given():
    with pytest.raises(TypeError):
        SklearncomPYre.split(X,0.6,0.2,0.2)
        SklearncomPYre.split(X,y,0.2,0.2)

#testing that output dimensions are match input dimensions
def test_output_col_dimension():
    X_train,y_train,X_validation, y_validation,X_train_validation,y_train_validation,X_test,y_test = SklearncomPYre.split(X,y,0.2,0.2,0.6)

    assert(X_train.shape[1] == X.shape[1]), "The number of columns in the ouput should be the same as the input"
    assert(len(y_train.shape) == len(y.shape)), "The number of columns in the ouput should be the same as the input"

    assert(X_validation.shape[1] == X.shape[1]), "The number of columns in the ouput should be the same as the input"
    assert(len(y_validation.shape) == len(y.shape)), "The number of columns in the ouput should be the same as the input"

    assert(X_train_validation.shape[1] == X.shape[1]), "The number of columns in the ouput should be the same as the input"
    assert(len(y_train_validation.shape) == len(y.shape)), "The number of columns in the ouput should be the same as the input"

    assert(X_test.shape[1] == X.shape[1]), "The number of columns in the ouput should be the same as the input"
    assert(len(y_test.shape) == len(y.shape)), "The number of columns in the ouput should be the same as the input"

#testing that there are 8 output dataframes
def test_output_total():
    X_train,y_train,X_validation, y_validation,X_train_validation,y_train_validation,X_test,y_test = SklearncomPYre.split(X,y,0.2,0.2,0.6)
    a = [X_train,y_train,X_validation, y_validation,X_train_validation,y_train_validation,X_test,y_test]
    assert(len(a) == 8), "missing outputs, there should be 8 dataframes"

#testing that output data set pairs match in shape
def test_output_sets_match():
    X_train,y_train,X_validation, y_validation,X_train_validation,y_train_validation,X_test,y_test = SklearncomPYre.split(X,y,0.2,0.2,0.6)

    assert(X_train.shape[0] == y_train.shape[0]), "The number of rows in the ouput should be the same as the input"
    assert(X_test.shape[0] == y_test.shape[0]), "The number of rows in the ouput should be the same as the input"
    assert(X_validation.shape[0] == y_validation.shape[0]), "The number of rows in the ouput should be the same as the input"
    assert(X_train_validation.shape[0] == y_train_validation.shape[0]), "The number of rows in the ouput should be the same as the input"
