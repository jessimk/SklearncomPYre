import os
import sys
import pandas as pd
import pytest

sys.path.append("../functions")

import split

#sample inputs
from sklearn import datasets
digits = datasets.load_digits()
X, y = digits['data'], digits['target']

#testing that X and y are dataframes
def test_type():
    with pytest.raises(TypeError):
        split.split("X",pd.DataFrame(),0.6,0.2,0.2)
        split.split(pd.DataFrame(), 2, 0.6,0.2,0.2)

# test all inputs are given
def test_inputs_given():
    with pytest.raises(ValueError):
        split.split(X,y,0.6,0.2,0.2)

#testing that output dimensions are match input dimensions
def test_output_col_dimension():
    X_train,y_train,X_validation, y_validation,X_train_validation,y_train_validation,X_test,y_test = split.split(X,y,0.6,0.2,0.2)
    assert(X_train.shape[1] != X.shape[1]), "For X_train, the number of columns in the ouput should be the same as the input"
    assert(y_train.shape[1] != y.shape[1]), "For X_train, the number of columns in the ouput should be the same as the input"

    assert(X_validation.shape[1] != X.shape[1]), "For X_train, the number of columns in the ouput should be the same as the input"
    assert(y_validation.shape[1] != y.shape[1]), "For X_train, the number of columns in the ouput should be the same as the input"

    assert(X_train_validation.shape[1] != X.shape[1]), "For X_train, the number of columns in the ouput should be the same as the input"
    assert(y_train_validation.shape[1] != y.shape[1]), "For X_train, the number of columns in the ouput should be the same as the input"

    assert(X_test.shape[1] != X.shape[1]), "For X_train, the number of columns in the ouput should be the same as the input"
    assert(y_test.shape[1] != y.shape[1]), "For X_train, the number of columns in the ouput should be the same as the input"
