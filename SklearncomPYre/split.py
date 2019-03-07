
# coding: utf-8

# In[ ]:
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def split(X, y, ptrain, pvalid, ptest):


    """
    The function splits the training input samples X, and target values y
    (class labels in classification, real numbers in regression) into train,
    test and validation sets according to specified proportions.

    The function Outputs four array like training, validation, test, and
    combined training and validation sets and four y arrays.

    Inputs:

    X data set, type: Array like
    Y data set, type: Array like
    proportion of training data , type: float
    proportion of test data , type: float
    proportion of validation data, type: float

    Outputs:

    X train set, type: Array like
    y train, type: Array like
    X validation set, type: Array like
    y validation, type: Array like
    X train and validation set, type: Array like
    y train and validation, type: Array like
    X test set, type: Array like
    y test, type: Array like

    Examples:
    # Splitting up datasets into 40% training, 20% vaildation, and 40% tests sets.
    X_train, y_train, X_val, y_val, X_train_val, y_train_val, X_test, y_test = split(X,y,0.4,0.2,0.4)

    See README for examples-- https://github.com/UBC-MDS/SklearncomPYre/blob/Jes/README.md

    """
    #testing that X and y are both either dataframe or array types

    # if  (type(X) != type(np.array(X)) and type(X) != type(pd.DataFrame(X))) or \
    #     (type(y) != type(np.array(y)) and type(y) != type(pd.DataFrame(y))) or \
    #     X.shape[0] != len(y):
    #     raise TypeError("X or y are not the right type. X and Y should be the \
    #     same length and they should be either arrays or dataframes.\
    #     See documentation and try again ¯\_(ツ)_/¯ ")


    if  type(X) != type(np.array([2,2])) and type(X) != type(pd.DataFrame([2,2])):
        raise TypeError("X isn't the right type. Make sure it's a dataframe or array ¯\_(ツ)_/¯ ")

    elif type(y) != type(np.array([2,2])) and type(y) != type(pd.DataFrame([2,2])):
        raise TypeError("y isn't the right type. Make sure it's a dataframe or array ¯\_(ツ)_/¯ ")

    #testing that X and y have the same number of rows and have at least 3 rows
    elif (X.shape[0] != len(y)) and (len(y) < 3):
        raise TypeError("X & y lengths don't match. Both X & y need to have at least 3 rows. Try again ¯\_(ツ)_/¯ ")

    else:
        X_train_validation, X_test, y_train_validation, y_test = train_test_split(X, y, test_size= ptest)

        validation_ratio = round(pvalid/(ptrain + pvalid),2)

        X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation, y_train_validation, test_size=validation_ratio)

    return X_train,y_train,X_validation, y_validation,X_train_validation,y_train_validation,X_test,y_test
