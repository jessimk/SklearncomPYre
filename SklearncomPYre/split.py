
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

    try:
        X_train_validation, X_test, y_train_validation, y_test = train_test_split(X, y, test_size= ptest)

        validation_ratio = round(pvalid/(ptrain + pvalid),2)

        X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation, y_train_validation, test_size=validation_ratio)

    except ValueError:
        print("¯\_(ツ)_/¯ Error: Check that X and y are dataframes or arrays. X and y must be the same length. Try again?")
        raise ValueError("¯\_(ツ)_/¯ Error: Check that X and y are dataframes or arrays. X and y must be the same length. Try again?")

    except TypeError:
        print("¯\_(ツ)_/¯ Error: Check that X and y are dataframes or arrays. X and y must be the same length. Try again?")
        raise TypeError("¯\_(ツ)_/¯ Error: Check that X and y are dataframes or arrays. X and y must be the same length. Try again?")

    except AttributeError:
        print("¯\_(ツ)_/¯ Error: Check that X and y are dataframes or arrays. X and y must be the same length. Try again?")
        raise AttributeError("¯\_(ツ)_/¯ Error: Check that X and y are dataframes or arrays. X and y must be the same length. Try again?")


    finally:
        X_train_validation, X_test, y_train_validation, y_test = train_test_split(X, y, test_size= ptest)

        validation_ratio = round(pvalid/(ptrain + pvalid),2)

        X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation, y_train_validation, test_size=validation_ratio)


    return X_train,y_train,X_validation, y_validation,X_train_validation,y_train_validation,X_test,y_test
