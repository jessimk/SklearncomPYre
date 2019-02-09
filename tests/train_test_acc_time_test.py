# import packages 
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

sys.path.append("../functions")

from train_test_acc_time import train_test_acc_time

# sample inputs
dictionary = {'knn': KNeighborsClassifier(), 'LogRegression':LogisticRegression() , 'RForest': RandomForestClassifier()}
digits = datasets.load_digits()
X, y = digits['data'], digits['target']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=123)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)


# test train_test_acc_time(dictionary, X_train, y_train, X_test, y_test)

# input type test for the Dictionary and the Input Arrays (X_train, y_train, X_test, y_test)
def test_dict_type():
    with pytest.raises(TypeError):
        train_test_acc_time("Vancouver", X_train, y_train, X_test, y_test)
        train_test_acc_time(55, X_train, y_train, X_test, y_test)    
        train_test_acc_time([1,2,3], X_train, y_train, X_test, y_test)    
        train_test_acc_time((4,44,5), X_train, y_train, X_test, y_test)                                   
 
# Checking the input types for X_train, y_train, X_test, y_test
def test_X_array_type():
    with pytest.raises(TypeError):
        train_test_acc_time(dictionary, np.array([1,2,3]), y_train, X_test, y_test)
        train_test_acc_time(dictionary, "Vancouver", y_train, X_test, y_test)
        train_test_acc_time(dictionary, X_train, [5,6,7], X_test, y_test)
        train_test_acc_time(dictionary, X_train,y_train, (7,8,9), y_test)

 
# Output type test for the function. The output should be a DataFrame
def test_output_type():
    result_dataframe = train_test_acc_time(dictionary, X_train, y_train, X_test, y_test)
    assert(type(result_dataframe) == int), "The type of output  is not DataFrame"
    assert(type(result_dataframe) == list), "The type of output is not DataFrame"
    assert(type(result_dataframe.iloc[:,1]) != pd.core.series.Series), "The type the second column is not Series"

    # output columns number check , The output should have 7 columns corresponding to accuracies and time taken.
def test_output_dimension():
    result_dataframe = train_test_acc_time(dictionary, X_train, y_train, X_test, y_test)
    assert(result_dataframe.shape[1] !=7), "The number of columns in output DataFrame should be 7 "
  
 # test all inputs are given (dictionary, X_train, y_train, X_test, y_test) otherwise there will be an error
def test_inputs_given():
    with pytest.raises(ValueError):
        train_test_acc_time(dictionary,  y_train, X_test, y_test)
            
