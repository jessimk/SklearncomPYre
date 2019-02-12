import os
import sys
import pandas as pd
import pytest
import matplotlib as plt

sys.path.insert(0, os.path.abspath("../SklearncomPYre"))

import comparison_viz

# Test Dataframe
df = pd.DataFrame({'index': [1, 2, 3, 4],
                   'models': ["Random Forest","MLPClassifier", "Logistic Regression","GaussianNB"],
                   'Train Accuracy': [90, 82, 73, 74], 
                   'Test Accuracy': [90, 82, 73, 74], 
                   'Variance': [90, 82, 73, 74],
                   'Predict Time (s)': [90, 82, 73, 74], 
                   'Fit Time (s)': [90, 82, 73, 74], 
                   'Total Time (s)': [90, 82, 73, 74]
                  })

def test_input_choice_value():
    """
    Testing for entering an invalid string as a choice.
    """
    try:
        comparison_viz.comparison_viz(comparison=df,  choice='model')
    except(ValueError):
        assert True
    else:
        assert False
        
def test_input_choice_type():
    """
    Testing for entering an invalid type for choice.
    """
    try:
        comparison_viz.comparison_viz(comparison=df,  choice=3)
    except(TypeError):
        assert True
    else:
        assert False


def test_input_comparison_value():
    """
    Testing for entering an invalid dataframe as a comparison.
    """
    try:
        comparison_viz.comparison_viz(comparison=df['Train Accuracy'],  choice='time')
    except(ValueError):
        assert True
    else:
        assert False
        
def test_input_comparison_type():
    """
    Testing for entering an invalid type for choice.
    """
    try:
        comparison_viz.comparison_viz(comparison=3,  choice='time')
    except(TypeError):
        assert True
    else:
        assert False
