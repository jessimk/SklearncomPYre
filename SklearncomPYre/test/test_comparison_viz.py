import os
import pandas as pd
import pytest
import glob
import matplotlib as plt

from SklearncomPYre import comparison_viz

# Test Dataframe
df = pd.DataFrame({'Models': ["Random Forest","MLPClassifier", "Logistic Regression","GaussianNB"],
                   'Train Accuracy': [90.1, 82.2, 73.3, 74.4],
                   'Test Accuracy': [85.4, 78.3, 69.2, 70.1],
                   'Variance': [5.1, 3.4, 5.6, 3.2],
                   'Fit Time (s)': [3, 2, 2.5, 1.2],
                   'Predict Time (s)': [1.1, 1.4, 0.9, 2.2],
                   'Total Time (s)': [4.5, 6.6, 4.3, 2.1]
                  })

def test_input_choice_value():
    """
    Testing for entering an invalid string as a choice.
    """
    try:
        comparison_viz(comparison=df, choice = 'model')
    except(ValueError):
        assert True
    else:
        assert False

def test_input_choice_type():
    """
    Testing for entering an invalid type for choice.
    """
    try:
        comparison_viz(comparison=df, choice=3)
    except(TypeError):
        assert True
    else:
        assert False


def test_input_comparison_value():
    """
    Testing for entering an invalid dataframe as a comparison.
    """
    try:
        comparison_viz(comparison=df[['Models','Variance']], choice='time')
    except(ValueError):
        assert True
    else:
        assert False

def test_input_comparison_type():
    """
    Testing for entering an invalid type for choice.
    """
    try:
        comparison_viz(comparison=3, choice='time')
    except(TypeError):
        assert True
    else:
        assert False

def test_output_comparison_viz_time():
    """
    Testing for output from appropriate comparison dataframe and choice time.
    """
    comparison_viz(comparison=df, choice='time')
    assert "comparison.png" in os.listdir()


def test_output_comparison_viz_accuracy():
    """
    Testing for output from appropriate comprison dataframe and choice.
    """
    comparison_viz(comparison=df, choice='accuracy')
    assert "comparison.png" in os.listdir()


def test_input_df_without_rows():
    """
    Comparison dataframe has no rows
    """
    try:
        comparison_viz(comparison=df.drop([0,1,2,3]), choice='time')
    except(ValueError):
        assert True


def test_input_df_model_column_not_string():
    """
    Comparison dataframe model column not string
    """
    try:
        df['Models'] = 1
        comparison_viz(comparison=df, choice='time')
    except(TypeError):
        assert True


def test_input_df_other_columns_not_float():
    """
    Comparison dataframe other columns not float
    """
    try:
        df['Variance'] = "a"
        comparison_viz(comparison=df, choice='time')
    except(TypeError):
        assert True
