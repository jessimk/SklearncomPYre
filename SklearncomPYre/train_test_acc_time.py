
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def train_test_acc_time(models,X_train,y_train,X_test,y_test):

    """
    The purpose of this function is to compare different sklearn regressors or classifiers in terms
    of training and test accuracies, and the time it takes to fit and predict.
    The function inputs are dictionary of models, input train samples `Xtrain`(input features),
    input test samples `Xtest`, target train values `ytrain` and target test values `ytest`
    (continuous or categorical).

    The function outputs a beautiful dataframe with training & test scores,
    model variance, and the time it takes to fit and predict using different models.
    Inputs:
    - Dictionary of ML classifiers or regressors.
    - X train set: `Array-like `
    - Y train set: `Array-like`
    - X test set: `Array-like `
    - Y test set: `Array-like`
    Outputs:
    Dataframe with 7 columns:
    - regressor or classifier name
    - training accuracy
    - test accuracy
    - model variance
    - time it takes to fit
    - time it takes to predict
    - total time

    The dataframe will be sorted by test score in descending order.

    """
    #initializing an empty results dictionary, an empty list and an empty dataframe
    results_dict = {'Model':[],
        'Train Accuracy':[],
        'Test Accuracy':[],
        'Fit Time':[],
        'Predict Time':[]}


    results_row =[]
    results_df = pd.DataFrame()

    if  type(models) != dict:
        raise TypeError("Models aren't the right type. Try again ¯\_(ツ)_/¯ ")

    inputs = [X_train, y_train, X_test, y_test]

    for dataset in inputs:
        if  type(dataset) != pd.DataFrame and type(dataset) != np.ndarray:
            raise TypeError("Input datasets aren't the right type. Try again ¯\_(ツ)_/¯ ")

    for model_name, model in models.items():
    # Train models and populate the results_dict
        t = time.time()
        model.fit(X_train, y_train)

        results_dict['Model'] = model_name
        results_dict['Fit Time'] = time.time() - t
        results_dict['Train Accuracy'] = model.score(X_train, y_train)

        t = time.time()
        model.predict(X_test)
        results_dict['Predict Time'] = time.time() - t

        results_dict['Test Accuracy'] = model.score(X_test, y_test)

        results_row.append(results_dict)
        results_df = results_df.append(results_row)
        results_row.pop()

    results_df = results_df.reset_index()
    results_df=results_df.drop(['index'],axis=1)

    results_df["Variance"] = results_df["Train Accuracy"] - results_df["Test Accuracy"]
    results_df["Total Time"] = results_df["Fit Time"] - results_df["Predict Time"]

    results_df.sort_values(by='Test Accuracy', ascending=False)

    cols = ['Model','Train Accuracy', 'Test Accuracy', 'Variance', 'Fit Time', 'Predict Time', 'Total Time']
    results_df = results_df[cols]
    results_df

    return results_df
