
# SklearncomPYre


| Contributor                  | Github              | CWL |
| --------------------- |-----------------------|-----------------------|
| Birinder Singh | [birinder1469](https://github.com/Birinder1469) | bsingh02 |
| Jes Simkin | [jessimk](https://github.com/jessimk) | jess354 |
| Talha Siddiqui | [talhaadnan100](https://github.com/talhaadnan100) | talhaas |


### Summary

A `python` package facilitating beautifully efficient comparisons of machine learning classifiers and regression models.

__SklearncomPYre__ harnesses the power of <a href="https://scikit-learn.org/">scikit-learn</a>, combining it with <a href="https://pandas.pydata.org/">pandas</a> dataframes for easy, breezy, and beautiful machine learning regressor and classifier exploration.

__Function 1 :__

- `train_test_acc()`
    Takes in a dictionary of sklearn regressors or classifiers along with input train samples `Xtrain` (input features) , input test samples `Xtest`, target train values `ytrain` and target test values `ytest` (continuous or categorical).
    The function outputs a beautiful dataframe with train and test scores, variance and is ordered in ascending order of test scores.  <br>

  __Inputs:__


    - Dictionary of ML classifiers or regressors.
	- X train set, type: `Array-like `
	- Y train set, type: `Array-like`
	- X test set, type: `Array-like `
	- Y test set, type: `Array-like`

  __Outputs:__ <br>

  - DataFrame of training and test scores sorted by test score in ascending order and the variance.


__Function 2 :__

- `train_test_time()`  <br>
  The function creates a comparison of time it takes to fit and predict using different models. It takes in a dictionary of sklearn regressors or classifiers along with input training samples `X`, and target values `y` (class labels in classification, real numbers in regression). <br>
  Outputs a beautiful dataframe with the time it takes for each model to fit and predict. <br>

  __Inputs:__   

    - Dictionary of ML classifiers or regressors.
	- X set, type: `Array-like `
	- Y set, type: `Array-like`

  __Outputs:__<br>

  - dataframe of time taken for the fit and predict for each classifier or regressor input.

__Function 3 :__

- `split()`

  The function splits the training input samples `X`, and target values `y` (class labels in classification, real numbers in regression) into train, test and validation sets according to specified proportions.

    Outputs four array like training, validation, test, and combined training and validation sets and four y arrays. <br>

    __Inputs:__
    - X data set, type: `Array like `
    - Y data set, type: `Array like`
    - proportion of training data  , type: `float`
    - proportion of test data , type: `float`
    - proportion of validation data, type: `float`<br>

    __Outputs:__  

    - X train set, type: `Array like`
    - y train, type: `Array like`
    - X validation set, type: `Array like`
    - y validation, type: `Array like`
    - X train and validation set, type: `Array like`
    - y train and validation, type: `Array like`
    - X test set, type: `Array like`
    - y test, type: `Array like`

### Where does this package fit in?

This package provides functions to help make the early stages of model selection and exploration easier to cycle through and meaningfully compare.

Our idea for this package was to facilitate the comparison of machine learning classifiers and models. Our inspiration came from <a href="https://ubc-mds.github.io/descriptions/">UBC MDS DSCI 573</a> lab assignments where we learned to combine python's `sci-kit learn` with `pandas` in order to produce interpretable comparisons of train and test accuracies and time efficiencies across models.

We are not currently aware of any packages that combine `sci-kit learn` and `pandas` for efficient and interpretable model-to-model comparisons. We expect that this combination is used in practice and after having used it while learning machine learning techniques during our UBC MDS coursework, we thought it would be a good combination of tools to formally package together.   

We are aware of a <a href="">new package</a>, `sklearn-pandas` that combines `sci-kit learn` and `pandas` powers but this new package is tailored towards providing full-cycle machine learning functionality (feature selection, transformations, inputting/outputting pandas dataframes, etc.) rather than focusing facilitating model-to-model comparisons via dataframes.
