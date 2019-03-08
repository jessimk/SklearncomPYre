# SklearncomPYre
[![Build Status](https://travis-ci.com/jessimk/SklearncomPYre.svg?branch=master)](https://travis-ci.com/jessimk/SklearncomPYre)
[![codecov](https://codecov.io/gh/jessimk/SklearncomPYre/branch/master/graph/badge.svg)](https://codecov.io/gh/jessimk/SklearncomPYre)
>>>>>>> 1940651bbb052f4830b0633d42338a799c479217

| Contributor                  | Github              | CWL |
| --------------------- |-----------------------|-----------------------|
| Birinder Singh | [birinder1469](https://github.com/Birinder1469) | bsingh02 |
| Jes Simkin | [jessimk](https://github.com/jessimk) | jess354 |
| Talha Siddiqui | [talhaadnan100](https://github.com/talhaadnan100) | talhaas |

<br>
A Python package facilitating beautifully efficient comparisons of machine learning classifiers and regression models.
<br>


### Dependencies
- `Python==3.6.8`
- `matplotlib==3.0.1`
- `numpy==1.15.4`
- `pandas==0.20.3`
- `scikit-learn==0.20.2`
- `scipy==1.2.0`


### Installation & Usage

Pleas use the following command to install the package. : <br>
`pip install git+https://github.com/UBC-MDS/SklearncomPYre.git`

Once installed, load the package using following commands :

`from SklearncomPYre.train_test_acc_time import train_test_acc_time` <br>
`from SklearncomPYre.comparison_viz import comparison_viz` <br>
`from SklearncomPYre.split import split`<br>

Here is an example of how you can use SklearncomPYre:

```
# Example usage

# Import libraries
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Importing SklearncomPYre
from SklearncomPYre.train_test_acc_time import train_test_acc_time
from SklearncomPYre.comparison_viz import comparison_viz
from SklearncomPYre.split import split

# Loading the handy iris dataset
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# Setting up a dictionary of classifiers to test

dictionary = {
    'knn': KNeighborsClassifier(),
    'LogRegression':LogisticRegression() ,
    'RForest': RandomForestClassifier()}

# Let's start by using the SklearncomPYre function split().

# Splitting up datasets into 40% training, 20% vaildation, and 40% tests sets.

X_train, y_train, X_val, y_val, X_train_val, y_train_val, X_test, y_test = split(X,y,0.4,0.2,0.4)

#Now, let's train some models and compare them in a pandas dataframe by using train_test_acc_time().

result = train_test_acc_time(dictionary,X_train,y_train,X_val,y_val)
result

# Next, let's take a look at some some plots with comparison_viz()

#Our plots will be saved to the working directory.

comparison_viz(result, "accuracy")
comparison_viz(result, 'time')

  ```

### Summary
__SklearncomPYre__ harnesses the power of <a href="https://scikit-learn.org/">scikit-learn</a>, combining it with <a href="https://pandas.pydata.org/">pandas</a> dataframes and <a href="https://matplotlib.org/">matplotlib</a> plots for easy, breezy, and beautiful machine learning exploration.  

#### <a href= "https://github.com/UBC-MDS/SklearncomPYre/blob/master/SklearncomPYre/split.py">Function 1:</a> `split()`

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

#### <a href="https://github.com/UBC-MDS/SklearncomPYre/blob/master/SklearncomPYre/train_test_acc_time.py">Function 2:</a>   `train_test_acc_time()`

The purpose of this function is to compare different sklearn regressors or classifiers in terms of training and test accuracies, and the time it takes to fit and predict. The function inputs are dictionary of models, input train samples `Xtrain`(input features), input test samples `Xtest`, target train values `ytrain` and target test values `ytest` (continuous or categorical).  

The function outputs a beautiful dataframe with training & test scores, model variance, and the time it takes to fit and predict using different models.  <br>

  __Inputs:__

   - Dictionary of ML classifiers or regressors.
   - X train set, type: `Array-like `
   - Y train set, type: `Array-like`
   - X test set, type: `Array-like `
   - Y test set, type: `Array-like`

  __Outputs:__

  - Dataframe with 7 columns: (1) regressor or classifier name, (2) training accuracy, (3) test accuracy, (4) model variance, (5) time it takes to fit, (6) time it takes to predict and (7) total time. The dataframe will be sorted by test score in descending order.


#### <a href="https://github.com/UBC-MDS/SklearncomPYre/blob/master/SklearncomPYre/comparison_viz.py">Function 3:</a> `comparison_viz()`

The purpose of this function is to visualize the output of `train_test_acc_time()` for easy communication and interpretation. The user has the choice to visualize a comparison of accuracies or time. It takes in a dataframe with 7 attributes i.e. model name, training & test scores, model variance, and the time it takes to fit, predict and total time.

Outputs a beautiful <a href="https://matplotlib.org">matplotlib</a> bar chart comparison of different models' training and test scores or the time it takes to fit and predict.

  __Inputs:__   

  - Dataframe with 7 columns: (1) regressor or classifier name, (2) training accuracy, (3) test accuracy, (4) model variance, (5) time it takes to fit, (6) time it takes to predict and (7) total time. Type: `pandas.Dataframe`
  - Choice of `accuracy` or `time`, with the default being 'accuracy' if no string is given. Type: `string`

  __Outputs:__

  - Bar chart of accuracies or time comparison by models saved to root directory. Type: `png`

### Where does this package fit in?

This package provides functions to help make the early stages of model selection and exploration easier to cycle through and meaningfully compare.

Our idea for this package was to facilitate the comparison of machine learning classifiers and models. Our inspiration came from <a href="https://ubc-mds.github.io/descriptions/">UBC MDS DSCI 573</a> lab assignments where we learned to combine python's `sci-kit learn` with `pandas` in order to produce interpretable comparisons of train and test accuracies and time efficiencies across models.

We are not currently aware of any packages that combine `sci-kit learn` and `pandas` for efficient and interpretable model-to-model comparisons. We expect that this combination is used in practice and after having used it while learning machine learning techniques during our UBC MDS coursework, we thought it would be a good combination of tools to formally package together.   

We are aware of a <a href="">new package</a>, `sklearn-pandas` that combines `sci-kit learn` and `pandas` powers but this new package is tailored towards providing full-cycle machine learning functionality (feature selection, transformations, inputting/outputting pandas dataframes, etc.) rather than focusing facilitating model-to-model comparisons via dataframes.


### Branch Coverage

The branch coverage results are as follows :

![branch_coverage](branch_coverage.PNG)
