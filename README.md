
# SklearncomPYre
Regression and Classification model comparison wrapper around scikit-learn

| Contributor                  | Github              | CWL |
| --------------------- |-----------------------|-----------------------|
| Jes Simkin | [jessimk](https://github.com/jessimk) | jess354 |
| Birinder Singh | [birinder1469](https://github.com/Birinder1469) | bsingh02 |
| Talha Siddiqui | [talhaadnan100](https://github.com/talhaadnan100) | talhaas |


### Motivation and Summary

The SklearncomPYre comprises of set functions doing a broad level comparison between the different Machine learning models for regression and Classification problems. One of the function has the functionality of splitting the X and y inputs in the Training.


a summary paragraph that describes the project at a high level

a bulleted list of the functions (and datasets if applicable) that will be included in the package (this should be a 1-2 sentence description for each function/dataset)

__Function 1 :__

- `train_test_acc()`
<br>
    Takes in a dictionary of sklearn regressors or classifiers, train input samples `Xtrain` (input features) , test input samples `Xtest`, train target values `ytrain` and test target values `ytest` (continuous or categorical).
    The function outputs a beautiful dataframe with train and test scores, variance and ordered in ascending order of test scores.  <br>

    __Inputs:__  <br>
	- Dictionary of ML classifiers or regressors,  format ->  ` dict { key: char, value: ML model }`
                where
                - key : is the name of the classifier such as 'SVM','RandomForest','LogisticRegression'
                - value : SVC(), LogisticRegression(), RandomForest()
	- X train set, type: `Array-like `
	- Y train set, type: `Array-like`
	- X test set, type: `Array-like `
	- Y test set, type: `Array-like`

  __Outputs:__ <br>
  - dataframe of training and test scores sorted by test score in ascending order and the variance.


__Function 2 :__

- `train_test_time()`  <br>
  The function creates a comparison of time it takes to fit and predict using different models. It takes in a dictionary of sklearn regressors or classifiers, training input samples `X`, and target values `y` (class labels in classification, real numbers in regression). <br>
  Outputs a beautiful dataframe with the time it takes for each model to fit and predict. <br>

  __Inputs:__   

  - Dictionary of ML classifiers,  format ->  ` dict{ key: char, value: ML model }`
                where
                - key : is the name of the classifier such as 'SVM','RandomForest','LogisticRegression'
                - value : SVC(), LogisticRegression(), RandomForest()
	- X set, type: `Array-like `
	- Y set, type: `Array-like`

  __Outputs:__<br>

  - dataframe of time taken for the fit and predict for each classifier or regressor input.

__Function 3 :__

- `split()`

  The function splits the training input samples `X`, and target values `y` (class labels in classification, real numbers in regression) into train, test and validation sets according to specified proportions.

    Outputs four array like training, validation, test, and combined training and validation sets and four y arrays. <br>

    __Inputs:__
    - X set, type: `Array like `
    - Y set, type: `Array like`
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

a paragraph describing where your packages fit into the Python and R ecosystems 
(are there any other Python or R software packages that have the same/similar functionality? 
Provide links to any that do. if none exist, then clearly state this as well).


