# SklearncomPYre

### Authors:
| Authors   | Github    | CWL |
| --------------------- |-----------------------|------------|
| Birinder Singh | [birinder1469](https://github.com/birinder1469/) | bsingh02 |
| Jes Simkin | [jessimk](https://github.com/jessimk/) | jess354 |
| Talha Siddiqui | [talhaadnan100](https://github.com/talhaadnan100/) | talhaas |

### Summary

A `Python` package facilitating beautifully efficient comparisons of machine learning classifiers and models.

__SklearncomPYre__ harnesses the power of <a href="https://scikit-learn.org/">scikit-learn</a>, combining it with <a href="https://pandas.pydata.org/">pandas</a> dataframes for easy, breezy, and beautiful machine learning model and classifier exploration. 

### Functions:

- `train_test_acc()`  
	*Takes in a dictionary of ML classifiers, an X train set, Y train set, X test set, Y test set.*

	*Outputs a beautiful dataframe with train and test accuracies for each of the given ML classifiers.*
	
	__Inputs:__   
	- Dictionary of ML classifiers,  type: `dict{key: char, value: ML model}`
	- X train set, type: ``
	- Y train set, type: ``
	- X test set, type: ``
	- Y test set, type: ``

	__Outputs:__
	
 
- `train_test_time()`  

	Takes in a list/dictionary of ML classifiers, an X train set, Y train set, X test set, Y test set.

	Outputs a beautiful dataframe with train and test accuracies for each of the given ML classifiers. 
	
	__Inputs:__. 
	
	__Outputs:__

- `split()`
	
	Takes in a list/dictionary of ML classifiers, an X train set, Y train set, X test set, Y test set.

	Outputs a beautiful dataframe with train and test accuracies for each of the given ML classifiers. 
	
	__Inputs:__. 
	
	__Outputs:__
 

### Where does this package fit in?

This package provides functions to help make the early stages of model selection and exploration easier to cycle through and meaningfully compare.

Our idea for this package was to facilitate the comparison of machine learning classifiers and models. Our inspiration came from <a href="https://ubc-mds.github.io/descriptions/">UBC MDS DSCI 573</a> lab assignments where we learned to combine python's `sci-kit learn` with `pandas` in order to produce interpretable comparisons of train and test accuracies and time efficiencies across models. 

We are not currently aware of any packages that combine `sci-kit learn` and `pandas` for efficient and interpretable model-to-model comparisons. We expect that this combination is used in practice and after having used it while learning machine learning techniques during our UBC MDS coursework, we thought it would be a good combination of tools to formally package together.   

We are aware of a <a href="">new package</a>, `sklearn-pandas` that combines `sci-kit learn` and `pandas` powers but this new package is tailored towards providing full-cycle machine learning functionality (feature selection, transformations, inputting/outputting pandas dataframes, etc.) rather than focusing facilitating model-to-model comparisons via dataframes.

