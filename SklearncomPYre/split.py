
# coding: utf-8

# In[ ]:


def split(X,y,ptrain,ptest,pvalid):
    
    
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
    
    """
    
    return (X_train,y_train,X_validation, y_validation,X_train_validation,y_train_validation,X_test,y_test)

