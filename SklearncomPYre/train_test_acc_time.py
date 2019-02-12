
# coding: utf-8

# In[ ]:


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
    results_dict = {'Classifier':[],
                    'Train Accuracy':[],
                    'Test Accuracy':[],
                    'Fit Time':[],
                    'Predict Time':[]}

    results_row =[]
    results_df = pd.DataFrame()

    for model_name, model in models.items():
        # Train & time models and populate the results_df
        t = time.time()
        model.fit(X_train, y_train)

        results_dict['Classifier'] = model_name
        results_dict['Fit Time'] = time.time() - t
        results_dict['Train Accuracy'] = model.score(X_train, y_train)

        #predict and time models and populate the results_df
        t = time.time()
        model.predict(X_test)
        results_dict['Predict Time'] = time.time() - t

        results_dict['Test Accuracy'] = model.score(X_test, y_test)

        results_row.append(results_dict)
        results_df = results_df.append(results_row)
        results_row.pop()

    #reset the index
    results_df = results_df.reset_index()
    results_df=results_df.drop(['index'],axis=1)

    #create new cols
    results_df["Variance"] = results_df["Train Accuracy"] - results_df["Test Accuracy"]
    results_df["Total Time"] = results_df["Fit Time"] - results_df["Predict Time"]

    #sort by test acc
    results_df.sort_values(by='Test Accuracy', ascending=False)

    return results_df
