
# coding: utf-8

# In[ ]:


def comparison_viz(comparison, choice):
    
      """
      
      The purpose of the function is to help User visualize the comparison of accuracies or time given in the comparison 
      dataframe. It takes in a dataframe with 7 attributes i.e. model name, training & test scores, model variance, 
      and the time it takes to fit, predict model and total time taken for both fit and predict. 
      
      The function outputs(saves in the root directory) a beautiful <a href="https://matplotlib.org">matplotlib</a> 
      bar chart comparison of different models' training and test scores or the time it takes to fit and predict. 
        
      Parameters :
      
      Inputs:   

      comparison : Dataframe with 7 columns: 
      - regressor or classifier name 
      - training accuracy 
      - test accuracy 
      - model variance 
      - time it takes to fit 
      - time it takes to predict and 
      - total time. Type: `pandas.Dataframe
      - Choice of `accuracy` or `time`. Type: `string`

      Outputs:

      - Bar chart of accuracies or time comparison by models. Type: `png` 
                    
        """
    
    

