import argparse
import pandas as pd
import matplotlib.pyplot as plt

# coding: utf-8

def comparison_viz(comparison, choice):
    '''
      
      The purpose of the function is to help User visualize the comparison of accuracies or time given in the comparison 
      dataframe. It takes in a dataframe with 7 attributes i.e. model name, training & test scores, model variance, 
      and the time it takes to fit, predict model and total time taken for both fit and predict. 
      
      The function outputs(saves in the root directory) a beautiful <a href="https://matplotlib.org">matplotlib</a> 
      bar chart comparison of different models' training and test scores or the time it takes to fit and predict. 
        
      Parameters :
      
      Inputs:   

      comparison : Dataframe with 7 columns
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
    
	inspiration: https://matplotlib.org/gallery/statistics/barchart_demo.html
    '''
    
    n_models = df.shape[0]

    if choice == 'accuracy':
        x = df.iloc[:,2]
        y = df.iloc[:,3]
        labels = ('Train Accuracy','Test Accuracy','Accuracy','Train and Test Accuracy by Model')
    elif choice == 'time':
        x = df.iloc[:,5]
        y = df.iloc[:,6]
        labels = ('Fit Time','Predict Time','Time (s)','Fit and Predict Time by Model')

    fig, ax = plt.subplots()

    index = np.arange(n_models)
    bar_width = 0.35
    opacity = 0.4

    rects1 = ax.bar(index, x,bar_width,
                    alpha=opacity, color='b',
                    label=labels[0])

    rects2 = ax.bar(index + bar_width, y, bar_width,
                    alpha=opacity, color='r',
                    label=labels[1])

    ax.set_xticklabels(df.iloc[:,1])
    ax.set_xlabel('Models')
    ax.set_ylabel(labels[2])
    ax.set_xticks(index + bar_width / 2)
    ax.set_title(labels[3])
    ax.legend()

    fig.tight_layout()

    plt.savefig('comparison.png', bbox_inches='tight')