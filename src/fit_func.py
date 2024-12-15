# fit_func.py
# author: Abdul Safdar
# date: 2024-12-12

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict

def fit_model(x_train, y_train, x_test, y_test, plot_to, max_depth):
    """
    Fits a decision tree classifier based off a specified max depth and 
    saves a confusion matrix as a display and an array as well as the
    score of the model as a csv. 

    Parameters:
    ----------
    x_train : DataFrame
        The input DataFrame with our train data for our features.
    y_train : Array
        The input array for our training data for our target column.
    x_test : DataFrame
        The input DataFrame containing our test data for all of the features of interest.
    y_test : Array
        The input array for our test data for our target column.
    plot_to : str
        The path to save figures as a file.
    max_depth : int
        Max depth value to choose for your decision tree classifier.

    Returns:
    -------
    null

    Examples:
    --------
    >>> fit_model(x_train, y_train, x_test, y_test, plot_to, 4)
    """
    # fit model with specified max depth with a random state of 123

    best_model = DecisionTreeClassifier(max_depth=max_depth, random_state=123)
    best_model.fit(x_train, y_train) 
    best_model.score(x_test, y_test)
  
    #generate a confusion matrix

    confmatrix = ConfusionMatrixDisplay.from_predictions(y_test, cross_val_predict(best_model, x_test, y_test))
    
    # if directory not exist, create 
    
    if not os.path.isdir(plot_to):
        os.makedirs(plot_to)

    # save confusion matrix as png file
    
    confmatrix.plot()
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(plot_to, "confmatrix.png"), dpi=300, bbox_inches="tight") 
    plt.close() 
    
    model_importances = best_model.feature_importances_
    feature_name = x_train.columns
    
    feature_importance_dataframe = pd.DataFrame(feature_name, 
        model_importances).reset_index().sort_values(by = 'index', ascending = False)

    # save feature importances as a csv
    
    feature_importance_dataframe.to_csv(os.path.join(plot_to, "feature_importance.csv"))

    # save best_model_score as a csv
    best_model_score = best_model.score(x_test, y_test)
    model_score_dataframe = pd.DataFrame({"Model Score": [best_model_score]})
    model_score_dataframe.to_csv(os.path.join(plot_to, "model_score_dataframe.csv"))

    # save confusion matrix as csv file
    conf_matrix_array = confusion_matrix(y_test, cross_val_predict(best_model, x_test, y_test))
    conf_matrix_df = pd.DataFrame(conf_matrix_array)
    conf_matrix_df.to_csv(os.path.join(plot_to, "conf_matrix_df.csv"))
