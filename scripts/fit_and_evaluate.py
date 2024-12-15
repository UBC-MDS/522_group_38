# fit_and_evaluate.py
# author: Tengwei Wang
# editor: Abdul Safdar
# date: 2024-12-04

import sys
import os
import altair as alt
import click
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.fit_func import fit_model



@click.command()
@click.option('--training-data', type=str, help="Path to training data")
@click.option('--test-data', type=str, help="Path to test data")
@click.option('--plot-to', type=str, help="Path to directory where the plot will be written to")
def main(training_data, test_data, plot_to):
    """
    Cleans the data and splits it into training and test sets and performs hyperparameter optimization.

    Fits a decision tree model with specified max depth and saves best features, confusion matrix
    and model score. 

    Args:
    - training_data (dataframe): Training data dataframe.
    - test_data (dataframe): Dataframe of the test data.
    - plot_to (str): Directory to save the files.
    """
    # split data
    
    train_data = pd.read_csv(training_data)
    test_data = pd.read_csv(test_data)
    x_train = (train_data.iloc[:, :-2])
    y_train = (train_data["is_good"])
    x_test = (test_data.iloc[:, :-2])
    y_test = (test_data["is_good"])
    
    depths = np.arange(1, 15, 1)

    # build result dict
    
    results_dict = {
        "depth": [],
        "mean_train_accuracy": [],
        "mean_cv_accuracy": []}
    param_grid = {"max_depth": np.arange(1, 15, 1)}

    # cross validate for each depth in max depths 
    
    for depth in param_grid["max_depth"]:
        model = DecisionTreeClassifier(max_depth=depth, random_state = 123)
        scores = cross_validate(model, x_train, y_train, cv=5, return_train_score=True)
        results_dict["depth"].append(depth)
        results_dict["mean_cv_accuracy"].append(np.mean(scores["test_score"]))
        results_dict["mean_train_accuracy"].append(np.mean(scores["train_score"]))
    
    results_df = pd.DataFrame(results_dict).set_index('depth')

    # save the results
    
    results_df[["mean_train_accuracy", "mean_cv_accuracy"]].plot()
    plt.title('Mean Train Accuracy vs Mean CV Accuracy by Max Depth')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy Score')
    plt.savefig(os.path.join(plot_to, "results.png"), dpi=300, bbox_inches="tight") 
    plt.close()
    
    #call function fit_func 
    fit_model(x_train, y_train, x_test, y_test, plot_to, 4)
    
if __name__ == '__main__':
    main()
    