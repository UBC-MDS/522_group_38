# eda.py
# author: Tengwei Wang
# date: 2024-12-04

import os
import altair as alt
import click
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

@click.command()
@click.option('--training-data', type=str, help="Path to training data")
@click.option('--test-data', type=str, help="Path to test data")
@click.option('--plot-to', type=str, help="Path to directory where the plot will be written to")
def main(training_data, test_data, plot_to):

    train_data = pd.read_csv(training_data)
    test_data = pd.read_csv(test_data)
    x_train = (train_data.iloc[:, :-2])
    y_train = (train_data["is_good"])
    x_test = (test_data.iloc[:, :-2])
    y_test = (test_data[])
    
    depths = np.arange(1, 15, 1)
    
    results_dict = {
        "depth": [],
        "mean_train_accuracy": [],
        "mean_cv_accuracy": []}
    param_grid = {"max_depth": np.arange(1, 15, 1)}
    
    for depth in param_grid["max_depth"]:
        model = DecisionTreeClassifier(max_depth=depth, random_state = 123)
        scores = cross_validate(model, x_train, y_train, cv=5, return_train_score=True)
        results_dict["depth"].append(depth)
        results_dict["mean_cv_accuracy"].append(np.mean(scores["test_score"]))
        results_dict["mean_train_accuracy"].append(np.mean(scores["train_score"]))
    
    results_df = pd.DataFrame(results_dict).set_index('depth')
    
    results_df[["mean_train_accuracy", "mean_cv_accuracy"]].plot()

    best_model = DecisionTreeClassifier(max_depth=4, random_state=123)
    best_model.fit(x_train, y_train) 
    best_model.score(x_test, y_test)

    confmatrix = ConfusionMatrixDisplay.from_predictions(y_test, cross_val_predict(best_model, x_test, y_test))

    model_importances = best_model.feature_importances_
    feature_name = x_train.columns
    
    feature_importance_dataframe = pd.DataFrame(feature_name, 
    model_importances).reset_index().sort_values(by = 'index', ascending = False)
    