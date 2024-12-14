import pytest
import pandas as pd
import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.fit_func import fit_model

#Creating valid data types to use for testing purposes
valid_x_train = pd.DataFrame({
    "fixed acidity": [7.1, 7.5, 7.6, 7.8, 7.3, 7.6, 7.2],
    "volatile acidity": [0.8, 0.7, 0.5, 0.9, 0.6, 0.8, 0.7],
    "citric acid": [0, 0.12, 0.11, 0, 0.1, 0.15, 0.08],
    "residual sugar": [1.9, 1.2, 1.6, 1.8, 1.4, 1.5, 1.3],
    "chlorides": [0.076, 0.045, 0.074, 0.056, 0.065, 0.053, 0.062],
    "free sulfur dioxide": [11, 16, 18, 12, 14, 17, 15],
    "total sulfur dioxide": [34, 26, 75, 34, 45, 50, 40],
    "density": [0.9987, 0.9876, 0.9786, 0.9564, 0.9970, 0.9850, 0.9925],
    "pH": [3.51, 4.21, 3.66, 3.21, 3.5, 3.8, 3.7],
    "sulphates": [0.54, 0.55, 0.56, 0.66, 0.60, 0.57, 0.59],
    "alcohol": [9.4, 9.9, 9.8, 9.6, 9.7, 9.5, 9.3]
})

valid_y_train = pd.Series([0, 1, 0, 1, 0, 1, 0], name="quality")

valid_x_test = pd.DataFrame({
    "fixed acidity": [7.2, 7.7],
    "volatile acidity": [0.6, 0.3],
    "citric acid": [0.15, 0.23],
    "residual sugar": [1.9, 1.3],
    "chlorides": [0.087, 0.073],
    "free sulfur dioxide": [13, 16],
    "total sulfur dioxide": [56, 74],
    "density": [0.9922, 0.9999],
    "pH": [3.01, 3.22],
    "sulphates": [0.53, 0.51],
    "alcohol": [9.6, 9.5]
})

valid_y_test = pd.Series([0, 1], name="quality")

#Creating a dataframe to input into y_train
invalid_y_train = pd.DataFrame({
        "quality": [0, 1, 0, 1, 1, 0, 1]
    })

#Case 1: You pass a dataframe into y_train
def test_invalid_y_train_df():
    with pytest.raises(ValueError):
        fit_model(valid_x_train, invalid_y_train, valid_x_test, valid_y_test, './plots', 4)

#Case 2: empty data frame is passed into x_train
empty_df_x_train = pd.DataFrame([])
def test_empty_df():
    with pytest.raises(ValueError):
        fit_model(empty_df_x_train, valid_y_train, valid_x_test, valid_y_test, './plots', 4)

#Case 3: Empty series is passed into y train
empty_y_train = pd.Series([], name="quality")
def test_empty_series():
    with pytest.raises(ValueError):
        fit_model(valid_x_train, empty_y_train, valid_x_test, valid_y_test, './plots', 4)

#Case 4: the y_train and y_test passed are floating points and not integers 
invalid_y_train_float = pd.Series([0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5], name="quality")
invalid_y_test_float = pd.Series([0.5, 1.5], name="quality")

def test_invalid_dtype_y():
    with pytest.raises(ValueError):
        fit_model(valid_x_train, invalid_y_train_float, valid_x_test, invalid_y_test_float, './plots', 4)

