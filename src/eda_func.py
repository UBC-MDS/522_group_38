# eda_func.py
# author: Tengwei Wang
# date: 2024-12-12

import pandas as pd
import os

def generate_describe(df : pd.DataFrame, path : str):
    """
    Generate describe content of input data frame and save the result as 
    a .csv file to input path.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data to describe.
    path : str
        The path to save result as .csv file.

    Returns:
    -------
    null

    Examples:
    --------
    >>> import pandas as pd
    >>> df = pd.read_csv(training_data)
    >>> generate_describe(df, 'result')

    Notes:
    -----
    This function uses the pandas library to generate describe content.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
        
    if not isinstance(path, str):
        raise TypeError("Input must be a string")
        
    if df.empty:
        raise ValueError("DataFrame must contain observations.")
        
    describe = df.describe().round(3)
    
    describe.to_csv(os.path.join(path, "data_describe.csv"))

def generate_example(df : pd.DataFrame, path : str):
    """
    Generate processed examples of input data frame and save the result as 
    a .csv file to input path.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data to describe.
    path : str
        The path to save result as .csv file.

    Returns:
    -------
    null

    Examples:
    --------
    >>> import pandas as pd
    >>> df = pd.read_csv(training_data)
    >>> generate_example(df, 'result')
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if not isinstance(path, str):
        raise TypeError("Input must be a string")
    
    if df.empty:
        raise ValueError("DataFrame must contain observations.")
        
    example = df.iloc[:5, :]

    example = example.copy()
    
    example['is_good'] =  (example["quality"]>5)*1

    example.to_csv(os.path.join(path, "example.csv"))


    