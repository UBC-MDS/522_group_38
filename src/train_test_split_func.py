# train_test_split.py
# author: Karlygash Zhakupbayeva
# date: 2024-12-16

from sklearn.model_selection import train_test_split
import pandas as pd
import os

def split_data(input_data_path, output_dir, train_size=0.8, random_state=123):
    """
    Splits the dataset into training and testing datasets and saves them.
    Parameters:
    ----------
    input_data_path : str
        Path to the cleaned dataset file.
    output_dir : str
        Directory to save the split datasets.
    train_size : float
        Proportion of the dataset to include in the training split (default: 0.8).
    random_state : int
        Random seed for reproducibility (default: 123).
    Returns:
    -------
    dict
        Paths to the saved training and testing datasets as a dictionary.
    Examples:
    --------
    >>> split_data("data/cleaned/cleaned_data.csv", "data/processed", train_size=0.8, random_state=123)
    """
    # Load the cleaned dataset
    data = pd.read_csv(input_data_path)

    # Verify that the required target column ('is_good') exists
    if 'is_good' not in data.columns:
        raise KeyError("Column 'is_good' not found in the dataset. Ensure the cleaned dataset contains the target column.")

    # Separate features (X) and target (y)
    x = data.drop(columns=['is_good'])
    y = data['is_good']

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=random_state)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the split datasets
    x_train_path = os.path.join(output_dir, "x_train.csv")
    y_train_path = os.path.join(output_dir, "y_train.csv")
    x_test_path = os.path.join(output_dir, "x_test.csv")
    y_test_path = os.path.join(output_dir, "y_test.csv")

    x_train.to_csv(x_train_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    x_test.to_csv(x_test_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    return {
        "x_train": x_train_path,
        "y_train": y_train_path,
        "x_test": x_test_path,
        "y_test": y_test_path
    }