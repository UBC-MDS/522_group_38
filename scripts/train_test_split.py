import sys
import os

# Add the project root directory to Python's search path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.train_test_split_func import split_data
import click
import pandas as pd

@click.command()
@click.option(
    '--input-data', 
    type=str, 
    help='Path to the cleaned dataset file', 
    required=True
)
@click.option(
    '--output-dir', 
    type=str, 
    default='data/processed', 
    help='Directory where split datasets will be saved (default: data/processed)'
)
@click.option(
    '--train-size', 
    type=float, 
    default=0.8, 
    help='Proportion of the dataset to include in the training split (default: 0.8)'
)
@click.option(
    '--random-state', 
    type=int, 
    default=123, 
    help='Random seed for reproducibility (default: 123)'
)
def train_test_split_script(input_data, output_dir, train_size, random_state):
    """
    Script to split the cleaned dataset into training and testing datasets,
    and save them as combined train_data.csv and test_data.csv.

    Args:
    - input_data (str): Path to the cleaned dataset to be split.
    - output_dir (str): Directory to save the split datasets.
    - train_size (float): Fraction of data to reserve for training (default: 80%).
    - random_state (int): Seed value to ensure consistent splits across runs.
    """
    try:
        # Load the cleaned dataset
        data = pd.read_csv(input_data)

        # Verify that the required target column ('is_good') exists
        if 'is_good' not in data.columns:
            raise KeyError("Column 'is_good' not found in the dataset. Ensure the cleaned dataset contains the target column.")

        # Split the data into training and testing sets
        print("Splitting data into training and testing sets...")
        train_data = data.sample(frac=train_size, random_state=random_state)
        test_data = data.drop(train_data.index)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save combined train_data.csv and test_data.csv
        train_path = os.path.join(output_dir, "train_data.csv")
        test_path = os.path.join(output_dir, "test_data.csv")

        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

        print(f"Training data saved to {train_path}")
        print(f"Test data saved to {test_path}")

    except Exception as e:
        print(f"Error during train-test split: {e}")

if __name__ == '__main__':
    train_test_split_script()
