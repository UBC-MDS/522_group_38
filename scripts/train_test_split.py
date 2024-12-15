import click
import pandas as pd
from sklearn.model_selection import train_test_split
import os

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
    Splits the cleaned dataset into training and testing datasets, separates features (X) and target (y),
    and saves them into specified directories.

    Args:
    - input_data (str): Path to the cleaned dataset to be split.
    - output_dir (str): Directory to save the split datasets (default: data/processed).
    - train_size (float): Fraction of data to reserve for training (default: 80%).
    - random_state (int): Seed value to ensure consistent splits across runs.
    """
    try:
        # Load the cleaned dataset
        print(f"Loading cleaned dataset from {input_data}...")
        data = pd.read_csv(input_data)

        # Verify that the required target column ('is_good') exists
        if 'is_good' not in data.columns:
            raise KeyError("Column 'is_good' not found in the dataset. Ensure the cleaned dataset contains the target column.")

        # Separate features (X) and target (y)
        print("Separating features (X) and target (y)...")
        x = data.drop(columns=['is_good'])
        y = data['is_good']

        # Split the dataset into training and testing sets
        print(f"Splitting dataset into training and testing sets (train_size={train_size})...")
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=random_state)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save the split datasets
        x_train_path = os.path.join(output_dir, "x_train.csv")
        y_train_path = os.path.join(output_dir, "y_train.csv")
        x_test_path = os.path.join(output_dir, "x_test.csv")
        y_test_path = os.path.join(output_dir, "y_test.csv")

        print("Saving train and test datasets...")
        x_train.to_csv(x_train_path, index=False)
        y_train.to_csv(y_train_path, index=False)
        x_test.to_csv(x_test_path, index=False)
        y_test.to_csv(y_test_path, index=False)

        print(f"Train features saved to {x_train_path}")
        print(f"Train target saved to {y_train_path}")
        print(f"Test features saved to {x_test_path}")
        print(f"Test target saved to {y_test_path}")

    except Exception as e:
        print(f"Error during train-test split: {e}")

if __name__ == '__main__':
    train_test_split_script()
