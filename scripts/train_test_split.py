import click
from src.train_test_split_func import split_data

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
    Script to split the cleaned dataset into training and testing datasets.

    Args:
    - input_data (str): Path to the cleaned dataset to be split.
    - output_dir (str): Directory to save the split datasets.
    - train_size (float): Fraction of data to reserve for training (default: 80%).
    - random_state (int): Seed value to ensure consistent splits across runs.
    """
    try:
        result = split_data(input_data, output_dir, train_size, random_state)
        print("Train-test split completed successfully.")
        print("Paths to saved files:")
        print(result)
    except Exception as e:
        print(f"Error during train-test split: {e}")

if __name__ == '__main__':
    train_test_split_script()
