import click
import pandas as pd
import os

@click.command()
@click.option('--input', type=str, help='Path to the input raw data file', required=True)
@click.option('--output_dir', type=str, help='Directory to save cleaned data', required=True)
@click.option('--missing_threshold', type=float, default=0.05, help='Threshold for missing values (default: 5%)')
def clean_data(input, output_dir, missing_threshold):
    """
    Cleans the data by handling missing values, validating data types, and adding a binary column.

    Args:
    - input (str): Path to the raw data file.
    - output_dir (str): Directory to save the cleaned file.
    - missing_threshold (float): Maximum acceptable proportion of missing values per column.
    """
    try:
        # Load raw data
        print(f"Loading raw data from {input}...")
        data = pd.read_csv(input, sep=';')

        # Validate required columns
        expected_columns = [
            "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
            "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
            "pH", "sulphates", "alcohol", "quality"
        ]
        missing_columns = [col for col in expected_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in dataset: {missing_columns}")
        
        # Drop excessive missing values
        print("Checking for missing values...")
        missing_percentage = data.isnull().mean()
        exceeding_columns = missing_percentage[missing_percentage > missing_threshold].index.tolist()
        if exceeding_columns:
            print(f"Dropping columns with excessive missingness: {exceeding_columns}")
            data = data.drop(columns=exceeding_columns)
        data = data.dropna()

        # Validate data types
        print("Validating data types...")
        expected_dtypes = {
            "fixed acidity": float, "volatile acidity": float, "citric acid": float,
            "residual sugar": float, "chlorides": float, "free sulfur dioxide": float,
            "total sulfur dioxide": float, "density": float, "pH": float, "sulphates": float,
            "alcohol": float, "quality": int
        }
        for column, dtype in expected_dtypes.items():
            if column in data.columns and not pd.api.types.is_dtype_equal(data[column].dtype, dtype):
                data[column] = data[column].astype(dtype)

        # Add binary column
        print("Creating binary column 'is_good'...")
        data["is_good"] = (data["quality"] > 5).astype(int)

        # Check for duplicate rows
        print("Checking for duplicates...")
        duplicate_rows = data.duplicated().sum()
        if duplicate_rows > 0:
            print(f"Removing {duplicate_rows} duplicate rows...")
            data = data.drop_duplicates()

        # Save cleaned dataset
        os.makedirs(output_dir, exist_ok=True)
        cleaned_data_path = os.path.join(output_dir, "cleaned_data.csv")
        data.to_csv(cleaned_data_path, index=False)
        print(f"Cleaned data saved to {cleaned_data_path}")

    except Exception as e:
        print(f"Error during data cleaning: {e}")

if __name__ == '__main__':
    clean_data()
