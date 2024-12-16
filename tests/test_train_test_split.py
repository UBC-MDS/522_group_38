import pytest
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.train_test_split_func import split_data

# Mock data for testing
valid_data = pd.DataFrame({
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
    "alcohol": [9.4, 9.9, 9.8, 9.6, 9.7, 9.5, 9.3],
    "is_good": [1, 0, 1, 0, 1, 0, 1]
})

# Test case 1: Valid input
def test_split_data_valid_input(tmp_path):
    input_path = tmp_path / "valid_data.csv"
    output_dir = tmp_path / "output"
    valid_data.to_csv(input_path, index=False)
    
    result = split_data(str(input_path), str(output_dir), train_size=0.8, random_state=123)
    
    # Assert files are created
    for key, path in result.items():
        assert os.path.exists(path)
    
    # Load split data and verify sizes
    x_train = pd.read_csv(result["x_train"])
    y_train = pd.read_csv(result["y_train"])
    x_test = pd.read_csv(result["x_test"])
    y_test = pd.read_csv(result["y_test"])
    
    assert len(x_train) == 5  # 80% of 7 rows
    assert len(x_test) == 2  # 20% of 7 rows
    assert len(y_train) == 5
    assert len(y_test) == 2

# Test case 2: Missing `is_good` column
def test_split_data_missing_target_column(tmp_path):
    input_path = tmp_path / "invalid_data.csv"
    output_dir = tmp_path / "output"
    
    invalid_data = valid_data.drop(columns=["is_good"])
    invalid_data.to_csv(input_path, index=False)
    
    with pytest.raises(KeyError):
        split_data(str(input_path), str(output_dir), train_size=0.8, random_state=123)

# Test case 3: Empty dataset
def test_split_data_empty_input(tmp_path):
    input_path = tmp_path / "empty_data.csv"
    output_dir = tmp_path / "output"
    
    empty_data = pd.DataFrame()
    empty_data.to_csv(input_path, index=False)
    
    with pytest.raises(ValueError):
        split_data(str(input_path), str(output_dir), train_size=0.8, random_state=123)

# Test case 4: Invalid `train_size`
def test_split_data_invalid_train_size(tmp_path):
    input_path = tmp_path / "valid_data.csv"
    output_dir = tmp_path / "output"
    valid_data.to_csv(input_path, index=False)
    
    with pytest.raises(ValueError):
        split_data(str(input_path), str(output_dir), train_size=1.5, random_state=123)

# Test case 5: Invalid file path
def test_split_data_invalid_file_path(tmp_path):
    output_dir = tmp_path / "output"
    
    with pytest.raises(FileNotFoundError):
        split_data("non_existent_file.csv", str(output_dir), train_size=0.8, random_state=123)
