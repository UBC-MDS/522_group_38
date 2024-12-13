import pytest
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.eda_func import generate_example, generate_describe

# test data

test_data = pd.DataFrame({
    "fixed acidity":[7.1,7.5,7.6,7.8,7.2,7.7],
    "volatile acidity":[0.8,0.7,0.5,0.9,0.6,0.3],
    "citric acid":[0,0.12,0.11,0,0.15,0.23],
    "residual sugar":[1.9,1.2,1.6,1.8,1.9,1.3],
    "chlorides":[0.076,0.045,0.074,0.056,0.087,0.073],
    "free sulfur dioxide":[11,16,18,12,13,16],
    "total sulfur dioxide":[34,26,75,34,56,74],
    "density":[0.9987,0.9876,0.9786,0.9564,0.9922,0.9999],
    "pH":[3.51,4.21,3.66,3.21,3.01,3.22],
    "sulphates":[0.54,0.55,0.56,0.66,0.53,0.51],
    "alcohol":[9.4,9.9,9.8,9.6,9.6,9.5],
    "quality":[5,6,1,2,6,7]
})

# case: wrong input data type

def test_generate_example_type():
    
    with pytest.raises(TypeError):
        generate_example(1,'tests')
        
    with pytest.raises(TypeError):
        generate_example(test_data,1)

def test_generate_describe_type():
    
    with pytest.raises(TypeError):
        generate_describe(1,'tests')
        
    with pytest.raises(TypeError):
        generate_describe(test_data,1)

# case: input empty data frame

def test_generate_example_empty():

    with pytest.raises(ValueError):
        generate_example(pd.DataFrame(),'tests')

def test_generate_describe_empty():

    with pytest.raises(ValueError):
        generate_describe(pd.DataFrame(),'tests')

# case: success

def test_generate_example_success():

    generate_example(test_data,'tests')

    # Check if file exists
    assert os.path.isfile('tests/example.csv')

    # Validate csv file
    example = pd.read_csv('tests/example.csv')
    assert len(example) == 5

    os.remove('tests/example.csv')

def test_generate_describe_success():

    generate_describe(test_data,'tests')

    # Check if file exists
    assert os.path.isfile('tests/data_describe.csv')

    # Validate csv file
    describe = pd.read_csv('tests/data_describe.csv')
    assert len(describe) > 0

    os.remove('tests/data_describe.csv')
    
    