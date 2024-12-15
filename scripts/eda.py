# eda.py
# author: Tengwei Wang
# date: 2024-12-04

import os
import altair as alt
import click
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.eda_func import generate_describe, generate_example

@click.command()
@click.option('--raw-data', type=str, help="Path to raw data")
@click.option('--training-data', type=str, help="Path to training data")
@click.option('--plot-to', type=str, help="Path to directory where the plot will be written to")
def main(raw_data, training_data, plot_to):
    
    eda(raw_data, training_data, plot_to)


def eda(raw_data, training_data, plot_to):

    # read data
    train_data = pd.read_csv(training_data)
    raw_data = pd.read_csv(raw_data,sep = ';')

    numeric_columns = numeric_columns = [
        'fixed acidity', 
        'volatile acidity', 
        'citric acid', 
        'residual sugar', 
        'chlorides', 
        'free sulfur dioxide', 
        'total sulfur dioxide', 
        'density', 
        'pH', 
        'sulphates', 
        'alcohol'
    ]
    
    plot = alt.Chart(train_data).mark_bar(opacity=0.7).encode(
        x=alt.X(alt.repeat()).type('quantitative').bin(maxbins=40),
        y=alt.Y('count()').stack(False),
        color='is_good:N'
    ).repeat(numeric_columns, columns = 3).properties(
    title="Overall Distribution of Numeric Columns by Wine Being Good or Not"
)

    # check directory not exist, create 
    
    if not os.path.isdir(plot_to):
        os.makedirs(plot_to)
        
    # save raw data describe

    generate_describe(raw_data, plot_to)

    # save examples of processed data

    generate_example(raw_data,plot_to)
    
    # save output
    
    plot.save(os.path.join(plot_to, "eda_plot.png"))



if __name__ == '__main__':
    main()