# eda.py
# author: Tengwei Wang
# date: 2024-12-04

import os
import altair as alt
import click
import pandas as pd

@click.command()
@click.option('--training-data', type=str, help="Path to training data")
@click.option('--plot-to', type=str, help="Path to directory where the plot will be written to")
def main(training_data, plot_to):

    # read data
    train_data = pd.read_csv(training_data)

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
    ).repeat(numeric_columns, columns = 3)

    plot.save(os.path.join(plot_to, "eda_plot.png"),
              scale_factor=2.0)

if __name__ == '__main__':
    main()