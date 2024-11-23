# Wine Quality Analysis

Authors: Abdul Safdar, Karlygash Zhakupbayeva, Tengwei Wang

## Summary

That data set that we used in this project can be found [here](https://archive.ics.uci.edu/dataset/186/wine+quality)). It contains various features that were obtained through physicochemical tests that could be used to judge wine quality. We performed a classification analysis on our data set, where wine quality \> 5 was encoded as 1 (to indicate good quality) and the rest were encoded as a zero (to indicate not good).

In our report, we fit a simple decision tree classifier model and evaluated the best max depth hyperparameter for our decision tree model. Our results indicated that the most important features in predicting wine quality were alcohol, sulphates and volatile acidity. More information about the specific analysis and results can be found in the report linked below.

## Report

The final report can be found [here](https://github.com/UBC-MDS/522_group_38/blob/main/wine_quality_analysis.ipynb).

## Usage

To run this project, first clone this repository to your local machine using git clone.

Then run the following in the root of the repository in your terminal :

``` bash
conda-lock install --name wine_quality_env conda-lock.yml
```

To run the analysis file, run the following commands in the root of this repository in your terminal:

``` bash
jupyter lab 
```

Open `wine_quality_analysis.ipynb` in Jupyter Lab and under Switch/Select Kernel choose "Python [conda env:wine_quality_env]". Then restart the kernel and run all cells.

## Dependencies

-   `conda` (version 24.9.1 or higher)
-   `conda-lock` (version 2.5.7 or higher)
-   `jupyterlab` (version 4.2.4 or higher)
-   `mamba` (version 1.5.11 or higher)
-   Python and packages listed in [`environment.yaml`](environment.yaml)

## License

The project report is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License. To view a copy of this license, visit <https://creativecommons.org/licenses/by-nc-nd/4.0/>, and the MIT License. More information can be found in the LICENSE.md file in the repository.

## References

UCI Machine Learning Repository. Wine Quality Dataset. Available at: <https://archive.ics.uci.edu/ml/datasets/wine+quality>

Scikit-learn Documentation. Decision Trees. Available at: <https://scikit-learn.org/stable/modules/tree.html>

Kolhatkar, V. (2024). DSCI 571 Supervised Learning I Lecture 2 ML Fundamentals <https://pages.github.ubc.ca/mds-2024-25/DSCI_571_sup-learn-1_students/lectures/notes/02_ml-fundamentals.html>

Ostblom, J. (2024). DSCI 573 Feature and Model Selection, Lecture 1 Classification Metrics <https://pages.github.ubc.ca/mds-2024-25/DSCI_573_feat-model-select_students/lectures/01_classification-metrics.html>

VanderPlas, J., & Satyanarayan, A. (2018). Altair: Declarative Visualization in Python. <https://altair-viz.github.io/>
