all : report/wine_quality_analysis.html

# download data
data/raw/wine_quality.zip \
data/raw/winequality-red.csv \
data/raw/winequality-white.csv \
data/raw/winequality.names : scripts/download_and_extract.py
	python scripts/download_and_extract.py \
		--url "https://archive.ics.uci.edu/static/public/186/wine+quality.zip" \
		--output_dir data/raw

# clean and split
data/processed/train_data.csv \
data/processed/test_data.csv : scripts/clean_and_split_data.py
	python scripts/clean_and_split_data.py \
		--input data/raw/winequality-red.csv \
		--output_dir data/processed

# EDA
results/data_describe.csv \
results/example.csv \
results/eda_plot.png : scripts/eda.py
	python scripts/eda.py \
		--raw-data=data/raw/winequality-red.csv \
		--training-data=data/processed/train_data.csv \
		--plot-to=results

# fit and evaluate
results/results.png \
results/confmatrix.png \
results/feature_importance.csv \
results/model_score_dataframe.csv \
results/conf_matrix_df.csv : scripts/fit_and_evaluate.py
	python scripts/fit_and_evaluate.py \
		--training-data=data/processed/train_data.csv \
		--test-data=data/processed/test_data.csv \
		--plot-to=results

# write the report
report/wine_quality_analysis.html : report/wine_quality_analysis.qmd \
results/data_describe.csv \
results/example.csv \
results/eda_plot.png \
results/results.png \
results/confmatrix.png \
results/feature_importance.csv \
results/model_score_dataframe.csv \
results/conf_matrix_df.csv
	quarto render report/wine_quality_analysis.qmd
	cp report/wine_quality_analysis.html .
	mv wine_quality_analysis.html index.html

clean :
	rm -f results/*
	rm -f report/wine_quality_analysis.html