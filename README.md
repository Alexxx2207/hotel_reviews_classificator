# Introduction (Short)

The project classifies hotel review by whether they are positive or negative. It uses multilayer perception model and logistic regression baseline. The vectorizer is TfIdf with 1 or 2 ngrams.

Some features are:

- streamlit project for testing the MLP and the baseline with your custom reviews
- plots for the mlp loss and f1 score in terms of the epochs
- confusion matrix plots for MLP and baseline
- MLP and baseline test reports in JSON for perception, recall, F1 asn support for each class and with accuracy, macro and weighted average.

# How to start it

This guide infers that you are in the root directory of the project(where this README is at).

For streamlit app demo:

1. `uv run python -m src.tripadvisor_setup`
2. `uv run python -m src.train`
3. `uv run python -m streamlit run ./app/streamlit_app.py`

For evaluation:

1. `uv run python -m src.tripadvisor_setup`
2. `uv run python -m src.train`
3. `uv run python -m src.evaluate`

You can run the tests with `uv run pytest`

# Structure

## Source code

The code is in the src folder. There are top level modules:

- tripadvisor setup: downloads tripadvisor reviews dataset from kagglehub. The reviews are from 1-5 with a text. It removes the reviews with score of 3 and sets the reviews with 1 and 2 to 0(negative review) and 4 and 5 to 1(positive review). It also splits the data into train and test with 80/20 ratio, respectively. It saves the csv data in ./data folder.

- train: trains baseline and MLP model with the training set. It uses pytorch with Adam optimizer and Cross Entropy loss function. It uses tqdm for simple training visualization in the terminal.

- mlp: defines the MLP model's parameters and forward behavior. Also defines how batches of data are created. The MLP has a single hidden layer with bias, ReLU activation and dropout for overfitting prevention.

- features: handles feature matrix operations like fitting and transformation of a review to a vector. It also saves the vectorizer in a joblib format.

- evaluate: tests MLP and baseline models with the test dataset. It creates test_report and confusion matrix for each one.

- constants: I think is self-explanatory - all constants for the project

- baseline: handles logistic regression classifier training and joblib format saves\loads.

## Tests

Pytest testing framework is used. A custom .tmp folder is created for saving the artifacts that are being expected. It has 98% coverage, but at the expense of some meaningless tests like constants checks.
