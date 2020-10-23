"""
Train Baseline Linear Regression model using just GrLivArea
"""

import logging
import pickle

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split

from kaggle_house_prices.data.make_dataset import load_training_dataset
from kaggle_house_prices.logs import configure_logging

mlflow.set_experiment("kaggle_house_prices")


def log_transform_sale_price(df):
    df.SalePrice = np.log1p(df.SalePrice)
    return df


def rmse_cv_score(model, feature_values, target_values, scorer):
    return np.sqrt(-cross_val_score(model,
                                    feature_values,
                                    target_values,
                                    scoring=scorer, cv=10))


class ModelDefinition:
    def __init__(self):
        self.estimator = LinearRegression()
        self.model = None
        self.target = "SalePrice"
        self.input_feature_columns = ["GrLivArea"]

    def target_variable(self):
        return self.target

    def input_feature_names(self):
        return self.input_feature_columns

    def fit(self, input_features, y):
        self.model = self.estimator.fit(input_features, y)

    def predict(self, input_features):
        return self.model.predict(input_features)

    def score(self, input_features, target_values):
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        return rmse_cv_score(self.model, input_features, target_values, scorer).mean()

    def save(self, model_filename):
        with open(model_filename, "wb") as model_file_pointer:
            pickle.dump(self.model, model_file_pointer)


class PredictionPlot:
    def __init__(self,
                 title="Predicted vs Actual",
                 training_colour="blue", test_colour="lightgreen"):
        self.training_colour = training_colour
        self.test_colour = test_colour
        self.plot_title = title

        self.fig, self.ax = plt.subplots(1, 1)

    def plot(self, training_predictions, test_predictions, training_actual, test_actual):
        self.ax.scatter(training_predictions,
                        training_actual,
                        c=self.training_colour,
                        marker="s",
                        label="Training data")
        self.ax.scatter(test_predictions,
                        test_actual,
                        c=self.test_colour,
                        marker="s",
                        label="Validation")
        self.ax.set_title(self.plot_title)
        self.ax.set_xlabel("Predicted Values")
        self.ax.set_ylabel("Real Values")
        self.ax.legend(loc="upper left")
        self.ax.plot([10.5, 13.5], [10.5, 13.5], c="red")

        return self.ax

    def save(self, file_path):
        self.fig.savefig(file_path)


def plot_residuals(training_predictions, test_predictions, training_actual, test_actual):
    plt.scatter(training_predictions,
                training_predictions - training_actual,
                c="blue",
                marker="s",
                label="Training data")
    plt.scatter(test_predictions,
                test_predictions - test_actual,
                c="lightgreen",
                marker="s",
                label="Validation data")
    plt.title("Linear Regression")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.legend(loc="upper left")
    plt.hlines(y=0, xmin=10.5, xmax=13.5, color="red")


def preprocessing_pipeline(model_definition):
    train_df = (load_training_dataset()
                .pipe(log_transform_sale_price))

    y = train_df[model_definition.target_variable()]
    train_df = train_df[model_definition.input_feature_names()]

    X_train, X_test, y_train, y_test = train_test_split(
        train_df,
        y,
        test_size=0.3,
        random_state=0
    )

    return X_train, X_test, y_train, y_test


def train_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    # assess vs training and test set
    training_rmse = model.score(X_train, y_train)
    logging.info(f"RMSE on Training set: {training_rmse}")

    test_rmse = model.score(X_test, y_test)
    logging.info(f"RMSE on Test set: {test_rmse}")

    # diagnostic plots
    y_training_predictions = model.predict(X_train)
    y_test_predictions = model.predict(X_test)

    # Plot residuals
    plot_residuals(y_training_predictions, y_test_predictions, y_train, y_test)
    residuals_path = "./reports/figures/baseline/baseline_linear_regression_residuals.png"
    plt.savefig(residuals_path)

    # Plot predictions vs actual
    predicted_plot = PredictionPlot(title="Linear Regression")
    predicted_plot.plot(y_training_predictions, y_test_predictions, y_train, y_test)

    predictions_path = "./reports/figures/baseline/baseline_linear_regression_predictions.png"
    predicted_plot.save(predictions_path)

    model_path = "models/baseline/baseline_model.pickle"
    model.save(model_path)


if __name__ == "__main__":
    configure_logging()

    model = ModelDefinition()

    X_train, X_test, y_train, y_test = preprocessing_pipeline(model)

    train_model(model, X_train, X_test, y_train, y_test)
