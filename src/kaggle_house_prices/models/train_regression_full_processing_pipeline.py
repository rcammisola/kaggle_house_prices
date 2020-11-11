import logging

import matplotlib.pyplot as plt

from kaggle_house_prices.data.make_dataset import load_training_dataset
from kaggle_house_prices.features.pipelines import full_pipeline
from kaggle_house_prices.logs import configure_logging
from kaggle_house_prices.visualization.diagnostic import PredictionPlot, plot_residuals


import pickle

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_score


def rmse_cv_score(model, feature_values, target_values, scorer):
    return np.sqrt(-cross_val_score(model,
                                    feature_values,
                                    target_values,
                                    scoring=scorer, cv=10))


def train_basic_model(dataset, preprocessing_pipeline):
    X_train, X_test, y_train, y_test = preprocessing_pipeline(dataset)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # assess vs training and test set
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    training_rmse =  rmse_cv_score(model, X_train, y_train, scorer).mean()
    logging.info(f"RMSE on Training set: {training_rmse}")

    test_rmse = rmse_cv_score(model, X_test, y_test, scorer).mean()
    logging.info(f"RMSE on Test set: {test_rmse}")

    # diagnostic plots
    y_training_predictions = model.predict(X_train)
    y_test_predictions = model.predict(X_test)

    # Plot residuals
    plot_residuals(y_training_predictions, y_test_predictions, y_train, y_test)
    residuals_path = "./reports/figures/full_pipeline/residuals.png"
    plt.savefig(residuals_path)

    # Plot predictions vs actual
    predicted_plot = PredictionPlot(title="Linear Regression (full pipeline)")
    predicted_plot.plot(y_training_predictions, y_test_predictions, y_train, y_test)

    predictions_path = "./reports/figures/full_pipeline/predictions_vs_actuals.png"
    predicted_plot.save(predictions_path)

    model_path = "models/full_pipeline/model.pickle"
    with open(model_path, "wb") as model_file_pointer:
        pickle.dump(model, model_file_pointer)


if __name__ == "__main__":
    configure_logging()

    dataset = load_training_dataset()

    logging.info("Linear Regression with full pipeline")
    train_basic_model(dataset, full_pipeline)
