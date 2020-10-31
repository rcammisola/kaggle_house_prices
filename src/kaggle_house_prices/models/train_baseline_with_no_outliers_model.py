import logging

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

import kaggle_house_prices.features.filter as feature_filter
import kaggle_house_prices.features.transform as transform
from kaggle_house_prices.data.make_dataset import load_training_dataset
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
    residuals_path = "./reports/figures/baseline_no_outliers/residuals.png"
    plt.savefig(residuals_path)

    # Plot predictions vs actual
    predicted_plot = PredictionPlot(title="Linear Regression (no_outliers)")
    predicted_plot.plot(y_training_predictions, y_test_predictions, y_train, y_test)

    predictions_path = "./reports/figures/baseline_no_outliers/predictions_vs_actuals.png"
    predicted_plot.save(predictions_path)

    model_path = "models/baseline_no_outliers/model.pickle"
    with open(model_path, "wb") as model_file_pointer:
        pickle.dump(model, model_file_pointer)


def preprocessing_pipeline_no_outliers(df):
    train_df = (df
                .copy()
                .pipe(transform.log_transform_sale_price)
                .pipe(feature_filter.filter_large_house_outliers))

    y = train_df["SalePrice"]
    train_df = train_df[["GrLivArea"]]
    X_train, X_test, y_train, y_test = train_test_split(
        train_df,
        y,
        test_size=0.3,
        random_state=0
    )
    return X_test, X_train, y_test, y_train


if __name__ == "__main__":
    configure_logging()

    dataset = load_training_dataset()

    logging.info("Baseline without outliers model")
    train_basic_model(dataset, preprocessing_pipeline_no_outliers)
