import logging

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

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


def training_process(dataset, preprocessing_pipeline):
    model_name = "baseline"

    X_train, X_test, y_train, y_test = preprocessing_pipeline(dataset)

    model = fit_model(X_train, y_train)

    evaluation_metrics(X_test, X_train, model, y_test, y_train)

    y_training_predictions = model.predict(X_train)
    y_test_predictions = model.predict(X_test)

    evaluation_plots(model_name, y_test, y_test_predictions, y_train, y_training_predictions)

    save_model(model, model_name)


def fit_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluation_metrics(X_test, X_train, model, y_test, y_train):
    # assess vs training and test set
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    training_rmse = rmse_cv_score(model, X_train, y_train, scorer).mean()
    logging.info(f"RMSE on Training set: {training_rmse}")
    test_rmse = rmse_cv_score(model, X_test, y_test, scorer).mean()
    logging.info(f"RMSE on Test set: {test_rmse}")


def evaluation_plots(model_name, y_test, y_test_predictions, y_train, y_training_predictions):
    # Plot residuals
    plot_residuals(y_training_predictions, y_test_predictions, y_train, y_test)
    residuals_path = f"./reports/figures/{model_name}/residuals.png"
    plt.savefig(residuals_path)
    # Plot predictions vs actual
    predicted_plot = PredictionPlot(title=f"Linear Regression ({model_name})")
    predicted_plot.plot(y_training_predictions, y_test_predictions, y_train, y_test)
    predictions_path = f"./reports/figures/{model_name}/predictions_vs_actuals.png"
    predicted_plot.save(predictions_path)


def save_model(model, model_name):
    model_path = f"models/{model_name}/model.pickle"
    with open(model_path, "wb") as model_file_pointer:
        pickle.dump(model, model_file_pointer)


def preprocessing_pipeline_baseline(df):
    train_df = (df
                .copy()
                .pipe(transform.log_transform_sale_price))

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

    logging.info("Baseline model")
    training_process(dataset, preprocessing_pipeline_baseline)
