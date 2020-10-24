import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from kaggle_house_prices.data.make_dataset import load_training_dataset
from kaggle_house_prices.logs import configure_logging
from kaggle_house_prices.models.baseline_model import ModelDefinition
from kaggle_house_prices.visualization.diagnostic import PredictionPlot, plot_residuals


def log_transform_sale_price(df):
    df.SalePrice = np.log1p(df.SalePrice)
    return df


def preprocessing_pipeline(dataset, model_definition):
    train_df = dataset.pipe(log_transform_sale_price)

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

    X_train, X_test, y_train, y_test = preprocessing_pipeline(load_training_dataset(), model)

    train_model(model, X_train, X_test, y_train, y_test)
