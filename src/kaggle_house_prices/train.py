import logging

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import kaggle_house_prices.features.filter as feature_filter
import kaggle_house_prices.features.transform as transform
from kaggle_house_prices.data.make_dataset import load_training_dataset
from kaggle_house_prices.logs import configure_logging
from kaggle_house_prices.models.baseline_model import ModelDefinition
from kaggle_house_prices.visualization.diagnostic import PredictionPlot, plot_residuals


def train_basic_model(model, dataset, preprocessing_pipeline):
    X_train, X_test, y_train, y_test = preprocessing_pipeline(dataset)

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
    residuals_path = "./reports/figures/baseline/residuals.png"
    plt.savefig(residuals_path)

    # Plot predictions vs actual
    predicted_plot = PredictionPlot(title="Linear Regression")
    predicted_plot.plot(y_training_predictions, y_test_predictions, y_train, y_test)

    predictions_path = "./reports/figures/baseline/predictions_vs_actuals.png"
    predicted_plot.save(predictions_path)

    model_path = "models/baseline/model.pickle"
    model.save(model_path)


def train_basic_model_with_outlier_removal(model, dataset, preprocessing_pipeline):
    X_train, X_test, y_train, y_test = preprocessing_pipeline(dataset)

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
    residuals_path = "./reports/figures/baseline_no_outliers/residuals.png"
    plt.savefig(residuals_path)

    # Plot predictions vs actual
    predicted_plot = PredictionPlot(title="Linear Regression (outliers removed)")
    predicted_plot.plot(y_training_predictions, y_test_predictions, y_train, y_test)

    predictions_path = "./reports/figures/baseline_no_outliers/predictions_vs_actuals.png"
    predicted_plot.save(predictions_path)

    model_path = "models/baseline_no_outliers/model.pickle"
    model.save(model_path)


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

    model = ModelDefinition()

    dataset = load_training_dataset()

    logging.info("Baseline model")
    train_basic_model(model, dataset, preprocessing_pipeline_baseline)

    logging.info("Outlier removed model")
    train_basic_model_with_outlier_removal(model, dataset, preprocessing_pipeline_no_outliers)
