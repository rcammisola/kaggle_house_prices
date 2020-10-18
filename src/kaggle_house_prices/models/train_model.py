"""
Train Baseline Linear Regression model using just GrLivArea
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split

from kaggle_house_prices.data.make_dataset import load_training_dataset
from kaggle_house_prices.logs import configure_logging


def log_transform_sale_price(df):
    df.SalePrice = np.log1p(df.SalePrice)
    return df


def rmse_cv_train(model, X_train, y_train, scorer):
    return np.sqrt(-cross_val_score(model,
                                    X_train,
                                    y_train,
                                    scoring=scorer, cv=10))


if __name__ == "__main__":
    configure_logging()

    # Load data
    train_df = load_training_dataset()

    # log transform sale price (target)
    train_df = log_transform_sale_price(train_df)

    y = train_df["SalePrice"]
    train_df = train_df[["GrLivArea"]]

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        train_df,
        y,
        test_size=0.3,
        random_state=0
    )

    # training fit
    model = LinearRegression()
    model.fit(X_train, y_train)

    # assess vs training and test set
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    logging.info(f"RMSE on Training set: {rmse_cv_train(model, X_train, y_train, scorer).mean()}")
    logging.info(f"RMSE on Test set: {rmse_cv_train(model, X_test, y_test, scorer).mean()}")

    # diagnostic plots
    y_training_predictions = model.predict(X_train)
    y_test_predictions = model.predict(X_test)

    # Plot residuals
    plt.scatter(y_training_predictions,
                y_training_predictions - y_train,
                c="blue",
                marker="s",
                label="Training data")
    plt.scatter(y_test_predictions,
                y_test_predictions - y_test,
                c="lightgreen",
                marker="s",
                label="Validation data")
    plt.title("Linear Regression")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.legend(loc="upper left")
    plt.hlines(y=0, xmin=10.5, xmax=13.5, color="red")
    plt.show()

    # Plot predictions
    plt.scatter(y_training_predictions,
                y_train,
                c="blue",
                marker="s",
                label="Training data")
    plt.scatter(y_test_predictions,
                y_test,
                c="lightgreen",
                marker="s",
                label="Validation")
    plt.title("Linear Regression")
    plt.xlabel("Predicted Values")
    plt.ylabel("Real Values")
    plt.legend(loc="upper left")
    plt.plot([10.5, 13.5], [10.5, 13.5], c="red")
    plt.show()
