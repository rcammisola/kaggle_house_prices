import pandas as pd
from sklearn.model_selection import train_test_split

from kaggle_house_prices.features import transform as transform, filter as feature_filter


def preprocessing_pipeline_baseline(df):
    train_df = (df
                .copy()
                .pipe(transform.log_transform_sale_price))

    y = train_df["SalePrice"]
    train_df = train_df[["GrLivArea"]]
    return split_dataset(train_df, y)


def split_dataset(train_df, y):
    X_train, X_test, y_train, y_test = train_test_split(
        train_df,
        y,
        test_size=0.3,
        random_state=0
    )
    return X_test, X_train, y_test, y_train


def preprocessing_pipeline_no_outliers(df):
    train_df = (df
                .copy()
                .pipe(transform.log_transform_sale_price)
                .pipe(feature_filter.filter_large_house_outliers))

    y = train_df["SalePrice"]
    train_df = train_df[["GrLivArea"]]
    return split_dataset(train_df, y)


def preprocessing_pipeline_handle_nulls(df):
    train_df = (df
                .copy()
                .pipe(transform.log_transform_sale_price)
                .pipe(transform.fill_null_values)
                .pipe(feature_filter.filter_large_house_outliers)
                .pipe(feature_filter.drop_id_column))

    y = train_df["SalePrice"]

    categorical_features = train_df.select_dtypes(include=["object"]).columns
    numerical_features = train_df.select_dtypes(exclude=["object"]).columns.drop("SalePrice")

    train_num = train_df[numerical_features]
    train_cat = train_df[categorical_features]

    # One hot encode categorical variables
    train_cat = pd.get_dummies(train_cat)

    train_df = pd.concat([train_num, train_cat], axis=1)
    return split_dataset(train_df, y)
