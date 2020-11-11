import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
    return X_train, X_test, y_train, y_test


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


def full_pipeline(df):
    train_df = (df
                .copy()
                .pipe(transform.log_transform_sale_price)
                .pipe(transform.fill_null_values)
                .pipe(feature_filter.filter_large_house_outliers)
                .pipe(feature_filter.drop_id_column)
                .pipe(transform.fill_null_values)
                .pipe(transform.subclass_numeric_to_category)
                .pipe(transform.month_sold_numeric_categorical)
                .pipe(transform.encode_categoricals_as_ordinal_features)
                .pipe(transform.discrete_categorical_transformation)
                .pipe(transform.add_interaction_variables)
                .pipe(transform.add_total_bathrooms)
                .pipe(transform.add_total_square_foot)
                .pipe(transform.add_total_square_foot_above_ground)
                .pipe(transform.add_total_porch_size)
                .pipe(transform.add_has_masonry_veneer)
                .pipe(transform.add_if_house_bought_before_completed_build)
                .pipe(transform.add_polynomials_for_top_10_correlated))

    y = train_df["SalePrice"]

    categorical_features = train_df.select_dtypes(include=["object"]).columns
    numerical_features = train_df.select_dtypes(exclude=["object"]).columns.drop("SalePrice")

    train_num = train_df[numerical_features].copy()
    train_num = transform.log_transform_skewed_numerical_variables(train_num)

    # One hot encode categorical variables
    train_cat = train_df[categorical_features].copy()
    train_cat = pd.get_dummies(train_cat)

    train_df = pd.concat([train_num, train_cat], axis=1)
    X_train, X_test, y_train, y_test = split_dataset(train_df, y)

    # Need to create copies the split data to avoid SettingWithCopyWarning
    # https://github.com/scikit-learn/scikit-learn/issues/8723#issuecomment-416513938
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Standard scale the features
    # Done after partitioning to avoid fitting scaler to observations in the test set
    # Should the scaler be pickled for deployment use cases then?
    scaler = StandardScaler()
    X_train.loc[:, numerical_features] = scaler.fit_transform(X_train.loc[:, numerical_features].values)
    X_test.loc[:, numerical_features] = scaler.transform(X_test.loc[:, numerical_features].values)

    return X_train, X_test, y_train, y_test
