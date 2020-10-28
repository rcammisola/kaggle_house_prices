def filter_large_house_outliers(df):
    """
    Drop the houses with more than 4000 sq feet following
    dataset author recommendations.
    """
    return df[df.GrLivArea < 4000]


def drop_id_column(df):
    return df.drop("Id", axis=1)
