import numpy as np


def log_transform_sale_price(df):
    df.SalePrice = np.log1p(df.SalePrice)
    return df
