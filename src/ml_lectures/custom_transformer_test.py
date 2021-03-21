from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_openml


class BigBedroomTransformer(TransformerMixin):
    def transform(self, X, y=None, **fit_params):
        X["big_bedrooms"] = X["bedrooms"] * 100
        return X


data = fetch_openml("house_sales", as_frame=True)
print(data.DESCR)

X = data.frame.drop(["price", "date", "zipcode"], axis=1)
y = data.frame.price

print(X.head())

b = BigBedroomTransformer()
new_X = b.transform(X)

print(new_X.head())
