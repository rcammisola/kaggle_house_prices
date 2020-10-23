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


class ModelDefinition:
    def __init__(self):
        self.estimator = LinearRegression()
        self.model = None
        self.target = "SalePrice"
        self.input_feature_columns = ["GrLivArea"]

    def target_variable(self):
        return self.target

    def input_feature_names(self):
        return self.input_feature_columns

    def fit(self, input_features, y):
        self.model = self.estimator.fit(input_features, y)

    def predict(self, input_features):
        return self.model.predict(input_features)

    def score(self, input_features, target_values):
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        return rmse_cv_score(self.model, input_features, target_values, scorer).mean()

    def save(self, model_filename):
        with open(model_filename, "wb") as model_file_pointer:
            pickle.dump(self.model, model_file_pointer)