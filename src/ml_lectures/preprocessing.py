"""
https://github.com/amueller/COMS4995-s20/blob/master/slides/aml-04-preprocessing/aml-07-preprocessing.ipynb

"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge, RidgeCV, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = fetch_openml("house_sales", as_frame=True)
print(data.DESCR)

X = data.frame.drop(["price", "date", "zipcode"], axis=1)
y = data.frame.price

# Scaling
# Tree based algorithms are insensitive to scaling
# Most common way is with 0 mean, unit variance (StandardScaler)

# * MinMaxScaler - scale between a min and max value of feature
#     useful for features with clear boundaries

# * Robust Scaler - median and inter-quantile ranges to scale
#     useful if you have outliers that affect the standard scaler

# Normalizer - does not work on per column, row based...
#     rows are divided by euclidean length
#     used when you only care about relative frequencies

# Sparse matrix - use MaxAbsScaler to avoid making the data dense

print("====" * 20)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

ridge = Ridge().fit(X_train, y_train)
print("Score without scaling:", ridge.score(X_test, y_test))
print("====" * 20)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

ridge = Ridge().fit(X_train_scaled, y_train)
X_test_scaled = scaler.transform(X_test)
print("Score with scaling:", ridge.score(X_test_scaled, y_test))
print("====" * 20)

print("")
print("Ridge Cross Validation (unscaled)")
scores = cross_val_score(RidgeCV(), X_train, y_train, cv=10)
print(np.mean(scores), np.std(scores))
print("====" * 20)

print("")
print("Ridge Cross Validation (scaled)")
scores = cross_val_score(RidgeCV(), X_train_scaled, y_train, cv=10)
print(np.mean(scores), np.std(scores))
print("====" * 20)

print("")
print("KNN Cross Validation (unscaled)")
scores = cross_val_score(KNeighborsRegressor(), X_train, y_train, cv=10)
print(np.mean(scores), np.std(scores))
print("====" * 20)

print("")
print("KNN Cross Validation (scaled)")
scores = cross_val_score(KNeighborsRegressor(), X_train_scaled, y_train, cv=10)
print(np.mean(scores), np.std(scores))

print("====" * 20)

print("")

print("PIPELINES!")

pipe = make_pipeline(StandardScaler(), Ridge())
pipe.fit(X_train, y_train)
print("Ridge score (pipeline):", pipe.score(X_test, y_test))
print("====" * 20)

print("Ridge Cross Validation (with pipeline)")
scores = cross_val_score(pipe, X_train, y_train, cv=10)
print(np.mean(scores), np.std(scores))
print("====" * 20)

# Can also build a pipeline with named steps
knn_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", KNeighborsRegressor())
])
print("KNeighorsRegressor Cross Validation (with pipeline)")
scores = cross_val_score(knn_pipe, X_train, y_train, cv=10)
print(np.mean(scores), np.std(scores))
print("====" * 20)



print("GridSearch with Pipelines")
print("====" * 20)
knn_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", KNeighborsRegressor())
])

print("KNeighorsRegressor GridSearch with pipeline")

# parameters applied to step name __dunder__ parameter name e.g. regressor__n_neighbors
param_grid = {
    'regressor__n_neighbors': range(1, 10)
}
grid = GridSearchCV(knn_pipe, param_grid, cv=10)

grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.score(X_test, y_test))


print("GridSearch with Diabetes dataset")
print("====" * 20)
diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, random_state=0)

pipe = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(),
    Ridge()
)

param_grid = {
    'polynomialfeatures__degree': [1, 2, 3],
    'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]
}
grid = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-1, return_train_score=True)
print(grid.fit(X_train, y_train))
print(grid.score(X_test, y_test))

# You can use pipelines to dynamically change the steps in the pipeline!
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", Ridge())
])

param_grid = {
    'scaler': [StandardScaler(), MinMaxScaler()],
    'regressor': [Ridge(), Lasso()],
    'regressor__alpha': np.logspace(-3, 3, 7)
}

grid = GridSearchCV(pipe, param_grid)
grid.fit(X_train, y_train)
print(grid.score(X_test, y_test))
