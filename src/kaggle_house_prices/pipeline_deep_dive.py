import os

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from kaggle_house_prices.constants import PROJECT_BASE_PATH

wine_df = pd.read_csv(os.path.join(PROJECT_BASE_PATH, "data/raw/wine_quality/wine_quality.csv"), sep=";")

print(wine_df.head(3))

X = wine_df.drop(["quality"], axis=1)
y = wine_df["quality"]

pipeline_steps = [
    ("scaler", StandardScaler()),
    ("SVM", SVC())
]

pipeline = Pipeline(pipeline_steps)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

# Show why we need to stratify target in train test split
print(wine_df["quality"].value_counts())

# Define parameters for SVM gamma and C
parameters = {
    "SVM__C": [0.001, 0.1, 10, 100, 10e5],
    "SVM__gamma": [0.1, 0.01]
}

# setup grid
grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)
grid.fit(X_train, y_train)
print(f"score = {grid.score(X_test, y_test)}")
print(grid.best_params_)
