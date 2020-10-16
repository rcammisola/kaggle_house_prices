import os

import pandas as pd

from kaggle_house_prices.constants import PROJECT_BASE_PATH


def load_training_dataset():
    return pd.read_csv(os.path.join(PROJECT_BASE_PATH, "data/raw/train.csv"))
