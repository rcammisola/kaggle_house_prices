import pandas as pd
import numpy as np
from scipy.stats import skew


def log_transform_sale_price(df):
    df.SalePrice = np.log1p(df.SalePrice)
    return df


def fill_null_values(train):
    # Alley : data description says NA means "no alley access"
    train.loc[:, "Alley"] = train.loc[:, "Alley"].fillna("None")
    # BedroomAbvGr : NA most likely means 0
    train.loc[:, "BedroomAbvGr"] = train.loc[:, "BedroomAbvGr"].fillna(0)

    # BsmtQual etc : data description says NA for basement features is "no basement"
    train.loc[:, "BsmtQual"] = train.loc[:, "BsmtQual"].fillna("No")
    train.loc[:, "BsmtCond"] = train.loc[:, "BsmtCond"].fillna("No")
    train.loc[:, "BsmtExposure"] = train.loc[:, "BsmtExposure"].fillna("No")
    train.loc[:, "BsmtFinType1"] = train.loc[:, "BsmtFinType1"].fillna("No")
    train.loc[:, "BsmtFinType2"] = train.loc[:, "BsmtFinType2"].fillna("No")

    train.loc[:, "BsmtFullBath"] = train.loc[:, "BsmtFullBath"].fillna(0)
    train.loc[:, "BsmtHalfBath"] = train.loc[:, "BsmtHalfBath"].fillna(0)
    train.loc[:, "BsmtUnfSF"] = train.loc[:, "BsmtUnfSF"].fillna(0)

    # CentralAir : NA most likely means No
    train.loc[:, "CentralAir"] = train.loc[:, "CentralAir"].fillna("N")

    # Condition : NA most likely means Normal
    train.loc[:, "Condition1"] = train.loc[:, "Condition1"].fillna("Norm")
    train.loc[:, "Condition2"] = train.loc[:, "Condition2"].fillna("Norm")

    # EnclosedPorch : NA most likely means no enclosed porch
    train.loc[:, "EnclosedPorch"] = train.loc[:, "EnclosedPorch"].fillna(0)

    # External stuff : NA most likely means average
    train.loc[:, "ExterCond"] = train.loc[:, "ExterCond"].fillna("TA")
    train.loc[:, "ExterQual"] = train.loc[:, "ExterQual"].fillna("TA")

    # Fence : data description says NA means "no fence"
    train.loc[:, "Fence"] = train.loc[:, "Fence"].fillna("No")

    # FireplaceQu : data description says NA means "no fireplace"
    train.loc[:, "FireplaceQu"] = train.loc[:, "FireplaceQu"].fillna("No")
    train.loc[:, "Fireplaces"] = train.loc[:, "Fireplaces"].fillna(0)

    # Functional : data description says NA means typical
    train.loc[:, "Functional"] = train.loc[:, "Functional"].fillna("Typ")

    # GarageType etc : data description says NA for garage features is "no garage"
    train.loc[:, "GarageType"] = train.loc[:, "GarageType"].fillna("No")
    train.loc[:, "GarageFinish"] = train.loc[:, "GarageFinish"].fillna("No")
    train.loc[:, "GarageQual"] = train.loc[:, "GarageQual"].fillna("No")
    train.loc[:, "GarageCond"] = train.loc[:, "GarageCond"].fillna("No")

    train.loc[train["GarageYrBlt"].isna(), "GarageYrBlt"] = train.loc[train["GarageYrBlt"].isna(), "YearBuilt"]

    train.loc[:, "GarageArea"] = train.loc[:, "GarageArea"].fillna(0)
    train.loc[:, "GarageCars"] = train.loc[:, "GarageCars"].fillna(0)

    # HalfBath : NA most likely means no half baths above grade
    train.loc[:, "HalfBath"] = train.loc[:, "HalfBath"].fillna(0)

    # HeatingQC : NA most likely means typical
    train.loc[:, "HeatingQC"] = train.loc[:, "HeatingQC"].fillna("TA")

    # KitchenAbvGr : NA most likely means 0
    train.loc[:, "KitchenAbvGr"] = train.loc[:, "KitchenAbvGr"].fillna(0)

    # KitchenQual : NA most likely means typical
    train.loc[:, "KitchenQual"] = train.loc[:, "KitchenQual"].fillna("TA")

    # LotFrontage : NA most likely means no lot frontage
    train.loc[:, "LotFrontage"] = train.loc[:, "LotFrontage"].fillna(0)

    # LotShape : NA most likely means regular
    train.loc[:, "LotShape"] = train.loc[:, "LotShape"].fillna("Reg")

    # MasVnrType : NA most likely means no veneer
    train.loc[:, "MasVnrType"] = train.loc[:, "MasVnrType"].fillna("None")
    train.loc[:, "MasVnrArea"] = train.loc[:, "MasVnrArea"].fillna(0)

    # MiscFeature : data description says NA means "no misc feature"
    train.loc[:, "MiscFeature"] = train.loc[:, "MiscFeature"].fillna("No")
    train.loc[:, "MiscVal"] = train.loc[:, "MiscVal"].fillna(0)

    # OpenPorchSF : NA most likely means no open porch
    train.loc[:, "OpenPorchSF"] = train.loc[:, "OpenPorchSF"].fillna(0)

    # PavedDrive : NA most likely means not paved
    train.loc[:, "PavedDrive"] = train.loc[:, "PavedDrive"].fillna("N")

    # PoolQC : data description says NA means "no pool"
    train.loc[:, "PoolQC"] = train.loc[:, "PoolQC"].fillna("No")
    train.loc[:, "PoolArea"] = train.loc[:, "PoolArea"].fillna(0)

    # SaleCondition : NA most likely means normal sale
    train.loc[:, "SaleCondition"] = train.loc[:, "SaleCondition"].fillna("Normal")

    # ScreenPorch : NA most likely means no screen porch
    train.loc[:, "ScreenPorch"] = train.loc[:, "ScreenPorch"].fillna(0)

    # TotRmsAbvGrd : NA most likely means 0
    train.loc[:, "TotRmsAbvGrd"] = train.loc[:, "TotRmsAbvGrd"].fillna(0)

    # Utilities : NA most likely means all public utilities
    train.loc[:, "Utilities"] = train.loc[:, "Utilities"].fillna("AllPub")

    # WoodDeckSF : NA most likely means no wood deck
    train.loc[:, "WoodDeckSF"] = train.loc[:, "WoodDeckSF"].fillna(0)

    return train


def subclass_numeric_to_category(df):
    return df.replace({
        "MSSubClass": {
            20: "SC20",
            30: "SC30",
            40: "SC40",
            45: "SC45",
            50: "SC50",
            60: "SC60",
            70: "SC70",
            75: "SC75",
            80: "SC80",
            85: "SC85",
            90: "SC90",
            120: "SC120",
            150: "SC150",
            160: "SC160",
            180: "SC180",
            190: "SC190"
        }
    })


def month_sold_numeric_categorical(df):
    return df.replace({
        "MoSold": {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec"
        }
    })


def encode_categoricals_as_ordinal_features(df):
    return df.replace({
        "Alley": {"Grvl": 1, "Pave": 2},
        "BsmtCond": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "BsmtExposure": {"No": 0, "Mn": 1, "Av": 2, "Gd": 3},
        "BsmtFinType1": {"No": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4,
                         "ALQ": 5, "GLQ": 6},
        "BsmtFinType2": {"No": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4,
                         "ALQ": 5, "GLQ": 6},
        "BsmtQual": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "ExterCond": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "ExterQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "FireplaceQu": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "Functional": {"Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5,
                       "Min2": 6, "Min1": 7, "Typ": 8},
        "GarageCond": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "GarageQual": {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "HeatingQC": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "KitchenQual": {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        "LandSlope": {"Sev": 1, "Mod": 2, "Gtl": 3},
        "LotShape": {"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4},
        "PavedDrive": {"N": 0, "P": 1, "Y": 2},
        "PoolQC": {"No": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
        "Street": {"Grvl": 1, "Pave": 2},
        "Utilities": {"ELO": 1, "NoSeWa": 2, "NoSewr": 3, "AllPub": 4}
    })


def discrete_categorical_transformation(df):
    # Simplifications of existing features
    df["SimplOverallQual"] = df.OverallQual.replace({
        1: 1, 2: 1, 3: 1,
        4: 2, 5: 2, 6: 2,
        7: 3, 8: 3, 9: 3, 10: 3,
    })

    df["SimplOverallCond"] = df.OverallCond.replace({
        # Bad
        1: 1, 2: 1, 3: 1,
        # Average
        4: 2, 5: 2, 6: 2,
        # Good
        7: 3, 8: 3, 9: 3, 10: 3,
    })

    df["SimplPoolQC"] = df.PoolQC.replace({
        1: 1, 2: 1,
        3: 2, 4: 2,
    })

    df["SimplGarageCond"] = df.GarageCond.replace({
        1: 1, 2: 1, 3: 1,
        4: 2, 5: 2,
    })

    df["SimplGarageQual"] = df.GarageQual.replace({1: 1,  # bad
                                                   2: 1, 3: 1,  # average
                                                   4: 2, 5: 2  # good
                                                   })
    df["SimplFireplaceQu"] = df.FireplaceQu.replace({1: 1,  # bad
                                                     2: 1, 3: 1,  # average
                                                     4: 2, 5: 2  # good
                                                     })
    df["SimplFireplaceQu"] = df.FireplaceQu.replace({1: 1,  # bad
                                                     2: 1, 3: 1,  # average
                                                     4: 2, 5: 2  # good
                                                     })
    df["SimplFunctional"] = df.Functional.replace({1: 1, 2: 1,  # bad
                                                   3: 2, 4: 2,  # major
                                                   5: 3, 6: 3, 7: 3,  # minor
                                                   8: 4  # typical
                                                   })
    df["SimplKitchenQual"] = df.KitchenQual.replace({1: 1,  # bad
                                                     2: 1, 3: 1,  # average
                                                     4: 2, 5: 2  # good
                                                     })
    df["SimplHeatingQC"] = df.HeatingQC.replace({1: 1,  # bad
                                                 2: 1, 3: 1,  # average
                                                 4: 2, 5: 2  # good
                                                 })
    df["SimplBsmtFinType1"] = df.BsmtFinType1.replace({1: 1,  # unfinished
                                                       2: 1, 3: 1,  # rec room
                                                       4: 2, 5: 2, 6: 2  # living quarters
                                                       })
    df["SimplBsmtFinType2"] = df.BsmtFinType2.replace({1: 1,  # unfinished
                                                       2: 1, 3: 1,  # rec room
                                                       4: 2, 5: 2, 6: 2  # living quarters
                                                       })
    df["SimplBsmtCond"] = df.BsmtCond.replace({1: 1,  # bad
                                               2: 1, 3: 1,  # average
                                               4: 2, 5: 2  # good
                                               })
    df["SimplBsmtQual"] = df.BsmtQual.replace({1: 1,  # bad
                                               2: 1, 3: 1,  # average
                                               4: 2, 5: 2  # good
                                               })
    df["SimplExterCond"] = df.ExterCond.replace({1: 1,  # bad
                                                 2: 1, 3: 1,  # average
                                                 4: 2, 5: 2  # good
                                                 })
    df["SimplExterQual"] = df.ExterQual.replace({1: 1,  # bad
                                                 2: 1, 3: 1,  # average
                                                 4: 2, 5: 2  # good
                                                 })

    return df


def add_interaction_variables(df):
    # Combinations of existing features
    # Overall quality of the house
    df["OverallGrade"] = df["OverallQual"] * df["OverallCond"]
    # Overall quality of the garage
    df["GarageGrade"] = df["GarageQual"] * df["GarageCond"]
    # Overall quality of the exterior
    df["ExterGrade"] = df["ExterQual"] * df["ExterCond"]
    # Overall kitchen score
    df["KitchenScore"] = df["KitchenAbvGr"] * df["KitchenQual"]
    # Overall fireplace score
    df["FireplaceScore"] = df["Fireplaces"] * df["FireplaceQu"]
    # Overall garage score
    df["GarageScore"] = df["GarageArea"] * df["GarageQual"]
    # Overall pool score
    df["PoolScore"] = df["PoolArea"] * df["PoolQC"]
    # Simplified overall quality of the house
    df["SimplOverallGrade"] = df["SimplOverallQual"] * df["SimplOverallCond"]
    # Simplified overall quality of the exterior
    df["SimplExterGrade"] = df["SimplExterQual"] * df["SimplExterCond"]
    # Simplified overall pool score
    df["SimplPoolScore"] = df["PoolArea"] * df["SimplPoolQC"]
    # Simplified overall garage score
    df["SimplGarageScore"] = df["GarageArea"] * df["SimplGarageQual"]
    # Simplified overall fireplace score
    df["SimplFireplaceScore"] = df["Fireplaces"] * df["SimplFireplaceQu"]
    # Simplified overall kitchen score
    df["SimplKitchenScore"] = df["KitchenAbvGr"] * df["SimplKitchenQual"]

    return df


def add_total_bathrooms(df):
    df["TotalBath"] = df["BsmtFullBath"] + \
                      (0.5 * df["BsmtHalfBath"]) + \
                      df["FullBath"] + \
                      (0.5 * df["HalfBath"])
    return df


def add_total_square_foot(df):
    df["AllSF"] = df["GrLivArea"] + df["TotalBsmtSF"]
    return df


def add_total_square_foot_above_ground(df):
    df["AllFlrsSF"] = df["1stFlrSF"] + df["2ndFlrSF"]
    return df


def add_total_porch_size(df):
    df["AllPorchSF"] = df["OpenPorchSF"] + df["EnclosedPorch"] + \
                       df["3SsnPorch"] + df["ScreenPorch"]
    return df


def add_has_masonry_veneer(df):
    df["HasMasVnr"] = df.MasVnrType.replace({"BrkCmn": 1, "BrkFace": 1, "CBlock": 1,
                                             "Stone": 1, "None": 0})
    return df


def add_if_house_bought_before_completed_build(df):
    df["BoughtOffPlan"] = df.SaleCondition.replace({"Abnorml": 0, "Alloca": 0, "AdjLand": 0,
                                                    "Family": 0, "Normal": 0, "Partial": 1})
    return df


def add_polynomials_for_top_10_correlated(df):
    feature_correlations = df.corr()
    top_10_price_correlated_features = (feature_correlations
                                        .sort_values(by="SalePrice", ascending=False)
                                        .SalePrice
                                        .head(11))

    for col in list(top_10_price_correlated_features.index):
        if col == "SalePrice":
            continue

        df[f"{col}-s2"] = df[col] ** 2
        df[f"{col}-s3"] = df[col] ** 3
        df[f"{col}-sq"] = np.sqrt(df[col])

    return df


def log_transform_skewed_numerical_variables(train_num):
    """
    Log transform skewed numerical features to lessen impact of outliers

    Inspired by Alexandru Papiu's script:
    https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models
    As a general rule of thumb, a skewness with an absolute value > 0.5 is considered at least moderately skewed
    """

    skewness = train_num.apply(lambda x: skew(x))
    skewness = skewness[abs(skewness) > 0.5]

    skewed_features = skewness.index
    train_num[skewed_features] = np.log1p(train_num[skewed_features])
    return train_num


def one_hot_encode_categoricals(train_cat):
    return pd.get_dummies(train_cat)
