import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error

from data_management import load_data, clean_data, show_new_parameters


############ VARIABLES ############

INTERPOLATION_NAN_LIMIT = 0 # 0=no interpolation, ... 2 = interpolates NaN series up to gap of 2

HORIZON = 2  # 2 x 10 min = 20 min
COLS = ["temp_c", "humidity_pct", "pressure_qff_hpa",
        "wind_dir_deg", "wind_speed_kmh", "wind_gust_max_kmh"]


TARGETS = ["wind_speed_kmh", "wind_gust_max_kmh", "dir_sin", "dir_cos"]
LAGS = [1, 2, 3, 6, 12, 36, 144]   # 10 min ... 24 h

##################################


def prepare(df, last_added_parameter):
    df = df[COLS].loc[last_added_parameter:].copy()   # start where all features exist

    ## WIND DIRECTION ##
    rad = np.deg2rad(df["wind_dir_deg"])
    df["dir_sin"] = np.sin(rad)
    df["dir_cos"] = np.cos(rad)
    df = df.drop(columns="wind_dir_deg")

    # if degrees needed : df["wind_dir_deg"] = np.rad2deg(np.arctan2(df["dir_sin"], df["dir_cos"])) % 360

    df = df.asfreq("10min")
    
    if INTERPOLATION_NAN_LIMIT > 0:
        df = df.interpolate(limit=INTERPOLATION_NAN_LIMIT) 
    return df


def split(df, test_from="2023-01-01"):
    return df[df.index < test_from], df[df.index >= test_from]


def make_xy(df):
    X = df.copy()
    for lag in LAGS:
        X = X.join(df.shift(lag).add_suffix(f"_lag{lag}"))
    y = df[TARGETS].shift(-HORIZON).add_prefix("y_")
    data = X.join(y).dropna()

    return data[X.columns], data[y.columns]



def main():
    df = load_data()
    df = clean_data(df, unit="kmh")
    last_added_parameter = show_new_parameters(df[COLS])
    df = prepare(df, last_added_parameter)

    df_train, df_test = split(df)
    # print(df_train.index.min(), df_train.index.max(), len(df_train))
    # print(df_test.index.min(), df_test.index.max(), len(df_test))

    X_train, Y_train = make_xy(df_train)
    X_test, Y_test = make_xy(df_test)


    for target in TARGETS:
        if "dir" in target:
            continue
        elif "gust" in target:
            continue

        col = f"y_{target}"
        base = mean_absolute_error(Y_test[col], X_test[target]) # base = score we get by saying : in HORIZON time it will be like rn

        print('TRAINING...')
        model = HistGradientBoostingRegressor(max_iter=300, random_state=42, learning_rate=0.0001)#, max_leaf_nodes=100)
        model.fit(X_train, Y_train[col])
        mae = mean_absolute_error(model.predict(X_test), Y_test[col])
        print(f"{target}: {base:.3f} -> model={mae:.3f}")

    
if __name__ == "__main__":
    main()