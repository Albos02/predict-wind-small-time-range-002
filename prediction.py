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


TARGETS = ["wind_speed_kmh"]
# TARGETS = ["wind_speed_kmh", "wind_gust_max_kmh", "dir_sin", "dir_cos"]
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


def charts(X_test, Y_test, target, model, dontshowlist):
    col = f"y_{target}"
    true = Y_test[col]
    pers = X_test[target]
    pred = pd.Series(model.predict(X_test), index=Y_test.index)
    residual = pred - true
    abs_err = residual.abs()
    mae_p, mae_m = mean_absolute_error(true, pers), mean_absolute_error(true, pred)
    print(f"{target}: persistence={mae_p:.3f} | model={mae_m:.3f}")

    # 1. MAE: model vs persistence
    if "1" not in dontshowlist:
        plt.figure(figsize=(5, 4))
        plt.bar(["persistence", "model"], [mae_p, mae_m])
        plt.title("MAE (km/h)")

    # 2. predicted vs actual
    if "2" not in dontshowlist:
        plt.figure(figsize=(6, 6))
        plt.scatter(true, pred, s=2, alpha=0.2)
        lim = [true.min(), true.max()]
        plt.plot(lim, lim, "r")
        plt.xlabel("actual"); plt.ylabel("predicted")
        plt.title(f"{target}: predicted vs actual (+{HORIZON * 10} min)")

    # 3. time series, first day of test
    if "3" not in dontshowlist:
        n = 144 * 1
        actual = true.to_numpy()[:n]
        pred_n = pred.to_numpy()[:n]
        pers_n = pers.to_numpy()[:n]
        err_n = np.abs(pred_n - actual)

        fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, height_ratios=[2, 1])
        ax[0].plot(actual, label="actual (t+HORIZON)")
        ax[0].plot(pred_n, label="model.predict(X_test)")
        ax[0].plot(pers_n, "--", color="gray", alpha=0.7, label="persistence (wind now)")
        ax[0].fill_between(range(n), actual, pred_n, color="red", alpha=0.25, label="error")
        ax[0].set_title(f"{target}: MAE on this extract = {err_n.mean():.2f}")
        ax[0].legend()

        ax[1].fill_between(range(n), err_n, color="red", alpha=0.5)
        ax[1].set_title("|pred - actual| (MAE = its mean)")
        fig.tight_layout()

    # 4. residual histogram
    if "4" not in dontshowlist:
        plt.figure(figsize=(6, 4))
        plt.hist(residual, bins=80)
        plt.title("Residuals (pred - actual)")

    # 5. MAE by hour and month
    if "5" not in dontshowlist:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        abs_err.groupby(abs_err.index.hour).mean().plot(ax=ax[0], title="MAE by hour")
        abs_err.groupby(abs_err.index.month).mean().plot(ax=ax[1], title="MAE by month")
        fig.tight_layout()

    # 6. MAE by current speed
    if "6" not in dontshowlist:
        bins = pd.cut(pers, [0, 5, 10, 15, 20, 30, 50, 200])
        abs_err.groupby(bins, observed=True).mean().plot.bar(figsize=(8, 4), title="MAE by current speed")

    # 7. permutation importance
    if "7" not in dontshowlist:
        idx = np.random.RandomState(0).choice(len(X_test), min(5000, len(X_test)), replace=False)
        pi = permutation_importance(model, X_test.iloc[idx], true.iloc[idx], n_repeats=5, scoring="neg_mean_absolute_error", random_state=0)
        pd.Series(pi.importances_mean, index=X_test.columns).nlargest(20)[::-1].plot.barh(figsize=(8, 7), title="Permutation importance (top 20)")

    plt.show()


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
        col = f"y_{target}"
        base = mean_absolute_error(Y_test[col], X_test[target]) # base = score we get by saying : in HORIZON time it will be like rn

        print('TRAINING...')
        model = HistGradientBoostingRegressor(max_iter=300, random_state=42, learning_rate=0.1, max_leaf_nodes=100)
        model.fit(X_train, Y_train[col])
        mae = mean_absolute_error(model.predict(X_test), Y_test[col])
        print(f"{target}: {base:.3f} -> model={mae:.3f}")

        charts(X_test, Y_test, target, model, "124567")

if __name__ == "__main__":
    main()
