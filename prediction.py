import time
from contextlib import contextmanager

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from data_management import load_prediction_data, clean_data, show_new_parameters


BASELINE_TIMES = {
    "LOAD DATA": 2.06,
    "CLEAN DATA": 0.08,
    "CHECK NEW PARAMETERS": 0.01,
    "PREPARE FEATURES": 0.08,
    "SPLIT TRAIN / TEST": 0.01,
    "BUILD X / Y": 0.58,
    "TRAIN MODEL": 8.31,
    "CHARTS": 0.12,
}


def timing_marker(elapsed, baseline):
    if not baseline:
        return ""
    ratio = elapsed / baseline
    if 0.8 <= ratio <= 1.2:
        return "🟨"
    if ratio < 0.8:
        return "🟩" * min(8, 1 + int((1 - ratio) / 0.25))
    return "🟥" * min(8, 1 + int((ratio - 1) / 0.25))


@contextmanager
def timed_step(name):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        key = name.split(" [")[0]
        marker = timing_marker(elapsed, BASELINE_TIMES.get(key))
        print(f"⏱  {name:<40} {elapsed:7.2f} s  {marker}")


######################## VARIABLES ########################

INTERPOLATION_NAN_LIMIT = 0 # 0=no interpolation, ... 2 = interpolates NaN series up to gap of 2

HORIZON = 2  # 2 x 10 min = 20 min
FEATURE_STATIONS = ("AIG", "ORO")
STATION_PREFIXES = ("", *(f"{station.lower()}_" for station in FEATURE_STATIONS))
WIND_COLS = ["wind_dir_deg", "wind_speed_kmh", "wind_gust_max_kmh"]
COLS = WIND_COLS + [
    f"{station.lower()}_{col}"
    for station in FEATURE_STATIONS
    for col in WIND_COLS
]
_DIRECTION_COLS = {
    "wind_dir_deg": "dir",
    **{
        f"{station.lower()}_wind_dir_deg": f"{station.lower()}_dir"
        for station in FEATURE_STATIONS
    },
}


TARGETS = ["wind_speed_kmh"]
LAGS = [1, 2, 3, 4, 5, 6, 9, 12, 36, 48]   # 10 min ... 24 h

# Window lengths are sample counts at the 10-minute data cadence. A 3-sample
# window spans 20 minutes (first to last observation), 7 spans 1 hour, and 19
# spans 3 hours.
ROLLING_WINDOWS = {"20m": 3, "1h": 7, "3h": 19}
DIRECTION_CHANGE_LAGS = {"10m": 1, "30m": 3, "1h": 6}
TREND_PERIODS = ("1-0", "2-0", "3-0")
ROLLING_COLS = [
    f"{prefix}{col}"
    for prefix in STATION_PREFIXES
    for col in ("wind_speed_kmh", "wind_gust_max_kmh")
]
TREND_COLS = [
    f"{prefix}{col}"
    for prefix in STATION_PREFIXES
    for col in (
        "wind_speed_kmh",
        "wind_gust_max_kmh",
        "dir_sin",
        "dir_cos",
        "wind_u_kmh",
        "wind_v_kmh",
    )
]


# XGBoost model configuration.
MODEL_PARAMETERS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.1,
    "objective": "reg:absoluteerror",
    "tree_method": "hist",
    "device": "cuda",
    "eval_metric": "rmse",
    "random_state": 42,
}
##########################################################


def prepare(df, last_added_parameter):
    df = df[COLS].loc[last_added_parameter:].copy()   # start where all features exist

    # Encode every station's compass direction without a discontinuity at north.
    for direction_col, feature_prefix in _DIRECTION_COLS.items():
        rad = np.deg2rad(df[direction_col])
        df[f"{feature_prefix}_sin"] = np.sin(rad)
        df[f"{feature_prefix}_cos"] = np.cos(rad)
    df = df.drop(columns=list(_DIRECTION_COLS))

    # Wind vectors combine speed and direction into physically meaningful
    # components (positive values point toward east/north respectively).
    for prefix in STATION_PREFIXES:
        rad = np.arctan2(df[f"{prefix}dir_sin"], df[f"{prefix}dir_cos"])
        speed = df[f"{prefix}wind_speed_kmh"]
        df[f"{prefix}wind_u_kmh"] = -speed * np.sin(rad)
        df[f"{prefix}wind_v_kmh"] = -speed * np.cos(rad)

    # Relative observations help the model distinguish shared regional flow
    # from local differences at BOU.
    for station in FEATURE_STATIONS:
        prefix = f"{station.lower()}_"
        df[f"{prefix}speed_delta_from_bou"] = (
            df[f"{prefix}wind_speed_kmh"] - df["wind_speed_kmh"]
        )
        df[f"{prefix}gust_delta_from_bou"] = (
            df[f"{prefix}wind_gust_max_kmh"] - df["wind_gust_max_kmh"]
        )
        for component in ("wind_u_kmh", "wind_v_kmh"):
            df[f"{prefix}{component}_delta_from_bou"] = (
                df[f"{prefix}{component}"] - df[component]
            )

    df = df.asfreq("10min")
    
    if INTERPOLATION_NAN_LIMIT > 0:
        df = df.interpolate(limit=INTERPOLATION_NAN_LIMIT) 
    return df


def split(df, test_from="2023-01-01"):
    return df[df.index < test_from], df[df.index >= test_from]


def make_xy(df):
    X = df.copy()

    # Trailing summaries capture the local level, variability, and rate of
    # change. These windows include the current observation but never look
    # forward in time.
    for col in ROLLING_COLS:
        series = df[col]
        for period, window in ROLLING_WINDOWS.items():
            rolling = series.rolling(window=window, min_periods=window)
            X[f"{col}_roll_mean_{period}"] = rolling.mean()
            X[f"{col}_roll_std_{period}"] = rolling.std(ddof=0)
            elapsed_hours = (window - 1) / 6
            X[f"{col}_slope_per_hour_{period}"] = (
                series - series.shift(window - 1)
            ) / elapsed_hours

    # Circular direction deltas avoid the discontinuity between 359 and 0
    # degrees. The sine/cosine pair describes the turn from the older
    # observation to the current one.
    for prefix in STATION_PREFIXES:
        for period, lag in DIRECTION_CHANGE_LAGS.items():
            old_sin = df[f"{prefix}dir_sin"].shift(lag)
            old_cos = df[f"{prefix}dir_cos"].shift(lag)
            X[f"{prefix}dir_change_sin_{period}"] = (
                df[f"{prefix}dir_sin"] * old_cos
                - df[f"{prefix}dir_cos"] * old_sin
            )
            X[f"{prefix}dir_change_cos_{period}"] = (
                df[f"{prefix}dir_sin"] * old_sin
                + df[f"{prefix}dir_cos"] * old_cos
            )

    # Cyclic calendar features let the model learn recurring daily and annual
    # patterns without artificial jumps at midnight or New Year's Day.
    hour = df.index.hour + df.index.minute / 60 + df.index.second / 3600
    hour_angle = 2 * np.pi * hour / 24
    X["hour_sin"] = np.sin(hour_angle)
    X["hour_cos"] = np.cos(hour_angle)

    fraction_of_day = hour / 24
    days_in_year = np.where(df.index.is_leap_year, 366, 365)
    year_position = df.index.dayofyear - 1 + fraction_of_day
    year_angle = 2 * np.pi * year_position / days_in_year
    X["dayofyear_sin"] = np.sin(year_angle)
    X["dayofyear_cos"] = np.cos(year_angle)

    for lag in LAGS:
        X = X.join(df.shift(lag).add_suffix(f"_lag{lag}"))
    for tr in TREND_PERIODS:
        a, b = map(int, tr.split("-"))   # a = older point, b = newer point (0 = now)
        X = X.join((df[TREND_COLS].shift(b) - df[TREND_COLS].shift(a)).add_suffix(f"_trend{tr}"))
    y = df[TARGETS].shift(-HORIZON).add_prefix("y_")
    data = X.join(y).dropna()

    return data[X.columns], data[y.columns]


def charts(
    X_test,
    Y_test,
    target,
    model,
    dontshowlist,
    Y_train=None,
    evals_result=None,
    pred_train=None,
    pred_test=None,
):
    col = f"y_{target}"
    true = Y_test[col]
    pers = X_test[target]
    pred = pd.Series(
        model.predict(X_test) + pers.to_numpy(),
        index=Y_test.index,
    )
    residual = pred - true
    abs_err = residual.abs()
    mae_p, mae_m = mean_absolute_error(true, pers), mean_absolute_error(true, pred)

    # Existing charts: model vs persistence MAE.
    if "1" not in dontshowlist:
        plt.figure(figsize=(5, 4))
        plt.bar(["persistence", "model"], [mae_p, mae_m])
        plt.title("MAE (km/h)")

    # Existing charts: predicted vs actual.
    if "2" not in dontshowlist:
        plt.figure(figsize=(6, 6))
        plt.scatter(true, pred, s=2, alpha=0.2)
        lim = [true.min(), true.max()]
        plt.plot(lim, lim, "r")
        plt.xlabel("actual")
        plt.ylabel("predicted")
        plt.title(f"{target}: predicted vs actual (+{HORIZON * 10} min)")

    # Existing charts: time series, first day of test.
    if "3" not in dontshowlist:
        n = 144
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

    # Existing charts: residual histogram.
    if "4" not in dontshowlist:
        plt.figure(figsize=(6, 4))
        plt.hist(residual, bins=80)
        plt.title("Residuals (pred - actual)")

    # Existing charts: MAE by hour and month.
    if "5" not in dontshowlist:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        abs_err.groupby(abs_err.index.hour).mean().plot(ax=ax[0], title="MAE by hour")
        abs_err.groupby(abs_err.index.month).mean().plot(ax=ax[1], title="MAE by month")
        fig.tight_layout()

    # Existing charts: MAE by current speed.
    if "6" not in dontshowlist:
        bins = pd.cut(pers, [0, 5, 10, 15, 20, 30, 50, 200])
        abs_err.groupby(bins, observed=True).mean().plot.bar(
            figsize=(8, 4), title="MAE by current speed"
        )

    # Existing charts: permutation importance.
    if "7" not in dontshowlist:
        idx = np.random.RandomState(0).choice(len(X_test), min(5000, len(X_test)), replace=False)
        pi = permutation_importance(
            model,
            X_test.iloc[idx],
            (true - pers).iloc[idx],
            n_repeats=5,
            scoring="neg_mean_absolute_error",
            random_state=0,
        )
        pd.Series(pi.importances_mean, index=X_test.columns).nlargest(20)[::-1].plot.barh(
            figsize=(8, 7), title="Permutation importance (top 20)"
        )


    if Y_train is None or evals_result is None:
        return

    y_train = Y_train[col].to_numpy()
    y_test = true.to_numpy()
    pred_train = np.asarray(pred_train)
    pred_test = np.asarray(pred_test if pred_test is not None else pred)
    train_rmse = np.asarray(evals_result["validation_0"]["rmse"])
    test_rmse = np.asarray(evals_result["validation_1"]["rmse"])
    epochs = len(train_rmse)
    train_r2_history = 1 - train_rmse**2 / np.var(y_train)
    test_r2_history = 1 - test_rmse**2 / np.var(y_test)

        # Appended charts: training curves, residual distribution, and feature importance.
    if "8" not in dontshowlist:
        fig, ax = plt.subplots(2, 2, figsize=(18, 15))
        ax[0, 0].plot(range(epochs), train_rmse, label="Train Loss", color="blue")
        ax[0, 0].plot(range(epochs), test_rmse, label="Test Loss", color="red", linestyle="--")
        ax[0, 0].set_title("Évolution du Loss (Erreur RMSE)")
        ax[0, 0].legend()
        ax[0, 0].grid(alpha=0.3)

        ax[0, 1].plot(range(epochs), train_r2_history, label="Train R²", color="green")
        ax[0, 1].plot(range(epochs), test_r2_history, label="Test R²", color="darkorange", linestyle="--")
        r2_min = min(float(np.min(train_r2_history)), float(np.min(test_r2_history)))
        r2_max = max(float(np.max(train_r2_history)), float(np.max(test_r2_history)))
        r2_center = (r2_min + r2_max) / 2
        r2_half_range = max((r2_max - r2_min) * 0.55, 0.001)
        ax[0, 1].set_ylim(r2_center - r2_half_range, r2_center + r2_half_range)
        ax[0, 1].set_title("Focus sur la convergence (Score R²)")
        ax[0, 1].set_xlabel("Nombre d'arbres")
        ax[0, 1].set_ylabel("R² (Zoomé)")
        ax[0, 1].legend()
        ax[0, 1].grid(alpha=0.3, which="both")

        residuals = y_test - pred_test
        ax[1, 0].hist(residuals, bins=100, density=True, color="purple", alpha=0.65)
        kde_sample = residuals
        if len(kde_sample) > 5000:
            kde_sample = np.random.RandomState(42).choice(kde_sample, 5000, replace=False)
        if len(kde_sample) > 1 and np.std(kde_sample) > 0:
            kde = gaussian_kde(kde_sample)
            x_grid = np.linspace(np.min(residuals), np.max(residuals), 500)
            ax[1, 0].plot(x_grid, kde(x_grid), color="black", linewidth=1.5)
        ax[1, 0].axvline(x=0, color="black", linestyle="--")
        ax[1, 0].set_title("Distribution des erreurs (Résidus)")

        importance = model.get_booster().get_score(importance_type="weight")
        feature_names = model.get_booster().feature_names or []
        importances = [importance.get(name, 0) for name in feature_names]
        colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, max(1, len(feature_names))))
        ax[1, 1].barh(feature_names, importances, color=colors)
        ax[1, 1].set_title("Importance des variables")

        fig.tight_layout()

    if "9" not in dontshowlist:

        # Appended chart: zoomed actual-versus-predicted test series.
        n = min(300, len(y_test))
        plt.figure(figsize=(18, 6))
        plt.plot(y_test[:n], label="Réel", color="black", alpha=0.7)
        plt.plot(pred_test[:n], label="Prédit", color="orange", linestyle="--")
        plt.title("Zoom 50h : Réel vs Prédit")
        plt.legend()

    plt.show()


def main():
    with timed_step("LOAD DATA"):
        df = load_prediction_data(feature_stations=FEATURE_STATIONS)

    with timed_step("CLEAN DATA"):
        df = clean_data(df, unit="kmh")

    with timed_step("CHECK NEW PARAMETERS"):
        last_added_parameter = show_new_parameters(df[COLS])

    with timed_step("PREPARE FEATURES"):
        df = prepare(df, last_added_parameter)

    with timed_step("SPLIT TRAIN / TEST"):
        df_train, df_test = split(df)

    with timed_step("BUILD X / Y"):
        X_train, Y_train = make_xy(df_train)
        X_test, Y_test = make_xy(df_test)


    for target in TARGETS:
        col = f"y_{target}"
        # Learn the change from persistence; add the current value back at inference.
        residual_train = Y_train[col] - X_train[target]
        residual_test = Y_test[col] - X_test[target]

        with timed_step(f"TRAIN MODEL [{target}]"):
            print(f"🚀 Lancement de l'entraînement GPU sur {len(X_train)} lignes...")
            start_time = time.perf_counter()
            model = XGBRegressor(**MODEL_PARAMETERS)
            model.fit(
                X_train,
                residual_train,
                eval_set=[(X_train, residual_train), (X_test, residual_test)],
                verbose=False,
            )

            # Previous HistGradientBoostingRegressor implementation:
            # from sklearn.ensemble import HistGradientBoostingRegressor
            # model = HistGradientBoostingRegressor(
            #     loss="absolute_error", max_iter=100, random_state=42, learning_rate=0.1
            # )
            # model.fit(X_train, residual_train)

            duration = time.perf_counter() - start_time
            print(f"✅ Terminé en {duration:.2f} secondes (soit {duration / 60:.2f} minutes)")
            # Predictions and diagnostics use CPU-backed pandas data.
            model.get_booster().set_param({"device": "cpu"})
            pred_train = pd.Series(
                model.predict(X_train) + X_train[target].to_numpy(),
                index=Y_train.index,
            )
            pred = pd.Series(
                model.predict(X_test) + X_test[target].to_numpy(),
                index=Y_test.index,
            )
            mae = mean_absolute_error(Y_test[col], pred)
            rmse = np.sqrt(mean_squared_error(Y_test[col], pred))
            r2 = r2_score(Y_test[col], pred)
            r2_train = r2_score(Y_train[col], pred_train)
            evals_result = model.evals_result()
            abs_error = np.abs(Y_test[col].to_numpy() - pred.to_numpy())

            print('')
            print("--- RÉSULTATS ---")
            print(f"Score R² (TEST) : {r2:.4f}")
            print(f"Score R² (TRAIN) : {r2_train:.4f} (Pour vérifier l'overfitting)")
            print(f"Erreur moyenne (MAE) : {mae:.2f} km/h")
            print(f"RMSE : {rmse:.2f} km/h")
            print("Pourcentage du temps où l'erreur est sous le seuil :")
            for threshold in (2, 5, 10, 15, 20):
                pct_within_threshold = np.mean(abs_error < threshold) * 100
                print(f"  < {threshold:2d} km/h : {pct_within_threshold:.1f}%")

        with open('last_mae.txt', 'r') as f:
            last_mae = float(f.readlines()[0])
        with open('last_mae.txt', 'w') as f:
            f.write(str(mae))

        delta = last_mae - mae                       # > 0 = improvement
        pct = abs(delta) / last_mae * 100
        k = min(20, 1 + int(pct / 0.5))              # 1 emoji per 0.5% change

        print(f"{target}: {last_mae:.3f} -> model={mae:.3f} ({-pct if delta < 0 else pct:+.1f}%)")
        if round(last_mae, 3) == round(mae, 3):
            print('🟨' * 3)
        elif delta > 0:
            print('🟩' * k)
        else:
            print('🟥' * k)

        with timed_step(f"CHARTS [{target}]"):
            charts(
                X_test,
                Y_test,
                target,
                model,
                "1 2 3 4 5 6 7 8 9",
                Y_train=Y_train,
                evals_result=evals_result,
                pred_train=pred_train,
                pred_test=pred,
            )

if __name__ == "__main__":
    main()
