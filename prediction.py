import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from data_management import load_data, clean_data, show_new_parameters


############ VARIABLES ############


HORIZON = 6  # 6 x 10 min = +1 h
COLS = ["temp_c", "humidity_pct", "pressure_qff_hpa",
        "wind_dir_deg", "wind_speed_kmh", "wind_gust_max_kmh"]
INTERPOLATION_NAN_LIMIT = 0 # 0=no interpolation, ... 2 = interpolates NaN series up to gap of 2

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
    df = df.dropna() if INTERPOLATION_NAN_LIMIT == 0 else df.interpolate(limit=INTERPOLATION_NAN_LIMIT).dropna()
    return df


def make_xy(df, horizon, target="wind_speed_kmh"):
    ws = df["wind_speed_kmh"]
    X = pd.DataFrame(index=df.index)
    for lag in (0, 1, 3, 6, 12):
        X[f"ws_lag{lag}"] = ws.shift(lag)
    X["gust"] = df["wind_gust_max_kmh"]
    X["dir_sin"] = df["dir_sin"]
    X["dir_cos"] = df["dir_cos"]
    X["temp"] = df["temp_c"]
    X["humidity"] = df["humidity_pct"]
    X["pressure"] = df["pressure_qff_hpa"]
    X["pressure_3h"] = df["pressure_qff_hpa"].diff(18)
    X["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    y = df[target].shift(-horizon).rename("target")
    data = X.join(y).dropna()
    return data.drop(columns="target"), data["target"]


def split(X, y, test_from="2023-01-01"):
    return (X[X.index < test_from], X[X.index >= test_from],
            y[y.index < test_from], y[y.index >= test_from])


def fit_predict(X_train, X_test, y_train, y_test):
    model = HistGradientBoostingRegressor(random_state=0).fit(X_train, y_train)
    return model, pd.Series(model.predict(X_test), index=y_test.index)


def mae(a, b):
    return (a - b).abs().mean()


def circ_mae(a_deg, b_deg):
    return ((a_deg - b_deg + 180) % 360 - 180).abs().mean()


# ---------- step 1: horizon sweep ----------
def horizon_sweep(df):
    print("\nHORIZON SWEEP (wind speed)")
    print(f"{'horizon':>8} {'persist':>8} {'model':>8} {'gain':>7}")
    rows = []
    for h in (1, 3, 6, 12, 18, 36):          # 10 min, 30 min, 1 h, 2 h, 3 h, 6 h
        X, y = make_xy(df, h)
        Xtr, Xte, ytr, yte = split(X, y)
        _, pred = fit_predict(Xtr, Xte, ytr, yte)
        p, m = mae(Xte["ws_lag0"], yte), mae(pred, yte)
        rows.append((h * 10 / 60, p, m))
        print(f"{h * 10:>5} min {p:8.2f} {m:8.2f} {(1 - m / p) * 100:6.1f}%")

    r = np.array(rows)
    plt.figure(figsize=(8, 4))
    plt.plot(r[:, 0], r[:, 1], marker="o", label="persistence")
    plt.plot(r[:, 0], r[:, 2], marker="o", label="model")
    plt.xlabel("horizon (h)")
    plt.ylabel("MAE km/h")
    plt.title("Error vs horizon")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()


# ---------- step 2: feature importance ----------
def feature_importance(df, horizon=6):
    X, y = make_xy(df, horizon)
    Xtr, Xte, ytr, yte = split(X, y)
    model, _ = fit_predict(Xtr, Xte, ytr, yte)

    idx = np.random.RandomState(0).choice(len(Xte), min(20000, len(Xte)), replace=False)
    r = permutation_importance(model, Xte.iloc[idx], yte.iloc[idx],
                               scoring="neg_mean_absolute_error",
                               n_repeats=3, random_state=0)
    imp = pd.Series(r.importances_mean, index=Xte.columns).sort_values()
    print("\nFEATURE IMPORTANCE (MAE increase when shuffled)")
    print(imp.sort_values(ascending=False).round(3))

    imp.plot.barh(figsize=(8, 5), title=f"Permutation importance (+{horizon * 10} min)")
    plt.xlabel("MAE increase km/h")
    plt.tight_layout()
    plt.show()


# ---------- step 3: gusts ----------
def gust_model(df, horizon=6):
    X, y = make_xy(df, horizon, target="wind_gust_max_kmh")
    Xtr, Xte, ytr, yte = split(X, y)
    _, pred = fit_predict(Xtr, Xte, ytr, yte)
    print(f"\nGUST +{horizon * 10} min: persistence MAE {mae(Xte['gust'], yte):.2f} | "
          f"model MAE {mae(pred, yte):.2f} km/h")


# ---------- step 4: direction (sin/cos, recombined) ----------
def direction_model(df, horizon=6, min_speed=5):
    Xs, ys = make_xy(df, horizon, target="dir_sin")
    Xc, yc = make_xy(df, horizon, target="dir_cos")
    Xtr, Xte, ys_tr, ys_te = split(Xs, ys)
    _, _, yc_tr, yc_te = split(Xc, yc)

    _, p_sin = fit_predict(Xtr, Xte, ys_tr, ys_te)
    _, p_cos = fit_predict(Xtr, Xte, yc_tr, yc_te)

    pred = np.rad2deg(np.arctan2(p_sin, p_cos)) % 360
    true = np.rad2deg(np.arctan2(ys_te, yc_te)) % 360
    now = np.rad2deg(np.arctan2(Xte["dir_sin"], Xte["dir_cos"])) % 360

    windy = Xte["ws_lag0"] > min_speed       # direction is noise when calm
    print(f"\nDIRECTION +{horizon * 10} min (speed > {min_speed} km/h): "
          f"persistence {circ_mae(now[windy], true[windy]):.1f}° | "
          f"model {circ_mae(pred[windy], true[windy]):.1f}°")


def split(X, y, test_from="2023-01-01"):
    return (X[X.index < test_from], X[X.index >= test_from],
            y[y.index < test_from], y[y.index >= test_from])


def main():
    df = load_data()
    df = clean_data(df, unit="kmh")
    last_added_parameter = show_new_parameters(df[COLS])
    df = prepare(df, last_added_parameter)

    horizon_sweep(df)
    feature_importance(df, horizon=6)
    gust_model(df, horizon=6)
    direction_model(df, horizon=6)
    
if __name__ == "__main__":
    main()