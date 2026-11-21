import time

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_management import load_prediction_data, clean_data, show_new_parameters
from prediction_base import (
    add_advection_features,
    make_xy,
    prepare,
)
from prediction_variables import (
    CHARTS_DIR,
    COLS,
    FEATURE_STATIONS,
    HORIZON,
    INTERPOLATION_NAN_LIMIT,
    LAGS,
    STATION_COORDINATES_PATH,
    STATION_PREFIXES,
    TARGETS,
    TARGET_STATION,
    TREND_COLS,
    TREND_PERIODS,
    WIND_COLS,
)
from prediction_charts import charts
from time_management import BASELINE_TIMES, timed_step, timing_marker


def main():
    from lightgbm import LGBMRegressor

    with timed_step("LOAD DATA"):
        df = load_prediction_data(target_station=TARGET_STATION, feature_stations=FEATURE_STATIONS)

    with timed_step("CLEAN DATA"):
        df = clean_data(df, unit="kmh")

    with timed_step("CHECK NEW PARAMETERS"):
        last_added_parameter = show_new_parameters(df[COLS])

    with timed_step("PREPARE FEATURES"):
        df = prepare(df, last_added_parameter)

    with timed_step("BUILD X / Y"):
        # Keep earlier observations available as lag features, while holding
        # out 2025 and purging training labels that cross into the test period.
        X, Y = make_xy(df)
        test_start = pd.Timestamp("2025-01-01")
        train_mask = X.index < test_start - pd.Timedelta(minutes=HORIZON * 10)
        test_mask = X.index >= test_start
        X_train, Y_train = X.loc[train_mask], Y.loc[train_mask]
        X_test, Y_test = X.loc[test_mask], Y.loc[test_mask]


    for target in TARGETS:
        col = f"y_{target}"
        # Learn the change from persistence; add the current value back at inference.
        residual_train = Y_train[col] - X_train[target]
        residual_test = Y_test[col] - X_test[target]

        with timed_step(f"TRAIN MODEL [{target}]"):
            print(f"🚀 Lancement de l'entraînement LightGBM sur {len(X_train)} lignes...")
            start_time = time.perf_counter()
            model = LGBMRegressor(
                objective="mae",
                n_estimators=400,
                learning_rate=0.04,
                num_leaves=31,
                max_depth=-1,
                min_child_samples=100,
                reg_lambda=2.0,
                colsample_bytree=0.85,
                device_type="gpu",
                gpu_device_id=0,
                max_bin=63,
                n_jobs=-1,
                verbosity=-1,
                random_state=42,
            )
            model.fit(
                X_train,
                residual_train,
                eval_set=[(X_train, residual_train), (X_test, residual_test)],
                eval_metric="rmse",
            )

            duration = time.perf_counter() - start_time
            print(f"✅ Terminé en {duration:.2f} secondes (soit {duration / 60:.2f} minutes)")
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
            raw_evals_result = model.evals_result_
            eval_names = list(raw_evals_result)
            eval_metric = "rmse" if "rmse" in raw_evals_result[eval_names[0]] else next(
                iter(raw_evals_result[eval_names[0]])
            )
            evals_result = {
                f"validation_{i}": {"rmse": raw_evals_result[name][eval_metric]}
                for i, name in enumerate(eval_names[:2])
            }
            abs_error = np.abs(Y_test[col].to_numpy() - pred.to_numpy())

            print("")
            print("--- RÉSULTATS ---")
            print(f"Score R² (TEST) : {r2:.4f}")
            print(f"Score R² (TRAIN) : {r2_train:.4f} (Pour vérifier l'overfitting)")
            print(f"Erreur moyenne (MAE) : {mae:.2f} km/h")
            print(f"RMSE : {rmse:.2f} km/h")
            print("Pourcentage du temps où l'erreur est sous le seuil :")
            for threshold in (2, 5, 10, 15, 20):
                pct_within_threshold = np.mean(abs_error < threshold) * 100
                print(f"  < {threshold:2d} km/h : {pct_within_threshold:.1f}%")

        with open("last_mae.txt", "r") as f:
            last_mae = float(f.readlines()[0])
        with open("last_mae.txt", "w") as f:
            f.write(str(mae))

        delta = last_mae - mae  # > 0 = improvement
        pct = abs(delta) / last_mae * 100
        k = min(20, 1 + int(pct / 0.5))  # 1 emoji per 0.5% change

        print(f"{target}: {last_mae:.3f} -> model={mae:.3f} ({-pct if delta < 0 else pct:+.1f}%)")
        if round(last_mae, 3) == round(mae, 3):
            print("🟨" * 3)
        elif delta > 0:
            print("🟩" * k)
        else:
            print("🟥" * k)

        with timed_step(f"CHARTS [{target}]"):
            charts(
                X_test,
                Y_test,
                target,
                model,
                "1 2  4 5 6 7 8 9",
                Y_train=Y_train,
                evals_result=evals_result,
                pred_train=pred_train,
                pred_test=pred,
            )


if __name__ == "__main__":
    main()
