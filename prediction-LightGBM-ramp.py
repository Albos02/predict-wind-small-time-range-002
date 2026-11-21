"""Forecast wind speed with a high-recall, ramp-aware LightGBM model."""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from data_management import clean_data, load_prediction_data, show_new_parameters
from prediction_base import make_xy, prepare
from prediction_charts import charts
from prediction_variables import COLS, FEATURE_STATIONS, HORIZON, TARGETS, TARGET_STATION
from time_management import timed_step


VALIDATION_START = pd.Timestamp("2023-01-01")
TEST_START = pd.Timestamp("2025-01-01")
RAMP_THRESHOLD_KMH = 5.0
MAX_VALIDATION_MAE_KMH = 2.6


def _model_parameters():
    return {
        "n_estimators": 400,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 100,
        "reg_lambda": 2.0,
        "colsample_bytree": 0.85,
        "device_type": "gpu",
        "gpu_device_id": 0,
        "max_bin": 63,
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": 42,
    }


def _classifier(labels):
    labels = pd.Series(labels)
    if labels.nunique() < 2:
        return DummyClassifier(strategy="prior")
    return LGBMClassifier(objective="binary", **_model_parameters())


class RampAwareModel:
    """Predict ramp probability, direction, and magnitude as separate tasks."""

    def __init__(self, ramp_threshold=RAMP_THRESHOLD_KMH):
        self.ramp_threshold = ramp_threshold

    def fit(self, X, delta):
        delta = pd.Series(delta, index=X.index)
        ramp = delta.abs() >= self.ramp_threshold
        self.event_model = _classifier(ramp.astype(int)).fit(X, ramp.astype(int))

        if ramp.any():
            ramp_X = X.loc[ramp]
            ramp_delta = delta.loc[ramp]
            direction = (ramp_delta > 0).astype(int)
            self.direction_model = _classifier(direction).fit(ramp_X, direction)
            self.magnitude_model = LGBMRegressor(
                objective="mae", **_model_parameters()
            ).fit(ramp_X, ramp_delta.abs())
        else:
            self.direction_model = None
            self.magnitude_model = None
        return self

    @staticmethod
    def _positive_probability(model, X):
        if model is None:
            return np.full(len(X), 0.5)
        classes = np.asarray(model.classes_)
        positive = np.flatnonzero(classes == 1)
        if len(positive) == 0:
            return np.zeros(len(X))
        return model.predict_proba(X)[:, positive[0]]

    def predict_parts(self, X):
        event_probability = self._positive_probability(self.event_model, X)
        up_probability = self._positive_probability(self.direction_model, X)
        if self.magnitude_model is None:
            magnitude = np.full(len(X), self.ramp_threshold)
        else:
            magnitude = np.maximum(
                self.magnitude_model.predict(X), self.ramp_threshold
            )
        return event_probability, up_probability, magnitude

    def predict_hybrid_delta(self, X, event_probability_cutoff, baseline_delta):
        event_probability, up_probability, magnitude = self.predict_parts(X)
        sign = np.where(up_probability >= 0.5, 1.0, -1.0)
        predicted_ramp = event_probability >= event_probability_cutoff
        delta = np.where(predicted_ramp, sign * magnitude, np.asarray(baseline_delta))
        return delta, predicted_ramp


class RampForecast:
    """Expose a selected ramp model as a delta predictor for chart utilities."""

    def __init__(self, ramp_model, baseline_model, event_probability_cutoff, target):
        self.ramp_model = ramp_model
        self.baseline_model = baseline_model
        self.event_probability_cutoff = event_probability_cutoff
        self.target = target

    def predict(self, X):
        baseline_delta = self.baseline_model.predict(X)
        delta, _ = self.ramp_model.predict_hybrid_delta(
            X, self.event_probability_cutoff, baseline_delta
        )
        current = X[self.target].to_numpy()
        return np.maximum(0.0, current + delta) - current


def _select_hybrid_cutoff(
    y_speed,
    current_speed,
    baseline_delta,
    ramp_model,
    X_validation,
    ramp_threshold,
    max_mae,
):
    y_speed = np.asarray(y_speed)
    current_speed = np.asarray(current_speed)
    baseline_delta = np.asarray(baseline_delta)
    event_probability, up_probability, magnitude = ramp_model.predict_parts(X_validation)
    specialist_delta = np.where(up_probability >= 0.5, magnitude, -magnitude)
    actual_ramp = np.abs(y_speed - current_speed) >= ramp_threshold

    cutoffs = [*np.linspace(0.05, 0.95, 91), 1.01]
    valid_candidates = []
    for cutoff in cutoffs:
        use_specialist = event_probability >= cutoff
        predicted_delta = np.where(use_specialist, specialist_delta, baseline_delta)
        predicted_speed = np.maximum(0.0, current_speed + predicted_delta)
        mae = mean_absolute_error(y_speed, predicted_speed)
        if mae > max_mae:
            continue

        final_delta = predicted_speed - current_speed
        predicted_ramp = np.abs(final_delta) >= ramp_threshold
        recall = recall_score(actual_ramp, predicted_ramp, zero_division=0)
        precision = precision_score(actual_ramp, predicted_ramp, zero_division=0)
        f1 = f1_score(actual_ramp, predicted_ramp, zero_division=0)
        valid_candidates.append(
            (
                (recall, precision, -mae, cutoff),
                cutoff,
                mae,
                precision,
                recall,
                f1,
                int(predicted_ramp.sum()),
            )
        )

    if not valid_candidates:
        raise ValueError(
            f"No ramp cutoff meets the validation MAE ceiling of {max_mae:.2f} km/h."
        )
    _, cutoff, mae, precision, recall, f1, predicted_ramps = max(
        valid_candidates, key=lambda item: item[0]
    )
    return {
        "cutoff": cutoff,
        "mae": mae,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_ramps": predicted_ramps,
        "actual_ramps": int(actual_ramp.sum()),
    }


def _report_metrics(name, y_speed, current_speed, predicted_speed):
    y_speed = np.asarray(y_speed)
    current_speed = np.asarray(current_speed)
    predicted_speed = np.asarray(predicted_speed)
    actual_delta = y_speed - current_speed
    predicted_delta = predicted_speed - current_speed

    print(f"\n{name}")
    print(f"  MAE:  {mean_absolute_error(y_speed, predicted_speed):.4f} km/h")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_speed, predicted_speed)):.4f} km/h")
    print(f"  R²:   {r2_score(y_speed, predicted_speed):.4f}")
    for threshold in (3.0, 5.0):
        actual_ramp = np.abs(actual_delta) >= threshold
        predicted_ramp = np.abs(predicted_delta) >= threshold
        selected = actual_ramp & predicted_ramp
        direction_accuracy = (
            accuracy_score(
                actual_delta[selected] > 0,
                predicted_delta[selected] > 0,
            )
            if selected.any()
            else float("nan")
        )
        print(
            f"  ±{threshold:.0f} km/h ramps: "
            f"precision={precision_score(actual_ramp, predicted_ramp, zero_division=0):.3f}, "
            f"recall={recall_score(actual_ramp, predicted_ramp, zero_division=0):.3f}, "
            f"F1={f1_score(actual_ramp, predicted_ramp, zero_division=0):.3f}, "
            f"direction_accuracy={direction_accuracy:.3f} "
            f"({int(predicted_ramp.sum()):,} predicted / {int(actual_ramp.sum()):,} actual)"
        )


def _report_standard_results(target, y_test, y_train, predicted_test, predicted_train):
    mae = mean_absolute_error(y_test, predicted_test)
    rmse = np.sqrt(mean_squared_error(y_test, predicted_test))
    r2 = r2_score(y_test, predicted_test)
    r2_train = r2_score(y_train, predicted_train)
    abs_error = np.abs(np.asarray(y_test) - np.asarray(predicted_test))

    print("\n--- RÉSULTATS ---")
    print(f"Score R² (TEST) : {r2:.4f}")
    print(f"Score R² (TRAIN) : {r2_train:.4f} (Pour vérifier l'overfitting)")
    print(f"Erreur moyenne (MAE) : {mae:.2f} km/h")
    print(f"RMSE : {rmse:.2f} km/h")
    print("Pourcentage du temps où l'erreur est sous le seuil :")
    for threshold in (2, 5, 10, 15, 20):
        pct_within_threshold = np.mean(abs_error < threshold) * 100
        print(f"  < {threshold:2d} km/h : {pct_within_threshold:.1f}%")
    return mae


def main():
    with timed_step("LOAD DATA"):
        df = load_prediction_data(
            target_station=TARGET_STATION,
            feature_stations=FEATURE_STATIONS,
        )

    with timed_step("CLEAN DATA"):
        df = clean_data(df, unit="kmh")

    with timed_step("CHECK NEW PARAMETERS"):
        last_added_parameter = show_new_parameters(df[COLS])

    with timed_step("PREPARE FEATURES"):
        df = prepare(df, last_added_parameter)

    with timed_step("BUILD X / Y AND TIME SPLITS"):
        # Build lags over the full series; purge rows whose future labels cross
        # validation or test boundaries, matching evaluate_dynamics.py.
        X, Y = make_xy(df)
        horizon_delta = pd.Timedelta(minutes=HORIZON * 10)
        train_mask = X.index < VALIDATION_START - horizon_delta
        validation_mask = (X.index >= VALIDATION_START) & (
            X.index < TEST_START - horizon_delta
        )
        test_mask = X.index >= TEST_START
        print(
            f"Samples: train={train_mask.sum():,}, "
            f"validation={validation_mask.sum():,}, test={test_mask.sum():,}; "
            f"ramp threshold=±{RAMP_THRESHOLD_KMH:.1f} km/h / {HORIZON * 10} min"
        )

    for target in TARGETS:
        col = f"y_{target}"
        current = X[target]
        delta = Y[col] - current

        with timed_step(f"TRAIN VALIDATION MODELS [{target}]"):
            validation_delta_model = LGBMRegressor(
                objective="mae", **_model_parameters()
            ).fit(X.loc[train_mask], delta.loc[train_mask])
            validation_baseline_delta = validation_delta_model.predict(
                X.loc[validation_mask]
            )
            validation_ramp_model = RampAwareModel().fit(
                X.loc[train_mask], delta.loc[train_mask]
            )
            validation_selection = _select_hybrid_cutoff(
                Y.loc[validation_mask, col],
                current.loc[validation_mask],
                validation_baseline_delta,
                validation_ramp_model,
                X.loc[validation_mask],
                RAMP_THRESHOLD_KMH,
                MAX_VALIDATION_MAE_KMH,
            )
        print(
            f"Validation-selected hybrid cutoff: {validation_selection['cutoff']:.2f}; "
            f"MAE={validation_selection['mae']:.4f} km/h "
            f"(ceiling {MAX_VALIDATION_MAE_KMH:.2f}), "
            f"±{RAMP_THRESHOLD_KMH:.0f} recall={validation_selection['recall']:.3f}, "
            f"precision={validation_selection['precision']:.3f}, "
            f"F1={validation_selection['f1']:.3f}; "
            f"{validation_selection['predicted_ramps']:,} predicted / "
            f"{validation_selection['actual_ramps']:,} actual ramps"
        )

        final_train_mask = train_mask | validation_mask
        with timed_step(f"TRAIN MODEL [{target}]"):
            # Keep the ordinary delta model as an MAE reference.
            delta_model = LGBMRegressor(
                objective="mae", **_model_parameters()
            ).fit(X.loc[final_train_mask], delta.loc[final_train_mask])
            ramp_model = RampAwareModel().fit(
                X.loc[final_train_mask], delta.loc[final_train_mask]
            )

        X_test = X.loc[test_mask]
        Y_test = Y.loc[test_mask]
        current_test = current.loc[test_mask].to_numpy()
        actual_test = Y_test[col].to_numpy()

        delta_prediction = delta_model.predict(X_test)
        baseline_prediction = np.maximum(0.0, current_test + delta_prediction)
        _report_metrics(
            "Single LightGBM delta baseline", actual_test, current_test, baseline_prediction
        )

        hybrid_delta, _ = ramp_model.predict_hybrid_delta(
            X_test,
            validation_selection["cutoff"],
            delta_prediction,
        )
        hybrid_prediction = np.maximum(0.0, current_test + hybrid_delta)
        _report_metrics(
            f"Hybrid baseline + ≥{RAMP_THRESHOLD_KMH:.0f} km/h ramp specialist "
            f"(cutoff={validation_selection['cutoff']:.2f})",
            actual_test,
            current_test,
            hybrid_prediction,
        )

        X_train = X.loc[final_train_mask]
        current_train = current.loc[final_train_mask].to_numpy()
        y_train = Y.loc[final_train_mask, col].to_numpy()
        train_baseline_delta = delta_model.predict(X_train)
        train_hybrid_delta, _ = ramp_model.predict_hybrid_delta(
            X_train,
            validation_selection["cutoff"],
            train_baseline_delta,
        )
        hybrid_train_prediction = np.maximum(
            0.0, current_train + train_hybrid_delta
        )

        chart_model = RampForecast(
            ramp_model,
            delta_model,
            validation_selection["cutoff"],
            target,
        )
        with timed_step(f"CHARTS [{target}]"):
            charts(
                X_test,
                Y_test,
                target,
                chart_model,
                "1 2 4 5 6 7 8 9 10",
                pred_test=hybrid_prediction,
            )

        baseline_mae = mean_absolute_error(actual_test, baseline_prediction)
        hybrid_mae = _report_standard_results(
            target,
            actual_test,
            y_train,
            hybrid_prediction,
            hybrid_train_prediction,
        )
        delta_mae = baseline_mae - hybrid_mae
        pct = abs(delta_mae) / baseline_mae * 100 if baseline_mae else 0.0
        print(
            f"{target}: {baseline_mae:.3f} -> model={hybrid_mae:.3f} "
            f"({-pct if delta_mae < 0 else pct:+.1f}%)"
        )
        if round(baseline_mae, 3) == round(hybrid_mae, 3):
            print("🟨" * 3)
        else:
            emoji = "🟩" if delta_mae > 0 else "🟥"
            print(emoji * min(20, 1 + int(pct / 0.5)))


if __name__ == "__main__":
    main()
