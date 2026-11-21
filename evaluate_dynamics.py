"""Evaluate 20-minute wind changes and ramp-aware forecasting candidates.

Run with ``python evaluate_dynamics.py`` after downloading the MeteoSwiss data.
The 2023-2024 period selects the ramp-event threshold; 2025 is held out for the
final comparison. No test labels are used for model or threshold selection.
"""

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    fbeta_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from data_management import clean_data, load_prediction_data, show_new_parameters
from prediction import (
    COLS,
    FEATURE_STATIONS,
    HORIZON,
    TARGET_STATION,
    make_xy,
    prepare,
)


VALIDATION_START = pd.Timestamp("2023-01-01")
TEST_START = pd.Timestamp("2025-01-01")
RAMP_THRESHOLD_KMH = 3.0


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


def _classifier(y):
    if pd.Series(y).nunique() < 2:
        return DummyClassifier(strategy="prior")
    return LGBMClassifier(objective="binary", **_model_parameters())


def _fit_classifier(model, X, y):
    model.fit(X, y)
    return model


@dataclass
class RampAwareModel:
    """Separate ramp occurrence, ramp direction, and ramp magnitude."""

    ramp_threshold: float = RAMP_THRESHOLD_KMH

    def fit(self, X, delta):
        delta = pd.Series(delta, index=X.index)
        ramp = delta.abs() >= self.ramp_threshold
        self.event_model = _fit_classifier(_classifier(ramp.astype(int)), X, ramp.astype(int))

        if ramp.any():
            ramp_X = X.loc[ramp]
            ramp_delta = delta.loc[ramp]
            self.direction_model = _fit_classifier(
                _classifier((ramp_delta > 0).astype(int)),
                ramp_X,
                (ramp_delta > 0).astype(int),
            )
            self.magnitude_model = LGBMRegressor(
                objective="mae", **_model_parameters()
            ).fit(ramp_X, ramp_delta.abs())
        else:
            self.direction_model = None
            self.magnitude_model = None
        self.ramp_rate = float(ramp.mean())
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

    def predict_delta(self, X, event_probability_cutoff):
        event_probability, up_probability, magnitude = self.predict_parts(X)
        sign = np.where(up_probability >= 0.5, 1.0, -1.0)
        is_ramp = event_probability >= event_probability_cutoff
        return np.where(is_ramp, sign * magnitude, 0.0), is_ramp


def choose_event_cutoffs(y_delta, event_probability, ramp_threshold):
    actual_event = np.abs(np.asarray(y_delta)) >= ramp_threshold
    candidates = np.linspace(0.05, 0.95, 91)
    cutoffs = {}
    for beta in (0.5, 1.0, 2.0):
        scored = [
            (
                fbeta_score(
                    actual_event,
                    event_probability >= cutoff,
                    beta=beta,
                    zero_division=0,
                ),
                cutoff,
            )
            for cutoff in candidates
        ]
        cutoffs[beta] = max(scored, key=lambda item: (item[0], item[1]))
    return cutoffs


def choose_delta_scales(y_delta, predicted_delta, ramp_threshold):
    """Calibrate change amplitude on validation without touching the test set."""
    actual_event = np.abs(np.asarray(y_delta)) >= ramp_threshold
    predicted_delta = np.asarray(predicted_delta)
    scales = np.arange(1.0, 3.01, 0.05)
    selected = {}
    for beta in (0.5, 1.0, 2.0):
        scored = [
            (
                fbeta_score(
                    actual_event,
                    np.abs(scale * predicted_delta) >= ramp_threshold,
                    beta=beta,
                    zero_division=0,
                ),
                scale,
            )
            for scale in scales
        ]
        # For tied validation scores, prefer the smaller adjustment.
        selected[beta] = max(scored, key=lambda item: (item[0], -item[1]))
    return selected


def _metrics(y_speed, y_now, predicted_speed, ramp_threshold):
    y_speed = np.asarray(y_speed)
    y_now = np.asarray(y_now)
    predicted_speed = np.asarray(predicted_speed)
    actual_delta = y_speed - y_now
    predicted_delta = predicted_speed - y_now
    actual_ramp = np.abs(actual_delta) >= ramp_threshold
    predicted_ramp = np.abs(predicted_delta) >= ramp_threshold
    directional = actual_ramp
    direction_coverage = (
        float(predicted_ramp[directional].mean()) if directional.any() else float("nan")
    )
    forecasted_actual_ramps = directional & predicted_ramp
    direction_accuracy = (
        accuracy_score(
            actual_delta[forecasted_actual_ramps] > 0,
            predicted_delta[forecasted_actual_ramps] > 0,
        )
        if forecasted_actual_ramps.any()
        else float("nan")
    )
    return {
        "mae_kmh": mean_absolute_error(y_speed, predicted_speed),
        "rmse_kmh": np.sqrt(mean_squared_error(y_speed, predicted_speed)),
        "r2": r2_score(y_speed, predicted_speed),
        "bias_kmh": float(np.mean(predicted_speed - y_speed)),
        "ramp_precision": precision_score(actual_ramp, predicted_ramp, zero_division=0),
        "ramp_recall": recall_score(actual_ramp, predicted_ramp, zero_division=0),
        "ramp_f1": f1_score(actual_ramp, predicted_ramp, zero_division=0),
        "actual_ramp_prevalence": float(actual_ramp.mean()),
        "always_predict_ramp_f1": f1_score(
            actual_ramp, np.ones(len(actual_ramp), dtype=bool), zero_division=0
        ),
        "directional_coverage_on_actual_ramps": direction_coverage,
        "direction_accuracy_when_forecasting_ramp": direction_accuracy,
    }


def _report_ramp_sweep(name, actual_delta, predicted_delta):
    actual_delta = np.asarray(actual_delta)
    predicted_delta = np.asarray(predicted_delta)
    print(f"\n{name}: ramp thresholds (actual and forecast change magnitude)")
    print("  threshold   prevalence   precision   recall      F1   dir-accuracy")
    for threshold in (0.5, 1.0, 2.0, 3.0, 5.0):
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
            f"  {threshold:7.1f}   {actual_ramp.mean():10.3f}   "
            f"{precision_score(actual_ramp, predicted_ramp, zero_division=0):9.3f}   "
            f"{recall_score(actual_ramp, predicted_ramp, zero_division=0):6.3f}   "
            f"{f1_score(actual_ramp, predicted_ramp, zero_division=0):6.3f}   "
            f"{direction_accuracy:12.3f}"
        )


def _split_masks(index, validation_start, test_start, horizon):
    # Purge labels that reach across a split boundary: labels are t + horizon.
    label_offset = pd.Timedelta(minutes=10 * horizon)
    return (
        index < validation_start - label_offset,
        (index >= validation_start) & (index < test_start - label_offset),
        index >= test_start,
    )


def _fit_delta_model(X, delta):
    return LGBMRegressor(objective="mae", **_model_parameters()).fit(X, delta)


def _report_metrics(name, y, now, prediction, threshold):
    scores = _metrics(y, now, prediction, threshold)
    print(f"\n{name}")
    for metric, value in scores.items():
        print(f"  {metric:38s} {value:.4f}")
    return scores


def run_evaluation(
    ramp_threshold=RAMP_THRESHOLD_KMH,
    validation_start=VALIDATION_START,
    test_start=TEST_START,
    station_ablation=True,
):
    raw = load_prediction_data(
        target_station=TARGET_STATION,
        feature_stations=FEATURE_STATIONS,
    )
    observations = clean_data(raw, unit="kmh")
    last_added_parameter = show_new_parameters(observations[COLS])
    prepared = prepare(observations, last_added_parameter)
    X, y_frame = make_xy(prepared)
    y = y_frame["y_wind_speed_kmh"].reindex(X.index)
    now = X["wind_speed_kmh"]
    delta = y - now

    train_mask, validation_mask, test_mask = _split_masks(
        X.index, validation_start, test_start, HORIZON
    )
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError(
            "Need training data before validation, validation observations before "
            f"{test_start.date()}, and test data from {test_start.date()} onward."
        )

    print(
        f"\nSamples: train={train_mask.sum():,}, validation={validation_mask.sum():,}, "
        f"test={test_mask.sum():,}; ramp threshold=±{ramp_threshold:.1f} km/h / "
        f"{HORIZON * 10} min"
    )
    print("LightGBM training device: GPU 0 (OpenCL)")

    # Select event cutoffs and continuous-change scales on 2023-2024. The final
    # 2025 period remains unseen until all model settings are fixed.
    validation_delta_model = _fit_delta_model(X.loc[train_mask], delta.loc[train_mask])
    validation_delta = validation_delta_model.predict(X.loc[validation_mask])
    delta_scales = choose_delta_scales(
        delta.loc[validation_mask], validation_delta, ramp_threshold
    )
    for beta, (validation_score, scale) in delta_scales.items():
        print(
            f"Validation-selected delta scale F{beta:g}: {scale:.2f}x "
            f"(validation F{beta:g}={validation_score:.3f})"
        )

    validation_model = RampAwareModel(ramp_threshold).fit(
        X.loc[train_mask], delta.loc[train_mask]
    )
    validation_probability = validation_model.predict_parts(X.loc[validation_mask])[0]
    event_cutoffs = choose_event_cutoffs(
        delta.loc[validation_mask], validation_probability, ramp_threshold
    )
    for beta, (validation_score, cutoff) in event_cutoffs.items():
        print(
            f"Validation-selected event cutoff F{beta:g}: {cutoff:.2f} "
            f"(validation F{beta:g}={validation_score:.3f})"
        )

    final_train_mask = train_mask | validation_mask
    X_train = X.loc[final_train_mask]
    delta_train = delta.loc[final_train_mask]
    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]
    now_test = now.loc[test_mask]

    # Simple extrapolation makes it clear whether the model sees more than
    # persistence or a continuation of the latest observed trend.
    previous_speed = X_test["wind_speed_kmh_lag1"]
    trend_prediction = np.maximum(
        0.0,
        now_test.to_numpy() + 2.0 * (now_test.to_numpy() - previous_speed.to_numpy()),
    )
    _report_metrics("Persistence", y_test, now_test, now_test, ramp_threshold)
    _report_metrics(
        "10-minute linear trend extrapolated to 20 minutes",
        y_test,
        now_test,
        trend_prediction,
        ramp_threshold,
    )

    delta_model = _fit_delta_model(X_train, delta_train)
    standard_prediction = now_test.to_numpy() + delta_model.predict(X_test)
    _report_metrics(
        "Single LightGBM delta regressor",
        y_test,
        now_test,
        standard_prediction,
        ramp_threshold,
    )
    raw_delta_prediction = standard_prediction - now_test.to_numpy()
    _report_ramp_sweep(
        "Single LightGBM delta regressor",
        y_test.to_numpy() - now_test.to_numpy(),
        raw_delta_prediction,
    )
    for beta, (validation_score, scale) in delta_scales.items():
        scaled_prediction = np.maximum(
            0.0,
            now_test.to_numpy() + scale * raw_delta_prediction,
        )
        _report_metrics(
            f"Delta regressor scaled {scale:.2f}x (validation F{beta:g})",
            y_test,
            now_test,
            scaled_prediction,
            ramp_threshold,
        )
        print(f"  validation_f{beta:g}: {validation_score:.3f}")
        if beta == 2.0:
            _report_ramp_sweep(
                "F2-calibrated delta regressor",
                y_test.to_numpy() - now_test.to_numpy(),
                scaled_prediction - now_test.to_numpy(),
            )

    final_ramp_model = RampAwareModel(ramp_threshold).fit(X_train, delta_train)
    for beta, (validation_score, event_cutoff) in event_cutoffs.items():
        ramp_delta, ramp_event = final_ramp_model.predict_delta(X_test, event_cutoff)
        ramp_prediction = np.maximum(0.0, now_test.to_numpy() + ramp_delta)
        _report_metrics(
            f"Ramp-aware model (validation F{beta:g} cutoff={event_cutoff:.2f})",
            y_test,
            now_test,
            ramp_prediction,
            ramp_threshold,
        )
        print(
            f"  validation_f{beta:g}: {validation_score:.3f}; "
            f"selected_ramp_events: {int(ramp_event.sum()):,} / {len(ramp_event):,}"
        )
    actual_ramp = np.abs(delta.loc[test_mask].to_numpy()) >= ramp_threshold
    if actual_ramp.any():
        _, up_probability, _ = final_ramp_model.predict_parts(X_test)
        direction_accuracy = accuracy_score(
            delta.loc[test_mask].to_numpy()[actual_ramp] > 0,
            up_probability[actual_ramp] >= 0.5,
        )
        print(
            "  direction_accuracy_on_all_actual_ramps: "
            f"{direction_accuracy:.4f}"
        )

    # Predict future wind components directly; direction changes can be
    # forecast even when scalar speed changes little.
    future_components = prepared[["wind_u_kmh", "wind_v_kmh"]].shift(-HORIZON).reindex(X.index)
    component_valid = future_components.notna().all(axis=1)
    component_train = final_train_mask & component_valid
    component_test = test_mask & component_valid
    u_model = LGBMRegressor(objective="mae", **_model_parameters()).fit(
        X.loc[component_train], future_components.loc[component_train, "wind_u_kmh"]
    )
    v_model = LGBMRegressor(objective="mae", **_model_parameters()).fit(
        X.loc[component_train], future_components.loc[component_train, "wind_v_kmh"]
    )
    vector_speed = np.hypot(
        u_model.predict(X.loc[component_test]),
        v_model.predict(X.loc[component_test]),
    )
    _report_metrics(
        "Future wind-vector components -> speed",
        y.loc[component_test],
        now.loc[component_test],
        vector_speed,
        ramp_threshold,
    )

    if station_ablation:
        print("\nNearby-station ablation (positive MAE difference means the station helps):")
        for station in FEATURE_STATIONS:
            prefix = f"{station.lower()}_"
            columns = [column for column in X.columns if not column.startswith(prefix)]
            ablated = _fit_delta_model(X_train[columns], delta_train)
            ablated_prediction = now_test.to_numpy() + ablated.predict(X_test[columns])
            ablated_mae = mean_absolute_error(y_test, ablated_prediction)
            all_feature_mae = mean_absolute_error(y_test, standard_prediction)
            print(
                f"  {station}: {ablated_mae - all_feature_mae:+.4f} "
                "km/h MAE when removed"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ramp-threshold",
        type=float,
        default=RAMP_THRESHOLD_KMH,
        help="Meaningful 20-minute speed change in km/h (default: 3.0).",
    )
    parser.add_argument(
        "--validation-start",
        type=pd.Timestamp,
        default=VALIDATION_START,
        help="Start of the event-threshold validation period.",
    )
    parser.add_argument(
        "--test-start",
        type=pd.Timestamp,
        default=TEST_START,
        help="Start of the final untouched test period.",
    )
    parser.add_argument(
        "--no-station-ablation",
        action="store_true",
        help="Skip per-station model refits.",
    )
    args = parser.parse_args()
    run_evaluation(
        ramp_threshold=args.ramp_threshold,
        validation_start=args.validation_start,
        test_start=args.test_start,
        station_ablation=not args.no_station_ablation,
    )


if __name__ == "__main__":
    main()
