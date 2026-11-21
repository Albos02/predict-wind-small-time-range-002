"""Shared feature-building logic for wind prediction."""

import json

import numpy as np

from prediction_variables import (
    COLS,
    FEATURE_STATIONS,
    HORIZON,
    INTERPOLATION_NAN_LIMIT,
    LAGS,
    MODEL_PARAMETERS,
    STATION_COORDINATES_PATH,
    STATION_PREFIXES,
    TARGETS,
    TARGET_STATION,
    TREND_COLS,
    TREND_PERIODS,
    _DIRECTION_COLS,
    _DIRECTION_FEATURE_COLS,
    _FEATURE_STATION_PREFIXES,
    _NEIGHBOR_SPEED_GUST_COLS,
    _NEIGHBOR_VECTOR_COLS,
)


def add_advection_features(df):
    """Describe whether each feature station is transporting wind toward BOU."""
    with STATION_COORDINATES_PATH.open(encoding="utf-8") as stations_file:
        stations = json.load(stations_file)
    coordinates = {
        station[0]: (float(station[1]), float(station[2]))
        for station in stations
        if station[1] is not None and station[2] is not None
    }
    target_lat, target_lon = coordinates[TARGET_STATION]
    result = df.copy()

    for station in FEATURE_STATIONS:
        station_lat, station_lon = coordinates[station]
        mean_lat = np.deg2rad((target_lat + station_lat) / 2)
        east_km = (target_lon - station_lon) * 111.32 * np.cos(mean_lat)
        north_km = (target_lat - station_lat) * 111.32
        distance_km = float(np.hypot(east_km, north_km))
        unit_east = east_km / distance_km if distance_km else 0.0
        unit_north = north_km / distance_km if distance_km else 0.0
        prefix = f"{station.lower()}_"
        u = result[f"{prefix}wind_u_kmh"]
        v = result[f"{prefix}wind_v_kmh"]
        speed = np.hypot(u, v)
        toward_bou = u * unit_east + v * unit_north
        cross_to_bou = -u * unit_north + v * unit_east

        result[f"{prefix}distance_to_bou_km"] = distance_km
        result[f"{prefix}flow_toward_bou_kmh"] = toward_bou
        result[f"{prefix}crossflow_to_bou_kmh"] = cross_to_bou
        result[f"{prefix}flow_alignment_to_bou"] = np.divide(
            toward_bou,
            speed,
            out=np.zeros(len(result), dtype=float),
            where=np.asarray(speed) > 0.1,
        )
        # A bounded travel-time feature distinguishes a useful upwind signal
        # from a station whose current flow points away from BOU.
        result[f"{prefix}advection_time_to_bou_min"] = np.where(
            toward_bou > 0.1,
            np.minimum(distance_km / np.maximum(toward_bou, 0.1) * 60, 180),
            180.0,
        )

    return result


def prepare(df, last_added_parameter):
    df = df[COLS].loc[last_added_parameter:].copy()  # start where all features exist

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

    df = add_advection_features(df)

    df = df.asfreq("10min")

    if INTERPOLATION_NAN_LIMIT > 0:
        df = df.interpolate(limit=INTERPOLATION_NAN_LIMIT)
    return df


def split(df, test_from="2023-01-01"):
    return df[df.index < test_from], df[df.index >= test_from]


def make_xy(df):
    # Neighbor absolute speed/gust and vectors are recoverable from BOU plus
    # the relative deltas, so keep those instead of duplicate raw readings.
    excluded_cols = [
        *_DIRECTION_FEATURE_COLS,
        *_NEIGHBOR_SPEED_GUST_COLS,
        *_NEIGHBOR_VECTOR_COLS,
    ]
    X = df.drop(columns=excluded_cols).copy()

    # Past direction is already encoded in the wind-vector components, so
    # avoid duplicating it with lagged sine/cosine direction values.
    lag_features = df.drop(
        columns=[*_DIRECTION_FEATURE_COLS, *_NEIGHBOR_SPEED_GUST_COLS]
    )
    for lag in LAGS:
        lag_columns = lag_features.columns
        if lag >= 6:
            # Neighbor readings older than one hour add little for this
            # 20-minute horizon; retain longer history only for the target.
            lag_columns = [
                col for col in lag_columns
                if not col.startswith(_FEATURE_STATION_PREFIXES)
            ]
        if lag in (3, 4):
            # These absolute neighbor vectors are represented by the target
            # vectors and relative vector deltas at the same lag.
            lag_columns = [
                col for col in lag_columns
                if not (
                    col.startswith(_NEIGHBOR_VECTOR_COLS)
                    and "_delta_from_bou" not in col
                )
            ]
        X = X.join(
            lag_features[lag_columns].shift(lag).add_suffix(f"_lag{lag}")
        )
    for tr in TREND_PERIODS:
        a, b = map(int, tr.split("-"))  # a = older point, b = newer point (0 = now)
        X = X.join((df[TREND_COLS].shift(b) - df[TREND_COLS].shift(a)).add_suffix(f"_trend{tr}"))
    y = df[TARGETS].shift(-HORIZON).add_prefix("y_")
    data = X.join(y).dropna()

    print("Shape of the data trained/tested on : ", data.shape)
    return data[X.columns], data[y.columns]
