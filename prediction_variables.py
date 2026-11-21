"""Configuration variables for the wind prediction workflow."""

from pathlib import Path


INTERPOLATION_NAN_LIMIT = 0  # 0=no interpolation, ... 2 = interpolates NaN series up to gap of 2

HORIZON = 2  # 2 x 10 min = 20 min
TARGET_STATION = "BOU"
# FEATURE_STATIONS = ("AIG", "ORO", "PUY", "EVI", "PRE", "CGI", "VEV", "CDM", "DOL", "MAH", "BIE", "MAR", "MOB", "SIO", "CHB", "MAS", "BOL", "THU", "ABO", "VIT", "MVE", "CHA", "NAP")
FEATURE_STATIONS = ("AIG", "ORO", "PUY", "VEV")
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
_DIRECTION_FEATURE_COLS = tuple(
    f"{prefix}_{component}"
    for prefix in _DIRECTION_COLS.values()
    for component in ("sin", "cos")
)
_FEATURE_STATION_PREFIXES = tuple(
    f"{station.lower()}_" for station in FEATURE_STATIONS
)
_NEIGHBOR_SPEED_GUST_COLS = tuple(
    f"{prefix}{col}"
    for prefix in _FEATURE_STATION_PREFIXES
    for col in ("wind_speed_kmh", "wind_gust_max_kmh")
)
_NEIGHBOR_VECTOR_COLS = tuple(
    f"{prefix}{col}"
    for prefix in _FEATURE_STATION_PREFIXES
    for col in ("wind_u_kmh", "wind_v_kmh")
)

TARGETS = ["wind_speed_kmh"]
LAGS = [1, 2, 3, 4, 5, 6, 9, 12, 36, 48]  # 10 min ... 24 h
CHARTS_DIR = Path(__file__).parent / "charts"
STATION_COORDINATES_PATH = Path(__file__).parent / "coord_station_meteosuisse.json"

TREND_PERIODS = ("1-0", "2-0", "3-0")
TREND_COLS = ["wind_speed_kmh", "wind_gust_max_kmh", "wind_u_kmh", "wind_v_kmh"] + [
    f"{prefix}{col}"
    for prefix in _FEATURE_STATION_PREFIXES
    for col in ("wind_u_kmh", "wind_v_kmh")
]
TREND_COLS += [
    f"{station.lower()}_{feature}"
    for station in FEATURE_STATIONS
    for feature in (
        "flow_toward_bou_kmh",
        "crossflow_to_bou_kmh",
        "flow_alignment_to_bou",
    )
]

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
