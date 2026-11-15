import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "meteoswiss_data"
PATTERN = "ogd-smn_*_t_historical_*.csv"

RENAME = {
    "station_abbr": "station",
    "tre200s0": "temp_c",
    "xchills0": "wind_chill_c",
    "ure200s0": "humidity_pct",
    "tde200s0": "dew_point_c",
    "pva200s0": "vapour_pressure_hpa",
    "prestas0": "pressure_station_hpa",
    "pp0qnhs0": "pressure_qnh_hpa",
    "pp0qffs0": "pressure_qff_hpa",
    "fu3010z0": "wind_speed_kmh",
    "fkl010z0": "wind_speed_ms",
    "fu3010z1": "wind_gust_max_kmh",
    "fkl010z1": "wind_gust_max_ms",
    "fu3010z3": "wind_gust_mean_kmh",
    "fkl010z3": "wind_gust_mean_ms",
    "dkl010z0": "wind_dir_deg",
    "fve010z0": "wind_vector_speed_ms",
    "rre150z0": "precip_mm",
}

def _load_stations(stations):
    stations = tuple(dict.fromkeys(stations))
    files = sorted(DATA_DIR.rglob(PATTERN))
    if not files:
        raise FileNotFoundError(f"no files matching {PATTERN} in {DATA_DIR}")

    parts = {station: [] for station in stations}
    for f in files:
        part = pd.read_csv(
            f,
            sep=";",
            encoding="utf-8",
            na_values=["-", ""],
            parse_dates=["reference_timestamp"],
            dayfirst=True,
        )
        for station in stations:
            station_part = part[part["station_abbr"] == station]
            if not station_part.empty:
                print(f"{f.name} [{station}]: {len(station_part)} rows")
            parts[station].append(station_part)

    result = {}
    for station, station_parts in parts.items():
        df = pd.concat(station_parts, ignore_index=True)
        df = df.rename(columns=RENAME)
        df = df.drop_duplicates(subset="reference_timestamp", keep="last")
        df = df.set_index("reference_timestamp").sort_index()
        df.index.name = "timestamp"
        result[station] = df
    return result


def load_data(station="BOU"):
    return _load_stations([station])[station]


def load_prediction_data(target_station="BOU", feature_stations=("AIG", "ORO")):
    """Load the target station and align nearby wind observations by timestamp."""
    stations = [target_station, *feature_stations]
    station_data = _load_stations(stations)
    df = station_data[target_station].copy()

    wind_columns = ["wind_speed_kmh", "wind_dir_deg", "wind_gust_max_kmh"]
    for station in feature_stations:
        features = station_data[station][wind_columns].add_prefix(f"{station.lower()}_")
        df = df.join(features, how="left")

    return df


def clean_data(df, unit="kmh"):
    if unit == "kmh":
        df = df.drop(columns=[c for c in df.columns if c.endswith("_ms")])
        required = ["wind_speed_kmh", "wind_dir_deg", "wind_gust_max_kmh"]
    else:
        df = df.drop(columns=[c for c in df.columns if c.endswith("_kmh")])
        required = ["wind_speed_ms", "wind_dir_deg", "wind_gust_max_ms"]

    df = df.dropna(axis=1, how="all")  # drop fully empty columns
    df = df.drop(columns=["station", "wind_chill_c", "dew_point_c", "vapour_pressure_hpa", "pressure_station_hpa", "pressure_qnh_hpa"])
    df = df.dropna(subset=required)    # drop rows missing the target columns
    return df

def show_new_parameters(df):
    first = pd.to_datetime(df.apply(pd.Series.first_valid_index)).dropna()
    start = df.index[0].normalize()

    last_added_parameter = first.max()
    print('')
    print('<<<<< PARAMETERS ADDING DATES >>>>>')
    for day, group in first.groupby(first.dt.normalize()):
        names = ", ".join(group.index)
        if day == start:
            print(f"SINCE START ({day.day} {day:%B %Y}) : {names}")
        else:
            print(f"ADDED ON THE {day.day} {day:%B %Y} : {names}")
        last_added_parameter = day
    print('')
    return last_added_parameter
