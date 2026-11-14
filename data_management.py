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

REQUIRED = ["wind_speed_kmh", "wind_dir_deg", "wind_gust_max_kmh"]  # adjust to your targets


def load_data(station="BOU"):
    files = sorted(DATA_DIR.glob(PATTERN))
    if not files:
        raise FileNotFoundError(f"no files matching {PATTERN} in {DATA_DIR}")

    parts = []
    for f in files:
        part = pd.read_csv(
            f,
            sep=";",
            encoding="utf-8",
            na_values=["-", ""],
            parse_dates=["reference_timestamp"],
            dayfirst=True,
        )
        part = part[part["station_abbr"] == station]
        print(f"{f.name}: {len(part)} rows")
        parts.append(part)

    df = pd.concat(parts, ignore_index=True)
    df = df.rename(columns=RENAME)
    df = df.drop_duplicates(subset="reference_timestamp", keep="last")
    df = df.set_index("reference_timestamp").sort_index()
    df.index.name = "timestamp"
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
