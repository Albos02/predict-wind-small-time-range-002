"""
Downloads BOU station 10-min ('t' granularity) historical data
from MeteoSwiss's STAC-based Open Data API (up to end of 2025),
concatenated into one CSV.
"""

import requests
import pandas as pd
from pathlib import Path

STATION = "bou"  # lowercase, as used in the STAC item id
OUT_DIR = Path("meteoswiss_data")
OUT_DIR.mkdir(exist_ok=True)

ITEM_URL = f"https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-smn/items/{STATION}"

# --- 1. Get the STAC item -> list of asset URLs ---
resp = requests.get(ITEM_URL)
resp.raise_for_status()
assets = resp.json()["assets"]

# --- 2. Keep only 10-min granularity ('_t_') historical CSVs (up to end of 2025) ---
wanted = [
    a["href"] for key, a in assets.items()
    if key.endswith(".csv") and "_t_" in key and "historical" in key
]

if not wanted:
    print("No matching assets found. Available asset keys:")
    for key in assets:
        print(" -", key)
    raise SystemExit(1)

# --- 3. Download each file ---
downloaded = []
for url in wanted:
    fname = OUT_DIR / url.split("/")[-1]
    if not fname.exists():
        print(f"Downloading {url} -> {fname}")
        r = requests.get(url)
        r.raise_for_status()
        fname.write_bytes(r.content)
    else:
        print(f"Using cached {fname}")
    downloaded.append(fname)

# --- 4. Concatenate into a single CSV (semicolon-separated, Windows-1252) ---
dfs = [pd.read_csv(f, sep=";", encoding="cp1252") for f in sorted(downloaded)]
full = pd.concat(dfs, ignore_index=True)

# Parse and ensure all records are up to end of 2025
timestamps = pd.to_datetime(full["reference_timestamp"], format="%d.%m.%Y %H:%M")
full = full[timestamps <= pd.Timestamp("2025-12-31 23:59:59")].reset_index(drop=True)

# full.to_csv(OUT_DIR / f"{STATION}_full.csv", sep=";", index=False)

print(f"\nDone. {len(full)} rows written to {OUT_DIR / f'{STATION}_full.csv'}")

