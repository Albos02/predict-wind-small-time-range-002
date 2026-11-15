"""Download MeteoSwiss 10-minute historical data for all listed stations."""

import json
from pathlib import Path

import requests


ROOT_DIR = Path(__file__).parent
STATIONS_FILE = ROOT_DIR / "coord_station_meteosuisse.json"
OUT_DIR = ROOT_DIR / "meteoswiss_data"
COLLECTION_URL = "https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-smn/items"


def download_station(session, station_code, station_name):
    station_id = station_code.lower()
    item_url = f"{COLLECTION_URL}/{station_id}"
    response = session.get(item_url, timeout=60)
    response.raise_for_status()
    assets = response.json()["assets"]

    urls = [
        asset["href"]
        for key, asset in assets.items()
        if key.endswith(".csv") and "_t_" in key and "historical" in key
    ]
    if not urls:
        raise ValueError("no matching historical 10-minute CSV assets found")

    station_dir = OUT_DIR / station_id
    station_dir.mkdir(exist_ok=True)
    for url in urls:
        basename = url.rsplit("/", 1)[-1]
        filename = station_dir / basename
        flat_filename = OUT_DIR / basename

        # Migrate files downloaded by older versions of this script.
        if not filename.exists() and flat_filename.exists():
            flat_filename.replace(filename)

        if filename.exists():
            print(f"Using cached {filename.name}")
            continue

        print(f"Downloading {station_code} ({station_name}): {filename.name}")
        download = session.get(url, timeout=120)
        download.raise_for_status()
        filename.write_bytes(download.content)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    with STATIONS_FILE.open(encoding="utf-8") as stations_file:
        stations = json.load(stations_file)

    failures = []
    with requests.Session() as session:
        for station in stations:
            station_code, _, _, station_name, *_ = station
            try:
                download_station(session, station_code, station_name or "unknown")
            except (requests.RequestException, KeyError, ValueError) as error:
                print(f"Failed {station_code}: {error}")
                failures.append((station_code, error))

    if failures:
        failed_codes = ", ".join(code for code, _ in failures)
        raise SystemExit(
            f"\nCompleted with {len(failures)} station failure(s): {failed_codes}"
        )

    print(f"\nDone. Downloaded historical data for all {len(stations)} stations.")


if __name__ == "__main__":
    main()
