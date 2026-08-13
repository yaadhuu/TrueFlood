"""
Fetch daily historical weather for a USGS gauge's coordinates (Open-Meteo,
free, no API key).

    GET https://archive-api.open-meteo.com/v1/archive
        ?latitude=&longitude=&start_date=&end_date=
        &daily=precipitation_sum,temperature_2m_mean&timezone=UTC

Coordinates come from the USGS site service (see fetch_usgs.site_metadata),
so the rainfall series is genuinely co-located with the gauge rather than
attached to a nearby city.

Cached to ml_pipeline/raw/openmeteo_<site>.csv; --refresh forces a re-fetch.

Note: this is the *archive* API.  backend/weather.py uses the separate
*forecast* API for Layer 3.  Different endpoints, different purposes.

Usage:
    python ml_pipeline/fetch_openmeteo.py --site 05389500
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "raw")
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

sys.path.insert(0, SCRIPT_DIR)
from fetch_usgs import DEFAULT_END, DEFAULT_SITE, DEFAULT_START, site_metadata  # noqa: E402


def fetch(site: str, lat: float, lon: float, start: str, end: str,
          *, refresh: bool = False, timeout: int = 120) -> pd.DataFrame:
    os.makedirs(RAW_DIR, exist_ok=True)
    cache = os.path.join(RAW_DIR, f"openmeteo_{site}.csv")
    if os.path.exists(cache) and not refresh:
        print(f"[openmeteo] cache hit -> {cache}")
        return pd.read_csv(cache)

    print(f"[openmeteo] fetching {lat},{lon} {start} .. {end}")
    r = requests.get(
        ARCHIVE_URL,
        params={
            "latitude": lat, "longitude": lon,
            "start_date": start, "end_date": end,
            "daily": "precipitation_sum,temperature_2m_mean",
            "timezone": "UTC",
        },
        timeout=timeout,
    )
    r.raise_for_status()
    daily = r.json()["daily"]
    df = pd.DataFrame({
        "date": daily["time"],
        "precip_mm": daily["precipitation_sum"],
        "temp_mean_c": daily["temperature_2m_mean"],
    })
    df.to_csv(cache, index=False)
    print(f"[openmeteo] wrote {len(df)} rows -> {cache}")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default=DEFAULT_SITE)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    if args.lat is None or args.lon is None:
        meta = site_metadata(args.site)
        args.lat, args.lon = meta["latitude"], meta["longitude"]
        print(f"[openmeteo] site {args.site}: {meta['station_nm']}")

    df = fetch(args.site, args.lat, args.lon, args.start, args.end,
               refresh=args.refresh)
    p = pd.to_numeric(df["precip_mm"], errors="coerce").dropna()
    print(f"[openmeteo] rows={len(df)}  precip: mean={p.mean():.2f} "
          f"max={p.max():.1f} mm/day")
    return 0


if __name__ == "__main__":
    sys.exit(main())
