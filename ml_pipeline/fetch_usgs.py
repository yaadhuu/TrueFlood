"""
Fetch daily river-gauge data from USGS Water Services (no API key required).

    GET https://waterservices.usgs.gov/nwis/dv/
        ?format=json&sites=<SITE>&startDT=&endDT=
        &parameterCd=00060,00065&statCd=00003

    00060 = discharge (cfs)      00065 = gage height (ft)      00003 = daily mean

Site selection matters more than the code here.  Most USGS gauges publish
daily *discharge* but not daily *stage*, and the NWS flood thresholds that
this project uses as an external label are published in feet of stage.  The
default site below was chosen by scanning state gauge catalogues for sites
that have both:

  * a long daily 00065 (gage height) record, and
  * an NWS/NWPS gauge page with a published minor-flood stage.

Results are cached to ml_pipeline/raw/usgs_<site>.csv so repeat runs (CI) do
not hammer the API.  Pass --refresh to force a re-fetch.

Usage:
    python ml_pipeline/fetch_usgs.py --site 05389500
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "raw")

DV_URL = "https://waterservices.usgs.gov/nwis/dv/"
SITE_URL = "https://waterservices.usgs.gov/nwis/site/"

DEFAULT_SITE = "05389500"
DEFAULT_START = "2006-01-01"
DEFAULT_END = "2024-12-31"
CHUNK_YEARS = 3          # the dv service degrades on very long single requests
MISSING = {"-999999", "-999999.0", "", "Ice", "Eqp"}


def site_metadata(site: str, timeout: int = 90) -> dict:
    """Station name and coordinates, straight from the site service (RDB)."""
    r = requests.get(SITE_URL, params={"format": "rdb", "sites": site},
                     timeout=timeout)
    r.raise_for_status()
    lines = [ln for ln in r.text.splitlines() if not ln.startswith("#")]
    header = lines[0].split("\t")
    values = lines[2].split("\t")          # line 1 is the RDB type row
    row = dict(zip(header, values))
    return {
        "site_no": row.get("site_no", site),
        "station_nm": row.get("station_nm", ""),
        "latitude": float(row["dec_lat_va"]),
        "longitude": float(row["dec_long_va"]),
    }


def _fetch_window(site: str, start: str, end: str, timeout: int) -> pd.DataFrame:
    r = requests.get(
        DV_URL,
        params={"format": "json", "sites": site, "startDT": start, "endDT": end,
                "parameterCd": "00060,00065", "statCd": "00003"},
        timeout=timeout,
    )
    r.raise_for_status()
    series = r.json()["value"]["timeSeries"]

    frames = []
    for s in series:
        code = s["variable"]["variableCode"][0]["value"]
        col = {"00060": "discharge_cfs", "00065": "gage_height_ft"}.get(code)
        if col is None:
            continue
        recs = [
            {"date": v["dateTime"][:10], col: float(v["value"])}
            for v in s["values"][0]["value"]
            if v["value"] not in MISSING
        ]
        if recs:
            frames.append(pd.DataFrame(recs).drop_duplicates("date"))

    if not frames:
        return pd.DataFrame(columns=["date", "discharge_cfs", "gage_height_ft"])

    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="date", how="outer")
    return out


def fetch(site: str, start: str, end: str, *, refresh: bool = False,
          timeout: int = 90) -> pd.DataFrame:
    os.makedirs(RAW_DIR, exist_ok=True)
    cache = os.path.join(RAW_DIR, f"usgs_{site}.csv")
    if os.path.exists(cache) and not refresh:
        print(f"[usgs] cache hit -> {cache}")
        return pd.read_csv(cache)

    start_year = int(start[:4])
    end_year = int(end[:4])
    frames = []
    for y0 in range(start_year, end_year + 1, CHUNK_YEARS):
        y1 = min(y0 + CHUNK_YEARS - 1, end_year)
        w_start = max(start, f"{y0}-01-01")
        w_end = min(end, f"{y1}-12-31")
        print(f"[usgs] fetching {site} {w_start} .. {w_end}")
        frames.append(_fetch_window(site, w_start, w_end, timeout))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        raise SystemExit(f"[usgs] no data returned for site {site}")
    df = (df.drop_duplicates("date").sort_values("date").reset_index(drop=True))
    for col in ("discharge_cfs", "gage_height_ft"):
        if col not in df.columns:
            df[col] = float("nan")
    df = df[["date", "discharge_cfs", "gage_height_ft"]]
    df.to_csv(cache, index=False)
    print(f"[usgs] wrote {len(df)} rows -> {cache}")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default=DEFAULT_SITE)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    meta = site_metadata(args.site)
    print(f"[usgs] site {meta['site_no']}: {meta['station_nm']}")
    print(f"[usgs] lat/lon {meta['latitude']}, {meta['longitude']}")

    df = fetch(args.site, args.start, args.end, refresh=args.refresh)
    gh = df["gage_height_ft"].dropna()
    print(f"[usgs] rows={len(df)}  "
          f"gage height: n={len(gh)} min={gh.min():.2f} max={gh.max():.2f} ft")
    if gh.empty:
        print("[usgs] WARNING: this site publishes no daily gage height; "
              "the stage-based flood label cannot be built from it.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
