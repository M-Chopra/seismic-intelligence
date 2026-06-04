"""
USGS Earthquake API Fetcher
Pulls real-time and historical earthquake data from USGS GeoJSON feeds.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import time

logger = logging.getLogger(__name__)

USGS_BASE = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USGS_FEED_BASE = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary"

FEEDS = {
    "significant_month": f"{USGS_FEED_BASE}/significant_month.geojson",
    "all_week": f"{USGS_FEED_BASE}/all_week.geojson",
    "2.5_month": f"{USGS_FEED_BASE}/2.5_month.geojson",
    "4.5_week": f"{USGS_FEED_BASE}/4.5_week.geojson",
}


def fetch_live_feed(feed_key: str = "2.5_month") -> pd.DataFrame:
    """Fetch from USGS real-time GeoJSON feeds."""
    url = FEEDS.get(feed_key, FEEDS["2.5_month"])
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return _parse_geojson(data)
    except Exception as e:
        logger.warning(f"Live USGS feed fetch failed: {e}. Using simulated data.")
        return _generate_simulated_data(500)


def fetch_historical(
    start_date: str = None,
    end_date: str = None,
    min_magnitude: float = 2.5,
    max_results: int = 5000,
) -> pd.DataFrame:
    """
    Fetch historical earthquake data via USGS FDSNWS API.
    start_date / end_date: 'YYYY-MM-DD' strings
    """
    if end_date is None:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")

    params = {
        "format": "geojson",
        "starttime": start_date,
        "endtime": end_date,
        "minmagnitude": min_magnitude,
        "limit": max_results,
        "orderby": "time",
    }
    try:
        resp = requests.get(USGS_BASE, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        df = _parse_geojson(data)
        logger.info(f"Fetched {len(df)} historical earthquakes from USGS.")
        return df
    except Exception as e:
        logger.warning(f"Historical USGS fetch failed: {e}. Using simulated data.")
        return _generate_simulated_data(max_results)


def _parse_geojson(geojson: dict) -> pd.DataFrame:
    """Parse USGS GeoJSON FeatureCollection into a DataFrame."""
    records = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        coords = feature.get("geometry", {}).get("coordinates", [None, None, None])
        records.append(
            {
                "id": feature.get("id", ""),
                "time": pd.to_datetime(props.get("time", 0), unit="ms", utc=True),
                "latitude": coords[1],
                "longitude": coords[0],
                "depth_km": coords[2],
                "magnitude": props.get("mag", np.nan),
                "mag_type": props.get("magType", ""),
                "place": props.get("place", "Unknown"),
                "status": props.get("status", ""),
                "tsunami": int(props.get("tsunami", 0)),
                "sig": props.get("sig", 0),
                "nst": props.get("nst", 0),
                "dmin": props.get("dmin", np.nan),
                "rms": props.get("rms", np.nan),
                "gap": props.get("gap", np.nan),
                "net": props.get("net", ""),
                "type": props.get("type", "earthquake"),
                "url": props.get("url", ""),
            }
        )
    df = pd.DataFrame(records)
    if df.empty:
        return _generate_simulated_data(100)
    df = df[df["type"] == "earthquake"].copy()
    df.dropna(subset=["latitude", "longitude", "magnitude"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _generate_simulated_data(n: int = 1000) -> pd.DataFrame:
    """
    Generate realistic simulated earthquake data for offline/fallback use.
    Uses known seismic belt distributions.
    """
    rng = np.random.default_rng(42)

    # Seismic hotspot clusters: (lat_center, lon_center, lat_std, lon_std, weight)
    clusters = [
        (35.0, 139.0, 3.0, 4.0, 0.18),   # Japan
        (37.0, -118.0, 2.0, 3.0, 0.12),  # California
        (-10.0, -75.0, 5.0, 4.0, 0.10),  # South America
        (38.0, 22.0, 2.0, 3.0, 0.08),    # Greece
        (28.0, 84.0, 3.0, 4.0, 0.08),    # Nepal/Himalayas
        (1.0, 126.0, 4.0, 5.0, 0.10),    # Indonesia
        (19.0, -155.0, 2.0, 2.0, 0.07),  # Hawaii
        (-38.0, -72.0, 3.0, 3.0, 0.07),  # Chile
        (40.0, 44.0, 3.0, 4.0, 0.06),    # Caucasus
        (36.0, 36.0, 2.0, 3.0, 0.05),    # Turkey
        (0.0, 0.0, 60.0, 120.0, 0.09),   # Global scatter
    ]
    weights = np.array([c[4] for c in clusters])
    weights /= weights.sum()

    chosen = rng.choice(len(clusters), size=n, p=weights)
    lats, lons = [], []
    for idx in chosen:
        c = clusters[idx]
        lats.append(np.clip(rng.normal(c[0], c[2]), -90, 90))
        lons.append(np.clip(rng.normal(c[1], c[3]), -180, 180))

    # Magnitude: Gutenberg-Richter distribution (log-linear)
    magnitudes = np.clip(rng.exponential(1.2, n) + 2.0, 2.0, 9.2)

    depths = np.abs(rng.lognormal(2.5, 1.0, n))
    depths = np.clip(depths, 0.5, 700)

    times = [
        datetime.utcnow() - timedelta(seconds=int(rng.uniform(0, 365 * 24 * 3600)))
        for _ in range(n)
    ]
    times_sorted = sorted(times)

    places = [f"Simulated Region ({la:.1f}N, {lo:.1f}E)" for la, lo in zip(lats, lons)]

    df = pd.DataFrame(
        {
            "id": [f"sim_{i:05d}" for i in range(n)],
            "time": pd.to_datetime(times_sorted, utc=True),
            "latitude": lats,
            "longitude": lons,
            "depth_km": depths,
            "magnitude": magnitudes,
            "mag_type": rng.choice(["ml", "mb", "mw", "ms"], n),
            "place": places,
            "status": ["reviewed"] * n,
            "tsunami": (magnitudes > 7.5).astype(int),
            "sig": (magnitudes * 100).astype(int),
            "nst": rng.integers(5, 120, n),
            "dmin": rng.uniform(0.01, 5.0, n),
            "rms": rng.uniform(0.01, 1.5, n),
            "gap": rng.uniform(10, 300, n),
            "net": rng.choice(["us", "ci", "nc", "ak", "uu"], n),
            "type": ["earthquake"] * n,
            "url": [""] * n,
        }
    )
    return df


def get_recent_significant(min_mag: float = 6.0, days: int = 30) -> pd.DataFrame:
    """Convenience: get recent significant earthquakes."""
    end = datetime.utcnow().strftime("%Y-%m-%d")
    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    return fetch_historical(start, end, min_magnitude=min_mag, max_results=200)
