"""
Shared Utilities and Helpers
"""

import numpy as np
import pandas as pd
import hashlib
import logging
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


# ─── LOGGING SETUP ────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(levelname)s | %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


# ─── MAGNITUDE HELPERS ────────────────────────────────────────────────────────

MMI_LABELS = {
    1: "Not felt", 2: "Weak", 3: "Weak", 4: "Light",
    5: "Moderate", 6: "Strong", 7: "Very Strong",
    8: "Severe", 9: "Violent", 10: "Extreme",
    11: "Extreme", 12: "Total Destruction",
}

RICHTER_LABELS = [
    (2.0, "Micro"),
    (3.0, "Minor"),
    (4.0, "Light"),
    (5.0, "Moderate"),
    (6.0, "Strong"),
    (7.0, "Major"),
    (8.0, "Great"),
    (10.0, "Catastrophic"),
]


def magnitude_label(mag: float) -> str:
    for threshold, label in RICHTER_LABELS:
        if mag < threshold:
            return label
    return "Catastrophic"


def magnitude_color(mag: float) -> str:
    if mag < 3:
        return "#4ade80"
    elif mag < 4:
        return "#a3e635"
    elif mag < 5:
        return "#facc15"
    elif mag < 6:
        return "#fb923c"
    elif mag < 7:
        return "#f87171"
    elif mag < 8:
        return "#dc2626"
    else:
        return "#7c3aed"


def energy_tnta(magnitude: float) -> float:
    """Convert magnitude to energy in tons of TNT equivalent."""
    joules = 10 ** (1.5 * magnitude + 4.8)
    return joules / 4.184e9  # 1 ton TNT = 4.184e9 J


def format_energy(magnitude: float) -> str:
    tnt = energy_tnta(magnitude)
    if tnt < 1000:
        return f"{tnt:.0f} tons TNT"
    elif tnt < 1e6:
        return f"{tnt/1000:.1f} kilotons TNT"
    elif tnt < 1e9:
        return f"{tnt/1e6:.2f} megatons TNT"
    else:
        return f"{tnt/1e9:.2f} gigatons TNT"


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great circle distance in km between two points."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ─── STATISTICS HELPERS ──────────────────────────────────────────────────────

def compute_global_stats(df: pd.DataFrame) -> dict:
    """Quick summary statistics for the dashboard header."""
    if df.empty:
        return {}
    recent_24h = df[df["time"] >= (pd.Timestamp.utcnow() - pd.Timedelta(hours=24))]
    recent_7d = df[df["time"] >= (pd.Timestamp.utcnow() - pd.Timedelta(days=7))]
    return {
        "total_events": len(df),
        "events_24h": len(recent_24h),
        "events_7d": len(recent_7d),
        "max_magnitude": round(df["magnitude"].max(), 1),
        "mean_magnitude": round(df["magnitude"].mean(), 2),
        "mean_depth_km": round(df["depth_km"].mean(), 1),
        "tsunami_events": int(df["tsunami"].sum()) if "tsunami" in df.columns else 0,
        "major_events": int((df["magnitude"] >= 6.0).sum()),
        "great_events": int((df["magnitude"] >= 7.0).sum()),
        "latest_event": df.sort_values("time").iloc[-1]["place"] if len(df) > 0 else "N/A",
        "latest_time": df["time"].max().strftime("%Y-%m-%d %H:%M UTC") if hasattr(df["time"].max(), "strftime") else "N/A",
    }


def top_recent_events(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return top-n most recent significant events."""
    return (
        df.sort_values("time", ascending=False)
        .head(n)[["time", "place", "magnitude", "depth_km", "latitude", "longitude"]]
        .reset_index(drop=True)
    )


# ─── CACHING ─────────────────────────────────────────────────────────────────

def cache_path(key: str, base_dir: str = "/tmp/eq_cache") -> Path:
    os.makedirs(base_dir, exist_ok=True)
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return Path(base_dir) / f"{h}.parquet"


def save_cache(df: pd.DataFrame, key: str) -> None:
    try:
        df.to_parquet(cache_path(key), index=False)
    except Exception as e:
        logger.debug(f"Cache save failed: {e}")


def load_cache(key: str, max_age_hours: int = 6) -> pd.DataFrame | None:
    p = cache_path(key)
    if not p.exists():
        return None
    age_h = (datetime.now().timestamp() - p.stat().st_mtime) / 3600
    if age_h > max_age_hours:
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None
