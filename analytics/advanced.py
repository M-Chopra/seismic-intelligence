"""
Advanced Analytics Module
Seismic gap analysis, energy release trends, fault stress, cross-correlation,
tectonic province classification, and magnitude recurrence intervals.
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.signal import find_peaks
import logging

logger = logging.getLogger(__name__)


# ─── SEISMIC GAP ANALYSIS ─────────────────────────────────────────────────────

def compute_seismic_gaps(df: pd.DataFrame, min_mag: float = 4.0) -> pd.DataFrame:
    """
    Identify seismic gaps — regions along plate boundaries with historically
    low seismicity despite being in active tectonic zones.
    Uses 20° grid cells, flags cells with low recent activity but high historical.
    """
    df = df[df["magnitude"] >= min_mag].copy()
    df["lat_cell"] = (df["latitude"] // 20) * 20
    df["lon_cell"] = (df["longitude"] // 20) * 20

    now = pd.Timestamp.utcnow()
    recent_cutoff = now - pd.Timedelta(days=180)

    all_activity = df.groupby(["lat_cell", "lon_cell"]).agg(
        total_events=("magnitude", "count"),
        max_mag=("magnitude", "max"),
        mean_mag=("magnitude", "mean"),
    ).reset_index()

    recent = df[df["time"] >= recent_cutoff].groupby(["lat_cell", "lon_cell"]).agg(
        recent_events=("magnitude", "count"),
    ).reset_index()

    merged = all_activity.merge(recent, on=["lat_cell", "lon_cell"], how="left")
    merged["recent_events"] = merged["recent_events"].fillna(0)

    # Gap score: high historical activity, low recent activity
    merged["gap_score"] = (
        np.log1p(merged["total_events"]) * merged["max_mag"] /
        (np.log1p(merged["recent_events"]) + 1)
    )
    merged["gap_score"] = (merged["gap_score"] / merged["gap_score"].max() * 100).round(1)
    return merged.sort_values("gap_score", ascending=False).head(15)


# ─── CUMULATIVE ENERGY RELEASE ────────────────────────────────────────────────

def cumulative_energy_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative seismic energy release over time."""
    df_s = df.sort_values("time").copy()
    df_s["energy_joules"] = 10 ** (1.5 * df_s["magnitude"] + 4.8)
    df_s["cumulative_energy"] = df_s["energy_joules"].cumsum()
    df_s["cumulative_energy_petajoules"] = df_s["cumulative_energy"] / 1e15
    df_s["log_cumulative"] = np.log10(df_s["cumulative_energy"] + 1)

    # Detect energy spikes (major events)
    energy_arr = df_s["energy_joules"].values
    peaks, _ = find_peaks(energy_arr, height=np.percentile(energy_arr, 95))
    df_s["is_major_spike"] = False
    df_s.iloc[peaks, df_s.columns.get_loc("is_major_spike")] = True

    return df_s[["time", "magnitude", "place", "energy_joules",
                 "cumulative_energy_petajoules", "log_cumulative", "is_major_spike"]]


# ─── MAGNITUDE RECURRENCE INTERVAL ───────────────────────────────────────────

def recurrence_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average recurrence interval (days) for different magnitude thresholds.
    Uses Poisson process assumption.
    """
    total_days = max(
        (df["time"].max() - df["time"].min()).days, 1
    )
    thresholds = [3.0, 4.0, 5.0, 6.0, 6.5, 7.0, 7.5, 8.0]
    rows = []
    for m in thresholds:
        count = int((df["magnitude"] >= m).sum())
        if count > 0:
            rate_per_day = count / total_days
            avg_interval = total_days / count
            annual_rate = rate_per_day * 365.25
        else:
            avg_interval = None
            annual_rate = 0.0
        rows.append({
            "Min Magnitude": f"M{m}+",
            "Event Count": count,
            "Avg Recurrence (days)": round(avg_interval, 1) if avg_interval else "—",
            "Annual Rate": round(annual_rate, 2),
            "Prob in 1 Year (%)": round((1 - np.exp(-annual_rate)) * 100, 1) if annual_rate > 0 else 0.0,
        })
    return pd.DataFrame(rows)


# ─── DEPTH vs TECTONIC TYPE ───────────────────────────────────────────────────

def classify_tectonic_type(depth_km: float, magnitude: float) -> dict:
    """
    Classify earthquake tectonic mechanism based on depth and magnitude.
    Returns type label, description, and color.
    """
    if depth_km < 20:
        if magnitude >= 6.0:
            return {"type": "Crustal Strike-Slip / Thrust", "color": "#ef4444",
                    "desc": "Shallow, high-damage potential. Common along transform faults."}
        return {"type": "Shallow Crustal", "color": "#f97316",
                "desc": "Uppermost crust. High MMI at surface despite moderate magnitude."}
    elif depth_km < 70:
        return {"type": "Upper Mantle / Subduction Interface", "color": "#f59e0b",
                "desc": "Transition zone. Includes megathrust interface events."}
    elif depth_km < 300:
        return {"type": "Intermediate Subduction", "color": "#0ea5e9",
                "desc": "Within subducting slab. Less surface damage but wide felt area."}
    else:
        return {"type": "Deep Slab (Wadati-Benioff)", "color": "#a855f7",
                "desc": "Deep within subducting slab. Can be felt over vast areas."}


# ─── CROSS-CORRELATION: MAG vs DEPTH ─────────────────────────────────────────

def mag_depth_correlation(df: pd.DataFrame) -> dict:
    """Pearson correlation between magnitude and depth by tectonic zone."""
    results = {}
    zones = {
        "Shallow (<70km)": df[df["depth_km"] < 70],
        "Intermediate (70–300km)": df[(df["depth_km"] >= 70) & (df["depth_km"] < 300)],
        "Deep (>300km)": df[df["depth_km"] >= 300],
    }
    for zone, subset in zones.items():
        if len(subset) > 10:
            r, p = pearsonr(subset["depth_km"], subset["magnitude"])
            results[zone] = {"r": round(r, 3), "p": round(p, 4),
                             "n": len(subset), "significant": p < 0.05}
    return results


# ─── HOURLY SEISMICITY PATTERN ────────────────────────────────────────────────

def hourly_seismicity(df: pd.DataFrame) -> pd.DataFrame:
    """Count events and mean magnitude by UTC hour."""
    df = df.copy()
    df["hour"] = df["time"].dt.hour
    result = df.groupby("hour").agg(
        event_count=("magnitude", "count"),
        mean_mag=("magnitude", "mean"),
        max_mag=("magnitude", "max"),
    ).reset_index()
    result["mean_mag"] = result["mean_mag"].round(2)
    result["max_mag"] = result["max_mag"].round(1)
    return result


# ─── REGIONAL SEISMICITY TREND ────────────────────────────────────────────────

def weekly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly event counts and max magnitude — detects upticks."""
    df = df.copy()
    df["week"] = df["time"].dt.to_period("W").dt.start_time
    weekly = df.groupby("week").agg(
        events=("magnitude", "count"),
        max_mag=("magnitude", "max"),
        mean_mag=("magnitude", "mean"),
        energy_sum=("log_energy", "sum") if "log_energy" in df.columns else ("magnitude", "sum"),
    ).reset_index()
    weekly["trend"] = weekly["events"].diff().fillna(0)
    weekly["trend_label"] = weekly["trend"].apply(
        lambda x: "↑ Uptick" if x > 5 else ("↓ Quiet" if x < -5 else "→ Stable")
    )
    return weekly.tail(12)  # last 12 weeks
