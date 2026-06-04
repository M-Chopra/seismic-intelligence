"""
Seismic Risk Zone Analytics
Computes grid-based risk scores, hotspot detection, and historical exposure.
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
import logging

logger = logging.getLogger(__name__)


# ─── RISK SCORING ─────────────────────────────────────────────────────────────

def compute_risk_grid(
    df: pd.DataFrame,
    resolution: int = 60,      # grid cells per axis
    energy_weight: bool = True,
) -> dict:
    """
    Build a 2D global risk grid using KDE on earthquake locations,
    weighted by energy release.

    Returns dict with keys: lats, lons, risk (2D array), max_risk, hotspots.
    """
    lat_bins = np.linspace(-90, 90, resolution + 1)
    lon_bins = np.linspace(-180, 180, resolution * 2 + 1)
    lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    lon_centers = 0.5 * (lon_bins[:-1] + lon_bins[1:])

    lats = df["latitude"].values
    lons = df["longitude"].values

    if energy_weight and "log_energy" in df.columns:
        weights = df["log_energy"].fillna(df["magnitude"] * 1.5).values
        weights = np.clip(weights, 0, None)
        weights = weights / weights.max()
    else:
        weights = np.ones(len(df))

    # 2D histogram weighted
    grid, _, _ = np.histogram2d(
        lats, lons,
        bins=[lat_bins, lon_bins],
        weights=weights,
        density=False,
    )

    # Smooth the grid
    grid = gaussian_filter(grid.astype(float), sigma=1.5)

    # Normalize 0–100
    if grid.max() > 0:
        risk = (grid / grid.max()) * 100
    else:
        risk = grid

    # Identify hotspot cells (top 5%)
    threshold = np.percentile(risk[risk > 0], 95) if (risk > 0).any() else 50
    hotspot_mask = risk > threshold
    hotspot_lats, hotspot_lons, hotspot_scores = [], [], []
    rows, cols = np.where(hotspot_mask)
    for r, c in zip(rows, cols):
        hotspot_lats.append(lat_centers[r])
        hotspot_lons.append(lon_centers[c])
        hotspot_scores.append(risk[r, c])

    hotspots = pd.DataFrame({
        "latitude": hotspot_lats,
        "longitude": hotspot_lons,
        "risk_score": hotspot_scores,
    }).sort_values("risk_score", ascending=False).head(20)

    return {
        "lat_centers": lat_centers,
        "lon_centers": lon_centers,
        "risk_grid": risk,
        "max_risk": float(risk.max()),
        "hotspots": hotspots,
        "threshold": float(threshold),
    }


def zone_risk_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise seismic risk by 10×10 degree grid cell.
    Returns DataFrame with zone stats.
    """
    df = df.copy()
    df["lat_zone"] = (df["latitude"] // 10) * 10
    df["lon_zone"] = (df["longitude"] // 10) * 10

    summary = df.groupby(["lat_zone", "lon_zone"]).agg(
        event_count=("magnitude", "count"),
        max_mag=("magnitude", "max"),
        mean_mag=("magnitude", "mean"),
        mean_depth=("depth_km", "mean"),
        tsunami_events=("tsunami", "sum"),
    ).reset_index()

    # Composite risk score
    summary["risk_score"] = (
        np.log1p(summary["event_count"]) * 0.4 +
        summary["max_mag"] * 0.35 +
        summary["mean_mag"] * 0.15 +
        (1 / (summary["mean_depth"] + 1)) * 10 * 0.1
    )
    summary["risk_score"] = (
        (summary["risk_score"] - summary["risk_score"].min()) /
        (summary["risk_score"].max() - summary["risk_score"].min() + 1e-9) * 100
    )
    return summary.sort_values("risk_score", ascending=False)


def classify_severity(magnitude: float) -> tuple[str, str]:
    """Return (label, css_color) for a magnitude."""
    if magnitude < 3.0:
        return "Minor", "#4ade80"
    elif magnitude < 4.0:
        return "Light", "#a3e635"
    elif magnitude < 5.0:
        return "Moderate", "#facc15"
    elif magnitude < 6.0:
        return "Strong", "#fb923c"
    elif magnitude < 7.0:
        return "Major", "#f87171"
    elif magnitude < 8.0:
        return "Great", "#e11d48"
    else:
        return "Catastrophic", "#7c3aed"


def magnitude_to_mmi(magnitude: float, depth_km: float = 10) -> int:
    """
    Estimate Modified Mercalli Intensity at epicenter.
    Simple empirical formula.
    """
    # Attenuation with depth
    r = depth_km
    mmi = 2.085 * magnitude - 0.044 * np.log(r + 1) - 0.002 * r + 1.5
    return int(np.clip(mmi, 1, 12))
