"""
Data Preprocessing & Feature Engineering
Transforms raw USGS earthquake records into ML-ready feature matrices.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
import logging

logger = logging.getLogger(__name__)


# ─── CLEANING ────────────────────────────────────────────────────────────────

def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop duplicates, fix types, filter valid rows."""
    df = df.copy()
    df.drop_duplicates(subset=["id"], keep="first", inplace=True)

    # Ensure numeric
    for col in ["magnitude", "depth_km", "latitude", "longitude", "sig", "nst", "rms", "gap", "dmin"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sane physical bounds
    df = df[
        df["magnitude"].between(0, 10) &
        df["latitude"].between(-90, 90) &
        df["longitude"].between(-180, 180) &
        df["depth_km"].between(0, 800)
    ]

    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

    df.dropna(subset=["time", "latitude", "longitude", "magnitude"], inplace=True)
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(f"After cleaning: {len(df)} records.")
    return df


# ─── FEATURE ENGINEERING ─────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all derived features for ML models."""
    df = df.copy()

    # ── Temporal features
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month
    df["year"] = df["time"].dt.year
    df["day_of_year"] = df["time"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ── Energy proxy (Richter energy release is ~10^(1.5*M))
    df["energy_joules"] = 10 ** (1.5 * df["magnitude"] + 4.8)
    df["log_energy"] = np.log10(df["energy_joules"] + 1)

    # ── Depth classification
    df["depth_class"] = pd.cut(
        df["depth_km"],
        bins=[0, 70, 300, 800],
        labels=["shallow", "intermediate", "deep"],
    ).astype(str)
    df["is_shallow"] = (df["depth_km"] < 70).astype(int)

    # ── Seismic moment proxy
    df["seismic_moment"] = 10 ** (1.5 * df["magnitude"] + 9.1)
    df["log_moment"] = np.log10(df["seismic_moment"] + 1)

    # ── Spatial features (convert lat/lon to 3D unit sphere)
    lat_r = np.radians(df["latitude"])
    lon_r = np.radians(df["longitude"])
    df["x_sphere"] = np.cos(lat_r) * np.cos(lon_r)
    df["y_sphere"] = np.cos(lat_r) * np.sin(lon_r)
    df["z_sphere"] = np.sin(lat_r)

    # ── Quality metrics
    if "gap" in df.columns:
        df["gap_quality"] = 1 / (df["gap"].fillna(180) + 1)
    if "rms" in df.columns:
        df["rms_log"] = np.log1p(df["rms"].fillna(0))
    if "nst" in df.columns:
        df["nst_log"] = np.log1p(df["nst"].fillna(0))

    # ── Rolling seismicity rate (events per day in neighbourhood)
    df = _add_seismicity_rate(df)

    # ── b-value proxy (Gutenberg-Richter)
    df = _add_gutenberg_richter_features(df)

    # ── Distance to nearest plate boundary (rough proxy)
    df["dist_to_boundary"] = _approx_plate_boundary_dist(df["latitude"], df["longitude"])

    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


def _add_seismicity_rate(df: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    """Count number of earthquakes in same region within a rolling window."""
    df = df.copy()
    df_sorted = df.sort_values("time").reset_index(drop=True)
    times = df_sorted["time"].values.astype(np.int64)
    lats = df_sorted["latitude"].values
    lons = df_sorted["longitude"].values
    window_ns = window_days * 24 * 3600 * 1_000_000_000

    rates = np.zeros(len(df_sorted), dtype=float)
    for i in range(len(df_sorted)):
        mask_time = (times[i] - times) < window_ns
        mask_time &= times <= times[i]
        dist = np.sqrt((lats - lats[i]) ** 2 + (lons - lons[i]) ** 2)
        mask_space = dist < 3.0  # ~300 km radius
        rates[i] = float(np.sum(mask_time & mask_space))
    df_sorted["seismicity_rate_7d"] = rates
    return df_sorted


def _add_gutenberg_richter_features(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate b-value in rolling window (simplistic)."""
    df = df.copy()
    mags = df["magnitude"].values
    b_values = np.zeros(len(mags))
    for i in range(len(mags)):
        start = max(0, i - 100)
        subset = mags[start:i + 1]
        if len(subset) > 10:
            mean_m = np.mean(subset)
            mc = subset.min()
            b_est = np.log10(np.e) / (mean_m - mc + 1e-9)
            b_values[i] = np.clip(b_est, 0.3, 2.5)
        else:
            b_values[i] = 1.0
    df["b_value_est"] = b_values
    return df


def _approx_plate_boundary_dist(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """
    Very rough proxy for distance to plate boundary.
    Based on simplified Ring of Fire and other major boundaries.
    """
    boundaries = [
        # Ring of Fire segments (lat, lon)
        (52, 160), (40, 142), (35, 137), (25, 122), (10, 126),
        (-5, 130), (-15, 167), (-38, -72), (-20, -65), (0, -80),
        (10, -84), (18, -100), (28, -110), (38, -120), (48, -124),
        # Mediterranean / Alpide
        (37, 23), (39, 40), (35, 58), (30, 65), (28, 84), (27, 92),
        # Mid-Atlantic Ridge
        (65, -19), (52, -33), (0, -20), (-38, -13), (-55, -10),
    ]
    b_lats = np.array([b[0] for b in boundaries])
    b_lons = np.array([b[1] for b in boundaries])
    dists = []
    for la, lo in zip(lat.values, lon.values):
        d = np.sqrt((b_lats - la) ** 2 + (b_lons - lo) ** 2)
        dists.append(d.min())
    return pd.Series(dists, index=lat.index)


# ─── IMPUTATION & SCALING ─────────────────────────────────────────────────────

FEATURE_COLS = [
    "latitude", "longitude", "depth_km", "magnitude",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "log_energy", "log_moment", "is_shallow",
    "x_sphere", "y_sphere", "z_sphere",
    "b_value_est", "dist_to_boundary",
    "seismicity_rate_7d",
]

TARGET_COLS = ["magnitude"]


def prepare_ml_matrix(df: pd.DataFrame, feature_cols: list = None, scale: bool = True):
    """Return (X, y, scaler, feature_names) ready for sklearn/torch models."""
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLS if c in df.columns and c not in TARGET_COLS]

    X = df[feature_cols].copy()

    # Impute
    imputer = KNNImputer(n_neighbors=5)
    X_arr = imputer.fit_transform(X)

    # Scale
    scaler = None
    if scale:
        scaler = RobustScaler()
        X_arr = scaler.fit_transform(X_arr)

    y = df["magnitude"].values.reshape(-1, 1) if "magnitude" in df.columns else None

    return X_arr, y, scaler, feature_cols


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 30):
    """Create overlapping sequences for LSTM training."""
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i: i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


def train_test_split_temporal(df: pd.DataFrame, test_ratio: float = 0.2):
    """Split preserving temporal order (no data leakage)."""
    split_idx = int(len(df) * (1 - test_ratio))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
