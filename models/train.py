"""
Classical ML Model Training
Trains Random Forest, XGBoost, GradientBoosting, Ridge, SVR and returns
ranked results.
"""

import numpy as np
import pandas as pd
import logging
import time
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

logger = logging.getLogger(__name__)

_XGB_AVAILABLE = False
try:
    from xgboost import XGBRegressor
    _XGB_AVAILABLE = True
except ImportError:
    pass

_LGB_AVAILABLE = False
try:
    from lightgbm import LGBMRegressor
    _LGB_AVAILABLE = True
except ImportError:
    pass


# ─── MODEL REGISTRY ──────────────────────────────────────────────────────────

def get_model_registry() -> dict:
    """Returns all classical models to train."""
    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01, max_iter=2000),
        "ElasticNet": ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=2000),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            n_jobs=-1, random_state=42
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=200, max_depth=12, n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=42
        ),
    }
    if _XGB_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=42, verbosity=0,
        )
    if _LGB_AVAILABLE:
        models["LightGBM"] = LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=63,
            n_jobs=-1, random_state=42, verbose=-1,
        )
    return models


# ─── TRAINING ────────────────────────────────────────────────────────────────

def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_dir: str = None,
) -> list[dict]:
    """
    Train all classical models. Returns sorted list of result dicts.
    """
    registry = get_model_registry()
    results = []

    for name, model in registry.items():
        logger.info(f"  ▶ Training {name} ...")
        t0 = time.time()
        try:
            model.fit(X_train, y_train.ravel())
            preds = model.predict(X_test)
            elapsed = time.time() - t0

            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)

            logger.info(f"    MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}  ({elapsed:.1f}s)")

            result = {
                "model": model,
                "name": name,
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "r2": round(r2, 4),
                "train_time_s": round(elapsed, 2),
                "predictions": preds,
            }

            # Feature importance (if available)
            if hasattr(model, "feature_importances_"):
                result["feature_importances"] = model.feature_importances_

            results.append(result)

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                joblib.dump(model, os.path.join(save_dir, f"{name.replace(' ', '_')}.pkl"))

        except Exception as e:
            logger.error(f"    ✗ {name} failed: {e}")

    # Sort by MAE ascending
    results.sort(key=lambda r: r["mae"])
    return results


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate a single trained model."""
    preds = model.predict(X_test)
    return {
        "mae": round(mean_absolute_error(y_test, preds), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_test, preds)), 4),
        "r2": round(r2_score(y_test, preds), 4),
        "predictions": preds,
    }


def get_feature_importance_df(result: dict, feature_names: list) -> pd.DataFrame:
    """Extract feature importance as sorted DataFrame."""
    fi = result.get("feature_importances_")
    if fi is None:
        return pd.DataFrame()
    df = pd.DataFrame({"feature": feature_names, "importance": fi})
    df.sort_values("importance", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
