"""
Ensemble Model
Combines predictions from all trained models using stacking / weighted averaging.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)


class EarthquakeEnsemble:
    """
    Stacking ensemble: uses a meta-learner (RidgeCV) on top of base model predictions.
    Fallback: inverse-MAE weighted average when insufficient data for stacking.
    """

    def __init__(self, results: list[dict]):
        self.results = results  # list of dicts with 'model', 'name', 'mae', 'predictions'
        self.meta_model = None
        self.weights = None
        self._fitted = False

    def fit_meta(self, X_val: np.ndarray, y_val: np.ndarray):
        """Fit the meta-learner on validation set predictions."""
        base_preds = np.column_stack(
            [r["model"].predict(X_val) for r in self.results if r.get("model") is not None]
        )
        self.meta_model = RidgeCV(alphas=[0.1, 1.0, 10.0])
        self.meta_model.fit(base_preds, y_val.ravel())
        self._fitted = True
        logger.info("Ensemble meta-learner fitted.")

        # Also compute inverse-MAE weights
        maes = np.array([r["mae"] for r in self.results if r.get("model") is not None])
        inv_mae = 1.0 / (maes + 1e-9)
        self.weights = inv_mae / inv_mae.sum()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using ensemble."""
        valid = [r for r in self.results if r.get("model") is not None]
        if not valid:
            return np.zeros(len(X))

        base_preds = np.column_stack([r["model"].predict(X) for r in valid])

        if self._fitted and self.meta_model is not None:
            return self.meta_model.predict(base_preds)
        else:
            # Weighted average fallback
            return base_preds @ self.weights

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        preds = self.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(np.mean((preds - y_test.ravel()) ** 2))
        r2 = r2_score(y_test, preds)
        return {
            "name": "Stacked Ensemble",
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "predictions": preds,
        }

    def model_contributions(self) -> pd.DataFrame:
        """Show how much each model contributes to the ensemble."""
        if self.weights is None:
            return pd.DataFrame()
        valid = [r for r in self.results if r.get("model") is not None]
        return pd.DataFrame({
            "model": [r["name"] for r in valid],
            "weight": self.weights,
            "mae": [r["mae"] for r in valid],
        }).sort_values("weight", ascending=False)


def simple_weighted_ensemble(results: list[dict], X_test: np.ndarray) -> np.ndarray:
    """Quick weighted average prediction — no fitting needed."""
    valid = [r for r in results if r.get("model") is not None]
    if not valid:
        return np.zeros(len(X_test))
    maes = np.array([r["mae"] for r in valid])
    weights = (1.0 / (maes + 1e-9))
    weights /= weights.sum()
    preds = np.column_stack([r["model"].predict(X_test) for r in valid])
    return preds @ weights
