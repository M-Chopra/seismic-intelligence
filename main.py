"""
main.py — Entry point for Seismic Intelligence project.
Run:  streamlit run app/dashboard.py
Or:   python main.py  (runs a quick data pipeline test)
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(__file__))
logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pipeline_test():
    """Quick smoke test of the full data → features → ML pipeline."""
    from api.usgs_fetch import fetch_live_feed
    from data.preprocess import clean_raw, engineer_features, prepare_ml_matrix, train_test_split_temporal, FEATURE_COLS
    from models.train import train_all_models
    from models.ensemble import EarthquakeEnsemble
    from analytics.risk_zones import compute_risk_grid, zone_risk_summary
    from analytics.aftershock import full_aftershock_report
    from utils.helpers import compute_global_stats

    logger.info("=" * 55)
    logger.info("  SEISMIC INTELLIGENCE — PIPELINE TEST")
    logger.info("=" * 55)

    # 1. Data
    logger.info("\n[1/5] Fetching seismic data...")
    raw = fetch_live_feed("2.5_month")
    logger.info(f"      Raw records: {len(raw)}")

    # 2. Preprocess
    logger.info("\n[2/5] Preprocessing...")
    cleaned = clean_raw(raw)
    featured = engineer_features(cleaned)
    logger.info(f"      Features shape: {featured.shape}")

    # 3. Stats
    logger.info("\n[3/5] Global statistics...")
    stats = compute_global_stats(featured)
    for k, v in stats.items():
        logger.info(f"      {k}: {v}")

    # 4. ML
    logger.info("\n[4/5] Training ML models...")
    feat_cols = [c for c in FEATURE_COLS if c in featured.columns and c != "magnitude"]
    train_df, test_df = train_test_split_temporal(featured, test_ratio=0.2)
    X_tr, y_tr, scaler, fnames = prepare_ml_matrix(train_df, feat_cols)
    X_te, y_te, _, _ = prepare_ml_matrix(test_df, feat_cols, scale=False)
    if scaler:
        X_te = scaler.transform(X_te)
    results = train_all_models(X_tr, y_tr, X_te, y_te)
    ens = EarthquakeEnsemble(results)
    try:
        ens.fit_meta(X_te, y_te)
    except Exception:
        pass
    ens_metrics = ens.evaluate(X_te, y_te)
    logger.info(f"      Best model: {results[0]['name']}  MAE={results[0]['mae']}")
    logger.info(f"      Ensemble:   MAE={ens_metrics['mae']}  R²={ens_metrics['r2']}")

    # 5. Aftershock
    logger.info("\n[5/5] Aftershock analysis (M7.5 mainshock)...")
    report = full_aftershock_report(7.5, mainshock_depth=15)
    logger.info(f"      Largest aftershock: M{report['largest_aftershock']}")
    logger.info(f"      Expected M2+ (30d): {report['expected_m2plus_30days']:.0f}")
    logger.info(f"      P(M≥5 / 24h): {report['prob_m5_next_24h']*100:.1f}%")
    logger.info(f"      Aftershock zone: {report['aftershock_zone_radius_km']} km")

    logger.info("\n✅  Pipeline test complete.\n")
    logger.info("Run dashboard:  streamlit run app/dashboard.py")


if __name__ == "__main__":
    run_pipeline_test()
