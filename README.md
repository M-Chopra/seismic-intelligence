# 🌋 SEISMIC INTELLIGENCE

> Real-time earthquake analysis · Multi-model ML prediction · Risk mapping · Aftershock forecasting

A production-grade ML project for seismic intelligence — built as a full multi-module Python project with a Streamlit dashboard, not a single script.

---

## Project Structure

```
earthquake-predictor/
│
├── app/
│   └── dashboard.py          # Streamlit dashboard (6-tab UI)
│
├── api/
│   └── usgs_fetch.py         # USGS GeoJSON + FDSNWS live/historical fetch
│
├── data/
│   └── preprocess.py         # Cleaning, 43-feature engineering, scaling
│
├── models/
│   ├── train.py              # Ridge, Lasso, RF, ExtraTrees, XGBoost, LightGBM
│   ├── lstm_model.py         # BiLSTM + Multi-Head Attention (TensorFlow)
│   └── ensemble.py           # Stacking meta-learner ensemble
│
├── analytics/
│   ├── risk_zones.py         # Grid-based risk scoring, KDE heatmap
│   └── aftershock.py         # Omori-Utsu law, Bath's law, Poisson probability
│
├── visuals/
│   └── map_renderer.py       # All Plotly figures (maps, charts, timelines)
│
├── utils/
│   └── helpers.py            # Stats, caching, magnitude helpers
│
├── notebooks/
│   └── eda.ipynb             # Exploratory analysis notebook
│
├── main.py                   # Pipeline smoke-test entry point
├── requirements.txt
└── runtime.txt
```

---

## Features

### 🌍 Global Map
- Interactive Scattergeo map with magnitude-sized markers coloured by depth
- Top-8 largest events sidebar panel
- Live severity alerts for M6+ events

### 📊 Analytics
- Magnitude timeline with rolling maximum
- **Gutenberg-Richter frequency-magnitude** distribution with fitted b-value
- Depth distribution by class (shallow / intermediate / deep)
- Magnitude vs depth scatter

### 🤖 ML Models
Trains **7 models** in parallel with temporal train/test split (no data leakage):

| Model | Type |
|-------|------|
| Ridge Regression | Linear |
| Lasso Regression | Linear (sparse) |
| ElasticNet | Regularised linear |
| Random Forest | Ensemble |
| Extra Trees | Ensemble |
| Gradient Boosting | Boosting |
| XGBoost | Boosting |
| LightGBM | Boosting |
| **Stacked Ensemble** | Meta-learner (RidgeCV) |
| **BiLSTM + Attention** | Deep learning (optional) |

**43 engineered features** including:
- Temporal (hour/month cyclical encoding)
- Spatial (unit-sphere XYZ projection)
- Energy / seismic moment (log-scaled)
- Seismicity rate (rolling 7-day spatial window)
- Gutenberg-Richter b-value estimate
- Distance to plate boundary proxy
- Depth classification

### ⚠️ Risk Zones
- **KDE-weighted risk grid** (energy-weighted, Gaussian-smoothed)
- Hotspot detection (top 5% percentile)
- 10° grid-cell risk summary table

### 🔁 Aftershock Forecasting
- **Omori-Utsu decay law**: λ(t) = K / (t + c)^p
- **Bath's law**: largest aftershock ≈ mainshock − 1.2
- **Wells-Coppersmith** rupture zone radius
- Poisson probability of M≥5 / M≥6 in next 24 hours
- Interactive 30-day daily forecast chart

### 📋 Recent Events
- Live scrollable event feed with magnitude colour coding
- Energy equivalent (tons of TNT), depth, coordinates
- Tsunami flag detection

---

## Quick Start

```bash
# 1. Clone / navigate to project
cd earthquake-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Enable LSTM — uncomment tensorflow in requirements.txt
# pip install tensorflow>=2.15.0

# 4. Run pipeline smoke-test
python main.py

# 5. Launch dashboard
streamlit run app/dashboard.py
```

Dashboard opens at **http://localhost:8501**

---

## Data Sources

| Source | Description |
|--------|-------------|
| [USGS GeoJSON Feeds](https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php) | Real-time M2.5+ global feed |
| [USGS FDSNWS API](https://earthquake.usgs.gov/fdsnws/event/1/) | Historical query with custom date/mag range |
| Simulated fallback | Realistic synthetic data when USGS unreachable |

---

## Science References

- **Gutenberg & Richter (1944)** — Frequency-magnitude distribution
- **Omori (1894), Utsu (1961)** — Aftershock decay law
- **Bath (1965)** — Largest aftershock scaling
- **Wells & Coppersmith (1994)** — Rupture zone dimensions from magnitude
- **Reasenberg & Jones (1989)** — Short-term aftershock probabilities
- **Ogata (1988)** — ETAS (Epidemic-Type Aftershock Sequence) model

---

## Dashboard Theme

**Brutalist dark terminal aesthetic** — Space Mono monospace + Barlow Condensed display, deep black backgrounds (#080808), orange (#f97316) accent, seismic red (#e11d48) for alerts.
