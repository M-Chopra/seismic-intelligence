"""
╔═══════════════════════════════════════════════════════════╗
║  SEISMIC INTELLIGENCE DASHBOARD  v2.0                     ║
║  +7 new features · Animations · Advanced analytics        ║
╚═══════════════════════════════════════════════════════════╝
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from api.usgs_fetch import fetch_live_feed, fetch_historical
from data.preprocess import clean_raw, engineer_features, prepare_ml_matrix, train_test_split_temporal, FEATURE_COLS
from models.train import train_all_models
from models.ensemble import EarthquakeEnsemble
from models.lstm_model import _TF_AVAILABLE
from analytics.risk_zones import compute_risk_grid, zone_risk_summary
from analytics.aftershock import full_aftershock_report, expected_aftershocks
from analytics.advanced import (
    compute_seismic_gaps, cumulative_energy_timeline,
    recurrence_intervals, classify_tectonic_type,
    mag_depth_correlation, hourly_seismicity, weekly_trend,
)
from utils.helpers import (
    setup_logging, compute_global_stats,
    magnitude_color, magnitude_label, format_energy,
    load_cache, save_cache,
)
from visuals.map_renderer import (
    global_earthquake_map, magnitude_time_series, magnitude_histogram,
    depth_distribution, model_performance_chart, aftershock_forecast_chart,
    magnitude_depth_scatter, animated_magnitude_bubble, cumulative_energy_chart,
    hourly_heatmap, weekly_trend_chart, seismic_gap_map, recurrence_bar_chart,
)

setup_logging()

st.set_page_config(page_title="SEISMIC INTELLIGENCE", page_icon="🌋", layout="wide")

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Barlow+Condensed:wght@300;500;700;900&display=swap');

:root{
    --bg:#080808;--surface:#111;--surface2:#181818;
    --border:#2a2a2a;--text:#d4d4d4;--muted:#666;
    --orange:#f97316;--blue:#0ea5e9;--red:#e11d48;--purple:#a855f7;
}
.stApp{background:var(--bg)!important;}
.main .block-container{padding:1.5rem 2rem 3rem;max-width:1400px;}
html,body,[class*="st-"]{font-family:'Barlow Condensed',sans-serif!important;color:var(--text);}
#MainMenu,footer,header{visibility:hidden;}
.stDeployButton{display:none;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border);}
[data-testid="stMetric"]{background:var(--surface2);border:1px solid var(--border);border-radius:4px;padding:1rem;transition:border-color .3s;}
[data-testid="stMetric"]:hover{border-color:var(--orange);}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:.75rem!important;letter-spacing:.1em;text-transform:uppercase;}
[data-testid="stMetricValue"]{color:var(--orange)!important;font-family:'Space Mono',monospace!important;font-size:2rem!important;}
[data-testid="stTabs"] button{font-family:'Space Mono',monospace!important;font-size:.72rem!important;letter-spacing:.08em;color:var(--muted)!important;}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--orange)!important;border-bottom:2px solid var(--orange);}
[data-testid="stPlotlyChart"]{border:1px solid var(--border);border-radius:4px;}
label{font-size:.8rem!important;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)!important;}
.stDataFrame{border:1px solid var(--border)!important;}

/* ── Animated header */
.eq-title{
    font-family:'Barlow Condensed',sans-serif;font-weight:900;
    font-size:3.5rem;letter-spacing:-.02em;line-height:.95;
    color:#fff;text-transform:uppercase;
    animation:fadeSlideIn .6s ease-out;
}
.eq-title span{color:var(--orange);}
@keyframes fadeSlideIn{from{opacity:0;transform:translateY(-16px);}to{opacity:1;transform:translateY(0);}}

.eq-subtitle{font-family:'Space Mono',monospace;font-size:.7rem;letter-spacing:.2em;color:var(--muted);margin-top:.4rem;animation:fadeSlideIn .8s ease-out;}
.eq-badge{display:inline-block;padding:2px 8px;border:1px solid;border-radius:2px;font-family:'Space Mono',monospace;font-size:.65rem;letter-spacing:.1em;margin-right:6px;margin-top:6px;transition:all .2s;}
.eq-badge:hover{opacity:.8;}

/* ── Pulse dot */
.live-dot{display:inline-block;width:7px;height:7px;background:var(--red);border-radius:50%;animation:pulse 1.4s infinite;margin-right:5px;vertical-align:middle;}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(225,29,72,.6);}50%{box-shadow:0 0 0 6px rgba(225,29,72,0);}}

/* ── Seismic wave animation on header */
.wave-bar{display:inline-block;width:3px;height:16px;background:var(--orange);margin:0 1px;border-radius:2px;animation:wave 1.2s ease-in-out infinite;}
.wave-bar:nth-child(2){animation-delay:.1s;height:24px;}
.wave-bar:nth-child(3){animation-delay:.2s;height:32px;}
.wave-bar:nth-child(4){animation-delay:.3s;height:20px;}
.wave-bar:nth-child(5){animation-delay:.4s;height:28px;}
.wave-bar:nth-child(6){animation-delay:.2s;height:18px;}
.wave-bar:nth-child(7){animation-delay:.1s;height:12px;}
@keyframes wave{0%,100%{transform:scaleY(1);}50%{transform:scaleY(.3);}}

.section-title{font-family:'Space Mono',monospace;font-size:.7rem;letter-spacing:.18em;color:var(--orange);text-transform:uppercase;border-left:3px solid var(--orange);padding-left:8px;margin:1.5rem 0 .8rem;}

/* ── Stat card with glow */
.stat-card{background:var(--surface2);border:1px solid var(--border);border-radius:6px;padding:1.2rem;text-align:center;transition:all .3s;position:relative;overflow:hidden;}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--orange),transparent);animation:scanline 2s linear infinite;}
@keyframes scanline{0%{transform:translateX(-100%);}100%{transform:translateX(100%);}}
.stat-card:hover{border-color:var(--orange);box-shadow:0 0 20px rgba(249,115,22,.15);}

/* ── Tectonic card */
.tectonic-card{background:var(--surface2);border-left:4px solid;border-radius:0 6px 6px 0;padding:1rem 1.2rem;margin-bottom:.5rem;}

/* ── Event row */
.event-row{display:flex;align-items:center;padding:.6rem .8rem;margin-bottom:.3rem;background:var(--surface);border-radius:0 4px 4px 0;transition:background .2s;}
.event-row:hover{background:#1a1a1a;}
</style>
""", unsafe_allow_html=True)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style='padding:1rem 0 .5rem;'>
        <div style='font-family:Space Mono,monospace;font-size:.65rem;letter-spacing:.2em;color:#666;'>SEISMIC INTELLIGENCE v2.0</div>
        <div style='font-family:Barlow Condensed,sans-serif;font-weight:900;font-size:1.8rem;color:#fff;'>CONTROLS</div>
    </div><hr style='border-color:#2a2a2a;margin:.5rem 0 1rem;'>""", unsafe_allow_html=True)

    data_source = st.selectbox("Data Source", ["Live Feed (USGS)", "Historical Query (USGS)"])
    min_mag = st.slider("Minimum Magnitude", 2.0, 6.0, 2.5, 0.5)
    days_back = 365
    if data_source == "Historical Query (USGS)":
        days_back = st.slider("Days of History", 30, 730, 365, 30)

    st.markdown("---")
    show_ml = st.checkbox("Run ML Pipeline", value=True)
    show_animation = st.checkbox("Show Animated Map", value=True)

    st.markdown("---")
    st.markdown("<div style='font-family:Space Mono,monospace;font-size:.65rem;letter-spacing:.15em;color:#666;margin-bottom:.5rem;'>AFTERSHOCK SIMULATOR</div>", unsafe_allow_html=True)
    sim_mag = st.slider("Mainshock Magnitude", 5.0, 9.5, 7.0, 0.1)
    sim_depth = st.slider("Mainshock Depth (km)", 5, 200, 15, 5)

    st.markdown("---")
    st.markdown("<div style='font-family:Space Mono,monospace;font-size:.65rem;letter-spacing:.15em;color:#666;margin-bottom:.5rem;'>TECTONIC CLASSIFIER</div>", unsafe_allow_html=True)
    tect_depth = st.slider("Depth (km)", 1, 700, 10, 5)
    tect_mag = st.slider("Magnitude", 2.0, 9.5, 6.0, 0.1)

    st.markdown("---")
    if st.button("🔄  Refresh Data", use_container_width=True):
        st.cache_data.clear(); st.cache_resource.clear(); st.rerun()
    st.markdown(f"<div style='font-family:Space Mono,monospace;font-size:.6rem;color:#444;margin-top:1rem;'>TF: {'✓' if _TF_AVAILABLE else '✗'}<br>UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}</div>", unsafe_allow_html=True)


# ─── DATA LOADING ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def load_data(source, min_mag, days):
    ckey = f"{source}_{min_mag}_{days}"
    cached = load_cache(ckey)
    if cached is not None:
        return cached, True
    if source == "Live Feed (USGS)":
        raw = fetch_live_feed("2.5_month")
    else:
        end = datetime.utcnow().strftime("%Y-%m-%d")
        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        raw = fetch_historical(start, end, min_magnitude=min_mag, max_results=5000)
    raw = raw[raw["magnitude"] >= min_mag].copy()
    cleaned = clean_raw(raw)
    featured = engineer_features(cleaned)
    save_cache(featured, ckey)
    return featured, False

@st.cache_resource(show_spinner=False)
def run_ml(data_hash, _df):
    feat_cols = [c for c in FEATURE_COLS if c in _df.columns and c != "magnitude"]
    train_df, test_df = train_test_split_temporal(_df, test_ratio=0.2)
    X_tr, y_tr, scaler, fnames = prepare_ml_matrix(train_df, feat_cols)
    X_te, y_te, _, _ = prepare_ml_matrix(test_df, feat_cols, scale=False)
    if scaler: X_te = scaler.transform(X_te)
    results = train_all_models(X_tr, y_tr, X_te, y_te)
    ens = EarthquakeEnsemble(results)
    try: ens.fit_meta(X_te, y_te)
    except: pass
    return {"results": results, "ensemble": ens, "ensemble_metrics": ens.evaluate(X_te, y_te),
            "feature_names": fnames, "X_test": X_te, "y_test": y_te,
            "train_size": len(train_df), "test_size": len(test_df)}

with st.spinner("Fetching seismic data..."):
    df, from_cache = load_data(data_source, min_mag, days_back)
stats = compute_global_stats(df)


# ─── HEADER ───────────────────────────────────────────────────────────────────
wave_bars = "".join([f"<span class='wave-bar'></span>" for _ in range(7)])

st.markdown(f"""
<div style='padding:1.5rem 0 1rem;border-bottom:1px solid #2a2a2a;margin-bottom:1.5rem;'>
    <div style='display:flex;align-items:flex-end;gap:1.5rem;'>
        <div>
            <div class='eq-title'>SEISMIC<br><span>INTELLIGENCE</span></div>
            <div class='eq-subtitle'>EARTHQUAKE ANALYSIS · ML PREDICTION · RISK MAPPING · AFTERSHOCK FORECASTING</div>
        </div>
        <div style='padding-bottom:.5rem;display:flex;align-items:flex-end;gap:2px;'>{wave_bars}</div>
    </div>
    <div style='margin-top:.8rem;'>
        <span class='live-dot'></span>
        <span style='font-family:Space Mono,monospace;font-size:.7rem;color:#888;'>{"LIVE — USGS" if not from_cache else "CACHED"}</span>
        <span class='eq-badge' style='color:#f97316;border-color:#f97316;'>M{min_mag:.1f}+ FILTER</span>
        <span class='eq-badge' style='color:#0ea5e9;border-color:#0ea5e9;'>{stats.get("total_events",0):,} EVENTS</span>
        <span class='eq-badge' style='color:#16a34a;border-color:#16a34a;'>{stats.get("latest_time","N/A")}</span>
        <span class='eq-badge' style='color:#a855f7;border-color:#a855f7;'>v2.0 · 7 NEW FEATURES</span>
    </div>
</div>""", unsafe_allow_html=True)


# ─── KPI STRIP ────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
c1.metric("Events (24h)", stats.get("events_24h",0))
c2.metric("Events (7d)", stats.get("events_7d",0))
c3.metric("Max Magnitude", stats.get("max_magnitude",0))
c4.metric("Mean Magnitude", stats.get("mean_magnitude",0))
c5.metric("Mean Depth km", stats.get("mean_depth_km",0))
c6.metric("M6+ Events", stats.get("major_events",0))
c7.metric("Tsunami Alerts", stats.get("tsunami_events",0))


# ─── TABS ─────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🌍 GLOBAL MAP",
    "🎬 ANIMATED",
    "📊 ANALYTICS",
    "⚡ ADVANCED",
    "🤖 ML MODELS",
    "⚠️ RISK ZONES",
    "🔁 AFTERSHOCK",
    "🧬 TECTONIC",
    "📋 RECENT EVENTS",
])


# ══ TAB 1: GLOBAL MAP ════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("<div class='section-title'>GLOBAL SEISMIC ACTIVITY</div>", unsafe_allow_html=True)
    m_col, i_col = st.columns([3,1])
    with m_col:
        st.plotly_chart(global_earthquake_map(df, max_events=3000), use_container_width=True, config={"displayModeBar":False})
    with i_col:
        st.markdown("<div class='section-title'>LARGEST EVENTS</div>", unsafe_allow_html=True)
        for _, row in df.nlargest(8,"magnitude")[["magnitude","place","depth_km"]].reset_index(drop=True).iterrows():
            color = magnitude_color(row.magnitude)
            label = magnitude_label(row.magnitude)
            place_s = str(row.place)[:35]+("..." if len(str(row.place))>35 else "")
            st.markdown(f"""<div style='display:flex;align-items:center;padding:.45rem 0;border-bottom:1px solid #1e1e1e;'>
                <span style='font-family:Space Mono,monospace;font-weight:700;font-size:1.2rem;color:{color};min-width:42px;'>{row.magnitude:.1f}</span>
                <div style='margin-left:.5rem;'><div style='font-size:.8rem;color:#ddd;'>{place_s}</div>
                <div style='font-size:.65rem;color:#555;font-family:Space Mono,monospace;'>{row.depth_km:.0f}km · {label}</div></div></div>""", unsafe_allow_html=True)
    if stats.get("max_magnitude",0)>=7.0:
        st.error(f"⚠️ MAJOR SEISMIC EVENT — M{stats['max_magnitude']} · {stats.get('latest_event','')}")
    elif stats.get("max_magnitude",0)>=6.0:
        st.warning(f"⚡ STRONG EARTHQUAKE — M{stats['max_magnitude']} · {stats.get('latest_event','')}")
    else:
        st.info(f"✅ Monitoring active · Max M{stats.get('max_magnitude','N/A')} in dataset")


# ══ TAB 2: ANIMATED MAP ══════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("<div class='section-title'>ANIMATED SEISMIC TIMELINE</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Space Mono,monospace;font-size:.7rem;color:#666;margin-bottom:1rem;'>Press ▶ Play to watch earthquakes unfold week by week across the globe.</div>", unsafe_allow_html=True)
    if show_animation:
        with st.spinner("Building animation frames..."):
            anim_fig = animated_magnitude_bubble(df, max_events=800)
        st.plotly_chart(anim_fig, use_container_width=True, config={"displayModeBar":False})
    else:
        st.info("Enable 'Show Animated Map' in the sidebar to view.")

    st.markdown("<div class='section-title'>CUMULATIVE ENERGY RELEASE</div>", unsafe_allow_html=True)
    with st.spinner("Computing energy timeline..."):
        energy_df = cumulative_energy_timeline(df)
    st.plotly_chart(cumulative_energy_chart(energy_df), use_container_width=True, config={"displayModeBar":False})

    # Energy stat
    total_pj = energy_df["cumulative_energy_petajoules"].iloc[-1]
    major_spikes = energy_df["is_major_spike"].sum()
    e1, e2, e3 = st.columns(3)
    e1.metric("Total Energy Released", f"{total_pj:.2f} PJ")
    e2.metric("Major Energy Spikes", int(major_spikes))
    e3.metric("Avg Energy per Event", f"{total_pj/max(len(df),1)*1000:.2f} TJ")


# ══ TAB 3: ANALYTICS ═════════════════════════════════════════════════════════
with tabs[2]:
    a1,a2 = st.columns(2)
    with a1:
        st.markdown("<div class='section-title'>MAGNITUDE TIMELINE</div>", unsafe_allow_html=True)
        st.plotly_chart(magnitude_time_series(df), use_container_width=True, config={"displayModeBar":False})
    with a2:
        st.markdown("<div class='section-title'>FREQUENCY–MAGNITUDE (G-R LAW)</div>", unsafe_allow_html=True)
        st.plotly_chart(magnitude_histogram(df), use_container_width=True, config={"displayModeBar":False})
    b1,b2 = st.columns(2)
    with b1:
        st.markdown("<div class='section-title'>DEPTH DISTRIBUTION</div>", unsafe_allow_html=True)
        st.plotly_chart(depth_distribution(df), use_container_width=True, config={"displayModeBar":False})
    with b2:
        st.markdown("<div class='section-title'>MAGNITUDE vs DEPTH</div>", unsafe_allow_html=True)
        st.plotly_chart(magnitude_depth_scatter(df), use_container_width=True, config={"displayModeBar":False})
    st.markdown("<div class='section-title'>MAGNITUDE CLASS DISTRIBUTION</div>", unsafe_allow_html=True)
    bins=[0,3,4,5,6,7,8,10]; lbs=["Micro (<3)","Minor (3–4)","Light (4–5)","Moderate (5–6)","Strong (6–7)","Major (7–8)","Great (8+)"]
    df2=df.copy(); df2["mc"]=pd.cut(df2["magnitude"],bins=bins,labels=lbs,right=False)
    cc=df2["mc"].value_counts().sort_index()
    st.dataframe(pd.DataFrame({"Class":cc.index.astype(str),"Count":cc.values,"% Total":(cc.values/len(df2)*100).round(2)}), use_container_width=True, hide_index=True)


# ══ TAB 4: ADVANCED ANALYTICS ════════════════════════════════════════════════
with tabs[3]:
    adv1, adv2 = st.columns(2)

    with adv1:
        st.markdown("<div class='section-title'>SEISMICITY BY HOUR (UTC)</div>", unsafe_allow_html=True)
        hourly_df = hourly_seismicity(df)
        st.plotly_chart(hourly_heatmap(hourly_df), use_container_width=True, config={"displayModeBar":False})

    with adv2:
        st.markdown("<div class='section-title'>WEEKLY SEISMICITY TREND</div>", unsafe_allow_html=True)
        weekly_df = weekly_trend(df)
        st.plotly_chart(weekly_trend_chart(weekly_df), use_container_width=True, config={"displayModeBar":False})

    st.markdown("<div class='section-title'>MAGNITUDE RECURRENCE INTERVALS</div>", unsafe_allow_html=True)
    rec_df = recurrence_intervals(df)
    r1, r2 = st.columns([1.2, 1])
    with r1:
        st.dataframe(rec_df, use_container_width=True, hide_index=True)
    with r2:
        st.plotly_chart(recurrence_bar_chart(rec_df), use_container_width=True, config={"displayModeBar":False})

    st.markdown("<div class='section-title'>SEISMIC GAP ZONES</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:Space Mono,monospace;font-size:.7rem;color:#666;margin-bottom:.8rem;'>Regions with historically high seismicity but recent quiet periods — potential stress accumulation zones.</div>", unsafe_allow_html=True)
    with st.spinner("Detecting seismic gaps..."):
        gap_df = compute_seismic_gaps(df)
    g1, g2 = st.columns([1.5, 1])
    with g1:
        st.plotly_chart(seismic_gap_map(gap_df), use_container_width=True, config={"displayModeBar":False})
    with g2:
        st.markdown("<div class='section-title'>TOP GAPS</div>", unsafe_allow_html=True)
        st.dataframe(gap_df[["lat_cell","lon_cell","gap_score","max_mag","total_events","recent_events"]].head(10).rename(
            columns={"lat_cell":"Lat","lon_cell":"Lon","gap_score":"Gap Score","max_mag":"Max M","total_events":"Total","recent_events":"Recent"}
        ).round(1), use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>MAG–DEPTH CORRELATION BY TECTONIC ZONE</div>", unsafe_allow_html=True)
    corr = mag_depth_correlation(df)
    corr_cols = st.columns(len(corr))
    for col, (zone, vals) in zip(corr_cols, corr.items()):
        sig = "✓ Significant" if vals["significant"] else "✗ Not significant"
        col.markdown(f"""<div style='background:#111;border:1px solid #2a2a2a;border-radius:4px;padding:.8rem;text-align:center;'>
            <div style='font-family:Space Mono,monospace;font-size:.6rem;color:#666;'>{zone}</div>
            <div style='font-family:Space Mono,monospace;font-size:1.6rem;color:#f97316;font-weight:700;'>r={vals["r"]}</div>
            <div style='font-family:Space Mono,monospace;font-size:.6rem;color:#555;'>p={vals["p"]} · n={vals["n"]}<br>{sig}</div>
        </div>""", unsafe_allow_html=True)


# ══ TAB 5: ML MODELS ═════════════════════════════════════════════════════════
with tabs[4]:
    if not show_ml:
        st.info("Enable 'Run ML Pipeline' in the sidebar.")
    else:
        with st.spinner("Training ML models... (~30–60s first run)"):
            dhash = str(hash(str(df["time"].max())+str(len(df))+str(min_mag)))
            ml = run_ml(dhash, df)
        s1,s2,s3,s4,s5 = st.columns(5)
        s1.metric("Train Events",f"{ml['train_size']:,}"); s2.metric("Test Events",f"{ml['test_size']:,}")
        s3.metric("Features",len(ml["feature_names"])); s4.metric("Best Model",ml["results"][0]["name"].split()[0] if ml["results"] else "—")
        s5.metric("Ensemble MAE",ml["ensemble_metrics"]["mae"])
        st.markdown("<div class='section-title'>MODEL RESULTS</div>", unsafe_allow_html=True)
        all_r = ml["results"]+[ml["ensemble_metrics"]]
        mc1,mc2 = st.columns(2)
        for i,res in enumerate(all_r):
            col = mc1 if i%2==0 else mc2
            is_best=(i==0); is_ens=(res["name"]=="Stacked Ensemble")
            accent="#f97316" if is_best else ("#0ea5e9" if is_ens else "#2a2a2a")
            badge="  🏆" if is_best else ("  🔷" if is_ens else "")
            with col:
                st.markdown(f"""<div style='background:#111;border:1px solid {accent};border-radius:4px;padding:.8rem 1rem;margin-bottom:.6rem;transition:all .2s;'>
                    <div style='display:flex;justify-content:space-between;'><span style='font-family:Space Mono,monospace;font-weight:700;color:#fff;'>{res["name"]}{badge}</span>
                    <span style='font-family:Space Mono,monospace;font-size:.7rem;color:{accent};'>MAE {res["mae"]}</span></div>
                    <div style='display:flex;gap:1.5rem;margin-top:.4rem;'>
                    <span style='font-size:.75rem;color:#888;font-family:Space Mono,monospace;'>RMSE: {res["rmse"]}</span>
                    <span style='font-size:.75rem;color:#888;font-family:Space Mono,monospace;'>R²: {res["r2"]}</span></div></div>""", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>MODEL COMPARISON</div>", unsafe_allow_html=True)
        st.plotly_chart(model_performance_chart(ml["results"]), use_container_width=True, config={"displayModeBar":False})
        if ml["results"] and "feature_importances" in ml["results"][0]:
            st.markdown("<div class='section-title'>FEATURE IMPORTANCE</div>", unsafe_allow_html=True)
            fi=ml["results"][0]["feature_importances"]
            fi_df=pd.DataFrame({"Feature":ml["feature_names"],"Importance":fi}).sort_values("Importance",ascending=False).head(12)
            fig_fi=px.bar(fi_df,x="Importance",y="Feature",orientation="h",color="Importance",color_continuous_scale=[[0,"#333"],[1,"#f97316"]])
            fig_fi.update_layout(paper_bgcolor="#080808",plot_bgcolor="#111",font_color="#d4d4d4",showlegend=False,coloraxis_showscale=False,
                                 xaxis=dict(gridcolor="#2a2a2a"),yaxis=dict(gridcolor="#2a2a2a"),margin=dict(l=0,r=10,t=10,b=10),height=380)
            st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar":False})


# ══ TAB 6: RISK ZONES ════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("<div class='section-title'>SEISMIC RISK HOTSPOT MAP</div>", unsafe_allow_html=True)
    with st.spinner("Computing risk zones..."):
        risk_data = compute_risk_grid(df, resolution=72)
        zone_df = zone_risk_summary(df).head(20)
    hs = risk_data["hotspots"]
    rfig = go.Figure()
    rfig.add_trace(go.Scattergeo(lat=df["latitude"],lon=df["longitude"],mode="markers",
        marker=dict(size=2,color="#ffffff",opacity=0.06),hoverinfo="skip"))
    rfig.add_trace(go.Scattergeo(lat=hs["latitude"],lon=hs["longitude"],mode="markers",
        marker=dict(size=(hs["risk_score"]/hs["risk_score"].max()*30+8).tolist(),
                    color=hs["risk_score"].tolist(),colorscale=[[0,"#1d4ed8"],[.4,"#f59e0b"],[.7,"#ea580c"],[1,"#7c3aed"]],
                    opacity=0.8,colorbar=dict(title="Risk",thickness=10,x=0.02,len=0.5),line=dict(width=0)),
        text=[f"Risk:{s:.0f}/100" for s in hs["risk_score"]],hoverinfo="text"))
    rfig.update_geos(projection_type="natural earth",showland=True,landcolor="#0e0e0e",showocean=True,oceancolor="#0a1628",
                     showcoastlines=True,coastlinecolor="#1e1e1e",showframe=False,bgcolor="#080808")
    rfig.update_layout(paper_bgcolor="#080808",font_color="#d4d4d4",margin=dict(l=0,r=0,t=0,b=0),height=460,showlegend=False)
    st.plotly_chart(rfig, use_container_width=True, config={"displayModeBar":False})
    st.markdown("<div class='section-title'>TOP RISK ZONES (10° GRID)</div>", unsafe_allow_html=True)
    dz=zone_df[["lat_zone","lon_zone","event_count","max_mag","mean_mag","risk_score","tsunami_events"]].copy()
    dz.columns=["Lat Zone","Lon Zone","Events","Max Mag","Avg Mag","Risk Score","Tsunamis"]
    st.dataframe(dz.round(2), use_container_width=True, hide_index=True)


# ══ TAB 7: AFTERSHOCK ════════════════════════════════════════════════════════
with tabs[6]:
    report = full_aftershock_report(sim_mag, sim_depth)
    st.markdown(f"""<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:1.5rem;'>
        <div class='stat-card'><div style='font-family:Space Mono,monospace;font-size:.6rem;color:#666;'>LARGEST AFTERSHOCK</div>
            <div style='font-family:Space Mono,monospace;font-size:2rem;color:#f97316;font-weight:700;'>M{report["largest_aftershock"]:.1f}</div>
            <div style='font-family:Space Mono,monospace;font-size:.6rem;color:#555;'>Bath's Law</div></div>
        <div class='stat-card'><div style='font-family:Space Mono,monospace;font-size:.6rem;color:#666;'>EXPECTED M2+ (30d)</div>
            <div style='font-family:Space Mono,monospace;font-size:2rem;color:#0ea5e9;font-weight:700;'>{report["expected_m2plus_30days"]:.0f}</div>
            <div style='font-family:Space Mono,monospace;font-size:.6rem;color:#555;'>Omori-Utsu</div></div>
        <div class='stat-card'><div style='font-family:Space Mono,monospace;font-size:.6rem;color:#666;'>P(M≥5 NEXT 24H)</div>
            <div style='font-family:Space Mono,monospace;font-size:2rem;color:#e11d48;font-weight:700;'>{report["prob_m5_next_24h"]*100:.1f}%</div>
            <div style='font-family:Space Mono,monospace;font-size:.6rem;color:#555;'>Poisson</div></div>
        <div class='stat-card'><div style='font-family:Space Mono,monospace;font-size:.6rem;color:#666;'>ZONE RADIUS</div>
            <div style='font-family:Space Mono,monospace;font-size:2rem;color:#a855f7;font-weight:700;'>{report["aftershock_zone_radius_km"]:.0f}<span style='font-size:1rem;'>km</span></div>
            <div style='font-family:Space Mono,monospace;font-size:.6rem;color:#555;'>Wells-Coppersmith</div></div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>30-DAY AFTERSHOCK FORECAST</div>", unsafe_allow_html=True)
    st.plotly_chart(aftershock_forecast_chart(report), use_container_width=True, config={"displayModeBar":False})
    op=report["omori_params"]
    st.markdown(f"""<div style='background:#111;border:1px solid #2a2a2a;border-radius:4px;padding:.8rem 1.2rem;
        font-family:Space Mono,monospace;font-size:.75rem;color:#888;'>
        <span style='color:#f97316;'>OMORI-UTSU</span> λ(t)=K/(t+c)^p | K={op["K"]:.4f} c={op["c"]}d p={op["p"]} |
        P(M≥6/24h)={report["prob_m6_next_24h"]*100:.2f}%</div>""", unsafe_allow_html=True)


# ══ TAB 8: TECTONIC CLASSIFIER ═══════════════════════════════════════════════
with tabs[7]:
    st.markdown("<div class='section-title'>TECTONIC TYPE CLASSIFIER</div>", unsafe_allow_html=True)
    tect = classify_tectonic_type(tect_depth, tect_mag)
    st.markdown(f"""
    <div class='tectonic-card' style='border-color:{tect["color"]};'>
        <div style='font-family:Space Mono,monospace;font-size:.6rem;color:#666;letter-spacing:.1em;'>CLASSIFIED AS</div>
        <div style='font-family:Barlow Condensed,sans-serif;font-weight:700;font-size:1.8rem;color:{tect["color"]};'>{tect["type"]}</div>
        <div style='font-size:.9rem;color:#aaa;margin-top:.3rem;'>{tect["desc"]}</div>
        <div style='margin-top:.6rem;font-family:Space Mono,monospace;font-size:.7rem;color:#555;'>
            Input: {tect_depth}km depth · M{tect_mag:.1f}
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>ALL TECTONIC TYPES</div>", unsafe_allow_html=True)
    tectonic_examples = [
        (5, 7.0), (12, 5.5), (35, 4.5), (150, 6.5), (500, 7.5),
    ]
    for d, m in tectonic_examples:
        t = classify_tectonic_type(d, m)
        st.markdown(f"""<div class='tectonic-card' style='border-color:{t["color"]};'>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
                <div>
                    <span style='font-family:Space Mono,monospace;font-size:.65rem;color:#555;'>{d}km · M{m}</span>
                    <div style='font-weight:700;color:{t["color"]};font-size:1rem;'>{t["type"]}</div>
                    <div style='font-size:.8rem;color:#888;'>{t["desc"]}</div>
                </div>
            </div></div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>DEPTH CLASS BREAKDOWN IN CURRENT DATASET</div>", unsafe_allow_html=True)
    if "depth_class" in df.columns:
        dc = df["depth_class"].value_counts()
        d1,d2,d3 = st.columns(3)
        colors_map = {"shallow":"#f97316","intermediate":"#0ea5e9","deep":"#a855f7"}
        for col, (cls, cnt) in zip([d1,d2,d3], dc.items()):
            pct = cnt/len(df)*100
            col.markdown(f"""<div style='background:#111;border:1px solid {colors_map.get(str(cls),"#333")};border-radius:4px;padding:1rem;text-align:center;'>
                <div style='font-family:Space Mono,monospace;font-size:.6rem;color:#666;'>{str(cls).upper()}</div>
                <div style='font-family:Space Mono,monospace;font-size:1.8rem;color:{colors_map.get(str(cls),"#fff")};font-weight:700;'>{cnt:,}</div>
                <div style='font-family:Space Mono,monospace;font-size:.7rem;color:#555;'>{pct:.1f}% of total</div>
            </div>""", unsafe_allow_html=True)


# ══ TAB 9: RECENT EVENTS ═════════════════════════════════════════════════════
with tabs[8]:
    st.markdown("<div class='section-title'>MOST RECENT SEISMIC EVENTS</div>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    filter_min_mag = f1.slider("Min magnitude", 2.0, 8.0, 2.5, 0.5, key="filter_mag")
    filter_depth = f2.selectbox("Depth class", ["All", "shallow", "intermediate", "deep"])
    n_ev = f3.slider("Events to show", 10, 100, 25, 5)

    recent = df.copy()
    recent = recent[recent["magnitude"] >= filter_min_mag]
    if filter_depth != "All" and "depth_class" in recent.columns:
        recent = recent[recent["depth_class"] == filter_depth]
    recent = recent.sort_values("time", ascending=False).head(n_ev).reset_index(drop=True)

    st.markdown(f"<div style='font-family:Space Mono,monospace;font-size:.7rem;color:#555;margin-bottom:.8rem;'>Showing {len(recent)} events</div>", unsafe_allow_html=True)

    for _, row in recent.iterrows():
        color=magnitude_color(row.magnitude); label=magnitude_label(row.magnitude)
        energy=format_energy(row.magnitude)
        t_str=row.time.strftime("%Y-%m-%d %H:%M UTC") if hasattr(row.time,"strftime") else str(row.time)
        tsf=f"&nbsp;<span style='color:#e11d48;'>🌊</span>" if row.get("tsunami",0) else ""
        depth_cls = str(row.get("depth_class","")).upper() if "depth_class" in row else ""
        st.markdown(f"""<div class='event-row' style='border-left:3px solid {color};'>
            <div style='min-width:56px;text-align:center;'>
                <div style='font-family:Space Mono,monospace;font-weight:700;font-size:1.4rem;color:{color};line-height:1;'>{row.magnitude:.1f}</div>
                <div style='font-family:Space Mono,monospace;font-size:.55rem;color:#555;'>{label.upper()}</div></div>
            <div style='margin-left:1rem;flex:1;'>
                <div style='font-size:.9rem;color:#e5e5e5;'>{row.place}</div>
                <div style='font-family:Space Mono,monospace;font-size:.65rem;color:#555;margin-top:2px;'>
                    {t_str} | {row.depth_km:.0f}km {depth_cls} | {energy}{tsf}</div></div>
            <div style='text-align:right;font-family:Space Mono,monospace;font-size:.65rem;color:#444;'>
                {row.latitude:.2f}°N<br>{row.longitude:.2f}°E</div></div>""", unsafe_allow_html=True)
