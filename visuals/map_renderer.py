"""
Map & Visualization Renderer
Builds Plotly and Folium maps/charts for the dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

# ─── COLOR PALETTE ────────────────────────────────────────────────────────────
MAGNITUDE_COLORSCALE = [
    [0.0, "#0ea5e9"],
    [0.2, "#22d3ee"],
    [0.4, "#a3e635"],
    [0.6, "#facc15"],
    [0.8, "#fb923c"],
    [1.0, "#dc2626"],
]

THEME = dict(
    bg="#0d0d0d",
    surface="#141414",
    border="#2a2a2a",
    text="#e5e5e5",
    accent="#f97316",
    accent2="#0ea5e9",
    red="#e11d48",
    purple="#a855f7",
    green="#16a34a",
)


def global_earthquake_map(df: pd.DataFrame, max_events: int = 2000) -> go.Figure:
    """
    Interactive global scatter map of earthquake events.
    Size = magnitude, Color = depth.
    """
    df = df.copy().head(max_events)
    df["size"] = (df["magnitude"] ** 2.5).clip(2, 40)

    fig = go.Figure()

    fig.add_trace(
        go.Scattergeo(
            lat=df["latitude"],
            lon=df["longitude"],
            mode="markers",
            marker=dict(
                size=df["size"],
                color=df["depth_km"],
                colorscale="Inferno",
                colorbar=dict(title="Depth (km)", x=0.02, thickness=12, len=0.5),
                opacity=0.75,
                line=dict(width=0.3, color="rgba(255,255,255,0.3)"),
            ),
            text=[
                f"<b>{row.place}</b><br>"
                f"M{row.magnitude:.1f} | {row.depth_km:.0f} km deep<br>"
                f"{row.time.strftime('%Y-%m-%d %H:%M UTC') if hasattr(row.time, 'strftime') else row.time}"
                for _, row in df.iterrows()
            ],
            hoverinfo="text",
            name="Earthquakes",
        )
    )

    fig.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="#1c1c1c",
        showocean=True,
        oceancolor="#0a1628",
        showcoastlines=True,
        coastlinecolor="#333333",
        showframe=False,
        bgcolor=THEME["bg"],
    )
    fig.update_layout(
        paper_bgcolor=THEME["bg"],
        plot_bgcolor=THEME["bg"],
        font_color=THEME["text"],
        margin=dict(l=0, r=0, t=0, b=0),
        height=480,
        showlegend=False,
    )
    return fig


def magnitude_depth_scatter(df: pd.DataFrame) -> go.Figure:
    """Magnitude vs Depth coloured by time."""
    fig = px.scatter(
        df,
        x="depth_km",
        y="magnitude",
        color="magnitude",
        color_continuous_scale=MAGNITUDE_COLORSCALE,
        opacity=0.6,
        size_max=8,
        labels={"depth_km": "Depth (km)", "magnitude": "Magnitude"},
    )
    fig.update_layout(**_base_layout(title="Magnitude vs Depth"))
    return fig


def magnitude_time_series(df: pd.DataFrame) -> go.Figure:
    """Time series of magnitudes with rolling max."""
    df_sorted = df.sort_values("time")
    rolling_max = df_sorted["magnitude"].rolling(50, min_periods=1).max()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sorted["time"], y=df_sorted["magnitude"],
        mode="markers",
        marker=dict(color=df_sorted["magnitude"], colorscale=MAGNITUDE_COLORSCALE,
                    size=4, opacity=0.5),
        name="Events",
    ))
    fig.add_trace(go.Scatter(
        x=df_sorted["time"], y=rolling_max,
        mode="lines",
        line=dict(color=THEME["accent"], width=2),
        name="Rolling Max",
    ))
    fig.update_layout(**_base_layout(title="Earthquake Timeline"))
    return fig


def magnitude_histogram(df: pd.DataFrame) -> go.Figure:
    """Frequency-magnitude distribution (Gutenberg-Richter)."""
    mags = df["magnitude"].dropna()
    bins = np.arange(mags.min(), mags.max() + 0.1, 0.2)
    counts, edges = np.histogram(mags, bins=bins)
    log_counts = np.log10(counts + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Fit G-R b-value
    valid = counts > 0
    if valid.sum() > 3:
        m_c = edges[np.argmax(counts)]  # completeness magnitude
        above = mags[mags >= m_c]
        if len(above) > 10:
            b_val = np.log10(np.e) / (above.mean() - m_c + 1e-9)
        else:
            b_val = 1.0
    else:
        b_val = 1.0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=centers, y=log_counts,
        marker_color=THEME["accent"], opacity=0.8,
        name="log10(N)",
    ))
    # G-R fit line
    m_range = np.linspace(centers.min(), centers.max(), 100)
    gr_line = log_counts.max() - b_val * (m_range - centers[np.argmax(log_counts)])
    fig.add_trace(go.Scatter(
        x=m_range, y=gr_line,
        mode="lines",
        line=dict(color=THEME["accent2"], dash="dash", width=2),
        name=f"G-R fit (b={b_val:.2f})",
    ))
    fig.update_layout(**_base_layout(
        title=f"Frequency-Magnitude (b={b_val:.2f})",
        xaxis_title="Magnitude",
        yaxis_title="log₁₀(N)",
    ))
    return fig


def depth_distribution(df: pd.DataFrame) -> go.Figure:
    """Depth distribution by class."""
    fig = go.Figure()
    colors = {"shallow": "#f97316", "intermediate": "#0ea5e9", "deep": "#a855f7"}
    for label, color in colors.items():
        subset = df[df["depth_class"] == label] if "depth_class" in df.columns else df
        if len(subset) == 0:
            continue
        fig.add_trace(go.Histogram(
            x=subset["depth_km"],
            name=label.title(),
            marker_color=color,
            opacity=0.75,
            xbins=dict(size=10),
        ))
    fig.update_layout(barmode="overlay", **_base_layout(
        title="Depth Distribution",
        xaxis_title="Depth (km)",
        yaxis_title="Count",
    ))
    return fig


def model_performance_chart(results: list[dict]) -> go.Figure:
    """Horizontal bar chart of model MAEs."""
    names = [r["name"] for r in results]
    maes = [r["mae"] for r in results]
    r2s = [r.get("r2", 0) for r in results]
    colors = [THEME["accent"] if i == 0 else "#444" for i in range(len(names))]

    fig = make_subplots(rows=1, cols=2, subplot_titles=["MAE (lower = better)", "R² (higher = better)"])
    fig.add_trace(go.Bar(x=maes, y=names, orientation="h", marker_color=colors, name="MAE"), row=1, col=1)
    fig.add_trace(go.Bar(x=r2s, y=names, orientation="h", marker_color=colors, name="R²"), row=1, col=2)
    layout = _base_layout(title="Model Comparison")
    layout["height"] = 350
    fig.update_layout(**layout)
    return fig


def aftershock_forecast_chart(forecast: dict) -> go.Figure:
    """Bar chart of aftershock daily forecast."""
    days = list(range(1, len(forecast["daily_forecast_30d"]) + 1))
    counts = forecast["daily_forecast_30d"]
    fig = go.Figure(go.Bar(
        x=days, y=counts,
        marker=dict(
            color=counts,
            colorscale=[[0, "#1d4ed8"], [0.5, "#f97316"], [1, "#dc2626"]],
        ),
        name="Expected aftershocks",
    ))
    fig.update_layout(**_base_layout(
        title=f"Aftershock Forecast — M{forecast['mainshock_magnitude']} event",
        xaxis_title="Days after mainshock",
        yaxis_title="Expected events (M≥2)",
    ))
    return fig


def animated_magnitude_bubble(df: pd.DataFrame, max_events: int = 800) -> go.Figure:
    """
    Animated scatter map: earthquakes appearing over time with pulsing bubbles.
    Uses Plotly animation frames by week.
    """
    df = df.copy().sort_values("time").head(max_events)
    df["week"] = df["time"].dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")
    df["size"] = (df["magnitude"] ** 2.2).clip(3, 35)
    df["color"] = df["magnitude"]
    df["label"] = df.apply(lambda r: f"M{r.magnitude:.1f} — {str(r.place)[:40]}", axis=1)

    fig = px.scatter_geo(
        df, lat="latitude", lon="longitude",
        color="magnitude", size="size",
        animation_frame="week",
        hover_name="label",
        color_continuous_scale=MAGNITUDE_COLORSCALE,
        projection="natural earth",
        size_max=25,
    )
    fig.update_geos(
        showland=True, landcolor="#111", showocean=True, oceancolor="#0a1628",
        showcoastlines=True, coastlinecolor="#222", showframe=False, bgcolor=THEME["bg"],
    )
    fig.update_layout(
        paper_bgcolor=THEME["bg"], font_color=THEME["text"],
        margin=dict(l=0, r=0, t=0, b=0), height=500,
        coloraxis_colorbar=dict(title="Mag", thickness=10, x=0.02, len=0.5),
        updatemenus=[dict(
            type="buttons", showactive=False,
            x=0.05, y=-0.05, xanchor="left",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, {"frame": {"duration": 400, "redraw": True}, "fromcurrent": True}]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
            ],
            bgcolor="#1a1a1a", bordercolor="#333", font=dict(color="#fff"),
        )],
    )
    return fig


def cumulative_energy_chart(energy_df: pd.DataFrame) -> go.Figure:
    """Animated cumulative energy release over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=energy_df["time"], y=energy_df["cumulative_energy_petajoules"],
        mode="lines", fill="tozeroy",
        line=dict(color=THEME["accent"], width=2),
        fillcolor="rgba(249,115,22,0.12)",
        name="Cumulative Energy (PJ)",
    ))
    # Spike markers
    spikes = energy_df[energy_df["is_major_spike"]]
    fig.add_trace(go.Scatter(
        x=spikes["time"], y=spikes["cumulative_energy_petajoules"],
        mode="markers",
        marker=dict(color=THEME["red"], size=10, symbol="star",
                    line=dict(width=1, color="#fff")),
        text=[f"M{m:.1f} — {p[:30]}" for m, p in zip(spikes["magnitude"], spikes["place"])],
        hoverinfo="text", name="Major Event",
    ))
    fig.update_layout(**_base_layout(
        title="Cumulative Seismic Energy Release",
        xaxis_title="Date", yaxis_title="Petajoules",
    ))
    return fig


def hourly_heatmap(hourly_df: pd.DataFrame) -> go.Figure:
    """Polar bar chart of seismicity by hour of day."""
    hours = hourly_df["hour"].tolist()
    counts = hourly_df["event_count"].tolist()
    fig = go.Figure(go.Barpolar(
        r=counts, theta=[h * 15 for h in hours],
        width=[14] * len(hours),
        marker=dict(
            color=counts,
            colorscale=[[0, "#1a1a2e"], [0.5, "#f97316"], [1, "#e11d48"]],
            line=dict(color=THEME["bg"], width=0.5),
        ),
        name="Events",
    ))
    fig.update_layout(
        paper_bgcolor=THEME["bg"], plot_bgcolor=THEME["bg"],
        font_color=THEME["text"],
        polar=dict(
            bgcolor=THEME["surface"],
            radialaxis=dict(showticklabels=False, gridcolor=THEME["border"]),
            angularaxis=dict(
                tickvals=[0,45,90,135,180,225,270,315],
                ticktext=["0h","3h","6h","9h","12h","15h","18h","21h"],
                gridcolor=THEME["border"], color=THEME["text"],
            ),
        ),
        title=dict(text="Seismicity by Hour (UTC)", font=dict(color=THEME["text"], size=13)),
        margin=dict(l=40, r=40, t=50, b=40), height=360,
        showlegend=False,
    )
    return fig


def weekly_trend_chart(weekly_df: pd.DataFrame) -> go.Figure:
    """Weekly seismicity trend with uptick indicators."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=weekly_df["week"].astype(str), y=weekly_df["events"],
        marker=dict(
            color=weekly_df["events"],
            colorscale=[[0, "#1d3557"], [0.5, "#f97316"], [1, "#e11d48"]],
        ),
        name="Events/Week",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=weekly_df["week"].astype(str), y=weekly_df["max_mag"],
        mode="lines+markers",
        line=dict(color=THEME["accent2"], width=2, dash="dot"),
        marker=dict(size=6, color=THEME["accent2"]),
        name="Max Magnitude",
    ), secondary_y=True)
    layout = _base_layout(title="Weekly Seismicity Trend", xaxis_title="Week")
    layout["yaxis2"] = dict(title="Max Mag", gridcolor=THEME["border"],
                            zeroline=False, color=THEME["accent2"])
    fig.update_layout(**layout)
    return fig


def seismic_gap_map(gap_df: pd.DataFrame) -> go.Figure:
    """Map showing seismic gap zones."""
    fig = go.Figure(go.Scattergeo(
        lat=gap_df["lat_cell"] + 10,
        lon=gap_df["lon_cell"] + 10,
        mode="markers",
        marker=dict(
            size=(gap_df["gap_score"] / gap_df["gap_score"].max() * 40 + 15).tolist(),
            color=gap_df["gap_score"].tolist(),
            colorscale=[[0, "#1d4ed8"], [0.5, "#f59e0b"], [1, "#7c3aed"]],
            opacity=0.7,
            colorbar=dict(title="Gap Score", thickness=10, x=0.02, len=0.5),
            symbol="square",
            line=dict(width=1, color="rgba(255,255,255,0.2)"),
        ),
        text=[f"Gap Score: {s:.0f} | Max M{m:.1f}" for s, m in
              zip(gap_df["gap_score"], gap_df["max_mag"])],
        hoverinfo="text", name="Seismic Gap",
    ))
    fig.update_geos(
        showland=True, landcolor="#0e0e0e", showocean=True, oceancolor="#0a1628",
        showcoastlines=True, coastlinecolor="#1e1e1e", showframe=False, bgcolor=THEME["bg"],
    )
    fig.update_layout(
        paper_bgcolor=THEME["bg"], font_color=THEME["text"],
        margin=dict(l=0, r=0, t=30, b=0), height=420,
        title=dict(text="Seismic Gap Zones", font=dict(color=THEME["text"], size=13)),
        showlegend=False,
    )
    return fig


def recurrence_bar_chart(rec_df: pd.DataFrame) -> go.Figure:
    """Bar chart for magnitude recurrence intervals."""
    valid = rec_df[rec_df["Annual Rate"] > 0].copy()
    fig = go.Figure(go.Bar(
        x=valid["Min Magnitude"], y=valid["Annual Rate"],
        marker=dict(
            color=valid["Annual Rate"],
            colorscale=[[0, "#1d4ed8"], [0.5, "#f97316"], [1, "#e11d48"]],
            line=dict(color=THEME["bg"], width=1),
        ),
        text=[f"{v:.1f}/yr" for v in valid["Annual Rate"]],
        textposition="outside", textfont=dict(color=THEME["text"], size=11),
    ))
    fig.update_layout(**_base_layout(
        title="Annual Earthquake Rates by Magnitude",
        xaxis_title="Minimum Magnitude",
        yaxis_title="Events per Year",
    ))
    return fig


def _base_layout(title: str = "", xaxis_title: str = "", yaxis_title: str = "") -> dict:
    return dict(
        title=dict(text=title, font=dict(color=THEME["text"], size=14)),
        paper_bgcolor=THEME["bg"],
        plot_bgcolor=THEME["surface"],
        font=dict(color=THEME["text"]),
        xaxis=dict(title=xaxis_title, gridcolor=THEME["border"], zeroline=False),
        yaxis=dict(title=yaxis_title, gridcolor=THEME["border"], zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=THEME["border"]),
        margin=dict(l=40, r=20, t=40, b=40),
        height=360,
    )
