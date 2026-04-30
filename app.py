# ============================================================
#  Cholera Outbreak Prediction System
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Cholera Outbreak Prediction", page_icon="🦠", layout="wide")

st.markdown("""<style>
.main{background:#0e1117!important}
.stApp{background:#0e1117!important}
[data-testid="stSidebar"]{background:#111827!important;border-right:1px solid #1f2937!important}
.stButton>button{background:linear-gradient(135deg,#3b82f6,#8b5cf6)!important;color:#fff!important;
  border:none!important;border-radius:10px!important;font-weight:600!important;padding:10px 24px!important}
.stButton>button:hover{box-shadow:0 6px 20px rgba(59,130,246,.35)!important}
[data-testid="metric-container"]{background:#1e293b!important;border:1px solid #334155!important;
  border-radius:12px!important;padding:16px!important}
[data-testid="metric-container"] label{color:#94a3b8!important;font-size:11px!important;
  text-transform:uppercase;letter-spacing:.05em}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:#e2e8f0!important;
  font-size:28px!important;font-weight:700!important}
</style>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════
def load_data():
    try:
        df = pd.read_csv("cholera_data_v3.csv", encoding="utf-8")
    except Exception:
        return None
    # Drop unnecessary index column
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    if "Cases" not in df.columns:
        return None
    df["Target"] = (df["Cases"] > df["Cases"].median()).astype(int)
    try:
        df["date"] = pd.to_datetime({"year": df["year"], "month": df["mon"], "day": df["day"]}, errors="coerce")
    except Exception:
        pass
    return df


def engineer(df):
    d = df.copy()
    if "date" in d.columns:
        d = d.sort_values("date").reset_index(drop=True)
    d["Temp_x_Preci"] = d["Temp"] * d["preci"]
    d["preci_per_temp"] = d["preci"] / (d["Temp"] + 1e-6)
    for lag in [1, 2, 4]:
        d[f"preci_lag_{lag}"] = d["preci"].shift(lag)
        d[f"Temp_lag_{lag}"] = d["Temp"].shift(lag)
        d[f"Cases_lag_{lag}"] = d["Cases"].shift(lag)
    d["preci_roll_mean_4w"] = d["preci"].rolling(4, min_periods=1).mean()
    d["preci_roll_std_4w"] = d["preci"].rolling(4, min_periods=1).std().fillna(0)
    d["cases_roll_sum_4w"] = d["Cases"].rolling(4, min_periods=1).sum()
    d["week_num"] = d["week_of_outbreak"].astype(str).str.extract(r"(\d+)").astype(float)
    d = d.dropna(subset=["preci_lag_1", "Cases_lag_1"]).reset_index(drop=True)
    nf = [c for c in ["Latitude","Longitude","is_coastal_district","week_num","preci","LAI",
        "Temp","Fatality_Rate","Temp_x_Preci","preci_per_temp","preci_roll_mean_4w",
        "preci_roll_std_4w","cases_roll_sum_4w","preci_lag_1","preci_lag_2","preci_lag_4",
        "Temp_lag_1","Temp_lag_2","Temp_lag_4","Cases_lag_1","Cases_lag_2","Cases_lag_4"]
        if c in d.columns]
    cf = [c for c in ["state_ut","Season"] if c in d.columns]
    X = d[nf+cf].copy(); y = d["Target"].copy()
    mc = [c for c in ["date","state_ut","district","Cases","Target"] if c in d.columns]
    return X, y, d[mc].copy().reset_index(drop=True)


def get_df():
    if "raw_df" not in st.session_state:
        st.session_state["raw_df"] = load_data()
    return st.session_state["raw_df"]


def clean_series(s):
    """Convert to list, replace inf/nan — safe for plotly."""
    return s.replace([np.inf, -np.inf], np.nan).dropna().tolist()


def dark_layout(title="", height=None, **extra):
    """Return a layout dict for dark theme. Height only added if specified."""
    layout = dict(
        paper_bgcolor="#111827",
        plot_bgcolor="#1e293b",
        font=dict(color="#94a3b8", size=11, family="Segoe UI, sans-serif"),
        margin=dict(l=50, r=25, t=45, b=45),
    )
    if height is not None:
        layout["height"] = height
    if title:
        layout["title"] = dict(text=title, font=dict(size=14, color="#e2e8f0"))
    layout.update(extra)
    return layout


def dark_axes():
    return dict(gridcolor="#1e293b", color="#64748b", zerolinecolor="#334155")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════
def main():

    with st.sidebar:
        st.markdown("### Cholera Prediction")
        page = st.radio("Navigate", [
            "Overview", "Data Analysis",
            "Model Training", "Forecasting"], key="nav")
        st.markdown("---")
        if "model" in st.session_state and "metrics" in st.session_state:
            m = st.session_state["metrics"]
            st.success(f"Model ready — F2: {m['F2']:.4f}")

    df = get_df()
    if df is None:
        st.error("**Dataset not found.** Place `cholera_data_v3.csv` in the app folder.")
        st.stop()

    # ════════════════════════════════════════════════════════
    #  OVERVIEW
    # ════════════════════════════════════════════════════════
    if "Overview" in page:
        st.title("Cholera Outbreak Prediction")
        st.caption("ML Early Warning System | Environmental & Temporal Drivers | District-Level Risk")
        st.divider()

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.metric("Records", f"{len(df):,}")
        with c2: st.metric("Features", f"{df.shape[1]}")
        with c3: st.metric("Outbreak Rate", f"{df['Target'].mean():.1%}")
        with c4: st.metric("Districts", f"{df['district'].nunique() if 'district' in df.columns else 0}")
        with c5: st.metric("States", f"{df['state_ut'].nunique() if 'state_ut' in df.columns else 0}")
        st.divider()

        # Row 1: Cases Histogram + Target Pie
        col1, col2 = st.columns(2)

        with col1:
            try:
                cases = df["Cases"].replace([np.inf, -np.inf], np.nan).dropna()
                if len(cases) == 0:
                    st.info("No Cases data.")
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=cases.value_counts().sort_index().index.tolist()[:40],
                        y=cases.value_counts().sort_index().values.tolist()[:40],
                        marker_color="#3b82f6",
                        marker_line=dict(color="#1e3a5f", width=1),
                        opacity=0.85,
                    ))
                    fig.update_layout(**dark_layout("Cases Count", 420))
                    fig.update_xaxes(title_text="Number of Cases", **dark_axes())
                    fig.update_yaxes(title_text="Frequency", **dark_axes())
                    st.plotly_chart(fig, use_container_width=True, key="ov_cases")
            except Exception as e:
                st.error(f"Cases chart error: {e}")

        with col2:
            try:
                tc = df["Target"].value_counts()
                fig = go.Figure()
                fig.add_trace(go.Pie(
                    labels=["Below Median", "Above Median"],
                    values=[int(tc.get(0, 0)), int(tc.get(1, 0))],
                    marker=dict(colors=["#3b82f6", "#ef4444"], line=dict(color="#111827", width=2)),
                    hole=0.55,
                    textinfo="label+percent",
                    textfont=dict(color="#e2e8f0", size=12),
                ))
                fig.update_layout(**dark_layout("Target Split", 380))
                st.plotly_chart(fig, use_container_width=True, key="ov_target")
            except Exception as e:
                st.error(f"Target chart error: {e}")

        # Row 2: Temp + Precipitation
        col1, col2 = st.columns(2)

        with col1:
            try:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=clean_series(df["Temp"]),
                    marker_color="#f59e0b",
                    marker_line=dict(color="#78350f", width=1),
                    opacity=0.85,
                    autobinx=True,
                ))
                fig.update_layout(**dark_layout("Temperature Distribution", 420))
                fig.update_xaxes(title_text="Temperature", **dark_axes())
                fig.update_yaxes(title_text="Count", **dark_axes())
                st.plotly_chart(fig, use_container_width=True, key="ov_temp")
            except Exception as e:
                st.error(f"Temperature chart error: {e}")

        with col2:
            try:
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=clean_series(df["preci"]),
                    marker_color="#22c55e",
                    marker_line=dict(color="#14532d", width=1),
                    opacity=0.85,
                    autobinx=True,
                ))
                fig.update_layout(**dark_layout("Precipitation Distribution", 420))
                fig.update_xaxes(title_text="Precipitation", **dark_axes())
                fig.update_yaxes(title_text="Count", **dark_axes())
                st.plotly_chart(fig, use_container_width=True, key="ov_preci")
            except Exception as e:
                st.error(f"Precipitation chart error: {e}")

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""**Model Architecture**
- Stacking: RF + GradBoost + AdaBoost → LogReg
- Temporal Lags: 1, 2, 4 weeks | Rolling stats: 4-week
- Metric: F2-Score & PR-AUC""")
        with c2:
            st.markdown("""**Key Findings**
- Past cases (lag 1-4) are top predictors
- Temp × Precipitation = key env. driver
- Coastal districts = elevated risk""")

    # ════════════════════════════════════════════════════════
    #  DATA ANALYSIS
    # ════════════════════════════════════════════════════════
    elif "Data" in page:
        st.title("Exploratory Data Analysis")
        st.write(f"**{len(df):,}** rows × **{df.shape[1]}** columns")

        view = st.selectbox("Choose Analysis", [
            "📈 Distributions", "🔀 Correlations", "🌡️ Temporal",
            "🗺️ Geographic", "📦 Box Plots", "📊 Scatter",
            "📉 Trends",
        ], key="da_view")

        # ─── DISTRIBUTIONS ──────────────────────────────────
        if view == "📈 Distributions":
            num_cols = [c for c in df.select_dtypes(include="number").columns
                        if c not in ["year", "mon", "day", "Target"]]
            if not num_cols:
                st.warning("No numeric columns found.")
            else:
                sel = st.multiselect("Pick features", num_cols,
                                      default=num_cols[:4] if len(num_cols) >= 4 else num_cols,
                                      key="da_dist")
                if sel:
                    try:
                        rows = (len(sel) + 1) // 2
                        fig = make_subplots(rows=rows, cols=2,
                                            subplot_titles=sel)
                        colors = ["#3b82f6", "#8b5cf6", "#ef4444", "#22c55e", "#f59e0b", "#0ea5e9"]
                        for i, col in enumerate(sel):
                            if col == "Cases":
                                vc = df["Cases"].replace([np.inf, -np.inf], np.nan).dropna().value_counts().sort_index().head(40)
                                fig.add_trace(go.Bar(
                                    x=vc.index.tolist(),
                                    y=vc.values.tolist(),
                                    name=col, showlegend=False,
                                    marker_color=colors[i % len(colors)],
                                    marker_line=dict(color="#111827", width=1),
                                    opacity=0.85,
                                ), row=i // 2 + 1, col=i % 2 + 1)
                            else:
                                fig.add_trace(go.Histogram(
                                    x=clean_series(df[col]),
                                    name=col, showlegend=False,
                                    marker_color=colors[i % len(colors)],
                                    marker_line=dict(color="#111827", width=1),
                                    opacity=0.85,
                                ), row=i // 2 + 1, col=i % 2 + 1)
                        fig.update_layout(
                            **dark_layout(height=250 * rows),
                            bargap=0.06,
                        )
                        fig.update_xaxes(**dark_axes())
                        fig.update_yaxes(**dark_axes())
                        st.plotly_chart(fig, use_container_width=True, key="da_hist")
                    except Exception as e:
                        st.error(f"Histogram error: {e}")

            # Target pie
            try:
                tc = df["Target"].value_counts()
                fig = go.Figure()
                fig.add_trace(go.Pie(
                    labels=["Below Median", "Above Median"],
                    values=[int(tc.get(0, 0)), int(tc.get(1, 0))],
                    marker=dict(colors=["#3b82f6", "#ef4444"], line=dict(color="#111827", width=3)),
                    hole=0.6, textinfo="label+percent+value",
                    textfont=dict(color="#e2e8f0", size=13),
                ))
                fig.update_layout(**dark_layout("Target Distribution", 400))
                st.plotly_chart(fig, use_container_width=True, key="da_pie")
            except Exception as e:
                st.error(f"Target pie error: {e}")

        # ─── CORRELATIONS ───────────────────────────────────
        elif view == "🔀 Correlations":
            cc = [c for c in df.select_dtypes(include="number").columns
                  if c not in ["year", "mon", "day"]]
            if len(cc) < 2:
                st.warning("Need at least 2 numeric columns.")
            else:
                cc = cc[:15]
                try:
                    corr = df[cc].corr().round(2)
                    fig = go.Figure()
                    fig.add_trace(go.Heatmap(
                        z=corr.values.tolist(),
                        x=corr.columns.tolist(),
                        y=corr.columns.tolist(),
                        colorscale=[[0, "#3b82f6"], [0.5, "#111827"], [1, "#ef4444"]],
                        zmin=-1, zmax=1,
                        text=[[f"{v:.2f}" for v in row] for row in corr.values.tolist()],
                        texttemplate="%{text}",
                        textfont=dict(size=9, color="#e2e8f0"),
                        colorbar=dict(thickness=12),
                    ))
                    h = max(500, len(cc) * 32)
                    fig.update_layout(**dark_layout("Correlation Heatmap", h))
                    fig.update_xaxes(tickangle=-45, **dark_axes())
                    fig.update_yaxes(**dark_axes())
                    st.plotly_chart(fig, use_container_width=True, key="da_corr")
                except Exception as e:
                    st.error(f"Heatmap error: {e}")

                if "Target" in cc:
                    try:
                        tc = corr["Target"].drop("Target", errors="ignore").abs().sort_values(ascending=False).head(10)
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=tc.values.tolist(),
                            y=tc.index.tolist(),
                            orientation="h",
                            marker_color=["#ef4444" if v > 0.3 else "#3b82f6" for v in tc.values],
                        ))
                        fig.update_layout(**dark_layout("|Correlation with Target|", 350))
                        fig.update_yaxes(autorange="reversed")
                        fig.update_xaxes(title_text="|r|", **dark_axes())
                        fig.update_yaxes(**dark_axes())
                        st.plotly_chart(fig, use_container_width=True, key="da_corrtarget")
                    except Exception as e:
                        st.error(f"Target corr error: {e}")

        # ─── TEMPORAL ───────────────────────────────────────
        elif view == "🌡️ Temporal":
            if "date" not in df.columns:
                st.info("No date column available.")
            else:
                try:
                    ts = df.groupby("date").agg(
                        Cases=("Cases", "sum"), Temp=("Temp", "mean"), Preci=("preci", "mean"),
                    ).reset_index()
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                        subplot_titles=["Total Cases", "Avg Temperature", "Avg Precipitation"])
                    fig.add_trace(go.Scatter(
                        x=ts["date"].tolist(), y=ts["Cases"].tolist(), mode="lines",
                        line=dict(color="#ef4444", width=2),
                        fill="tozeroy", fillcolor="rgba(239,68,68,0.08)"), row=1, col=1)
                    fig.add_trace(go.Scatter(
                        x=ts["date"].tolist(), y=ts["Temp"].tolist(), mode="lines",
                        line=dict(color="#f59e0b", width=2),
                        fill="tozeroy", fillcolor="rgba(245,158,11,0.08)"), row=2, col=1)
                    fig.add_trace(go.Scatter(
                        x=ts["date"].tolist(), y=ts["Preci"].tolist(), mode="lines",
                        line=dict(color="#3b82f6", width=2),
                        fill="tozeroy", fillcolor="rgba(59,130,246,0.08)"), row=3, col=1)
                    fig.update_layout(
                        height=650, showlegend=False,
                        paper_bgcolor="#111827", plot_bgcolor="#1e293b",
                        font=dict(color="#94a3b8", size=10),
                        margin=dict(l=45, r=15, t=55, b=25),
                    )
                    fig.update_xaxes(**dark_axes())
                    fig.update_yaxes(**dark_axes())
                    st.plotly_chart(fig, use_container_width=True, key="da_temporal")
                except Exception as e:
                    st.error(f"Time series error: {e}")

                if "mon" in df.columns:
                    try:
                        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                        ml = df.groupby("mon").agg(Cases=("Cases", "mean"), Temp=("Temp", "mean")).reset_index()
                        ml["Month"] = ml["mon"].map({i + 1: months[i] for i in range(12)})
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        fig.add_trace(go.Bar(
                            x=ml["Month"].tolist(), y=ml["Cases"].tolist(),
                            name="Mean Cases", marker_color="#3b82f6"), secondary_y=False)
                        fig.add_trace(go.Scatter(
                            x=ml["Month"].tolist(), y=ml["Temp"].tolist(),
                            name="Mean Temp", mode="lines+markers",
                            line=dict(color="#f59e0b", width=2)), secondary_y=True)
                        fig.update_layout(
                            title=dict(text="Monthly: Cases vs Temperature", font=dict(color="#e2e8f0", size=14)),
                            height=380, paper_bgcolor="#111827", plot_bgcolor="#1e293b",
                            font=dict(color="#94a3b8", size=11),
                            legend=dict(font=dict(color="#94a3b8"), orientation="h", y=1.12),
                            margin=dict(l=50, r=25, t=45, b=45),
                        )
                        fig.update_xaxes(**dark_axes())
                        fig.update_yaxes(**dark_axes(), secondary_y=False, title_text="Cases")
                        fig.update_yaxes(**dark_axes(), secondary_y=True, title_text="Temp")
                        st.plotly_chart(fig, use_container_width=True, key="da_monthly")
                    except Exception as e:
                        st.error(f"Monthly error: {e}")

        # ─── GEOGRAPHIC ─────────────────────────────────────
        elif view == "🗺️ Geographic":
            col1, col2 = st.columns(2)
            with col1:
                if "state_ut" in df.columns:
                    try:
                        sd = df.groupby("state_ut")["Cases"].sum().sort_values(ascending=False).head(15).reset_index()
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=sd["Cases"].tolist(), y=sd["state_ut"].tolist(),
                            orientation="h",
                            marker=dict(color=sd["Cases"].tolist(),
                                        colorscale=[[0, "#3b82f6"], [1, "#ef4444"]]),
                        ))
                        fig.update_layout(**dark_layout("Cases by State (Top 15)", 480))
                        fig.update_yaxes(autorange="reversed")
                        fig.update_xaxes(**dark_axes())
                        fig.update_yaxes(**dark_axes())
                        st.plotly_chart(fig, use_container_width=True, key="da_state")
                    except Exception as e:
                        st.error(f"State error: {e}")
                else:
                    st.info("No 'state_ut' column.")

            with col2:
                if "district" in df.columns:
                    try:
                        dd = df.groupby("district")["Cases"].sum().sort_values(ascending=False).head(15).reset_index()
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=dd["Cases"].tolist(), y=dd["district"].tolist(),
                            orientation="h",
                            marker=dict(color=dd["Cases"].tolist(), colorscale="Reds"),
                        ))
                        fig.update_layout(**dark_layout("Cases by District (Top 15)", 480))
                        fig.update_yaxes(autorange="reversed")
                        fig.update_xaxes(**dark_axes())
                        fig.update_yaxes(**dark_axes())
                        st.plotly_chart(fig, use_container_width=True, key="da_district")
                    except Exception as e:
                        st.error(f"District error: {e}")
                else:
                    st.info("No 'district' column.")

            if "is_coastal_district" in df.columns:
                try:
                    cd = df.groupby("is_coastal_district")["Cases"].mean().reset_index()
                    cd["Type"] = cd["is_coastal_district"].map({0: "Inland", 1: "Coastal"})
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=cd["Type"].tolist(), y=cd["Cases"].tolist(),
                        marker=dict(color=["#3b82f6", "#f59e0b"]),
                        text=[f"{v:.1f}" for v in cd["Cases"].tolist()],
                        textposition="outside", textfont=dict(color="#e2e8f0"),
                    ))
                    fig.update_layout(**dark_layout("Mean Cases: Coastal vs Inland", 320))
                    fig.update_yaxes(title_text="Mean Cases", **dark_axes())
                    fig.update_xaxes(**dark_axes())
                    st.plotly_chart(fig, use_container_width=True, key="da_coastal")
                except Exception as e:
                    st.error(f"Coastal error: {e}")

            if "Latitude" in df.columns and "Longitude" in df.columns:
                try:
                    gd = df.groupby(["district", "Latitude", "Longitude"]).agg(
                        Total=("Cases", "sum")).reset_index().head(100)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=gd["Longitude"].tolist(), y=gd["Latitude"].tolist(),
                        mode="markers",
                        marker=dict(
                            size=10, color=gd["Total"].tolist(),
                            colorscale=[[0, "#3b82f6"], [1, "#ef4444"]],
                            opacity=0.7, colorbar=dict(thickness=12),
                        ),
                        text=[f"{d}<br>{t} cases" for d, t in zip(gd["district"].tolist(), gd["Total"].tolist())],
                        hovertemplate="%{text}<extra></extra>",
                    ))
                    fig.update_layout(**dark_layout("Geographic Spread", 400))
                    fig.update_xaxes(title_text="Longitude", **dark_axes())
                    fig.update_yaxes(title_text="Latitude", **dark_axes())
                    st.plotly_chart(fig, use_container_width=True, key="da_geo")
                except Exception as e:
                    st.error(f"Map error: {e}")

        # ─── BOX PLOTS ──────────────────────────────────────
        elif view == "📦 Box Plots":
            bc = [c for c in ["Cases", "Temp", "preci", "Fatality_Rate", "LAI"] if c in df.columns]
            if not bc:
                st.warning("None of the standard columns found.")
            else:
                sel = st.multiselect("Features", bc, default=bc[:3], key="da_box")
                if sel:
                    try:
                        rows = (len(sel) + 1) // 2
                        fig = make_subplots(rows=rows, cols=2, subplot_titles=sel)
                        for i, col in enumerate(sel):
                            for t, clr, nm in [(0, "#3b82f6", "Below"), (1, "#ef4444", "Above")]:
                                vals = df[df["Target"] == t][col]
                                vals = vals.replace([np.inf, -np.inf], np.nan).dropna().tolist()
                                fig.add_trace(go.Box(
                                    y=vals, name=nm,
                                    legendgroup=nm, showlegend=(i == 0),
                                    marker_color=clr,
                                ), row=i // 2 + 1, col=i % 2 + 1)
                        fig.update_layout(
                            height=260 * rows,
                            paper_bgcolor="#111827", plot_bgcolor="#1e293b",
                            font=dict(color="#94a3b8", size=10),
                            legend=dict(font=dict(color="#94a3b8")),
                            margin=dict(l=45, r=15, t=55, b=25),
                        )
                        fig.update_xaxes(**dark_axes())
                        fig.update_yaxes(**dark_axes())
                        st.plotly_chart(fig, use_container_width=True, key="da_boxplot")
                    except Exception as e:
                        st.error(f"Box plot error: {e}")

        # ─── SCATTER ────────────────────────────────────────
        elif view == "📊 Scatter":
            nc = [c for c in df.select_dtypes(include="number").columns
                  if c not in ["year", "mon", "day", "Target"]]
            if len(nc) < 2:
                st.warning("Need at least 2 numeric columns.")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    xi = nc.index("Temp") if "Temp" in nc else 0
                    xc = st.selectbox("X axis", nc, index=xi, key="da_sx")
                with c2:
                    yi = nc.index("preci") if "preci" in nc else (1 if len(nc) > 1 else 0)
                    yc = st.selectbox("Y axis", nc, index=yi, key="da_sy")
                if xc == yc:
                    st.info("Pick different axes.")
                else:
                    try:
                        sample = df[[xc, yc, "Target"]].dropna().head(800)
                        fig = go.Figure()
                        for t, clr, nm in [(0, "#3b82f6", "Below Median"), (1, "#ef4444", "Above Median")]:
                            sub = sample[sample["Target"] == t]
                            fig.add_trace(go.Scatter(
                                x=sub[xc].tolist(), y=sub[yc].tolist(),
                                mode="markers", name=nm,
                                marker=dict(color=clr, size=5, opacity=0.6),
                            ))
                        fig.update_layout(**dark_layout(f"{xc} vs {yc}", 480))
                        fig.update_xaxes(title_text=xc, **dark_axes())
                        fig.update_yaxes(title_text=yc, **dark_axes())
                        fig.update_layout(legend=dict(font=dict(color="#94a3b8")))
                        st.plotly_chart(fig, use_container_width=True, key="da_scatter")
                    except Exception as e:
                        st.error(f"Scatter error: {e}")

        # ─── TRENDS ─────────────────────────────────────────
        elif view == "📉 Trends":
            if "year" in df.columns:
                try:
                    yl = df.groupby("year").agg(
                        Total=("Cases", "sum"), Mean=("Cases", "mean"),
                        Max=("Cases", "max"), Count=("Cases", "count"),
                    ).reset_index()
                    fig = make_subplots(rows=2, cols=2,
                                        subplot_titles=["Total Cases", "Mean Cases", "Max Cases", "Record Count"])
                    fig.add_trace(go.Bar(x=yl["year"].tolist(), y=yl["Total"].tolist(),
                                         marker_color="#3b82f6", showlegend=False), row=1, col=1)
                    fig.add_trace(go.Bar(x=yl["year"].tolist(), y=yl["Mean"].tolist(),
                                         marker_color="#8b5cf6", showlegend=False), row=1, col=2)
                    fig.add_trace(go.Bar(x=yl["year"].tolist(), y=yl["Max"].tolist(),
                                         marker_color="#ef4444", showlegend=False), row=2, col=1)
                    fig.add_trace(go.Bar(x=yl["year"].tolist(), y=yl["Count"].tolist(),
                                         marker_color="#22c55e", showlegend=False), row=2, col=2)
                    fig.update_layout(
                        height=500, showlegend=False,
                        paper_bgcolor="#111827", plot_bgcolor="#1e293b",
                        font=dict(color="#94a3b8", size=10),
                        margin=dict(l=40, r=15, t=55, b=25),
                    )
                    fig.update_xaxes(**dark_axes())
                    fig.update_yaxes(**dark_axes())
                    st.plotly_chart(fig, use_container_width=True, key="da_yearly")
                except Exception as e:
                    st.error(f"Yearly error: {e}")

            if "Season" in df.columns:
                try:
                    sl = df.groupby("Season")["Cases"].mean().reset_index()
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=sl["Season"].tolist(), y=sl["Cases"].tolist(),
                        marker=dict(color=sl["Cases"].tolist(), colorscale="Viridis"),
                        text=[f"{v:.1f}" for v in sl["Cases"].tolist()],
                        textposition="outside", textfont=dict(color="#e2e8f0"),
                    ))
                    fig.update_layout(**dark_layout("Mean Cases by Season", 350))
                    fig.update_yaxes(title_text="Mean Cases", **dark_axes())
                    fig.update_xaxes(**dark_axes())
                    st.plotly_chart(fig, use_container_width=True, key="da_season")
                except Exception as e:
                    st.error(f"Season error: {e}")

            if "date" in df.columns and "district" in df.columns:
                try:
                    top_d = df.groupby("district")["Cases"].sum().nlargest(5).index.tolist()
                    sel_d = st.multiselect("Districts", top_d, default=top_d[:3], key="da_td")
                    if sel_d:
                        fig = go.Figure()
                        colors = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6"]
                        for i, d in enumerate(sel_d):
                            dd = df[df["district"] == d].sort_values("date")
                            fig.add_trace(go.Scatter(
                                x=dd["date"].tolist(), y=dd["Cases"].tolist(),
                                mode="lines", name=d,
                                line=dict(color=colors[i % len(colors)], width=2),
                            ))
                        fig.update_layout(**dark_layout("Case Trends by District", 400))
                        fig.update_xaxes(title_text="Date", **dark_axes())
                        fig.update_yaxes(title_text="Cases", **dark_axes())
                        fig.update_layout(legend=dict(font=dict(color="#94a3b8")))
                        st.plotly_chart(fig, use_container_width=True, key="da_trend")
                except Exception as e:
                    st.error(f"Trend error: {e}")

    # ════════════════════════════════════════════════════════
    #  MODEL TRAINING
    # ════════════════════════════════════════════════════════
    elif "Model" in page:
        from joblib import parallel_backend
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.ensemble import (
            StackingClassifier, RandomForestClassifier,
            AdaBoostClassifier, HistGradientBoostingClassifier)
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            classification_report, confusion_matrix,
            fbeta_score, precision_recall_curve, auc)

        X, y, meta = engineer(df)
        st.title("Model Training & Evaluation")
        st.info("**Protocol:** Temporal 80/20 | Pipeline | Stacking (RF+GB+Ada→LR) | F2 & PR-AUC")

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Training", f"{int(len(X) * .8):,}")
        with c2: st.metric("Test", f"{len(X) - int(len(X) * .8):,}")
        with c3: st.metric("Features", f"{X.shape[1]}")

        if st.button("Train Stacking Ensemble", use_container_width=True):
            try:
                with st.spinner("Training (2-4 min)..."):
                    prog = st.progress(0, text="Splitting...")
                    si = int(len(X) * .8)
                    X_tr, X_te = X.iloc[:si], X.iloc[si:]
                    y_tr, y_te = y.iloc[:si], y.iloc[si:]
                    meta_te = meta.iloc[si:].reset_index(drop=True)
                    prog.progress(10, text="Preprocessor...")

                    ncol = X_tr.select_dtypes(include="number").columns.tolist()
                    ccol = X_tr.select_dtypes(include="object").columns.tolist()
                    preproc = ColumnTransformer([
                        ("n", Pipeline([("i", SimpleImputer(strategy="median")),
                                        ("s", StandardScaler())]), ncol),
                        ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                                        ("e", OneHotEncoder(handle_unknown="ignore"))]), ccol),
                    ])
                    prog.progress(25, text="Learners...")
                    ests = [
                        ("rf", RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=1)),
                        ("hgb", HistGradientBoostingClassifier(max_iter=80, max_depth=6, learning_rate=.1, random_state=42)),
                        ("ada", AdaBoostClassifier(n_estimators=50, learning_rate=.1, random_state=42)),
                    ]
                    clf = StackingClassifier(
                        estimators=ests,
                        final_estimator=LogisticRegression(C=.4, class_weight="balanced", max_iter=1000, n_jobs=1),
                        cv=3, n_jobs=1, passthrough=False)
                    pipe = Pipeline([("p", preproc), ("c", clf)])
                    prog.progress(50, text="Fitting...")
                    with parallel_backend("threading", n_jobs=1):
                        pipe.fit(X_tr, y_tr)
                    prog.progress(85, text="Metrics...")
                    yp = pipe.predict(X_te)
                    ypr = pipe.predict_proba(X_te)[:, 1]
                    f2 = fbeta_score(y_te, yp, beta=2)
                    pa, ra, _ = precision_recall_curve(y_te, ypr)
                    prauc = auc(ra, pa)
                    acc = pipe.score(X_te, y_te)
                    st.session_state.update({
                        "model": pipe, "X_test": X_te, "y_test": y_te,
                        "y_pred": yp, "y_proba": ypr, "meta_test": meta_te,
                        "metrics": {"F2": f2, "PR-AUC": prauc, "Accuracy": acc},
                    })
                    prog.progress(100, text="Done!")
                st.success("Done! Go to **Forecasting** for SHAP.")
            except Exception as exc:
                st.error(f"Failed: {exc}")
                import traceback
                st.code(traceback.format_exc())

        # ── Results ──
        if "model" in st.session_state and "metrics" in st.session_state:
            m = st.session_state["metrics"]
            st.markdown("---")
            k1, k2, k3 = st.columns(3)
            with k1: st.metric("F2-Score", f"{m['F2']:.4f}")
            with k2: st.metric("PR-AUC", f"{m['PR-AUC']:.4f}")
            with k3: st.metric("Accuracy", f"{m['Accuracy']:.2%}")

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            try:
                cm = confusion_matrix(
                    st.session_state["y_test"],
                    st.session_state["y_pred"]
                )
                fig = go.Figure()
                fig.add_trace(go.Heatmap(
                    z=cm.tolist(),
                    x=["No Outbreak", "Outbreak"],
                    y=["No Outbreak", "Outbreak"],
                    colorscale=[[0, "#1e3a8a"], [0.5, "#3b82f6"], [1, "#60a5fa"]],
                    text=[[str(v) for v in row] for row in cm.tolist()],
                    texttemplate="%{text}",
                    textfont=dict(size=16, color="#fff"),
                ))
                fig.update_layout(**dark_layout("Confusion Matrix", 400))
                fig.update_xaxes(title_text="Predicted", **dark_axes())
                fig.update_yaxes(title_text="Actual", **dark_axes())
                st.plotly_chart(fig, use_container_width=True, key="mt_cm")
            except Exception as e:
                st.error(f"Confusion matrix error: {e}")

            # PR Curve
            st.subheader("Precision-Recall Curve")
            try:
                pa2, ra2, _ = precision_recall_curve(
                    st.session_state["y_test"],
                    st.session_state["y_proba"]
                )
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ra2.tolist(), y=pa2.tolist(),
                    mode="lines",
                    line=dict(color="#8b5cf6", width=2.5),
                    fill="tozeroy", fillcolor="rgba(139,92,246,0.08)",
                    name=f"PR-AUC = {m['PR-AUC']:.4f}",
                ))
                fig.update_layout(**dark_layout("Precision-Recall Curve", 400))
                fig.update_xaxes(title_text="Recall", **dark_axes())
                fig.update_yaxes(title_text="Precision", **dark_axes())
                fig.update_layout(legend=dict(font=dict(color="#94a3b8")))
                st.plotly_chart(fig, use_container_width=True, key="mt_pr")
            except Exception as e:
                st.error(f"PR curve error: {e}")

            # Classification Report
            st.subheader("Classification Report")
            try:
                st.code(classification_report(
                    st.session_state["y_test"], st.session_state["y_pred"],
                    target_names=["No Outbreak", "Outbreak"]))
            except Exception as e:
                st.error(f"Report error: {e}")

    # ════════════════════════════════════════════════════════
    #  FORECASTING
    # ════════════════════════════════════════════════════════
    elif "Forecast" in page:
        if "model" not in st.session_state:
            st.warning("Train the model first.")
            st.stop()

        from joblib import parallel_backend
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        X, y, meta = engineer(df)
        st.title("Forecasting & Explainability")
        model = st.session_state["model"]
        mt = st.session_state["meta_test"]
        Xt = st.session_state["X_test"]
        prep = model.named_steps["p"]
        with parallel_backend("threading", n_jobs=1):
            Xp = prep.transform(Xt)
        try:
            rf = model.named_steps["c"].estimators_[0]
        except Exception:
            rf = None

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("District Prediction")
            if "district" in mt.columns and "date" in mt.columns:
                opts = [f"{r['district']} ({r['date'].date()})" for _, r in mt.iterrows()]
            else:
                opts = [f"Row {i}" for i in range(len(mt))]
            sel = st.selectbox("Select", range(len(opts)),
                               format_func=lambda i: opts[i], key="fc_sel")
            row = mt.iloc[sel]
            Xi = Xt.iloc[[sel]]
            with parallel_backend("threading", n_jobs=1):
                pred = model.predict(Xi)[0]
                prob = model.predict_proba(Xi)[0][1]
            if "Cases" in row.index:
                st.metric("Actual Cases", f"{int(row['Cases']):,}")

            try:
                if prob < 0.3:
                    gauge_color = "#22c55e"
                elif prob < 0.6:
                    gauge_color = "#f59e0b"
                else:
                    gauge_color = "#ef4444"

                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob * 100,
                    gauge=dict(
                        axis=dict(range=[0, 100], tickfont=dict(color="#64748b")),
                        bar=dict(color=gauge_color),
                        bgcolor="#1e293b",
                        steps=[
                            dict(range=[0, 30], color="rgba(34,197,94,0.15)"),
                            dict(range=[30, 60], color="rgba(245,158,11,0.15)"),
                            dict(range=[60, 100], color="rgba(239,68,68,0.15)"),
                        ],
                        threshold=dict(line=dict(color="#fff", width=2), value=50),
                    ),
                    number=dict(suffix="%", font=dict(size=32, color="#e2e8f0")),
                ))
                fig.update_layout(height=260, paper_bgcolor="#111827",
                                  margin=dict(l=25, r=25, t=25, b=10))
                st.plotly_chart(fig, use_container_width=True, key="fc_gauge")
            except Exception as e:
                st.write(f"Probability: **{prob:.1%}**")

            if prob >= 0.6:
                st.error("🔴 **HIGH RISK** — Outbreak Predicted")
            elif prob >= 0.3:
                st.warning("🟠 **MEDIUM RISK** — Monitor Closely")
            else:
                st.success("🟢 **LOW RISK** — No Outbreak")

        with c2:
            st.subheader("SHAP Explanation")
            try:
                import shap
            except ImportError:
                st.warning("Install: `pip install shap`")
                shap = None

            if shap is not None and rf is not None:
                try:
                    ex = shap.TreeExplainer(rf, feature_perturbation="tree_path_dependent")
                    sv = ex.shap_values(Xp[sel:sel + 1], check_additivity=False)
                    if isinstance(sv, list):
                        s = sv[1][0] if len(sv) > 1 else sv[0][0]
                    elif isinstance(sv, np.ndarray):
                        s = sv[0, :, 1] if sv.ndim == 3 else (sv[0] if sv.ndim == 2 else sv)
                    else:
                        s = np.array(sv).ravel()
                    try:
                        fn = list(prep.get_feature_names_out())
                    except Exception:
                        fn = [f"f{i}" for i in range(len(s))]
                    ml = min(len(s), len(fn))
                    s, fn = s[:ml], fn[:ml]
                    if ml > 0:
                        tn = min(15, ml)
                        ix = np.argsort(np.abs(s))[-tn:]
                        v = s[ix]
                        names = [str(fn[i]) for i in ix]
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            y=names, x=v.tolist(), orientation="h",
                            marker=dict(
                                color=["rgba(239,68,68,0.8)" if x > 0 else "rgba(59,130,246,0.8)" for x in v],
                                line=dict(color=["#ef4444" if x > 0 else "#3b82f6" for x in v], width=1),
                            ),
                        ))
                        fig.add_vline(x=0, line_dash="dash", line_color="#475569")
                        fig.update_layout(**dark_layout("Feature Impact (🔴 Risk+ | 🔵 Risk-)", 480))
                        fig.update_yaxes(autorange="reversed")
                        fig.update_xaxes(title_text="SHAP Value", **dark_axes())
                        fig.update_yaxes(**dark_axes())
                        st.plotly_chart(fig, use_container_width=True, key="fc_shap")
                except Exception as e:
                    st.error(f"SHAP error: {e}")
                    import traceback
                    st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
