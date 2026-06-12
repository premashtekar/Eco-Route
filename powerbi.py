"""
VANET / 5G Dashboard (Streamlit + Plotly)
Usage: streamlit run vanet_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------------
# Configuration
# ---------------------
DATA_PATH = "CCR_5G_VANETs.csv"  # change if needed

PALETTE = {
    "primary": "#1F77B4",   # deep blue
    "secondary": "#2CA02C", # green
    "danger": "#D62728",    # red
    "accent": "#9467BD",    # purple
    "bg": "#F7F7F7"
}

st.set_page_config(layout="wide", page_title="VANET / 5G Dashboard")

# ---------------------
# Helpers
# ---------------------
@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    numeric_cols = [
        "Packet_Size_KB", "Distance_to_RSU_m", "Energy_Consumption_J",
        "Carbon_Emission_gCO2", "Traffic_Density_Vehicles_per_km",
        "Speed_kmph", "Acceleration_mps2", "RSU_Power_kW", "Optimal_Route_Score"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df = load_data()

st.title("VANET / 5G Performance & Sustainability Dashboard")
st.markdown("Use the sidebar filters to slice the dataset interactively.")

# ---------------------
# Sidebar filters
# ---------------------
st.sidebar.header("Filters")

regions = st.sidebar.multiselect(
    "Region Type",
    options=sorted(df["Region_Type"].dropna().unique()) if "Region_Type" in df.columns else [],
    default=sorted(df["Region_Type"].dropna().unique()) if "Region_Type" in df.columns else []
)

vtypes = st.sidebar.multiselect(
    "Vehicle Type",
    options=sorted(df["Vehicle_Type"].dropna().unique()) if "Vehicle_Type" in df.columns else [],
    default=sorted(df["Vehicle_Type"].dropna().unique()) if "Vehicle_Type" in df.columns else []
)

def numeric_range_slider(df, col, label):
    if col in df.columns:
        mn, mx = float(df[col].min()), float(df[col].max())
        return st.sidebar.slider(label, min_value=mn, max_value=mx, value=(mn, mx))
    return None

dist_range = numeric_range_slider(df, "Distance_to_RSU_m", "Distance to RSU (m)")
traffic_range = numeric_range_slider(df, "Traffic_Density_Vehicles_per_km", "Traffic density (vehicles/km)")

# Apply filters
df_filtered = df.copy()
if regions:
    df_filtered = df_filtered[df_filtered["Region_Type"].isin(regions)]
if vtypes:
    df_filtered = df_filtered[df_filtered["Vehicle_Type"].isin(vtypes)]
if dist_range:
    df_filtered = df_filtered[
        (df_filtered["Distance_to_RSU_m"] >= dist_range[0]) &
        (df_filtered["Distance_to_RSU_m"] <= dist_range[1])
    ]
if traffic_range:
    df_filtered = df_filtered[
        (df_filtered["Traffic_Density_Vehicles_per_km"] >= traffic_range[0]) &
        (df_filtered["Traffic_Density_Vehicles_per_km"] <= traffic_range[1])
    ]

# ---------------------
# KPIs
# ---------------------
kpi1 = df_filtered["Energy_Consumption_J"].mean() if "Energy_Consumption_J" in df_filtered else np.nan
kpi2 = df_filtered["Carbon_Emission_gCO2"].sum() if "Carbon_Emission_gCO2" in df_filtered else np.nan
kpi3 = df_filtered["Packet_Size_KB"].mean() if "Packet_Size_KB" in df_filtered else np.nan
kpi4 = df_filtered["Distance_to_RSU_m"].mean() if "Distance_to_RSU_m" in df_filtered else np.nan

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Energy (J)", f"{kpi1:,.2f}")
col2.metric("Total Carbon (gCO2)", f"{kpi2:,.2f}")
col3.metric("Avg Packet Size (KB)", f"{kpi3:,.2f}")
col4.metric("Avg Distance to RSU (m)", f"{kpi4:,.2f}")

st.markdown("---")

# ---------------------
# Trend & Distribution
# ---------------------
r1c1, r1c2 = st.columns((2,1))

# Trend chart
if "Distance_to_RSU_m" in df_filtered:
    bins = np.linspace(df_filtered["Distance_to_RSU_m"].min(), df_filtered["Distance_to_RSU_m"].max(), 25)
    df_filtered["dist_bin"] = pd.cut(df_filtered["Distance_to_RSU_m"], bins)
    trend = df_filtered.groupby("dist_bin").agg({"Energy_Consumption_J":"mean"}).reset_index()
    trend["bin_mid"] = trend["dist_bin"].apply(lambda x: x.mid if pd.notnull(x) else np.nan)
    fig_trend = px.line(trend, x="bin_mid", y="Energy_Consumption_J",
                        title="Avg Energy vs Distance to RSU (binned)", markers=True)
    r1c1.plotly_chart(fig_trend, use_container_width=True)

# Distribution
if "Speed_kmph" in df_filtered:
    fig_hist = px.histogram(df_filtered, x="Speed_kmph", nbins=40, title="Speed Distribution", marginal="box")
    r1c2.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# ---------------------
# Comparison Charts
# ---------------------
c1, c2, c3 = st.columns(3)

# Avg Energy by Region & Vehicle Type
if all(x in df_filtered.columns for x in ["Region_Type","Vehicle_Type","Energy_Consumption_J"]):
    agg = df_filtered.groupby(["Region_Type","Vehicle_Type"]).agg({"Energy_Consumption_J":"mean"}).reset_index()
    fig_bar = px.bar(agg, x="Region_Type", y="Energy_Consumption_J", color="Vehicle_Type",
                     barmode="group", title="Avg Energy by Region & Vehicle Type")
    c1.plotly_chart(fig_bar, use_container_width=True)

# Avg Carbon by Region
if all(x in df_filtered.columns for x in ["Region_Type","Carbon_Emission_gCO2"]):
    agg2 = df_filtered.groupby("Region_Type").agg({"Carbon_Emission_gCO2":"mean"}).reset_index()
    fig_carbon = px.bar(agg2, x="Region_Type", y="Carbon_Emission_gCO2",
                        title="Avg Carbon Emission by Region", color_discrete_sequence=[PALETTE["accent"]])
    c2.plotly_chart(fig_carbon, use_container_width=True)

# Packet Size vs Energy scatter
if all(x in df_filtered.columns for x in ["Packet_Size_KB","Energy_Consumption_J","Traffic_Density_Vehicles_per_km"]):
    fig_scatter = px.scatter(
        df_filtered, x="Packet_Size_KB", y="Energy_Consumption_J",
        size="Traffic_Density_Vehicles_per_km", color="Region_Type",
        hover_data=["Vehicle_ID"], title="Packet Size vs Energy (bubble size = traffic density)"
    )
    c3.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# ---------------------
# Route vs RSU distance + table
# ---------------------
n1, n2 = st.columns((2,1))

if all(x in df_filtered.columns for x in ["Distance_to_RSU_m","Optimal_Route_Score"]):
    fig_route = px.scatter(df_filtered, x="Distance_to_RSU_m", y="Optimal_Route_Score",
                           color="Region_Type", title="Distance to RSU vs Optimal Route Score",
                           hover_data=["Vehicle_ID"])
    n1.plotly_chart(fig_route, use_container_width=True)

show_cols = [c for c in ["Vehicle_ID","Vehicle_Type","Region_Type","Speed_kmph","Energy_Consumption_J","Carbon_Emission_gCO2"] if c in df_filtered]
n2.write("Sample Data (Top 50 Rows):")
n2.dataframe(df_filtered[show_cols].head(50))

st.markdown("---")

# ---------------------
# Correlation heatmap
# ---------------------
st.subheader("Correlation Matrix (Numeric Columns)")
numeric = df_filtered.select_dtypes(include=np.number)
if numeric.shape[1] >= 2:
    corr = numeric.corr()
    fig_heat = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation matrix")
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.write("Not enough numeric columns for correlation matrix.")

# ---------------------
# Footer
# ---------------------
st.markdown("---")
st.write(f"Dataset rows: {len(df)} | After filters: {len(df_filtered)}")

csv = df_filtered.to_csv(index=False).encode("utf-8")
st.download_button("Download Filtered CSV", data=csv, file_name="Filtered_VANETs.csv", mime="text/csv")

st.caption("Dashboard automatically generated with Streamlit + Plotly.")
