import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Data Visualizer",
    layout="wide"
)

st.title("Smart Data Visualization Dashboard")

# ==================================================
# SIDEBAR — CONTROL PANEL
# ==================================================
st.sidebar.header("Controls")

uploaded_files = st.sidebar.file_uploader(
    "Upload CSV files",
    type=["csv"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload CSV files to start exploring data.")
    st.stop()

file_names = [f.name for f in uploaded_files]
selected_file = st.sidebar.selectbox("Select Dataset", file_names)

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

df = load_csv(next(f for f in uploaded_files if f.name == selected_file))

# --------------------------------------------------
# Data Cleaning
# --------------------------------------------------
st.sidebar.divider()
st.sidebar.subheader("Data Cleaning")

missing_option = st.sidebar.selectbox(
    "Missing Values",
    ["Do nothing", "Drop rows", "Fill numeric with mean", "Fill numeric with median"]
)

if missing_option == "Drop rows":
    df = df.dropna()

elif missing_option == "Fill numeric with mean":
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

elif missing_option == "Fill numeric with median":
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# ==================================================
# KPI CARDS (Power BI style)
# ==================================================
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows", f"{df.shape[0]:,}")
k2.metric("Columns", df.shape[1])
k3.metric("Numeric Columns", len(numeric_cols))
k4.metric("Missing Values", int(df.isnull().sum().sum()))

# ==================================================
# TABS
# ==================================================
tab1, tab2, tab3 = st.tabs(
    ["Overview", "Visual Analysis", "Advanced Analysis"]
)

# ==================================================
# TAB 1 — OVERVIEW
# ==================================================
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), width="stretch")

# ==================================================
# TAB 2 — VISUAL ANALYSIS
# ==================================================
with tab2:
    st.subheader("Interactive Visuals")

    plot_type = st.selectbox(
        "Select Visualization",
        [
            "Line Plot",
            "Bar Chart",
            "Scatter Plot",
            "Histogram",
            "Count Plot",
            "Correlation Heatmap"
        ]
    )

    fig = None

    if plot_type in ["Line Plot", "Bar Chart", "Scatter Plot"]:
        c1, c2 = st.columns(2)
        x = c1.selectbox("X-axis", df.columns)
        y = c2.selectbox("Y-axis", numeric_cols)

        if plot_type == "Line Plot":
            fig = px.line(df, x=x, y=y)
        elif plot_type == "Bar Chart":
            fig = px.bar(df, x=x, y=y)
        else:
            fig = px.scatter(df, x=x, y=y)

    elif plot_type == "Histogram":
        x = st.selectbox("Numeric Column", numeric_cols)
        fig = px.histogram(df, x=x)

    elif plot_type == "Count Plot":
        x = st.selectbox("Categorical Column", categorical_cols)
        fig = px.histogram(df, x=x)

    elif plot_type == "Correlation Heatmap":
        corr = df[numeric_cols].corr()
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r"
        )

    if fig is not None:
        st.plotly_chart(
            fig,
            width="stretch",
            config={
                "displaylogo": False,
                "modeBarButtonsToRemove": ["toImage"]
            }
        )

# ==================================================
# TAB 3 — ADVANCED ANALYSIS (NaN-safe PCA)
# ==================================================
with tab3:
    st.subheader("Advanced Analytics")

    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for PCA.")
    else:
        st.info("Missing values are automatically handled for PCA (mean imputation).")

        # ---- PCA pipeline (SAFE) ----
        X = df[numeric_cols]

        imputer = SimpleImputer(strategy="mean")
        scaler = StandardScaler()
        pca = PCA(n_components=2)

        X_imputed = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_imputed)
        components = pca.fit_transform(X_scaled)

        pca_df = pd.DataFrame(
            components,
            columns=["PC1", "PC2"]
        )

        fig = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            title="PCA Projection (2D)"
        )

        st.plotly_chart(
            fig,
            width="stretch",
            config={
                "displaylogo": False,
                "modeBarButtonsToRemove": ["toImage"]
            }
        )
