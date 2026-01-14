import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Data Visualizer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Smart Data Visualization Dashboard")

# --------------------------------------------------
# Sidebar: File Upload
# --------------------------------------------------
st.sidebar.header("📁 Upload Datasets")

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more CSV files",
    type=["csv"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload CSV files from the sidebar to begin.")
    st.stop()

# --------------------------------------------------
# Load selected dataset
# --------------------------------------------------
file_names = [file.name for file in uploaded_files]
selected_file = st.sidebar.selectbox("Select a dataset", file_names)

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

df = load_csv(
    next(file for file in uploaded_files if file.name == selected_file)
)

# --------------------------------------------------
# Sidebar: Data Cleaning
# --------------------------------------------------
st.sidebar.header("🧹 Data Cleaning")

missing_option = st.sidebar.selectbox(
    "Handle missing values",
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

# --------------------------------------------------
# Dataset Overview
# --------------------------------------------------
st.subheader("📄 Dataset Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing Values", int(df.isnull().sum().sum()))

st.dataframe(df.head(), width="stretch")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# --------------------------------------------------
# Auto-detect best plot
# --------------------------------------------------
st.subheader("🤖 Auto-Detected Insight")

if len(numeric_cols) >= 2:
    auto_plot = "Correlation Heatmap"
elif len(categorical_cols) >= 1:
    auto_plot = "Count Plot"
elif len(numeric_cols) == 1:
    auto_plot = "Histogram"
else:
    auto_plot = "No suitable plot detected"

st.info(f"Suggested visualization: **{auto_plot}**")

# --------------------------------------------------
# Visualization Section
# --------------------------------------------------
st.subheader("📈 Visualization")

plot_type = st.selectbox(
    "Select Plot Type",
    [
        "Line Plot",
        "Bar Chart",
        "Scatter Plot",
        "Histogram",
        "Count Plot",
        "Correlation Heatmap",
        "PCA Plot (sklearn)"
    ]
)

fig = None

if plot_type in ["Line Plot", "Bar Chart", "Scatter Plot"]:
    x = st.selectbox("X-axis", df.columns)
    y = st.selectbox("Y-axis", numeric_cols)

    if plot_type == "Line Plot":
        fig = px.line(df, x=x, y=y)
    elif plot_type == "Bar Chart":
        fig = px.bar(df, x=x, y=y)
    else:
        fig = px.scatter(df, x=x, y=y)

elif plot_type == "Histogram":
    x = st.selectbox("Column", numeric_cols)
    fig = px.histogram(df, x=x)

elif plot_type == "Count Plot":
    x = st.selectbox("Column", categorical_cols)
    fig = px.histogram(df, x=x)

elif plot_type == "Correlation Heatmap":
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap"
    )

elif plot_type == "PCA Plot (sklearn)":
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for PCA.")
    else:
        scaled = StandardScaler().fit_transform(df[numeric_cols])
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled)

        pca_df = pd.DataFrame(pca_data, columns=["PC1", "PC2"])
        fig = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            title="PCA Projection (2D)"
        )

# --------------------------------------------------
# Display Plot
# --------------------------------------------------
if fig is not None:
    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displaylogo": False,
            "modeBarButtonsToRemove": ["toImage"]
        }
    )
