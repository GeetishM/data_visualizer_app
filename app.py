import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="Data Visualizer",
    layout="centered",
    page_icon="📊"
)

st.title("📊 Data Visualization Web App")

# Upload file
uploaded_file = st.file_uploader(
    "Upload your dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    # Read dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Show dataset info
    st.subheader("Dataset Info")
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    # Missing values summary
    st.subheader("Missing Values")
    missing_data = df.isnull().sum()
    st.write(missing_data[missing_data > 0])

    # Handle missing values
    st.subheader("Handle Missing Values")
    missing_option = st.selectbox(
        "Choose how to handle missing values",
        ["Do nothing", "Drop rows with missing values", "Fill numeric with mean", "Fill numeric with median"]
    )

    if missing_option == "Drop rows with missing values":
        df = df.dropna()
        st.success("Rows with missing values dropped.")

    elif missing_option == "Fill numeric with mean":
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        st.success("Missing numeric values filled with mean.")

    elif missing_option == "Fill numeric with median":
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        st.success("Missing numeric values filled with median.")

    # Column lists
    all_columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    st.subheader("Visualization Settings")

    plot_type = st.selectbox(
        "Select plot type",
        ["Line Plot", "Bar Chart", "Scatter Plot", "Histogram", "Count Plot"]
    )

    # Axis selection based on plot type
    if plot_type in ["Line Plot", "Bar Chart", "Scatter Plot"]:
        x_axis = st.selectbox("Select X-axis", all_columns)
        y_axis = st.selectbox("Select Y-axis", numeric_columns)

    elif plot_type == "Histogram":
        x_axis = st.selectbox("Select column", numeric_columns)
        y_axis = None

    elif plot_type == "Count Plot":
        x_axis = st.selectbox("Select column", all_columns)
        y_axis = None

    if st.button("Generate Plot"):
        fig, ax = plt.subplots(figsize=(6, 4))

        try:
            if plot_type == "Line Plot":
                sns.lineplot(data=df, x=x_axis, y=y_axis, ax=ax)

            elif plot_type == "Bar Chart":
                sns.barplot(data=df, x=x_axis, y=y_axis, ax=ax)

            elif plot_type == "Scatter Plot":
                sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)

            elif plot_type == "Histogram":
                sns.histplot(df[x_axis], kde=True, ax=ax)

            elif plot_type == "Count Plot":
                sns.countplot(data=df, x=x_axis, ax=ax)

            ax.set_title(plot_type)
            ax.tick_params(axis="x", rotation=45)

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error generating plot: {e}")
