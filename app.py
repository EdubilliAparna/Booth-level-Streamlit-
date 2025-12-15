import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(page_title="Booth Level Analytics", layout="wide")
st.title("Booth Level Analytics Dashboard")

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Booth-Level Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")
    st.write("### Data Preview")
    st.dataframe(df.head())

    # ----------------------------------------------------
    # DISTRICT FILTER
    # ----------------------------------------------------
    if "district_name" not in df.columns:
        st.error("Column 'district_name' not found in dataset")
        st.stop()

    districts = df["district_name"].dropna().unique()
    selected_dist = st.selectbox("Select District", districts)

    dist_df = df[df["district_name"] == selected_dist].copy()

    st.write(f"### Data for District: **{selected_dist}**")
    st.dataframe(dist_df)

    # ----------------------------------------------------
    # VOTE SHARE + SWING / LOYAL / NEUTRAL
    # ----------------------------------------------------
    st.subheader("Vote Share Clustering (Swing / Loyal / Neutral)")

    vote_cols = [col for col in dist_df.columns if col.endswith("_votes")]

    if not vote_cols:
        st.warning("No *_votes columns found in dataset")
    elif "valid_votes" not in dist_df.columns:
        st.error("Column 'valid_votes' is missing")
    else:
        # Convert to numeric safely
        dist_df["valid_votes"] = pd.to_numeric(
            dist_df["valid_votes"], errors="coerce"
        )

        for col in vote_cols:
            dist_df[col] = pd.to_numeric(dist_df[col], errors="coerce")

        # Vote share calculation (safe for zero / NaN)
        for col in vote_cols:
            dist_df[f"{col}_share"] = np.where(
                dist_df["valid_votes"] > 0,
                dist_df[col] / dist_df["valid_votes"],
                0
            )

        # Winner & margin
        dist_df["winner"] = dist_df[vote_cols].idxmax(axis=1)

        dist_df["margin"] = (
            dist_df[vote_cols].max(axis=1)
            - dist_df[vote_cols].nlargest(2, axis=1).iloc[:, -1]
        )

        dist_df["margin_share"] = np.where(
            dist_df["valid_votes"] > 0,
            dist_df["margin"] / dist_df["valid_votes"],
            0
        )

        # Classification
        def classify(row):
            if row["margin_share"] < 0.05:
                return "Swing"
            elif row["margin_share"] >= 0.15:
                return "Loyal"
            else:
                return "Neutral"

        dist_df["cluster"] = dist_df.apply(classify, axis=1)

        st.success("Clustering completed successfully")
        st.dataframe(
            dist_df[
                [
                    "polling_station_name",
                    "winner",
                    "margin_share",
                    "cluster"
                ]
            ]
        )

    # ----------------------------------------------------
    # HEATMAP: CASTE / RELIGION / AGE
    # ----------------------------------------------------
    st.subheader("Heatmap: Caste × Religion × Age Category")

    required_cols = ["caste_group", "religion_group", "age_category"]

    if all(col in dist_df.columns for col in required_cols):
        try:
            pivot = pd.crosstab(
                [dist_df["caste_group"], dist_df["religion_group"]],
                dist_df["age_category"]
            )

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(
                pivot,
                annot=True,
                fmt="d",
                cmap="coolwarm",
                ax=ax
            )
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Heatmap error: {e}")
    else:
        st.warning(
            "Required columns missing: caste_group, religion_group, age_category"
        )

    # ----------------------------------------------------
    # HEATMAP: SWING / LOYAL / NEUTRAL
    # ----------------------------------------------------
    st.subheader("Heatmap: Swing / Loyal / Neutral")

    if "cluster" in dist_df.columns:
        try:
            clust_count = dist_df.pivot_table(
                index="polling_station_name",
                columns="cluster",
                aggfunc="size",
                fill_value=0
            )

            fig2, ax2 = plt.subplots(figsize=(12, 6))
            sns.heatmap(
                clust_count,
                annot=True,
                fmt="d",
                cmap="viridis",
                ax=ax2
            )
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"Cluster heatmap error: {e}")
    else:
        st.warning("Clustering not completed yet")

else:
    st.info("Please upload a CSV file to begin booth analytics.")
