import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
st.title("Career Progression Dashboard")

# Load Dataset
df = pd.read_csv(
    "employee_data.csv",
    encoding="latin1",
    on_bad_lines="skip"
)
# Feature Engineering

df["PromotionGapRatio"] = (
    df["YearsSinceLastPromotion"] /
    (df["YearsAtCompany"] + 1)
)

df["RoleStagnationIndex"] = (
    df["YearsInCurrentRole"] /
    (df["YearsAtCompany"] + 1)
)

df["TrainingIntensityScore"] = (
    df["TrainingTimesLastYear"] /
    (df["YearsAtCompany"] + 1)
)
# Dataset Preview
st.subheader("Dataset Preview")
st.write(df.head())

# Dataset Shape
st.write("Rows and Columns:", df.shape)
# Show New Features
st.subheader("Feature Engineered Columns")

st.write(
    df[[
        "PromotionGapRatio",
        "RoleStagnationIndex",
        "TrainingIntensityScore"
    ]].head()
)
# Department Wise Employees

st.subheader("Department Wise Employee Count")

department_count = df["Department"].value_counts()

st.bar_chart(department_count)
# KPI Section

st.subheader("Key Performance Indicators")

col1, col2, col3 = st.columns(3)

col1.metric("Total Employees", len(df))

col2.metric(
    "Average Promotion Gap",
    round(df["PromotionGapRatio"].mean(), 2)
)

col3.metric(
    "Average Training Score",
    round(df["TrainingIntensityScore"].mean(), 2)
)
# Department Filter

st.subheader("Department Filter")

department = st.selectbox(
    "Select Department",
    df["Department"].unique()
)

filtered_df = df[df["Department"] == department]

st.write(filtered_df.head())
# Attrition Analysis

st.subheader("Attrition Count")

attrition_count = df["Attrition"].value_counts()

st.bar_chart(attrition_count)
# High Promotion Gap Employees

st.subheader("High Promotion Gap Employees")

high_gap = df[df["PromotionGapRatio"] > 0.5]

st.write(
    high_gap[[
        "EmployeeNumber",
        "Department",
        "JobRole",
        "PromotionGapRatio"
    ]].head(10)
)
# Career Path Clustering

st.subheader("Career Path Clustering")

cluster_data = df[[
    "PromotionGapRatio",
    "RoleStagnationIndex",
    "TrainingIntensityScore"
]]

kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

df["CareerCluster"] = kmeans.fit_predict(cluster_data)

st.write(
    df[[
        "EmployeeNumber",
        "CareerCluster",
        "PromotionGapRatio",
        "RoleStagnationIndex"
    ]].head(10)
)

cluster_count = df["CareerCluster"].value_counts()

st.bar_chart(cluster_count)
# Cluster Labels

cluster_labels = {
    0: "Fast Track Employees",
    1: "Stable Contributors",
    2: "High Risk Stagnation"
}

df["ClusterLabel"] = df["CareerCluster"].map(cluster_labels)

st.subheader("Cluster Labels")

st.write(
    df[[
        "EmployeeNumber",
        "ClusterLabel"
    ]].head(10)
)
st.success("Career Progression Dashboard Loaded Successfully")