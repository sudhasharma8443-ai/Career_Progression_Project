import streamlit as st
import pandas as pd
import os
from sklearn.cluster import KMeans

st.set_page_config(page_title="Career Progression Dashboard", layout="wide")

st.title("📊 Career Progression Dashboard")

# Safe file path (IMPORTANT FIX)
file_path = os.path.join(os.path.dirname(__file__), "employee_data.csv")

df = pd.read_csv(file_path, encoding="latin1", on_bad_lines="skip")

# Feature Engineering
df["PromotionGapRatio"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
df["RoleStagnationIndex"] = df["YearsInCurrentRole"] / (df["YearsAtCompany"] + 1)
df["TrainingIntensityScore"] = df["YearsAtCompany"] / (df["YearsAtCompany"] + 1)

# Dataset Preview
st.subheader("Dataset Preview")
st.write(df.head())

st.subheader("Dataset Shape")
st.write(df.shape)

# KPIs
st.subheader("Key Performance Indicators")

col1, col2, col3 = st.columns(3)

col1.metric("Total Employees", len(df))
col2.metric("Avg Promotion Gap", round(df["PromotionGapRatio"].mean(), 2))
col3.metric("Avg Training Score", round(df["TrainingIntensityScore"].mean(), 2))

# Department wise employees
st.subheader("Department Wise Employee Count")
st.bar_chart(df["Department"].value_counts())

# Filter
st.subheader("Department Filter")
dept = st.selectbox("Select Department", df["Department"].unique())
st.write(df[df["Department"] == dept].head())

# Attrition
st.subheader("Attrition Count")
st.bar_chart(df["Attrition"].value_counts())

# Clustering
st.subheader("Career Path Clustering")

cluster_data = df[
    ["PromotionGapRatio", "RoleStagnationIndex", "TrainingIntensityScore"]
]

kmeans = KMeans(n_clusters=3, random_state=42)
df["CareerCluster"] = kmeans.fit_predict(cluster_data)

cluster_labels = {
    0: "Fast Track Employees",
    1: "Stable Contributors",
    2: "High Risk Stagnation"
}

df["ClusterLabel"] = df["CareerCluster"].map(cluster_labels)

st.subheader("Cluster Results")
st.write(df[["EmployeeNumber", "CareerCluster", "ClusterLabel"]].head(10))

st.success("Dashboard Loaded Successfully 🚀")
