import streamlit as st
import pandas as pd
import plotly.express as px


st.set_page_config(page_title="Cluster Analysis", layout="centered")

if "cluster_id" not in st.session_state:
    st.error("⚠️ Please go to the main page and find your cluster first.")
    st.stop()

cluster_id = st.session_state["cluster_id"]

st.title(f"📊 Detailed Analysis for Cluster {cluster_id}")


df = pd.read_csv(r"C:\Users\France Tech\OneDrive\سطح المكتب\Customer Segmentation Project\Project\labeled_data.csv")

df_cluster = df[df["Cluster"] == cluster_id]


st.subheader("📌 Summary Statistics")
st.write(df_cluster.describe())


fig_gender = px.pie(
    df_cluster,
    names="Gender",
    title="Gender Distribution",
    color="Gender",
    color_discrete_map={"Male": "#636EFA", "Female": "#EF553B"}
)
st.plotly_chart(fig_gender, use_container_width=True)


fig_age = px.histogram(
    df_cluster,
    x="Age",
    nbins=10,
    title="Age Distribution",
    color_discrete_sequence=["#00CC96"]
)
fig_age.update_traces(marker_line_width=1, marker_line_color="black")
st.plotly_chart(fig_age, use_container_width=True)


fig_income = px.histogram(
    df_cluster,
    x="Annual Income (k$)",
    nbins=10,
    title="Annual Income Distribution",
    color_discrete_sequence=["#AB63FA"]
)
fig_income.update_traces(marker_line_width=1, marker_line_color="black")
st.plotly_chart(fig_income, use_container_width=True)

fig_spending = px.histogram(
    df_cluster,
    x="Spending Score (1-100)",
    nbins=10,
    title="Spending Score Distribution",
    color_discrete_sequence=["#FFA15A"]
)
fig_spending.update_traces(marker_line_width=1, marker_line_color="black")
st.plotly_chart(fig_spending, use_container_width=True)


st.subheader("📊 Comparison with Other Clusters")
comparison_df = df.groupby("Cluster").mean(numeric_only=True).reset_index()
fig_compare = px.bar(
    comparison_df,
    x="Cluster",
    y=["Age", "Annual Income (k$)", "Spending Score (1-100)"],
    title="Average Metrics per Cluster",
    barmode="group"
)
st.plotly_chart(fig_compare, use_container_width=True)
