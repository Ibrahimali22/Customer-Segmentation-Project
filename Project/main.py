import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

@st.cache_resource
def load_model_and_scaler():
    model = load(r"logistic_regression_classifier.pkl")
    scaler = load(r"Scale2.pkl")
    label_encoders = load(r"encoding.pkl")
    return model, scaler, label_encoders

model, scaler, label_encoders = load_model_and_scaler()

df_final = pd.read_csv('labeled_data.csv')

st.set_page_config(page_title="Customer Clustering", layout="centered")

st.title("📊 Customer Clustering Tool")
st.write("Easily predict customer clusters using either manual input or CSV upload.")

tab1, tab2 = st.tabs(["🧍 Single User", "📄 CSV Upload"])

# ----------- Single User -----------
with tab1:
    st.markdown("### 📝 Enter Customer Details")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=70, step=1, value=30)
    with col2:
        income = st.number_input("Annual Income (k$)", min_value=15, max_value=150, step=1, value=50)
        score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, step=1, value=50)

    if st.button("🔍 Predict Cluster", key="single_predict"):

        gender_num = label_encoders.transform([gender])[0]


        input_data = np.array([[gender_num, age, income, score]])
        input_scaled = scaler.transform(input_data)

        cluster = model.predict(input_scaled)[0]
        st.session_state["cluster_id"] = cluster
        st.success(f"✅ This customer belongs to Cluster: {cluster}")

        st.session_state["single_user_data"] = {
            "Gender": gender,
            "Age": age,
            "Annual Income (k$)": income,
            "Spending Score (1-100)": score,
            "Cluster": cluster
        }

        st.page_link("pages/cluster_analysis.py", label="📊 Go to Cluster Analysis", icon="📈")

# ----------- CSV Upload -----------
with tab2:
    st.markdown("### 📄 Upload CSV Data")
    uploaded_csv = st.file_uploader("Upload CSV with customer data", type=["csv"])

    if uploaded_csv is not None:
        try:
            df_csv = pd.read_csv(uploaded_csv)
            st.write("### 📋 Uploaded Data")
            st.dataframe(df_csv)

            required_cols = ["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]

            if all(col in df_csv.columns for col in required_cols):

                df_csv["Gender"] = label_encoders.transform(df_csv["Gender"])


                X_scaled = scaler.transform(df_csv[required_cols])


                predictions = model.predict(X_scaled)
                df_csv["Predicted Cluster"] = predictions

                st.success("✅ Predictions completed!")
                st.dataframe(df_csv)


                csv = df_csv.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name="cluster_results.csv",
                    mime="text/csv"
                )
            else:
                st.error(f"⚠️ The CSV must contain columns: {required_cols}")

        except Exception as e:
            st.error(f"Error reading CSV: {e}")














