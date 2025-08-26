🧠 Customer Segmentation Project






This project applies machine learning techniques for customer segmentation to help businesses better understand their customers and optimize marketing strategies.
🏆 Winner of 2nd place in the Applai Training Student Activity!

📌 Project Overview

Dataset: Mall Customers CSV (200 entries) with:

Gender, Age, Annual Income, Spending Score

EDA:

Used visualizations like histograms, pair plots, and a correlation heatmap to identify patterns in the data.

Preprocessing:

Label Encoding (Gender)

StandardScaler (feature normalization)

PCA (dimensionality reduction)

Clustering (KMeans):

Tested k = 2 to 9 clusters.

Evaluated with Silhouette Score.

Best performance: k = 6 (score = 0.55)

Classification (Logistic Regression):

Trained on the original 4 features using the KMeans cluster labels.

Accuracy:

Train: 99.38%

Test: 97.5%

Insights:

Identified 6 customer segments:

🏅 VIPs: Young, high income, high spending

⚡ Impulsive Buyers: Young, low income, high spending

...and others (e.g., conservative, low-spenders, etc.)

Deployment:

Built with Streamlit

Models saved using Joblib

🚀 Installation
# 1. Clone the repository
git clone https://github.com/Ibrahimali22/Customer-Segmentation-Project.git
cd Customer-Segmentation-Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run main.py


    Make sure requirements.txt includes:
    pandas, numpy, scikit-learn, matplotlib, seaborn, streamlit, joblib, plotly


💡 Usage

Single User Prediction:
Input customer details (Gender, Age, Income, Spending Score) via UI.

Batch Prediction:
Upload a CSV file to classify multiple customers at once.

Full Analysis:
Run the Jupyter notebook Applai_project_final(2).ipynb for detailed steps.


🎥 Video Demo

📽️ []

🙏 Acknowledgments

Special thanks to:

Ahmed Ebrahim

Ahmed Ayman
for their participation in this project.

📄 License

This project is licensed under the MIT License

