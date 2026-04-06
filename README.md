# 🏥 NephroCare — Chronic Kidney Disease Clinical Dashboard

![NephroCare Banner](https://img.shields.io/badge/Status-Active-success) ![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python) ![Dash](https://img.shields.io/badge/Plotly_Dash-UI-informational?logo=plotly) ![XGBoost](https://img.shields.io/badge/XGBoost-Machine_Learning-orange)

**NephroCare** is a professional, interactive clinical web dashboard designed for the exploration, analytics, and risk prediction of Chronic Kidney Disease (CKD). Developed as a final project for the ITI Data Exploration and Preparation track.

## 🌟 Key Features

* **📊 Population Overview**: Real-time epidemiological analytics across clinical cohorts, including stage distribution, eGFR variance, and demographic profiles.
* **🔬 Clinical Analysis**: Deep-dive statistical visualizations on biomarkers like Serum Creatinine, Blood Urea Nitrogen (BUN), Hemoglobin, and Comorbidity prevalence.
* **🤖 Risk Prediction (XGBoost)**: A built-in inference engine utilizing a trained XGBoost model to instantly predict a patient's CKD stage (Healthy to Kidney Failure) based on 31 clinical features.
* **📋 Data Explorer**: Interactive data tables to browse raw patient records and compute descriptive statistics on the fly.

## 🛠️ Technology Stack

* **Frontend Framework**: [Plotly Dash](https://dash.plotly.com/) & HTML/CSS
* **Data Manipulation**: [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)
* **Visualizations**: [Plotly Express](https://plotly.com/python/plotly-express/) & Graph Objects
* **Machine Learning**: [XGBoost](https://xgboost.readthedocs.io/) (Classification)

## 🚀 Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Fayad-nullPointer-Data_Exploration_Prepration_Projects.git
   cd Fayad-nullPointer-Data_Exploration_Prepration_Projects
   ```

2. **Install dependencies**:
   Make sure you have your Python environment set up, then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Dashboard**:
   ```bash
   python Script_final.py
   ```

4. **Access the Web App**:
   Open your browser and navigate to: `http://127.0.0.1:8050/`

## 📁 Project Structure

```text
├── Script_final.py            # Main application script (Dash UI & Logic)
├── notebook.ipynb             # ML Model training and Data preparation Notebook
├── best_model_xgb.pkl         # Trained XGBoost inference model
├── Training_CKD_dataset.csv   # Training data subset
├── Testing_CKD_dataset.csv    # Testing data subset
└── data/                      # Processed and Raw data directory
```

## 🧠 Predictive Pipeline
The machine learning pipeline accurately mirrors the data transformations used during training. Real-time patient inputs (e.g. `Systolic BP` and `eGFR`) are dynamically log-transformed (`np.log1p`) before being fed into the serialized XGBoost model, guaranteeing accurate clinical inference.

## 👨‍💻 Author
**Ahmed Fayad (nullPointer)**  
*ITI — Data Exploration & Preparation Project*
