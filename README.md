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

## ❓ Business & Clinical Questions Answered
This dashboard is designed to answer several critical questions regarding patient populations and Chronic Kidney Disease risk:

### 1. Population Health & Demographics
* **What is the overall burden of Chronic Kidney Disease in our patient population?** (Answered by the CKD Prevalence KPI and Stage Distribution donut chart).
* **Are older patients disproportionately affected by advanced CKD?** (Answered by the Age Distribution by CKD Stage histogram).
* **What are the baseline averages for critical kidney function markers across our network?** (Answered by the Mean eGFR and Mean Creatinine KPIs).

### 2. Clinical Biomarker Analysis
* **How do primary kidney indicators degrade as the disease progresses?** (Answered by the eGFR, Serum Creatinine, and Blood Urea Nitrogen trend charts across stages).
* **Which secondary metrics fluctuate most noticeably with severe kidney disease?** (Answered by Hemoglobin and Pulse Pressure distributions).
* **Which biomarkers and patient statistics are most strongly correlated with one another?** (Answered by the Biomarker Correlation Matrix heatmap).

### 3. Comorbidities & Risk Factors
* **How heavily do lifestyle and underlying conditions (Diabetes, Hypertension, Smoking) correlate with advanced CKD stages?** (Answered by the Comorbidity Prevalence bar chart).

### 4. Patient Risk Stratification
* **Based on routine lab results and vitals, what is a specific patient's risk of having advanced CKD?** (Answered by the XGBoost Risk Prediction form).
* **Can we accurately predict a patient's exact CKD stage without requiring invasive procedures?** (Answered by the ML prediction output and class probabilities).
* **How does altering a specific health metric (e.g., lowering Blood Pressure or managing Glucose) theoretically impact a patient's risk stage?** (Answered by adjusting inputs in the prediction tool and re-running).

### 5. Data Quality & Auditing
* **What does the raw patient profile look like for anomalous cases?** (Answered by searching/filtering the raw Data Explorer table).
* **Are there any skewness or outliers in our clinical data collection?** (Answered by the embedded Descriptive Statistics table).

## 👨‍💻 Author
**Ahmed Fayad (nullPointer)**  
*ITI — Data Exploration & Preparation Project*
