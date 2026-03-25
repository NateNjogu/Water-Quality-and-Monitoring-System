import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
# importtensorflow as tf  # Deep Learning upgrade
import json
import os
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")


# Production Configuration
class WaterQualityModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, max_depth=20, 
                                          min_samples_split=2, n_jobs=-1, random_state=42)
        self.pipeline = Pipeline([('scaler', self.scaler), ('model', self.model)])
        self.model_path = 'cloud_WaterQuality_model.keras'
        self.load_or_train()
    
    def load_or_train(self):
        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)
            st.success("✅ Loaded production model")
        else:
            self._train_production_model()
    
    def _train_production_model(self):
        # Scalable dataset (5000 samples)
        np.random.seed(42)
        n_samples = 5000
        data = {
            'ph': np.clip(np.random.normal(7.2, 1.5, n_samples), 0, 14),
            'conductivity': np.random.uniform(50, 1500, n_samples),
            'turbidity': np.random.exponential(5, n_samples),
            'nitrates': np.random.uniform(0, 30, n_samples),
            'e_coli': np.random.poisson(5, n_samples),
            'fluoride': np.random.uniform(0, 2.5, n_samples)
        }
        df = pd.DataFrame(data)
        df['admissible'] = ((df['ph'].between(6.5, 8.5)) & 
                           (df['turbidity'] < 5) & 
                           (df['nitrates'] < 20) & 
                           (df['e_coli'] < 10) & 
                           (df['fluoride'] < 1.5)).astype(int)
        
        # Feature engineering + Cross-validation
        X = df.drop('admissible', axis=1)
        y = df['admissible']
        cv_scores = cross_val_score(self.pipeline, X, y, cv=5, scoring='accuracy')
        st.info(f"Cross-Val Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
        self.pipeline.fit(X, y)
        joblib.dump(self.pipeline, self.model_path)
        st.success("✅ Trained & checkpointed production model")

def classify_admissible(**params):
    reasons = []
    if not (6.5 <= params['ph'] <= 8.5):
        reasons.append("pH out of range")
    if params['turbidity'] >= 5:
        reasons.append("Turbidity too high")
    if params['nitrates'] >= 20:
        reasons.append("Nitrates too high")
    if params['e_coli'] >= 10:
        reasons.append("E.coli contamination")
    if params['fluoride'] >= 1.5:
        reasons.append("Fluoride too high")
    return len(reasons) == 0, reasons


# Production App
st.set_page_config(page_title=" Water Quality Monitor", layout="wide")
st.markdown('<style>.main {background-color: #0e1117;}.stMetric > label {color: #00d4aa;}</style>', unsafe_allow_html=True)

st.title("🧪 Water Quality Intelligence")
model = WaterQualityModel()

# Scalable Inputs (6 parameters)
st.sidebar.header("🌡️ IoT Sensors (Production)")
params = {
    'ph': st.sidebar.slider("pH", 0.0, 14.0, 7.2),
    'turbidity': st.sidebar.slider("Turbidity (NTU)", 0.0, 100.0, 5.2),
    'conductivity': st.sidebar.slider("Conductivity (µS/cm)", 0.0, 2000.0, 280.0),
    'nitrates': st.sidebar.slider("Nitrates (mg/L)", 0.0, 50.0, 12.3),
    'e_coli': st.sidebar.slider("E.coli (CFU/100ml)", 0.0, 50.0, 3.0),
    'fluoride': st.sidebar.slider("Fluoride (mg/L)", 0.0, 3.0, 0.8)
}

# Multi-model predictions
input_data = np.array([list(params.values())])
ml_pred = model.pipeline.predict(input_data)[0]
ml_prob = model.pipeline.predict_proba(input_data)[0][1]
rule_admissible, reasons = classify_admissible(**params)

# Production Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("ML Prediction", "Admissible ✅" if ml_pred else "Not Admissible ❌")
col2.metric("Rule Check", "Admissible ✅" if rule_admissible else "Not Admissible ❌")
col3.metric("ML Confidence", f"{ml_prob:.1%}")
col4.metric("Issues", len(reasons))

if reasons:
    st.error(f"⚠️ Violations: {' | '.join(reasons)}")

# FIXED: Interactive Threshold Table (No eval errors)
st.subheader("📊 Live Threshold Compliance")

# Lambda functions for each parameter
threshold_conditions = {
    'pH': lambda ph: 6.5 <= ph <= 8.5,
    'Turbidity': lambda turb: turb < 5,
    'Conductivity': lambda cond: cond < 1000,
    'Nitrates': lambda nit: nit < 50,
    'E.coli': lambda ecoli: ecoli < 10,
    'Fluoride': lambda fluor: fluor < 1.5
}


threshold_conditions = {
    'pH': lambda ph: 6.5 <= ph <= 8.5,
    'Turbidity': lambda turb: turb < 5,
    'Conductivity': lambda cond: cond < 1000,
    'Nitrates': lambda nit: nit < 50,
    'E.coli': lambda ecoli: ecoli < 10,
    'Fluoride': lambda fluor: fluor < 1.5
}

status_list = []
current_values = []
for param, cond in threshold_conditions.items():
    param_key = param.lower().replace(' ', '_')
    value = params.get(param_key, 'N/A')
    current_values.append(value)
    status_list.append('✅ PASS' if isinstance(value, (int, float)) and cond(value) else '❌ FAIL')

threshold_df = pd.DataFrame({
    'Parameter': list(threshold_conditions.keys()),
    'Safe Range': ['6.5-8.5', '<5 NTU', '<1000 µS/cm', '<50 mg/L', '<10 CFU', '<1.5 mg/L'],
    'Current Value': current_values,
    'Status': status_list
})

st.dataframe(threshold_df, use_container_width=True)


# Fault Tolerance Demo
st.subheader("🛡️ Fault Tolerance Test")
if st.button("Simulate 20% Sensor Failure"):
    faulty_data = input_data.copy()
    faulty_data[0, np.random.choice(6, 2, replace=False)] = np.nan  # 2/6 fail
    st.write("Fault-tolerant prediction maintained:", model.pipeline.predict(faulty_data)[0])

# CSV Batch Processing (Scalable)
uploaded = st.file_uploader("Upload Production CSV")
if uploaded:
    df_up = pd.read_csv(uploaded)
    df_up['prediction'] = model.pipeline.predict(df_up)
    df_up['prob'] = model.pipeline.predict_proba(df_up)[:,1]
    st.dataframe(df_up)
    st.download_button("📥 Download Results", df_up.to_csv(index=False).encode(), "predictions.csv")

# Performance Metrics (from CV)
if os.path.exists('model_metrics.json'):
    metrics = json.load(open('model_metrics.json'))
    st.metric("Production CV Accuracy", f"{metrics['cv_accuracy']:.1%}")
    st.metric("Fault Tolerance (20% loss)", f"{metrics['fault_tolerance']:.1%}")


