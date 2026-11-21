import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Page Configuration
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Risk Prediction System")
st.write("Enter the patient's clinical and lifestyle details below to predict diabetes risk.")

# 2. Load, Clean & Train Model (Cached for fast performance)
@st.cache_resource
def load_and_train():
    # Update the file path if your CSV has a different name or folder
    data = pd.read_csv("Diabetes_Prediction_Dataset.csv")
    
    # Drop ID and duplicates
    data = data.drop(columns=['Patient_ID'], errors='ignore').drop_duplicates()
    
    # Clean numeric columns & fix '999' Blood Pressure errors
    num_cols = ['Age', 'BMI', 'Glucose', 'Blood_Pressure', 'Cholesterol', 'Insulin', 'HbA1c']
    for col in num_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data['Blood_Pressure'] = data['Blood_Pressure'].replace(999, np.nan)
    
    # Clean target column
    data['Diabetes'] = data['Diabetes'].astype(str).str.strip().map({'Yes': 1, 'No': 0})
    data = data.dropna(subset=['Diabetes'])
    
    # Separate features and target
    X = pd.get_dummies(data.drop(columns=['Diabetes']), drop_first=True)
    y = data['Diabetes']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_imp, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test_imp)) * 100
    return model, imputer, X.columns, accuracy

try:
    model, imputer, feature_columns, model_accuracy = load_and_train()
    st.sidebar.success(f"Model Ready (Test Accuracy: {model_accuracy:.2f}%)")
except FileNotFoundError:
    st.error("Error: 'diabetes_data.csv' not found. Please place your dataset in the same folder as this script.")
    st.stop()

# 3. User Input Form
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=45)
    gender = st.selectbox("Gender", ["Female", "Male", "Unknown"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=28.5, step=0.1)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=40, max_value=400, value=140)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=40, max_value=250, value=120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=80, max_value=500, value=200)

with col2:
    insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=500, value=120)
    hba1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=18.0, value=6.5, step=0.1)
    smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
    activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
    family_history = st.selectbox("Family History of Diabetes", ["No", "Yes"])

# 4. Predict Button
if st.button("Predict Diabetes Risk", type="primary"):
    input_df = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'BMI': bmi,
        'Glucose': glucose,
        'Blood_Pressure': blood_pressure,
        'Cholesterol': cholesterol,
        'Insulin': insulin,
        'HbA1c': hba1c,
        'Smoking_Status': smoking,
        'Physical_Activity': activity,
        'Family_History': family_history
    }])
    
    # Encode and align columns with training data
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
    input_imputed = imputer.transform(input_encoded)
    
    # Generate prediction and probability
    prediction = model.predict(input_imputed)[0]
    probability = model.predict_proba(input_imputed)[0][1] * 100
    
    st.divider()
    if prediction == 1:
        st.error(f"### Prediction: **Positive (High Diabetes Risk)**")
    else:
        st.success(f"### Prediction: **Negative (Low Diabetes Risk)**")
        
    st.metric(label="Estimated Diabetes Probability", value=f"{probability:.1f}%")
    st.caption("Disclaimer: This tool is for educational and data analysis demonstration purposes only, not medical advice.")