import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and expected columns
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# Background color function
def set_background(color_hex):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {color_hex};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set initial background color (neutral orange)
set_background("#000000")

# App Title
st.title("💓 Heart Stroke Prediction By Ansh")

st.markdown("Please fill in the details below to assess your heart stroke risk.")

# 📘 Educational Section
with st.expander("📘 Know the Medical Terms (Click to Expand)"):
    st.markdown("""
    #### 🧠 Key Medical Terms Explained:

    - **Chest Pain Type:**
        - **ATA (Atypical Angina):** Chest pain not related to the heart.
        - **NAP (Non-Anginal Pain):** Pain not due to angina or heart.
        - **TA (Typical Angina):** Heart-related chest pain.
        - **ASY (Asymptomatic):** No symptoms but possible hidden issues.

    - **Resting Blood Pressure (RestingBP):**
        - Blood pressure while resting.
        - Normal: ~120/80 mm Hg. 
        - High BP increases heart risk.

    - **Cholesterol:**
        - Fat content in your blood.
        - High cholesterol can block arteries.

    - **Fasting Blood Sugar (FastingBS):**
        - Blood sugar after 8-12 hrs of fasting.
        - 120 mg/dL indicates diabetes risk.

    - **Resting ECG:**
        - Heart electrical activity at rest.
        - **Normal:** No issue. 
        - **ST/LVH:** Possible heart problems.

    - **Max Heart Rate (MaxHR):**
        - Highest HR during exercise.
        - Normal Max = 220 - age.

    - **Exercise-Induced Angina:**
        - Chest pain caused by physical activity.

    - **Oldpeak (ST Depression):**
        - Drop in ST segment during exercise.
        - Shows poor blood supply to heart.

    - **ST Slope:**
        - **Up:** Normal  
        - **Flat:** Mild abnormality  
        - **Down:** Serious concern
    """)

# Collect user input
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Prediction Logic
if st.button("🔍 Predict"):

    # Create raw input dictionary
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([raw_input])

    # Fill missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[expected_columns]

    # Scale input
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]

    # Display result and change background
    if prediction == 1:
        set_background("#000000")  # Red
        st.error("⚠️ High Risk of Heart Disease")
    else:
        set_background("#000000")  # Green
        st.success("✅ Low Risk of Heart Disease")

# 📜 Terms and Conditions
with st.expander("📄 Terms and Conditions"):
    st.markdown("""
    - This tool is for educational and awareness purposes only.
    - It is not a substitute for professional medical advice, diagnosis, or treatment.
    - Prediction results are based on historical data and may not be accurate for all individuals.
    - Always consult a certified healthcare provider for health-related decisions.
    """)