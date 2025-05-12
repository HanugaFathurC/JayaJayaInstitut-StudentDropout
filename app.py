import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model components
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

# Updated selected features
selected_features = [
    'Marital_status', 'Application_order', 'Admission_grade', 'Displaced',
    'Debtor', 'Gender', 'Scholarship_holder', 'Age_at_enrollment'
]

# Mappings for categorical features
marital_status_mapping = {'Single': 1, 'Married': 2, 'Widower': 3, 'Divorced': 4, 'Facto union': 5, 'Legally separated': 6}
displaced_mapping = {'Yes': 1, 'No': 0}
debt_status_mapping = {'Yes': 1, 'No': 0}
gender_mapping = {'Male': 0, 'Female': 1}
scholarship_holder_mapping = {'Yes': 1, 'No': 0}

# App layout
st.title("🧑🏻‍🎓 Student Dropout Prediction")
st.markdown("""
This tool predicts whether a student is likely to drop out or graduate based on key academic and personal information.
Please fill in the following form and click **Make Prediction**.
""")

# Input fields
col1, col2 = st.columns(2)
inputs = {}

for idx, feature in enumerate(selected_features):
    if feature == 'Marital_status':
        inputs[feature] = col1.selectbox("Marital Status", marital_status_mapping.keys(), key=f"{feature}_{idx}")
    elif feature == 'Application_order':
        inputs[feature] = col2.number_input("Application Order (1 = first choice, 9 = last choice)", min_value=1, max_value=9, step=1, key=f"{feature}_{idx}")
    elif feature == 'Admission_grade':
        inputs[feature] = col1.number_input("Admission Grade (0 - 200)", min_value=0, max_value=200, key=f"{feature}_{idx}")
    elif feature == 'Displaced':
        inputs[feature] = col2.selectbox("Is the student displaced?", displaced_mapping.keys(), key=f"{feature}_{idx}")
    elif feature == 'Debtor':
        inputs[feature] = col1.selectbox("Is the student a debtor?", debt_status_mapping.keys(), key=f"{feature}_{idx}")
    elif feature == 'Gender':
        inputs[feature] = col2.selectbox("Gender", gender_mapping.keys(), key=f"{feature}_{idx}")
    elif feature == 'Scholarship_holder':
        inputs[feature] = col1.selectbox("Is the student a scholarship holder?", scholarship_holder_mapping.keys(), key=f"{feature}_{idx}")
    elif feature == 'Age_at_enrollment':
        inputs[feature] = col2.number_input("Age at Enrollment", min_value=10, max_value=100, key=f"{feature}_{idx}")

# Validation
valid_inputs = True
error_messages = []

for feature in selected_features:
    if inputs[feature] in ["", None]:
        valid_inputs = False
        error_messages.append(f"{feature} is required.")

# Show errors if any
if not valid_inputs:
    st.error("Please correct the following errors:")
    for msg in error_messages:
        st.write(f"- {msg}")
else:
    # Convert to numerical values
    input_values = []
    for feature in selected_features:
        if feature == 'Marital_status':
            input_values.append(marital_status_mapping[inputs[feature]])
        elif feature == 'Displaced':
            input_values.append(displaced_mapping[inputs[feature]])
        elif feature == 'Debtor':
            input_values.append(debt_status_mapping[inputs[feature]])
        elif feature == 'Gender':
            input_values.append(gender_mapping[inputs[feature]])
        elif feature == 'Scholarship_holder':
            input_values.append(scholarship_holder_mapping[inputs[feature]])
        else:
            input_values.append(float(inputs[feature]))

    # Prepare for prediction
    input_df = pd.DataFrame([input_values], columns=selected_features)
    input_scaled = scaler.transform(input_df)

    if st.button("Make Prediction"):
        prediction = model.predict(input_scaled)
        prediction_label = le.inverse_transform(prediction)
        proba = model.predict_proba(input_scaled)[0]

        st.subheader(f"🎯 Predicted Status: {prediction_label[0]}")
        st.write(f"Confidence: {np.max(proba) * 100:.2f}%")
