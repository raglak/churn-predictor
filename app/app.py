import streamlit as st
import joblib
import numpy as np

model = joblib.load('models/churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.set_page_config(page_title="Churn Predictor", page_icon="📊")
st.title("Customer Churn Predictor")
st.markdown("Enter customer details to predict churn probability.")

col1, col2 = st.columns(2)

with col1:
    gender            = st.radio("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    senior            = st.radio("Senior Citizen", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    partner           = st.radio("Has Partner", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    dependents        = st.radio("Has Dependents", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    tenure            = st.slider("Tenure (months)", 0, 72, 12)
    phone_service     = st.radio("Phone Service", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    multiple_lines    = st.selectbox("Multiple Lines", [0, 1, 2],
                                     format_func=lambda x: ["No","Yes","No phone service"][x])
    internet          = st.selectbox("Internet Service", [0, 1, 2],
                                     format_func=lambda x: ["DSL","Fiber optic","No"][x])
    online_security   = st.selectbox("Online Security", [0, 1, 2],
                                     format_func=lambda x: ["No","Yes","No internet service"][x])
    online_backup     = st.selectbox("Online Backup", [0, 1, 2],
                                     format_func=lambda x: ["No","Yes","No internet service"][x])

with col2:
    device_protection = st.selectbox("Device Protection", [0, 1, 2],
                                     format_func=lambda x: ["No","Yes","No internet service"][x])
    tech_support      = st.selectbox("Tech Support", [0, 1, 2],
                                     format_func=lambda x: ["No","Yes","No internet service"][x])
    streaming_tv      = st.selectbox("Streaming TV", [0, 1, 2],
                                     format_func=lambda x: ["No","Yes","No internet service"][x])
    streaming_movies  = st.selectbox("Streaming Movies", [0, 1, 2],
                                     format_func=lambda x: ["No","Yes","No internet service"][x])
    contract          = st.selectbox("Contract Type", [0, 1, 2],
                                     format_func=lambda x: ["Month-to-month","One year","Two year"][x])
    paperless         = st.radio("Paperless Billing", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    payment_method    = st.selectbox("Payment Method", [0, 1, 2, 3],
                                     format_func=lambda x: ["Bank transfer","Credit card","Electronic check","Mailed check"][x])
    monthly_charges   = st.number_input("Monthly Charges ($)", 0.0, 120.0, 65.0)
    total_charges     = st.number_input("Total Charges ($)", 0.0, 9000.0, 780.0)

# Engineered features (same as notebook)
charges_per_tenure = monthly_charges / (tenure + 1)
service_count      = sum([phone_service, multiple_lines == 1, internet != 2,
                          online_security == 1, online_backup == 1,
                          device_protection == 1, tech_support == 1,
                          streaming_tv == 1, streaming_movies == 1])

# Must match exact column order: 21 features
input_data = np.array([[gender, senior, partner, dependents, tenure,
                        phone_service, multiple_lines, internet,
                        online_security, online_backup, device_protection,
                        tech_support, streaming_tv, streaming_movies,
                        contract, paperless, payment_method,
                        monthly_charges, total_charges,
                        charges_per_tenure, service_count]])

input_scaled = scaler.transform(input_data)

if st.button("Predict Churn", use_container_width=True):
    prob  = model.predict_proba(input_scaled)[0][1]
    label = "Will Churn" if prob > 0.5 else "Will Stay"
    color = "red" if prob > 0.5 else "green"
    st.markdown(f"### Prediction: :{color}[{label}]")
    st.metric("Churn Probability", f"{prob*100:.1f}%")
    st.progress(float(prob))
