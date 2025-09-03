import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# ==============================
# Load Model & Scalers
# ==============================
model = tf.keras.models.load_model("insurance_model.h5", compile=False)
scaler_x = joblib.load("scaler_x.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# ==============================
# Page Config & Custom Style
# ==============================
st.set_page_config(
    page_title="Medical Insurance Premium Predictor",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .main {
            background-color: #f9f9f9;
        }
        .stApp {
            background-color: #f5f7fa;
        }
        h1 {
            color: #2C3E50;
            text-align: center;
            font-family: 'Helvetica Neue', sans-serif;
        }
        .stButton>button {
            color: white;
            background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
            border-radius: 10px;
            height: 3em;
            width: 100%;
            font-size: 16px;
        }
        .stSuccess {
            font-size: 20px;
            color: #27ae60;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================
# App Title
# ==============================
st.title("🏥 Medical Insurance Premium Prediction")
st.markdown("#### Predict your expected **insurance premium** based on lifestyle and health factors.")

# ==============================
# Sidebar Inputs
# ==============================
st.sidebar.header("📌 Input Your Details")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
children = st.sidebar.number_input("Children", min_value=0, max_value=10, value=0)
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# ==============================
# Encode Categorical Variables
# ==============================
sex = 1 if sex == "male" else 0
smoker = 1 if smoker == "yes" else 0

# One-hot encode region (northeast = base case)
region_dict = {
    "northeast": [0, 0, 0],
    "northwest": [1, 0, 0],
    "southeast": [0, 1, 0],
    "southwest": [0, 0, 1]
}
region_encoded = region_dict[region]

# Build feature vector (8 features total)
features = np.array([[age, sex, bmi, children, smoker] + region_encoded])

# ==============================
# Prediction
# ==============================
if st.sidebar.button("💡 Predict Premium"):
    # Scale input
    features_scaled = scaler_x.transform(features)

    # Predict using ANN
    prediction_scaled = model.predict(features_scaled)

    # Inverse transform to original scale
    prediction = scaler_y.inverse_transform(prediction_scaled)
    
    # Clip negative values (so no negative premiums)
    premium = max(prediction[0][0], 0)

    st.success(f"💰 Estimated Annual Premium: **${premium:,.2f}**")


    st.markdown("""
    ---
    ✅ **Note:** Predictions are based on the dataset used for training.  
    Lifestyle factors like **smoking** and **BMI** significantly increase premium costs.
    """)

# ==============================
# Footer
# ==============================
st.markdown("---")
st.markdown("Made with ❤️ using **Streamlit & TensorFlow** | Demo Project")
