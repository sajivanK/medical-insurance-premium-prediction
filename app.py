import streamlit as st
import numpy as np
import tensorflow as tf

# Load your trained ANN model
model = tf.keras.models.load_model("insurance_model.h5")

# 🎨 Page configuration
st.set_page_config(
    page_title="Medical Insurance Premium Predictor",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 🌟 Custom CSS for premium look
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

# 🏥 App Title
st.title("🏥 Medical Insurance Premium Prediction")
st.markdown("#### Predict your expected **insurance premium** based on lifestyle and health factors.")

# 📊 Sidebar for inputs
st.sidebar.header("📌 Input Your Details")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
children = st.sidebar.number_input("Number of Children", min_value=0, max_value=10, value=0)
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# 🔄 Encode categorical variables
sex = 1 if sex == "male" else 0
smoker = 1 if smoker == "yes" else 0
region_dict = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
region = region_dict[region]

# 🧮 Feature array
features = np.array([[age, sex, bmi, children, smoker, region]])

# 🎯 Prediction
if st.sidebar.button("💡 Predict Premium"):
    prediction = model.predict(features)
    st.success(f"💰 Estimated Annual Premium: **${prediction[0][0]:,.2f}**")

    # Extra info
    st.markdown("""
    ---
    ✅ **Note:** Predictions are based on the dataset used for training.  
    Lifestyle factors like **smoking** and **BMI** significantly increase premium costs.
    """)

# 📌 Footer
st.markdown("---")
st.markdown("Made with ❤️ using **Streamlit & TensorFlow** | Demo Project")
