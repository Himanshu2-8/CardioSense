import streamlit as st
import numpy as np
import joblib
import base64

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="CardioSense ❤️",
    page_icon="❤️",
    layout="centered",
)

# ===============================
# LOAD AND ENCODE BACKGROUND IMAGE
# ===============================
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Encode the background image
try:
    bg_image = get_base64_image("assets/image.jpg")
    bg_style = f'background-image: url("data:image/jpeg;base64,{bg_image}");'
except FileNotFoundError:
    # Fallback gradient if image not found
    bg_style = 'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'

# ===============================
# CUSTOM CSS (Gradient Background + Glassmorphic Card)
# ===============================
st.markdown(f"""
<style>
/* Background */
[data-testid="stAppViewContainer"] {{
    {bg_style}
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    font-family: 'Poppins', sans-serif;
}}

/* Hide Streamlit header */
[data-testid="stHeader"] {{
    background: transparent;
    display: none;
}}

/* Hide top padding */
.block-container {{
    padding-top: 2rem;
}}

/* Glass effect card */
.main-card {{
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 3rem 2.5rem;
    box-shadow: 0 4px 40px rgba(0,0,0,0.25);
    max-width: 800px;
    margin: 3rem auto;
}}

/* Titles */
h1 {{
    text-align: center;
    color: #ffffff;
    text-shadow: 0px 0px 15px rgba(255,255,255,0.4);
}}
h3 {{
    color: #f1f1f1;
}}

/* Labels and text */
label, .stSelectbox label, .stNumberInput label {{
    color: #f0f0f0 !important;
    font-weight: 500;
}}

/* Button styling */
.stButton button {{
    background: linear-gradient(135deg, #ff4b2b, #ff416c);
    color: white;
    border: none;
    padding: 0.6rem 1.5rem;
    border-radius: 10px;
    font-weight: 600;
    font-size: 1rem;
    transition: 0.3s ease-in-out;
}}
.stButton button:hover {{
    background: linear-gradient(135deg, #ff6b6b, #ff8e53);
    transform: scale(1.03);
}}

/* Result messages */
.success, .error {{
    text-align: center;
    font-size: 1.2rem;
    padding: 1rem;
    border-radius: 12px;
    font-weight: 600;
    backdrop-filter: blur(8px);
}}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL
# ===============================
model = joblib.load("adaboost_model.pkl")

# ===============================
# PAGE CONTENT
# ===============================
st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.title("❤️ CardioSense")
st.markdown("### AI-Powered Heart Disease Prediction")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ("Female", "Male"))
    cp = st.selectbox("Chest Pain Type", (0, 1, 2, 3))
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", (0, 1))
    restecg = st.selectbox("Resting ECG Results", (0, 1, 2))

with col2:
    thalach = st.number_input("Max Heart Rate Achieved", 70, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", (0, 1))
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of ST Segment", (0, 1, 2))
    ca = st.selectbox("Major Vessels Colored (0–3)", (0, 1, 2, 3))
    thal = st.selectbox("Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)", (1, 2, 3))

st.markdown("---")

if st.button("🔍 Predict"):
    sex_val = 1 if sex == "Male" else 0
    features = np.array([[age, sex_val, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])

    with st.spinner("Analyzing patient data... ⏳"):
        prediction = model.predict(features)

    st.markdown("---")
    if prediction[0] == 1:
        st.markdown(
            '<div class="error" style="background:rgba(255,0,0,0.2);color:#ffb3b3;">🚨 The patient is likely to have heart disease.</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="success" style="background:rgba(0,255,0,0.2);color:#b3ffb3;">✅ The patient is unlikely to have heart disease.</div>',
            unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<center><p style='color:#fff;font-size:0.9rem;'>Developed with ❤️ using Streamlit and scikit-learn</p></center>",
    unsafe_allow_html=True)