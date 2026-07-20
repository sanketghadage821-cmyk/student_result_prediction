import streamlit as st
import pickle
import numpy as np

# Set up page configurations with a modern title and icon
st.set_page_config(
    page_title="Student Outcome Predictor",
    page_icon="🎓",
    layout="centered"
)

# Custom CSS for a premium, clean interface with a cohesive color palette
st.markdown("""
    <style>
    /* Main background and font styling */
    .stApp {
        background-color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header card styling */
    .main-header {
        background: linear-gradient(135deg, #4f46e5 0%, #3730a3 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(79, 70, 229, 0.15);
    }
    
    /* Section headers */
    h3 {
        color: #1e293b !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
    }
    
    /* Customize the predict button */
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
        color: white !important;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4);
        background: linear-gradient(135deg, #4338ca 0%, #3730a3 100%);
    }
    
    /* Prediction output card */
    .result-card {
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
        animation: fadeIn 0.5s ease-in-out;
    }
    .result-yes {
        background-color: #ecfdf5;
        color: #065f46;
        border: 2px solid #a7f3d0;
    }
    .result-no {
        background-color: #fef2f2;
        color: #991b1b;
        border: 2px solid #fecaca;
    }
    </style>
""", unsafe_style_allowed=True)

# Load the trained SVM Model safely
@st.cache_resource
def load_model():
    with open("SVMModel.pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure 'SVMModel.pkl' is in the same directory.")
    st.stop()

# Header Display
st.markdown("""
    <div class="main-header">
        <h1 style='margin:0; font-size: 2.2rem;'>Performance & Analytics Predictor</h1>
        <p style='margin: 10px 0 0 0; opacity: 0.9;'>Enter student metrics below to evaluate final prediction status.</p>
    </div>
""", unsafe_style_allowed=True)

# Form to collect features based on model specifications
st.markdown("### 📋 Student Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", options=["Female", "Male"], index=0)
    age = st.number_input("Age", min_value=10, max_value=100, value=18, step=1)
    study_hours = st.number_input("Study Hours per Week", min_value=0.0, max_value=168.0, value=15.0, step=0.5)
    attendance_rate = st.slider("Attendance Rate (%)", min_value=0, max_value=100, value=90, step=1)
    parent_education = st.selectbox(
        "Parent Education Level", 
        options=["None/Primary", "High School", "Associate Degree", "Bachelor's", "Master's/PhD"],
        index=1
    )

with col2:
    internet_access = st.selectbox("Has Internet Access?", options=["Yes", "No"], index=0)
    extracurricular = st.selectbox("Participates in Extracurriculars?", options=["Yes", "No"], index=1)
    previous_score = st.number_input("Previous Score", min_value=0.0, max_value=100.0, value=75.0, step=0.5)
    final_score = st.number_input("Final Term/Exam Score Input", min_value=0.0, max_value=100.0, value=78.0, step=0.5)

# Mapping text inputs back to numeric formats matching model expectations
gender_encoded = 1 if gender == "Male" else 0
internet_encoded = 1 if internet == "Yes" else 0
extra_encoded = 1 if extracurricular == "Yes" else 0

# Mapping typical label encoding categories (0 to 4 hierarchy)
edu_mapping = {"None/Primary": 0, "High School": 1, "Associate Degree": 2, "Bachelor's": 3, "Master's/PhD": 4}
edu_encoded = edu_mapping[parent_education]

# Normalize attendance to standard scale if your model trained on 0.0-1.0 rather than 0-100
attendance_scaled = attendance_rate / 100.0 if model.support_vectors_[0][3] <= 1.0 else float(attendance_rate)

# Collate features exactly in order: [gender, age, study_hours_per_week, attendance_rate, parent_education, internet_access, extracurricular, previous_score, final_score]
features = np.array([[
    gender_encoded, 
    float(age), 
    study_hours, 
    attendance_scaled, 
    float(edu_encoded), 
    internet_encoded, 
    extra_encoded, 
    previous_score, 
    final_score
]])

st.markdown("---")

# Predict Trigger
if st.button("Generate Prediction Result"):
    prediction = model.predict(features)[0]
    
    if prediction == "Yes":
        st.markdown(f"""
            <div class="result-card result-yes">
                🎉 Target Achieved / Passed (Prediction: {prediction})
            </div>
        """, unsafe_style_allowed=True)
    else:
        st.markdown(f"""
            <div class="result-card result-no">
                ⚠️ Attention Needed / Risk Detected (Prediction: {prediction})
            </div>
        """, unsafe_style_allowed=True)
