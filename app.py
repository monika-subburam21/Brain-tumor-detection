import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sqlite3
import pandas as pd
from datetime import datetime

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# Custom CSS Styling
# ---------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;800&display=swap');
    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #ff9a9e, #fad0c4, #fbc2eb, #a18cd1);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    h1 {
        color: #2c003e;
        text-align: center;
        font-size: 48px !important;
        font-weight: 800 !important;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 22px;
        color: #222;
        font-weight: 600;
        margin-bottom: 25px;
    }
    .prediction-box {
        background: #ffffffdd;
        padding: 25px;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0px 6px 18px rgba(0,0,0,0.2);
        font-size: 22px;
        font-weight: 600;
        color: #2c003e;
    }
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Database Setup
# ---------------------------
conn_users = sqlite3.connect("users.db", check_same_thread=False)
c_users = conn_users.cursor()
c_users.execute('''CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password TEXT,
                name TEXT,
                age INTEGER,
                email TEXT
            )''')
conn_users.commit()

conn_preds = sqlite3.connect("predictions.db", check_same_thread=False)
c_preds = conn_preds.cursor()
c_preds.execute('''CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                image_name TEXT,
                prediction TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )''')
conn_preds.commit()

# ---------------------------
# Helper Functions
# ---------------------------
def signup(username, password, name, age, email):
    try:
        c_users.execute("INSERT INTO users (username, password, name, age, email) VALUES (?, ?, ?, ?, ?)",
                        (username, password, name, age, email))
        conn_users.commit()
        return True
    except:
        return False

def login(username, password):
    c_users.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    return c_users.fetchone()

def save_prediction(user_id, image_name, prediction, confidence):
    c_preds.execute("INSERT INTO predictions (user_id, image_name, prediction, confidence) VALUES (?, ?, ?, ?)",
                    (user_id, image_name, prediction, confidence))
    conn_preds.commit()

def get_predictions(user_id):
    c_preds.execute("SELECT image_name, prediction, confidence, timestamp FROM predictions WHERE user_id=? ORDER BY timestamp DESC", (user_id,))
    return c_preds.fetchall()

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_brain_model():
    model = load_model("brain_tumor_model.h5")
    return model

class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# ---------------------------
# Login & Signup Page
# ---------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None

if "show_prediction" not in st.session_state:
    st.session_state.show_prediction = False

def login_page():
    st.title("🔐 Login / Signup")
    choice = st.radio("Choose an option:", ["Login", "Sign Up"])

    if choice == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user = login(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.user_id = user[0]
                st.session_state.username = user[1]
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    else:  # Sign Up
        name = st.text_input("Full Name")
        age = st.number_input("Age", min_value=1, max_value=120, step=1)
        email = st.text_input("Email")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Sign Up"):
            if signup(username, password, name, age, email):
                st.success("Signup successful! Please log in.")
            else:
                st.error("Username already exists or error occurred.")

# ---------------------------
# Prediction Page
# ---------------------------
def prediction_page():
    st.title("🧠 Brain Tumor Detection")
    st.markdown(f'<div class="subtitle">Welcome {st.session_state.username}!</div>', unsafe_allow_html=True)

    # ---------------------------
    # Awareness Section
    # ---------------------------
    with st.expander("📖 Brain Tumor Awareness", expanded=True):
        st.markdown("""
        **What is a Brain Tumor?**  
        A brain tumor is an abnormal growth of cells in the brain. Some tumors are noncancerous (benign), while others are cancerous (malignant).  
        
        **Why Classification Matters?**  
        Detecting the type of brain tumor (Glioma, Meningioma, Pituitary, or No Tumor) is crucial for planning treatment, surgery, or therapy.  
        
        **Importance of Early Detection:**  
        Early and accurate detection can save lives by preventing tumor growth, reducing complications, and improving recovery outcomes.  
        
        **How Our System Helps:**  
        This system uses AI and MRI images to assist in fast, accurate classification, supporting doctors in decision-making and raising awareness about timely checkups.
        """)

    # ---------------------------
    # Proceed Button
    # ---------------------------
    if not st.session_state.show_prediction:
        if st.button("👉 Proceed to Prediction"):
            st.session_state.show_prediction = True
            st.rerun()
        return

    # ---------------------------
    # File Uploader & Prediction
    # ---------------------------
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded MRI Image", use_container_width=True)

        img_resized = img.resize((150,150))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        model = load_brain_model()
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

        # Save prediction in predictions.db
        save_prediction(st.session_state.user_id, uploaded_file.name, predicted_class, float(confidence))

        st.markdown(
            f"""
            <div class="prediction-box">
                <b>Prediction:</b> {predicted_class}<br>
                <b>Confidence:</b> {confidence:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(figsize=(6,4))
        bars = ax.bar(class_names, predictions[0], color=["#6a1b9a","#f50057","#009688","#ff9800"])
        ax.set_ylabel("Probability", fontsize=12, fontweight="bold")
        ax.set_title("Prediction Confidence", fontsize=14, fontweight="bold")
        ax.bar_label(bars, fmt="%.2f")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("📜 Your Prediction History")
    history = get_predictions(st.session_state.user_id)
    if history:
        for h in history:
            st.write(f"🖼 {h[0]} | **{h[1]}** ({h[2]:.2f}%) | ⏰ {h[3]}")
    else:
        st.info("No previous records found.")

    st.markdown("---")
    st.subheader("⚙️ Admin Dashboard (View Data)")
    if st.session_state.username == "admin":
        st.write("### Users Table")
        df_users = pd.read_sql_query("SELECT * FROM users", conn_users)
        st.dataframe(df_users)

        st.write("### Predictions Table")
        df_preds = pd.read_sql_query("SELECT * FROM predictions", conn_preds)
        st.dataframe(df_preds)

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.username = None
        st.session_state.show_prediction = False
        st.rerun()

# ---------------------------
# Run App
# ---------------------------
if not st.session_state.logged_in:
    login_page()
else:
    prediction_page()

