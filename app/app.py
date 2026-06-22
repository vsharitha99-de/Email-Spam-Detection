"""
app.py

Streamlit app: paste an email/SMS, get a Spam / Not Spam prediction.

Run with:
    streamlit run app/app.py
"""

import pickle
import sys
from pathlib import Path

import streamlit as st

# Make src/ importable so we can reuse the same cleaning function
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))
from preprocess import clean_text  # noqa: E402

MODELS_DIR = ROOT / "models"

st.set_page_config(
    page_title="Spam Detector",
    page_icon="📧",
    layout="centered",
)

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.stTextArea textarea {
    border-radius: 12px;
    border: 2px solid #dcdcdc;
    font-size: 16px;
}

.stButton button {
    width: 100%;
    border-radius: 12px;
    height: 3rem;
    font-weight: 600;
    font-size: 16px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 15px;
}

.footer {
    text-align: center;
    color: gray;
    font-size: 14px;
    margin-top: 40px;
}
.header-card {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    background: #111827;
    color: white;
    margin-bottom: 20px;
    border: none;
}
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_artifacts():
    with open(MODELS_DIR / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open(MODELS_DIR / "model.pkl", "rb") as f:
        model = pickle.load(f)

    model_name = (MODELS_DIR / "model_name.txt").read_text().strip()

    return vectorizer, model, model_name


# ---------------- HEADER ---------------- #
st.markdown("""
<div class="header-card">
    <h1>📧 Spam Email & SMS Detector</h1>
    <p style="font-size:18px;color:gray;">
        AI-Powered Message Classification using Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- MODEL LOADING ---------------- #
try:
    vectorizer, model, model_name = load_artifacts()

except FileNotFoundError:
    st.error(
        "No trained model found. Run "
        "`python src/train.py --data data/spam.csv` "
        "first."
    )
    st.stop()

# ---------------- SIDEBAR ---------------- #
st.sidebar.markdown("## 🤖 Model Information")
st.sidebar.success("Model Loaded Successfully")
st.sidebar.info(f"Active Model: {model_name}")

metrics_path = MODELS_DIR / "metrics.csv"

if metrics_path.exists():
    import pandas as pd

    st.sidebar.markdown("### 📊 Performance")
    metrics_df = pd.read_csv(metrics_path, index_col=0)
    st.sidebar.dataframe(metrics_df)

# ---------------- INPUT ---------------- #
message = st.text_area(
    "✉️ Enter Email or SMS Message",
    height=200,
    placeholder="Paste your email or SMS message here..."
)

check_clicked = st.button(
    "🔍 Analyze Message",
    type="primary",
    use_container_width=True
)

# ---------------- PREDICTION ---------------- #
if check_clicked:

    if not message.strip():
        st.warning("⚠️ Please enter a message first.")

    else:

        cleaned = clean_text(message)

        vec = vectorizer.transform([cleaned])

        prediction = model.predict(vec)[0]

        confidence = None

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vec)[0]
            classes = list(model.classes_)
            confidence = proba[classes.index(prediction)]

        st.divider()

        if prediction == "spam":

            st.markdown("""
            <div class="result-box"
                style="background:#dc2626;border:1px solid #b91c1c;color:white;">
                <h2>🚨 SPAM DETECTED</h2>
                <p>This message appears suspicious and may contain spam content.</p>
            </div>
            """, unsafe_allow_html=True)
        else:

            st.markdown("""
            <div class="result-box"
                style="background:#059669;border:1px solid #047857;color:white;">
                <h2>✅ NOT SPAM</h2>
                <p>This message appears legitimate and safe.</p>
            </div>
            """, unsafe_allow_html=True)
        # else:

        #         st.markdown("""
        #         <div class="result-box"
        #             style="background:#e8fff0;border:1px solid #9be7b1;color:black;">
        #             <h2>✅ NOT SPAM</h2>
        #             <p>This message appears legitimate and safe.</p>
        #         </div>
        #         """, unsafe_allow_html=True)
            

        if confidence is not None:
            st.metric(
                label="🎯 Prediction Confidence",
                value=f"{confidence:.1%}"
            )

# ---------------- FOOTER ---------------- #
st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
Built with ❤️ using Python, Scikit-Learn & Streamlit<br>
NLP and Machine Learning Portfolio Project
</div>
""", unsafe_allow_html=True)