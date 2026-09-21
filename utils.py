import os
import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "model/spam_model.pkl"
VECTORIZER_PATH = "model/tfidf_vectorizer.pkl"
DATA_PATH = "data/SMSSpamCollection"


def load_custom_css():
    st.markdown(
        """
        <style>

        /* Main background */
        .stApp {
            background: #f6f8fc;
        }

        /* Main title */
        .main-title {
            font-size: 42px;
            font-weight: 800;
            background: linear-gradient(
                90deg,
                #2563eb,
                #7c3aed,
                #db2777
            );
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
        }

        .subtitle {
            color: #64748b;
            font-size: 17px;
            margin-bottom: 25px;
        }

        /* Cards */
        .info-card {
            background: white;
            padding: 24px;
            border-radius: 16px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            margin-bottom: 18px;
        }

        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            border: 1px solid #e2e8f0;
            text-align: center;
            box-shadow: 0 3px 12px rgba(0,0,0,0.05);
        }

        .metric-number {
            font-size: 30px;
            font-weight: 800;
            color: #2563eb;
        }

        .metric-label {
            color: #64748b;
            font-size: 14px;
        }

        /* Prediction cards */
        .spam-result {
            background: #fff1f2;
            border: 2px solid #fb7185;
            padding: 30px;
            border-radius: 18px;
            text-align: center;
            margin-top: 20px;
        }

        .safe-result {
            background: #ecfdf5;
            border: 2px solid #34d399;
            padding: 30px;
            border-radius: 18px;
            text-align: center;
            margin-top: 20px;
        }

        .result-title {
            font-size: 30px;
            font-weight: 800;
        }

        .confidence {
            font-size: 18px;
            margin-top: 10px;
        }

        /* Section heading */
        .section-title {
            font-size: 26px;
            font-weight: 750;
            color: #172554;
            margin-top: 15px;
            margin-bottom: 15px;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #0f172a,
                #172554
            );
        }

        [data-testid="stSidebar"] * {
            color: white;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None, None

    if not os.path.exists(VECTORIZER_PATH):
        return None, None

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


@st.cache_data
def load_dataset():

    if not os.path.exists(DATA_PATH):
        return None

    df = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8"
    )

    return df