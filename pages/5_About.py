import streamlit as st

from utils import load_custom_css


st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)


load_custom_css()


st.markdown(
    '<div class="main-title">ℹ️ About the Project</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'A small NLP and Machine Learning application for SMS classification.'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------
# Project description
# ---------------------------------------

st.markdown(
    """
    <div class="info-card">

    <h2>📧 Spam Message Detector</h2>

    <p>
    Spam Message Detector is a Natural Language Processing
    and Machine Learning project that classifies SMS messages
    as Spam or Not Spam.
    </p>

    <p>
    The application uses TF-IDF to transform text into
    numerical features and Logistic Regression to perform
    binary classification.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------
# Technologies
# ---------------------------------------

st.subheader(
    "🛠️ Technologies"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.info(
        "🐍 Python\n\n"
        "Programming language"
    )


with col2:

    st.info(
        "📊 Pandas\n\n"
        "Data processing"
    )


with col3:

    st.info(
        "🧠 Scikit-learn\n\n"
        "Machine Learning"
    )


col4, col5, col6 = st.columns(3)


with col4:

    st.info(
        "📝 NLP\n\n"
        "Text processing"
    )


with col5:

    st.info(
        "🔢 TF-IDF\n\n"
        "Feature extraction"
    )


with col6:

    st.info(
        "🎨 Streamlit\n\n"
        "Web interface"
    )


# ---------------------------------------
# Features
# ---------------------------------------

st.subheader(
    "✨ Features"
)


features = [
    "Real SMS dataset",
    "Natural Language Processing",
    "TF-IDF feature extraction",
    "Machine Learning classification",
    "Spam probability",
    "Interactive analytics",
    "Dataset explorer",
    "Modern multi-page interface"
]


for feature in features:

    st.write(
        f"✅ {feature}"
    )


# ---------------------------------------
# Disclaimer
# ---------------------------------------

st.divider()

st.warning(
    "This application is an educational machine learning "
    "project. Its predictions should not be treated as "
    "a guaranteed determination that a message is safe or malicious."
)


st.caption(
    "AI Spam Message Detector • Data Science Project"
)