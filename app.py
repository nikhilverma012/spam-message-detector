import streamlit as st

from utils import (
    load_custom_css,
    load_dataset
)




st.set_page_config(
    page_title="AI Spam Message Detector",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)



load_custom_css()



with st.sidebar:

    st.markdown(
        """
        <h1 style="text-align:center;">
        📧
        </h1>

        <h2 style="text-align:center;">
        Spam Detector
        </h2>

        <p style="text-align:center;">
        AI-powered SMS classification
        </p>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.success("🟢 ML Model Online")

    st.divider()

    st.caption(
        "Built with Python + NLP + Machine Learning"
    )



df = load_dataset()



st.markdown(
    '<div class="main-title">'
    'AI Spam Message Detector'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Detect spam messages using Natural Language Processing '
    'and Machine Learning.'
    '</div>',
    unsafe_allow_html=True
)



st.markdown(
    """
    <div class="info-card">

    <h2>🛡️ Keep Your Inbox Safe</h2>

    <p>
    This application uses TF-IDF text vectorization
    and Logistic Regression to classify SMS messages
    as Spam or Not Spam.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)



if df is not None:

    total = len(df)

    spam = len(
        df[df["label"] == "spam"]
    )

    ham = len(
        df[df["label"] == "ham"]
    )

else:

    total = 0
    spam = 0
    ham = 0


col1, col2, col3 = st.columns(3)


with col1:

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-number">
        {total:,}
        </div>

        <div class="metric-label">
        Total Messages
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-number">
        {ham:,}
        </div>

        <div class="metric-label">
        Normal Messages
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        f"""
        <div class="metric-card">

        <div class="metric-number">
        {spam:,}
        </div>

        <div class="metric-label">
        Spam Messages
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")



st.markdown(
    '<div class="section-title">🚀 Explore the Application</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


with col1:

    st.page_link(
        "pages/1_Check_Message.py",
        label="🔍 Check Message",
        icon="🔍"
    )


with col2:

    st.page_link(
        "pages/2_Analytics.py",
        label="📊 Analytics",
        icon="📊"
    )


with col3:

    st.page_link(
        "pages/3_Dataset.py",
        label="📁 Dataset",
        icon="📁"
    )


col4, col5 = st.columns(2)


with col4:

    st.page_link(
        "pages/4_How_It_Works.py",
        label="⚙️ How It Works",
        icon="⚙️"
    )


with col5:

    st.page_link(
        "pages/5_About.py",
        label="ℹ️ About Project",
        icon="ℹ️"
    )



st.divider()

st.caption(
    "📧 AI Spam Message Detector • "
    "Data Science Project • Built with Streamlit"
)