import streamlit as st

from utils import (
    load_custom_css,
    load_dataset
)


st.set_page_config(
    page_title="Dataset",
    page_icon="📁",
    layout="wide"
)


load_custom_css()


st.markdown(
    '<div class="main-title">📁 Dataset Explorer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Explore the SMS Spam Collection used by the project.'
    '</div>',
    unsafe_allow_html=True
)


df = load_dataset()


if df is None:

    st.error(
        "Dataset not found."
    )

    st.stop()


# ---------------------------------------
# Statistics
# ---------------------------------------

total = len(df)

spam = len(
    df[df["label"] == "spam"]
)

ham = len(
    df[df["label"] == "ham"]
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Total Records",
        f"{total:,}"
    )


with col2:

    st.metric(
        "Normal",
        f"{ham:,}"
    )


with col3:

    st.metric(
        "Spam",
        f"{spam:,}"
    )


st.divider()


# ---------------------------------------
# Dataset preview
# ---------------------------------------

st.subheader(
    "👀 Dataset Preview"
)


preview = df.head(50)


st.dataframe(
    preview,
    use_container_width=True,
    height=450
)


# ---------------------------------------
# Dataset information
# ---------------------------------------

st.subheader(
    "📌 Dataset Information"
)


st.write(
    """
The SMS Spam Collection is a labeled SMS dataset.

Each record contains:

• label — ham or spam

• message — the SMS text

The data is used to train and evaluate the
machine learning classifier.
"""
)


# ---------------------------------------
# Class distribution
# ---------------------------------------

st.subheader(
    "📊 Class Distribution"
)


st.bar_chart(
    df["label"].value_counts()
)