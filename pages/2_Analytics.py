import re
from collections import Counter

import streamlit as st
import pandas as pd
import plotly.express as px

from utils import (
    load_custom_css,
    load_dataset
)


st.set_page_config(
    page_title="Analytics",
    page_icon="📊",
    layout="wide"
)


load_custom_css()


st.markdown(
    '<div class="main-title">📊 Dataset Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Explore the SMS dataset used to train the spam classifier.'
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

spam_count = len(
    df[df["label"] == "spam"]
)

ham_count = len(
    df[df["label"] == "ham"]
)

spam_percentage = (
    spam_count / total * 100
)

ham_percentage = (
    ham_count / total * 100
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Messages",
        f"{total:,}"
    )


with col2:

    st.metric(
        "Normal",
        f"{ham_count:,}"
    )


with col3:

    st.metric(
        "Spam",
        f"{spam_count:,}"
    )


with col4:

    st.metric(
        "Spam %",
        f"{spam_percentage:.1f}%"
    )


# ---------------------------------------
# Pie chart
# ---------------------------------------

st.subheader(
    "📌 Spam vs Not Spam"
)


chart_df = pd.DataFrame(
    {
        "Category": [
            "Not Spam",
            "Spam"
        ],

        "Count": [
            ham_count,
            spam_count
        ]
    }
)


fig = px.pie(
    chart_df,
    names="Category",
    values="Count",
    hole=0.45
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ---------------------------------------
# Message length
# ---------------------------------------

df["message_length"] = (
    df["message"]
    .astype(str)
    .str.len()
)


st.subheader(
    "📏 Message Length Distribution"
)


fig2 = px.histogram(
    df,
    x="message_length",
    color="label",
    nbins=50,
    labels={
        "message_length": "Characters",
        "label": "Type"
    }
)


st.plotly_chart(
    fig2,
    use_container_width=True
)


# ---------------------------------------
# Word analysis
# ---------------------------------------

st.subheader(
    "🔤 Most Common Words"
)


def get_words(messages):

    words = []

    for message in messages:

        tokens = re.findall(
            r"\b[a-zA-Z]{3,}\b",
            str(message).lower()
        )

        words.extend(tokens)

    stop_words = {
        "the",
        "and",
        "you",
        "for",
        "are",
        "this",
        "that",
        "your",
        "with",
        "have",
        "will",
        "from",
        "was",
        "not",
        "but",
        "can",
        "has",
        "our"
    }

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return Counter(words)


col1, col2 = st.columns(2)


with col1:

    spam_words = get_words(
        df[df["label"] == "spam"]["message"]
    )

    spam_words_df = pd.DataFrame(
        spam_words.most_common(10),
        columns=["Word", "Frequency"]
    )

    st.write("🚨 Spam Words")

    st.bar_chart(
        spam_words_df.set_index("Word")
    )


with col2:

    ham_words = get_words(
        df[df["label"] == "ham"]["message"]
    )

    ham_words_df = pd.DataFrame(
        ham_words.most_common(10),
        columns=["Word", "Frequency"]
    )

    st.write("✅ Normal Message Words")

    st.bar_chart(
        ham_words_df.set_index("Word")
    )