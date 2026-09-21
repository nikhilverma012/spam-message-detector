import streamlit as st

from utils import load_custom_css


st.set_page_config(
    page_title="How It Works",
    page_icon="⚙️",
    layout="wide"
)


load_custom_css()


st.markdown(
    '<div class="main-title">⚙️ How It Works</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Understand the complete machine learning pipeline.'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------
# Step 1
# ---------------------------------------

with st.expander(
    "1️⃣ Data Collection",
    expanded=True
):

    st.write(
        """
        The project uses the SMS Spam Collection dataset.
        Each message is labeled as either ham (normal)
        or spam.
        """
    )


# ---------------------------------------
# Step 2
# ---------------------------------------

with st.expander(
    "2️⃣ Data Cleaning"
):

    st.write(
        """
        Before training, the dataset is cleaned.

        • Missing values are removed.

        • Duplicate messages are removed.

        • Text is converted to lowercase.

        • Labels are converted into numerical values.
        """
    )


# ---------------------------------------
# Step 3
# ---------------------------------------

with st.expander(
    "3️⃣ TF-IDF Vectorization"
):

    st.write(
        """
        Machine learning algorithms cannot directly
        understand raw text.

        TF-IDF converts text into numerical features
        that represent the importance of words and
        word combinations.
        """
    )


# ---------------------------------------
# Step 4
# ---------------------------------------

with st.expander(
    "4️⃣ Logistic Regression"
):

    st.write(
        """
        Logistic Regression is used as the classification
        algorithm.

        The model learns patterns from labeled SMS messages
        and predicts whether a new message belongs to
        the spam or normal class.
        """
    )


# ---------------------------------------
# Step 5
# ---------------------------------------

with st.expander(
    "5️⃣ Model Evaluation"
):

    st.write(
        """
        The dataset is divided into training and testing
        portions.

        The model is trained on the training data and
        evaluated on unseen test data.

        Metrics include:

        • Accuracy

        • Precision

        • Recall

        • F1-score

        • Confusion matrix
        """
    )


# ---------------------------------------
# Step 6
# ---------------------------------------

with st.expander(
    "6️⃣ Prediction"
):

    st.write(
        """
        When the user enters a new message:

        Message
        ↓
        TF-IDF
        ↓
        Machine Learning Model
        ↓
        Spam / Not Spam

        The application also displays the model's
        predicted probability for the selected class.
        """
    )


st.divider()


st.subheader(
    "🧠 Complete Pipeline"
)


st.code(
    """
SMS Dataset
     ↓
Data Cleaning
     ↓
Train/Test Split
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression
     ↓
Model Evaluation
     ↓
Prediction
     ↓
Streamlit Interface
    """,
    language="text"
)