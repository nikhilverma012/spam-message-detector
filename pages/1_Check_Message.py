import streamlit as st

from utils import (
    load_custom_css,
    load_model
)


st.set_page_config(
    page_title="Check Message",
    page_icon="🔍",
    layout="wide"
)


load_custom_css()


# ---------------------------------------
# Header
# ---------------------------------------

st.markdown(
    '<div class="main-title">🔍 Check Your Message</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Enter an SMS and let the machine learning model analyze it.'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------
# Load model
# ---------------------------------------

model, vectorizer = load_model()


if model is None:

    st.error(
        "Model not found. Please run train_model.py first."
    )

    st.stop()


# ---------------------------------------
# Session state
# ---------------------------------------

if "message_text" not in st.session_state:

    st.session_state.message_text = ""


# ---------------------------------------
# Example buttons
# ---------------------------------------

st.markdown(
    '<div class="section-title">💬 Message Analyzer</div>',
    unsafe_allow_html=True
)


col1, col2, col3 = st.columns(3)


with col1:

    if st.button(
        "🚨 Spam Example",
        use_container_width=True
    ):

        st.session_state.message_text = (
            "Congratulations! You have won a free "
            "cash prize. Call now to claim your reward!"
        )


with col2:

    if st.button(
        "✅ Safe Example",
        use_container_width=True
    ):

        st.session_state.message_text = (
            "Hey, are you coming to college tomorrow?"
        )


with col3:

    if st.button(
        "🧹 Clear",
        use_container_width=True
    ):

        st.session_state.message_text = ""


# ---------------------------------------
# Message box
# ---------------------------------------

message = st.text_area(
    "Enter or paste your message:",
    value=st.session_state.message_text,
    height=180,
    placeholder="Type your SMS message here..."
)


# ---------------------------------------
# Character counter
# ---------------------------------------

st.caption(
    f"Characters: {len(message)}"
)


# ---------------------------------------
# Prediction
# ---------------------------------------

if st.button(
    "🔍 Analyze Message",
    type="primary",
    use_container_width=True
):

    if not message.strip():

        st.warning(
            "Please enter a message first."
        )

    else:

        message_vector = vectorizer.transform(
            [message]
        )

        prediction = model.predict(
            message_vector
        )[0]

        probabilities = model.predict_proba(
            message_vector
        )[0]

        safe_probability = probabilities[0]

        spam_probability = probabilities[1]


        # ---------------------------------------
        # Spam result
        # ---------------------------------------

        if prediction == 1:

            confidence = spam_probability

            st.markdown(
                f"""
                <div class="spam-result">

                <div class="result-title">
                🚨 SPAM MESSAGE
                </div>

                <div class="confidence">
                Confidence: <b>{confidence * 100:.2f}%</b>
                </div>

                <p>
                This message contains patterns
                commonly associated with spam.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                float(confidence)
            )


        # ---------------------------------------
        # Safe result
        # ---------------------------------------

        else:

            confidence = safe_probability

            st.markdown(
                f"""
                <div class="safe-result">

                <div class="result-title">
                ✅ NOT SPAM
                </div>

                <div class="confidence">
                Confidence: <b>{confidence * 100:.2f}%</b>
                </div>

                <p>
                This message appears similar to
                normal messages in the training data.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                float(confidence)
            )


# ---------------------------------------
# Information
# ---------------------------------------

st.divider()

st.info(
    "⚠️ Machine learning predictions are probabilistic. "
    "Do not use the result as the sole basis for deciding "
    "whether a real-world message is safe."
)