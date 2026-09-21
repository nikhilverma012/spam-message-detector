# 📧 AI Spam Message Detector

A Machine Learning and Natural Language Processing project that
classifies SMS messages as **Spam** or **Not Spam**.

The project uses **TF-IDF Vectorization** and **Logistic Regression**
and provides an interactive multi-page **Streamlit** interface.

---

## 🚀 Features

- 🔍 Spam / Not Spam message detection
- 📊 Dataset analytics
- 📁 Dataset explorer
- 📈 Message statistics and charts
- 🤖 Machine Learning prediction
- 🎯 Prediction confidence
- ⚙️ Machine Learning pipeline explanation
- 🎨 Multi-page Streamlit interface
- 💾 Saved ML model using Joblib

---

## 🧠 Machine Learning

The project uses:

- **TF-IDF Vectorization** — converts text into numerical features
- **Logistic Regression** — classifies messages into Spam or Not Spam

### Pipeline

```text
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
Spam Prediction
     ↓
Streamlit Interface