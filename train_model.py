import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


DATA_PATH = "data/SMSSpamCollection"

MODEL_DIR = "model"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "spam_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)


if not os.path.exists(DATA_PATH):

    print("ERROR: Dataset not found!")

    print(
        "\nPlease put the file 'SMSSpamCollection' "
        "inside the data folder."
    )

    print(
        "\nCorrect location:"
        "\ndata/SMSSpamCollection"
    )

    exit()



print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

print("Dataset loaded successfully!")

print(
    f"Total messages: {len(df)}"
)



print("\nFirst 5 messages:")

print(df.head())


print("\nClass distribution:")

print(df["label"].value_counts())



print("\nCleaning data...")

# Remove missing values
df = df.dropna()

# Remove duplicate messages
df = df.drop_duplicates()

# Convert messages to lowercase
df["message"] = df["message"].str.lower()

# Convert labels
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Remove unknown labels
df = df.dropna()

df["label"] = df["label"].astype(int)


print(
    f"Messages after cleaning: {len(df)}"
)



X = df["message"]

y = df["label"]


print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print(
    f"Training messages: {len(X_train)}"
)

print(
    f"Testing messages: {len(X_test)}"
)



print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(

    lowercase=True,

    stop_words="english",

    max_features=5000,

    ngram_range=(1, 2)
)


X_train_tfidf = vectorizer.fit_transform(
    X_train
)


X_test_tfidf = vectorizer.transform(
    X_test
)


print(
    "TF-IDF transformation completed!"
)



print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000
)


model.fit(
    X_train_tfidf,
    y_train
)


print(
    "Model training completed!"
)



print("\nMaking predictions...")

y_pred = model.predict(
    X_test_tfidf
)



accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n================================")
print("MODEL PERFORMANCE")
print("================================")


print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Not Spam",
            "Spam"
        ]
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)



os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


joblib.dump(
    model,
    MODEL_PATH
)


joblib.dump(
    vectorizer,
    VECTORIZER_PATH
)



print("\n================================")
print("TRAINING COMPLETED!")
print("================================")

print(
    f"\nModel saved to:"
    f"\n{MODEL_PATH}"
)

print(
    f"\nVectorizer saved to:"
    f"\n{VECTORIZER_PATH}"
)

print(
    "\nYou can now run the Streamlit application."
)