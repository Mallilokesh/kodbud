import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------
# Load Dataset
# ---------------------------
df = pd.read_csv("spam.csv", sep="\t", header=None, names=["label", "message"])

print(df.head())

# ---------------------------
# Convert Labels
# ham = 0
# spam = 1
# ---------------------------
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# ---------------------------
# Features and Labels
# ---------------------------
X = df["message"]
y = df["label"]

# ---------------------------
# Split Dataset
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------
# TF-IDF Vectorization
# ---------------------------
vectorizer = TfidfVectorizer(stop_words="english")

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ---------------------------
# Train Naive Bayes Model
# ---------------------------
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ---------------------------
# Predictions
# ---------------------------
y_pred = model.predict(X_test_tfidf)

# ---------------------------
# Accuracy
# ---------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------------------------
# Test New Messages
# ---------------------------
while True:
    msg = input("\nEnter a message (type 'exit' to quit): ")

    if msg.lower() == "exit":
        print("Program Ended.")
        break

    msg_vector = vectorizer.transform([msg])

    prediction = model.predict(msg_vector)

    if prediction[0] == 1:
        print("Prediction: 🚨 SPAM")
    else:
        print("Prediction: ✅ NOT SPAM")