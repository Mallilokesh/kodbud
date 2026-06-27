# ============================================================
# Task 3: Sentiment Analysis on Product Reviews
# Dataset: Women's Clothing E-Commerce Reviews
# ============================================================

import pandas as pd
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Download stopwords
nltk.download("stopwords")

# ============================================================
# Load Dataset
# ============================================================

# If you renamed your dataset
data = pd.read_csv("reviews.csv")

# OR use this:
# data = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")

print("=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)

print(data.head())

# ============================================================
# Select Required Columns
# ============================================================

data = data[['Review Text', 'Rating']]

# Remove missing reviews
data.dropna(inplace=True)

# Remove Neutral Reviews
data = data[data['Rating'] != 3]

# Create Sentiment Column
data['Sentiment'] = data['Rating'].apply(
    lambda x: "Positive" if x >= 4 else "Negative"
)

print("\nDataset Size :", data.shape)

print("\nSentiment Distribution")
print(data['Sentiment'].value_counts())

# ============================================================
# Text Cleaning
# ============================================================

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

print("\nCleaning Reviews...")

data['Clean Review'] = data['Review Text'].apply(clean_text)

print("Cleaning Completed!")

# ============================================================
# TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)

X = vectorizer.fit_transform(data['Clean Review'])

y = data['Sentiment']

print("TF-IDF Completed!")

# ============================================================
# Train Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ============================================================
# Logistic Regression
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

print("\nTraining Model...")

model.fit(X_train, y_train)

print("Model Trained Successfully!")

# ============================================================
# Prediction
# ============================================================

prediction = model.predict(X_test)

# ============================================================
# Evaluation
# ============================================================

accuracy = accuracy_score(y_test, prediction)

print("\n" + "="*60)
print("MODEL PERFORMANCE")
print("="*60)

print(f"Accuracy : {accuracy*100:.2f}%")

print("\nClassification Report\n")
print(classification_report(y_test, prediction))

print("Confusion Matrix\n")
print(confusion_matrix(y_test, prediction))

# ============================================================
# Custom Prediction
# ============================================================

print("\n" + "="*60)
print("CUSTOM SENTIMENT PREDICTION")
print("="*60)

while True:

    review = input("\nEnter Review (type quit to exit): ")

    if review.lower() == "quit":
        print("\nProgram Ended!")
        break

    cleaned = clean_text(review)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector).max()

    print("\nPredicted Sentiment :", prediction)
    print(f"Confidence : {probability*100:.2f}%")
