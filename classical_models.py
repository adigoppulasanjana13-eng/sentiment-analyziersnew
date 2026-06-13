"""
Step 1: VADER rule-based baseline
Step 2: Classical ML (TF-IDF + Gradient Boosting)

Both are evaluated against the ground-truth 3-class sentiment labels
(Positive / Neutral / Negative) in data/reviews.csv.
"""

import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("data/reviews.csv")
print(f"Total reviews: {len(df)}")
print(df["sentiment"].value_counts())

LABELS = ["Negative", "Neutral", "Positive"]

# ---------------------------------------------------------------------------
# Step 1: VADER rule-based baseline
# ---------------------------------------------------------------------------
print("\n--- VADER Rule-Based Baseline ---")
analyzer = SentimentIntensityAnalyzer()


def vader_label(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


df["vader_pred"] = df["review"].apply(vader_label)
vader_acc = accuracy_score(df["sentiment"], df["vader_pred"])
print(f"VADER baseline accuracy: {vader_acc * 100:.2f}%")
print(classification_report(df["sentiment"], df["vader_pred"], labels=LABELS))

# ---------------------------------------------------------------------------
# Step 2: TF-IDF + Gradient Boosting
# ---------------------------------------------------------------------------
print("\n--- TF-IDF + Gradient Boosting ---")

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["review"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
)

vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

gb_model = GradientBoostingClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42
)
gb_model.fit(X_train, y_train)

preds = gb_model.predict(X_test)
gb_acc = accuracy_score(y_test, preds)
print(f"Gradient Boosting accuracy: {gb_acc * 100:.2f}%")
print(classification_report(y_test, preds, labels=LABELS))

# Confusion matrix
cm = confusion_matrix(y_test, preds, labels=LABELS)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=LABELS, yticklabels=LABELS)
plt.title("Confusion Matrix - Gradient Boosting (TF-IDF)")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.savefig("outputs/gb_confusion_matrix.png", dpi=120)
plt.close()
print("Saved outputs/gb_confusion_matrix.png")

# ---------------------------------------------------------------------------
# Save model, vectorizer, and results summary
# ---------------------------------------------------------------------------
joblib.dump(gb_model, "outputs/model.pkl")
joblib.dump(vectorizer, "outputs/vectorizer.pkl")

# Also save the test split so the DistilBERT script uses the SAME split
X_test_text.to_csv("data/test_reviews.csv", index=False, header=["review"])
y_test.to_csv("data/test_labels.csv", index=False, header=["sentiment"])
X_train_text.to_csv("data/train_reviews.csv", index=False, header=["review"])
y_train.to_csv("data/train_labels.csv", index=False, header=["sentiment"])

summary = {
    "dataset_size": len(df),
    "vader_baseline_accuracy": round(vader_acc * 100, 2),
    "gradient_boosting_accuracy": round(gb_acc * 100, 2),
}
with open("outputs/classical_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved outputs/model.pkl, outputs/vectorizer.pkl, outputs/classical_results.json")
print("Done.")
