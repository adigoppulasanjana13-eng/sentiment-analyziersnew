"""
Fine-tune DistilBERT for 3-class sentiment classification
(Positive / Neutral / Negative) and compare against the classical
TF-IDF + Gradient Boosting model.

NOTE: This script downloads pretrained weights from Hugging Face
(`distilbert-base-uncased`), so it needs internet access. Recommended:
run on Google Colab with a GPU runtime (Runtime > Change runtime type > GPU).

Usage:
    python distilbert_finetune.py
"""

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

LABELS = ["Negative", "Neutral", "Positive"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# Load the same train/test split used by the classical models
# (created by classical_models.py)
# ---------------------------------------------------------------------------
train_texts = pd.read_csv("data/train_reviews.csv")["review"].tolist()
train_labels_raw = pd.read_csv("data/train_labels.csv")["sentiment"].tolist()
test_texts = pd.read_csv("data/test_reviews.csv")["review"].tolist()
test_labels_raw = pd.read_csv("data/test_labels.csv")["sentiment"].tolist()

train_labels = [LABEL2ID[l] for l in train_labels_raw]
test_labels = [LABEL2ID[l] for l in test_labels_raw]

print(f"Train size: {len(train_texts)} | Test size: {len(test_texts)}")

# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=64)


class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = ReviewDataset(train_encodings, train_labels)
test_dataset = ReviewDataset(test_encodings, test_labels)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="model/checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=20,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

print("\nFine-tuning DistilBERT...")
trainer.train()

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
eval_results = trainer.evaluate()
print(f"\nDistilBERT test accuracy: {eval_results['eval_accuracy'] * 100:.2f}%")

preds_output = trainer.predict(test_dataset)
preds = np.argmax(preds_output.predictions, axis=-1)
print(classification_report(test_labels, preds, target_names=LABELS))

# ---------------------------------------------------------------------------
# Save model + per-prediction confidence scores for a few examples
# ---------------------------------------------------------------------------
model.save_pretrained("model/distilbert-sentiment")
tokenizer.save_pretrained("model/distilbert-sentiment")

probs = torch.softmax(torch.tensor(preds_output.predictions), dim=1).numpy()
sample_results = []
for i in range(5):
    sample_results.append({
        "review": test_texts[i],
        "true_label": ID2LABEL[test_labels[i]],
        "predicted_label": ID2LABEL[int(preds[i])],
        "confidence": {ID2LABEL[j]: round(float(probs[i][j]), 4) for j in range(3)},
    })

with open("outputs/distilbert_results.json", "w") as f:
    json.dump({
        "test_accuracy": round(eval_results["eval_accuracy"] * 100, 2),
        "sample_predictions": sample_results,
    }, f, indent=2)

print("\nSaved model to model/distilbert-sentiment/ and outputs/distilbert_results.json")
print("Done.")
