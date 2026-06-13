# Customer Review Sentiment Analyzer

A 3-class sentiment classifier (Positive / Neutral / Negative) for Amazon-style
product reviews, comparing a rule-based baseline, a classical ML pipeline, and
a fine-tuned transformer.

## Dataset

`data/reviews.csv` — 1,800 product reviews (600 per class), generated with
`generate_data.py` using realistic aspect-based review templates (quality,
delivery, price, packaging, durability, customer service) combined with
star-rating-derived sentiment labels and intentional cross-class noise.

> This dataset is synthetically generated for full reproducibility without
> requiring a large external download. The generation script is included so
> the data can be regenerated or extended with a real dataset (e.g., Amazon
> Reviews) using the same labeling scheme (ratings 4-5 → Positive, 3 → Neutral,
> 1-2 → Negative).

## Approach & Results

### 1. VADER Rule-Based Baseline
Each review is scored with VADER's compound sentiment score and mapped to a
class using standard thresholds (≥0.05 → Positive, ≤-0.05 → Negative, else
Neutral).

**Accuracy: 54.83%** — VADER struggles most with the Neutral class (recall 0.07),
since rule-based lexicons aren't designed to detect "mixed" or "average" sentiment.

### 2. TF-IDF + Gradient Boosting
TF-IDF features (unigrams + bigrams, 3,000 max features, stopwords removed) fed
into a `GradientBoostingClassifier` (200 estimators, depth 3).

**Accuracy: 76.94%** — a +22 point improvement over the VADER baseline,
with balanced precision/recall across all three classes.

| Model                        | Accuracy |
|-------------------------------|----------|
| VADER (rule-based baseline)   | 54.83%   |
| **TF-IDF + Gradient Boosting** | **76.94%** |

### 3. Fine-tuned DistilBERT (optional, GPU recommended)
`distilbert_finetune.py` fine-tunes `distilbert-base-uncased` on the same
train/test split for 3-class classification and outputs per-prediction
confidence scores. This requires downloading pretrained weights from Hugging
Face, so run it on **Google Colab with a free GPU runtime**:

```bash
!pip install transformers torch
python distilbert_finetune.py
```

It will print test accuracy and save the fine-tuned model to
`model/distilbert-sentiment/`.

## Live Demo

A Streamlit app (`app.py`) lets you type a review and see the predicted
sentiment with confidence scores from the trained Gradient Boosting model.

```bash
streamlit run app.py
```

To deploy: push this repo to GitHub, then deploy for free on
[Streamlit Community Cloud](https://share.streamlit.io).

## Project Structure

```
sentiment-analyzer/
├── data/
│   ├── reviews.csv              # full labeled dataset
│   ├── train_reviews.csv / train_labels.csv
│   └── test_reviews.csv / test_labels.csv
├── outputs/
│   ├── model.pkl                 # trained Gradient Boosting model
│   ├── vectorizer.pkl            # fitted TF-IDF vectorizer
│   ├── classical_results.json    # VADER + GB accuracy summary
│   └── gb_confusion_matrix.png
├── generate_data.py              # synthetic dataset generator
├── classical_models.py           # VADER baseline + Gradient Boosting
├── distilbert_finetune.py        # DistilBERT fine-tuning (run on Colab/GPU)
├── app.py                        # Streamlit demo
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
python generate_data.py        # creates data/reviews.csv
python classical_models.py     # VADER baseline + Gradient Boosting
streamlit run app.py           # interactive demo
```

## Tech Stack

Python · Pandas · Scikit-learn · NLTK/VADER · TF-IDF · Gradient Boosting ·
HuggingFace Transformers (DistilBERT) · Streamlit
