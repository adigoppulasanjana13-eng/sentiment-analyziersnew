"""
Customer Review Sentiment Analyzer — Streamlit Demo
----------------------------------------------------
Loads the trained TF-IDF vectorizer + Gradient Boosting classifier and
predicts sentiment (Positive / Neutral / Negative) for a review, with
per-class confidence scores.

Run locally:
    streamlit run app.py

Deploy:
    Push this repo to GitHub, then deploy on share.streamlit.io
    (Streamlit Community Cloud) or Hugging Face Spaces.
"""

import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Review Sentiment Analyzer", page_icon="💬", layout="centered")

st.title("💬 Customer Review Sentiment Analyzer")
st.write(
    "Paste a product review below and the model will predict whether the "
    "sentiment is **Positive**, **Neutral**, or **Negative**, along with "
    "confidence scores for each class."
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("outputs/model.pkl")
    vectorizer = joblib.load("outputs/vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_artifacts()

review_text = st.text_area(
    "Enter a product review:",
    height=140,
    placeholder="e.g., The product quality was amazing and it arrived earlier than expected!",
)

if st.button("Analyze Sentiment", type="primary"):
    if not review_text.strip():
        st.error("Please enter a review before analyzing.")
    else:
        X = vectorizer.transform([review_text])
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        classes = model.classes_

        label_colors = {"Positive": "green", "Neutral": "orange", "Negative": "red"}
        color = label_colors.get(str(prediction), "blue")

        st.markdown(
            f"### Predicted Sentiment: "
            f"<span style='color:{color}; font-weight:bold;'>{prediction}</span>",
            unsafe_allow_html=True,
        )

        st.subheader("Confidence Scores")
        proba_df = pd.DataFrame({"Sentiment": classes, "Confidence": proba}).set_index("Sentiment")
        st.bar_chart(proba_df)
        st.dataframe(proba_df.style.format({"Confidence": "{:.2%}"}), use_container_width=True)

st.markdown("---")
st.caption(
    "Model: TF-IDF + Gradient Boosting, trained on a labeled set of Amazon-style "
    "product reviews. Part of the Customer Review Sentiment Analyzer project."
)
