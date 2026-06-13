"""
Generate a synthetic dataset of Amazon-style product reviews labeled with
3-class sentiment (Positive / Neutral / Negative).

Why synthetic data?
This project is built for full reproducibility without requiring a large
external download. The generator combines realistic review templates,
product names, and aspect-based phrases (quality, delivery, price, etc.)
with controlled sentiment labels, then adds noise (mixed-sentiment clauses)
so the task isn't trivially easy -- this is what creates the gap between
the VADER rule-based baseline and a trained classifier.

Output: data/reviews.csv with columns [review, rating, sentiment]
"""

import random
import pandas as pd

random.seed(42)

products = [
    "wireless earbuds", "laptop stand", "coffee maker", "running shoes",
    "bluetooth speaker", "yoga mat", "office chair", "phone case",
    "backpack", "smart watch", "desk lamp", "blender", "water bottle",
    "gaming mouse", "kitchen knife set", "air fryer", "winter jacket",
    "wireless charger", "bookshelf", "electric toothbrush",
]

positive_phrases = [
    "exceeded my expectations",
    "works perfectly and feels premium",
    "arrived earlier than expected and well packaged",
    "great value for the price",
    "exactly as described, very happy with this purchase",
    "build quality is excellent and sturdy",
    "I would definitely recommend this to others",
    "battery life is amazing, lasts all day",
    "customer service was helpful and quick to respond",
    "easy to set up and use right away",
]

negative_phrases = [
    "stopped working after a few days",
    "feels cheap and poorly made",
    "arrived damaged and the packaging was torn",
    "not worth the price at all",
    "completely different from the description",
    "very disappointed with the build quality",
    "would not recommend this to anyone",
    "battery drains way too quickly",
    "customer service never responded to my emails",
    "difficult to set up and the instructions were unclear",
]

neutral_phrases = [
    "it's okay, does the job but nothing special",
    "average quality, about what I expected for the price",
    "some features are good, others could be improved",
    "packaging was fine, delivery took a bit longer than expected",
    "works as advertised but the design feels a little outdated",
    "decent product overall, neither great nor bad",
    "price is reasonable for what you get",
    "battery life is acceptable but not impressive",
    "customer service was responsive but slow to resolve the issue",
    "setup was straightforward, though the manual could be clearer",
]

intensifiers = ["Honestly, ", "Overall, ", "To be fair, ", "So far, ", ""]


def make_review(sentiment):
    """Build a review string for the given sentiment, with light noise
    from the other classes to avoid trivially easy classification."""
    n_sentences = random.randint(2, 3)
    sentences = []

    if sentiment == "Positive":
        primary, secondary = positive_phrases, neutral_phrases
        primary_share = 0.8
    elif sentiment == "Negative":
        primary, secondary = negative_phrases, neutral_phrases
        primary_share = 0.8
    else:  # Neutral
        primary, secondary = neutral_phrases, random.choice([positive_phrases, negative_phrases])
        primary_share = 0.6

    for _ in range(n_sentences):
        pool = primary if random.random() < primary_share else secondary
        sentences.append(random.choice(pool))

    product = random.choice(products)
    intro = f"{random.choice(intensifiers)}This {product} {sentences[0]}."
    rest = " ".join(f"{s.capitalize()}." for s in sentences[1:])
    return f"{intro} {rest}".strip()


# Star rating consistent with sentiment label (adds realism, mirrors how
# real e-commerce sentiment is often derived from ratings)
rating_map = {
    "Positive": [4, 5],
    "Neutral": [3],
    "Negative": [1, 2],
}

rows = []
n_per_class = 600  # 600 * 3 = 1800 reviews total
for sentiment in ["Positive", "Neutral", "Negative"]:
    for _ in range(n_per_class):
        review = make_review(sentiment)
        rating = random.choice(rating_map[sentiment])
        rows.append({"review": review, "rating": rating, "sentiment": sentiment})

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("data/reviews.csv", index=False)

print(f"Generated {len(df)} reviews -> data/reviews.csv")
print(df["sentiment"].value_counts())
print("\nSample reviews:")
for s in ["Positive", "Neutral", "Negative"]:
    print(f"\n[{s}] {df[df.sentiment == s].iloc[0]['review']}")
