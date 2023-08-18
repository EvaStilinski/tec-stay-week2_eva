# pip install nltk

# The next code snippet shows how to perform basic sentiment analysis using the VADER sentiment analyzer from NLTK. The sentiment analysis assigns a positive, negative, or neutral sentiment label based on the sentiment score.
# Simulates how a customer service agent would analyze customer feedback to determine the customer's sentiment.

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('vader_lexicon')

# Sample customer feedback
feedbacks = [
    "The product is great, I love it!",
    "The customer service was terrible.",
    "Shipping was fast and efficient.",
    "I had a bad experience with the product quality.",
    "The product arrived broken and unusable.",
    "The product was not as described.",
    "I don't recommend buying this product.",
    "I like the colors and the design."
]

# Tokenize sentences
sentences = [sent_tokenize(feedback) for feedback in feedbacks]

# Initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Analyze sentiments and tokenize words
for i, feedback in enumerate(feedbacks):
    sentiment_scores = sid.polarity_scores(feedback)
    sentiment = "positive" if sentiment_scores[
        'compound'] > 0 else "negative" if sentiment_scores['compound'] < 0 else "neutral"

    words = word_tokenize(feedback)  # Tokenize words for the current feedback

    print(f"Feedback: {feedback}")
    print(f"Sentiment: {sentiment}")
    print("Words:", words)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
