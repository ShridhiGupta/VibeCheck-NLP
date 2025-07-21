import pandas as pd
import re
import nltk
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
from nltk.corpus import stopwords

# Preprocess review text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stopwords.words('english'))
    return text

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df['cleaned_review'] = df['Review'].apply(clean_text)
    df['Sentiment'] = df['Sentiment'].map({'positive': 1, 'negative': 0})
    return df

def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_review'], df['Sentiment'], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    return model, vectorizer

def predict_sentiment(model, vectorizer, text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    result = model.predict(vec)[0]
    return "Positive 😊" if result == 1 else "Negative 😠"

if __name__ == "__main__":
    print("📥 Loading and training model...")
    df = load_and_prepare_data("dataset/reviews.csv")
    model, vectorizer = train_model(df)
    print("✅ Model is ready!\n")

    while True:
        user_input = input("💬 Enter a review (or type 'exit' to quit):\n> ")
        if user_input.lower() == 'exit':
            print("👋 Goodbye!")
            break
        prediction = predict_sentiment(model, vectorizer, user_input)
        print("🧠 Sentiment Prediction:", prediction, "\n")
