import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Download NLTK resources (run once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# Text cleaning function
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = text.lower()  # lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # remove punctuation
    tokens = text.split()  # tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # remove stopwords and lemmatize
    return ' '.join(tokens)

# Train model on example data or your dataset
def train_model(csv_path=None):
    if csv_path is not None:
        df = pd.read_csv(csv_path)
    else:
        # Dummy data if no CSV provided
        data = {
            'review': [
                'I love this product! It is amazing.',
                'This is the worst service ever.',
                'Pretty average experience.',
                'I will never buy this again.',
                'Absolutely fantastic! Highly recommend it.',
                'The service was terrible and disappointing.',
                'Great quality and fast delivery.',
                'Not satisfied with the purchase.',
                'Excellent customer support.',
                'Horrible experience, do not recommend.',
                'Neutral, nothing special.',
                'Best product I have ever used.',
                'Waste of money, very bad.',
                'Good value for money.',
                'Awful, never again.',
                'Superb, exceeded expectations.',
                'Mediocre at best.',
                'Fantastic service.',
                'Disappointing and overpriced.',
                'Highly satisfied.'
            ],
            'sentiment': ['positive', 'negative', 'neutral', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'neutral', 'positive', 'negative', 'positive', 'negative', 'positive', 'neutral', 'positive', 'negative', 'positive']
        }
        df = pd.DataFrame(data)
    df['cleaned_review'] = df['review'].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_review'])

    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    y = df['sentiment'].map(sentiment_map)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model, vectorizer, sentiment_map

# Predict sentiment for a new dataframe of reviews
def predict(model, vectorizer, sentiment_map, reviews):
    cleaned = [clean_text(text) for text in reviews]
    features = vectorizer.transform(cleaned)
    preds = model.predict(features)
    inv_map = {v: k for k, v in sentiment_map.items()}
    return [inv_map[p] for p in preds]
