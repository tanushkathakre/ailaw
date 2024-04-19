import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Load CSV files
x_train_df = pd.read_csv("X_train.csv")
y_train_df = pd.read_csv("y_train.csv")

# Preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize words
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into string
    processed_text = ' '.join(filtered_tokens)
    return processed_text

# Apply preprocessing to 'Facts' column
x_train_df['Facts'] = x_train_df['Facts'].apply(preprocess_text)

# Split data into train and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train_df['Facts'], y_train_df['winner_index'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
x_val_tfidf = tfidf_vectorizer.transform(x_val)

# Train a classifier
classifier = LogisticRegression()
classifier.fit(x_train_tfidf, y_train)

# Evaluate the model
y_pred = classifier.predict(x_val_tfidf)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:")
print(classification_report(y_val, y_pred))

# Predict winning probability for new inputs
def predict_winning_probability(petitioner, respondent, facts):
    # Preprocess the input facts
    processed_facts = preprocess_text(facts)
    # Vectorize the input using the trained TF-IDF vectorizer
    input_tfidf = tfidf_vectorizer.transform([processed_facts])
    # Predict the winning probability
    winning_probability = classifier.predict_proba(input_tfidf)
    return winning_probability

# Example usage
petitioner = "James L. Kisor"
respondent = "Robert L. Wilkie"
facts = "Petitioner James L. Kisor is a veteran of the US Marine Corps..."
winning_probability = predict_winning_probability(petitioner, respondent, facts)
print("Winning Probability:", winning_probability)
