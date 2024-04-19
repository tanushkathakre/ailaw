import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
data = pd.read_csv("ipc_sections.csv")

# Data preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if pd.isnull(text):
        return ""
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)


data['Processed_Offense'] = data['Offense'].apply(preprocess_text)

# Feature Engineering
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data['Processed_Offense'])
y = data['Section']

# Model Building
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Prediction
# def predict_section(text):
#     processed_text = preprocess_text(text)
#     vectorized_text = tfidf_vectorizer.transform([processed_text])
#     predicted_section = model.predict(vectorized_text)
#     return predicted_section[0]

# Example usage
def predict_section_and_punishment(text):
    processed_text = preprocess_text(text)
    vectorized_text = tfidf_vectorizer.transform([processed_text])
    predicted_section = model.predict(vectorized_text)[0]
    predicted_punishment = data[data['Section'] == predicted_section]['Punishment'].iloc[0]
    return predicted_section, predicted_punishment
