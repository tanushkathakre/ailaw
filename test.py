import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score
import torch
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load data
x_train_df = pd.read_csv("x_train.csv")
y_train_df = pd.read_csv("y_train.csv")
x_test_df = pd.read_csv("x_test.csv")
y_test_df = pd.read_csv("y_test.csv")

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
x_test_df['Facts'] = x_test_df['Facts'].apply(preprocess_text)

# Load BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Streamlit app
st.title("Case Winning Probability Predictor")

# User inputs
petitioner = st.text_input("Petitioner")
respondent = st.text_input("Respondent")
facts = st.text_area("Facts")

# Prediction function
def predict_winning_probability(petitioner, respondent, facts):
    # Preprocess the input facts
    processed_facts = preprocess_text(facts)
    # Combine petitioner, respondent, and facts into a single input text
    input_text = f"{petitioner} {respondent} {processed_facts}"
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    # Predict the winning probability using the BERT model
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1).tolist()[0]
    return probabilities

# Predict and display result
if st.button("Predict"):
    if petitioner.strip() == "" or respondent.strip() == "" or facts.strip() == "":
        st.warning("Please enter all inputs.")
    else:
        winning_probabilities = predict_winning_probability(petitioner, respondent, facts)
        st.success(f"Winning Probability for Petitioner: {winning_probabilities[1]}")
        st.success(f"Winning Probability for Respondent: {winning_probabilities[0]}")

# Display accuracy
st.subheader("Model Accuracy:")
x_test_processed = x_test_df['Facts'].apply(preprocess_text)
inputs_test = tokenizer(x_test_processed.tolist(), return_tensors="pt", max_length=512, truncation=True, padding=True)
with torch.no_grad():
    outputs_test = model(**inputs_test)
predicted_labels = torch.argmax(outputs_test.logits, dim=1)
accuracy = accuracy_score(predicted_labels.cpu().numpy(), y_test_df['winner_index'])
st.write(f"Accuracy on Test Set: {accuracy:.4f}")
