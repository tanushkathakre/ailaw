import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import ne_chunk, pos_tag
from nltk.tree import Tree
nltk.download('stopwords')
# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Load data
x_train_df = pd.read_csv('X_train.csv')
y_train_df = pd.read_csv('y_train.csv')
x_test_df = pd.read_csv('X_test.csv')
y_test_df = pd.read_csv('y_test.csv')

# Define cleaning functions
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = text.replace('x', '')
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text

def anonymisation(text, first_party, second_party):
    first_words = first_party.split()
    second_words = second_party.split()

    nltk_results = ne_chunk(pos_tag(word_tokenize(text)))

    for nltk_result in nltk_results:
        if isinstance(nltk_result, Tree):
            name = ''
            for nltk_result_leaf in nltk_result.leaves():
                name += nltk_result_leaf[0] + ' '
                if nltk_result.label() != 'PERSON':
                    text = text.replace(name, 'anonymized')

    text = text.replace(first_party, 'first_party')
    text = text.replace(second_party, 'second_party')
    for f_w in first_words:
        text = text.replace(f_w, 'first_party')
    for s_w in second_words:
        text = text.replace(s_w, 'second_party')

    return text

# Define model
model = TFBertModel.from_pretrained('bert-base-cased') # Load your Keras model here

# Function for predicting winning percentage
def predict_winning_percentage(model, text_file_path):
    with open(text_file_path, 'r') as file:
        text = file.read()

    respondent = input("Enter respondent's name: ")
    petitioner = input("Enter petitioner's name: ")

    anonymized_text = anonymisation(text, respondent, petitioner)
    preprocessed_text = clean_text(anonymized_text)

    tokenized_text = tokenizer.encode_plus(
        preprocessed_text,
        max_length=256,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='tf'
    )

    prediction = model.predict(tokenized_text.input_ids)

    winning_percentage_respondent = prediction[0][1] * 100
    winning_percentage_petitioner = prediction[0][0] * 100

    return winning_percentage_respondent, winning_percentage_petitioner

# Usage example:
respondent_percentage, petitioner_percentage = predict_winning_percentage(model, 'case.txt')
print(f'Respondent winning percentage: {respondent_percentage}%')
print(f'Petitioner winning percentage: {petitioner_percentage}%')
