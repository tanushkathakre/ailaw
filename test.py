import streamlit as st
import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def process_sentence(input_sentence):
    # Tokenize input sentence
    tokens = tokenizer.tokenize(input_sentence)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Convert token IDs to PyTorch tensor
    token_tensor = torch.tensor([token_ids])

    # Run input through BERT model
    with torch.no_grad():
        outputs = model(token_tensor)

    # Extract the hidden states
    hidden_states = outputs[0]

    return tokens, hidden_states

# Streamlit UI
st.title("BERT Sentence Processor")
input_sentence = st.text_input("Enter a sentence:")
if input_sentence:
    tokens, hidden_states = process_sentence(input_sentence)
    st.write("Tokenized sentence:", tokens)
    st.write("Hidden states shape:", hidden_states.shape)
