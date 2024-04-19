from bert import *
import streamlit as st # Import the prediction function
import os

os.environ['TF_USE_LEGACY_KERAS'] = 'True'

# Load BERT model (assuming it's already initialized elsewhere)  # Replace with the appropriate method to load your BERT model

def main():
    st.title("Winning Percentage Predictor")

    # File uploader
    uploaded_file = st.file_uploader("Upload your case file (.txt)", type=["txt"])
    if uploaded_file is not None:
        # Read uploaded file
        file_contents = uploaded_file.getvalue().decode("utf-8")

        # Get respondent and petitioner names
        respondent = st.text_input("Enter respondent's name:")
        petitioner = st.text_input("Enter petitioner's name:")

        # Predict winning percentages
        if st.button("Predict"):
            respondent_percentage, petitioner_percentage = predict_winning_percentage(the_model_1_2, file_contents)
            st.write(f'Respondent winning percentage: {respondent_percentage:.2f}%')
            st.write(f'Petitioner winning percentage: {petitioner_percentage:.2f}%')

if __name__ == "__main__":
    main()
