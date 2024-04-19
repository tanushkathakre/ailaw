import streamlit as st
from gensim.summarization import summarize 
#from bert import predict_winning_percentage  # Import the prediction function from prediction_model.py
from section import *
# from bert import loaded_model
# Function for text summarization using TextRank
def summarize_text(text):
    summarized = summarizer.summarize(text)
    return summarized

st.set_page_config(
    page_title="JudgerAI",
    page_icon="🧊",
    layout="wide")

# application header
left_col, right_col = st.columns(2)

with left_col:
    st.header("Summarize your Case")
    with st.expander(label="Case Summarizer", expanded=True):
        option = st.selectbox(
            'Choose a Method for Entering your Case Facts',
            ('Upload a File', 'Write it Myself'))

        if option == "Upload a File":
            uploaded_file = st.file_uploader(
                label='Upload your Case File (.txt)', type=['txt'])
            if uploaded_file is not None:
                content = uploaded_file.getvalue().decode("utf-8")
                summarized_case_facts = summarize_text(content)

                st.write('<p class="bold-text"> Summarized Case Facts </p>', unsafe_allow_html=True)
                st.success(summarized_case_facts)

                summarized_case_facts_file = summarized_case_facts

                btn = st.download_button(
                    label="Download",
                    data=summarized_case_facts_file,
                    file_name="summarized_case_facts.txt",
                    mime="file/txt"
                )

                # Add a button to trigger prediction of section and punishment
                if st.button("Predict Section and Punishment"):
                    predicted_section, predicted_punishment = predict_section_and_punishment(content)
                    st.success(f"Predicted Section: {predicted_section}")
                    st.success(f"Predicted Punishment: {predicted_punishment}")

with right_col:
    st.header("Predict the Outcome")

    # input form
    with st.expander(label="Case Outcome Predictor", expanded=True):
        option = st.selectbox(
            'Choose a Method for Entering your Case Information',
            ('Upload a File', 'Write it Myself'))

        if option == 'Upload a File':
            uploaded_file = st.file_uploader(
                label='Upload your Case File (.txt)', type=['txt'], key="prediction_case_uploader")
            if uploaded_file is not None:
                content = uploaded_file.getvalue().decode("utf-8")
                summarized_case_facts = summarize_text(content)

                st.write('<p class="bold-text"> Summarized Case Facts </p>', unsafe_allow_html=True)
                st.success(summarized_case_facts)

                summarized_case_facts_file = summarized_case_facts

                btn = st.download_button(
                    label="Download",
                    data=summarized_case_facts_file,
                    file_name="summarized_case_facts.txt",
                    mime="file/txt"
                )

        option = st.selectbox(
            "Select Model",
            ( "BERT")
        )

        col1, col2 = st.columns(2)

        with col1:
            petitioner = st.text_input(
                label="Petitioner", key="petitioner")

        with col2:
            respondent = st.text_input(
                label="Respondent", key="respondent")

        global facts
        facts = st.text_area(label="Case Facts",
                             height=300, key="facts")

        submitted = st.button(label="Predict")

        if submitted:
            if petitioner and respondent and facts:
                # For prediction, call the BERT model prediction logic here
                # Import tokenizer and other necessary functions from the prediction_model.py file
                print("hi i will add predictor later")
            else:
                st.error("Please, fill in all fields!")
