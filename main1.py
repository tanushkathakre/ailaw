import predict_winning_percentage from bert
import streamlit as st
respondent_percentage, petitioner_percentage = predict_winning_percentage(the_model_1_2, 'case.txt')
print(f'Respondent winning percentage: {respondent_percentage}%')
print(f'Petitioner winning percentage: {petitioner_percentage}%')
