
import tensorflow as tf
import transformers
from tqdm.auto import tqdm
from transformers import BertTokenizer
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report,confusion_matrix
from scipy.special import softmax
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import string
import pandas as pd
import random

#!pip install nltk
import nltk
nltk.download('all')
nltk.download('stopwords')
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

x_train_df = pd.read_csv('X_train.csv')
y_train_df = pd.read_csv('y_train.csv')
x_test_df = pd.read_csv('X_test.csv')
y_test_df = pd.read_csv('y_test.csv')

df = pd.DataFrame(index=range(len(x_train_df)),columns=["Facts","winner_index","first_party","second_party"])
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
for i in range (len(df)):
    df['Facts'].iloc[i] = x_train_df['Facts'].iloc[i]
    df['first_party'].iloc[i] = x_train_df['first_party'].iloc[i]
    df['second_party'].iloc[i] = x_train_df['second_party'].iloc[i]
    df['winner_index'].iloc[i] = y_train_df['winner_index'].iloc[i].item()

df_pred = pd.DataFrame(index=range(len(x_test_df)),columns=["Facts","winner_index","first_party","second_party"])

for i in range (len(df_pred)):
    df_pred['Facts'].iloc[i] = x_test_df['Facts'].iloc[i]
    df_pred['first_party'].iloc[i] = x_test_df['first_party'].iloc[i]
    df_pred['second_party'].iloc[i] = x_test_df['second_party'].iloc[i]
    df_pred['winner_index'].iloc[i] = y_test_df['winner_index'].iloc[i].item()

def generate_training_data(df, ids, tokenizer):
    for i, text in tqdm(enumerate(df['Facts'])):
        tokenized_text = tokenizer.encode_plus(
            text,
            max_length=256,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_tensors='tf'
        )
        ids[i, :] = tokenized_text.input_ids
    return ids


def making_dataset(df):
    X_input_ids = np.zeros((len(df), 256))
    X_input_ids = generate_training_data(df, X_input_ids,tokenizer)

    labels = np.zeros((len(df), 2))
    labels[np.arange(len(df)), df['winner_index'].values.astype(int)] = 1

    dataset = tf.data.Dataset.from_tensor_slices((X_input_ids, labels))
    dataset = dataset.shuffle(10000).batch(32, drop_remainder=True)

    return dataset

target_labels =[]
for i in range(len(y_test_df)):
    target_labels.append(y_test_df['winner_index'].iloc[i].item())

target_labels=np.array(target_labels)
target_labels_pred = target_labels
target_labels_pred

X_input_ids_pred = np.zeros((len(x_test_df), 256))
X_input_ids_pred = generate_training_data(x_test_df, X_input_ids_pred,tokenizer)

def K_fold (train_data):
    n=4
    i=0
    kf = KFold(n_splits=n, random_state=42, shuffle=True)

    for train_index, val_index in kf.split(train_data):

        # splitting Dataframe (dataset not included)
        i=i+1
        train_df = train_data.iloc[train_index]
        val_df = train_data.iloc[val_index]

        if i ==1 :
            train_1 = train_df
            test_1 = val_df

        if i ==2 :
            train_2 = train_df
            test_2 = val_df
        if i ==3 :
            train_3 = train_df
            test_3 = val_df
        if i ==4 :
            train_4 = train_df
            test_4 = val_df

    return train_1,test_1,train_2,test_2,train_3,test_3,train_4,test_4


optim = tf.keras.optimizers.legacy.Adam(learning_rate=1e-5, decay=1e-6)
loss_func = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')
model = TFBertModel.from_pretrained('bert-base-cased')

def creating_model (train_dataset , val_dataset , test_dataset):

    input_ids = tf.keras.layers.Input(shape=(256,), name='input_ids', dtype='int32')

    bert_embds = model.bert(input_ids)[1] # 0 -> activation layer (3D), 1 -> pooled output layer (2D)
    intermediate_layer = tf.keras.layers.Dense(512, activation='relu', name='intermediate_layer')(bert_embds)
    output_layer = tf.keras.layers.Dense(2, activation='softmax', name='output_layer')(intermediate_layer) # softmax -> calcs probs of classes

    sentiment_model = tf.keras.Model(inputs=input_ids, outputs=output_layer)

    sentiment_model.compile(optimizer=optim, loss=loss_func, metrics=[acc])
    hist = sentiment_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10
        )

# model training

dff = df
p = 0.9

dff_pred = df_pred

X_input_ids_pred = np.zeros((len(dff_pred), 256))
X_input_ids_pred = generate_training_data(dff_pred , X_input_ids_pred , tokenizer)

#-----------------------------------

train_1,test_1,train_2,test_2,train_3,test_3,train_4,test_4 = K_fold(dff)

#----------------------------------
train_set_1 = making_dataset(train_1)
train_size = int((len(train_1)/32)*p)
train_1_dataset = train_set_1.take(train_size)
val_1_dataset = train_set_1.skip(train_size)

test_1_dataset = making_dataset(test_1)

train_set_2 = making_dataset(train_2)
train_size = int((len(train_2)/32)*p)
train_2_dataset = train_set_2.take(train_size)
val_2_dataset = train_set_2.skip(train_size)

test_2_dataset = making_dataset(test_2)

train_set_3 = making_dataset(train_3)
train_size = int((len(train_3)/32)*p)
train_3_dataset = train_set_3.take(train_size)
val_3_dataset = train_set_3.skip(train_size)

test_3_dataset = making_dataset(test_3)

train_set_4 = making_dataset(train_4)
train_size = int((len(train_4)/32)*p)
train_4_dataset = train_set_4.take(train_size)
val_4_dataset = train_set_4.skip(train_size)

test_4_dataset= making_dataset(test_4)

#------------------------------------
acc_result = []
the_model_1_2 = creating_model(train_2_dataset,val_2_dataset,test_2_dataset)
y_predict = the_model_1_2.predict(X_input_ids_pred)
y_predict = np.argmax(y_predict,axis=1)
cm = confusion_matrix(target_labels_pred,y_predict)
cm_df = pd.DataFrame(cm,index = ['winner_ZERO','winner_ONE'], columns = ['winner_ZERO','winner_ONE'])
# plt.figure(figsize=(5,4))
# sns.heatmap(cm_df, annot=True)
# plt.title('Confusion Matrix')
# plt.ylabel('Actal Values')
# plt.xlabel('Predicted Values')
# plt.show()
# print('classification_report\n',classification_report(target_labels_pred,y_predict))
# report = classification_report(target_labels_pred,y_predict,output_dict=True)
# acc_result.append(report['accuracy'])


# print('Acuuracy results of the 4 models : ',acc_result)
# print('Avrage accuracy of this probability =' , sum(acc_result)/len(acc_result))

# probability_1 = sum(acc_result)/len(acc_result)


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

x_train_df = pd.read_csv('X_train.csv')
y_train_df = pd.read_csv('y_train.csv')
x_test_df = pd.read_csv('X_test.csv')
y_test_df = pd.read_csv('y_test.csv')

df = pd.DataFrame(index=range(len(x_train_df)),columns=["Facts","winner_index","first_party","second_party"])

for i in range (len(df)):
    df['Facts'].iloc[i] = x_train_df['Facts'].iloc[i]
    df['first_party'].iloc[i] = x_train_df['first_party'].iloc[i]
    df['second_party'].iloc[i] = x_train_df['second_party'].iloc[i]
    df['winner_index'].iloc[i] = y_train_df['winner_index'].iloc[i].item()

df_pred = pd.DataFrame(index=range(len(x_test_df)),columns=["Facts","winner_index","first_party","second_party"])

for i in range (len(df_pred)):
    df_pred['Facts'].iloc[i] = x_test_df['Facts'].iloc[i]
    df_pred['first_party'].iloc[i] = x_test_df['first_party'].iloc[i]
    df_pred['second_party'].iloc[i] = x_test_df['second_party'].iloc[i]
    df_pred['winner_index'].iloc[i] = y_test_df['winner_index'].iloc[i].item()


token = tokenizer.encode_plus(
    df['Facts'].iloc[0],
    max_length=256,
    truncation=True,
    padding='max_length',
    add_special_tokens=True,
    return_tensors='tf'
)

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing.
    text = text.replace('x', '')
    #    text = re.sub(r'\W+', '', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text

def anonymisation (text , first_party , second_party):

    first_words = first_party.split()
    second_words = second_party.split()

    nltk_results = ne_chunk(pos_tag(word_tokenize(text)))

    for nltk_result in nltk_results:
        if type(nltk_result) == Tree:
            name = ''
            for nltk_result_leaf in nltk_result.leaves():
                name += nltk_result_leaf[0] + ' '

                #print ('Type: ', nltk_result.label(), 'Name: ', name)
                if nltk_result.label() != 'PERSON':
                    text = text.replace(name,'anonymized')

    text = text.replace(first_party ,'first_party')
    text = text.replace(second_party , 'second_party')
    for f_w in first_words:
        text = text.replace(f_w , 'first_party')
    for s_w in second_words :
        text = text.replace(s_w , 'second_party')

    return text


def predict_winning_percentage(model, text_file_path):
    # Load and preprocess the text file
    with open(text_file_path, 'r') as file:
        text = file.read()

    # Extract respondent, petitioner, and facts
    respondent = input("Enter respondent's name: ")
    petitioner = input("Enter petitioner's name: ")
    facts = text

    # Anonymize text
    anonymized_text = anonymisation(facts, respondent, petitioner)

    # Preprocess text
    preprocessed_text = clean_text(anonymized_text)

    # Tokenize text
    tokenized_text = tokenizer.encode_plus(
        preprocessed_text,
        max_length=256,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='tf'
    )

    # Predict using the model
    prediction = model.predict(tokenized_text.input_ids)

    # Calculate winning percentage
    winning_percentage_respondent = prediction[0][1] * 100
    winning_percentage_petitioner = prediction[0][0] * 100

    return winning_percentage_respondent, winning_percentage_petitioner

# # Usage example:
respondent_percentage, petitioner_percentage = predict_winning_percentage(the_model_1_2, 'case.txt')
print(f'Respondent winning percentage: {respondent_percentage}%')
print(f'Petitioner winning percentage: {petitioner_percentage}%')
