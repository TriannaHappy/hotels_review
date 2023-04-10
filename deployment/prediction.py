# Load the Models
import streamlit as st
import pandas as pd
import re
import string
import numpy as np
from tensorflow import keras
from keras.models import load_model
# import libraries and download package of nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download package nltk
nltk.download(['stopwords', 'punkt','averaged_perceptron_tagger',
               'vader_lexicon', 'wordnet', 'omw-1.4'])

# load the model
model_load1=load_model('model_lstm_hotel')


def run():
    with st.form(key='reviews from visitors'):
        property = st.text_input('Property Name', help='Name of the hotel')
        Rtitle = st.text_input('Review Title')
        Rtext = st.text_input('Review Text')
        location = st.text_input('Location Of The Reviewer')
        date_r = st.text_input('Date Of Review')
        
        st.markdown('---')
        submitted = st.form_submit_button('Predict')

    data_inf = {
        'Property Name' : property,
        'Review Title' : Rtitle,
        'Review Text' : Rtext,
        'Location Of The Reviewer' : location,
        'Date Of Review' : date_r,
    }

    df=pd.DataFrame([data_inf])
    st.dataframe(df)

    if submitted:
        # Get the rate and review text only to be analyzed
        data=df[['Review Text' ]]
        data.rename({'Review Rating': 'rate', 'Review Text': 'text'}, axis=1, inplace=True)
        
        # Change to lowercase
        teks = data.text[0]
        teks_lower = teks.lower()

        # check punctuation
        punctuation = re.findall(r'[^\w\s]', teks_lower)

        # remove punctuation
        teks_punc = teks_lower.translate(str.maketrans('', '', string.punctuation))

        # remove Hashtag
        teks_punc = re.sub("#[A-Za-z0-9_]+", " ", teks_punc)

        # remove \n
        teks_punc = re.sub(r"\\n", " ", teks_punc)

        # remove Whitespace
        teks_punc = teks_punc.strip()

        # Remove Emoji, Mathematic symbols (ex : μ), etc
        teks_punc = re.sub("[^A-Za-z\s']", " ", teks_punc)

        # List Stopwords
        stpwds_en = list(set(stopwords.words('english')))

        # tokenize words
        tokens = word_tokenize(teks_punc)
        teks_stopwords = [word for word in tokens if word not in stpwds_en]

        # Define the stemmer and lemmatizer
        stemmer = nltk.stem.PorterStemmer()

        # Stemming the sentence
        stemmed_words = []

        for word in teks_stopwords:
            stemmed_words.append(stemmer.stem(word))
            
        stemmed_sentence = " ".join(stemmed_words)
        stemmed_sentence

        # Predict the text using the model
        y_pred=model_load1.predict(np.array(stemmed_sentence).reshape(-1))
        

        if np.argmax(y_pred)==0:
            st.write("# This visitor will give us rate 1-2 star rating")
        elif np.argmax(y_pred)==1:
            st.write("# This visitor will give us rate 3 star rating")
        elif np.argmax(y_pred)==2:
            st.write("# This visitor will give us rate 4 star rating")
        elif np.argmax(y_pred)==3:
            st.write("# This visitor will give us rate 5 star rating")


        
if __name__ == '__main__':
    run()