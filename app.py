import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import streamlit as st
import pickle
import sklearn

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    valid_text = []
    for i in text:
        if i.isalnum():
            if i not in stopwords.words('english') and i not in string.punctuation:
                valid_text.append(i)
    stem_words = []
    for i in valid_text:
        stem_words.append(ps.stem(i))

    return " ".join(stem_words)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load((open('model.pkl', 'rb')))

st.title('Email/SMS spam classifier')
sms = st.text_area("Enter the messege")

# the Framework will be as below
# Preprocess
# Vectorize
# Predict
# Display

if st.button("predict"):
    # Preprocess
    transform_sms = transform_text(sms)

    # Vectorize the text
    vector_text = tfidf.transform([transform_sms])

    # Predict
    result = model.predict(vector_text)[0]

    # Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
