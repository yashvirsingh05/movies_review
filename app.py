import streamlit as st
import pickle
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
import string

tfid=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Movies Review Classifier")

input_sms=st.text_input("Please enter the review related to movie")

print(input_sms)

def transform_text(text):
  text=text.lower()
  text=nltk.word_tokenize(text)
  y=[]
  text=[i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
  ps=PorterStemmer()
  for i in text:
    y.append(ps.stem(i))
  text=" ".join(y)
  return text


if st.button('Predict'):
  transformed = transform_text(input_sms)
  vector_input = tfid.transform([transformed])
  result = model.predict(vector_input)[0]

  if result == 1:
    st.header("Positive Review of Movie")
  else:
    st.header("Negative Review of Movie")


