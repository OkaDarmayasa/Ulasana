import streamlit as st

# Model
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemover, ArrayDictionary
import string
import regex as re
from sklearn.utils import resample        
from tensorflow.keras.models import load_model
import tensorflow
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

nltk.download('stopwords')
alay_dict = pd.read_csv('new_kamusalay.csv', encoding='latin-1', header=None)
alay_dict = alay_dict.rename(columns={0: 'original', 1: 'replacement'})

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def remove_emojis(text):
  return str(text.encode('ascii', 'ignore'))

def remove_punctuation(text):
    # Make a regular expression that matches all punctuation
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    # Use the regex
    return regex.sub(' ', text)

# stopwords removal
stopword_factory = StopWordRemoverFactory()
stopword = stopword_factory.create_stop_word_remover()
def remove_stopwords_sastrawi(text):
  return stopword.remove(text)

# stemming
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()
def stem_words(text):
  return stemmer.stem(text)

# normalisasi kata tidak baku
slang_dict = pd.read_csv('/content/drive/MyDrive/Penambangan Data Tekstual/Dataset/new_kamusalay.csv', encoding='latin-1', header=None)
slang_dict = slang_dict.rename(columns={0: 'original',
                                      1: 'replacement'})
slang_dict_map = dict(zip(slang_dict['original'], slang_dict['replacement']))
def normalize_slang(text):
  return ' '.join([slang_dict_map[word] if word in slang_dict_map else word for word in text.split(' ')])

def preprocess(text):
  text1 = text.lower()   # case folding
  text4 = remove_emojis(text1)
  text5 = re.sub(r"\d+", "", text4)   # remove numbers
  text6 = text5.replace('\\n',' ')    # hapus karakter '\n'
  text7 = remove_punctuation(text6)
  text8 = normalize_slang(text7)
  text9 = stem_words(text8)
  text10 = remove_stopwords_sastrawi(text9)
  result = text10.strip()   # remove whitespace
  return result

def returnSentiment(score):
    if (score >= 4):
        return "positive"
    elif (score <= 3):
        return "negative"

st.sidebar.subheader('About the App')
st.sidebar.write('Text Classification App with Streamlit using a trained SVM model')
st.sidebar.write("This is just a small text classification app.")

loaded_model = joblib.load("svm_tfidf_model.pkl")

#start the user interface
st.title("Text Classification App")
st.write("Type in your text below and don't forget to press the enter button before clicking/pressing the 'Classify' button")

uploaded_file = st.file_uploader("Upload file with format .csv", type="csv")
my_text = st.text_input("Enter the text you want to classify", "Change this...", max_chars=100, key='to_classify')

df = pd.DataFrame()

if my_text is not None:
    df1['content'] = my_text

if uploaded_file is not None:
    # read csv
    df2 = pd.read_csv(uploaded_file)

df = pd.concat([df1, df2])

with open('best_tfidf_vocabulary_SVM.pkl', 'rb') as f:
    best_vocabulary_SVM = pickle.load(f)

if st.button('Classify', key='classify_button'):  
    # Create a new TfidfVectorizer with the loaded vocabulary
    tfidf_vectorizer_svm = TfidfVectorizer(vocabulary=best_vocabulary_SVM)  # You can adjust the max_features parameter
    X_TFIDF_SVM = tfidf_vectorizer_svm.fit_transform(df["preprocessed"]).toarray()  
    
    y = df["score"].apply(returnSentiment)
    
    result = model_tfidf_SVM.score(X_TFIDF_SVM, y)
    
    st.write("Accuracy: ", result)