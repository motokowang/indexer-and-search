# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from email.encoders import encode_7or8bit
from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin

# -----
import pandas as pd
import json
import subprocess as sp
import spacy
from string import punctuation
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from os import listdir, mkdir
from os.path import isfile, join
import os

nlp = spacy.load('en_core_web_lg')
stopword = stopwords.words('english')

# Cache file name
CACHE_FILE_NAME = 'cache.csv'
CACHE_FILE_PATH = '/app/'

# Where the JSON test cases are uploaded
JSON_TEST_CASE_PATH = '/app/json_test_cases'

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)

# Lemmatise str into a dict
def lemmatise(s):
    s = s.lower()
    s = s.replace('\n','')
    s = s.replace('\x0c','')
    s = ''.join(c for c in s if not c.isdigit())
    s = ''.join(c for c in s if c not in punctuation)
    s = nlp(s)

    return s

# For a given file, append:
# 1. File name as the key of a dict
# 2. Contents as the value of a dict
def append_file_contents_to_dict(file,  d):
    error = False
    try:
        f = open(file)
        f = f.read()
        j = json.loads(f)
        txt = j.get('text')

        # Lemmitise
        txt = lemmatise(txt)
        lemma_t = str()
        for t in txt:
            lemma_t += (t.lemma_ + ' ')
            print('--', t.lemma_, 'added to txts')

        # Create a dictionary based on the txts, or the complete contents of that literary work as lemmas
        d[file] = lemma_t
    except:
        # Case where the JSON is badly formed or something else
        print('Bad JSON candidate')
        error = True
    return d, error

# Use CountVectoriser to do TF_IDF computations. Here's the place where we'd daisy chain other ML models
# or algos, if we have time. If here is where we stop, then the vectorisation tf_idf_vector, which can be
# used to construct a df
def compute_tf_idf_vector(files_contents):
    cv = CountVectorizer()

    word_count_vector = cv.fit_transform(files_contents)
    tfidf_transformer = TfidfTransformer(smooth_idf=True, 
                                        use_idf=True) 
    tfidf_transformer.fit(word_count_vector)
    df_idf = pd.DataFrame(tfidf_transformer.idf_, 
                        index=cv.get_feature_names_out(), 
                        columns=["idf_weights"]) 
    df_idf.sort_values(by=['idf_weights'], 
                    ascending=True)
    count_vector = cv.transform(files_contents)
    tf_idf_vector = tfidf_transformer.transform(count_vector)
    feature_names = cv.get_feature_names_out()

    return tf_idf_vector, feature_names

@app.route('/index', methods=['POST', 'GET'])
@cross_origin()
def index():
    # Read the JSON test cases (.json) files
    os.chdir(JSON_TEST_CASE_PATH)
    
    # Basically ls that directory to get the list of such test case files
    files = [f for f in listdir() if isfile(join(f))]# and f[-5:-1] == '.json']
    
    # Is there a cache file?
    if CACHE_FILE_NAME in listdir(CACHE_FILE_PATH):
        os.chdir(CACHE_FILE_PATH)
        print('Cache found')
        return pd.read_csv(CACHE_FILE_NAME).to_json()
    else:
        # Create a dictionary based on the txts, or the complete contents of that literary work as lemmas
        txts_dict = dict()

        # For each file, do NLP preprocessing, such as lowercasing, lemmitisation, so forth:
        for file in files:
            txts_dict, _ = append_file_contents_to_dict(file, txts_dict)

        # TF_IDF vector
        tf_idf_vector, feature_names = compute_tf_idf_vector(txts_dict.values())

        # Transform TF_IDF vector into a df
        df = pd.DataFrame(tf_idf_vector.T.todense(), 
                        index=['.'+w for w in feature_names])
        df = df.transpose()
        df['File'] = txts_dict.keys()
        
        # Save df as cache
        df.to_csv(CACHE_FILE_PATH+CACHE_FILE_NAME,
                index=False)

        return df.to_json()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7776)