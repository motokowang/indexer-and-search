# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from email.encoders import encode_7or8bit
from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
import requests
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

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__,
            template_folder='templates')

# Performs NLP preprocessing
def nlp_preproc(s, inplace=False):
    s = s.lower()
    s = s.replace('\n','')
    s = s.replace('\x0c','')
    s = ''.join(c for c in s if not c.isdigit())
    s = ''.join(c for c in s if c not in punctuation)
    s = nlp(s)
    
    return s

# Lemmatise str into a dict
def lemmatise(q):
    query_dict = dict()
    for token in q:
        if token.lemma_ not in stopword:
            if token.lemma_ in query_dict:
                query_dict[token.lemma_] += 1
            else:
                query_dict[token.lemma_] = 1
    return query_dict

# Finds the LCD between a df and a dict in terms of columns
def lcd_of_df_and_query_dict(query_dict, df):
    dot_query_words = ['.'+w for w in list(query_dict.keys())]
    dot_col_words = df.columns[0:-2]
    lcd_cols = list()

    for c in dot_col_words:
        if c in dot_query_words:
            lcd_cols.append(c)

    return df[lcd_cols+['File']]

# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/', methods=['POST', 'GET'])
@cross_origin()
def route_root(name=None):
    return render_template('search.html', name=name)

@app.route('/search', methods=['POST', 'GET'])
@cross_origin()
# ‘/search’ URL is bound with search() function.
def search():
    # User's query (e.g., 127.0.0.1:7777/search?q=haiii%20everyone)
    q = request.args.get('q', 
                         default='', 
                         type=str)

    # NLP preprocessing on the user's search query
    q = nlp_preproc(q)

    # Lematise the query
    query_dict = lemmatise(q)

    # Turn into a df
    index_resp = requests.get(url='http://indexer:7776/index').json()
    df = pd.DataFrame(data=index_resp)

    # Which columns appear in the query dictionary?
    df = lcd_of_df_and_query_dict(query_dict, df)

    # We'll compute the total column in a very simple way:
    df = df.infer_objects()
    df['total'] = df.sum(axis=1, 
                         numeric_only=True)

    # Display results
    df = df.sort_values(by='total', 
                               ascending=False)
    return df['File'].to_json()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7777)