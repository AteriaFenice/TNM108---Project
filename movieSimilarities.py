# MASKININLÄRNING FÖR SOCIALA MEDIER TNM108
# PROJECT: Use NLP algorithm to decide what movie plots are most similar to eachother
# AUTHORS: Group 14, Malva Jansson and Maria Brunned
#____________________________________________________________________________________

import numpy as np
import pandas as pd
import re 
import nltk

# Normalize function
from nltk.stem.snowball import SnowballStemmer

# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import FunctionTransformer


# Import database
movies_db = pd.read_csv('imdb (1000 movies) in june 2022.csv')

# Print database
#print(movies_db.columns)

# Print summary column 
#print(movies_db['DETAIL ABOUT MOVIE\r\n'])

# Normalize function that will tokenize, stem and filter special characters
stemmer = SnowballStemmer("english", ignore_stopwords=False)

def normalize(X):
    normalized = []
    for x in X:
        words = nltk.word_tokenize(x)
        normalized.append(' '.join([stemmer.stem(word) for word in words if re.match('[a-zA-Z]+', word)]))
        return normalized
    
# Define pipline
pipe = Pipeline([
    # Apply the normalize function
    ('normalize', FunctionTransformer(normalize, validate=False)),

    # Vectorize all the documents using Bag of Words
    # This step also removes stopwords
    ('counter_vectorizer', CountVectorizer(
        max_features=200000,
        max_df = 1.0 ,min_df=0.9, stop_words='english',
        ngram_range=(1,3)
    )),

    # Transform the Bag of Words into a TF-IDF matrix
    ('tfidf_transform', TfidfTransformer())
])

# Creating the TF-IDF matrix
tfidf_matrix = pipe.fit_transform([x for x in movies_db['DETAIL ABOUT MOVIE\r\n']])

print(tfidf_matrix.toarray())







