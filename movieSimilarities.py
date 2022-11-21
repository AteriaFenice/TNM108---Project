# MASKININLÄRNING FÖR SOCIALA MEDIER TNM108
# PROJECT: Use NLP algorithm to decide what movie plots are most similar to eachother
# AUTHORS: Group 14, Malva Jansson and Maria Brunned
#____________________________________________________________________________________

import numpy as np
import pandas as pd
import re 
import nltk
import math

# Split data
from sklearn.model_selection import train_test_split

# Normalize function
from nltk.stem.snowball import SnowballStemmer

# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer

# Stopwords
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')

# Cosine simluarity
from sklearn.metrics.pairwise import cosine_similarity

# Import database
movies_db = pd.read_csv('imdb (1000 movies) in june 2022.csv')

# Print database
print(movies_db.columns)

# Print summary column 
#print(movies_db['DETAIL ABOUT MOVIE\r\n'])

# Split data into training and test 
#movies_train, movies_test, target_train, target_test = train_test_split(movies_db['DETAIL ABOUT MOVIE\r\n'], movies_db['DETAIL ABOUT MOVIE\r\n'], test_size = 0.20, random_state = 12)

# Create TFI-DF matrix
tfidf_vectorizer = TfidfVectorizer(stop_words=set(stopwords.words('english')))
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_db['DETAIL ABOUT MOVIE\r\n'])

print(tfidf_matrix.shape) # Consits of 1000 rows (movies) and 5715 columns (tf-idf terms)

# Calculate simularity 
search_movie = 223
cos_similarity = cosine_similarity(tfidf_matrix[search_movie], tfidf_matrix)
print(cos_similarity)
print(cos_similarity.shape)

# Find the movie that is closes to the given movie
# Search for the cloest to the value sent in, returns the index number
# calculate the difference array
x = 1.0
difference_array = np.absolute(cos_similarity-x)

# To remove those that are 100% the same cause then it's the same movie
index = difference_array.argmin()
difference_array = np.delete(difference_array, index)

# Find the index of minimum element from the array
index = difference_array.argmin()
print(index)

# Take the cos similarity and calculate the angle
angle_in_radius = math.acos(cos_similarity[search_movie][index])

print("Movie title: " + movies_db['movie name\r\n'][search_movie])
print("\nMovie plot: " + movies_db['DETAIL ABOUT MOVIE\r\n'][search_movie])

print("\nFound movie title: " + movies_db['movie name\r\n'][index])
print("\nFound movie plot: " + movies_db['DETAIL ABOUT MOVIE\r\n'][index])
print("\nFound movie degree: ")
print(math.degrees(angle_in_radians))





