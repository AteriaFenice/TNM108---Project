# MASKININLÄRNING FÖR SOCIALA MEDIER TNM108
# PROJECT: Use NLP algorithm to decide what movie plots are most similar to eachother
# AUTHORS: Group 14, Malva Jansson and Maria Brunned
#____________________________________________________________________________________

import numpy as np
import pandas as pd
import re 
import nltk
import math
import random

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
#nltk.download('stopwords')

# Cosine simluarity
from sklearn.metrics.pairwise import cosine_similarity

# Dendogram
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# Import database
movies_db = pd.read_csv('imdb (1000 movies) in june 2022.csv')

# Print database
#print(movies_db.columns)

# Print summary column 
#print(movies_db['DETAIL ABOUT MOVIE\n'])

# Create TFI-DF matrix
tfidf_vectorizer = TfidfVectorizer(stop_words=set(stopwords.words('english')))
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_db['DETAIL ABOUT MOVIE\n'])

#print(tfidf_matrix.shape) # Consits of 1000 rows (movies) and 5715 columns (tf-idf terms)

# Calculate simularity 
search_movie = random.randint(0,999) # Take a random movie from the database
print("Index of movie: ", search_movie)
cos_similarity = cosine_similarity(tfidf_matrix[search_movie], tfidf_matrix)
#print(cos_similarity)
#print(cos_similarity.shape)

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

# Take the cos similarity and calculate the angle
angle_in_radius = math.acos(cos_similarity[0][index])

print("Index of found movie:", index)
print("Max similarity:", 1.0-difference_array.min()) # print max similarity
print("Found movie degree:", math.degrees(angle_in_radius), "\n")

print("Movie title: " + movies_db['movie name\r\n'][search_movie])
print("\nMovie plot: " + movies_db['DETAIL ABOUT MOVIE\n'][search_movie])

print("\nFound movie title: " + movies_db['movie name\r\n'][index])
print("\nFound movie plot: " + movies_db['DETAIL ABOUT MOVIE\n'][index])

# Show 5 most similar movies
print("_________________________________________")
print("\nTop 5 most similar movies: \n")
k = 5
index_list = np.argpartition(difference_array, k)
five = index_list[:k]
#print(five)
for x in five:
    print("Movie title: " + movies_db['movie name\r\n'][x]," | Index of movie:", x)
    print("Movie plot: " + movies_db['DETAIL ABOUT MOVIE\n'][x])
    print("Similarity:", 1.0-difference_array[x])
    angle_in_radius = math.acos(cos_similarity[0][x])
    print("Degree:", math.degrees(angle_in_radius), "\n")




# DENDOGRAM - works but messy!
'''similarity_distance = 1 - cosine_similarity(tfidf_matrix)
mergings = linkage(similarity_distance, method='complete')
dendrogram_ = dendrogram(mergings, labels=[x for x in movies_db["movie name\r\n"]], leaf_rotation=90, leaf_font_size=16,)
fig = plt.gcf()
_ = [lbl.set_color('r') for lbl in plt.gca().get_xmajorticklabels()]
fig.set_size_inches(108, 21)
plt.show()'''
