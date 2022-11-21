# MASKININLÄRNING FÖR SOCIALA MEDIER TNM108
# PROJECT: Use NLP algorithm to decide what movie plots are most similar to eachother
# AUTHORS: Group 14, Malva Jansson and Maria Brunned
#____________________________________________________________________________________

import numpy as np
import pandas as pd
import re 
import nltk

# Import database
movies_db = pd.read_csv('imdb (1000 movies) in june 2022.csv')

# Print database
#print(movies_db.columns)

# Print summary column 
#print(movies_db['DETAIL ABOUT MOVIE\r\n'])




