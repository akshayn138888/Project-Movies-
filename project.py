import numpy as np
import pandas as pd
import nltk 
import matplotlib.pyplot as plt

genres = pd.read_json('genres.json.gz', orient='record', lines=True)
wiki_movies = pd.read_json('wikidata-movies.json.gz', orient='record', lines=True)
rt = pd.read_json('rotten-tomatoes.json.gz', orient='record', lines=True)
omdb= pd.read_json('omdb-data.json.gz', orient='record', lines=True)


# Country of Origin Stats. 

import matplotlib.pyplot as plt
#plt.hist(wiki1['original_language'].values)
#plt.show()
summ =  wiki1['enwiki_title'].sum()

def calc(row):
    return row['enwiki_title']/summ * 100

wiki1['percent']=wiki1.apply(func=calc,axis=1)
wiki_bylanguage = wiki1.sort_values(by='percent',ascending=False).reset_index()
# wiki_bylanguage[:10]



