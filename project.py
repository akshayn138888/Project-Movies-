import numpy as np
import pandas as pd
import nltk 
import matplotlib.pyplot as plt

genres = pd.read_json('genres.json.gz', orient='record', lines=True)
wiki_movies = pd.read_json('wikidata-movies.json.gz', orient='record', lines=True)
rt = pd.read_json('rotten-tomatoes.json.gz', orient='record', lines=True)
omdb= pd.read_json('omdb-data.json.gz', orient='record', lines=True)


# Country of Origin Stats. 

#plt.hist(wiki1['original_language'].values)
#plt.show()
summ =  wiki1['enwiki_title'].sum()

def calc(row):
    return row['enwiki_title']/summ * 100

wiki1['percent']=wiki1.apply(func=calc,axis=1)
wiki_bylanguage = wiki1.sort_values(by='percent',ascending=False).reset_index()
# wiki_bylanguage[:10]


#ploting our data entries / What we have and what we dont have. 

rt1 = rt[['imdb_id', 'critic_percent', 'audience_percent', 'audience_ratings','critic_average']]
#rt.count().plot.bar()

#Plotting how many movies made profit and how many did not. 

wiki_movies2 = wiki_movies[['imdb_id', 'made_profit']].dropna()

profit_movie = pd.merge(rt1, wiki_movies2, on='imdb_id', how='outer')
profit_movie = profit_movie[['imdb_id', 'critic_percent', 'audience_percent','made_profit']]
#profit_movie = profit_movie.dropna()
#profit_movie.count().plot.bar() #made profit or not. 

################################

profit_movie.groupby('made_profit').mean().plot.bar() #profitable movies have higher critic and audience ratings. 
profit_movie.groupby('made_profit').mean()


#profit_movie = pd.merge(rt1, wiki_movies2, on='imdb_id', how='outer')
#profit_movie

#profit_movie['audience_percent'].plot.bar()
profit_movie['audience_percent'].hist()

profit_movie['critic_percent'].hist()

################################

director_data = wiki_movies[['director','imdb_id']]
movie_ratings = rt[['imdb_id', 'critic_percent', 'audience_percent' ]]
director_ratings = pd.merge( movie_ratings, director_data, on='imdb_id', how ='outer')
director_ratings = director_ratings[['imdb_id', 'critic_percent', 'audience_percent' , 'director']]

director_ratings.count().plot.bar()

director_audiencR = director_ratings[['audience_percent','director']]

def astype(row):
    return str(row['director'])

director_audiencR['director']=director_audiencR.apply(func=astype,axis=1)
director_audiencR = director_audiencR.groupby('director').mean().reset_index()
director_audiencR= director_audiencR.dropna()


director_audiencR = director_ratings[['critic_percent','director']]
def astype(row):
    return str(row['director'])

director_audiencR['director']=director_audiencR.apply(func=astype,axis=1)
director_audiencR = director_audiencR.groupby('director').mean().reset_index()
director_audiencR= director_audiencR.dropna()



director_audiencR['critic_percent'].hist( bins = 20, color = 'orange')

################################

