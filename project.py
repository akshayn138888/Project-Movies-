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

################################ Ratings based on Directors#############
director_audiencR = director_ratings[['critic_percent','director']]
def astype(row):
    return str(row['director'])

director_audiencR['director']=director_audiencR.apply(func=astype,axis=1)
director_audiencR = director_audiencR.groupby('director').mean().reset_index()
director_audiencR= director_audiencR.dropna()
director_audiencR['critic_percent'].hist( bins = 20, color = 'orange')

######################## DateTime NS ##### Exploration Data Using MOnths and Years ######
import time
from datetime import date

wiki_movies['publication_date'] = pd.to_datetime(wiki_movies['publication_date'])

movie_season = wiki_movies[['imdb_id', 'publication_date']]
def getmonth(month):
    return month['publication_date'].month
movie_season['month']= movie_season.apply(func=getmonth,axis=1)

def getyear(year):
    return year['publication_date'].year

movie_season['year']= movie_season.apply(func=getyear,axis=1)

ratings = rt[['imdb_id' , 'critic_percent', 'audience_percent' ]]

movie_year = pd.merge( ratings , movie_season, on='imdb_id', how ='outer')

audience_percent = movie_year[['audience_percent', 'year']] 
critic_percent = movie_year[['critic_percent', 'year']] 

#audience_percent.groupby('year').mean().plot.bar()
#critic_percent.groupby('year').mean().plot.bar()

################Date Time ns ##### Exploration Using MOnths and critic Ratings and audience Ratings
# If critics and audience rates movies differently. 

ratings = rt1[['imdb_id' , 'critic_percent', 'audience_percent' ]]
movie_season = pd.merge( ratings , movie_season, on='imdb_id', how ='outer')
audience_percent = movie_season[['audience_percent', 'month']] 
critic_percent = movie_season[['critic_percent', 'month']] 


#audience_percent.groupby('month').mean().plot.bar()
#critic_percent.groupby('month').mean().plot.bar() 

################# Regular Experession Extracting Number of Awards, 'imdb_id','num_nominations','num_awards', 'num_oscars', 'num_golden_globe', 'num_bafta' ######

import re 

def process_awards1(row) : 
    awards_text=row['omdb_awards'].lower()
    NOMINATION_REGEX=r'[0-9]* nominations'
    if re.search(NOMINATION_REGEX,awards_text) is not None:       
        nomination_num=int(re.search(NOMINATION_REGEX,awards_text).group()[0])
    else:
        nomination_num=0
    return nomination_num

def process_awards2(row) : 
    awards_text=row['omdb_awards'].lower()
    WIN_REGEX =r'[0-9]* wins'
    if re.search(WIN_REGEX,awards_text) is not None:
        win_num = int(re.search(WIN_REGEX,awards_text).group()[0])
    else:
        win_num=0
        
    return win_num

def process_awards3(row) : 
    awards_text=row['omdb_awards'].lower()
    OSCAR_REGEX=r'[0-9]* oscars'
    if re.search(OSCAR_REGEX,awards_text) is not None:      
        oscar_num=int(re.search(OSCAR_REGEX,awards_text).group()[0])
    else:
        oscar_num=0
    return oscar_num
    
def process_awards4(row) : 
    awards_text=row['omdb_awards'].lower()
    GOLDENGATE_REGEX=r'[0-9]* golden globe'
    if re.search(GOLDENGATE_REGEX,awards_text) is not None: 
        goldengate_num=int(re.search(GOLDENGATE_REGEX,awards_text).group()[0])
    else:
        goldengate_num=0
    return goldengate_num


def process_awards5(row) : 
    awards_text=row['omdb_awards'].lower()     
    BAFTA_REGEX=r'[0-9]* bafta'
            
    if re.search(BAFTA_REGEX,awards_text) is not None:
        bafta_num = int(re.search(BAFTA_REGEX,awards_text).group()[0])
    else:
        bafta_num=0
    return bafta_num
       
omdb['num_nominations']=omdb.apply(func=process_awards1,axis=1)
omdb['num_awards']=omdb.apply(func=process_awards2,axis=1) 
omdb['num_oscars']=omdb.apply(func=process_awards3,axis=1) 
omdb['num_golden_globe']=omdb.apply(func=process_awards4,axis=1) 
omdb['num_bafta']=omdb.apply(func=process_awards5,axis=1) 

omdb_awards = omdb[['imdb_id','num_nominations','num_awards', 'num_oscars', 'num_golden_globe', 'num_bafta']]


################## Count Vectorizer ########################

#np.hstack(wiki_movies['genre']).tolist()
from sklearn.feature_extraction.text import CountVectorizer
vec=CountVectorizer()
wiki_movies['genre2']=wiki_movies.apply(func=lambda row:" ".join(row['genre']),axis=1)
df1=pd.DataFrame(vec.fit_transform(wiki_movies['genre2']).toarray(),columns=vec.get_feature_names()).reset_index()


best_vars = pd.merge(wiki_movies, df1 , on= 'index', how='outer')

best_vars2 = pd.merge(best_variables, best_vars, on= 'imdb_id', how='outer')

#best_vars_final = best_vars2.drop(col = ['index', 'based_on', 'cast_member', 'country_of_origin', 'eniki_title','filming_location', 'genre', 'label', 'made_profit_y', 'main_subject', 'metacritic_id', 'original_language', 'rotten_tomatoes_id', 'series','wikidata_id',
 #'genre2'])
best_vars_final = best_vars2[['imdb_id',
 'critic_percent','audience_percent','audience_ratings','audience_average','critic_average','made_profit_x',
 'num_nominations','num_awards','num_oscars',
 'num_golden_globe',
 'num_bafta',
 'director',
 'made_profit_y',
 'publication_date',
 'q1033891',
 'q1054574',
 'q10654943',
 'q1067324',
 'q1080374',
 'q1091334',
 'q1096120',
 'q1097273',
 'q1107',
 'q1113999',
 'q1115187',
 'q1117103',
 'q11356864',
 'q1135802',
 'q11366',
 'q11396323',
 'q11399',
 'q11425',
 'q11452132',
 'q1146335',
 'q1147197',
 'q1150666',
 'q1164334',
 'q11661562',
 'q1190502',
 'q1194365',
 'q1196408',
 'q1200678',
 'q1257444',
 'q1259759',
 'q1261214',
 'q12767035',
 'q128758',
 'q12912091',
 'q130232',
 'q130352',
 'q131539',
 'q1320115',
 'q13209138',
 'q132311',
 'q1332055',
 'q13377795',
 'q1339864',
 'q1341051',
 'q1342372',
 'q1344',
 'q1356411',
 'q1361932',
 'q136472',
 'q1366112',
 'q13717554',
 'q1377546',
 'q1395566',
 'q1401416',
 'q1433443',
 'q1436734',
 'q145806',
 'q1464369',
 'q14699093',
 'q15062348',
 'q1519335',
 'q15286013',
 'q1535153',
 'q1538137',
 'q15428604',
 'q1548170',
 'q15637293',
 'q15637299',
 'q15637301',
 'q15637310',
 'q15712918',
 'q15712927',
 'q157394',
 'q157443',
 'q15858553',
 'q15898171',
 'q16049832',
 'q1615638',
 'q1654577',
 'q16575965',
 'q16861950',
 'q16909344',
 'q16950433',
 'q169672',
 'q17013749',
 'q170238',
 'q170539',
 'q17113138',
 'q17175676',
 'q172067',
 'q1723850',
 'q172980',
 'q1740789',
 'q174526',
 'q1747837',
 'q1760864',
 'q1762165',
 'q1776156',
 'q1782964',
 'q1786567',
 'q17884',
 'q1788980',
 'q1800833',
 'q181001',
 'q182015',
 'q182154',
 'q182415',
 'q185529',
 'q185867',
 'q18620604',
 'q186424',
 'q1864294',
 'q188473',
 'q188784',
 'q1894374',
 'q191489',
 'q1919632',
 'q192239',
 'q192881',
 'q193541',
 'q1935609',
 'q193606',
 'q19367312',
 'q193979',
 'q1941707',
 'q1957385',
 'q197949',
 'q19842222',
 'q199701',
 'q1999690',
 'q200092',
 'q20220309',
 'q20267837',
 'q202866',
 'q20442589',
 'q20443008',
 'q20650540',
 'q20652466',
 'q20656232',
 'q20656352',
 'q20664331',
 'q20667180',
 'q20737414',
 'q208505',
 'q208555',
 'q21010853',
 'q2116008',
 'q2118696',
 'q21188110',
 'q21192427',
 'q21209409',
 'q212781',
 'q21322403',
 'q2137852',
 'q21401869',
 'q2143665',
 'q21590660',
 'q217117',
 'q217199',
 'q21802675',
 'q218248',
 'q219557',
 'q222639',
 'q222926',
 'q223685',
 'q223770',
 'q223945',
 'q224700',
 'q224989',
 'q2254193',
 'q2254211',
 'q2254548',
 'q226730',
 'q2290276',
 'q2292320',
 'q229390',
 'q2297927',
 'q22981906',
 'q231302',
 'q2321734',
 'q2356541',
 'q23653',
 'q237338',
 'q23739',
 'q23745',
 'q2376899',
 'q2389651',
 'q23916',
 'q240911',
 'q2421031',
 'q242492',
 'q2439025',
 'q2447078',
 'q2484376',
 'q248583',
 'q24862',
 'q2490520',
 'q24925',
 'q25110269',
 'q25372',
 'q253732',
 'q25379',
 'q2561390',
 'q2561438',
 'q2584671',
 'q2593937',
 'q261636',
 'q2625243',
 'q26268098',
 'q263734',
 'q2642760',
 'q2678111',
 'q270948',
 'q2724311',
 'q2743',
 'q275934',
 'q28026639',
 'q289',
 'q28968258',
 'q28968511',
 'q29197',
 'q2973181',
 'q2973201',
 'q2975633',
 'q2991560',
 'q2991565',
 'q3038946',
 'q304538',
 'q3056541',
 'q3072024',
 'q3072031',
 'q3072039',
 'q3072042',
 'q3072043',
 'q3072049',
 'q31235',
 'q319221',
 'q319226',
 'q320568',
 'q3249257',
 'q326439',
 'q3272147',
 'q332102',
 'q336059',
 'q336107',
 'q336144',
 'q343782',
 'q352904',
 'q36279',
 'q3634883',
 'q3641550',
 'q369747',
 'q37073',
 'q37484',
 'q38072107',
 'q38926',
 'q39427',
 'q39892385',
 'q3990883',
 'q4075563',
 'q40831',
 'q4164344',
 'q41664487',
 'q416747',
 'q4184',
 'q4220915',
 'q4235011',
 'q4292083',
 'q430525',
 'q4382232',
 'q4461646',
 'q457832',
 'q459290',
 'q459435',
 'q4674071',
 'q468478',
 'q4686573',
 'q47009776',
 'q471839',
 'q472637',
 'q4765080',
 'q4774498',
 'q482',
 'q483352',
 'q4836991',
 'q484641',
 'q486263',
 'q4875794',
 'q48834789',
 'q49084',
 'q491158',
 'q4925568',
 'q49451',
 'q4949058',
 'q496523',
 'q4984974',
 'q5035283',
 'q505119',
 'q5145881',
 'q5151495',
 'q5151497',
 'q517386',
 'q52162262',
 'q525350',
 'q5258881',
 'q53094',
 'q531067',
 'q5366020',
 'q5366097',
 'q542475',
 'q5434357',
 'q5442753',
 'q546440',
 'q5769084',
 'q5769663',
 'q5774663',
 'q5778924',
 'q580850',
 'q581714',
 'q583768',
 'q586250',
 'q5897543',
 'q590103',
 'q59126',
 'q5967378',
 'q599558',
 'q603291',
 'q604725',
 'q608862',
 'q622291',
 'q622370',
 'q622548',
 'q622812',
 'q623787',
 'q624771',
 'q628165',
 'q629917',
 'q643684',
 'q643873',
 'q645717',
 'q6457531',
 'q645928',
 'q652256',
 'q6585139',
 'q663106',
 'q665478',
 'q678345',
 'q681737',
 'q690342',
 'q699',
 'q7168625',
 'q7210294',
 'q7225114',
 'q7311396',
 'q7362831',
 'q7379160',
 'q743934',
 'q7444356',
 'q752321',
 'q754803',
 'q7551315',
 'q7569',
 'q7603925',
 'q761469',
 'q7643432',
 'q7644030',
 'q7645884',
 'q7696995',
 'q775169',
 'q775344',
 'q790192',
 'q794912',
 'q80930',
 'q8253',
 'q8261',
 'q8274',
 'q83267',
 'q838368',
 'q842256',
 'q846544',
 'q850412',
 'q851213',
 'q853630',
 'q853873',
 'q858330',
 'q859369',
 'q860626',
 'q8812380',
 'q883179',
 'q904447',
 'q909586',
 'q9168',
 'q9259727',
 'q926324',
 'q93196',
 'q93204',
 'q9335577','q9503','q959790','q986699' ]].dropna().reset_index(drop=True)


best_vars = best_vars_final.drop(columns=['director','made_profit_x', 'publication_date'])


from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt 
res = mutual_info_classif(best_vars.drop(columns=['imdb_id','made_profit_y']).values, best_vars['made_profit_y'].values, discrete_features=True)
dep=dict(list(zip(best_vars.drop(columns=['imdb_id','made_profit_y']).columns,res)))
d=sorted(dep,key=lambda x:x[1],reverse=True)
var_names=['audience_percent','audience_ratings','audience_average','num_nominations','num_awards','num_oscars','num_golden_globe','num_bafta','critic_percent','critic_average']

import seaborn as sns
sns.set()
import seaborn as sns
sns.set()

# plt.figure(figsize=(20, 12))
# plt.grid(True, color= 'skyblue')
# plt.plot(list(dep.keys())[:10],list(dep.values())[:10], color = 'red', linewidth = 5.0, ) 

##################### Modelling ###############################################

################Gaussian NB ###################

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

X = best_vars.drop(columns=['imdb_id','made_profit_y']).values
y = best_vars['made_profit_y'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
from sklearn.naive_bayes import GaussianNB
bayes_model = GaussianNB()
bayes_model.fit(X_train, y_train)
y_prediction = bayes_model.predict(X_test) 
print('GaussianNB Score: ' + accuracy_score(y_test, y_prediction))

############## K-nearest neighbours Model#########

from sklearn.neighbors import KNeighborsClassifier 

 knn_model = make_pipeline(
 KNeighborsClassifier(n_neighbors=13)
  )
 knn_model.fit(X_train, y_train)
 y_prediction = knn_model.predict(X_test) 
 print('knn_model Score (13 clusters): ' + accuracy_score(y_test, y_prediction))

 ##############SVC Model ##################

from sklearn.svm import SVC    
    
svc_model = make_pipeline(
        
        SVC(kernel='linear',C=0.0001)
    )
    
#svc_model.fit(X_train, y_train)
#y_prediction = svc_model.predict(X_test) 
#print(accuracy_score(y_test, y_prediction))

############ Finding How Many variables and clusters give the Highest Ratings #############

X = best_vars.drop(columns=['imdb_id','made_profit_y'])
y = best_vars['made_profit_y'].values

feature_names=X.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
ARR=[]
max_config=[]
def hyper_parametre(n,kmax):
    max_score=0.0
    for i in range(1,n+1):
        f_list=feature_names[:i]
        #print("f_list=",f_list)
        x = X_train[f_list]
        x_test=X_test[f_list]
        ar=[]
        for j in range(1,kmax+1):
            knn_model = make_pipeline( KNeighborsClassifier(n_neighbors=j) )
            knn_model.fit(x.values, y_train)
            y_prediction = knn_model.predict(x_test) 
            score=accuracy_score(y_test, y_prediction)
            if score>max_score:
                max_score=score
                max_config.append([i,j,f_list])
            ar.append(accuracy_score(y_test, y_prediction))
        ARR.append(ar)
        
    return max_score,max_config
            
mscore,mconfig=hyper_parametre(10,30)        
print("MAX SCORE = {} @ variables = {} ({}) & num_clusters = {}".format(mscore,mconfig[-1][0],mconfig[-1][2],mconfig[-1][1]))

#sns.heatmap(ARR) ######### Heat Map of Variables. How different variables affect our Model accuracy score. 

############################ svc ##############################

# from sklearn.svm import SVC    
# X = best_vars.drop(columns=['imdb_id','made_profit_y'])
# y = best_vars['made_profit_y'].values

# feature_names=X.columns.tolist()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
# ARR=[]
# max_config=[]
# def hyper_parametre(n):
#     max_score=0.0
#     for i in range(1,n):
#         f_list=feature_names[:i]
#         #print("f_list=",f_list)
#         x = X_train[f_list]
#         x_test=X_test[f_list]
#         ar=[]
#         from sklearn.svm import SVC    
#         svc_model = make_pipeline(SVC(kernel='linear',C=0.0001))
#         svc_model.fit(X_train, y_train)
#         y_prediction = svc_model.predict(X_test) 
#         score = accuracy_score(y_test, y_prediction)
#         if score>max_score:
#             max_score=score
#             max_config.append([i,f_list])
#         ar.append(accuracy_score(y_test, y_prediction))
#         ARR.append(ar)
#     return max_score,max_config


#print (hyper_parametre(6))  
#import seaborn as sns
#sns.set()
#plt.figure(figsize=(20, 12))
#plt.grid(True, color= 'skyblue')
#plt.plot(ARR, color = 'red', linewidth = 5.0 ) #





################################ Predict made profit using directors ################################### Ids 
#np.hstack(wiki_movies['genre']).tolist()
df2 = wiki_movies
df2 = df2[['imdb_id', 'director']].dropna()
from sklearn.feature_extraction.text import CountVectorizer
vec=CountVectorizer()
df2['director2']=df2.apply(func=lambda row:" ".join(row['director']),axis=1)
df2=pd.DataFrame(vec.fit_transform(df2['director2']).toarray(),columns=vec.get_feature_names()).reset_index()

wiki_movies[['imdb_id','director']]

df2['level_0'] = df2['index'].values

df3 = wiki_movies[['imdb_id', 'director']].dropna().reset_index().reset_index()

directors = pd.merge(df3, df2, on= 'level_0', how='outer')

cast_dir = wiki_movies[['imdb_id', 'director', 'cast_member', 'made_profit']].reset_index(drop=True)
cast_dir = cast_dir.dropna()

m_dir_profit = pd.merge(directors, cast_dir , on= 'imdb_id', how='outer')


m_dir_profit = m_dir_profit.dropna()

X = m_dir_profit.drop(columns=['imdb_id','director_x', 'director_y', 'cast_member']).values
y = m_dir_profit['made_profit'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

from sklearn.neighbors import KNeighborsClassifier 
knn_model = make_pipeline(
    KNeighborsClassifier(n_neighbors=20)
    )
knn_model.fit(X_train, y_train)
y_prediction = knn_model.predict(X_test) 
print( 'Director is a good indicator os profit (KNN Model result)'+ accuracy_score(y_test, y_prediction))

# It tells that directors are the good indicator of profit. Using data we have. 
















