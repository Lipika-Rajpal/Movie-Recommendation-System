import numpy as np
import pandas as pd
import os 
import shutil
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings

# ignore all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Recommender_new_user:
    def __init__(self, movies, ratings, users):
        '''
        input: user_id
        '''
        self.movies = movies
        self.ratings = ratings
        self.users = users.copy()
        self.users = self.users.drop('zip-code',axis=1)
        self.users = self.users.drop('occupation',axis=1)
        self.nn_algo = NearestNeighbors(metric='cosine')
        self.users["gender"] = np.where(self.users["gender"] == "F", 0, 1)
        # self.users = self.users.drop('gender',axis=1)

    def fit(self):
        self.nn_algo.fit(self.users)
        return self.nn_algo

    def recommend_movies(self, user_id, n_recommend=10):
        # distance, neighbors = self.nn_algo.kneighbors([self.users.loc[self.users.userId == user_id]], n_neighbors=n_recommend+1)
        distance, neighbors = self.nn_algo.kneighbors(self.users.loc[self.users.userId == user_id], n_neighbors=n_recommend+1)

        userids = [self.users.iloc[i].name for i in neighbors[0]]
        recommends = [movieId for uid in userids for movieId in self.ratings.loc[(self.ratings.userId == uid) & (self.ratings.rating == 5)].dropna()['movieId']]
        recommends = np.array(recommends)
        # np.random.shuffle(recommends)
        rec_movies = [str(self.movies.loc[self.movies.movieId == movieId].title.values[0]) for movieId in recommends]
        return rec_movies[:n_recommend]
    


class MatrixFactorization(nn.Module):
    '''
    input: users_id
    '''
    def __init__(self, num_users, num_movies, n_factors=20):
        super().__init__()
        self.user_factors = nn.Embedding(num_users, n_factors)
        self.movie_factors = nn.Embedding(num_movies, n_factors)

    def forward(self, user, movie):
        return (self.user_factors(user) * self.movie_factors(movie)).sum(1)  
    
    def recommend(self, movies, ratings, users, user_id, n_recommend=10):
        rating_pivot = ratings.pivot_table(values='rating',columns='userId',index='movieId')
        pivot_tensor = torch.tensor(rating_pivot.values, dtype=torch.float32)
        mask = ~torch.isnan(pivot_tensor)

        # Get the indices of the non-NaN values
        i, j = torch.where(mask)

        # Get the values of the non-NaN values
        v = pivot_tensor[mask]

        # Store in PyTorch tensors
        users_nn= i.to(torch.int64)
        movies_nn = j.to(torch.int64)
        ratings_nn = v.to(torch.float32) 

        movie_user_df = pd.DataFrame({'user': users_nn, 'movie': movies_nn, 'rating': ratings_nn})
        

        n_users = rating_pivot.shape[0]
        n_movies = rating_pivot.shape[1]
        model = MatrixFactorization(n_users, n_movies, 5)
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        for i in range(300):
            # Compute the loss
            pred = model(users_nn, movies_nn)
            loss = F.mse_loss(pred, ratings_nn)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Backpropagate
            loss.backward()
            
            # Update the parameters
            optimizer.step()
            
            # Print the loss
            if i % 100 == 0:
                print(loss.item()) 
        with torch.no_grad():
            pred_raing_pivot = pd.DataFrame(model.user_factors.weight @ model.movie_factors.weight.t(), index=rating_pivot.index, columns=rating_pivot.columns)
            # round to nearest integer
            pred_raing_pivot = pred_raing_pivot.round() 
            pred_raing_pivot = np.clip(pred_raing_pivot, 1, 5)   
        movie_ids = pred_raing_pivot.loc[:, user_id][rating_pivot.loc[:, user_id].isna()].sort_values(ascending=False).index.values
        recommended_movies = movies[movies['movieId'].isin(movie_ids)]['title'].head(n_recommend).values
        return recommended_movies  


class Recommender_content_based:
    '''
    input: movie as string
    '''
    def __init__(self, movies, ratings, users):
        # This list will stored movies that called atleast ones using recommend_on_movie method
        self.hist = [] 
        self.ishist = False # Check if history is empty 
        self.movies = movies
        self.ratings = ratings
        self.users = users
        self.nn_algo = NearestNeighbors(metric='cosine')

        vectorizer = CountVectorizer(stop_words='english')
        genres = vectorizer.fit_transform(self.movies.genres).toarray()
        feature_names = vectorizer.get_feature_names_out() 
        self.contents = pd.DataFrame(genres,columns= feature_names)
    
    # This method will recommend movies based on a movie that passed as the parameter
    def recommend_on_movie(self,movie,n_reccomend = 10):
        self.ishist = True
        iloc = self.movies[self.movies['title']==movie].index[0]
        self.hist.append(iloc)
        distance,neighbors = self.nn_algo.kneighbors([self.contents.iloc[iloc]],n_neighbors=n_reccomend+1)
        recommeds = [self.movies.iloc[i]['title'] for i in neighbors[0] if i not in [iloc]]
        return recommeds[:n_reccomend]
    
    # This method will recommend movies based on history stored in self.hist list
    def recommend_on_history(self,n_reccomend = 10):
        if self.ishist == False:
            return print('No history found')
        history = np.array([list(self.contents.iloc[iloc]) for iloc in self.hist])
        distance,neighbors = self.nn_algo.kneighbors([np.average(history,axis=0)],n_neighbors=n_reccomend + len(self.hist))
        recommeds = [self.movies.iloc[i]['title'] for i in neighbors[0] if i not in self.hist]
        return recommeds[:n_reccomend] 
    
    def fit(self):
        self.nn_algo.fit(self.contents)
        return self.nn_algo
    


class Evaluator:
    def __init__(self, movies):
        self.movies = movies
        self.nn_algo = NearestNeighbors(metric='cosine')
        vectorizer = CountVectorizer(stop_words='english')
        genres = vectorizer.fit_transform(self.movies.genres).toarray()
        feature_names = vectorizer.get_feature_names_out() 
        self.contents = pd.DataFrame(genres,columns= feature_names)
    
    def recommend_on_movie(self,movie,n_reccomend = 10):
        iloc = self.movies[self.movies['title']==movie].index[0]
        distance,neighbors = self.nn_algo.kneighbors([self.contents.iloc[iloc]],n_neighbors=n_reccomend+1)
        recommeds = [self.movies.iloc[i]['title'] for i in neighbors[0] if i not in [iloc]]
        return recommeds[:n_reccomend]

    
    def fit(self):
        self.nn_algo.fit(self.contents)
        return self.nn_algo
    

def get_movies(user_id, movie_name, isNew):
    movies = pd.read_csv('.\our_dataset\movies.csv',sep=';',encoding='latin-1').drop('Unnamed: 3',axis=1)
    ratings = pd.read_csv('.\our_dataset\\ratings.csv',sep=';')
    users = pd.read_csv('.\our_dataset\\users.csv',sep=';') 

    if user_id in users['userId'].values:
        isNew = False

    # Running all the models
    if isNew:
        new_user = Recommender_new_user(movies, ratings, users) 
        new_user.fit() 
        set1 = new_user.recommend_movies(user_id)
        # print(set1) 
    
    else:
        mf = MatrixFactorization(users.shape[0], movies.shape[0], 5) 
        set2 = mf.recommend(movies, ratings, users, user_id, 10)
        # print(set2)


        content_based = Recommender_content_based(movies, ratings, users)
        content_based.fit()
        set3 = (content_based.recommend_on_movie(movie_name))
        # print(set3)


        movie_names = list(set3) + list(set2)
        movie_names.append(movie_name)
        data_movies = pd.DataFrame()
        movie_ids = []
        for movie_name in movie_names:
            u = movies[movies['title'] == movie_name]
            j = movies[movies['title'] == movie_name]['movieId']
            movie_ids.append(j.iloc[0])
            data_movies = data_movies.append(u, ignore_index = True)

        evaluator = Evaluator(data_movies)
        evaluator.fit()
        ans = evaluator.recommend_on_movie(movie_name)
        print(ans)
        return ans

# get_movies(1, 'Waiting to Exhale (1995)', True)



