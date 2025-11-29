import pandas as pd

def load_data():
    movies = pd.read_csv('data/movies.csv')
    ratings = pd.read_csv('data/ratings.csv')
    return movies, ratings

def get_top_n_movies(ratings, n=10):
    top = ratings.groupby('movieId').size().sort_values(ascending=False).head(n)
    return top
