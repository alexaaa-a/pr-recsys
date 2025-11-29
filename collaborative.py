import pandas as pd
import numpy as np
from utils import load_data
from sklearn.model_selection import train_test_split

movies, ratings = load_data()

rating_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)


def user_based_cf(user_id, top_n=5):
    user_corr = rating_matrix.T.corr()
    if user_id not in user_corr.columns:
        return []

    similar_users = user_corr[user_id].sort_values(ascending=False)[1:]

    similar_users_idx = similar_users.index
    recs = rating_matrix.loc[similar_users_idx].mean(axis=0)

    watched = rating_matrix.loc[user_id]
    recs = recs[watched == 0]

    top_movies = recs.sort_values(ascending=False).head(top_n).index.tolist()
    return top_movies


def item_based_cf(movie_id, top_n=5):
    item_corr = rating_matrix.corr()
    if movie_id not in item_corr.columns:
        return []

    similar_movies = item_corr[movie_id].sort_values(ascending=False)[1:top_n + 1]
    return similar_movies.index.tolist()
