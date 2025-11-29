import pandas as pd
import numpy as np
from utils import load_data
from scipy.sparse.linalg import svds


movies, ratings = load_data()

rating_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
R = rating_matrix.values
user_ids = rating_matrix.index.tolist()
movie_ids = rating_matrix.columns.tolist()

R_demeaned = R - R.mean(axis=1).reshape(-1, 1)

U, sigma, Vt = svds(R_demeaned, k=20)
sigma = np.diag(sigma)

predicted_ratings = np.dot(np.dot(U, sigma), Vt) + R.mean(axis=1).reshape(-1, 1)
preds_df = pd.DataFrame(predicted_ratings, index=user_ids, columns=movie_ids)

def recommend_svd(user_id, top_n=5):
    if user_id not in preds_df.index:
        return []

    user_row = preds_df.loc[user_id]

    watched = rating_matrix.loc[user_id]
    user_row = user_row[watched == 0]

    top_movies = user_row.sort_values(ascending=False).head(top_n).index.tolist()
    return top_movies
