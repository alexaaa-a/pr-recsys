from utils import load_data, get_top_n_movies

movies, ratings = load_data()

def recommend_popular(top_n=10):
    top = get_top_n_movies(ratings, n=top_n)
    top_titles = movies[movies['movieId'].isin(top.index)]['title'].tolist()
    return top_titles
