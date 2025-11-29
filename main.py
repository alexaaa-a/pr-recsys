from content_based import recommend_content
from collaborative import user_based_cf, item_based_cf
from heuristic import recommend_popular
from svd import recommend_svd
from utils import load_data

movies, ratings = load_data()
user_id = 1
movie_id = 1

print("~~~ SVD Model (Fashion) ~~~")
svd_recs = recommend_svd(user_id=user_id, top_n=5)
print(f"Recommendations for user {user_id}:")
print(movies[movies['movieId'].isin(svd_recs)]['title'].tolist())

print("\n~~~ Content-Based ~~~")
print(recommend_content('Toy Story (1995)'))

print("\n~~~ Collaborative Filtering ~~~")
user_recs = user_based_cf(user_id=user_id, top_n=5)
print(f"User-based CF recommendations for user {user_id}:")
print(movies[movies['movieId'].isin(user_recs)]['title'].tolist())

item_recs = item_based_cf(movie_id=movie_id, top_n=5)
print(f"Item-based CF recommendations similar to movie {movie_id}:")
print(movies[movies['movieId'].isin(item_recs)]['title'].tolist())

print("\n~~~ Heuristic Popular ~~~")
print(recommend_popular(5))
