from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_data

movies, ratings = load_data()

count = CountVectorizer(token_pattern='[a-zA-Z0-9]+')
count_matrix = count.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)

def recommend_content(movie_title, top_n=5):
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()
