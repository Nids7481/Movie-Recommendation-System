import streamlit as st
import numpy as np
import pandas as pd
import difflib 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your movie dataset (replace 'movies.csv' with your actual file path)
movies_data = pd.read_csv('movies.csv', usecols=range(24))

# Feature selection and preprocessing
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# Vectorize the features
vectorizer = TfidfVectorizer() 
feature_vectors = vectorizer.fit_transform(combined_features) 

# Calculate cosine similarity
similarity = cosine_similarity(feature_vectors)

# Streamlit app
st.title("Movie Recommender")

movie_name = st.text_input('Enter your favourite movie name:')

if movie_name:
    list_of_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, [str(title) for title in list_of_titles])
    
    if find_close_match:
        close_match = find_close_match[0]
        index_of_movie = movies_data[movies_data.title == close_match].index[0]
        similarity_score = list(enumerate(similarity[index_of_movie]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        st.subheader('Movies suggested for you:')
        i = 1
        for movie in sorted_similar_movies[:30]:
            index = movie[0]
            title_from_index = movies_data.iloc[index]['title']
            st.write(f"{i}. {title_from_index}")
            i += 1
    else:
        st.write("No close matches found for your movie.")
