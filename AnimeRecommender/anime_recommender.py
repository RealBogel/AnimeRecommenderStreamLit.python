"""
Anime Recommendation System (Machine Learning Version)

This Streamlit app provides an interactive anime recommendation system using real-time data
from MyAnimeList (via the Jikan API). It retrieves top anime data and caches it locally. 
For recommendations, it uses a pre-trained Sentence Transformer model (all-MiniLM-L6-v2) 
to generate semantic embeddings based on combined synopsis and genre text for each anime. 
Cosine similarity is computed on these embeddings to measure content-based similarity.

The app features fuzzy string matching to handle user input flexibly, allowing approximate
title matches. Once a title is selected, the app recommends the most semantically similar 
anime based on their descriptions and genres. The UI is built using Streamlit for easy 
interaction and visualization of recommendations.
"""
import streamlit as st
import numpy as np
from local_cachingJSON import get_top_animes, get_cache_last_updated  # import caching function
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

#-------------------------------------
# similarity matrix (ML version)
#-------------------------------------
# Uses pre-trained sentence transformer to encode combined synopsis and genres
# into semantic embeddings, and computes cosine similarity between all anime.
#-------------------------------------
@st.cache_data(show_spinner="Building semantic similarity matrix...")
def build_similarity_matrix(df):
    combined_features = df['synopsis'] + " " + df['genres']
    model = SentenceTransformer('all-MiniLM-L6-v2')  
    embeddings = model.encode(combined_features.tolist(), convert_to_numpy=True)
    cosine_sim = cosine_similarity(embeddings, embeddings)
    return cosine_sim

#-------------------------------------
# recommend
#-------------------------------------
# Given a selected anime title (already fuzzy-matched by the user interface),
# this function locates the corresponding anime in the dataset by checking
# if the title appears in any of the title-related fields (original title,
# English title, Japanese title, or synonyms). Once the anime is located,
# it retrieves the precomputed similarity scores from the similarity matrix,
# sorts them in descending order (highest similarity first), excludes the anime
# itself, and returns the top N most similar anime as recommendations.
#-------------------------------------
def recommend(title, df, similarity_matrix, top_n=5):
    idx = df.apply(
        lambda row: title in str(row['title']) 
                    or title in str(row['title_english']) 
                    or title in str(row['title_japanese']) 
                    or title in str(row['title_synonyms']), axis=1
    ).idxmax()

    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]
    recommendations = df.iloc[[i[0] for i in sim_scores]]
    return recommendations

#-------------------------------------
# Streamlit App
#-------------------------------------
# Sets up the user interface with a title and description. Loads anime data and
# builds the similarity matrix once. Displays the last cache update time.
# Allows the user to type an anime name with fuzzy matching to suggest close matches.
# When the user clicks the button, displays recommended anime with titles, images,
# genres, and synopsis.
#-------------------------------------
# Streamlit UI
st.title("🎌 Anime Recommendation System")
st.write("Using real-time data from MyAnimeList via Jikan API")

# Load data and build similarity matrix
df = get_top_animes(pages = 20)

last_updated = get_cache_last_updated()
if last_updated:
    st.write(f"📅 Data last updated: {last_updated}")
else:
    st.write("📅 Data last updated: Unknown")

if 'title_english' not in df.columns:
    df['title_english'] = ""
if 'title_japanese' not in df.columns:
    df['title_japanese'] = ""
if 'title_synonyms' not in df.columns:
    df['title_synonyms'] = ""

similarity_matrix = build_similarity_matrix(df)

# User input
user_input = st.text_input("Type an anime name:")

if user_input:
    # Get top 5 fuzzy matches for input
    choices = (
        df['title'].tolist() +
        df['title_english'].dropna().tolist() +
        df['title_japanese'].dropna().tolist() +
        df['title_synonyms'].dropna().apply(lambda x: x.split(", ")).explode().tolist()
    )
    choices = list(set([c for c in choices if c]))  # remove duplicates and empty strings
    matches = process.extract(user_input, choices, limit=5)
    options = [match[0] for match in matches]
    
    selected_title = st.selectbox("Did you mean:", options)
    
    if st.button("Get Recommendations"):
        recommendations = recommend(selected_title, df, similarity_matrix)
        st.write(f"Because you liked **{selected_title}**, you may also like:")
        for _, row in recommendations.iterrows():
            st.subheader(f"{row['title']} ({row['title_english']})")
            st.image(row['image_url'], width=200)
            st.write(f"**Genres:** {row['genres']}")
            st.write(f"**Synopsis:** {row['synopsis']}")
            st.write("---")
else:
    st.write("Start typing to see anime suggestions...")
