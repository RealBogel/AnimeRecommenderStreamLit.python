"""
This Streamlit app provides an anime recommendation system using real-time data
from MyAnimeList (via the Jikan API). It leverages cached anime metadata to
compute content-based similarity using TF-IDF and cosine similarity on combined
synopsis and genres text. Users input an anime title, and the app suggests similar
anime based on textual similarity, enhanced by fuzzy matching for flexible input.
"""
import streamlit as st
from local_cachingJSON import get_top_animes, get_cache_last_updated  # import caching function
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

#-------------------------------------
# similarity matrix
#-------------------------------------
# Constructs a similarity matrix for the anime dataset by combining the synopsis
# and genres of each anime into a single string. These combined texts are converted
# into TF-IDF vectors to weight terms by importance, then cosine similarity is
# calculated to quantify similarity between every pair of anime.
#-------------------------------------
@st.cache_data(show_spinner="Building similarity matrix...")
def build_similarity_matrix(df):
    combined_features = df['synopsis'] + " " + df['genres']
    tfidf = TfidfVectorizer(stop_words = 'english') 
    tfidf_matrix = tfidf.fit_transform(combined_features)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

#-------------------------------------
# recommend
#-------------------------------------
# Given a user-selected anime title, this function finds the anime's index and
# retrieves its similarity scores to all other anime. It sorts these scores in
# descending order, excludes the anime itself, and returns the top N most similar
# anime as recommendations.
#-------------------------------------
def recommend(title, df, similarity_matrix, top_n=5):
    match_row = df[
        (df['title'] == title) | 
        (df['title_english'] == title) | 
        (df['title_japanese'] == title)
    ]

    if match_row.empty:
        raise ValueError(f"No anime found matching title '{title}'.")
    
    idx = match_row.index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]  # exclude itself
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
