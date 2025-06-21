# Anime Recommender
This Python Streamlit app builds a content-based anime recommendation system using real-time data fetched from the MyAnimeList API via Jikan. It utilizes TF-IDF vectorization and cosine similarity to recommend anime based on synopsis and genres, enhanced by fuzzy matching for flexible user input.

Objectives:
- Practice building interactive web apps with Streamlit

- Implement content-based recommendation algorithms using natural language processing

- Use TF-IDF and cosine similarity to compute item similarity

- Apply fuzzy string matching to improve user experience

- Manage API data caching to optimize performance and reduce redundant requests

- Present data in a clean, user-friendly interface with images and descriptive text

Breakdown:
- Fetch and cache top anime data from Jikan API with local JSON storage

- Build a TF-IDF similarity matrix on combined synopsis and genre text

- Develop a recommendation function that retrieves the most similar anime given a title

- Implement fuzzy matching to handle flexible user inputs and suggest closest matches

- Create a Streamlit UI with input box, selection dropdown, and recommendation display

- Display recommended anime with title, English title, image, genres, and synopsis

- Provide cache freshness indicator for transparency on data recency

