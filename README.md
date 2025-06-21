# Anime Recommender
This project is a content-based anime recommendation system that uses real-time data from MyAnimeList (via the Jikan API) combined with natural language processing (NLP) techniques to suggest similar anime based on synopsis and genres.

The system is fully interactive using Streamlit, allowing users to type any anime name (with fuzzy matching support) and receive high-quality recommendations based on semantic similarity of anime descriptions.

## Features
- Real-time anime data from MyAnimeList using the Jikan API

- Caching system to avoid redundant API calls and speed up processing

- Semantic similarity using Sentence Transformers (pre-trained language model)

- Cosine similarity used for recommendation scoring

- Fuzzy matching to handle approximate or partial user input

- Interactive Streamlit interface with anime posters, genres, and synopses

## Machine Learning Techniques Used
### Natural Language Processing (NLP):
- Anime descriptions and genres are combined into text representations.

### Sentence Embeddings:
- The sentence-transformers library (all-MiniLM-L6-v2 model) generates dense vector embeddings that capture semantic meaning of the anime content.

### Cosine Similarity:
- Similarity between embeddings is calculated using cosine similarity, allowing the system to recommend anime with the most similar content.

### Fuzzy String Matching:
- The fuzzywuzzy library helps match user input with available anime titles, handling typos, synonyms, and different title formats.

