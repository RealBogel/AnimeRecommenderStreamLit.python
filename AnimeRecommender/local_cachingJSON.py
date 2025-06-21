"""
This module handles fetching, caching, and loading of top anime data from the Jikan API.
It implements a caching mechanism to reduce redundant API requests by storing data locally
in a JSON file and refreshing it only if the cache is older than 24 hours. The main function
returns a pandas DataFrame containing detailed anime information such as titles, synopsis,
genres, and images for use in further analysis or display.
"""
import os
import json
import pandas as pd
import requests
import time

CACHE_FILE = 'anime_cache.json'
CACHE_EXPIRATION_HOURS = 24

#-------------------------------------
# last cache update
#-------------------------------------
# This function returns the last modification time of the cache file as a formatted string,
# or None if the cache file does not exist. Useful for checking cache freshness.
#-------------------------------------
def get_cache_last_updated():
    if os.path.exists(CACHE_FILE):
        last_modified = os.path.getmtime(CACHE_FILE)
        # Convert to readable string
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_modified))
    return None

#-------------------------------------
# cache validation
#-------------------------------------
# This function checks if the cache file exists and whether it is still valid based on
# a preset expiration time (24 hours). Returns True if cache is valid, False otherwise.
#-------------------------------------
def is_cache_valid():
    if not os.path.exists(CACHE_FILE):
        return False
    last_modified = os.path.getmtime(CACHE_FILE)
    age_hours = (time.time() - last_modified) / 3600
    return age_hours <= CACHE_EXPIRATION_HOURS

#-------------------------------------
# load cached anime
#-------------------------------------
# Loads the cached anime data from a JSON file if it exists and is fresh enough.
# If the cache is stale or missing, it returns None to signal the need for data fetching.
#-------------------------------------
def load_cached_anime():
    if os.path.exists(CACHE_FILE):
        last_modified = os.path.getmtime(CACHE_FILE)
        age_hours = (time.time() - last_modified) / 3600
        if age_hours > CACHE_EXPIRATION_HOURS:
            print("Cache is stale. Refreshing...")
            return None
        else:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                print("Loading data from cache...")
                return json.load(f)
    return None

#-------------------------------------
# save cached anime
#-------------------------------------
# Saves the given anime data (a list of dictionaries) to the JSON cache file,
# writing it with indentation for readability and printing a confirmation message.
#-------------------------------------
def save_cached_anime(data):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Saved data to cache.")

#-------------------------------------
# get top animes
#-------------------------------------
# Main function that retrieves top anime data either from the local cache or
# by querying the Jikan API. It supports fetching multiple pages of data and
# returns a pandas DataFrame with relevant fields for each anime.
#-------------------------------------
def get_top_animes(pages=5):
    if is_cache_valid():
        cached_data = load_cached_anime()
        return pd.DataFrame(cached_data)
    
    print("Fetching data from Jikan API...")
    anime_list = []

    for page in range(1, pages + 1):
        url = f"https://api.jikan.moe/v4/top/anime?page={page}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()['data']
            for anime in data:
                anime_list.append({
                    'id': anime['mal_id'],
                    'title': anime['title'],
                    'title_english': anime.get('title_english', ""),
                    'title_japanese': anime.get('title_japanese', ""),
                    'title_synonyms': ", ".join(anime.get('title_synonyms', [])),
                    'synopsis': anime['synopsis'] or "",
                    'genres': ", ".join([g['name'] for g in anime['genres']]),
                    'image_url': anime['images']['jpg']['image_url']
                })
            time.sleep(0.5)  # To avoid hitting API rate limits
        else:
            print(f"Warning: Failed to fetch page {page} (status code {response.status_code})")
    save_cached_anime(anime_list)
    return pd.DataFrame(anime_list)