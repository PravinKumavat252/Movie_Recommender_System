# 🎬 Movie Recommender System

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/) 
[![Streamlit](https://img.shields.io/badge/Streamlit-App-orange)](https://streamlit.io/) 
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A **smart and interactive movie recommendation app** built with **Streamlit**. Explore top movies by genre, search for your favorite films, and get personalized movie recommendations based on content similarity. Movie details and posters are fetched in real-time using the **TMDB API**.

---

## 🚀 Key Features

- **Top 5 Trending Movies:** Quickly see what’s popular.  
- **Browse by Genre:** Adventure, Romance, Horror, Action, Comedy, and more.  
- **Smart Movie Search:** Type any movie name to get **5 personalized recommendations**.  
- **Movie Details & Posters:** Fetches real-time posters, ratings, release year, and genres using **TMDB API**.  
- **Interactive UI:** Streamlit-based interface that’s **easy to navigate**.  
- **Personalized Recommendations:** Uses **TF-IDF vectorization** and **cosine similarity** to suggest movies similar to your search.  
- **Safe API Key Handling:** TMDB API key is stored securely in `.streamlit/secrets.toml`.  
- **Responsive Layout:** Works on desktop and mobile screens.  
- **Fast & Efficient:** Optimized caching with `@st.cache_data` for quicker responses.  

--- 

## 🔄 How It Works

**Workflow Diagram:** 

**User Search → Dataset Lookup → TF-IDF Vectorization → Cosine Similarity → Top 5 Recommendations → TMDB API → Display Results**

Or as a step-by-step list:

1. **User Search** – The user enters a movie name in the search bar.  
2. **Dataset Lookup** – The app finds the movie in the Kaggle dataset.  
3. **TF-IDF Vectorization** – Converts movie genres/descriptions into vectors.  
4. **Cosine Similarity** – Calculates similarity between movies.  
5. **Top 5 Recommendations** – Selects the 5 most similar movies.  
6. **TMDB API** – Fetches movie posters, release year, and ratings.  
7. **Display Results** – Shows recommendations with posters and details in the app.
