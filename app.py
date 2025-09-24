import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

# ----------------- Streamlit Page Config -----------------
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Custom CSS -----------------
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #1F77B4;
        margin: 1rem 0;
    }
     .movie-card {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 0.5rem;
        text-align: center;
        height: 280px;
        overflow: hidden;
    }
    .search-box {
        margin: 2rem 0;
    }
    .rating-high {
        color: #00D100;
        font-weight: bold;
    }
    .rating-medium {
        color: #FFA500;
        font-weight: bold;
    }
    .rating-low {
        color: #FF4B4B;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Recommender Class -----------------
class MovieRecommender:
    def __init__(self):
        self.movies_df = None
        self.tmdb_api_key = st.secrets["TMDB_API_KEY"]
        self.load_data()

    def load_data(self):
        url = "https://github.com/PravinKumavat252/Movie_Recommender_System/raw/main/tmdb_5000_movies.csv"
        try:
            self.movies_df = pd.read_csv(url)
            self.preprocess_data()
        except FileNotFoundError:
            st.error("❌ Dataset not found. Please check the path.")

    def preprocess_data(self):
        # Fill missing values
        for col in ['overview', 'tagline', 'genres', 'keywords']:
            if col not in self.movies_df.columns:
                self.movies_df[col] = ""
            self.movies_df[col] = self.movies_df[col].fillna("")

        # Weighted Rating for popularity
        C = self.movies_df['vote_average'].mean()
        m = self.movies_df['vote_count'].quantile(0.7)

        def weighted_rating(x):
            v = x['vote_count']
            R = x['vote_average']
            return (v / (v + m) * R) + (m / (m + v) * C)

        self.movies_df['score'] = self.movies_df.apply(weighted_rating, axis=1)

        # Combined features
        self.movies_df['combined_features'] = (
            self.movies_df['overview'] + " " +
            self.movies_df['tagline'] + " " +
            self.movies_df['genres'] + " " +
            self.movies_df['keywords']
        )

        # TF-IDF with bigrams
        self.tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies_df['combined_features'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def get_movie_poster(self, movie_title):
        try:
            search_url = f"https://api.themoviedb.org/3/search/movie"
            params = {'api_key': self.tmdb_api_key, 'query': movie_title, 'language': 'en-US'}
            response = requests.get(search_url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data['results']:
                    poster_path = data['results'][0].get('poster_path')
                    if poster_path:
                        return f"https://image.tmdb.org/t/p/w500{poster_path}"
            return "https://via.placeholder.com/500x750/333333/FFFFFF?text=No+Poster"
        except:
            return "https://via.placeholder.com/500x750/333333/FFFFFF?text=No+Poster"

    def get_recommendations(self, movie_title, n_recommendations=5):
        try:
            idx = self.movies_df[self.movies_df['title'].str.lower() == movie_title.lower()].index
            if len(idx) == 0:
                idx = self.movies_df[self.movies_df['title'].str.lower().str.contains(movie_title.lower())].index
                if len(idx) == 0:
                    return pd.DataFrame()
            idx = idx[0]
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:n_recommendations+1]
            indices = [i[0] for i in sim_scores]
            return self.movies_df.iloc[indices][['title', 'vote_average', 'score']]
        except:
            return pd.DataFrame()

    def get_top_movies_by_genre(self, genre, n_movies=5):
        genre_movies = self.movies_df[self.movies_df['genres'].str.lower().str.contains(genre.lower(), na=False)]
        return genre_movies.nlargest(n_movies, 'score')


# ----------------- Streamlit App -----------------
def main():
    recommender = MovieRecommender()

    if recommender.movies_df is None:
        return

    st.markdown('<h1 class="main-header">🎬 Movie Recommender System</h1>', unsafe_allow_html=True)

    # Sidebar info
    st.sidebar.write("**Dataset loaded successfully!**")
    st.sidebar.write(f"**Total movies:** {len(recommender.movies_df)}")
    # st.sidebar.write(f"**Columns available:** {list(recommender.movies_df.columns)}")

    # Dataset statistics
    st.sidebar.markdown("### 📊 Dataset Statistics")
    if "vote_average" in recommender.movies_df.columns and "vote_count" in recommender.movies_df.columns:
        st.sidebar.write(f"**Average Rating:** {recommender.movies_df['vote_average'].mean():.2f}/10")
        st.sidebar.write(f"**Total Votes:** {recommender.movies_df['vote_count'].sum():,}")

    # Handle release date safely
    if "release_date" in recommender.movies_df.columns:
        valid_dates = pd.to_datetime(recommender.movies_df['release_date'], errors="coerce").dropna()
        if not valid_dates.empty:
            min_year = valid_dates.min().year
            max_year = valid_dates.max().year
            st.sidebar.write(f"**Date Range:** {min_year} - {max_year}")
        else:
            st.sidebar.write("**Date Range:** N/A")

    # Search box
    st.markdown("---")
    st.markdown('<h2 class="section-header">🔍 Search Movies</h2>', unsafe_allow_html=True)
    search_query = st.text_input("Enter a movie title:", placeholder="Type a movie title...")

    if search_query:
        st.subheader(f'Recommendations for: "{search_query}"')
        recs = recommender.get_recommendations(search_query)
        if not recs.empty:
            cols = st.columns(5)
            for idx, (_, movie) in enumerate(recs.iterrows()):
                with cols[idx % 5]:
                    st.image(recommender.get_movie_poster(movie['title']), width=220, caption=movie['title'])
                    st.write(f"⭐ {movie['vote_average']}")
        else:
            st.warning("No recommendations found.")

    # Top Movies
    st.markdown("---")
    st.markdown('<h2 class="section-header">🎯 Top 5 Most Viewed Movies</h2>', unsafe_allow_html=True)
    top_viewed = recommender.movies_df.nlargest(5, 'vote_count')
    cols = st.columns(5)
    for idx, (_, movie) in enumerate(top_viewed.iterrows()):
        with cols[idx]:
            st.image(recommender.get_movie_poster(movie['title']), width=220, caption=movie['title'])
            st.write(f"⭐ {movie['vote_average']} 👥 {movie['vote_count']:,}")

    # Genre Sections
    genres = [
        ('Adventure', '🚀 Adventure Movies'),
        ('Romance', '💖 Love & Romance'),
        ('Horror', '👻 Horror Movies'),
        ('Drama', '🎭 Drama Movies'),
        ('Action', '💥 Action Movies')
    ]
    for genre, title in genres:
        st.markdown("---")
        st.markdown(f'<h2 class="section-header">{title}</h2>', unsafe_allow_html=True)
        genre_movies = recommender.get_top_movies_by_genre(genre)
        if not genre_movies.empty:
            cols = st.columns(5)
            for idx, (_, movie) in enumerate(genre_movies.iterrows()):
                with cols[idx % 5]:
                    st.image(recommender.get_movie_poster(movie['title']), width=220, caption=movie['title'])
                    st.write(f"⭐ {movie['vote_average']}")
        else:
            st.info(f"No {genre} movies found.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Movie Recommender System • Built with Streamlit • Data from TMDB 5000 Movie Dataset</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
