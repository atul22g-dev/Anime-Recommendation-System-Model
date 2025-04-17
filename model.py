# %%
import pandas as pd
import dill
import gzip
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

# %%
# ------------------------------------------
# Step 1: Load and clean the dataset
# ------------------------------------------

# Load the dataset
df = pd.read_csv("./dataset/anime.csv", low_memory=False, usecols=['anime_id','image_url',"describe",'english_name','score', 'genres','episodes','producers','studios', 'themes', 'rank', 'popularity']) 

# %%
# ------------------------------------------
# Step 2: Preprocess the data
# ------------------------------------------

# Drop duplicates and rows with missing values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Combine multiple features into one string for content
def combine_features(row):
        return f"{row['describe']} {row['genres']} {row['producers']} {row['studios']} {row['themes']}"

df['combined_content'] = df.apply(combine_features, axis=1)

# Float to int
df['episodes'] = df['episodes'].astype(int)
df['rank'] = df['rank'].astype(int)

# Combine used columns
df = df[['anime_id', 'image_url', 'english_name', 'score', 'genres','episodes' ,'rank','studios','describe' ,'popularity','combined_content']].reset_index(drop=True)

# %%
# ------------------------------------------
# Step 3: Define the AnimeRecommender class
# ------------------------------------------
class AnimeRecommender(BaseEstimator, TransformerMixin):
    """Simple content-based anime recommender using TF-IDF and cosine similarity."""
    
    # Fit method
    def fit(self, X, y=None):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(X['combined_content'])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        self.anime_df = X.reset_index(drop=True)
        self.anime_indices = pd.Series(self.anime_df.index, index=self.anime_df['english_name'].str.lower())
        return self

    # Transform method
    def transform(self, X):
        return self
    
    # Get anime index
    def get_anime_index(self, anime_name):
        # return self.anime_indices.get(anime_name.lower(), -1)

        matches = self.anime_df[self.anime_df['english_name'].str.lower() == anime_name.lower()]

        if not matches.empty:
            return matches.index[0]

        # Try partial matching if exact match not found
        matches = self.anime_df[self.anime_df['english_name'].str.lower().str.contains(anime_name.lower())]

        if not matches.empty:
            return matches.index[0]

        return -1

    # Recommend method
    def get_recommendations(self, anime_name, n=10):
        """""
        Args:
            anime_name (str): Name of the anime to base recommendations on
            n (int): Number of recommendations to return
        
        Returns:
            pandas.DataFrame: Dataframe with recommended anime
        """
        anime_idx = self.get_anime_index(anime_name)
        if anime_idx == -1:
            print(f"Anime '{anime_name}' not found.")
            print("Try searching for similar Anime :")

        similarity_scores = list(enumerate(self.similarity_matrix[anime_idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similar_indices = [i for i, _ in similarity_scores[1:n+1]]
        return self.anime_df.iloc[similar_indices].reset_index(drop=True)
    
    # Recommend by genre
    def get_recommendations_by_genre(self, genre, n=10):
        """"
        Args:
            genre (str): Genre to filter by
            n (int): Number of recommendations to return
        
        Returns:
            pandas.DataFrame: Dataframe with recommended anime
        """
        # Filter anime by genre
        # genre_matches = self.data[self.data['genres'].str.lower().str.contains(genre.lower())]
        genre_matches = self.anime_df[self.anime_df['genres'].str.lower().str.contains(genre.lower())]

        if genre_matches.empty:
            print(f"No anime found for genre '{genre}'.")
            # return pd.DataFrame()
        

        # Sort by score (descending)
        return genre_matches.sort_values('score', ascending=False).head(n).reset_index(drop=True)
    
    # Get top rated
    def get_top_rated_anime(self, n=10):
        """
        Get the top-rated anime.
        Args:
            n (int): Number of anime to return
        Returns:
            pandas.DataFrame: Dataframe with top-rated anime
        """
        # return self.data.sort_values('score', ascending=False).head(n).reset_index(drop=True)
        return self.anime_df.sort_values('score', ascending=False).head(n).reset_index(drop=True)
    
    # Get popular
    def get_popular_anime(self, n=10):
        """
        Get the most popular anime.

        Args:
            n (int): Number of anime to return

        Returns:
            pandas.DataFrame: Dataframe with popular anime
        """
        # return self.anime_df.sort_values('popularity').head(n).reset_index(drop=True)
        return self.anime_df.sort_values('popularity', ascending=False).head(n).reset_index(drop=True)
    
    # Search Anime
    def search_anime(self, query, n=10):
        """
        Search for anime by name.

        Args:
            query (str): Search query
            n (int): Maximum number of results to return

        Returns:
            pandas.DataFrame: Dataframe with search results
        """
        matches = self.anime_df[self.anime_df['english_name'].str.lower().str.contains(query.lower())]
        return matches.head(n).reset_index(drop=True)


# %%
# --------------------------------------------------
# Step 4: Build Pipeline
# --------------------------------------------------
anime_pipeline = Pipeline([
    ('AnimeRecommender', AnimeRecommender())
])

# Fit the pipeline
anime_pipeline.fit(df)

# %%
# --------------------------------------------------
# Step 5: Test the model
# --------------------------------------------------
# Get the recommender model
recommender_model = anime_pipeline.named_steps['AnimeRecommender']

#  Get recommendations  By Anime Name
get_recommendations = recommender_model.get_recommendations("fboiefbioearjhoier")
get_recommendations

# %%
# Get recommendations  By Genre
get_recommendations_by_genre = recommender_model.get_recommendations_by_genre("action")
get_recommendations_by_genre.head(2)

# %%
# Get top rated anime
get_top_rated_anime = recommender_model.get_top_rated_anime()
get_top_rated_anime.head(2)

# %%
# Get popular anime
get_popular_anime = recommender_model.get_popular_anime()
get_popular_anime.head(2)

# %%
# Search anime
search_anime = recommender_model.search_anime("naruto")
search_anime.head(2)

# %%
# --------------------------------------------------
# Step 6: Save the model
# --------------------------------------------------
# dill.dump(anime_pipeline, open("model/anime_recommender.pkl", "wb"))
with gzip.open('model/anime_recommender.pkl.gz', 'wb') as f:
    dill.dump(anime_pipeline, f)