"""
Test script for the Anime Recommendation System.

This script demonstrates how to load and use the AnimeRecommender class
from the saved model file.

This is a beginner-friendly example that shows how to:
1. Load the pre-trained anime recommendation model
2. Get anime recommendations based on a title you like
3. Display the results in a simple format
"""

# Import necessary libraries
import dill
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define the AnimeRecommender class (needed to load the model)
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
        return self.anime_indices.get(anime_name.lower(), -1)
    
    """""
    Args:
        anime_name (str): Name of the anime to base recommendations on
        n (int): Number of recommendations to return
            
    Returns:
        pandas.DataFrame: Dataframe with recommended anime
    """

    # Recommend method
    def get_recommendations(self, anime_name, n=10):
        anime_idx = self.get_anime_index(anime_name)
        if anime_idx == -1:
            print(f"Anime '{anime_name}' not found.")
            return pd.DataFrame()

        similarity_scores = list(enumerate(self.similarity_matrix[anime_idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        similar_indices = [i for i, _ in similarity_scores[1:n+1]]
        return self.anime_df.iloc[similar_indices].reset_index(drop=True)

# Step 1: Load the saved model
# The dill.load function reads our saved model from a file
print("Loading the anime recommendation model...")
try:
    dilld_model = dill.load(open("../model/anime_recommender.pkl", "rb"))
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: Model file 'anime_recommender.pkl' not found!")
    print("Make sure to run the model.ipynb notebook first to create the model.")
    exit()

# Step 2: Access the recommender component from our model
# Our model is a pipeline with a 'recommender' step that does the actual work
recommender = dilld_model.named_steps['recommender']

# Step 3: Get user input for an anime they like
print("\n" + "-"*50)
print("Welcome to the Anime Recommendation System!")
print("-"*50)
print("This system will recommend anime similar to one you already like.")
anime_name = input("\nEnter the name of an anime you like: ")

# Step 4: Get recommendations based on the input
print("\nFinding recommendations for '", anime_name, "'...")
recommendations = recommender.get_recommendations(anime_name, n=5)

# Step 5: Display the results
print("\n" + "-"*50)
if isinstance(recommendations, str):
    # If the anime wasn't found, the recommender returns a string message
    print(recommendations)
else:
    # If recommendations were found, display them in a nice format
    print(f"Top 5 Recommendations similar to '{anime_name}':\n")

    # Loop through each recommendation
    for i in range(len(recommendations)):
        # Get the data for the current recommendation
        anime = recommendations.iloc[i]

        # Print the information in a nice format
        print(f"{i+1}. {anime['english_name']}")
        print(f"   Genres: {anime['genres']}")
        print(f"   Score: {anime['score']}")
        print()

print("-"*50)
print("Thank you for using the Anime Recommendation System!")
