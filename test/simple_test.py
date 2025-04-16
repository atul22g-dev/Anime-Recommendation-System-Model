"""
Simple Test Script for the Anime Recommendation System

This is a very basic example that shows how to use the anime recommendation system.
It loads the model and gets recommendations for a fixed anime title.

Perfect for beginners to understand how the system works!
"""

# Import the dill library to load our saved model
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

# Step 1: Print a welcome message
print("="*60)
print("ANIME RECOMMENDATION SYSTEM - SIMPLE TEST")
print("="*60)
print("This program will show you anime similar to 'Naruto'")
print("="*60)

# Step 2: Load our pre-trained model
print("\nLoading the recommendation model...")
try:
    # Open and load the model file using a safer approach
    with open("../model/anime_recommender.pkl", "rb") as f:
        model = dill.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    # Show an error if the file doesn't exist
    print("ERROR: Could not find the model file 'anime_recommender.pkl'")
    print("Please run the model.ipynb notebook first to create the model.")
    exit()
except ModuleNotFoundError as e:
    # Handle dill compatibility issues
    print(f"ERROR: dill compatibility issue - {e}")
    print("This usually happens when the model was created with a different Python version.")
    print("Please regenerate the model using the current Python environment.")
    exit()
except Exception as e:
    # Handle any other unexpected errors
    print(f"ERROR: Failed to load the model - {e}")
    print("Please check that the model file is not corrupted and try again.")
    exit()

# Step 3: Get the recommender part from our model
# The model is a pipeline with a 'recommender' component
recommender = model.named_steps['recommender']

# Step 4: Get recommendations for "Naruto"
print("\nGetting recommendations for 'Naruto'...")
anime_to_find = "Naruto"
results = recommender.get_recommendations(anime_to_find, n=3)

# Step 5: Show the results
print("\n" + "="*60)
if isinstance(results, str):
    # If Naruto wasn't found, show the error message
    print(results)
else:
    # If recommendations were found, show them
    print(f"TOP 3 ANIME SIMILAR TO '{anime_to_find}':\n")

    # Go through each recommendation
    for i in range(len(results)):
        # Get the current anime recommendation
        anime = results.iloc[i]

        # Print details in a nice format
        print(f"RECOMMENDATION #{i+1}:")
        print(f"Title: {anime['english_name']}")
        print(f"Genres: {anime['genres']}")
        print(f"Score: {anime['score']}")
        print("-"*30)

print("\n" + "="*60)
print("END OF RECOMMENDATIONS")
print("="*60)
print("\nTry running test.py to search for your own favorite anime!")
