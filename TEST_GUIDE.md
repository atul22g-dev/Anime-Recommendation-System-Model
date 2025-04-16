# Anime Recommendation System - Beginner's Guide

Welcome to the Anime Recommendation System! This guide will help you understand how to use the test files to get anime recommendations.

## What is This System?

This is a content-based recommendation system that suggests anime similar to ones you already like. It works by:

1. Analyzing the descriptions, genres, producers, studios, and themes of anime
2. Finding anime with similar characteristics to your favorites
3. Recommending the most similar anime to you

## Files in This Project

- **model.ipynb**: The main Jupyter notebook that creates and trains the recommendation model
- **test.py**: An interactive script that lets you enter any anime title and get recommendations
- **simple_test.py**: A basic script that shows recommendations for "Naruto" without requiring input
- **anime_recommender.pkl**: The saved model file (created when you run model.ipynb)

## How to Use the Test Files

### Option 1: Simple Test (Easiest)

If you're new to programming or just want to see how the system works:

1. Make sure you've run the `model.ipynb` notebook first to create the model
2. Run the simple test script:
   ```
   python simple_test.py
   ```
3. This will show you anime similar to "Naruto" without requiring any input

### Option 2: Interactive Test

If you want to search for recommendations based on your favorite anime:

1. Make sure you've run the `model.ipynb` notebook first to create the model
2. Run the interactive test script:
   ```
   python test.py
   ```
3. When prompted, enter the name of an anime you like
4. The system will show you the top 5 similar anime

## Understanding the Code

Both test files follow these basic steps:

1. **Load the model**: The system loads the pre-trained recommendation model from a file
2. **Get the recommender**: It accesses the recommendation component from the model
3. **Find similar anime**: It searches for anime similar to the one you specified
4. **Display results**: It shows you the most similar anime with their details

## Tips for Beginners

- If you get an error about the model file not being found, make sure to run the `model.ipynb` notebook first
- Try entering anime names exactly as they appear in the dataset (English titles)
- If your anime isn't found, the system will suggest similar titles you might try instead
- The recommendations are based on content similarity, not user ratings

## Next Steps

Once you understand how the basic system works, you might want to:

1. Look at the `model.ipynb` file to understand how the recommendation system is built
2. Try modifying the test files to show different information about the recommendations
3. Experiment with different anime titles to see what gets recommended

Happy anime watching!
